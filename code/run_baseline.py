"""Reproducible ItemCF -> SASRec -> DIN baseline, isolated from historical runs."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import argparse
import gc
import hashlib
import json
import pickle
import time
from itertools import islice
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.decomposition import TruncatedSVD
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from baseline_data import BenchmarkData, PrefixDataset, USER_KEYS, ITEM_KEYS, collate, fingerprint, load_brands, load_data
from baseline_runtime import atomic_save, pair_auc, quota_merge, ranking_metrics, restore_checkpoint, save_checkpoint, seed_all
from model import TwoTowerV2Model
from model_ext import DINExtendedModel


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temp, path)


def to_device(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def build_matrix(data):
    rows, cols = data.uid[data.train_mask], data.iid[data.train_mask]
    return sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)),
                         shape=(len(data.users), len(data.items)))


def prepare(args, run):
    source_info = []
    for category in config.AMAZON_CATEGORIES:
        path = Path(args.data_dir) / (category + ".csv")
        stat = path.stat()
        source_info.append([str(path.resolve()), stat.st_size, stat.st_mtime_ns])
    for path in sorted(Path(args.data_dir).glob("raw/meta_categories/*.jsonl")):
        stat = path.stat()
        source_info.append([str(path.resolve()), stat.st_size, stat.st_mtime_ns])
    prep_id = fingerprint([source_info, args.sample_users, args.seed, args.hist_len])
    cache = run / "data.pkl"
    if cache.exists():
        with cache.open("rb") as stream:
            saved = pickle.load(stream)
        if saved["prep_id"] != prep_id:
            raise ValueError("Input data/config changed: use a new run directory")
        return saved["data"]
    print("Loading positive first interactions...", flush=True)
    frame = load_data(args.data_dir, config.AMAZON_CATEGORIES, args.sample_users, args.seed)
    print(f"Loaded {len(frame):,} interactions; reading static brands...", flush=True)
    brands, sources = load_brands(args.data_dir, config.AMAZON_CATEGORIES, frame.parent_asin.unique())
    data = BenchmarkData(frame, brands, args.hist_len, args.seed)
    del frame
    gc.collect()
    data.manifest["brand_sources"] = sources
    data.manifest["source_files"] = source_info
    write_json(run / "data_manifest.json", data.manifest)
    with cache.with_suffix(".tmp").open("wb") as stream:
        pickle.dump({"prep_id": prep_id, "data": data}, stream, protocol=5)
    os.replace(cache.with_suffix(".tmp"), cache)
    return data


def fit_assets(data, args, run):
    svd_path, cf_path = run / "svd.npy", run / "itemcf.pkl"
    matrix = None
    if not svd_path.exists():
        matrix = build_matrix(data)
        print("Fitting SVD on train interactions only...", flush=True)
        svd = TruncatedSVD(n_components=args.dim, n_iter=10, random_state=args.seed)
        svd.fit(matrix)
        factors = svd.components_.T.astype(np.float32)
        factors /= np.maximum(np.linalg.norm(factors, axis=1, keepdims=True), 1e-8)
        factors[:2] = 0
        with svd_path.with_suffix(".tmp").open("wb") as stream:
            np.save(stream, factors)
        os.replace(svd_path.with_suffix(".tmp"), svd_path)
    if not cf_path.exists():
        matrix = build_matrix(data) if matrix is None else matrix
        deg_u = np.asarray(matrix.sum(axis=1)).ravel()
        deg_i = np.asarray(matrix.sum(axis=0)).ravel()
        weighted = sp.diags(1 / np.sqrt(np.log1p(deg_u).clip(min=1))) @ matrix
        weighted = (weighted @ sp.diags(1 / np.sqrt(deg_i.clip(min=1)))).tocsr()
        columns = weighted.tocsc()
        k = min(args.cf_neighbors, len(data.items) - 3)
        neighbors = np.zeros((len(data.items), k), dtype=np.int32)
        similarities = np.zeros((len(data.items), k), dtype=np.float32)
        for start in tqdm(range(2, len(data.items), 256), desc="ItemCF train-only cosine/IUF"):
            end = min(start + 256, len(data.items))
            block = (columns[:, start:end].T @ weighted).tocsr()
            for row in range(end - start):
                ids = block.indices[block.indptr[row]:block.indptr[row + 1]]
                scores = block.data[block.indptr[row]:block.indptr[row + 1]]
                valid = (ids >= 2) & (ids != start + row) & (scores > 0)
                ids, scores = ids[valid], scores[valid]
                if len(ids) > k:
                    selection = np.argpartition(-scores, k - 1)[:k]
                    ids, scores = ids[selection], scores[selection]
                order = np.lexsort((ids, -scores))
                neighbors[start + row, :len(order)] = ids[order]
                similarities[start + row, :len(order)] = scores[order]
        with cf_path.with_suffix(".tmp").open("wb") as stream:
            pickle.dump((neighbors, similarities), stream, protocol=5)
        os.replace(cf_path.with_suffix(".tmp"), cf_path)
    with cf_path.open("rb") as stream:
        return np.load(svd_path, mmap_mode="r"), pickle.load(stream)


def model_config(data, args, kind):
    common = {"num_users": len(data.users), "num_items": len(data.items),
              "num_categories": len(data.categories), "num_brands": len(data.brands),
              "embed_dim": args.dim, "brand_embed_dim": min(64, args.dim),
              "hist_len": args.hist_len, "dropout": .1}
    if kind == "v2":
        common.update(hidden_dims=[512, 384, 256, 128] if args.dim >= 128 else [64, 32],
                      num_heads=2, num_blocks=2, temperature=.07,
                      use_brand_pref=True, use_time_decay=True, time_decay_lambda=.3)
    else:
        common["hidden_dims"] = [256, 128, 64] if args.dim >= 128 else [64, 32]
    return common


def make_model(data, args, kind, device):
    cls = TwoTowerV2Model if kind == "v2" else DINExtendedModel
    return cls(**model_config(data, args, kind)).to(device)


def encode_items(model, data, device):
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(data.items), 4096):
            batch = {k: torch.as_tensor(v, device=device) for k, v in
                     data.item_features(np.arange(start, min(start + 4096, len(data.items)))).items()}
            chunks.append(model.get_item_embedding(batch).float())
    return torch.cat(chunks)


def recall(model, data, records, device, k):
    model.eval()
    item_vectors = encode_items(model, data, device)
    rankings, scored = [], []
    with torch.inference_mode():
        for start in tqdm(range(0, len(records), 128), desc="Full-catalog V2"):
            chunk = records[start:start + 128]
            batch = to_device(collate([data.user_features(uid, pos) for uid, pos in chunk]), device)
            scores = model.get_user_embedding(batch).float() @ item_vectors.T
            scores[:, :2] = -torch.inf
            for row, (uid, pos) in enumerate(chunk):
                seen = data.iid[data.starts[uid]:data.history_end(uid, pos)]
                scores[row, torch.as_tensor(seen.astype(np.int64), device=device)] = -torch.inf
            values, indices = scores.topk(min(k, scores.shape[1] - 2), dim=1)
            for ids, vals in zip(indices.cpu().numpy(), values.cpu().numpy()):
                mapping = {int(i): float(v) for i, v in zip(ids, vals) if np.isfinite(v)}
                rankings.append(list(mapping))
                scored.append(mapping)
    del item_vectors
    return rankings, scored


def candidate_pools(data, records, v2_scores, cf, budget):
    neighbors, similarity = cf
    hot = np.argsort(-data.train_counts, kind="stable")
    hot = [int(i) for i in hot if i >= 2 and data.train_counts[i] > 0]
    category_hot = {cat: [i for i in hot if data.item_category[i] == cat]
                    for cat in range(2, len(data.categories))}
    pools, cf_rankings = [], []
    for (uid, pos), v2 in tqdm(zip(records, v2_scores), total=len(records), desc="Candidate fusion"):
        end = data.history_end(uid, pos)
        seen = set(data.iid[data.starts[uid]:end].tolist())
        hist = data.iid[max(data.starts[uid], end - data.hist_len):end]
        cf_score = {}
        for iid in hist:
            for candidate, score in zip(neighbors[iid], similarity[iid]):
                if candidate >= 2 and candidate not in seen and score > 0:
                    cf_score[int(candidate)] = cf_score.get(int(candidate), 0.) + float(score)
        cf_score = dict(sorted(cf_score.items(), key=lambda x: (-x[1], x[0]))[:budget])
        for iid in hot:
            if len(cf_score) >= budget:
                break
            if iid not in seen and iid not in cf_score:
                cf_score[iid] = -1.0
        cf_rankings.append(list(cf_score))
        cats, counts = np.unique(data.item_category[hist], return_counts=True)
        cat_score = {}
        for cat, count in sorted(zip(cats, counts), key=lambda x: -x[1])[:3]:
            eligible = list(islice((i for i in category_hot.get(cat, ()) if i not in seen), 20))
            cat_score.update({i: float(count / max(len(hist), 1)) for i in eligible})
        hot_score = {i: 1 / (rank + 1) for rank, i in enumerate(islice((i for i in hot if i not in seen), 5))}
        merged = quota_merge([cf_score, v2, cat_score, hot_score], [1.5, 1., .7, .05], budget, seen)
        for iid in hot:
            if len(merged) >= budget:
                break
            if iid not in seen and iid not in merged:
                merged[iid] = -1.0
        pools.append(merged)
    return pools, cf_rankings


def score_din(model, data, records, pools, device, microbatch=512):
    model.eval()
    ranked = []
    with torch.inference_mode():
        for offset in tqdm(range(0, len(records), 16), desc="Pure DIN rerank"):
            chunk = records[offset:offset + 16]
            chunk_ids = [list(pool) for pool in pools[offset:offset + 16]]
            sizes = [len(ids) for ids in chunk_ids]
            users = collate([data.user_features(uid, pos) for uid, pos in chunk])
            ids = [iid for part in chunk_ids for iid in part]
            if not ids:
                ranked.extend([[] for _ in chunk])
                continue
            row_index = torch.repeat_interleave(torch.arange(len(chunk)), torch.as_tensor(sizes))
            scores = []
            for start in range(0, len(ids), microbatch):
                subset = ids[start:start + microbatch]
                rows = row_index[start:start + microbatch]
                batch = {k: value[rows].to(device) for k, value in users.items()}
                batch.update({k: torch.as_tensor(v, device=device) for k, v in data.item_features(subset).items()})
                values = model(batch)
                scores.extend(values.float().cpu().tolist())
            cursor = 0
            for part in chunk_ids:
                values = scores[cursor:cursor + len(part)]
                order = sorted(range(len(part)), key=lambda i: (-values[i], part[i]))
                ranked.append([part[i] for i in order])
                cursor += len(part)
    return ranked


def infonce(model, batch):
    users, items = model(batch)
    logits = users.float() @ items.float().T / model.temperature
    ids, uids = batch["pos_item_id"], batch["user_id"]
    diagonal = torch.eye(len(ids), dtype=torch.bool, device=ids.device)
    known_positive = (ids[:, None] == ids[None, :]) | (uids[:, None] == uids[None, :])
    # Known history positives in the batch must not be used as negatives.
    known_positive |= (batch["hist_items"][:, :, None] == ids[None, None, :]).any(dim=1)
    logits = logits.masked_fill(known_positive & ~diagonal, -torch.inf)
    loss = F.cross_entropy(logits, torch.arange(len(ids), device=ids.device))
    hard = {key: batch["neg_" + key].reshape(-1) for key in ITEM_KEYS}
    negative_vecs = model.get_item_embedding(hard).float().reshape(len(ids), -1, users.shape[1])
    pos_score = (users.float() * items.float()).sum(-1, keepdim=True) / model.temperature
    neg_scores = torch.einsum("bd,bkd->bk", users.float(), negative_vecs) / model.temperature
    return loss + .3 * F.cross_entropy(torch.cat([pos_score, neg_scores], 1), torch.zeros(len(ids), dtype=torch.long, device=ids.device))


def train_model(data, args, run, kind, factors, records, pools, identity, device,
                evaluation_targets=None, selection_metric="hr", latest_dir=None):
    seed_all(args.seed)
    model = make_model(data, args, kind, device)
    item_layer = model.item_tower.item_embedding if kind == "v2" else model.item_embedding
    with torch.no_grad():
        item_layer.weight.copy_(torch.as_tensor(np.array(factors), device=device))
    lr = .0005 if kind == "v2" else .001
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    manifest = {"run_identity": identity, "data_id": data.manifest["data_id"], "kind": kind,
                "model_config": model_config(data, args, kind)}
    state_dir = Path(latest_dir) if latest_dir else run
    state_dir.mkdir(parents=True, exist_ok=True)
    latest, best_path = state_dir / (kind + "_latest.pth"), run / (kind + "_best.pth")
    start_epoch, best, history = 0, -1., []
    if latest.exists():
        start_epoch, best, history = restore_checkpoint(latest, model, optimizer, scheduler, manifest)
        print(f"Resumed {kind} at epoch {start_epoch}, best={best:.6f}", flush=True)
    dataset = PrefixDataset(data, args.negatives)
    targets = evaluation_targets if evaluation_targets is not None else [int(data.iid[pos]) for _, pos in records]
    for epoch in range(start_epoch, args.epochs):
        dataset.epoch = epoch
        generator = torch.Generator().manual_seed(args.seed + epoch)
        loader = DataLoader(dataset, batch_size=args.v2_batch if kind == "v2" else args.din_batch,
                            shuffle=True, generator=generator, num_workers=args.workers,
                            collate_fn=collate, pin_memory=device.type == "cuda", drop_last=True)
        model.train()
        total, steps = 0., 0
        started = time.time()
        for batch in tqdm(loader, desc=f"{kind} epoch {epoch + 1}/{args.epochs}", mininterval=20):
            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                if kind == "v2":
                    loss = infonce(model, batch)
                else:
                    user = {k: batch[k] for k in USER_KEYS}
                    positive = {k: batch["pos_" + k] for k in ITEM_KEYS}
                    negative = {k: batch["neg_" + k] for k in ITEM_KEYS}
                    pos, neg = model.forward_bpr(user, positive, negative)
                    loss = F.softplus(neg.float() - pos.float()[:, None]).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{kind}: nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach())
            steps += 1
        if not steps:
            raise ValueError("No complete training batches; reduce batch size")
        scheduler.step()
        if kind == "v2":
            rankings, _ = recall(model, data, records, device, args.candidates)
            metrics = ranking_metrics(rankings, targets, args.candidates)
        else:
            rankings = score_din(model, data, records, pools, device)
            metrics = ranking_metrics(rankings, targets, 5)
        metric = metrics[selection_metric]
        history.append({"epoch": epoch + 1, "train_loss": total / steps, "steps": steps,
                        "seconds": time.time() - started, "validation": metrics})
        if metric > best:
            best = metric
            atomic_save({"model": model.state_dict(), "manifest": manifest, "epoch": epoch + 1,
                         "metrics": metrics}, best_path)
        save_checkpoint(latest, model, optimizer, scheduler, epoch + 1, best, manifest, history)
        write_json(run / (kind + "_history.json"), history)
        print(json.dumps({"stage": kind, **history[-1]}, ensure_ascii=False), flush=True)
    saved = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(saved["model"])
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=os.environ.get("AMAZON_DATA_PATH", config.DATA_PATH))
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--sample-users", type=int, default=0)
    parser.add_argument("--eval-users", type=int, default=10000)
    parser.add_argument("--final-users", type=int, default=100000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--hist-len", type=int, default=50)
    parser.add_argument("--negatives", type=int, default=4)
    parser.add_argument("--candidates", type=int, default=100)
    parser.add_argument("--cf-neighbors", type=int, default=100)
    parser.add_argument("--v2-batch", type=int, default=1024)
    parser.add_argument("--din-batch", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.dim % 2 or args.epochs < 1 or args.negatives < 1:
        parser.error("dim must be even; epochs and negatives must be positive")
    if min(args.dim, args.hist_len, args.candidates, args.cf_neighbors) < 1 or min(args.v2_batch, args.din_batch) < 2:
        parser.error("dimensions and candidate counts must be positive; batch sizes must be at least 2")
    if min(args.sample_users, args.eval_users, args.final_users, args.workers) < 0:
        parser.error("user limits and workers must be nonnegative")
    run = Path(args.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)
    code_hash = fingerprint({p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                             Path(__file__).parent.glob("*.py")})
    run_args = {key: value for key, value in vars(args).items() if key != "prepare_only"}
    spec = {"args": run_args, "code_hash": code_hash, "torch": torch.__version__,
            "numpy": np.__version__, "device": str(device)}
    identity = fingerprint(spec)
    identity_path = run / "run_manifest.json"
    if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf-8")) != spec:
        raise ValueError("Run manifest differs; use a new output directory")
    write_json(identity_path, spec)
    print(json.dumps(spec, indent=2), flush=True)
    data = prepare(args, run)
    print(json.dumps(data.manifest, indent=2), flush=True)
    if not len(data.val_users):
        raise ValueError("No eligible validation/test users")
    factors, cf = fit_assets(data, args, run)
    if args.prepare_only:
        return
    records = data.evaluation("val", args.eval_users)
    model = train_model(data, args, run, "v2", factors, records, None, identity, device)
    _, scores = recall(model, data, records, device, args.candidates)
    pools, _ = candidate_pools(data, records, scores, cf, args.candidates)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    din = train_model(data, args, run, "din", factors, records, pools, identity, device)
    del din
    gc.collect()
    torch.cuda.empty_cache()
    results = {"data_id": data.manifest["data_id"], "run_identity": identity,
               "selection_metric": f"val V2 HR@{args.candidates}, val pure DIN HR@5", "splits": {}}
    for split in ("val", "test"):
        records = data.evaluation(split, args.final_users)
        targets = [int(data.iid[pos]) for _, pos in records]
        model = make_model(data, args, "v2", device)
        saved = torch.load(run / "v2_best.pth", map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"])
        rankings, scores = recall(model, data, records, device, args.candidates)
        pools, cf_rankings = candidate_pools(data, records, scores, cf, args.candidates)
        del model, saved
        gc.collect()
        torch.cuda.empty_cache()
        din = make_model(data, args, "din", device)
        saved = torch.load(run / "din_best.pth", map_location="cpu", weights_only=False)
        din.load_state_dict(saved["model"])
        final_rankings = score_din(din, data, records, pools, device)
        metrics = {"v2": ranking_metrics(rankings, targets, args.candidates),
                   "itemcf": ranking_metrics(cf_rankings, targets, args.candidates),
                   "candidate_pool": ranking_metrics([list(p) for p in pools], targets, args.candidates),
                   "recall_only": ranking_metrics([list(p) for p in pools], targets, 5),
                   "din": ranking_metrics(final_rankings, targets, 5),
                   "pool_size_mean": float(np.mean([len(p) for p in pools]))}
        cover = metrics["candidate_pool"]["hr"]
        metrics["din_conditional_hr5"] = metrics["din"]["hr"] / cover if cover else 0.
        results["splits"][split] = metrics
        write_json(run / "results.json", results)
        predictions = [{"user_id": data.users[uid], "target": data.items[data.iid[pos]],
                        "items": [data.items[i] for i in ranked[:5]],
                        "candidate_hit": int(data.iid[pos] in pool)}
                       for (uid, pos), ranked, pool in zip(records, final_rankings, pools)]
        write_json(run / (split + "_predictions.json"), predictions)
        print(json.dumps({"split": split, **metrics}, indent=2), flush=True)
        del din, saved
        gc.collect()
        torch.cuda.empty_cache()
    write_json(run / "COMPLETED.json", {"status": "complete", "results": results})


if __name__ == "__main__":
    main()
