"""Future-window validation for semantic-token DIN rankers.

The V2 checkpoint and candidate pools remain fixed.  Only the DIN token
builder/fusion head is trained in this suite, so Top-5 changes are attributable
to the ranker architecture.
"""
import argparse
import gc
import hashlib
import json
import os
import pickle
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from baseline_data import ITEM_KEYS, USER_KEYS, PrefixDataset, collate, fingerprint
from baseline_runtime import ranking_metrics, seed_all
from run_baseline import candidate_pools, make_model, recall, score_din, write_json
from token_models import SemanticTokenDIN


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def model_config(data, args, fusion):
    return {
        "num_users": len(data.users), "num_items": len(data.items),
        "num_brands": len(data.brands), "num_categories": len(data.categories),
        "embed_dim": int(args.dim), "brand_embed_dim": min(64, int(args.dim)),
        "hidden_dims": [256, 128, 64] if args.dim >= 128 else [64, 32],
        "hist_len": int(args.hist_len), "dropout": 0.1,
        "token_dim": int(args.token_dim), "fusion": fusion,
    }


def init_svd(model, data, factors):
    """Use the shared train-only item factors and derive user means.

    This is a fresh ranker initialization: only the common CF warm-start is
    reused; the token projections and ranking head remain newly initialized.
    """
    factors = np.array(factors, dtype=np.float32, copy=True)
    dim = model.embed_dim
    if factors.shape[0] != len(data.items) or factors.shape[1] != dim:
        raise ValueError(f"SVD shape {factors.shape} does not match {(len(data.items), dim)}")
    user = np.zeros((len(data.users), dim), dtype=np.float32)
    counts = np.zeros(len(data.users), dtype=np.int64)
    train_positions = np.flatnonzero(data.train_mask)
    for uid, iid in zip(data.uid[train_positions], data.iid[train_positions]):
        if iid >= 2:
            user[int(uid)] += factors[int(iid)]
            counts[int(uid)] += 1
    valid = counts > 0
    user[valid] /= counts[valid, None]
    with torch.no_grad():
        model.item_embedding.weight.copy_(torch.as_tensor(factors))
        model.user_embedding.weight.copy_(torch.as_tensor(user))
        model.item_embedding.weight[:2].zero_()
        model.user_embedding.weight[:2].zero_()


class TokenSuite:
    def __init__(self, args):
        self.args = args
        self.base = Path(args.base_run).resolve()
        self.cf_source = Path(args.cf_run or args.base_run).resolve()
        self.run_dir = Path(args.run_dir).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if not (self.base / "data.pkl").exists():
            raise FileNotFoundError(self.base / "data.pkl")
        if not (self.base / "v2_best.pth").exists():
            raise FileNotFoundError(self.base / "v2_best.pth")
        with (self.base / "data.pkl").open("rb") as stream:
            self.data = pickle.load(stream)["data"]
        with (self.cf_source / "itemcf.pkl").open("rb") as stream:
            self.cf = pickle.load(stream)
        source_manifest = load_json(self.base / "run_manifest.json")
        source_args = source_manifest["args"]
        self.args.dim = int(source_args["dim"])
        self.args.hist_len = int(source_args["hist_len"])
        self.args.workers = int(source_args["workers"])
        self.args.din_batch = int(source_args["din_batch"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_num_threads(4)
        self.data_id = self.data.manifest["data_id"]
        self.source = {
            "base_run": str(self.base),
            "base_manifest": sha256_file(self.base / "run_manifest.json"),
            "base_v2": sha256_file(self.base / "v2_best.pth"),
            "svd": sha256_file(self.base / "svd.npy"),
            "itemcf": sha256_file(self.cf_source / "itemcf.pkl"),
            "data_id": self.data_id,
        }
        self.fusion_mode = args.fusion_mode
        self.fusion_weights = list(args.fusion_weights)
        self.half_life = float(args.itemcf_half_life_days)
        self.cache_dir = self.run_dir / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        self.factors = np.load(self.base / "svd.npy", mmap_mode="r")
        self.v2 = self._load_v2()
        self.manifest = {
            "source": self.source, "data_id": self.data_id,
            "protocol": "future-window", "candidate_budget": args.candidates,
            "fusion_mode": self.fusion_mode, "fusion_weights": self.fusion_weights,
            "itemcf_half_life_days": self.half_life,
            "token_dim": args.token_dim, "epochs": args.epochs,
            "variants": ["semantic_concat", "semantic_rankmixer"],
            "screen_users": args.screen_users, "test_users": args.test_users,
            "seed": args.seed,
        }
        write_json(self.run_dir / "suite_manifest.json", self.manifest)

    def _load_v2(self):
        source_args = load_json(self.base / "run_manifest.json")["args"]
        model_args = SimpleNamespace(**source_args)
        model = make_model(self.data, model_args, "v2", self.device)
        saved = torch.load(self.base / "v2_best.pth", map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"])
        model.eval()
        return model

    def records(self, split, count):
        return self.data.evaluation(split, count)

    def targets(self, records):
        return self.data.targets(records)

    def pools(self, records, label):
        key = fingerprint({"label": label, "records": records,
                           "data_id": self.data_id, "v2": self.source["base_v2"],
                           "itemcf": self.source["itemcf"], "budget": self.args.candidates,
                           "fusion_mode": self.fusion_mode, "weights": self.fusion_weights,
                           "half_life": self.half_life})
        path = self.cache_dir / f"pools_{label}_{key}.pkl"
        if path.exists():
            with path.open("rb") as stream:
                return pickle.load(stream)
        _, scores = recall(self.v2, self.data, records, self.device, self.args.candidates)
        pools, _ = candidate_pools(
            self.data, records, scores, self.cf, self.args.candidates,
            self.fusion_mode, self.half_life, self.fusion_weights,
        )
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            pickle.dump(pools, stream, protocol=5)
        os.replace(temporary, path)
        return pools

    def score(self, model, records, pools):
        model.eval()
        ranked = []
        with torch.inference_mode():
            for offset in tqdm(range(0, len(records), 16), desc="Token DIN rerank", mininterval=20):
                chunk = records[offset:offset + 16]
                parts = [list(pool) for pool in pools[offset:offset + 16]]
                sizes = [len(part) for part in parts]
                ids = [iid for part in parts for iid in part]
                if not ids:
                    ranked.extend([[] for _ in chunk])
                    continue
                users = collate([self.data.user_features(uid, pos) for uid, pos in chunk])
                encoded = model._encode_user({key: value.to(self.device) for key, value in users.items()})
                rows = torch.repeat_interleave(torch.arange(len(chunk), device=self.device),
                                               torch.as_tensor(sizes, device=self.device))
                values = []
                for start in range(0, len(ids), self.args.microbatch):
                    end = min(start + self.args.microbatch, len(ids))
                    index = rows[start:end]
                    user = {key: value[index] if value is not None else None
                            for key, value in encoded.items()}
                    items = {key: torch.as_tensor(value, device=self.device)
                             for key, value in self.data.item_features(ids[start:end]).items()}
                    values.extend(model._score_item(user, items).float().cpu().tolist())
                cursor = 0
                for part in parts:
                    scores = values[cursor:cursor + len(part)]
                    order = sorted(range(len(part)), key=lambda i: (-scores[i], part[i]))
                    ranked.append([part[i] for i in order])
                    cursor += len(part)
        return ranked

    def evaluate(self, model, records, pools, label):
        targets = self.targets(records)
        rankings = self.score(model, records, pools)
        metrics = {
            "candidate_pool": ranking_metrics([list(pool) for pool in pools], targets, self.args.candidates),
            "din": ranking_metrics(rankings, targets, 5),
            "pool_size_mean": float(np.mean([len(pool) for pool in pools])),
            "label": label, "users": len(records),
        }
        coverage = metrics["candidate_pool"]["hr"]
        metrics["din_conditional_hr5"] = metrics["din"]["hr"] / coverage if coverage else 0.
        arrays = {
            "hit5": np.asarray([bool(set(r[:5]) & set(t)) for r, t in zip(rankings, targets)], dtype=np.int8),
            "ndcg5": np.asarray([
                sum(1.0 / np.log2(i + 2) for i, iid in enumerate(r[:5]) if iid in set(t)) /
                max(sum(1.0 / np.log2(i + 2) for i in range(min(5, len(set(t))))), 1e-12)
                for r, t in zip(rankings, targets)
            ], dtype=np.float32),
            "pool_hit": np.asarray([bool(set(t) & set(p)) for t, p in zip(targets, pools)], dtype=np.int8),
        }
        np.savez_compressed(self.run_dir / f"{label}_users.npz", **arrays)
        write_json(self.run_dir / f"{label}_metrics.json", metrics)
        return metrics, arrays

    def evaluate_baseline(self, records, pools, label):
        source_args = SimpleNamespace(**load_json(self.base / "run_manifest.json")["args"])
        model = make_model(self.data, source_args, "din", self.device)
        saved = torch.load(self.base / "din_best.pth", map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"])
        model.eval()
        rankings = score_din(model, self.data, records, pools, self.device)
        targets = self.targets(records)
        metrics = {
            "candidate_pool": ranking_metrics([list(pool) for pool in pools], targets, self.args.candidates),
            "din": ranking_metrics(rankings, targets, 5),
            "pool_size_mean": float(np.mean([len(pool) for pool in pools])),
            "label": label, "users": len(records),
        }
        coverage = metrics["candidate_pool"]["hr"]
        metrics["din_conditional_hr5"] = metrics["din"]["hr"] / coverage if coverage else 0.
        arrays = {"hit5": np.asarray([bool(set(r[:5]) & set(t)) for r, t in zip(rankings, targets)], dtype=np.int8),
                  "ndcg5": np.asarray([
                      sum(1.0 / np.log2(i + 2) for i, iid in enumerate(r[:5]) if iid in set(t)) /
                      max(sum(1.0 / np.log2(i + 2) for i in range(min(5, len(set(t))))), 1e-12)
                      for r, t in zip(rankings, targets)
                  ], dtype=np.float32),
                  "pool_hit": np.asarray([bool(set(t) & set(p)) for t, p in zip(targets, pools)], dtype=np.int8)}
        np.savez_compressed(self.run_dir / f"{label}_users.npz", **arrays)
        write_json(self.run_dir / f"{label}_metrics.json", metrics)
        del model, saved, rankings
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return metrics, arrays

    @staticmethod
    def paired(candidate, baseline):
        delta = candidate["hit5"].astype(np.int8) - baseline["hit5"].astype(np.int8)
        counts = np.array([(delta == v).sum() for v in (-1, 0, 1)])
        rng = np.random.default_rng(42)
        boot = rng.multinomial(len(delta), counts / max(len(delta), 1), size=10000)
        interval = np.quantile((boot[:, 2] - boot[:, 0]) / max(len(delta), 1), [0.025, 0.975])
        return {
            "hr5_delta": float(delta.mean()), "hr5_delta_ci95": interval.tolist(),
            "ndcg5_delta": float(candidate["ndcg5"].mean() - baseline["ndcg5"].mean()),
            "gained": int(counts[2]), "lost": int(counts[0]),
            "users": int(len(delta)),
            "confirmed_gain": bool(interval[0] > 0 and candidate["ndcg5"].mean() >= baseline["ndcg5"].mean()),
        }

    def build_model(self, fusion):
        return SemanticTokenDIN(**model_config(self.data, self.args, fusion)).to(self.device)

    def train_variant(self, fusion, epochs, seed):
        name = "semantic_" + fusion
        directory = self.run_dir / name
        directory.mkdir(exist_ok=True)
        manifest = {"variant": name, "fusion": fusion, "seed": seed,
                    "epochs": epochs, "source": self.source,
                    "model_config": model_config(self.data, self.args, fusion),
                    "negative_count": self.args.negatives}
        completed = directory / "COMPLETED.json"
        if completed.exists():
            saved = torch.load(directory / "best.pth", map_location=self.device, weights_only=False)
            model = self.build_model(fusion)
            model.load_state_dict(saved["model"])
            return model, load_json(directory / "history.json")
        seed_all(seed)
        model = self.build_model(fusion)
        init_svd(model, self.data, self.factors)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        dataset = PrefixDataset(self.data, self.args.negatives)
        records = self.records("val", self.args.screen_users)
        pools = self.pools(records, "screen")
        history = []
        best = -1.
        loader = None
        for epoch in range(epochs):
            dataset.epoch = epoch
            generator = torch.Generator().manual_seed(seed + epoch)
            loader = DataLoader(dataset, batch_size=self.args.din_batch, shuffle=True,
                                generator=generator, num_workers=self.args.workers,
                                collate_fn=collate, pin_memory=self.device.type == "cuda",
                                drop_last=True, persistent_workers=self.args.workers > 0)
            model.train()
            total, steps = 0., 0
            started = time.monotonic()
            for batch in tqdm(loader, desc=f"{name} epoch {epoch + 1}/{epochs}", mininterval=20):
                batch = {key: value.to(self.device, non_blocking=True) for key, value in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16,
                                    enabled=self.device.type == "cuda"):
                    pos, neg = model.forward_bpr(
                        {key: batch[key] for key in USER_KEYS},
                        {key: batch["pos_" + key] for key in ITEM_KEYS},
                        {key: batch["neg_" + key] for key in ITEM_KEYS},
                    )
                    loss = F.softplus(neg.float() - pos.float()[:, None]).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"{name}: nonfinite loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                optimizer.step()
                total += float(loss.detach())
                steps += 1
                if self.args.max_train_steps and steps >= self.args.max_train_steps:
                    break
            scheduler.step()
            model.eval()
            metrics, _ = self.evaluate(model, records, pools, f"{name}_epoch{epoch + 1}")
            entry = {"epoch": epoch + 1, "loss": total / max(steps, 1),
                     "steps": steps, "seconds": time.monotonic() - started,
                     "metrics": metrics}
            history.append(entry)
            if metrics["din"]["hr"] > best:
                best = metrics["din"]["hr"]
                torch.save({"model": model.state_dict(), "manifest": manifest,
                            "epoch": epoch + 1, "metrics": metrics}, directory / "best.pth")
            write_json(directory / "history.json", history)
        write_json(completed, {"status": "complete", "manifest": manifest, "best_hr5": best})
        saved = torch.load(directory / "best.pth", map_location=self.device, weights_only=False)
        model.load_state_dict(saved["model"])
        model.eval()
        return model, history

    def run(self):
        screen = self.records("val", self.args.screen_users)
        test = self.records("test", self.args.test_users)
        screen_pools = self.pools(screen, "screen")
        test_pools = self.pools(test, "test")
        baseline_screen, baseline_screen_arrays = self.evaluate_baseline(screen, screen_pools, "baseline_screen")
        baseline_test, baseline_test_arrays = self.evaluate_baseline(test, test_pools, "baseline_test")
        results = {"manifest": self.manifest,
                   "baseline": {"screen": baseline_screen, "test": baseline_test},
                   "variants": {}}
        for fusion in ("concat", "rankmixer"):
            model, history = self.train_variant(fusion, self.args.epochs, self.args.seed)
            screen_metrics, _ = self.evaluate(model, screen, screen_pools, fusion + "_screen_final")
            test_metrics, _ = self.evaluate(model, test, test_pools, fusion + "_test_final")
            results["variants"][fusion] = {"history": history,
                                            "screen": screen_metrics,
                                            "test": test_metrics,
                                            "screen_vs_baseline": self.paired(
                                                np.load(self.run_dir / f"{fusion}_screen_final_users.npz"),
                                                baseline_screen_arrays),
                                            "test_vs_baseline": self.paired(
                                                np.load(self.run_dir / f"{fusion}_test_final_users.npz"),
                                                baseline_test_arrays)}
            del model
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        write_json(self.run_dir / "results.json", results)
        write_json(self.run_dir / "COMPLETED.json", {"status": "complete", "results": results})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run", required=True)
    parser.add_argument("--cf-run", default="")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--screen-users", type=int, default=100000)
    parser.add_argument("--test-users", type=int, default=100000)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--token-dim", type=int, default=256)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--hist-len", type=int, default=50)
    parser.add_argument("--negatives", type=int, default=4)
    parser.add_argument("--din-batch", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--microbatch", type=int, default=512)
    parser.add_argument("--max-train-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--candidates", type=int, default=75)
    parser.add_argument("--fusion-mode", choices=("quota", "rrf"), default="rrf")
    parser.add_argument("--fusion-weights", type=float, nargs=4, default=[2.0, 1.0, 0.7, 0.05])
    parser.add_argument("--itemcf-half-life-days", type=float, default=180.)
    args = parser.parse_args()
    if args.epochs < 1 or args.token_dim < 1 or args.screen_users < 1 or args.test_users < 1:
        parser.error("epochs, token_dim and user counts must be positive")
    suite = TokenSuite(args)
    suite.run()


if __name__ == "__main__":
    main()
