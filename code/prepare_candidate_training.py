"""Mine recall negatives from a retrieval model fitted on an internal training split."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import argparse
import gc
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from tqdm import tqdm

from baseline_data import BenchmarkData
from run_baseline import build_matrix, write_json
from run_fusion_experiments import Channels, RECIPES
from run_optimization import file_hash


def training_internal_data(base):
    positions = np.flatnonzero(base.train_mask)
    items = base.iid[positions]
    frame = pd.DataFrame({
        "user_id": np.asarray(base.users, dtype=object)[base.uid[positions]],
        "parent_asin": np.asarray(base.items, dtype=object)[items],
        "rating": base.rating[positions], "timestamp": base.ts[positions],
        "category": np.asarray(base.categories, dtype=object)[base.item_category[items]],
    })
    brands = {base.items[i]: base.brands[base.item_brand[i]] for i in base.active_items
              if base.item_brand[i] >= 2}
    inner = BenchmarkData(frame, brands, base.hist_len, seed=73)
    if len(inner.iid) != len(positions):
        raise ValueError("Internal row mapping changed")
    raw_to_original = {raw: i for i, raw in enumerate(base.items)}
    item_map = np.asarray([raw_to_original[raw] for raw in inner.items], dtype=np.int32)
    if not np.array_equal(item_map[inner.iid], base.iid[positions]):
        raise ValueError("Internal item mapping differs")
    return inner, positions, item_map


def fit_cf(data, count=100):
    matrix = build_matrix(data)
    deg_u = np.asarray(matrix.sum(axis=1)).ravel()
    deg_i = np.asarray(matrix.sum(axis=0)).ravel()
    weighted = sp.diags(1 / np.sqrt(np.log1p(deg_u).clip(min=1))) @ matrix
    weighted = (weighted @ sp.diags(1 / np.sqrt(deg_i.clip(min=1)))).tocsr()
    columns = weighted.tocsc()
    k = min(count, len(data.items) - 3)
    neighbors = np.zeros((len(data.items), k), dtype=np.int32)
    similarity = np.zeros((len(data.items), k), dtype=np.float32)
    for start in tqdm(range(2, len(data.items), 256), desc="Inner-train ItemCF", mininterval=20):
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
            similarity[start + row, :len(order)] = scores[order]
    return neighbors, similarity


def mine(base, inner, original_positions, item_map, cf, records, budget=100):
    builder = Channels(inner, cf)
    mapped = np.zeros((len(records), 2), dtype=np.int64)
    negatives = np.zeros((len(records), budget), dtype=np.int32)
    hits, sizes = [], []
    for index, record in enumerate(tqdm(records, desc="Inner-heldout recall negatives", mininterval=20)):
        channels, seen = builder.build(record, {}, budget, RECIPES["rrf"])
        pool = builder.merge(channels, seen, budget, RECIPES["rrf"])
        pos = int(original_positions[record[1]])
        uid, target = int(base.uid[pos]), int(base.iid[pos])
        if not base.train_mask[pos] or int(inner.iid[record[1]]) in inner.train_sets[record[0]]:
            raise ValueError("Held-out retrieval target leaked into fitting data")
        mapped[index] = (uid, pos)
        original_pool = [int(item_map[i]) for i in pool]
        hits.append(target in original_pool)
        eligible = [i for i in original_pool if i in base.active_set and i not in base.train_sets[uid]]
        negatives[index, :len(eligible)] = eligible
        sizes.append(len(eligible))
    return mapped, negatives, {"retrieval_target_coverage": float(np.mean(hits)),
                              "mean_eligible_negatives": float(np.mean(sizes)),
                              "empty_pool_fraction": float(np.mean(np.asarray(sizes) == 0))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--users", type=int, default=400000)
    args = parser.parse_args()
    if args.users < 1:
        parser.error("users must be positive")
    base_path, run = Path(args.baseline).resolve(), Path(args.run_dir).resolve()
    run.mkdir(parents=True, exist_ok=True)
    with (base_path / "data.pkl").open("rb") as stream:
        base = pickle.load(stream)["data"]
    inner, positions, item_map = training_internal_data(base)
    source = {p.name: file_hash(p) for p in Path(__file__).parent.glob("*.py")}
    manifest = {"baseline_data_id": base.manifest["data_id"], "inner": inner.manifest,
                "source": source, "max_users": args.users, "candidate_budget": 100,
                "policy": "Only original training rows; inner leave-two-out; mine inner validation positions",
                "channels": "inner-fitted ItemCF, inner category popularity, inner global popularity; RRF",
                "negative_filter": "exclude all known ORIGINAL TRAIN positives; never consult outer val/test",
                "limitation": "three-channel recall negatives; no inner SASRec or learned fusion features"}
    path = run / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Candidate preparation identity changed")
    write_json(path, manifest)
    if (run / "PREPARED.json").exists():
        for name, digest in json.loads((run / "PREPARED.json").read_text())["sha256"].items():
            if file_hash(run / name) != digest:
                raise ValueError("Prepared candidate artifact corrupted")
        return
    cf_path = run / "itemcf.npz"
    if cf_path.exists():
        with np.load(cf_path) as saved:
            cf = saved["neighbors"], saved["similarity"]
    else:
        cf = fit_cf(inner)
        with cf_path.with_suffix(".tmp").open("wb") as stream:
            np.savez(stream, neighbors=cf[0], similarity=cf[1])
        os.replace(cf_path.with_suffix(".tmp"), cf_path)
    records = inner.evaluation("val", args.users)
    if not records:
        raise ValueError("No eligible internal holdout users")
    mapped, negatives, metrics = mine(base, inner, positions, item_map, cf, records)
    for name, values in (("records.npy", mapped), ("negatives.npy", negatives)):
        path = run / name
        with path.with_suffix(".tmp").open("wb") as stream:
            np.save(stream, values)
        os.replace(path.with_suffix(".tmp"), path)
    result = {"users": len(records), **metrics, "baseline_data_id": base.manifest["data_id"],
              "sha256": {name: file_hash(run / name) for name in ("records.npy", "negatives.npy", "itemcf.npz")}}
    write_json(run / "PREPARED.json", result)
    print(json.dumps(result), flush=True)
    del base, inner, cf
    gc.collect()


if __name__ == "__main__":
    main()
