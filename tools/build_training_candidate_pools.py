"""Build train-fit medium-negative pools for every training prefix.

The training pool uses ItemCF/category/hot scores fit on the complete train
split, with the V2 channel intentionally empty.  This is suitable for a fair
relative hard-negative screen, but it is not a strict prefix-causal pool:
other training targets influence the ItemCF and popularity statistics.  The
limitation is recorded in the output manifest; evaluation pools remain the
fixed four-channel production pools.
"""
import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm

from run_baseline import candidate_pools


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-run", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--budget", type=int, default=75)
    p.add_argument("--chunk", type=int, default=2000)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--stop", type=int, default=0,
                   help="Exclusive row stop; zero means all rows")
    args = p.parse_args()
    base = Path(args.base_run).resolve()
    with (base / "data.pkl").open("rb") as f:
        data = pickle.load(f)["data"]
    with (base / "itemcf.pkl").open("rb") as f:
        cf = pickle.load(f)
    records = [(int(data.uid[pos]), int(pos)) for pos in data.train_positions]
    start_row = max(0, args.start)
    stop_row = min(len(records), args.stop or len(records))
    if start_row >= stop_row:
        raise ValueError("empty training-pool range")
    pools = np.zeros((stop_row - start_row, args.budget), dtype=np.int32)
    for start in range(start_row, stop_row, args.chunk):
        part = records[start:min(start + args.chunk, stop_row)]
        # Empty V2 channel keeps this screen causal and inexpensive.  The
        # production evaluation still uses the fixed V2+RRF candidate pool.
        merged, _ = candidate_pools(
            data, part, [{} for _ in part], cf, args.budget, "rrf", 180.,
            [2.0, 0.0, 0.7, 0.05],
        )
        for row, pool in enumerate(merged, start - start_row):
            ids = np.asarray(list(pool), dtype=np.int32)[:args.budget]
            pools[row, :len(ids)] = ids
        print(f"built {min(start + len(part), stop_row)}/{stop_row}", flush=True)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    np.save(tmp, pools)
    tmp_path = Path(str(tmp) + ".npy") if not tmp.exists() else tmp
    tmp_path.replace(output)
    manifest = {
        "data_id": data.manifest["data_id"], "train_samples": len(records),
        "start": start_row, "stop": stop_row,
        "budget": args.budget, "candidate_policy": "rrf_itemcf_category_hot",
        "fit_scope": "entire train split; not prefix-causal",
        "v2_weight": 0.0, "itemcf_half_life_days": 180.0,
        "weights": [2.0, 0.0, 0.7, 0.05], "pool_sha256": sha256(output),
        "rank_bands": [[11, 25, 2], [26, 50, 2]],
    }
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
