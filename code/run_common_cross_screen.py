"""Common-negative 4/16 control plus normalized/gated cross-token screen."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch

from run_semantic_negative_screen import NegativeScreenSuite
from run_baseline import write_json


class CommonCrossSuite(NegativeScreenSuite):
    VARIANTS = (
        "random4_common",
        "random16_common",
        "cross_raw",
        "cross_normalized",
        "cross_gated_normalized",
    )
    SPECS = {
        "random4_common": {"negatives": 4, "policy": "random", "loss": "bpr", "cross_mode": "raw"},
        "random16_common": {"negatives": 16, "policy": "random", "loss": "bpr", "cross_mode": "raw"},
        "cross_raw": {"negatives": 16, "policy": "mixed_rrf", "loss": "bpr", "cross_mode": "raw"},
        "cross_normalized": {"negatives": 16, "policy": "mixed_rrf", "loss": "bpr", "cross_mode": "normalized"},
        "cross_gated_normalized": {"negatives": 16, "policy": "mixed_rrf", "loss": "bpr", "cross_mode": "normalized_gated"},
    }

    def __init__(self, args):
        super().__init__(args)
        self.VARIANTS = type(self).VARIANTS
        self._prepare_common_pool()
        self.manifest.update({
            "suite": "common_negative_cross_screen_v1",
            "variants": list(self.VARIANTS),
            "variant_specs": self.SPECS,
            "common_negative_pool": "one fixed randomized 16-negative row per train prefix; 4 uses first 4",
            "common_negative_pool_sha256": self.fixed_random_pool_sha256,
            "common_negative_pool_shape": list(self.fixed_random_pools.shape),
            "common_negative_pool_data_id": self.data_id,
            "common_negative_pool_validation_rows": self._pool_validation_rows,
            "negative_order": "preserved RNG order, unsorted",
            "cross_modes": {v: self.SPECS[v]["cross_mode"] for v in self.VARIANTS},
            "cross_gate": "all variants retain scalar gate; raw/normalized sigmoid~1, gated normalized starts 0.05",
        })
        write_json(self.run_dir / "suite_manifest.json", self.manifest)

    def _prepare_common_pool(self):
        path = self.run_dir / "common_random16.npy"
        if path.exists():
            pool = np.load(path, mmap_mode="r")
        else:
            pool = np.empty((len(self.data.train_positions), 16), dtype=np.int32)
            for row, pos in enumerate(self.data.train_positions):
                uid, target = int(self.data.uid[pos]), int(self.data.iid[pos])
                rng = np.random.default_rng(np.random.SeedSequence([self.data.seed, row]))
                # BenchmarkData.negatives performs rejection sampling over the
                # active catalog; permute its sorted result to avoid an ID-order
                # artifact while keeping the 4-prefix a true subset of 16.
                pool[row] = rng.permutation(self.data.negatives(uid, target, 16, rng))
            np.save(path, pool)
        if pool.shape != (len(self.data.train_positions), 16):
            raise ValueError(f"common pool shape {pool.shape}")
        self._pool_validation_rows = sorted(set([0, len(pool) // 2, len(pool) - 1] + list(range(min(len(pool), 1000)))) )
        for row in self._pool_validation_rows:
            uid, pos = int(self.data.uid[self.data.train_positions[row]]), int(self.data.train_positions[row])
            target = int(self.data.iid[pos]); vals = np.asarray(pool[row])
            if len(set(vals.tolist())) != 16 or any(int(x) < 2 or int(x) in self.data.train_sets[uid] or int(x) == target for x in vals):
                raise AssertionError("invalid common negative pool row")
        h = hashlib.sha256()
        with Path(path).open("rb") as f:
            for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
                h.update(block)
        self.fixed_random_pools = pool
        self.fixed_random_pool_sha256 = h.hexdigest()

    def _new_model(self, kind, user_mode):
        model = super()._new_model(kind, user_mode)
        spec = self.SPECS[self._active_variant]
        mode = spec["cross_mode"]
        model.cross_mode = mode
        with torch.no_grad():
            model.cross_gate_logit.fill_(-2.944439 if "gated" in mode else 10.0)
        return model

    def _train_one(self, variant, records, pools):
        self._active_variant = variant
        return super()._train_one(variant, records, pools)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-run", required=True); p.add_argument("--cf-run", default=""); p.add_argument("--run-dir", required=True)
    p.add_argument("--screen-users", type=int, default=20000); p.add_argument("--test-users", type=int, default=20000)
    p.add_argument("--epochs", type=int, default=3); p.add_argument("--token-dim", type=int, default=256)
    p.add_argument("--dim", type=int, default=256); p.add_argument("--hist-len", type=int, default=50)
    p.add_argument("--din-batch", type=int, default=256); p.add_argument("--negatives", type=int, default=16)
    p.add_argument("--workers", type=int, default=4); p.add_argument("--microbatch", type=int, default=512)
    p.add_argument("--seed", type=int, default=42); p.add_argument("--init-seed", type=int, default=424242)
    p.add_argument("--max-train-steps", type=int, default=0); p.add_argument("--retain-checkpoints", action="store_true")
    p.add_argument("--candidates", type=int, default=75); p.add_argument("--fusion-mode", default="rrf")
    p.add_argument("--fusion-weights", type=float, nargs=4, default=[2.0, 1.0, 0.7, 0.05])
    p.add_argument("--itemcf-half-life-days", type=float, default=180.); p.add_argument("--training-pools", required=True)
    args = p.parse_args(); CommonCrossSuite(args).run()


if __name__ == "__main__": main()
