"""Controlled single-token ablations for semantic_concat.

Every variant keeps the six token slots and parameter shapes unchanged. One
slot is zeroed after token-type offsets are added, so the comparison measures
the contribution of that source while preserving the same optimizer and head.
The negative policy is fixed to the selected 12-random + 4-medium mixed set.
"""
import hashlib
from pathlib import Path

from run_semantic_negative_screen import NegativeScreenSuite
from run_baseline import write_json


class TokenAblationSuite(NegativeScreenSuite):
    VARIANTS = (
        "full",
        "zero_seq",
        "zero_user",
        "zero_item",
        "zero_context",
        "zero_cross",
        "zero_dense",
    )
    SPECS = {
        name: {"negatives": 16, "policy": "mixed_rrf", "loss": "bpr"}
        for name in VARIANTS
    }
    ABLATIONS = {
        "full": (),
        "zero_seq": ("seq",),
        "zero_user": ("user",),
        "zero_item": ("item",),
        "zero_context": ("context",),
        "zero_cross": ("cross",),
        "zero_dense": ("dense",),
    }

    def __init__(self, args):
        super().__init__(args)
        self.VARIANTS = tuple(self.VARIANTS)
        self.manifest.update({
            "suite": "semantic_concat_token_ablation_v1",
            "variants": list(self.VARIANTS),
            "fixed_architecture": "semantic_concat",
            "fixed_user_initialization": "shared random user embedding",
            "negative_policy": "mixed_rrf: 12 random + 2 ranks 11-25 + 2 ranks 26-50",
            "ablation_contract": "six fixed token slots; one slot zeroed after token-type offsets",
            "ablation_map": self.ABLATIONS,
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        })
        write_json(self.run_dir / "suite_manifest.json", self.manifest)

    def _new_model(self, kind, user_mode):
        model = super()._new_model(kind, user_mode)
        if kind != "concat":
            raise ValueError("token ablation only supports semantic_concat")
        model.ablate_tokens = tuple(self.ABLATIONS[self._active_variant])
        return model

    def _train_one(self, variant, records, pools):
        self._active_variant = variant
        return super()._train_one(variant, records, pools)


def main():
    parser = __import__("argparse").ArgumentParser(description=__doc__)
    parser.add_argument("--base-run", required=True)
    parser.add_argument("--cf-run", default="")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--screen-users", type=int, default=20000)
    parser.add_argument("--test-users", type=int, default=20000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--token-dim", type=int, default=256)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--hist-len", type=int, default=50)
    parser.add_argument("--din-batch", type=int, default=256)
    parser.add_argument("--negatives", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--microbatch", type=int, default=512)
    parser.add_argument("--max-train-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--init-seed", type=int, default=424242)
    parser.add_argument("--candidates", type=int, default=75)
    parser.add_argument("--fusion-mode", choices=("quota", "rrf"), default="rrf")
    parser.add_argument("--fusion-weights", type=float, nargs=4, default=[2.0, 1.0, 0.7, 0.05])
    parser.add_argument("--itemcf-half-life-days", type=float, default=180.)
    parser.add_argument("--retain-checkpoints", action="store_true")
    parser.add_argument("--training-pools", required=True)
    args = parser.parse_args()
    TokenAblationSuite(args).run()


if __name__ == "__main__":
    main()
