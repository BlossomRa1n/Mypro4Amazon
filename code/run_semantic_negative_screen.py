"""Screen semantic_concat negative-sampling and listwise-loss variants.

All variants use the same semantic_concat model, random user initialization,
data subset, positive rows, initialization, order and evaluation pools.  The
script is intentionally separate from the four-way architecture suite so a
negative-sampling result cannot be confused with a token-architecture result.
"""
import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from baseline_data import ITEM_KEYS, USER_KEYS, PrefixDataset, collate, fingerprint
from baseline_runtime import sampled_softmax_loss, seed_all
from run_controlled_token_experiments import ControlledSuite, load_json
from run_baseline import write_json


class NegativeScreenSuite(ControlledSuite):
    """A single-architecture suite with fixed random user initialization."""

    VARIANTS = (
        "concat_4_random_bpr",
        "concat_16_random_bpr",
    )

    SPECS = {
        "concat_4_random_bpr": {"negatives": 4, "policy": "random", "loss": "bpr"},
        "concat_16_random_bpr": {"negatives": 16, "policy": "random", "loss": "bpr"},
        "concat_16_mixed_bpr": {"negatives": 16, "policy": "mixed_rrf", "loss": "bpr"},
        "concat_16_mixed_softmax": {"negatives": 16, "policy": "mixed_rrf", "loss": "softmax"},
    }

    def __init__(self, args):
        super().__init__(args)
        self.training_pools = None
        if args.training_pools and type(self) is NegativeScreenSuite:
            self.training_pools = np.load(args.training_pools, mmap_mode="r")
            if len(self.training_pools) != len(self.data.train_positions):
                raise ValueError("training pool count does not match train positions")
            self.VARIANTS = self.VARIANTS + (
                "concat_16_mixed_bpr", "concat_16_mixed_softmax")
        elif args.training_pools:
            self.training_pools = np.load(args.training_pools, mmap_mode="r")
            if len(self.training_pools) != len(self.data.train_positions):
                raise ValueError("training pool count does not match train positions")
        self.manifest.update({
            "suite": "semantic_concat_negative_screen_v1",
            "variants": list(self.VARIANTS),
            "fixed_architecture": "semantic_concat",
            "fixed_user_initialization": "shared random user embedding",
            "variant_specs": self.SPECS,
            "training_pools": str(args.training_pools or ""),
            "training_pool_fit_scope": (
                "entire train split ItemCF/category/hot fit; target and known positives filtered "
                "from sampled rows; exploratory hard-negative screen, not prefix-causal"
                if args.training_pools else "none"),
        })
        write_json(self.run_dir / "suite_manifest.json", self.manifest)

    def _train_one(self, variant, records, pools):
        spec = self.SPECS[variant]
        directory = self.run_dir / variant
        directory.mkdir(exist_ok=True)
        completed = directory / "COMPLETED.json"
        model = self._new_model("concat", "random")
        seed_all(self.args.seed)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.args.epochs, eta_min=1e-6)
        dataset = PrefixDataset(
            self.data, spec["negatives"], negative_policy=spec["policy"],
            candidate_pools=self.training_pools if spec["policy"] == "mixed_rrf" else None,
        )
        history = []
        manifest = {
            "variant": variant, "spec": spec, "seed": self.args.seed,
            "init_seed": self.args.init_seed, "epochs": self.args.epochs,
            "data_id": self.data_id, "source": self.source,
            "model_config": {"fusion": "concat", "user_init": "random"},
            "train_samples": len(dataset),
        }
        write_json(directory / "manifest.json", manifest)
        for epoch in range(self.args.epochs):
            dataset.epoch = epoch
            loader = DataLoader(
                dataset, batch_size=self.args.din_batch, shuffle=True,
                generator=torch.Generator().manual_seed(self.args.seed + epoch),
                num_workers=self.args.workers, collate_fn=collate,
                pin_memory=self.device.type == "cuda", drop_last=True,
                persistent_workers=self.args.workers > 0,
            )
            model.train(); total = 0.; steps = 0
            started = time.monotonic()
            for batch in tqdm(loader, desc=f"{variant} epoch {epoch + 1}/{self.args.epochs}", mininterval=20):
                batch = {key: value.to(self.device, non_blocking=True) for key, value in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16,
                                    enabled=self.device.type == "cuda"):
                    user = {key: batch[key] for key in USER_KEYS}
                    pos = {key: batch["pos_" + key] for key in ITEM_KEYS}
                    neg = {key: batch["neg_" + key] for key in ITEM_KEYS}
                    positive, negative = model.forward_bpr(user, pos, neg)
                    if spec["loss"] == "bpr":
                        loss = F.softplus(negative.float() - positive.float()[:, None]).mean()
                    else:
                        loss = sampled_softmax_loss(positive.float(), negative.float())
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"nonfinite loss: {variant}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                optimizer.step(); total += float(loss.detach()); steps += 1
                if self.args.max_train_steps and steps >= self.args.max_train_steps:
                    break
            if not steps:
                raise ValueError("No complete training batches")
            scheduler.step(); model.eval()
            metrics, _ = self.evaluate(model, records, pools, f"{variant}_screen_epoch{epoch + 1}")
            history.append({"epoch": epoch + 1, "loss": total / steps, "steps": steps,
                            "seconds": time.monotonic() - started, "metrics": metrics})
            write_json(directory / "history.json", history)
        torch.save({"model": model.state_dict(), "manifest": manifest,
                    "history": history}, directory / "best.pth")
        write_json(completed, {"status": "complete", "variant": variant,
                               "manifest": manifest})
        return model, history

    def run(self):
        screen = self.records("val", self.args.screen_users)
        test = self.records("test", self.args.test_users)
        screen_pools = self.pools(screen, "screen")
        test_pools = self.pools(test, "test")
        self.manifest["evaluation_records"] = {"screen": fingerprint(screen), "test": fingerprint(test)}
        write_json(self.run_dir / "suite_manifest.json", self.manifest)
        del self.v2; gc.collect()
        if self.device.type == "cuda": torch.cuda.empty_cache()
        self._shared_state, self._fusion_tail_state, self._common_head_keys, self.random_user_state = self._make_templates()
        results = {"manifest": self.manifest, "variants": {}}
        arrays = {}
        for variant in self.VARIANTS:
            model, history = self._train_one(variant, screen, screen_pools)
            screen_metrics, screen_arrays = self.evaluate(model, screen, screen_pools, variant + "_screen_final")
            test_metrics, test_arrays = self.evaluate(model, test, test_pools, variant + "_test_final")
            arrays[variant + "_screen_final"] = screen_arrays
            arrays[variant + "_test_final"] = test_arrays
            results["variants"][variant] = {"history": history, "screen": screen_metrics, "test": test_metrics}
            if not self.args.retain_checkpoints:
                (self.run_dir / variant / "best.pth").unlink(missing_ok=True)
            del model; gc.collect()
            if self.device.type == "cuda": torch.cuda.empty_cache()
        write_json(self.run_dir / "results.json", results)
        write_json(self.run_dir / "COMPLETED.json", {"status": "complete", "variants": list(self.VARIANTS), "results": "results.json"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-run", required=True); parser.add_argument("--cf-run", default="")
    parser.add_argument("--run-dir", required=True); parser.add_argument("--screen-users", type=int, default=20000)
    parser.add_argument("--test-users", type=int, default=20000); parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--token-dim", type=int, default=256); parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--hist-len", type=int, default=50); parser.add_argument("--din-batch", type=int, default=256)
    parser.add_argument("--negatives", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4); parser.add_argument("--microbatch", type=int, default=512)
    parser.add_argument("--max-train-steps", type=int, default=0); parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--init-seed", type=int, default=424242); parser.add_argument("--candidates", type=int, default=75)
    parser.add_argument("--fusion-mode", choices=("quota", "rrf"), default="rrf")
    parser.add_argument("--fusion-weights", type=float, nargs=4, default=[2.0, 1.0, 0.7, 0.05])
    parser.add_argument("--itemcf-half-life-days", type=float, default=180.)
    parser.add_argument("--retain-checkpoints", action="store_true")
    parser.add_argument("--training-pools", default="",
                        help="Optional [train_samples,75] causal candidate pools for mixed variants")
    args = parser.parse_args(); NegativeScreenSuite(args).run()


if __name__ == "__main__": main()
