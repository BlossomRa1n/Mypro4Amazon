"""Four-way controlled DIN/token reranker experiment.

The existing token suite compares two freshly initialized token models with a
previous DIN checkpoint.  This runner removes that confounding comparison:
DIN and semantic_concat are both trained from scratch, with either a random
or train-only-SVD user initialization.  Candidate pools, records, negatives,
and the compatible part of the initial state are shared explicitly.
"""
import argparse
import copy
import gc
import hashlib
import json
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
from model_ext import DINExtendedModel
from run_baseline import make_model, recall, score_din, write_json
from run_token_experiments import TokenSuite, model_config as token_model_config
from token_models import SemanticTokenDIN


def controlled_model_config(data, args, kind):
    if kind == "din":
        # run_baseline's model_config is not imported under this name because
        # the token runner has its own config helper.
        return {
            "num_users": len(data.users), "num_items": len(data.items),
            "num_brands": len(data.brands), "num_categories": len(data.categories),
            "embed_dim": int(args.dim), "brand_embed_dim": min(64, int(args.dim)),
            "hidden_dims": [256, 128, 64] if args.dim >= 128 else [64, 32],
            "hist_len": int(args.hist_len), "dropout": 0.1,
        }
    return token_model_config(data, args, "concat")


def user_svd_factors(data, factors):
    factors = np.asarray(factors, dtype=np.float32)
    user = np.zeros((len(data.users), factors.shape[1]), dtype=np.float32)
    counts = np.zeros(len(data.users), dtype=np.int64)
    train_positions = np.flatnonzero(data.train_mask)
    for uid, iid in zip(data.uid[train_positions], data.iid[train_positions]):
        if iid >= 2:
            user[int(uid)] += factors[int(iid)]
            counts[int(uid)] += 1
    valid = counts > 0
    user[valid] /= counts[valid, None]
    user[:2] = 0.
    return user


def copy_module_state(source, target, name):
    source_module = getattr(source, name)
    target_module = getattr(target, name)
    target_module.load_state_dict(copy.deepcopy(source_module.state_dict()))


def copy_shape_compatible_common_state(din, token):
    """Copy shared embeddings/attention and compatible tail-head parameters.

    The first fusion weight is not copied: DIN consumes 1358 inputs while
    semantic_concat consumes 1536. Its compatible bias and all later head
    parameters/buffers are copied explicitly.
    """
    copied = []
    for name in ("user_embedding", "item_embedding", "brand_embedding",
                 "category_embedding", "attention_layer"):
        copy_module_state(din, token, name)
        copied.extend(name + "." + key for key in getattr(din, name).state_dict())
    source = din.state_dict()
    target = token.state_dict()
    for key, value in source.items():
        if not key.startswith("mlp."):
            continue
        mapped = "head." + key[len("mlp."):]
        if mapped in target and target[mapped].shape == value.shape:
            target[mapped] = value.detach().clone()
            copied.append(mapped)
    token.load_state_dict(target)
    return copied


def apply_embedding_initialization(model, item_factors, user_factors, user_mode,
                                   random_user_state):
    with torch.no_grad():
        model.item_embedding.weight.copy_(torch.as_tensor(item_factors))
        if user_mode == "random":
            model.user_embedding.weight.copy_(random_user_state)
        elif user_mode == "svd":
            model.user_embedding.weight.copy_(torch.as_tensor(user_factors))
        else:
            raise ValueError(user_mode)
        model.item_embedding.weight[:2].zero_()
        model.user_embedding.weight[:2].zero_()


class ControlledSuite(TokenSuite):
    VARIANTS = ("din_random", "din_svd", "semantic_concat_random", "semantic_concat_svd")

    def __init__(self, args):
        super().__init__(args)
        self.user_factors = user_svd_factors(self.data, self.factors)
        self.manifest.update({
            "suite": "controlled_din_token_v2",
            "variants": list(self.VARIANTS),
            "initialization_modes": {"random": "shared DIN template user embedding",
                                     "svd": "train-only SVD item-factor mean per user"},
            "negative_count": int(args.negatives),
            "init_seed": int(args.init_seed),
            "training_rng": "reset to seed after model construction; architecture-specific dropout call counts can differ",
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "retain_checkpoints": bool(args.retain_checkpoints),
            "train_order": "PrefixDataset deterministic seed=data.seed+epoch+sample; DataLoader generator=seed+epoch",
            "shared_initial_state": "copy embeddings/attention and all shape-compatible fusion parameters/buffers; only first fusion weight is shape-incompatible",
        })
        write_json(self.run_dir / "suite_manifest.json", self.manifest)

    def _make_templates(self):
        # Templates are built on CPU.  They are copied into each fresh model,
        # so CUDA allocation does not alter the shared initialization.
        din_seed = int(self.args.init_seed)
        seed_all(din_seed)
        din = DINExtendedModel(**controlled_model_config(self.data, self.args, "din"))
        din_state = din.state_dict()
        shared_state = {}
        # User/item tables are handled separately below: item is the shared
        # train-only SVD matrix, and user is either the shared random table or
        # the shared derived SVD table.  Keeping them out of this template
        # avoids holding a second copy of the largest tensors in RAM.
        for name in ("brand_embedding", "category_embedding", "attention_layer"):
            for key, value in getattr(din, name).state_dict().items():
                shared_state[name + "." + key] = value.detach().cpu().clone()
        fusion_tail_state = {}
        for key, value in din_state.items():
            if key.startswith("mlp.") and key != "mlp.0.weight":
                fusion_tail_state[key[len("mlp."):]] = value.detach().cpu().clone()
        common_head_keys = ["head." + key for key in fusion_tail_state]
        random_user_state = din_state["user_embedding.weight"].detach().cpu().clone()
        del din
        gc.collect()
        return shared_state, fusion_tail_state, common_head_keys, random_user_state

    def _new_model(self, kind, user_mode):
        seed_all(int(self.args.init_seed) if kind == "din" else int(self.args.init_seed) + 1)
        if kind == "din":
            model = DINExtendedModel(**controlled_model_config(self.data, self.args, "din"))
        elif kind == "concat":
            model = SemanticTokenDIN(**controlled_model_config(self.data, self.args, "concat"))
        else:
            raise ValueError(kind)
        state = model.state_dict()
        for key, value in self._shared_state.items():
            if key not in state or state[key].shape != value.shape:
                raise AssertionError(f"Incompatible shared parameter: {key}")
            state[key] = value.to(dtype=state[key].dtype).clone()
        fusion_prefix = "mlp." if kind == "din" else "head."
        for suffix, value in self._fusion_tail_state.items():
            key = fusion_prefix + suffix
            if key not in state or state[key].shape != value.shape:
                raise AssertionError(f"Incompatible shared fusion parameter: {key}")
            state[key] = value.to(dtype=state[key].dtype).clone()
        model.load_state_dict(state)
        apply_embedding_initialization(model, self.factors, self.user_factors,
                                       user_mode, self.random_user_state)
        return model.to(self.device)

    def _initial_state_audit(self, model, kind, user_mode):
        """Assert copied tensors and record actual initial state for review."""
        state = model.state_dict()
        prefix = "mlp." if kind == "din" else "head."
        expected = dict(self._shared_state)
        expected.update({prefix + key: value for key, value in self._fusion_tail_state.items()})
        hashes = {}
        for key, value in state.items():
            actual = value.detach().cpu().contiguous()
            if key in expected and not torch.equal(actual, expected[key]):
                raise AssertionError(f"Shared initialization differs: {key}")
            # memoryview avoids making another large byte copy of embeddings.
            hashes[key] = hashlib.sha256(memoryview(actual.numpy()).cast("B")).hexdigest()
        return {"kind": kind, "user_init": user_mode,
                "copied_parameters_verified": sorted(expected),
                "state_sha256": hashes,
                "parameter_count": sum(p.numel() for p in model.parameters()),
                "training_seed": int(self.args.seed),
                "cpu_rng_sha256": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
                "cuda_rng_sha256": [hashlib.sha256(s.cpu().numpy().tobytes()).hexdigest()
                                    for s in torch.cuda.get_rng_state_all()] if self.device.type == "cuda" else [],
                "dropout_note": "same initial RNG state; extra token dropout calls prevent identical masks across architectures"}

    def evaluate(self, model, records, pools, label):
        metrics, arrays = super().evaluate(model, records, pools, label)
        # Preserve paired-row identities, not just equal array lengths.
        arrays["uid"] = np.asarray([record[0] for record in records], dtype=np.int64)
        arrays["position"] = np.asarray([record[1] for record in records], dtype=np.int64)
        np.savez_compressed(self.run_dir / f"{label}_users.npz", **arrays)
        return metrics, arrays

    def _train_one(self, variant, records, pools):
        if variant.startswith("semantic_concat_"):
            kind, user_mode = "concat", variant[len("semantic_concat_"):]
        else:
            kind, user_mode = "din", variant[len("din_"):]
        directory = self.run_dir / variant
        directory.mkdir(exist_ok=True)
        completed = directory / "COMPLETED.json"
        if completed.exists() and (directory / "best.pth").exists():
            model = self._new_model(kind, user_mode)
            saved = torch.load(directory / "best.pth", map_location=self.device, weights_only=False)
            model.load_state_dict(saved["model"])
            return model, load_json(directory / "history.json")

        # Reset the training RNG for every variant.  Constructors consume RNG
        # while creating modules even though their states are replaced by the
        # shared templates; resetting here keeps dropout and other training
        # randomness comparable across the four groups.
        model = self._new_model(kind, user_mode)
        seed_all(self.args.seed)
        write_json(directory / "initial_state_audit.json",
                   self._initial_state_audit(model, kind, user_mode))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.args.epochs, eta_min=1e-6)
        dataset = PrefixDataset(self.data, self.args.negatives)
        history, best = [], -1.
        manifest = {"variant": variant, "kind": kind, "user_init": user_mode,
                    "seed": self.args.seed, "init_seed": self.args.init_seed,
                    "epochs": self.args.epochs, "source": self.source,
                    "model_config": controlled_model_config(self.data, self.args, kind),
                    "negative_count": self.args.negatives,
                    "shared_head_keys": self._common_head_keys}
        for epoch in range(self.args.epochs):
            dataset.epoch = epoch
            generator = torch.Generator().manual_seed(self.args.seed + epoch)
            loader = DataLoader(
                dataset, batch_size=self.args.din_batch, shuffle=True,
                generator=generator, num_workers=self.args.workers,
                collate_fn=collate, pin_memory=self.device.type == "cuda",
                drop_last=True, persistent_workers=self.args.workers > 0)
            model.train()
            total, steps = 0., 0
            started = time.monotonic()
            for batch in tqdm(loader, desc=f"{variant} epoch {epoch + 1}/{self.args.epochs}",
                              mininterval=20):
                batch = {key: value.to(self.device, non_blocking=True)
                         for key, value in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16,
                                    enabled=self.device.type == "cuda"):
                    user = {key: batch[key] for key in USER_KEYS}
                    positive = {key: batch["pos_" + key] for key in ITEM_KEYS}
                    negative = {key: batch["neg_" + key] for key in ITEM_KEYS}
                    pos, neg = model.forward_bpr(user, positive, negative)
                    loss = F.softplus(neg.float() - pos.float()[:, None]).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"{variant}: nonfinite loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                optimizer.step()
                total += float(loss.detach())
                steps += 1
                if self.args.max_train_steps and steps >= self.args.max_train_steps:
                    break
            if not steps:
                raise ValueError("No complete training batches; reduce batch size")
            scheduler.step()
            model.eval()
            metrics, _ = self.evaluate(model, records, pools, f"{variant}_screen_epoch{epoch + 1}")
            entry = {"epoch": epoch + 1, "loss": total / steps, "steps": steps,
                     "seconds": time.monotonic() - started, "metrics": metrics}
            history.append(entry)
            if metrics["din"]["hr"] > best:
                best = metrics["din"]["hr"]
                torch.save({"model": model.state_dict(), "manifest": manifest,
                            "epoch": epoch + 1, "metrics": metrics}, directory / "best.pth")
            write_json(directory / "history.json", history)
        write_json(completed, {"status": "complete", "variant": variant,
                               "best_hr5": best, "manifest": manifest})
        saved = torch.load(directory / "best.pth", map_location=self.device, weights_only=False)
        model.load_state_dict(saved["model"])
        model.eval()
        return model, history

    def run(self):
        screen = self.records("val", self.args.screen_users)
        test = self.records("test", self.args.test_users)
        screen_pools = self.pools(screen, "screen")
        test_pools = self.pools(test, "test")
        self.manifest["evaluation_records"] = {
            "screen": fingerprint(screen), "test": fingerprint(test)}
        self.manifest["candidate_cache_files"] = sorted(path.name for path in self.cache_dir.glob("pools_*.pkl"))
        write_json(self.run_dir / "suite_manifest.json", self.manifest)
        # Candidate pools are now fully materialized and cached.  The V2
        # recall model is no longer needed for the four ranker trainings;
        # release it before constructing the large DIN/token templates.
        del self.v2
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        self._shared_state, self._fusion_tail_state, self._common_head_keys, self.random_user_state = self._make_templates()
        results = {"manifest": self.manifest, "variants": {}, "comparisons": {}}
        arrays = {}
        for variant in self.VARIANTS:
            model, history = self._train_one(variant, screen, screen_pools)
            screen_metrics, screen_arrays = self.evaluate(
                model, screen, screen_pools, variant + "_screen_final")
            test_metrics, test_arrays = self.evaluate(
                model, test, test_pools, variant + "_test_final")
            arrays[variant + "_screen_final"] = screen_arrays
            arrays[variant + "_test_final"] = test_arrays
            results["variants"][variant] = {
                "history": history, "screen": screen_metrics, "test": test_metrics,
            }
            if not self.args.retain_checkpoints:
                checkpoint = self.run_dir / variant / "best.pth"
                if checkpoint.exists():
                    checkpoint.unlink()
            del model
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        def compare(left, right, split):
            left_arrays, right_arrays = arrays[f"{left}_{split}_final"], arrays[f"{right}_{split}_final"]
            for key in ("uid", "position", "pool_hit"):
                if not np.array_equal(left_arrays[key], right_arrays[key]):
                    raise AssertionError(f"Paired evaluation differs: {split}/{key}")
            return self.paired(left_arrays, right_arrays)

        for split in ("screen", "test"):
            results["comparisons"][split] = {
                "semantic_concat_random_vs_din_random": compare("semantic_concat_random", "din_random", split),
                "semantic_concat_svd_vs_din_svd": compare("semantic_concat_svd", "din_svd", split),
                "din_svd_vs_din_random": compare("din_svd", "din_random", split),
                "semantic_concat_svd_vs_semantic_concat_random": compare("semantic_concat_svd", "semantic_concat_random", split),
            }
        write_json(self.run_dir / "results.json", results)
        write_json(self.run_dir / "COMPLETED.json", {
            "status": "complete", "suite": self.manifest["suite"],
            "variants": list(self.VARIANTS), "results": "results.json"})


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run", required=True)
    parser.add_argument("--cf-run", default="")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--screen-users", type=int, default=100000)
    parser.add_argument("--test-users", type=int, default=100000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--token-dim", type=int, default=256)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--hist-len", type=int, default=50)
    parser.add_argument("--negatives", type=int, default=4)
    parser.add_argument("--din-batch", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--microbatch", type=int, default=512)
    parser.add_argument("--max-train-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--init-seed", type=int, default=424242)
    parser.add_argument("--candidates", type=int, default=75)
    parser.add_argument("--fusion-mode", choices=("quota", "rrf"), default="rrf")
    parser.add_argument("--fusion-weights", type=float, nargs=4,
                        default=[2.0, 1.0, 0.7, 0.05])
    parser.add_argument("--itemcf-half-life-days", type=float, default=180.)
    parser.add_argument("--retain-checkpoints", action="store_true",
                        help="Keep each best.pth; default deletes it after final evaluation to limit disk use")
    args = parser.parse_args()
    if args.epochs < 1 or args.token_dim < 1 or args.screen_users < 1 or args.test_users < 1:
        parser.error("epochs, token-dim and user counts must be positive")
    ControlledSuite(args).run()


if __name__ == "__main__":
    main()
