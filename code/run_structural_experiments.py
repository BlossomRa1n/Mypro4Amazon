"""CUDA equivalence, cached inference measurement and isolated DIN ablations."""
import argparse
import gc
import json
import os
import pickle
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from baseline_data import PrefixDataset, USER_KEYS, ITEM_KEYS, collate, fingerprint
from baseline_runtime import seed_all
from optimization_models import RawSliceDIN, score_din_cached
from run_baseline import make_model, model_config, recall, score_din, to_device, write_json
from run_fusion_experiments import Channels, RECIPES
from run_optimization import Optimization, file_hash, paired_comparison
from structural_models import StructuralDIN, VARIANTS, variant_for_experiment


class StructuralExperiments(Optimization):
    def __init__(self, args):
        fusion = Path(args.fusion_run).resolve()
        completed = json.loads((fusion / "FUSION_COMPLETED.json").read_text())
        previous = json.loads((fusion / "manifest.json").read_text())
        for name, digest in previous["source"].items():
            if file_hash(Path(__file__).parent / name) != digest:
                raise ValueError(f"Frozen fusion source differs: {name}")
        for key in ("screen_users", "confirm_users", "max_train_steps"):
            if getattr(args, key) != previous[key]:
                raise ValueError(f"Evaluation selection differs: {key}")
        super().__init__(args)
        if previous["baseline"] != self.manifest["baseline"]:
            raise ValueError("Baseline identity differs")
        selection = json.loads((fusion / "fusion_inputs.json").read_text())["selected"]
        if selection["budget"] != 100:
            raise ValueError("This continuation predeclares fixed budget 100")
        self.base_din, self.base_v2 = Path(selection["din"]), Path(selection["v2"])
        self.recipe_name = completed["selected_recipe"]
        self.recipe = RECIPES[self.recipe_name]
        self.channels = Channels(self.data, self.cf)
        self.training_variant = None
        inputs = {"fusion_sha256": file_hash(fusion / "FUSION_COMPLETED.json"),
                  "din": self.identity(self.base_din), "v2": self.identity(self.base_v2),
                  "recipe": self.recipe, "variants": VARIANTS,
                  "initialization": "same accepted DIN; fresh matched AdamW, one epoch",
                  "category32": "SVD-convert category embedding and first-layer columns before continuation",
                  "user_removal": "remove both direct user ID and its user-item product",
                  "seed_replication": "continuation seed only"}
        path = self.run / "structural_inputs.json"
        if path.exists() and json.loads(path.read_text()) != json.loads(json.dumps(inputs)):
            raise ValueError("Structural inputs changed")
        write_json(path, inputs)

    def model(self, kind, path):
        if kind == "v2":
            return super().model(kind, path)
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if saved["manifest"]["data_id"] != self.data.manifest["data_id"]:
            raise ValueError("Checkpoint data differs")
        checkpoint_variant = variant_for_experiment(saved["manifest"].get("experiment", ""))
        variant = checkpoint_variant or self.training_variant or "control"
        model = StructuralDIN(variant=variant, **model_config(self.data, self.args, "din"))
        if checkpoint_variant is None:
            model.load_baseline(saved["model"])
        else:
            model.load_state_dict(saved["model"])
        del saved
        return model.to(self.device).eval()

    def train_variant(self, variant, seed=42):
        self.training_variant = variant
        try:
            return self.train(f"structural_{variant}_seed{seed}", seed=seed)
        finally:
            self.training_variant = None

    def pools(self, records, budget, v2_path=None):
        v2_path = v2_path or self.base_v2
        key = fingerprint([self.data.manifest["data_id"], records, self.identity(v2_path), budget, self.recipe])
        path = self.run / "cache" / (key + ".pkl")
        if path.exists():
            with path.open("rb") as stream:
                return pickle.load(stream)
        self.event("structural_build_pools", users=len(records), recipe=self.recipe_name)
        model = super().model("v2", v2_path)
        _, scores = recall(model, self.data, records, self.device, budget)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        pools = []
        for record, v2 in zip(records, scores):
            channels, seen = self.channels.build(record, v2, budget, self.recipe)
            pools.append(self.channels.merge(channels, seen, budget, self.recipe))
        with path.with_suffix(".tmp").open("wb") as stream:
            pickle.dump(pools, stream, protocol=5)
        os.replace(path.with_suffix(".tmp"), path)
        return pools

    def runtime_gate(self):
        path = self.run / "runtime_gate.json"
        if path.exists():
            return
        self.event("runtime_gate_start", device=str(self.device))
        saved = torch.load(self.base_din, map_location="cpu", weights_only=False)
        original = make_model(self.data, self.args, "din", self.device)
        original.load_state_dict(saved["model"])
        sliced = RawSliceDIN(**model_config(self.data, self.args, "din")).to(self.device)
        sliced.load_state_dict(saved["model"])
        del saved
        gc.collect()
        batch = to_device(collate([PrefixDataset(self.data)[i] for i in range(min(8, len(self.data.train_positions)))]), self.device)
        user = {k: batch[k] for k in USER_KEYS}
        pos = {k: batch["pos_" + k] for k in ITEM_KEYS}
        neg = {k: batch["neg_" + k] for k in ITEM_KEYS}
        original.eval()
        sliced.eval()
        with torch.no_grad():
            a, b = original({**user, **pos}), sliced({**user, **pos})
            torch.testing.assert_close(a, b, atol=0, rtol=0)
        optimizers = [torch.optim.AdamW(model.parameters(), lr=.0001) for model in (original, sliced)]
        for step in range(3):
            losses = []
            for model, optimizer in zip((original, sliced), optimizers):
                seed_all(100 + step)
                model.train()
                optimizer.zero_grad()
                positive, negative = model.forward_bpr(user, pos, neg)
                loss = F.softplus(negative.float() - positive.float()[:, None]).mean()
                loss.backward()
                losses.append(loss.detach())
            torch.testing.assert_close(losses[0], losses[1], atol=0, rtol=0)
            for first, second in zip(original.parameters(), sliced.parameters()):
                if first.grad is not None:
                    torch.testing.assert_close(first.grad, second.grad, atol=0, rtol=0)
                else:
                    assert second.grad is None
            for optimizer in optimizers:
                optimizer.step()
            for key, value in original.state_dict().items():
                torch.testing.assert_close(value, sliced.state_dict()[key], atol=0, rtol=0)
        del optimizers, optimizer, original, sliced, model, first, second, batch, user, pos, neg, loss
        del positive, negative, losses, a, b, value
        gc.collect()
        torch.cuda.empty_cache()
        # Reload accepted weights so timing never uses the short equivalence-training weights.
        model = self.model("din", self.base_din)
        records = self.screen[:min(1024, len(self.screen))]
        pools = self.pools(self.screen, 100)[:len(records)]
        timing = {"original": [], "cached": []}
        for label, scorer in (("original", score_din), ("cached", score_din_cached)):
            scorer(model, self.data, records[:16], pools[:16], self.device)
            for _ in range(3):
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                started = time.monotonic()
                result = scorer(model, self.data, records, pools, self.device)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                timing[label].append(time.monotonic() - started)
                if label == "original":
                    reference = result
                elif result != reference:
                    raise ValueError("Cached inference changes trained-checkpoint rankings")
        report = {"device": str(self.device), "checkpoint": self.identity(self.base_din),
                  "exact_fp32_logits_gradients_and_three_updates": True,
                  "cached_rankings_equal": True, "timing_users": len(records), "seconds": timing,
                  "timing_note": "warm batched throughput, not online request latency",
                  "cached_speedup": float(np.median(timing["original"]) / np.median(timing["cached"]))}
        write_json(path, report)
        self.event("runtime_gate_complete", report=report)
        del model, pools
        gc.collect()
        torch.cuda.empty_cache()

    def execute(self):
        self.runtime_gate()
        if (self.run / "STRUCTURAL_COMPLETED.json").exists():
            return
        checkpoints = {"accepted": self.base_din}
        for variant in VARIANTS:
            path = self.train_variant(variant)
            if path is None:
                return
            checkpoints[variant] = path
        screen = {}
        for name, checkpoint in checkpoints.items():
            if not self.available():
                return
            screen[name], _ = self.evaluate(name + "_screen", self.screen, din_path=checkpoint)
        winner = max(screen, key=lambda k: (screen[k]["hr5"], screen[k]["ndcg5"], k == "accepted"))
        write_json(self.run / "screen_decision.json", {"screen": screen, "winner": winner})
        confirmed = {}
        for name in dict.fromkeys(("accepted", "control", winner)):
            if not self.available():
                return
            confirmed[name] = self.evaluate(name + "_confirm", self.confirm, din_path=checkpoints[name])
        comparison = paired_comparison(confirmed[winner][1], confirmed["accepted"][1])
        control_effect = paired_comparison(confirmed[winner][1], confirmed["control"][1])
        selected = "accepted"
        eligible = comparison["confirmed_gain"] and (winner == "control" or control_effect["confirmed_gain"])
        replication = None
        if eligible:
            replicate = self.train_variant(winner, seed=43)
            if replicate is None:
                return
            _, values = self.evaluate("winner_seed43_confirm", self.confirm, din_path=replicate)
            replication = {"against_accepted": paired_comparison(values, confirmed["accepted"][1])}
            if winner != "control":
                control43 = self.train_variant("control", seed=43)
                if control43 is None:
                    return
                _, reference = self.evaluate("control_seed43_confirm", self.confirm, din_path=control43)
                replication["against_control43"] = paired_comparison(values, reference)
            stable = replication["against_accepted"]["confirmed_gain"]
            if winner != "control":
                stable &= replication["against_control43"]["confirmed_gain"]
            if stable:
                selected = winner
        write_json(self.run / "STRUCTURAL_COMPLETED.json", {"screen": screen, "screen_winner": winner,
                   "against_accepted": comparison, "against_control": control_effect, "replication": replication,
                   "selected": selected, "din_checkpoint": str(checkpoints[selected]),
                   "fusion_recipe": self.recipe_name, "overall_complete": False})
        self.event("structural_complete", selected=selected, overall_complete=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--fusion-run", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--screen-users", type=int, default=10000)
    parser.add_argument("--confirm-users", type=int, default=90000)
    parser.add_argument("--hours", type=float, default=3)
    parser.add_argument("--max-train-steps", type=int, default=0)
    args = parser.parse_args()
    if min(args.screen_users, args.confirm_users, args.hours) <= 0:
        parser.error("counts and hours must be positive")
    runner = StructuralExperiments(args)
    try:
        runner.execute()
    except Exception as error:
        runner.event("failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
