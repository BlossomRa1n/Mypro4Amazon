"""Lock the recipe, accept on untouched validation users, evaluate test, export."""
import argparse
import csv
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

from baseline_data import fingerprint
from optimization_models import score_din_cached
from optimized_inference import load_model, predict, verify_bundle
from run_baseline import score_din, write_json
from run_fusion_experiments import RECIPES
from run_optimization import paired_comparison, file_hash
from run_structural_experiments import StructuralExperiments


def same_or_copy(source, target, prefer_link=True):
    source, target = Path(source), Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if file_hash(source) != file_hash(target):
            raise ValueError("Export artifact differs: " + str(target))
        return
    if prefer_link:
        try:
            os.link(source, target)
            return
        except OSError:
            pass
    temporary = target.with_suffix(target.suffix + ".tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, target)


class FinalEvaluation(StructuralExperiments):
    def __init__(self, args):
        candidate = Path(args.candidate_run).resolve()
        selection = json.loads((candidate / "CANDIDATE_COMPLETED.json").read_text())
        super().__init__(args)
        previous = json.loads((candidate / "manifest.json").read_text())
        if previous["baseline"] != self.manifest["baseline"]:
            raise ValueError("Candidate baseline differs")
        for name, digest in previous["source"].items():
            if file_hash(Path(__file__).parent / name) != digest:
                raise ValueError("Candidate dependency changed: " + name)
        self.base_din = Path(selection["din_checkpoint"])
        self.structural = Path(args.structural_run).resolve()
        candidate_inputs = json.loads((candidate / "candidate_inputs.json").read_text())
        if file_hash(self.structural / "STRUCTURAL_COMPLETED.json") != candidate_inputs["structural_result_sha256"]:
            raise ValueError("Structural decisions differ from candidate training")
        self.original_din, self.original_v2 = self.base / "din_best.pth", self.base / "v2_best.pth"
        self.base_recipe = dict(self.recipe)
        self.recipes = {"current": self.base_recipe,
                        "with_brand": {**self.base_recipe, "brand": True},
                        "with_cf90": {**self.base_recipe, "half_life": 90}}
        self.fresh_offset = args.screen_users + args.confirm_users
        self.fresh_count = args.fresh_users
        if len(self.data.val_users) < self.fresh_offset + self.fresh_count:
            raise ValueError("Insufficient untouched validation users")
        inputs = {"candidate_result_sha256": file_hash(candidate / "CANDIDATE_COMPLETED.json"),
                  "din_sha256": self.identity(self.base_din), "recipes": self.recipes,
                  "fresh_offset": self.fresh_offset, "fresh_users": args.fresh_users,
                  "test_users": args.test_users,
                  "policy": "recipe refinement on development only; lock before fresh val; test never selects"}
        path = self.run / "final_inputs.json"
        if path.exists() and json.loads(path.read_text()) != inputs:
            raise ValueError("Final evaluation inputs changed")
        write_json(path, inputs)

    def model(self, kind, path):
        return load_model(path, self.data, self.args, kind, self.device)

    def evaluate(self, name, records, budget=100, din_path=None, v2_path=None):
        name = fingerprint(self.recipe)[:16] + "_" + name
        metrics, arrays = super().evaluate(name, records, budget, din_path, v2_path)
        if "recipe" in metrics and metrics["recipe"] != self.recipe:
            raise ValueError("Evaluation recipe differs")
        if "recipe" not in metrics:
            metrics["recipe"] = dict(self.recipe)
            write_json(self.run / "evaluations" / name / "metrics.json", metrics)
        return metrics, arrays

    def compare_original(self, name, records, proposal):
        self.recipe, self.recipe_name = proposal["recipe"], proposal["recipe_name"]
        candidate_metrics, candidate = self.evaluate(name + "_proposal", records, din_path=Path(proposal["din"]),
                                                     v2_path=Path(proposal["v2"]))
        self.recipe, self.recipe_name = RECIPES["quota"], "original_quota"
        reference_metrics, reference = self.evaluate(name + "_original", records, din_path=self.original_din,
                                                     v2_path=self.original_v2)
        return {"proposal": candidate_metrics, "original": reference_metrics,
                "paired": paired_comparison(candidate, reference)}, candidate

    def efficiency(self, records, choice):
        path = self.run / "final_efficiency.json"
        if path.exists():
            return json.loads(path.read_text())
        self.recipe, self.recipe_name = choice["recipe"], choice["recipe_name"]
        records = records[:min(1024, len(records))]
        pools = self.pools(records, 100, Path(choice["v2"]))
        model = self.model("din", Path(choice["din"]))
        times, peak, reference = {}, {}, None
        for name, scorer in (("reference", score_din), ("cached", score_din_cached)):
            scorer(model, self.data, records[:16], pools[:16], self.device)
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            samples = []
            for _ in range(3):
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                started = time.monotonic()
                ranking = scorer(model, self.data, records, pools, self.device)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                samples.append(time.monotonic() - started)
                if name == "reference":
                    reference = ranking
                elif ranking != reference:
                    raise ValueError("Final checkpoint cached rankings differ")
            times[name] = samples
            peak[name] = torch.cuda.max_memory_allocated() if self.device.type == "cuda" else None
        report = {"cached_rankings_equal": True, "users": len(records), "times": times,
                  "peak_cuda_allocated_bytes": peak, "parameters": sum(p.numel() for p in model.parameters()),
                  "cached_speedup": float(np.median(times["reference"]) / np.median(times["cached"])),
                  "note": "warm batched rerank throughput; excludes recall and network latency"}
        write_json(path, report)
        del model, pools
        gc.collect()
        torch.cuda.empty_cache()
        return report

    def export(self, choice, result, efficiency):
        release = self.run / "release"
        release.mkdir(exist_ok=True)
        files = {"din.pth": Path(choice["din"]), "v2.pth": Path(choice["v2"]),
                 "data.pkl": self.base / "data.pkl", "itemcf.pkl": self.base / "itemcf.pkl"}
        exported_paths = list(files)
        for filename, source in files.items():
            same_or_copy(source, release / filename)
        for source in Path(__file__).parent.glob("*.py"):
            same_or_copy(source, release / "code" / source.name, prefer_link=False)
            exported_paths.append("code/" + source.name)
        same_or_copy(Path(__file__).parent.parent / "requirements.txt", release / "requirements.txt", prefer_link=False)
        exported_paths.append("requirements.txt")
        sha256 = {name: file_hash(release / name) for name in exported_paths}
        manifest = {"format_version": 1, "data_id": self.data.manifest["data_id"],
                    "baseline_args": vars(self.args), "recipe": choice["recipe"], "candidate_budget": 100,
                    "sha256": sha256, "cached_inference_verified": True, "choice": choice,
                    "environment": {name: importlib.metadata.version(name)
                                    for name in ("torch", "numpy", "pandas", "scipy", "scikit-learn")},
                    "test": result, "efficiency": efficiency,
                    "protocol": self.data.manifest["protocol"],
                    "limitations": ["user-wise next-positive benchmark, not global deployment-time backtest",
                                    "static category/brand availability assumed; frozen train statistics",
                                    "latest inference uses all observed events; val/test modes reproduce evaluation"]}
        write_json(release / "bundle.json", manifest)
        verify_bundle(release)
        # Reproduce the exported test predictions before declaring the bundle usable.
        count = min(32, self.options.test_users)
        exported_csv = self.run / "bundle_test_verification.csv"
        subprocess.run([sys.executable, str(release / "code" / "optimized_inference.py"),
                        "--bundle", str(release), "--output", str(exported_csv),
                        "--split", "test", "--max-users", str(count), "--device", self.device.type],
                       cwd=release, check=True)
        with exported_csv.open(encoding="utf-8", newline="") as stream:
            exported = list(csv.DictReader(stream))
        records = self.data.evaluation("test", count)
        self.recipe, self.recipe_name = choice["recipe"], choice["recipe_name"]
        _, expected = self.evaluate("bundle_verification", records, din_path=Path(choice["din"]), v2_path=Path(choice["v2"]))
        if len(exported) != len(records):
            raise ValueError("Exported prediction count differs")
        for row, uid, ids in zip(exported, expected["uid"], expected["top5"]):
            if row["user_id"] != self.data.users[uid]:
                raise ValueError("Exported user IDs differ")
            if [row["article_" + str(i + 1)] for i in range(5)] != [self.data.items[i] if i else "" for i in ids]:
                raise ValueError("Exported inference differs from final evaluation")
        latest = predict(release, "latest", count, str(self.device))
        with (self.run / "latest_recommendations_sample.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["user_id"] + ["article_" + str(i) for i in range(1, 6)])
            writer.writeheader()
            writer.writerows(latest)
        write_json(self.run / "BUNDLE_VERIFIED.json", {"users": count, "test_predictions_equal": True,
                   "latest_inference_completed": True, "bundle_manifest_sha256": file_hash(release / "bundle.json")})

    def execute(self):
        if (self.run / "FINAL_EVALUATION_COMPLETED.json").exists():
            return
        scope_path = self.run / "scope_decisions.json"
        if not scope_path.exists():
            structural = json.loads((self.structural / "STRUCTURAL_COMPLETED.json").read_text())
            control_history = json.loads((self.structural / "experiments/structural_control_seed42/history.json").read_text())
            gate = json.loads((self.structural / "runtime_gate.json").read_text())
            evidence = {"train_loss": control_history[-1]["loss"],
                        "accepted_hr5": structural["screen"]["accepted"]["hr5"],
                        "continued_control_hr5": structural["screen"]["control"]["hr5"]}
            capacity_not_indicated = evidence["train_loss"] < .005 and evidence["continued_control_hr5"] <= evidence["accepted_hr5"]
            if not capacity_not_indicated and not self.options.max_train_steps:
                self.event("scope_review_required", reason="Mixer applicability needs evidence review", evidence=evidence)
                return
            if not gate["exact_fp32_logits_gradients_and_three_updates"]:
                raise ValueError("Raw-slice equivalence gate has not passed")
            write_json(scope_path, {"raw_slice": "implemented and exact-equivalence verified",
                       "mixer": {"decision": "defer added capacity", "evidence": evidence,
                                 "reason": "very low train loss while continued validation accuracy does not improve",
                                 "limitation": "does not prove mixers cannot help; no mixer gain is claimed"},
                       "learned_fusion": "current batch uses validated RRF; supervised four-channel fusion remains future work requiring inner-trained SASRec features",
                       "new_channels": "brand recall and history decay evaluated; no claim of standard HSTU or MIND retraining",
                       "smoke_only": bool(self.options.max_train_steps)})
        lock_path = self.run / "LOCKED_SCHEME.json"
        if not lock_path.exists():
            screens = {}
            for name, recipe in self.recipes.items():
                if not self.available():
                    return
                self.recipe, self.recipe_name = recipe, name
                screens[name], _ = self.evaluate(name + "_screen", self.screen)
            chosen = max(screens, key=lambda k: (screens[k]["hr5"], screens[k]["ndcg5"], k == "current"))
            confirmed = {}
            for name in dict.fromkeys(("current", chosen)):
                self.recipe, self.recipe_name = self.recipes[name], name
                confirmed[name] = self.evaluate(name + "_confirm", self.confirm)
            comparison = paired_comparison(confirmed[chosen][1], confirmed["current"][1])
            selected = chosen if comparison["confirmed_gain"] else "current"
            write_json(self.run / "refinement.json", {"screen": screens, "winner": chosen,
                       "selected": selected, "paired": comparison})
            choice = {"din": str(self.base_din), "v2": str(self.base_v2),
                      "recipe": self.recipes[selected], "recipe_name": selected, "budget": 100}
            # This file is written before fetching any reserved validation records.
            write_json(lock_path, {"scheme": choice, "source": self.manifest["source"],
                                  "fresh_offset": self.fresh_offset, "fresh_users": self.fresh_count})
            self.event("scheme_locked", scheme=choice)
        proposal = json.loads(lock_path.read_text())["scheme"]
        fresh = self.data.evaluation("val", self.fresh_offset + self.fresh_count)[self.fresh_offset:]
        acceptance_path = self.run / "fresh_acceptance.json"
        if not acceptance_path.exists():
            if not self.available():
                return
            report, _ = self.compare_original("fresh_validation", fresh, proposal)
            choice = proposal if report["paired"]["confirmed_gain"] else {
                "din": str(self.original_din), "v2": str(self.original_v2),
                "recipe": RECIPES["quota"], "recipe_name": "original_quota", "budget": 100}
            write_json(acceptance_path, {"comparison": report, "accepted": report["paired"]["confirmed_gain"],
                       "final_scheme": choice, "policy": "binary acceptance only; no retuning on these users"})
            self.event("fresh_validation_complete", accepted=report["paired"]["confirmed_gain"], paired=report["paired"])
        acceptance = json.loads(acceptance_path.read_text())
        choice = acceptance["final_scheme"]
        test_path = self.run / "final_test.json"
        if not test_path.exists():
            if not self.available():
                return
            records = self.data.evaluation("test", self.options.test_users)
            result, predictions = self.compare_original("final_test", records, choice)
            result["scheme"] = choice
            result["selection_policy"] = "test results never change the selected scheme"
            csv_path = self.run / "test_recommendations.csv"
            with csv_path.with_suffix(".tmp").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["user_id"] + ["article_" + str(i) for i in range(1, 6)])
                for uid, ids in zip(predictions["uid"], predictions["top5"]):
                    writer.writerow([self.data.users[uid]] + [self.data.items[i] if i else "" for i in ids])
            os.replace(csv_path.with_suffix(".tmp"), csv_path)
            write_json(test_path, result)
            self.event("final_test_complete", hr5=result["proposal"]["hr5"], paired=result["paired"])
        result = json.loads(test_path.read_text())
        efficiency = self.efficiency(fresh, choice)
        self.export(choice, result, efficiency)
        write_json(self.run / "FINAL_EVALUATION_COMPLETED.json", {"status": "evaluation_and_bundle_verified",
                   "overall_complete": False, "local_sync_verified": False,
                   "pending": ["scope_decisions_review", "report", "local_artifact_sync", "shutdown"]})
        self.event("final_evaluation_complete", local_sync_still_required=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--fusion-run", required=True)
    parser.add_argument("--structural-run", required=True)
    parser.add_argument("--candidate-run", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--screen-users", type=int, default=10000)
    parser.add_argument("--confirm-users", type=int, default=90000)
    parser.add_argument("--fresh-users", type=int, default=100000)
    parser.add_argument("--test-users", type=int, default=100000)
    parser.add_argument("--hours", type=float, default=2)
    parser.add_argument("--max-train-steps", type=int, default=0)
    args = parser.parse_args()
    if min(args.screen_users, args.confirm_users, args.fresh_users, args.test_users, args.hours) <= 0:
        parser.error("counts and hours must be positive")
    runner = FinalEvaluation(args)
    try:
        runner.execute()
    except Exception as error:
        runner.event("failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
