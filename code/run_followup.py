"""Replicate core gains and compare ranking objectives without using test labels."""
import argparse
import json
import os
from pathlib import Path
import shutil

from run_baseline import write_json
from run_optimization import Optimization, file_hash, paired_comparison
from baseline_data import fingerprint


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def link_completed_artifacts(source, destination):
    """Reuse immutable completed evaluation files, never optimizer checkpoints."""
    for folder, pattern in (("cache", "*.pkl"), ("evaluations", "*/*")):
        for path in (source / folder).glob(pattern):
            if not path.is_file() or path.suffix not in (".pkl", ".json", ".npz"):
                continue
            target = destination / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                try:
                    os.link(path, target)
                except OSError:
                    shutil.copy2(path, target)


class Followup(Optimization):
    def __init__(self, args):
        self.core_run = Path(args.core_run).resolve()
        if not (self.core_run / "CORE_COMPLETED.json").exists():
            raise ValueError("Core queue is incomplete; resume it before follow-up")
        core_manifest = read_json(self.core_run / "manifest.json")
        # Reuse is allowed only when data, sampling and implementation still match.
        for name, digest in core_manifest["source"].items():
            if file_hash(Path(__file__).parent / name) != digest:
                raise ValueError(f"Frozen core source differs: {name}")
        for key in ("screen_users", "confirm_users", "max_train_steps"):
            if getattr(args, key) != core_manifest[key]:
                raise ValueError(f"Core sampling differs: {key}")
        super().__init__(args)
        if core_manifest["baseline"] != self.manifest["baseline"]:
            raise ValueError("Baseline identity differs from core")
        self.decisions = {name: read_json(self.core_run / (name + "_decision.json"))
                          for name in ("pool", "negative", "v2")}
        inputs = {name: file_hash(self.core_run / (name + "_decision.json")) for name in self.decisions}
        path = self.run / "core_inputs.json"
        if path.exists() and read_json(path) != inputs:
            raise ValueError("Core decisions changed")
        write_json(path, inputs)
        link_completed_artifacts(self.core_run, self.run)

    def evaluate(self, name, records, budget=100, din_path=None, v2_path=None):
        spec = {"records_hash": fingerprint(records), "data_id": self.data.manifest["data_id"],
                "budget": budget, "din": self.identity(din_path or self.base_din),
                "v2": self.identity(v2_path or self.base_v2)}
        destination = self.run / "evaluations" / name
        if not (destination / "metrics.json").exists():
            for existing in (self.run / "evaluations").glob("*/metrics.json"):
                if read_json(existing)["spec"] == spec:
                    destination.mkdir(exist_ok=True)
                    for filename in ("users.npz", "metrics.json"):
                        target = destination / filename
                        if not target.exists():
                            try:
                                os.link(existing.parent / filename, target)
                            except OSError:
                                shutil.copy2(existing.parent / filename, target)
                    break
        return super().evaluate(name, records, budget, din_path, v2_path)

    def comparison(self, name, candidate, reference, budget=100):
        if not self.available():
            return None
        metrics, arrays = self.evaluate(name, self.confirm, budget, **candidate)
        ref_metrics, ref_arrays = self.evaluate(name + "_reference", self.confirm, budget, **reference)
        return {"candidate": metrics, "reference": ref_metrics,
                "paired": paired_comparison(arrays, ref_arrays)}

    def followup(self):
        path = self.run / "replication.json"
        if not path.exists():
            report = {"seed": 43, "scope": "continuation seed, not independent baseline retraining",
                      "din_checkpoint": str(self.base_din), "v2_checkpoint": str(self.base_v2)}
            winner = self.decisions["negative"]["confirmed_winner"]
            if winner != "baseline":
                random = self.train("din_random_seed43", seed=43)
                if random is None:
                    return
                candidate = random
                if winner == "mixed":
                    candidate = self.train("din_mixed_seed43", seed=43, hard_count=2)
                    if candidate is None:
                        return
                    effect = self.comparison("replicate_mixed_control", {"din_path": candidate}, {"din_path": random})
                    if effect is None:
                        return
                    report["mixed_vs_random"] = effect
                result = self.comparison("replicate_din", {"din_path": candidate}, {})
                if result is None:
                    return
                report["din"] = result
                stable = result["paired"]["confirmed_gain"]
                if winner == "mixed":
                    stable &= report["mixed_vs_random"]["paired"]["hr5_delta"] > 0
                if stable:
                    report["din_checkpoint"] = self.decisions["negative"]["winner_checkpoint"]
            else:
                report["din_skipped"] = "No core DIN gain confirmed; preserve baseline, proceed to objective experiment"
            if self.decisions["v2"]["fixed_din_comparison"]["confirmed_gain"]:
                candidate = self.train("v2_extra2_seed43", kind="v2", epochs=2, seed=43)
                if candidate is None:
                    return
                result = self.comparison("replicate_v2", {"v2_path": candidate}, {})
                if result is None:
                    return
                report["v2"] = result
                if result["paired"]["confirmed_gain"]:
                    report["v2_checkpoint"] = self.decisions["v2"]["checkpoint"]
            else:
                report["v2_skipped"] = "No end-to-end core V2 gain confirmed with fixed DIN"
            write_json(path, report)
            self.event("replication_complete", report=report)
        replicated = read_json(path)

        path = self.run / "objective_decision.json"
        if not path.exists():
            hard = 2 if self.decisions["negative"]["confirmed_winner"] == "mixed" else 0
            control_name = "din_mixed_seed42" if hard else "din_random_seed42"
            control = self.core_run / "experiments" / control_name / "best.pth"
            softmax = self.train("din_softmax_seed42", hard_count=hard, objective="softmax")
            if softmax is None:
                return
            candidate_screen, _ = self.evaluate("softmax_screen", self.screen, din_path=softmax)
            control_screen, _ = self.evaluate("softmax_control_screen", self.screen, din_path=control)
            result = self.comparison("softmax_control_confirm", {"din_path": softmax}, {"din_path": control})
            if result is None:
                return
            result.update(hard_count=hard, screen_candidate=candidate_screen, screen_control=control_screen,
                          initial="same baseline best; identical seed, optimizer, order and negatives",
                          accepted=False, checkpoint=str(softmax))
            screen_gain = (candidate_screen["hr5"], candidate_screen["ndcg5"]) > (control_screen["hr5"], control_screen["ndcg5"])
            if screen_gain and result["paired"]["confirmed_gain"]:
                control43 = self.train("din_mixed_seed43" if hard else "din_random_seed43", hard_count=hard, seed=43)
                if control43 is None:
                    return
                softmax43 = self.train("objective_softmax_seed43", hard_count=hard, seed=43, objective="softmax")
                if softmax43 is None:
                    return
                replicate = self.comparison("softmax_seed43", {"din_path": softmax43}, {"din_path": control43})
                if replicate is None:
                    return
                result["replication"] = replicate
                if replicate["paired"]["confirmed_gain"]:
                    against_accepted = self.comparison("softmax_accepted", {"din_path": softmax},
                                                       {"din_path": Path(replicated["din_checkpoint"])})
                    if against_accepted is None:
                        return
                    result["against_accepted_din"] = against_accepted
                    result["accepted"] = against_accepted["paired"]["confirmed_gain"]
            write_json(path, result)
            self.event("objective_complete", accepted=result["accepted"], paired=result["paired"])
        objective = read_json(path)

        path = self.run / "combination_decision.json"
        if not path.exists():
            din = Path(objective["checkpoint"] if objective["accepted"] else replicated["din_checkpoint"])
            v2 = Path(replicated["v2_checkpoint"])
            budget = self.decisions["pool"]["confirmed_budget"]
            if not self.available():
                return
            screen, _ = self.evaluate("combination_screen", self.screen, budget, din, v2)
            result = self.comparison("combination_confirm", {"din_path": din, "v2_path": v2}, {}, budget)
            if result is None:
                return
            baseline_metrics, baseline = self.evaluate("pool_100_confirm", self.confirm)
            metrics, candidate = self.evaluate("combination_confirm", self.confirm, budget, din, v2)
            result.update(screen=screen, against_original=paired_comparison(candidate, baseline),
                          original_baseline=baseline_metrics,
                          proposed={"din": str(din), "v2": str(v2), "budget": budget})
            result["selected"] = result["proposed"] if result["against_original"]["confirmed_gain"] else {
                "din": str(self.base_din), "v2": str(self.base_v2), "budget": 100}
            write_json(path, result)
            self.event("combination_complete", selected=result["selected"])
        write_json(self.run / "FOLLOWUP_COMPLETED.json", {
            "status": "replication_objective_combination_complete", "overall_complete": False,
            "pending": ["fusion", "structure_ablations", "raw_slice_gate", "recall_channels",
                        "fresh_validation_gate", "final_test", "sync", "shutdown"]})
        self.event("followup_complete", overall_complete=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--core-run", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--screen-users", type=int, default=10000)
    parser.add_argument("--confirm-users", type=int, default=90000)
    parser.add_argument("--hours", type=float, default=3)
    parser.add_argument("--max-train-steps", type=int, default=0)
    args = parser.parse_args()
    if min(args.screen_users, args.confirm_users, args.hours) <= 0:
        parser.error("user counts and hours must be positive")
    runner = Followup(args)
    try:
        runner.followup()
    except Exception as error:
        runner.event("failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
