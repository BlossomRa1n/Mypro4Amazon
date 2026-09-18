import argparse
import copy
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import auto_select_future_window as selection


def row(uid, targets, items, candidate_hit=1):
    return {"user_id": uid, "targets": targets, "target_count": len(targets),
            "items": items, "candidate_hit": candidate_hit}


class PairedSelectionTests(unittest.TestCase):
    def test_larger_baseline_is_recomputed_on_candidate_users(self):
        baseline = {"a": row("a", ["x", "y"], ["wrong"]),
                    "b": row("b", ["q"], ["q"])}
        for i in range(8):
            baseline[str(i)] = row(str(i), ["z"], ["z"])
        candidate = {"b": row("b", ["q"], ["q"]),
                     "a": row("a", ["y", "x"], ["x", "y"])}
        result = selection.compare_predictions(baseline, candidate)
        self.assertEqual(result["status"], "accepted")
        self.assertEqual(result["baseline_available_users"], 10)
        self.assertEqual(result["aligned_users"], 2)
        self.assertEqual(result["baseline_val"]["din"]["hr"], .5)
        self.assertEqual(result["candidate_val"]["din"]["hr"], 1.)
        self.assertEqual(result["candidate_val"]["din"]["recall"], 1.)
        self.assertEqual(result["candidate_val"]["din"]["ndcg"], 1.)
        self.assertEqual(result["paired_hr5"]["gained"], 1)
        self.assertEqual(result["paired_hr5"]["lost"], 0)
        self.assertEqual(len(result["paired_hr5"]["ci95"]), 2)
        self.assertEqual(result, selection.compare_predictions(baseline, candidate))

    def test_higher_hr_cannot_hide_ndcg_regression(self):
        baseline = {str(i): row(str(i), ["x"], ["x"] if i < 2 else ["a"]) for i in range(3)}
        candidate = {str(i): row(str(i), ["x"], ["a", "b", "c", "d", "x"]) for i in range(3)}
        result = selection.compare_predictions(baseline, candidate)
        self.assertGreater(result["candidate_val"]["din"]["hr"], result["baseline_val"]["din"]["hr"])
        self.assertLess(result["candidate_val"]["din"]["ndcg"], result["baseline_val"]["din"]["ndcg"])
        self.assertEqual(result["status"], "rejected")

    def test_coverage_alone_cannot_accept_tied_din_hr(self):
        baseline = {"a": row("a", ["x"], ["a"], candidate_hit=0)}
        candidate = {"a": row("a", ["x"], ["a"], candidate_hit=1)}
        self.assertEqual(selection.compare_predictions(baseline, candidate)["status"], "rejected")

    def test_missing_users_and_full_target_changes_fail_closed(self):
        baseline = {"a": row("a", ["x", "y"], ["x"])}
        for candidate in ({"b": row("b", ["x"], ["x"])}, {"a": row("a", ["x"], ["x"])}):
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                selection.compare_predictions(baseline, candidate)

    def test_multitarget_ndcg_recall_and_paired_changes(self):
        baseline = {"a": row("a", ["x", "y"], ["x"]),
                    "b": row("b", ["x", "y"], ["a"])}
        candidate = {"a": row("a", ["x", "y"], ["a"]),
                     "b": row("b", ["x", "y"], ["a", "x", "y"])}
        result = selection.compare_predictions(baseline, candidate)
        expected_ndcg = ((1 / math.log2(3) + 1 / math.log2(4)) / (1 + 1 / math.log2(3))) / 2
        self.assertAlmostEqual(result["candidate_val"]["din"]["ndcg"], expected_ndcg)
        self.assertEqual(result["candidate_val"]["din"]["recall"], .5)
        self.assertEqual(result["paired_hr5"]["gained"], 1)
        self.assertEqual(result["paired_hr5"]["lost"], 1)
        self.assertEqual(result["paired_hr5"]["difference"], 0.)


class SelectionWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.baseline, self.candidate, self.final = [self.root / name for name in ("base", "candidate", "final")]
        self.source = Path(__file__).resolve().parents[1]
        self.data = self.root / "data"
        self.base_rows = [row("a", ["x", "y"], ["q"], candidate_hit=0), row("b", ["z"], ["z"])]
        self.trial_rows = [row("a", ["x", "y"], ["x", "y"])]
        common = {"protocol": "future-window", "future_test_users": 10, "sample_users": 30,
                  "seed": 42, "hist_len": 5, "dim": 8, "candidates": 100, "cf_neighbors": 10,
                  "data_dir": str(self.data), "epochs": 1, "negatives": 4, "eval_users": 1,
                  "v2_batch": 8, "din_batch": 4, "workers": 0}
        self.base_spec = {"args": dict(common, run_dir=str(self.baseline), final_users=2,
                                       validation_only=False, fusion_mode="quota",
                                       itemcf_half_life_days=0., eval_only_run=""), "code_hash": "source"}
        self.trial_spec = {"args": dict(common, run_dir=str(self.candidate), final_users=1,
                                        validation_only=True, fusion_mode="rrf",
                                        itemcf_half_life_days=90., eval_only_run=str(self.baseline)), "code_hash": "source"}
        base_id = selection.fingerprint(self.base_spec)
        self.base_result = self.result(self.base_spec, self.base_rows,
                                      {"run_dir": str(self.baseline), "run_identity": base_id})
        self.trial_result = self.result(self.trial_spec, self.trial_rows,
                                       {"run_dir": str(self.baseline), "run_identity": base_id,
                                        "checkpoint_run_identities": [base_id, base_id]})
        self.write_fixture()
        for kind in ("v2", "din"):
            torch.save({"manifest": {"data_id": "same-data", "kind": kind, "run_identity": base_id},
                        "model": {"weight": torch.ones(1)}}, self.baseline / (kind + "_best.pth"))
        for name in ("data.pkl", "svd.npy", "itemcf.pkl"):
            (self.baseline / name).write_bytes(b"fixture")
        self.args = argparse.Namespace(baseline_dir=str(self.baseline), candidate_dir=str(self.candidate),
                                       final_dir=str(self.final), data_dir=str(self.data), source_dir=str(self.source))

    def result(self, spec, rows, source):
        val, _ = selection.metrics(rows)
        val["candidate_pool"]["k"] = 100
        splits = {"val": val}
        if not spec["args"]["validation_only"]:
            splits["test"] = {"unused": True}
        return {"data_id": "same-data", "run_identity": selection.fingerprint(spec),
                "protocol": "future-window", "validation_only": spec["args"]["validation_only"],
                "checkpoint_source": source, "fusion_mode": spec["args"]["fusion_mode"],
                "itemcf_half_life_days": spec["args"]["itemcf_half_life_days"], "splits": splits}

    def write_fixture(self):
        for run, result, spec, rows in ((self.baseline, self.base_result, self.base_spec, self.base_rows),
                                         (self.candidate, self.trial_result, self.trial_spec, self.trial_rows)):
            selection.write_json(run / "results.json", result)
            selection.write_json(run / "run_manifest.json", spec)
            selection.write_json(run / "val_predictions.json", rows)
            selection.write_json(run / "COMPLETED.json", {
                "status": "validation-only" if result["validation_only"] else "complete",
                "validation_only": result["validation_only"], "splits": list(result["splits"]), "results": result})

    def argv(self):
        arguments = []
        for name, value in vars(self.args).items():
            arguments.extend(["--" + name.replace("_", "-"), value])
        return arguments

    def test_complete_validation_pair_is_accepted(self):
        decision, recipe = selection.select(self.baseline, self.candidate)
        self.assertEqual(decision["status"], "accepted")
        self.assertEqual(decision["baseline_val"]["din"]["hr"], 0.)
        self.assertEqual(recipe["fusion_mode"], "rrf")

    def test_wrong_data_protocol_or_checkpoint_identity_is_rejected(self):
        original = copy.deepcopy(self.trial_result)
        changes = ({"data_id": "wrong"}, {"protocol": "leave-two-out"},
                   {"checkpoint_source": dict(original["checkpoint_source"], checkpoint_run_identities=["wrong", "wrong"])})
        for change in changes:
            with self.subTest(change=change):
                self.trial_result = dict(copy.deepcopy(original), **change)
                self.write_fixture()
                with self.assertRaises(ValueError):
                    selection.select(self.baseline, self.candidate)

    def test_candidate_scope_and_stale_marker_are_rejected(self):
        self.trial_result["splits"]["test"] = {"unexpected": True}
        self.write_fixture()
        with self.assertRaisesRegex(ValueError, "split mismatch"):
            selection.select(self.baseline, self.candidate)
        self.trial_result["splits"].pop("test")
        self.write_fixture()
        selection.write_json(self.candidate / "COMPLETED.json", {"status": "complete"})
        with self.assertRaisesRegex(ValueError, "scope mismatch"):
            selection.select(self.baseline, self.candidate)

    def test_missing_targets_duplicate_users_and_stale_metrics_fail(self):
        variants = [self.trial_rows * 2, [dict(self.trial_rows[0], targets=[])],
                    [dict(self.trial_rows[0], target_count=1)],
                    [dict(self.trial_rows[0], candidate_hit=0)]]
        for rows in variants:
            with self.subTest(rows=rows):
                selection.write_json(self.candidate / "val_predictions.json", rows)
                with self.assertRaises(ValueError):
                    selection.select(self.baseline, self.candidate)
        self.write_fixture()
        self.trial_result["splits"]["val"]["din"]["hr"] = .5
        self.write_fixture()
        with self.assertRaisesRegex(ValueError, "disagree"):
            selection.select(self.baseline, self.candidate)

    def test_failed_monitor_without_completion_marker_stops_wait(self):
        (self.baseline / "COMPLETED.json").unlink()
        selection.write_json(self.baseline / "monitor_status.json", {"status": "failed", "exit_code": 1})
        with mock.patch.object(selection.time, "sleep") as sleep, self.assertRaisesRegex(ValueError, "monitor failed"):
            selection.wait_for_results(self.baseline, self.candidate, 60, 3600)
        sleep.assert_not_called()

    def test_all_chain_failure_locations_stop_wait(self):
        for path in (self.candidate / "chain_status", self.candidate / ".chain_status",
                     self.candidate.with_name(self.candidate.name + ".chain_status")):
            with self.subTest(path=path):
                path.write_text("baseline failed", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "Chain stopped"):
                    selection.wait_for_results(self.baseline, self.candidate, 60, 3600)
                path.unlink()

    def test_wait_times_out(self):
        (self.candidate / "COMPLETED.json").unlink()
        with mock.patch.object(selection.time, "monotonic", side_effect=[0., 11.]), \
                mock.patch.object(selection.time, "sleep") as sleep, self.assertRaises(TimeoutError):
            selection.wait_for_results(self.baseline, self.candidate, 60, 10)
        sleep.assert_not_called()

    def test_timeout_and_unknown_monitor_write_failure_without_launch(self):
        (self.candidate / "COMPLETED.json").unlink()
        with mock.patch.object(selection.time, "monotonic", side_effect=[0., 11.]), \
                mock.patch.object(selection.subprocess, "Popen") as popen:
            self.assertEqual(selection.main(self.argv() + ["--timeout-seconds", "10"]), 1)
        popen.assert_not_called()
        self.assertIn("TimeoutError", selection.read_json(self.candidate / "AUTO_SELECT.json")["reason"])
        selection.write_json(self.candidate / "monitor_status.json", {"status": "unknown"})
        with mock.patch.object(selection.subprocess, "Popen") as popen:
            self.assertEqual(selection.main(self.argv()), 1)
        popen.assert_not_called()
        self.assertIn("unknown status", selection.read_json(self.candidate / "AUTO_SELECT.json")["reason"])

    def test_unknown_failure_is_atomic_and_never_launches(self):
        (self.candidate / "results.json").write_text("broken json", encoding="utf-8")
        with mock.patch.object(selection.subprocess, "Popen") as popen:
            self.assertEqual(selection.main(self.argv()), 1)
        popen.assert_not_called()
        decision = selection.read_json(self.candidate / "AUTO_SELECT.json")
        self.assertEqual(decision["status"], "failed")
        self.assertIn("JSONDecodeError", decision["reason"])
        self.assertFalse(list(self.candidate.glob("AUTO_SELECT.json.*.tmp")))

    def test_launch_is_monitored_matches_recipe_and_does_not_repeat(self):
        with mock.patch.object(selection.subprocess, "Popen", return_value=mock.Mock(pid=1234)) as popen:
            self.assertEqual(selection.main(self.argv()), 0)
            self.assertEqual(selection.main(self.argv()), 0)
        self.assertEqual(popen.call_count, 1)
        command = popen.call_args.args[0]
        self.assertIn(str(self.source / "tools/monitor_training.py"), command)
        self.assertNotIn("--validation-only", command)
        self.assertEqual(command[command.index("--final-users") + 1], "10")
        self.assertEqual(command[command.index("--dim") + 1], "8")
        self.assertEqual(command[command.index("--itemcf-half-life-days") + 1], "90.0")
        decision = selection.read_json(self.candidate / "AUTO_SELECT.json")
        self.assertEqual(decision["final_launch_status"], "already-started")
        self.assertEqual(decision["final_monitor_pid"], 1234)

    def test_rejected_metrics_never_create_final_directory(self):
        self.trial_rows = [row("a", ["x", "y"], ["q"])]
        self.trial_result = self.result(self.trial_spec, self.trial_rows, self.trial_result["checkpoint_source"])
        self.write_fixture()
        with mock.patch.object(selection.subprocess, "Popen") as popen:
            self.assertEqual(selection.main(self.argv()), 0)
        popen.assert_not_called()
        self.assertFalse(self.final.exists())
        self.assertEqual(selection.read_json(self.candidate / "AUTO_SELECT.json")["status"], "rejected")

    def test_failed_final_launch_cannot_be_retried_automatically(self):
        with mock.patch.object(selection.subprocess, "Popen", side_effect=OSError("launch unavailable")) as popen:
            self.assertEqual(selection.main(self.argv()), 1)
            self.assertEqual(selection.main(self.argv()), 1)
        self.assertEqual(popen.call_count, 1)
        self.assertEqual(selection.read_json(self.final / "AUTO_LAUNCH.json")["status"], "failed")

    def test_unknown_final_directory_is_not_reused(self):
        self.final.mkdir()
        (self.final / "train.log").write_text("existing run", encoding="utf-8")
        with mock.patch.object(selection.subprocess, "Popen") as popen:
            self.assertEqual(selection.main(self.argv()), 1)
        popen.assert_not_called()
        self.assertIn("nonempty final directory", selection.read_json(self.candidate / "AUTO_SELECT.json")["reason"])


if __name__ == "__main__":
    unittest.main()
