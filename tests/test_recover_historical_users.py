import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("recover_historical_users", ROOT / "tools/recover_historical_users.py")
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path

    def identity(self, name, users):
        return self.write(name + ".json", {"data_id": name, "users": ["__PAD__", "__UNKNOWN__"] + users,
                                          "source_sha256": "a" * 64})

    def test_merge_raw_ids_not_uid_and_partial_attestation(self):
        first = self.identity("first", ["alice", "bob"])
        second = self.identity("second", ["charlie", "alice"])
        assets = []
        for name in ("first", "second"):
            self.write(name + "/suite_manifest.json", {"source": {"data_id": name}})
            path = self.root / name / "eval_users.npz"
            np.savez(path, uid=np.array([2]), labels=np.array([object()], dtype=object))
            assets.append({"path": str(path)})
        predictions = self.write("historical_predictions.json", [{"user_id": "diana", "labels": [1, 2]},
                                                                   {"user_id": "alice", "metrics": {"a": 1}}])
        assets.append({"path": str(predictions)})
        csv_path = self.root / "test_recommendations.csv"
        csv_path.write_text("user_id,item\nerin,item1\n")
        assets.append({"path": str(csv_path)})
        # This is not an output CSV and must not be treated as historical evaluation.
        raw = self.root / "source.csv"
        raw.write_text("user_id,item\nNOT_EVALUATED,item1\n")
        assets.append({"path": str(raw)})
        inventory = self.write("inventory.json", {"assets": assets})
        output = self.root / "new-output"
        summary = recovery.recover(inventory, [first, second], [first], [], output)
        excluded = json.loads((output / "historical_users_partial.json").read_text())
        self.assertEqual(set(excluded["raw_user_ids"]), {"alice", "bob", "charlie", "diana", "erin"})
        self.assertEqual(excluded["schema_version"], 1)
        self.assertFalse(excluded["completeness"]["attested"])
        self.assertEqual(summary["unique_raw_user_count"], 5)
        with self.assertRaises(recovery.RecoveryError):
            recovery.recover(inventory, [first], [], [], output)

    def test_inventory_recovers_preds_csv_but_excludes_raw_source_directories(self):
        assets = []
        for name, user in (("preds_legacy.csv", "legacy-user"),
                           ("PREDS_SECOND.CSV", "second-user"),
                           ("other_preds_unrelated.csv", "not-a-prefix")):
            path = self.root / name
            path.write_text(f"user_id,item\n{user},item1\n")
            assets.append(str(path))
        for directory in ("raw", "raw_data", "source_data", "amazon_data",
                          "amazon_reviews", "5core", "rating_only"):
            path = self.root / directory / "preds_source.csv"
            path.parent.mkdir()
            path.write_text("user_id,item\nNOT_EVALUATED,item1\n")
            assets.append(str(path))
        inventory = self.write("inventory.json", {"assets": assets})
        output = self.root / "new-output"
        summary = recovery.recover(inventory, [], [], [], output)
        excluded = json.loads((output / "historical_users_partial.json").read_text())
        self.assertEqual(excluded["raw_user_ids"], ["legacy-user", "second-user"])
        self.assertEqual(summary["unique_raw_user_count"], 2)
        self.assertEqual({source["path"] for source in summary["sources"]}, set(assets[:2]))
        self.assertTrue(all(source["mode"] == "user_csv" and source["status"] == "recovered"
                            for source in summary["sources"]))
        self.assertFalse(summary["completeness"]["attested"])
        self.assertFalse(excluded["completeness"]["attested"])

    def test_mismatched_manifest_and_out_of_range_rejected(self):
        identity = recovery.load_identity(self.identity("first", ["alice"]))
        manifest = self.write("suite_manifest.json", {"data_id": "wrong"})
        path = self.root / "eval_users.npz"
        np.savez(path, uid=np.array([2]))
        with self.assertRaises(recovery.RecoveryError):
            recovery.npz_users(path, {"first": identity})
        manifest.write_text(json.dumps({"data_id": "first"}))
        for indices in ([3], [-1], [0]):
            np.savez(path, uid=np.array(indices))
            with self.subTest(indices=indices), self.assertRaises(recovery.RecoveryError):
                recovery.npz_users(path, {"first": identity})

    def test_streaming_predictions_cross_chunk_and_invalid_ids(self):
        path = self.write("predictions.json", [{"ignored": "x" * 130000, "user_id": "用户",
                                                "metrics": {"user_id": "not-a-user"},
                                                "labels": [{"user_id": "also-not-a-user"}]},
                                               {"user_id": "other"}])
        self.assertEqual(recovery.prediction_users(path), {"用户", "other"})
        path.write_text('[{"user_id": 2}]')
        with self.assertRaises(recovery.RecoveryError):
            recovery.prediction_users(path)
        path.write_text('[{"user_id": "u"}')
        with self.assertRaises(recovery.RecoveryError):
            recovery.prediction_users(path)


if __name__ == "__main__":
    unittest.main()
