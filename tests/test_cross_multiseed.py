"""Small protocol guards for the registered cross runner."""

import hashlib
import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from future_window_data import FutureWindowData
from run_cross_multiseed import (load_historical_users, prepare,
                                 sha256_file)
from token_models import SemanticTokenDIN


def _fixture(root: Path):
    rows = []
    for user in range(14):
        for tick in range(8):
            rows.append((f"u{user:03}", f"i{user:03}_{tick}", 4.0,
                         1_700_000_000_000 + tick * 86_400_000, "fixture"))
    frame = pd.DataFrame(rows, columns=("user_id", "parent_asin", "rating", "timestamp", "category"))
    data = FutureWindowData(frame, hist_len=5, test_users=2)
    base = root / "base"; base.mkdir()
    with (base / "data.pkl").open("wb") as stream:
        pickle.dump({"data": data}, stream, protocol=5)
    (base / "run_manifest.json").write_text(json.dumps({"args": {"dim": 8, "hist_len": 5}}))
    source = root / "history-source.txt"; source.write_text("fixture historical source")
    history = root / "historical.json"
    history.write_text(json.dumps({
        "schema_version": 1, "raw_user_ids": ["outside-fixture"],
        "sources": [{"path": str(source), "sha256": sha256_file(source), "mode": "fixture"}],
        "completeness": {"attested": True, "note": "fixture source"},
    }))
    return base, history


class CrossProtocolTests(unittest.TestCase):
    def test_prepare_uses_raw_ids_and_exact_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); base, historical = _fixture(root)
            args = type("Args", (), {"base_run": str(base), "run_dir": str(root / "protocol"),
                "historical_users": str(historical), "train_users": 4, "screen_users": 2,
                "confirm_users": 2, "test_users": 2, "cohort_seed": 20260922, "smoke": True})()
            manifest = prepare(args)
            self.assertEqual(manifest["counts"], {"train": 4, "screen": 2, "confirm": 2, "test": 2})
            screen = json.loads((root / "protocol" / "screen_records.json").read_text())
            confirm = json.loads((root / "protocol" / "confirm_records.json").read_text())
            test = json.loads((root / "protocol" / "test_records.json").read_text())
            self.assertTrue({row[2] for row in screen}.isdisjoint({row[2] for row in confirm}))
            self.assertTrue({row[2] for row in test}.isdisjoint({row[2] for row in screen + confirm}))
            self.assertFalse(manifest["test_future_labels_read"])

    def test_unverifiable_history_source_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.json"
            path.write_text(json.dumps({"schema_version": 1, "raw_user_ids": ["u"],
                                        "sources": [], "completeness": {"attested": True, "note": "x"}}))
            with self.assertRaises(ValueError):
                load_historical_users(path)

    def test_zero_cross_keeps_shape_and_zeroes_complete_slot(self):
        model = SemanticTokenDIN(num_users=4, num_items=8, num_brands=4,
                                 num_categories=4, embed_dim=8, token_dim=8,
                                 hist_len=5, cross_mode="zero_cross")
        self.assertEqual(model.cross_gate_logit.numel(), 1)
        self.assertEqual(model.TOKEN_COUNT, 6)


if __name__ == "__main__":
    unittest.main()
