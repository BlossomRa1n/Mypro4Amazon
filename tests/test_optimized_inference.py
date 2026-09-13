import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from baseline_data import BenchmarkData
from optimized_inference import LatestHistoryView
from run_fusion_experiments import Channels, RECIPES
from test_baseline import example


class LatestInferenceTests(unittest.TestCase):
    def test_latest_includes_last_event_and_filters_complete_observed_history(self):
        base = BenchmarkData(example(), hist_len=2)
        view = LatestHistoryView(base)
        times = base.ts.copy()
        graph = np.tile(np.arange(2, len(base.items)), (len(base.items), 1))
        channels = Channels(view, (graph, np.ones_like(graph, dtype=np.float32)))
        for uid in range(2, len(base.users)):
            record = (uid, int(base.ends[uid]))
            features = view.user_features(*record)
            self.assertEqual(features["hist_items"][-1], base.iid[base.ends[uid] - 1])
            parts, seen = channels.build(record, {}, 20, RECIPES["rrf"])
            pool = channels.merge(parts, seen, 20, RECIPES["rrf"])
            self.assertEqual(len(seen), 6)
            self.assertTrue(set(pool).isdisjoint(base.iid[base.starts[uid]:base.ends[uid]]))
        np.testing.assert_array_equal(times, base.ts)


if __name__ == "__main__":
    unittest.main()
