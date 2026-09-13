import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from baseline_data import BenchmarkData
from run_baseline import candidate_pools
from run_fusion_experiments import Channels, RECIPES
from test_baseline import example


class FusionTests(unittest.TestCase):
    def test_quota_matches_frozen_baseline_and_variants_exclude_complete_history(self):
        data = BenchmarkData(example(), hist_len=2)
        graph = np.tile(np.arange(2, len(data.items)), (len(data.items), 1))
        cf = (graph, np.ones_like(graph, dtype=np.float32))
        records = data.evaluation("test", 4)
        v2 = [{i: float(len(data.items) - i) for i in range(2, len(data.items))} for _ in records]
        expected, _ = candidate_pools(data, records, v2, cf, 20)
        builder = Channels(data, cf)
        for record, scores, pool in zip(records, v2, expected):
            for name, recipe in RECIPES.items():
                channels, seen = builder.build(record, scores, 20, recipe)
                actual = builder.merge(channels, seen, 20, recipe)
                self.assertEqual(len(actual), 20)
                self.assertTrue(seen.isdisjoint(actual))
                if name == "quota":
                    self.assertEqual(list(actual.items()), list(pool.items()))

    def test_recency_uses_last_history_event_and_brand_excludes_unknown(self):
        data = BenchmarkData(example(), hist_len=5)
        neighbors = np.zeros((len(data.items), 2), dtype=np.int32)
        scores = np.zeros_like(neighbors, dtype=np.float32)
        record = data.evaluation("val", 1)[0]
        uid, pos = record
        hist = data.iid[data.starts[uid]:data.history_end(uid, pos)]
        candidates = [int(i) for i in data.active_items if i not in hist][:2]
        neighbors[hist[0], 0], neighbors[hist[-1], 0] = candidates
        scores[hist[0], 0] = scores[hist[-1], 0] = 1.
        builder = Channels(data, (neighbors, scores))
        channels, _ = builder.build(record, {}, 20, RECIPES["cf30"])
        self.assertGreater(channels[0][candidates[1]], channels[0][candidates[0]])
        original = channels[0].copy()
        data.ts = data.ts.copy()
        data.ts[pos] += 86400000 * 10
        channels, _ = builder.build(record, {}, 20, RECIPES["cf30"])
        self.assertEqual(channels[0], original)
        channels, _ = builder.build(record, {}, 20, RECIPES["brand"])
        self.assertEqual(channels[-1], {})


if __name__ == "__main__":
    unittest.main()
