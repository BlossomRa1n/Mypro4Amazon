import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from baseline_data import BenchmarkData
from run_optimization import MixedNegativeDataset, paired_comparison
from test_baseline import example


class OptimizationTests(unittest.TestCase):
    def test_hard_samples_use_only_train_catalog_and_exclude_known_positives(self):
        data = BenchmarkData(example(), hist_len=3)
        neighbors = np.tile(np.arange(len(data.items)), (len(data.items), 1))
        scores = np.ones_like(neighbors, dtype=np.float32)
        dataset = MixedNegativeDataset(data, (neighbors, scores))
        for idx in range(len(dataset)):
            row = dataset[idx]
            negatives = set(row["neg_item_id"].tolist())
            uid, target = int(row["user_id"]), int(row["pos_item_id"])
            self.assertEqual(len(negatives), 4)
            self.assertTrue(negatives <= data.active_set)
            self.assertTrue(negatives.isdisjoint(data.train_sets[uid]))
            self.assertNotIn(target, negatives)
            self.assertNotIn(target, row["hist_items"])
            self.assertEqual(row["hard_count"], 2)
            np.testing.assert_array_equal(row["neg_item_id"], dataset[idx]["neg_item_id"])

    def test_empty_hard_pool_falls_back_to_four_distinct_random_samples(self):
        data = BenchmarkData(example(), hist_len=3)
        graph = np.zeros((len(data.items), 10), dtype=np.int32)
        dataset = MixedNegativeDataset(data, (graph, graph.astype(np.float32)))
        row = dataset[0]
        self.assertEqual(row["hard_count"], 0)
        self.assertEqual(len(set(row["neg_item_id"].tolist())), 4)

    def test_paired_interval_and_alignment_guard(self):
        reference = {"uid": np.arange(1000), "target": np.arange(1000),
                     "hit5": np.zeros(1000, dtype=bool), "ndcg5": np.zeros(1000)}
        candidate = {k: v.copy() for k, v in reference.items()}
        candidate["hit5"][:100] = True
        candidate["ndcg5"][:100] = 1.
        result = paired_comparison(candidate, reference)
        self.assertTrue(result["confirmed_gain"])
        self.assertAlmostEqual(result["hr5_delta"], .1)
        self.assertEqual(paired_comparison(reference, reference)["hr5_delta_ci95"], [0., 0.])
        candidate["uid"] = candidate["uid"][::-1]
        with self.assertRaises(ValueError):
            paired_comparison(candidate, reference)


if __name__ == "__main__":
    unittest.main()
