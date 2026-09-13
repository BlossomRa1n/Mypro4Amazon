import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from baseline_data import BenchmarkData
from prepare_candidate_training import training_internal_data, fit_cf, mine
from run_baseline import build_matrix
from run_candidate_experiments import CandidateDataset
from test_baseline import example


class CandidatePreparationTests(unittest.TestCase):
    def test_internal_split_never_fits_outer_or_internal_heldout_labels(self):
        base = BenchmarkData(example(), hist_len=5)
        inner, positions, item_map = training_internal_data(base)
        self.assertTrue(np.all(base.train_mask[positions]))
        matrix = build_matrix(inner)
        for uid, pos in inner.evaluation("val") + inner.evaluation("test"):
            self.assertEqual(matrix[uid, inner.iid[pos]], 0)
        for _, pos in base.evaluation("val") + base.evaluation("test"):
            self.assertNotIn(pos, positions)
        cf = fit_cf(inner, count=10)
        records, negatives, stats = mine(base, inner, positions, item_map, cf, inner.evaluation("val"), budget=20)
        self.assertGreater(stats["mean_eligible_negatives"], 0)
        for (uid, pos), row in zip(records, negatives):
            self.assertTrue(base.train_mask[pos])
            valid = set(row[row >= 2].tolist())
            self.assertTrue(valid <= base.active_set)
            self.assertTrue(valid.isdisjoint(base.train_sets[uid]))
            self.assertNotIn(base.iid[pos], base.user_features(uid, pos)["hist_items"])

    def test_candidate_sampler_rejects_known_positives_and_falls_back(self):
        base = BenchmarkData(example(), hist_len=5)
        positions = base.train_positions[:4]
        records = np.column_stack([base.uid[positions], positions])
        pools = np.tile(base.active_items, (len(records), 1))
        for count in (0, 1):
            dataset = CandidateDataset(base, records, pools, recall_count=count)
            for index in range(len(dataset)):
                row = dataset[index]
                values = set(row["neg_item_id"].tolist())
                self.assertEqual(len(values), 4)
                self.assertTrue(values.isdisjoint(base.train_sets[int(row["user_id"])]))
                self.assertEqual(row["hard_count"], count)
                np.testing.assert_array_equal(row["neg_item_id"], dataset[index]["neg_item_id"])
        fallback = CandidateDataset(base, records, np.zeros_like(pools), recall_count=1)
        self.assertEqual(fallback[0]["hard_count"], 0)


if __name__ == "__main__":
    unittest.main()
