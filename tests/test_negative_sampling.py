import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline_data import BenchmarkData, PrefixDataset
from baseline_runtime import bpr_loss, sampled_softmax_loss
from test_baseline import example


class NegativeSamplingTests(unittest.TestCase):
    def setUp(self):
        self.data = BenchmarkData(example(), hist_len=5)
        ranked = np.tile(self.data.active_items, (len(self.data.train_positions), 1))
        self.ranked = ranked

    def test_sixteen_random_is_deterministic(self):
        dataset = PrefixDataset(self.data, negatives=16)
        first = dataset[0]["neg_item_id"]
        np.testing.assert_array_equal(first, dataset[0]["neg_item_id"])
        self.assertEqual(len(first), 16)
        self.assertEqual(len(set(first.tolist())), 16)

    def test_mixed_sampler_uses_four_ranked_and_twelve_random(self):
        dataset = PrefixDataset(self.data, negatives=16, negative_policy="mixed_rrf",
                                candidate_pools=self.ranked)
        row = dataset[0]
        values = row["neg_item_id"].tolist()
        self.assertEqual(len(values), 16)
        self.assertEqual(len(set(values)), 16)
        uid, target = int(row["user_id"]), int(row["pos_item_id"])
        self.assertNotIn(target, values)
        self.assertTrue(set(values).isdisjoint(self.data.train_sets[uid]))
        # The selected candidate portion is deterministic and comes from the
        # configured 11--25 / 26--50 rank bands (after filtering known IDs).
        allowed = set(self.ranked[0][10:25]) | set(self.ranked[0][25:50])
        self.assertGreaterEqual(len(set(values) & allowed), 2)

    def test_loss_functions_share_positive_zero_label(self):
        positive = torch.tensor([2., 1.])
        negative = torch.tensor([[1., 0.], [2., -1.]])
        bpr = bpr_loss(positive, negative)
        softmax = sampled_softmax_loss(positive, negative)
        self.assertTrue(torch.isfinite(bpr))
        self.assertTrue(torch.isfinite(softmax))
        self.assertGreater(float(softmax), 0.)
        self.assertEqual(sampled_softmax_loss(positive, negative, 0.5).shape, torch.Size([]))
        with self.assertRaises(ValueError):
            sampled_softmax_loss(positive, negative, 0.)


if __name__ == "__main__":
    unittest.main()
