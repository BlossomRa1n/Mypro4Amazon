import sys
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from future_window_attribution import _compact_itemcf_topk


class CompactItemCFTests(unittest.TestCase):
    def test_compact_aggregation_matches_deterministic_catalog_ranking(self):
        # Duplicate neighbors exercise the segment sum; invalid and zero-score
        # entries exercise the same filtering used by the attribution loop.
        candidate_ids = np.asarray([
            [[2, 7, 2, 11], [7, 4, 11, 2]],
            [[5, 5, 9, 3], [9, 8, 5, 3]],
        ], dtype=np.int64)
        candidate_scores = np.asarray([
            [[.5, .4, .25, .1], [.2, .3, .7, 0.]],
            [[.1, .8, .2, .3], [.4, .2, .6, .5]],
        ], dtype=np.float32)
        valid = np.asarray([
            [[True, True, True, True], [True, True, True, False]],
            [[True, True, True, True], [True, True, True, True]],
        ])

        ids, values = _compact_itemcf_topk(
            candidate_ids, candidate_scores, valid, budget=3, item_count=20,
            device=torch.device("cpu"))

        self.assertEqual(ids.tolist(), [[11, 2, 7], [5, 3, 9]])
        np.testing.assert_allclose(values, [[.8, .75, .6], [1.5, .8, .6]],
                                   rtol=0., atol=1e-6)

    def test_empty_valid_batch_returns_empty_rows(self):
        ids, values = _compact_itemcf_topk(
            np.zeros((2, 1, 3), dtype=np.int64),
            np.zeros((2, 1, 3), dtype=np.float32),
            np.zeros((2, 1, 3), dtype=bool), budget=5, item_count=20,
            device=torch.device("cpu"))
        self.assertEqual(ids.shape, (2, 0))
        self.assertEqual(values.shape, (2, 0))


if __name__ == "__main__":
    unittest.main()
