import copy
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from baseline_data import BenchmarkData, PrefixDataset, ITEM_KEYS, USER_KEYS, collate
from future_window_data import FutureWindowData
from baseline_runtime import pair_auc, quota_merge, ranking_metrics, save_checkpoint, restore_checkpoint
from model_ext import DINExtendedModel
from run_baseline import build_matrix, candidate_pools, score_din, targets_for_protocol


def example():
    rows = []
    for u in range(12):
        for t in range(6):
            rows.append((f"u{u:02}", f"i{u * 6 + t:03}", 4. + t % 2, 1000000000000 + t * 86400000, "category"))
    return pd.DataFrame(rows, columns=["user_id", "parent_asin", "rating", "timestamp", "category"])


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.data = BenchmarkData(example(), hist_len=5)

    def test_prefix_and_holdouts(self):
        data = self.data
        dataset = PrefixDataset(data, negatives=4)
        matrix = build_matrix(data)
        for idx in range(len(dataset)):
            sample = dataset[idx]
            uid, target = int(sample["user_id"]), int(sample["pos_item_id"])
            self.assertNotIn(target, sample["hist_items"])
            pos = data.train_positions[idx]
            end = data.history_end(uid, pos)
            self.assertTrue(np.all(data.ts[data.starts[uid]:end] < data.ts[pos]))
            self.assertTrue(set(sample["neg_item_id"]).isdisjoint(data.train_sets[uid]))
            self.assertNotIn(target, sample["neg_item_id"])
        for uid, pos in data.evaluation("val") + data.evaluation("test"):
            self.assertEqual(matrix[uid, data.iid[pos]], 0)
        self.assertEqual(data.manifest["train_interactions"], 48)

    def test_reserved_and_deterministic(self):
        data = self.data
        self.assertEqual(data.items[:2], ["__PAD__", "__UNKNOWN__"])
        self.assertTrue(np.all(data.item_category[2:] >= 2))
        self.assertTrue(np.all(data.iid >= 2))
        self.assertEqual(data.evaluation(max_users=5), data.evaluation(max_users=5))
        ds = PrefixDataset(data)
        np.testing.assert_array_equal(ds[3]["neg_item_id"], ds[3]["neg_item_id"])

    def test_ties_never_cross_history(self):
        frame = example()
        frame.loc[1, "timestamp"] = frame.loc[0, "timestamp"]
        data = BenchmarkData(frame, hist_len=5)
        for uid, pos in data.evaluation("val"):
            end = data.history_end(uid, pos)
            self.assertLess(data.ts[end - 1], data.ts[pos])

    def test_future_window_uses_80_percent_prefix(self):
        data = FutureWindowData(example(), hist_len=5, test_users=1)
        uid, cutoff = data.evaluation("test", 1)[0]
        start, end = data.starts[uid], data.ends[uid]
        self.assertEqual(cutoff, start + 4)
        self.assertEqual(set(data.iid[cutoff:end]), set(data.targets([(uid, cutoff)])[0]))
        self.assertTrue(np.all(data.ts[start:cutoff] < data.ts[cutoff]))

    def test_future_targets_flow_through_runner_helper(self):
        data = FutureWindowData(example(), hist_len=5, test_users=1)
        records = data.evaluation("test", 1)
        targets = targets_for_protocol(data, records, "future-window")
        self.assertIsInstance(targets[0], set)
        self.assertEqual(targets, data.targets(records))
        self.assertEqual(targets_for_protocol(data, records, "leave-two-out"),
                         [int(data.iid[records[0][1]])])
        self.assertTrue(set.union(*targets).isdisjoint(set(data.iid[data.train_mask])))

    def test_metrics(self):
        self.assertEqual(pair_auc([.9, .2], [[.8], [.1]]), 1.)
        self.assertEqual(pair_auc([.5], [[.5]]), .5)
        metrics = ranking_metrics([[1, 2], [3]], [{1, 2}, {4}], 2)
        self.assertEqual(metrics["ndcg"], .5)
        self.assertEqual(metrics["hr"], .5)

    def test_fusion_refills_and_filters(self):
        channel = {i: 100. - i for i in range(100)}
        result = quota_merge([channel, channel], [1., 1.], 75, excluded={0, 1})
        self.assertEqual(len(result), 75)
        self.assertTrue(set(result).isdisjoint({0, 1}))
        self.assertEqual(quota_merge([channel], [0.], 75), {})

    def test_rrf_candidate_fusion_and_half_life(self):
        data = BenchmarkData(example(), hist_len=5)
        records = data.evaluation("test", 3)
        neighbors = np.tile(np.arange(2, len(data.items)), (len(data.items), 1))
        similarities = np.ones_like(neighbors, dtype=np.float32)
        scores = [{i: float(len(data.items) - i) for i in range(2, len(data.items))} for _ in records]
        pools, _ = candidate_pools(data, records, scores, (neighbors, similarities), 20,
                                   fusion_mode="rrf", half_life_days=90.)
        for (uid, pos), pool in zip(records, pools):
            seen = set(data.iid[data.starts[uid]:data.history_end(uid, pos)])
            self.assertEqual(len(pool), 20)
            self.assertTrue(seen.isdisjoint(pool))

    def test_candidates_exclude_history_beyond_model_window(self):
        data = BenchmarkData(example(), hist_len=2)
        records = data.evaluation("test", 3)
        neighbors = np.tile(np.arange(2, len(data.items)), (len(data.items), 1))
        similarities = np.ones_like(neighbors, dtype=np.float32)
        scores = [{i: float(len(data.items) - i) for i in range(2, len(data.items))} for _ in records]
        pools, cf = candidate_pools(data, records, scores, (neighbors, similarities), 20)
        for (uid, pos), pool, ranked in zip(records, pools, cf):
            seen = set(data.iid[data.starts[uid]:data.history_end(uid, pos)])
            self.assertGreater(len(seen), data.hist_len)
            self.assertTrue(seen.isdisjoint(pool))
            self.assertTrue(seen.isdisjoint(ranked))
            self.assertEqual(len(pool), 20)

    def test_din_evaluator_matches_single_candidate_scoring(self):
        data, model = self.data, self.model().eval()
        records = data.evaluation("val", 3)
        pools = [{9: 1., 11: 2., 20: 0.}, {}, {5: 1., 21: 2.}]
        ranked = score_din(model, data, records, pools, torch.device("cpu"), microbatch=2)
        expected = []
        with torch.no_grad():
            for (uid, pos), pool in zip(records, pools):
                scores = {}
                for iid in pool:
                    batch = collate([{**data.user_features(uid, pos), **data.item_features(iid)}])
                    scores[iid] = float(model(batch).item())
                expected.append(sorted(scores, key=lambda iid: (-scores[iid], iid)))
        self.assertEqual(ranked, expected)

    def model(self):
        data = self.data
        return DINExtendedModel(len(data.users), len(data.items), len(data.brands), len(data.categories),
                                embed_dim=8, brand_embed_dim=4, hidden_dims=[16], hist_len=5, dropout=0)

    def test_bundled_eval_equals_individual(self):
        model = self.model().eval()
        batch = collate([PrefixDataset(self.data)[i] for i in range(3)])
        users = {k: batch[k] for k in USER_KEYS}
        pos = {k: batch["pos_" + k] for k in ITEM_KEYS}
        neg = {k: batch["neg_" + k] for k in ITEM_KEYS}
        with torch.no_grad():
            p, n = model.forward_bpr(users, pos, neg)
            torch.testing.assert_close(p, model({**users, **pos}), atol=1e-6, rtol=1e-5)
            for i in range(4):
                torch.testing.assert_close(n[:, i], model({**users, **{k: v[:, i] for k, v in neg.items()}}), atol=1e-6, rtol=1e-5)

    def test_resume_matches_continuous_update(self):
        torch.manual_seed(42)
        model = torch.nn.Linear(3, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, .9)
        inputs = torch.randn(4, 3)
        def step(m, opt, sch):
            opt.zero_grad()
            m(inputs).square().mean().backward()
            opt.step()
            sch.step()
        step(model, optimizer, scheduler)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latest.pth"
            save_checkpoint(path, model, optimizer, scheduler, 1, .5, {"id": "test"}, [])
            restored = torch.nn.Linear(3, 1)
            opt = torch.optim.AdamW(restored.parameters(), lr=.01)
            sch = torch.optim.lr_scheduler.StepLR(opt, 1, .9)
            restore_checkpoint(path, restored, opt, sch, {"id": "test"})
            before = restored.weight.detach().clone()
            step(restored, opt, sch)
            step(model, optimizer, scheduler)
            self.assertFalse(torch.equal(before, restored.weight))
            for p, q in zip(model.parameters(), restored.parameters()):
                torch.testing.assert_close(p, q, atol=0, rtol=0)
            with self.assertRaises(ValueError):
                restore_checkpoint(path, restored, opt, sch, {"id": "wrong"})


if __name__ == "__main__":
    unittest.main()
