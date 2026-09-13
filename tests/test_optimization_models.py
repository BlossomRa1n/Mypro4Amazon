import copy
import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from baseline_data import BenchmarkData, PrefixDataset, USER_KEYS, ITEM_KEYS, collate
from baseline_runtime import seed_all
from model_ext import DINExtendedModel
from optimization_models import RawSliceDIN, score_din_cached
from run_baseline import score_din
from test_baseline import example


class RawSliceTests(unittest.TestCase):
    def setUp(self):
        seed_all(42)
        self.data = BenchmarkData(example(), hist_len=5)
        kwargs = dict(num_users=len(self.data.users), num_items=len(self.data.items),
                      num_brands=len(self.data.brands), num_categories=len(self.data.categories),
                      embed_dim=8, brand_embed_dim=4, hidden_dims=[16, 8], hist_len=5, dropout=.1)
        self.original = DINExtendedModel(**kwargs)
        self.sliced = RawSliceDIN(**kwargs)
        self.sliced.load_state_dict(copy.deepcopy(self.original.state_dict()))
        batch = collate([PrefixDataset(self.data)[i] for i in range(4)])
        self.user = {k: batch[k] for k in USER_KEYS}
        self.pos = {k: batch["pos_" + k] for k in ITEM_KEYS}
        self.neg = {k: batch["neg_" + k] for k in ITEM_KEYS}

    def test_raw_slice_logits_gradients_and_three_adam_updates(self):
        for model in (self.original, self.sliced):
            model.eval()
        torch.testing.assert_close(self.original({**self.user, **self.pos}),
                                   self.sliced({**self.user, **self.pos}), atol=0, rtol=0)
        optimizers = [torch.optim.AdamW(model.parameters(), lr=.001)
                      for model in (self.original, self.sliced)]
        for step in range(3):
            losses, gradients = [], []
            for model, optimizer in zip((self.original, self.sliced), optimizers):
                model.train()
                seed_all(100 + step)
                optimizer.zero_grad()
                pos, neg = model.forward_bpr(self.user, self.pos, self.neg)
                loss = F.softplus(neg - pos[:, None]).mean()
                loss.backward()
                losses.append(loss.detach())
                gradients.append({name: param.grad.clone() for name, param in model.named_parameters()
                                  if param.grad is not None})
                optimizer.step()
            torch.testing.assert_close(losses[0], losses[1], atol=0, rtol=0)
            self.assertEqual(gradients[0].keys(), gradients[1].keys())
            for name in gradients[0]:
                torch.testing.assert_close(gradients[0][name], gradients[1][name], atol=0, rtol=0)
            for name, tensor in self.original.state_dict().items():
                torch.testing.assert_close(tensor, self.sliced.state_dict()[name], atol=0, rtol=0)

    def test_cached_inference_variable_pools_padding_and_ties(self):
        records = self.data.evaluation("val", 4)
        pools = [{9: 1., 11: 2., 20: 0.}, {}, {5: 1., 21: 2.}, {8: 0.}]
        for model in (self.original, self.sliced):
            expected = score_din(model, self.data, records, pools, torch.device("cpu"), microbatch=2)
            actual = score_din_cached(model, self.data, records, pools, torch.device("cpu"), microbatch=2)
            self.assertEqual(actual, expected)
        with torch.no_grad():
            self.sliced.mlp[-1].weight.zero_()
            self.sliced.mlp[-1].bias.zero_()
        self.assertEqual(score_din_cached(self.sliced, self.data, records, pools, torch.device("cpu")),
                         [sorted(pool) for pool in pools])


if __name__ == "__main__":
    unittest.main()
