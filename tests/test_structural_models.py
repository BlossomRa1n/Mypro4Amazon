import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from baseline_data import BenchmarkData, PrefixDataset, collate, USER_KEYS, ITEM_KEYS
from model_ext import DINExtendedModel
from structural_models import StructuralDIN, VARIANTS, variant_for_experiment
from test_baseline import example


class StructuralTests(unittest.TestCase):
    def setUp(self):
        self.data = BenchmarkData(example(), hist_len=5)
        self.config = dict(num_users=len(self.data.users), num_items=len(self.data.items),
                           num_categories=len(self.data.categories), num_brands=len(self.data.brands),
                           embed_dim=64, brand_embed_dim=8, hidden_dims=[16], dropout=.1, hist_len=5)
        self.baseline = DINExtendedModel(**self.config).eval()
        batch = collate([PrefixDataset(self.data)[i] for i in range(4)])
        self.user = {k: batch[k] for k in USER_KEYS}
        self.pos = {k: batch["pos_" + k] for k in ITEM_KEYS}
        self.neg = {k: batch["neg_" + k] for k in ITEM_KEYS}

    def test_category_conversion_preserves_initial_logits_and_reduces_width(self):
        model = StructuralDIN("category32", **self.config).eval()
        model.load_baseline(self.baseline.state_dict())
        self.assertEqual(model.category_embedding.embedding_dim, 32)
        self.assertEqual(model.mlp[0].in_features, self.baseline.mlp[0].in_features - 32)
        with torch.no_grad():
            torch.testing.assert_close(model({**self.user, **self.pos}),
                                       self.baseline({**self.user, **self.pos}), atol=1e-6, rtol=1e-5)

    def test_variants_train_and_roundtrip_without_user_table(self):
        for variant in VARIANTS:
            model = StructuralDIN(variant, **self.config)
            model.load_baseline(self.baseline.state_dict())
            model.train()
            p, n = model.forward_bpr(self.user, self.pos, self.neg)
            F.softplus(n - p[:, None]).mean().backward()
            self.assertTrue(all(torch.isfinite(param.grad).all() for param in model.parameters() if param.grad is not None))
            other = StructuralDIN(variant, **self.config)
            other.load_state_dict(model.state_dict())
            self.assertEqual(variant_for_experiment("structural_" + variant + "_seed43"), variant)
            model.eval()
            other.eval()
            with torch.no_grad():
                torch.testing.assert_close(model({**self.user, **self.pos}), other({**self.user, **self.pos}), atol=0, rtol=0)
            if variant == "no_user_id":
                self.assertNotIn("user_embedding.weight", model.state_dict())
                encoded = model._encode_user(self.user)
                self.assertEqual(torch.count_nonzero(encoded["user_emb"]).item(), 0)
            elif variant == "no_popularity":
                dense = model.raw_slices(model._encode_user(self.user), self.pos)[6]
                self.assertEqual(torch.count_nonzero(dense[:, [6, 9]]).item(), 0)


if __name__ == "__main__":
    unittest.main()
