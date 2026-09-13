"""Isolated DIN ablations with explicit checkpoint conversion."""
import torch
from torch import nn

from optimization_models import RawSliceDIN


VARIANTS = ("control", "no_user_id", "category32", "no_popularity", "dropout20")


class ZeroEmbedding(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.width = width
        self.register_buffer("anchor", torch.zeros(1), persistent=False)

    def forward(self, ids):
        return self.anchor.new_zeros((*ids.shape, self.width))


class StructuralDIN(RawSliceDIN):
    def __init__(self, variant="control", **kwargs):
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.variant = variant
        kwargs = dict(kwargs)
        if variant == "no_user_id":
            kwargs["num_users"] = 2
        if variant == "dropout20":
            kwargs["dropout"] = .2
        super().__init__(**kwargs)
        if variant == "no_user_id":
            self.user_embedding = ZeroEmbedding(self.embed_dim)
        if variant == "category32":
            width = min(32, self.embed_dim)
            self.category_embedding = nn.Embedding(kwargs["num_categories"], width, padding_idx=0)
            first = self.mlp[0]
            self.mlp[0] = nn.Linear(first.in_features - self.embed_dim + width, first.out_features)

    def load_baseline(self, state):
        state = dict(state)
        if self.variant == "no_user_id":
            del state["user_embedding.weight"]
        elif self.variant == "category32":
            old = state["category_embedding.weight"]
            width = self.category_embedding.embedding_dim
            rank = min(width, *old.shape)
            u, s, vh = torch.linalg.svd(old.float(), full_matrices=False)
            embedding = old.new_zeros((old.shape[0], width))
            embedding[:, :rank] = u[:, :rank] * s[:rank]
            original = state["mlp.0.weight"]
            begin, end = 2 * self.embed_dim, 3 * self.embed_dim
            category_weight = original.new_zeros((original.shape[0], width))
            category_weight[:, :rank] = original[:, begin:end] @ vh[:rank].T
            state["category_embedding.weight"] = embedding
            state["mlp.0.weight"] = torch.cat([original[:, :begin], category_weight, original[:, end:]], 1)
        self.load_state_dict(state)

    def raw_slices(self, user, batch):
        tokens = super().raw_slices(user, batch)
        if self.variant == "no_popularity":
            dense = tokens[6].clone()
            dense[:, 6] = 0
            dense[:, 9] = 0
            tokens = (*tokens[:6], dense, tokens[7])
        return tokens


def variant_for_experiment(name):
    for variant in VARIANTS:
        if name.startswith("structural_" + variant + "_seed"):
            return variant
    return None
