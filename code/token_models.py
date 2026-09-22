"""Future-window DIN rankers with fixed-width semantic tokens.

The token builders keep DIN target attention candidate-dependent, then expose
the user, item, context, cross and dense sources as a fixed token sequence.
"""
import torch
from torch import nn
import torch.nn.functional as F

from model_ext import DINAttentionLayer


class RankMixerBlock(nn.Module):
    """One MLP-Mixer block for a fixed number of semantic tokens."""

    def __init__(self, token_count, token_dim, token_expansion=4, dropout=0.1):
        super().__init__()
        self.token_norm = nn.LayerNorm(token_dim)
        self.token_mlp = nn.Sequential(
            nn.Linear(token_count, token_count * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_count * 4, token_count),
        )
        self.channel_norm = nn.LayerNorm(token_dim)
        self.channel_mlp = nn.Sequential(
            nn.Linear(token_dim, token_dim * token_expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim * token_expansion, token_dim),
        )
        # Start close to independent tokens; the mixer contribution grows
        # during training instead of immediately destroying source identity.
        self.token_scale = nn.Parameter(torch.full((1,), 1e-3))
        self.channel_scale = nn.Parameter(torch.full((1,), 1e-3))

    def forward(self, tokens):
        mixed = self.token_norm(tokens).transpose(1, 2)
        mixed = self.token_mlp(mixed).transpose(1, 2)
        tokens = tokens + self.token_scale * mixed
        tokens = tokens + self.channel_scale * self.channel_mlp(self.channel_norm(tokens))
        return tokens


class SemanticTokenDIN(nn.Module):
    """DIN with six equal-width semantic tokens.

    Tokens are ordered as sequence, user, item, context, cross and dense.
    ``fusion='concat'`` isolates tokenization from the RankMixer.  The
    ``rankmixer`` variant keeps the same token builder and changes only the
    interaction head.
    """

    TOKEN_COUNT = 6
    TOKEN_NAMES = ("seq", "user", "item", "context", "cross", "dense")

    def __init__(self, num_users, num_items, num_brands, num_categories,
                 embed_dim=256, brand_embed_dim=64, hidden_dims=None,
                 hist_len=50, dropout=0.1, token_dim=None, fusion="concat",
                 ablate_tokens=None, cross_mode="raw"):
        super().__init__()
        if token_dim is None:
            token_dim = embed_dim
        if fusion not in ("concat", "rankmixer"):
            raise ValueError(fusion)
        hidden_dims = list(hidden_dims or [256, 128, 64])
        self.embed_dim = int(embed_dim)
        self.brand_embed_dim = int(brand_embed_dim)
        self.token_dim = int(token_dim)
        self.hist_len = int(hist_len)
        self.fusion = fusion
        if cross_mode not in ("raw", "normalized", "gated", "normalized_gated", "zero_cross"):
            raise ValueError(f"cross_mode={cross_mode}")
        self.cross_mode = str(cross_mode)
        unknown = set(ablate_tokens or ()) - set(self.TOKEN_NAMES)
        if unknown:
            raise ValueError(f"unknown token ablation: {sorted(unknown)}")
        self.ablate_tokens = tuple(ablate_tokens or ())

        self.user_embedding = nn.Embedding(num_users, embed_dim, padding_idx=0)
        self.item_embedding = nn.Embedding(num_items, embed_dim, padding_idx=0)
        self.brand_embedding = nn.Embedding(num_brands, brand_embed_dim, padding_idx=0)
        self.category_embedding = nn.Embedding(num_categories, embed_dim, padding_idx=0)
        self.attention_layer = DINAttentionLayer(embed_dim, brand_embed_dim)

        self.user_proj = self._projection(embed_dim, token_dim)
        self.item_proj = self._projection(embed_dim, token_dim)
        self.seq_proj = self._projection(embed_dim, token_dim)
        self.cross_proj = self._projection(embed_dim, token_dim)
        self.context_proj = nn.Sequential(
            nn.Linear(embed_dim + brand_embed_dim + 2, token_dim),
            nn.GELU(),
            nn.LayerNorm(token_dim),
        )
        self.dense_proj = nn.Sequential(
            nn.Linear(10, max(64, token_dim // 4)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(64, token_dim // 4), token_dim),
            nn.LayerNorm(token_dim),
        )
        # Fixed token positions already identify semantic roles for MLP-Mixer;
        # a zero-initialized type offset is available for later adaptation.
        self.token_type = nn.Parameter(torch.zeros(self.TOKEN_COUNT, token_dim))
        # Keep the gate parameter in every cross variant so parameter shapes
        # remain comparable. Raw/normalized start effectively ungated (1.0);
        # gated variants override this value to sigmoid(-2.944439) ~= 0.05.
        self.cross_gate_logit = nn.Parameter(torch.tensor(
            -2.944439 if "gated" in self.cross_mode else 10.0))

        if fusion == "rankmixer":
            self.mixer = RankMixerBlock(self.TOKEN_COUNT, token_dim, dropout=dropout)
        else:
            self.mixer = None

        # Keep the fusion head identical to DINExtendedModel.  This makes the
        # concat variant an input-tokenization comparison rather than a hidden
        # activation/normalization comparison.
        layers = []
        prev = self.TOKEN_COUNT * token_dim
        for width in hidden_dims:
            layers.extend([nn.Linear(prev, width), nn.BatchNorm1d(width), nn.PReLU(), nn.Dropout(dropout)])
            prev = width
        layers.append(nn.Linear(prev, 1))
        self.head = nn.Sequential(*layers)
        self._init_weights()

    @staticmethod
    def _projection(in_dim, out_dim):
        return nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0., std=0.02)
                if module.padding_idx is not None:
                    with torch.no_grad():
                        module.weight[module.padding_idx].zero_()

    def _encode_user(self, batch):
        user_emb = self.user_embedding(batch["user_id"])
        hist_items = batch["hist_items"]
        hist_brands = batch.get("hist_brands", torch.zeros_like(hist_items))
        return {
            "user_emb": user_emb,
            "hist_emb": self.item_embedding(hist_items),
            "hist_brand_emb": self.brand_embedding(hist_brands),
            "hist_ratings": batch.get("hist_ratings"),
            "hist_deltas": batch.get("hist_time_deltas"),
            "hist_verified": batch.get("hist_verified"),
            "hist_mask": (hist_items != 0).float(),
            "click_count": batch["click_count"].float().unsqueeze(-1),
            "time_span": batch["time_span"].float().unsqueeze(-1),
            "user_avg_rating": batch.get("user_avg_rating", torch.zeros_like(batch["click_count"])).float().unsqueeze(-1),
            "user_std_rating": batch.get("user_std_rating", torch.zeros_like(batch["click_count"])).float().unsqueeze(-1),
            "user_verified_ratio": batch.get("user_verified_ratio", torch.zeros_like(batch["click_count"])).float().unsqueeze(-1),
            "user_avg_helpful": batch.get("user_avg_helpful", torch.zeros_like(batch["click_count"])).float().unsqueeze(-1),
        }

    def _tokenize(self, user, batch):
        device = user["user_emb"].device
        size = user["user_emb"].shape[0]
        item = self.item_embedding(batch["item_id"])
        category = self.category_embedding(batch["category_id"])
        brand = self.brand_embedding(batch["brand_id"])
        zeros = torch.zeros(size, device=device)
        item_avg = batch.get("item_avg_rating", zeros).float().unsqueeze(-1)
        item_count = batch.get("item_rating_number", zeros).float().unsqueeze(-1)
        interest = self.attention_layer(
            user["hist_emb"], item,
            hist_brand_emb=user["hist_brand_emb"], target_brand_emb=brand,
            hist_ratings=user["hist_ratings"], hist_deltas=user["hist_deltas"],
            hist_verified=user["hist_verified"], hist_mask=user["hist_mask"],
        )
        dense = torch.cat([
            user["click_count"], user["time_span"], user["user_avg_rating"],
            user["user_std_rating"], user["user_verified_ratio"], user["user_avg_helpful"],
            batch.get("item_click_count", zeros).float().unsqueeze(-1),
            batch.get("created_at_ts", zeros).float().unsqueeze(-1),
            item_avg, item_count,
        ], dim=-1)
        context = torch.cat([category, brand, item_avg, item_count], dim=-1)
        cross_user = user["user_emb"]
        cross_item = item
        if "normalized" in self.cross_mode:
            cross_user = F.normalize(cross_user, dim=-1, eps=1e-6)
            cross_item = F.normalize(cross_item, dim=-1, eps=1e-6)
        cross = cross_user * cross_item
        if "normalized" in self.cross_mode:
            # Match the raw product's initial scale for the 256-dim SVD/item
            # and random/user initialization used by the controlled screen.
            cross = cross * 0.32
        cross_token = self.cross_proj(cross)
        if "gated" in self.cross_mode:
            cross_token = torch.sigmoid(self.cross_gate_logit) * cross_token
        tokens = torch.stack([
            self.seq_proj(interest),
            self.user_proj(user["user_emb"]),
            self.item_proj(item),
            self.context_proj(context),
            cross_token,
            self.dense_proj(dense),
        ], dim=1)
        tokens = tokens + self.token_type.unsqueeze(0)
        # ``zero_cross`` is a structural control: retain the cross projection,
        # gate parameter and token slot so checkpoints remain shape compatible,
        # but zero the complete slot after the type offset is applied.
        if self.cross_mode == "zero_cross":
            tokens[:, self.TOKEN_NAMES.index("cross"), :] = 0
        if self.ablate_tokens:
            keep = torch.ones(self.TOKEN_COUNT, device=device, dtype=tokens.dtype)
            for name in self.ablate_tokens:
                keep[self.TOKEN_NAMES.index(name)] = 0
            tokens = tokens * keep.view(1, -1, 1)
        return tokens

    def _score_item(self, user, batch):
        tokens = self._tokenize(user, batch)
        if self.mixer is not None:
            tokens = self.mixer(tokens)
        return self.head(tokens.reshape(tokens.shape[0], -1)).squeeze(-1)

    def _score(self, batch):
        return self._score_item(self._encode_user(batch), batch)

    def forward(self, batch):
        return self._score(batch)

    def forward_pairwise(self, user_batch, pos_item_batch, neg_item_batch):
        return self._score({**user_batch, **pos_item_batch}), self._score({**user_batch, **neg_item_batch})

    def forward_bpr(self, user_batch, pos_batch, neg_batch):
        batch_size = user_batch["user_id"].shape[0]
        negative_count = neg_batch["item_id"].shape[1]
        user = self._encode_user(user_batch)
        items = {key: torch.cat([pos_batch[key].unsqueeze(1), value], dim=1).reshape(-1)
                 for key, value in neg_batch.items()}
        repeated = {key: (value.repeat_interleave(negative_count + 1, dim=0)
                          if value is not None else None)
                    for key, value in user.items()}
        scores = self._score_item(repeated, items).reshape(batch_size, negative_count + 1)
        return scores[:, 0], scores[:, 1:]

    @staticmethod
    def compute_bpr_loss(pos_score, neg_score):
        if neg_score.dim() > pos_score.dim():
            pos_score = pos_score.unsqueeze(-1)
        return F.softplus(neg_score - pos_score).mean()
