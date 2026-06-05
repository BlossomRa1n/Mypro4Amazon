"""
Extended DIN Model for Amazon Reviews 2023 Raw (All_Beauty).

Architecture:
┌──────────────────────────────────────────────────────────────┐
│  Discrete Features (Embedding layers)                         │
│    user_emb (256d) + item_emb (256d) + brand_emb (64d)       │
│    + verified_emb (4d)                                        │
│                                                               │
│  Sequence Features (DIN Attention)                            │
│    hist_items, hist_brands, hist_ratings, hist_time_deltas    │
│    → Target Attention with candidate item                     │
│                                                               │
│  Dense Features (MLP)                                         │
│    user_stats (6d) + item_stats (4d) + interaction (4d)       │
│                                                               │
│  Final: concat → FC(512→256→128→64) → sigmoid                │
└──────────────────────────────────────────────────────────────┘
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Multi-Feature DIN Attention
# ============================================================

class DINAttentionLayer(nn.Module):
    """
    DIN Attention with extra signals (rating, time_delta) added to
    the concat before the attention MLP.

    Input concat per history position:
      [hist_item_emb, target_item_emb, diff, prod,
       hist_brand_emb, target_brand_emb,
       hist_rating, hist_delta, hist_verified]
    """
    def __init__(self, embed_dim, brand_embed_dim=64):
        super().__init__()
        # item_emb * 4 + brand_emb * 2 + 3 float signals
        input_dim = embed_dim * 4 + brand_embed_dim * 2 + 3
        layers = []
        prev_dim = input_dim
        for h_dim in [embed_dim * 2, embed_dim]:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.PReLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, hist_item_emb, target_item_emb,
                hist_brand_emb=None, target_brand_emb=None,
                hist_ratings=None, hist_deltas=None, hist_verified=None,
                hist_mask=None):
        """
        Args:
            hist_item_emb: (B, S, D) — history item embeddings
            target_item_emb: (B, D) — target item embedding
            hist_brand_emb: (B, S, D_b) — history brand embeddings
            target_brand_emb: (B, D_b) — target brand embedding
            hist_ratings: (B, S, 1)
            hist_deltas: (B, S, 1)
            hist_verified: (B, S, 1)
            hist_mask: (B, S) — 1 for valid positions, 0 for padding

        Returns:
            weighted_hist: (B, D) — attention-pooled history representation
        """
        B, S, D = hist_item_emb.shape
        target_expanded = target_item_emb.unsqueeze(1).expand_as(hist_item_emb)

        diff = hist_item_emb - target_expanded
        prod = hist_item_emb * target_expanded

        # Base concat
        concat_parts = [hist_item_emb, target_expanded, diff, prod]

        # Brand features
        if hist_brand_emb is not None and target_brand_emb is not None:
            D_b = hist_brand_emb.shape[-1]
            target_brand_expanded = target_brand_emb.unsqueeze(1).expand(B, S, D_b)
            concat_parts.extend([hist_brand_emb, target_brand_expanded])

        # Float signals
        if hist_ratings is not None:
            concat_parts.append(hist_ratings.unsqueeze(-1) if hist_ratings.dim() == 2 else hist_ratings)
        if hist_deltas is not None:
            concat_parts.append(hist_deltas.unsqueeze(-1) if hist_deltas.dim() == 2 else hist_deltas)
        if hist_verified is not None:
            concat_parts.append(hist_verified.unsqueeze(-1) if hist_verified.dim() == 2 else hist_verified)

        concat = torch.cat(concat_parts, dim=-1)

        attn_scores = self.mlp(concat).squeeze(-1)  # (B, S)

        if hist_mask is not None:
            attn_scores = attn_scores.masked_fill(hist_mask == 0, -1e9)

        attn_weights = F.softmax(attn_scores, dim=-1).unsqueeze(-1)
        weighted_hist = (hist_item_emb * attn_weights).sum(dim=1)
        return weighted_hist


# ============================================================
# Extended DIN Model
# ============================================================

class DINExtendedModel(nn.Module):
    """
    Extended DIN with brand embedding, item quality signals, and
    richer user history (brand sequence + rating + time_delta + verified).
    """
    def __init__(self, num_users, num_items, num_brands, num_categories,
                 embed_dim=256, brand_embed_dim=64,
                 hidden_dims=None, hist_len=50, dropout=0.1):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        self.hist_len = hist_len
        self.embed_dim = embed_dim
        self.brand_embed_dim = brand_embed_dim

        # ---- Discrete Embedding Layers ----
        self.user_embedding = nn.Embedding(num_users, embed_dim, padding_idx=0)
        self.item_embedding = nn.Embedding(num_items, embed_dim, padding_idx=0)
        self.brand_embedding = nn.Embedding(num_brands, brand_embed_dim, padding_idx=0)
        self.category_embedding = nn.Embedding(num_categories, embed_dim, padding_idx=0)
        # verified_purchase: tiny embedding as weight booster
        self.verified_embedding = nn.Embedding(2, 4)

        # ---- DIN Attention ----
        self.attention_layer = DINAttentionLayer(embed_dim, brand_embed_dim)

        # ---- Feature Dimension Calculation ----
        # Discrete: user(256) + item(256) + cat(256) + brand(64) + verified(4) = 836
        # Sequence: weighted_hist(256)
        # Dense user: click_count, time_span, avg_rating, std_rating, verified_ratio, avg_helpful = 6
        # Dense item: item_click_count, created_at_ts, item_avg_rating, item_rating_number = 4
        # Interaction: user_emb*item_emb(256) + user_brand_match(1) = 257
        feature_dim = (embed_dim * 3 + brand_embed_dim + 4 +  # discrete
                       embed_dim +                              # sequence
                       6 + 4 +                                  # dense
                       embed_dim)                               # interaction

        layers = []
        prev_dim = feature_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.PReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.mlp = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.02)
                # padding_idx = 0
                if m.padding_idx is not None:
                    with torch.no_grad():
                        m.weight[m.padding_idx].fill_(0)

    def _score(self, batch):
        """
        Forward pass for a single (user, item) pair.

        Expected batch keys:
          User: user_id, hist_items, hist_brands, hist_ratings,
                hist_time_deltas, hist_verified, hist_len,
                click_count, time_span, user_avg_rating, user_std_rating,
                user_verified_ratio, user_avg_helpful
          Item: item_id, category_id, brand_id, item_click_count,
                created_at_ts, item_avg_rating, item_rating_number
        """
        B = batch['user_id'].size(0)

        # --- User embeddings ---
        user_emb = self.user_embedding(batch['user_id'])          # (B, 256)

        # --- Item embeddings ---
        item_emb = self.item_embedding(batch['item_id'])          # (B, 256)
        cat_emb = self.category_embedding(batch['category_id'])   # (B, 256)
        brand_emb = self.brand_embedding(batch['brand_id'])       # (B, 64)
        verified_emb = self.verified_embedding(
            batch.get('verified', torch.zeros(B, dtype=torch.long, device=item_emb.device))
        )  # (B, 4)

        # --- History (sequence) ---
        hist_items = batch['hist_items']                      # (B, S)
        hist_mask = (hist_items != 0).float()
        hist_emb = self.item_embedding(hist_items)            # (B, S, 256)

        # Brand history
        hist_brands = batch.get('hist_brands', torch.zeros_like(hist_items))
        hist_brand_emb = self.brand_embedding(hist_brands)    # (B, S, 64)

        # Extra sequence signals
        hist_ratings = batch.get('hist_ratings', None)
        hist_deltas = batch.get('hist_time_deltas', None)
        hist_verified = batch.get('hist_verified', None)

        # --- DIN Target Attention ---
        interest_activation = self.attention_layer(
            hist_emb, item_emb,
            hist_brand_emb=hist_brand_emb,
            target_brand_emb=brand_emb,
            hist_ratings=hist_ratings,
            hist_deltas=hist_deltas,
            hist_verified=hist_verified,
            hist_mask=hist_mask,
        )  # (B, 256)

        # --- Dense user features ---
        click_count = batch['click_count'].float().unsqueeze(-1)      # (B, 1)
        time_span = batch['time_span'].float().unsqueeze(-1)           # (B, 1)
        user_avg_rating = batch.get('user_avg_rating',
                                     torch.zeros(B, device=click_count.device)).float().unsqueeze(-1)
        user_std_rating = batch.get('user_std_rating',
                                     torch.zeros(B, device=click_count.device)).float().unsqueeze(-1)
        user_verified_ratio = batch.get('user_verified_ratio',
                                         torch.zeros(B, device=click_count.device)).float().unsqueeze(-1)
        user_avg_helpful = batch.get('user_avg_helpful',
                                      torch.zeros(B, device=click_count.device)).float().unsqueeze(-1)

        # --- Dense item features ---
        item_click_count = batch['item_click_count'].float().unsqueeze(-1)
        created_at_ts = batch['created_at_ts'].float().unsqueeze(-1)
        item_avg_rating = batch.get('item_avg_rating',
                                     torch.zeros(B, device=item_click_count.device)).float().unsqueeze(-1)
        item_rating_number = batch.get('item_rating_number',
                                        torch.zeros(B, device=item_click_count.device)).float().unsqueeze(-1)

        # --- Interaction features ---
        user_item_interaction = user_emb * item_emb  # element-wise product

        # --- Final concat ---
        concat_features = torch.cat([
            # Discrete
            user_emb,           # 256
            item_emb,           # 256
            cat_emb,            # 256
            brand_emb,          # 64
            verified_emb,       # 4
            # Sequence
            interest_activation,  # 256
            # Dense user (6)
            click_count, time_span, user_avg_rating,
            user_std_rating, user_verified_ratio, user_avg_helpful,
            # Dense item (4)
            item_click_count, created_at_ts, item_avg_rating, item_rating_number,
            # Interaction (256)
            user_item_interaction,
        ], dim=-1)

        logit = self.mlp(concat_features).squeeze(-1)
        return logit

    # --- Public API (matches existing DINModel interface) ---

    def forward(self, batch):
        """BCE pointwise scoring (for evaluation/prediction)."""
        return self._score(batch)

    def forward_pairwise(self, user_batch, pos_item_batch, neg_item_batch):
        """BPR pairwise: score positive and negative items separately."""
        pos_score = self._score({**user_batch, **pos_item_batch})
        neg_score = self._score({**user_batch, **neg_item_batch})
        return pos_score, neg_score

    def compute_bpr_loss(self, pos_score, neg_score):
        return -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8).mean()

    def compute_loss(self, logits, labels, click_weight=None, pos_weight=None):
        kwargs = {}
        if pos_weight is not None:
            kwargs['pos_weight'] = torch.tensor([pos_weight], device=logits.device)
        if click_weight is not None:
            kwargs['weight'] = click_weight
        return F.binary_cross_entropy_with_logits(logits, labels.float(), **kwargs)

    def predict(self, batch):
        return torch.sigmoid(self.forward(batch))
