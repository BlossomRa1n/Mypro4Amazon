import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ItemTower(nn.Module):
    def __init__(self, num_items, num_categories, embed_dim, hidden_dims,
                 num_brands=0, brand_embed_dim=64):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items, embed_dim, padding_idx=0)
        self.category_embedding = nn.Embedding(num_categories, embed_dim, padding_idx=0)
        self.num_brands = num_brands
        self.brand_embed_dim = brand_embed_dim
        if num_brands > 0:
            self.brand_embedding = nn.Embedding(num_brands, brand_embed_dim, padding_idx=0)
        else:
            self.brand_embedding = None

        # input_dim calculation depends on num_brands
        # When num_brands > 0: embed_dim*2 + brand_embed_dim + 4 (brand + 2 quality scalars)
        # When num_brands == 0: embed_dim*2 + 2 (legacy: only click_count + created_at)
        if num_brands > 0:
            input_dim = embed_dim * 2 + brand_embed_dim + 4
        else:
            input_dim = embed_dim * 2 + 2
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, embed_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, item_id, category_id, brand_id=None,
                item_click_count=None, created_at_ts=None,
                item_avg_rating=None, item_rating_number=None):
        item_emb = self.item_embedding(item_id)
        cat_emb = self.category_embedding(category_id)

        B = item_emb.size(0)
        _zeros = lambda: torch.zeros(B, 1, device=item_emb.device)
        item_click_norm = _zeros() if item_click_count is None else item_click_count.float().unsqueeze(-1)
        created_norm = _zeros() if created_at_ts is None else created_at_ts.float().unsqueeze(-1)

        # Only include extended features when num_brands > 0 (V2+ config)
        if self.num_brands > 0:
            if self.brand_embedding is not None and brand_id is not None:
                brand_emb = self.brand_embedding(brand_id)
            else:
                brand_emb = torch.zeros(B, self.brand_embed_dim, device=item_emb.device)
            item_avg_rating_norm = _zeros() if item_avg_rating is None else item_avg_rating.float().unsqueeze(-1)
            item_rating_num_norm = _zeros() if item_rating_number is None else item_rating_number.float().unsqueeze(-1)
            x = torch.cat([item_emb, cat_emb, brand_emb, item_click_norm, created_norm,
                          item_avg_rating_norm, item_rating_num_norm], dim=-1)
        else:
            # No brand/quality features (num_brands == 0)
            x = torch.cat([item_emb, cat_emb, item_click_norm, created_norm], dim=-1)

        item_vec = self.mlp(x)
        item_vec = F.normalize(item_vec, p=2, dim=-1)
        return item_vec


# ============================================================
# Innovation 1: SASRec-based User Tower
# Self-Attentive Sequential Recommendation
# 用 Transformer 自注意力替代简单平均池化，捕获序列兴趣演变
# ============================================================

class SASRecBlock(nn.Module):
    """
    SASRec 核心模块: Causal Self-Attention + Feed-Forward
    因果自注意力确保只能看到历史，不能看到未来
    """
    def __init__(self, embed_dim, num_heads=2, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        residual = x
        x = self.layernorm1(x)
        attn_out, _ = self.attention(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        x = residual + attn_out
        residual = x
        x = self.layernorm2(x)
        x = residual + self.ffn(x)
        return x


class SASRecUserTower(nn.Module):
    """
    SASRec 用户塔: 用 Transformer 对用户行为序列建模
    相比平均池化的优势:
    1. 捕获兴趣随时间的演变 (短期 vs 长期)
    2. 自注意力自动学习哪些历史行为与当前预测更相关
    3. 位置编码保留时序信息
    """
    def __init__(self, num_users, num_items, embed_dim, hidden_dims, hist_len=50,
                 num_heads=2, num_blocks=2, dropout=0.1,
                 shared_item_embedding=None,  # 共享 ItemTower 的 embedding
                 shared_brand_embedding=None,  # 共享 ItemTower 的 brand_embedding (品牌偏好)
                 use_brand_pref=False, use_time_decay=False, time_decay_lambda=0.3):
        super().__init__()
        # user_id embedding 表已移除: 原 831494×256≈2.13亿参数, 平均每用户~8次交互下
        # 欠训练 + 构成绕过序列的"背答案"捷径。用户表示改为纯序列 + 统计特征。
        # num_users 参数保留仅为兼容 checkpoint 里的 config (不再使用)。
        if shared_item_embedding is not None:
            self.item_embedding = shared_item_embedding
        else:
            self.item_embedding = nn.Embedding(num_items, embed_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(hist_len, embed_dim)
        self.hist_len = hist_len
        self.embed_dim = embed_dim
        self.use_brand_pref = use_brand_pref
        self.use_time_decay = use_time_decay
        self.time_decay_lambda = time_decay_lambda
        if shared_brand_embedding is not None:
            self.brand_embedding = shared_brand_embedding
            self.brand_embed_dim = shared_brand_embedding.embedding_dim
        else:
            self.brand_embedding = None
            self.brand_embed_dim = 0

        self.sasrec_blocks = nn.ModuleList([
            SASRecBlock(embed_dim, num_heads, dropout) for _ in range(num_blocks)
        ])

        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(embed_dim)

        input_dim = embed_dim + 4  # hist_vec + 4 stats (user_emb 已移除)
        if self.use_brand_pref and self.brand_embedding is not None:
            input_dim += self.brand_embed_dim  # + 品牌偏好向量
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, embed_dim))
        self.mlp = nn.Sequential(*layers)

        causal_mask = torch.triu(torch.ones(hist_len, hist_len), diagonal=1).bool()
        self.register_buffer('causal_mask', causal_mask)

    def forward(self, hist_items, hist_len, click_count, time_span,
                user_avg_rating=None, user_std_rating=None,
                hist_brands=None, hist_time_deltas=None):
        seq_emb = self.item_embedding(hist_items)
        positions = torch.arange(self.hist_len, device=hist_items.device).unsqueeze(0)
        seq_emb = seq_emb + self.position_embedding(positions)
        seq_emb = self.layernorm(seq_emb)
        seq_emb = self.dropout(seq_emb)

        padding_mask = (hist_items == 0)

        for block in self.sasrec_blocks:
            seq_emb = block(seq_emb, attn_mask=self.causal_mask, key_padding_mask=padding_mask)

        hist_mask = (hist_items != 0).unsqueeze(-1).float()   # (B, hist_len, 1)
        seq_output = seq_emb * hist_mask

        # ---- 时间衰减加权池化 (距今时间 → exp 衰减) ----
        if self.use_time_decay and hist_time_deltas is not None:
            # recency[i] = 第 i 个交互距最后一个交互的总时间 (反向累积和)
            recency = torch.flip(torch.cumsum(torch.flip(hist_time_deltas, dims=[1]), dim=1), dims=[1])
            time_weight = torch.exp(-self.time_decay_lambda * recency)   # (B, hist_len)
            denom = (time_weight * hist_mask.squeeze(-1)).sum(dim=1).clamp(min=1)  # (B,)
            hist_vec = (seq_output * time_weight.unsqueeze(-1)).sum(dim=1) / denom.unsqueeze(-1)
        else:
            hist_vec = seq_output.sum(dim=1) / hist_len.unsqueeze(-1).clamp(min=1)

        B, D = hist_vec.shape
        _zeros = lambda: torch.zeros(B, 1, device=hist_vec.device)
        click_count_norm = _zeros() if click_count is None else click_count.float().unsqueeze(-1)
        time_span_norm = _zeros() if time_span is None else time_span.float().unsqueeze(-1)
        user_avg_rating_norm = _zeros() if user_avg_rating is None else user_avg_rating.float().unsqueeze(-1)
        user_std_rating_norm = _zeros() if user_std_rating is None else user_std_rating.float().unsqueeze(-1)

        parts = [hist_vec, click_count_norm, time_span_norm,
                 user_avg_rating_norm, user_std_rating_norm]

        # ---- 品牌偏好向量 (历史品牌 embedding 平均池化) ----
        if self.use_brand_pref and self.brand_embedding is not None:
            if hist_brands is not None:
                brand_emb = self.brand_embedding(hist_brands)          # (B, hist_len, brand_dim)
                brand_pref = (brand_emb * hist_mask).sum(dim=1) / hist_len.unsqueeze(-1).clamp(min=1)
            else:
                brand_pref = torch.zeros(B, self.brand_embed_dim, device=hist_vec.device)
            parts.append(brand_pref)

        x = torch.cat(parts, dim=-1)
        user_vec = self.mlp(x)
        user_vec = F.normalize(user_vec, p=2, dim=-1)
        return user_vec


# ============================================================
# Innovation 1b: HSTU-based User Tower (Meta 2024, 第 5 路召回)
# Hierarchical Sequential Transduction Unit — pointwise 传导替代 softmax 注意力
# ============================================================

class HSTUBlock(nn.Module):
    """
    HSTU (Hierarchical Sequential Transduction Unit) 核心模块 — Meta 2024。
    用 pointwise (逐元素) 交互 + 因果 cumsum 替代 softmax 注意力:
      - 复杂度 O(L·D)/层 vs Transformer 自注意力的 O(L²·D), 长序列更高效
      - 无 attention 矩阵, q⊙k 逐元素相乘 → cumsum 因果聚合 → v 门控
      - 与 SASRec 是「同角色异读法」: 两者读序列的方式不同 → 用户表示互补
    """
    def __init__(self, embed_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.w_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.w_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.w_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.w_o = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask=None):
        # --- Pointwise transduction (O(L·D)) ---
        residual = x
        n = self.norm1(x)
        q = self.w_q(n)
        k = self.w_k(n)
        v = self.w_v(n)
        att = q * k                               # (B, L, D) 逐元素交互
        if padding_mask is not None:
            att = att * (~padding_mask).unsqueeze(-1)
        att = att.cumsum(dim=1)                   # 因果聚合: 位置 t 只看 ≤ t
        if padding_mask is not None:
            att = att * (~padding_mask).unsqueeze(-1)
        out = v * att                             # v 门控聚合结果
        x = residual + self.dropout(out)

        # --- 前馈 (对应 Transformer FFN) ---
        residual = x
        x = x + self.dropout(self.w_o(F.silu(self.norm2(x))))
        return x


class HSTUUserTower(nn.Module):
    """
    HSTU 用户塔: pointwise 序列传导建模用户行为序列。
    相比 SASRec (softmax 自注意力):
    1. Pointwise 交互 + 因果 cumsum, O(L·D)/层, 长序列更高效
    2. 逐元素门控 (q⊙k⊙v) 学习序列局部→全局的层级结构
    3. 与 SASRec 同角色异读法 → 召回候选互补, 适合做独立第 5 路
    """
    def __init__(self, num_users, num_items, embed_dim, hidden_dims, hist_len=50,
                 num_blocks=2, dropout=0.1, shared_item_embedding=None):
        super().__init__()
        # user_id embedding 表已移除 (同 SASRecUserTower, 见其注释: 背答案捷径)
        if shared_item_embedding is not None:
            self.item_embedding = shared_item_embedding
        else:
            self.item_embedding = nn.Embedding(num_items, embed_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(hist_len, embed_dim)
        self.hist_len = hist_len
        self.embed_dim = embed_dim

        self.hstu_blocks = nn.ModuleList([
            HSTUBlock(embed_dim, dropout) for _ in range(num_blocks)
        ])

        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(embed_dim)

        input_dim = embed_dim + 4  # hist_vec + 4 stats (user_emb 已移除)
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, embed_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, hist_items, hist_len, click_count, time_span,
                user_avg_rating=None, user_std_rating=None,
                hist_brands=None, hist_time_deltas=None):
        seq_emb = self.item_embedding(hist_items)
        positions = torch.arange(self.hist_len, device=hist_items.device).unsqueeze(0)
        seq_emb = seq_emb + self.position_embedding(positions)
        seq_emb = self.layernorm(seq_emb)
        seq_emb = self.dropout(seq_emb)

        padding_mask = (hist_items == 0)

        for block in self.hstu_blocks:
            seq_emb = block(seq_emb, padding_mask=padding_mask)

        hist_mask = (hist_items != 0).unsqueeze(-1).float()
        seq_output = seq_emb * hist_mask
        hist_vec = seq_output.sum(dim=1) / hist_len.unsqueeze(-1).clamp(min=1)

        B, D = hist_vec.shape
        _zeros = lambda: torch.zeros(B, 1, device=hist_vec.device)
        click_count_norm = _zeros() if click_count is None else click_count.float().unsqueeze(-1)
        time_span_norm = _zeros() if time_span is None else time_span.float().unsqueeze(-1)
        user_avg_rating_norm = _zeros() if user_avg_rating is None else user_avg_rating.float().unsqueeze(-1)
        user_std_rating_norm = _zeros() if user_std_rating is None else user_std_rating.float().unsqueeze(-1)

        x = torch.cat([hist_vec, click_count_norm, time_span_norm,
                       user_avg_rating_norm, user_std_rating_norm], dim=-1)
        user_vec = self.mlp(x)
        user_vec = F.normalize(user_vec, p=2, dim=-1)
        return user_vec


# ============================================================
# Innovation 2: Two-Tower V2 with SASRec + InfoNCE Loss
# InfoNCE: In-batch Negative + Hard Negative Mining
# ============================================================

class TwoTowerV2Model(nn.Module):
    """
    增强版双塔模型:
    1. User Tower 使用 SASRec 序列建模 (替代平均池化)
    2. InfoNCE Loss (In-batch negatives, 一个batch内所有其他样本作为负样本)
    3. 支持 Hard Negative Mining (从 ItemCF 相似但未点击的 item 中挖掘)
    4. 扩展物品塔: brand_embedding + item_avg_rating + item_rating_number
    5. 扩展用户塔: user_avg_rating + user_std_rating
    """
    def __init__(self, num_users, num_items, num_categories, embed_dim=64, hidden_dims=None,
                 hist_len=50, temperature=0.07, num_heads=2, num_blocks=2, dropout=0.1,
                 num_hard_negatives=0, num_brands=0, brand_embed_dim=64, user_tower_type='sasrec',
                 use_brand_pref=False, use_time_decay=False, time_decay_lambda=0.3):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]
        # 先创建 ItemTower, 再让 UserTower 共享同一个 item_embedding
        self.item_tower = ItemTower(num_items, num_categories, embed_dim, hidden_dims,
                                     num_brands=num_brands, brand_embed_dim=brand_embed_dim)
        if user_tower_type == 'hstu':
            self.user_tower = HSTUUserTower(
                num_users, num_items, embed_dim, hidden_dims, hist_len,
                num_blocks, dropout,
                shared_item_embedding=self.item_tower.item_embedding
            )
        else:
            self.user_tower = SASRecUserTower(
                num_users, num_items, embed_dim, hidden_dims, hist_len,
                num_heads, num_blocks, dropout,
                shared_item_embedding=self.item_tower.item_embedding,
                shared_brand_embedding=self.item_tower.brand_embedding,
                use_brand_pref=use_brand_pref,
                use_time_decay=use_time_decay,
                time_decay_lambda=time_decay_lambda,
            )
        self.user_tower_type = user_tower_type
        self.temperature = temperature
        self.num_hard_negatives = num_hard_negatives

    def forward(self, batch):
        user_vec = self.get_user_embedding(batch)
        pos_item_vec = self.get_item_embedding({
            'item_id': batch['pos_item_id'],
            'category_id': batch['pos_category_id'],
            'item_click_count': batch['pos_item_click_count'],
            'created_at_ts': batch['pos_created_at_ts'],
            'brand_id': batch.get('pos_brand_id'),
            'item_avg_rating': batch.get('pos_item_avg_rating'),
            'item_rating_number': batch.get('pos_item_rating_number'),
        })
        return user_vec, pos_item_vec

    def compute_infonce_loss(self, user_vec, pos_item_vec, hard_neg_vecs=None, click_weight=None):
        """
        InfoNCE Loss (NT-Xent):
        - In-batch negatives: 同一batch内其他用户的正样本作为负样本
        - Hard negatives: 额外传入的困难负样本 (ItemCF 相似但未点击)
        - click_weight: 每个正样本的置信度权重 (5分=1.0, 4分=0.5)
        """
        user_vec = F.normalize(user_vec, p=2, dim=-1)
        pos_item_vec = F.normalize(pos_item_vec, p=2, dim=-1)

        pos_score = torch.sum(user_vec * pos_item_vec, dim=-1) / self.temperature

        batch_size = user_vec.size(0)
        device = user_vec.device

        sim_matrix = torch.matmul(user_vec, pos_item_vec.t()) / self.temperature  # (B, B)

        labels = torch.arange(batch_size, device=device)

        # 逐样本加权 cross-entropy
        ce_per_sample = F.cross_entropy(sim_matrix, labels, reduction='none')
        if click_weight is not None:
            loss = (ce_per_sample * click_weight).sum() / (click_weight.sum() + 1e-8)
        else:
            loss = ce_per_sample.mean()

        if hard_neg_vecs is not None and self.num_hard_negatives > 0:
            hard_neg_scores = torch.matmul(
                user_vec.unsqueeze(1),
                hard_neg_vecs.transpose(1, 2)
            ).squeeze(1) / self.temperature

            pos_score_exp = torch.exp(pos_score).unsqueeze(1)
            hard_neg_exp = torch.exp(hard_neg_scores)
            hard_loss = -torch.log(
                pos_score_exp / (pos_score_exp + hard_neg_exp.sum(dim=1, keepdim=True) + 1e-8)
            ).mean()
            loss = loss + 0.3 * hard_loss

        return loss

    def get_user_embedding(self, batch):
        return self.user_tower(
            batch.get('hist_items'),
            batch.get('hist_len'),
            batch.get('click_count'),
            batch.get('time_span'),
            user_avg_rating=batch.get('user_avg_rating'),
            user_std_rating=batch.get('user_std_rating'),
            hist_brands=batch.get('hist_brands'),
            hist_time_deltas=batch.get('hist_time_deltas'),
        )

    def get_item_embedding(self, batch):
        return self.item_tower(
            batch.get('item_id'),
            batch.get('category_id'),
            brand_id=batch.get('brand_id'),
            item_click_count=batch.get('item_click_count'),
            created_at_ts=batch.get('created_at_ts'),
            item_avg_rating=batch.get('item_avg_rating'),
            item_rating_number=batch.get('item_rating_number'),
        )


# ============================================================
# Innovation 3: MIND - Multi-Interest Network with Dynamic Routing
# 将用户表示为多个兴趣向量，每个向量代表一个兴趣方向
# ============================================================

class LabelAwareAttention(nn.Module):
    """
    标签感知注意力: 在推理时用候选 item 与多个兴趣向量计算注意力
    选择最相关的兴趣向量进行打分
    """
    def __init__(self, embed_dim, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.w = nn.Linear(embed_dim, 1, bias=False)

    def forward(self, interest_vectors, target_item_vec):
        """
        interest_vectors: (B, K, D) - K个兴趣向量
        target_item_vec: (B, D) - 目标item向量
        """
        target_expanded = target_item_vec.unsqueeze(1).expand_as(interest_vectors)
        dot_product = torch.sum(interest_vectors * target_expanded, dim=-1, keepdim=True)
        attention_weights = F.softmax(dot_product / self.temperature, dim=1)
        weighted_interest = (interest_vectors * attention_weights).sum(dim=1)
        return weighted_interest


class MINDUserTower(nn.Module):
    """
    MIND 用户塔: 多兴趣提取
    通过动态路由机制 (Dynamic Routing) 将用户历史行为聚类为 K 个兴趣胶囊
    每个胶囊代表用户的一个兴趣方向

    相比单向量表示的优势:
    1. 用户兴趣是多样的，单一向量无法同时表达"喜欢科技"和"喜欢娱乐"
    2. 多兴趣向量在召回时可以覆盖更多兴趣方向
    3. 动态路由自动发现兴趣聚类，无需人工标注
    """
    def __init__(self, num_users, num_items, embed_dim, hidden_dims, hist_len=50,
                 num_interests=3, num_routing_iterations=3, dropout=0.1,
                 shared_item_embedding=None):  # 共享 ItemTower 的 embedding
        super().__init__()
        # user_embedding 已移除 (背答案捷径, 同 SASRecUserTower / HSTUUserTower)
        if shared_item_embedding is not None:
            self.item_embedding = shared_item_embedding
        else:
            self.item_embedding = nn.Embedding(num_items, embed_dim, padding_idx=0)
        self.hist_len = hist_len
        self.embed_dim = embed_dim
        self.num_interests = num_interests
        self.num_routing_iterations = num_routing_iterations

        self.routing_linear = nn.Linear(embed_dim, embed_dim * num_interests)
        # 训练与评估统一直接使用 K 个兴趣胶囊 (interest_vectors), 不再接 label-aware + mlp
        # (原 mlp(label_aware(interest)) 与评估召回的 interest_vectors 空间断裂 → 训练不收敛)

    def dynamic_routing(self, hist_emb, hist_mask):
        """
        动态路由: 将历史行为 item embedding 路由到 K 个兴趣胶囊
        迭代更新路由权重，使每个胶囊专注于一个兴趣方向
        """
        B = hist_emb.size(0)
        S = hist_emb.size(1)

        primary_capsules = self.routing_linear(hist_emb)
        primary_capsules = primary_capsules.view(B, S, self.num_interests, self.embed_dim)

        routing_logits = torch.zeros(B, S, self.num_interests, device=hist_emb.device)

        for _ in range(self.num_routing_iterations):
            routing_weights = F.softmax(routing_logits, dim=-1)
            routing_weights = routing_weights * hist_mask.unsqueeze(-1)

            interest_capsules = torch.einsum('bsk,bskd->bkd', routing_weights, primary_capsules)

            # 真正的 squash: 方向不变, 长度压缩到 [0,1)
            # (原 * (norm/(norm+1e-8)) 是恒等变换, 无效)
            norm = interest_capsules.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            norm2 = (interest_capsules ** 2).sum(dim=-1, keepdim=True)
            interest_capsules = (norm2 / (1.0 + norm2)) * (interest_capsules / norm)

            agreement = torch.einsum('bskd,bkd->bsk', primary_capsules, interest_capsules)
            routing_logits = routing_logits + agreement

        interest_capsules = F.normalize(interest_capsules, p=2, dim=-1)
        return interest_capsules

    def forward(self, user_id, hist_items, hist_len, click_count, time_span,
                target_item_vec=None):
        hist_emb = self.item_embedding(hist_items)
        hist_mask = (hist_items != 0).float()

        interest_vectors = self.dynamic_routing(hist_emb, hist_mask)
        # 训练与评估统一返回 K 个兴趣胶囊 (B, K, D):
        #   评估 = 每个兴趣独立 top-N 召回; 训练 loss = 每个兴趣独立 InfoNCE 取均值
        return interest_vectors


class MINDModel(nn.Module):
    """
    MIND 完整模型: 多兴趣网络
    训练时用 Label-Aware Attention 选择最相关兴趣
    推理时用每个兴趣向量独立召回，合并结果
    """
    def __init__(self, num_users, num_items, num_categories, embed_dim=64, hidden_dims=None,
                 hist_len=50, temperature=0.07, num_interests=3, num_routing_iterations=3,
                 dropout=0.1, num_brands=0, brand_embed_dim=64):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]
        # 扩展 ItemTower (brand/quality), 与 V2/HSTU 同一物品空间
        self.item_tower = ItemTower(num_items, num_categories, embed_dim, hidden_dims,
                                     num_brands=num_brands, brand_embed_dim=brand_embed_dim)
        self.user_tower = MINDUserTower(
            num_users, num_items, embed_dim, hidden_dims, hist_len,
            num_interests, num_routing_iterations, dropout,
            shared_item_embedding=self.item_tower.item_embedding
        )
        self.temperature = temperature

    def forward(self, batch):
        pos_item_vec = self.item_tower(
            batch['pos_item_id'],
            batch['pos_category_id'],
            brand_id=batch.get('pos_brand_id'),
            item_click_count=batch['pos_item_click_count'],
            created_at_ts=batch['pos_created_at_ts'],
            item_avg_rating=batch.get('pos_item_avg_rating'),
            item_rating_number=batch.get('pos_item_rating_number'),
        )

        interest_vectors = self.user_tower(
            batch['user_id'],
            batch['hist_items'],
            batch['hist_len'],
            batch['click_count'],
            batch['time_span'],
            target_item_vec=None
        )

        return interest_vectors, pos_item_vec

    def compute_infonce_loss(self, interest_vectors, pos_item_vec, hard_neg_vecs=None):
        # 多兴趣 InfoNCE: K 个兴趣胶囊每个独立做 in-batch negative softmax, 取均值。
        # 训练 loss 直接作用在 interest_vectors 上 (评估召回的同一空间), 消除空间断裂。
        interest_vectors = F.normalize(interest_vectors, p=2, dim=-1)   # (B, K, D)
        pos_item_vec = F.normalize(pos_item_vec, p=2, dim=-1)           # (B, D)

        B, K, D = interest_vectors.shape
        labels = torch.arange(B, device=interest_vectors.device)
        losses = []
        for k in range(K):
            sim_matrix = torch.matmul(interest_vectors[:, k], pos_item_vec.t()) / self.temperature
            losses.append(F.cross_entropy(sim_matrix, labels))
        return torch.stack(losses).mean()

    def get_user_embedding(self, batch):
        interest_vectors = self.user_tower(
            batch['user_id'],
            batch['hist_items'],
            batch['hist_len'],
            batch['click_count'],
            batch['time_span']
        )
        return interest_vectors

    def get_item_embedding(self, batch):
        return self.item_tower(
            batch['item_id'],
            batch['category_id'],
            brand_id=batch.get('brand_id'),
            item_click_count=batch['item_click_count'],
            created_at_ts=batch['created_at_ts'],
            item_avg_rating=batch.get('item_avg_rating'),
            item_rating_number=batch.get('item_rating_number'),
        )


