import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class UserTower(nn.Module):
    def __init__(self, num_users, num_items, embed_dim, hidden_dims, hist_len=50):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embed_dim, padding_idx=0)
        self.item_embedding_for_hist = nn.Embedding(num_items, embed_dim, padding_idx=0)
        self.hist_len = hist_len
        self.embed_dim = embed_dim

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

    def forward(self, user_id, hist_items, hist_len, click_count, time_span):
        user_emb = self.user_embedding(user_id)
        hist_emb = self.item_embedding_for_hist(hist_items)
        hist_mask = (hist_items != 0).unsqueeze(-1).float()
        hist_emb = (hist_emb * hist_mask).sum(dim=1) / hist_len.unsqueeze(-1).clamp(min=1)

        click_count_norm = click_count.float().unsqueeze(-1)
        time_span_norm = time_span.float().unsqueeze(-1)

        x = torch.cat([user_emb, hist_emb, click_count_norm, time_span_norm], dim=-1)
        user_vec = self.mlp(x)
        user_vec = F.normalize(user_vec, p=2, dim=-1)
        return user_vec


class ItemTower(nn.Module):
    def __init__(self, num_items, num_categories, embed_dim, hidden_dims):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items, embed_dim, padding_idx=0)
        self.category_embedding = nn.Embedding(num_categories, embed_dim, padding_idx=0)

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

    def forward(self, item_id, category_id, item_click_count, created_at_ts):
        item_emb = self.item_embedding(item_id)
        cat_emb = self.category_embedding(category_id)

        item_click_norm = item_click_count.float().unsqueeze(-1)
        created_norm = created_at_ts.float().unsqueeze(-1)

        x = torch.cat([item_emb, cat_emb, item_click_norm, created_norm], dim=-1)
        item_vec = self.mlp(x)
        item_vec = F.normalize(item_vec, p=2, dim=-1)
        return item_vec


class TwoTowerModel(nn.Module):
    def __init__(self, num_users, num_items, num_categories, embed_dim=64, hidden_dims=None, hist_len=50, temperature=0.1):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]
        self.user_tower = UserTower(num_users, num_items, embed_dim, hidden_dims, hist_len)
        self.item_tower = ItemTower(num_items, num_categories, embed_dim, hidden_dims)
        self.temperature = temperature

    def forward(self, batch):
        user_vec = self.user_tower(
            batch['user_id'],
            batch['hist_items'],
            batch['hist_len'],
            batch['click_count'],
            batch['time_span']
        )
        pos_item_vec = self.item_tower(
            batch['pos_item_id'],
            batch['pos_category_id'],
            batch['pos_item_click_count'],
            batch['pos_created_at_ts']
        )
        neg_item_vec = self.item_tower(
            batch['neg_item_id'],
            batch['neg_category_id'],
            batch['neg_item_click_count'],
            batch['neg_created_at_ts']
        )

        pos_score = torch.sum(user_vec * pos_item_vec, dim=-1) / self.temperature
        neg_score = torch.sum(user_vec * neg_item_vec, dim=-1) / self.temperature

        return pos_score, neg_score, user_vec, pos_item_vec, neg_item_vec

    def compute_loss(self, pos_score, neg_score, click_weight=None):
        # BPR loss with per-sample weight
        loss_per_sample = -torch.log(torch.sigmoid(pos_score - neg_score))
        if click_weight is not None:
            loss = (loss_per_sample * click_weight).sum() / (click_weight.sum() + 1e-8)
        else:
            loss = loss_per_sample.mean()
        return loss

    def get_user_embedding(self, batch):
        return self.user_tower(
            batch['user_id'],
            batch['hist_items'],
            batch['hist_len'],
            batch['click_count'],
            batch['time_span']
        )

    def get_item_embedding(self, batch):
        return self.item_tower(
            batch['item_id'],
            batch['category_id'],
            batch['item_click_count'],
            batch['created_at_ts']
        )


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
                 num_heads=2, num_blocks=2, dropout=0.1):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embed_dim, padding_idx=0)
        self.item_embedding = nn.Embedding(num_items, embed_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(hist_len, embed_dim)
        self.hist_len = hist_len
        self.embed_dim = embed_dim

        self.sasrec_blocks = nn.ModuleList([
            SASRecBlock(embed_dim, num_heads, dropout) for _ in range(num_blocks)
        ])

        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(embed_dim)

        input_dim = embed_dim * 2 + 2
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

    def forward(self, user_id, hist_items, hist_len, click_count, time_span):
        user_emb = self.user_embedding(user_id)

        seq_emb = self.item_embedding(hist_items)
        positions = torch.arange(self.hist_len, device=hist_items.device).unsqueeze(0)
        seq_emb = seq_emb + self.position_embedding(positions)
        seq_emb = self.layernorm(seq_emb)
        seq_emb = self.dropout(seq_emb)

        padding_mask = (hist_items == 0)

        for block in self.sasrec_blocks:
            seq_emb = block(seq_emb, attn_mask=self.causal_mask, key_padding_mask=padding_mask)

        hist_mask = (hist_items != 0).unsqueeze(-1).float()
        seq_output = seq_emb * hist_mask
        hist_vec = seq_output.sum(dim=1) / hist_len.unsqueeze(-1).clamp(min=1)

        click_count_norm = click_count.float().unsqueeze(-1)
        time_span_norm = time_span.float().unsqueeze(-1)

        x = torch.cat([user_emb, hist_vec, click_count_norm, time_span_norm], dim=-1)
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
    """
    def __init__(self, num_users, num_items, num_categories, embed_dim=64, hidden_dims=None,
                 hist_len=50, temperature=0.07, num_heads=2, num_blocks=2, dropout=0.1,
                 num_hard_negatives=0):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]
        self.user_tower = SASRecUserTower(
            num_users, num_items, embed_dim, hidden_dims, hist_len,
            num_heads, num_blocks, dropout
        )
        self.item_tower = ItemTower(num_items, num_categories, embed_dim, hidden_dims)
        self.temperature = temperature
        self.num_hard_negatives = num_hard_negatives

    def forward(self, batch):
        user_vec = self.user_tower(
            batch['user_id'],
            batch['hist_items'],
            batch['hist_len'],
            batch['click_count'],
            batch['time_span']
        )
        pos_item_vec = self.item_tower(
            batch['pos_item_id'],
            batch['pos_category_id'],
            batch['pos_item_click_count'],
            batch['pos_created_at_ts']
        )
        return user_vec, pos_item_vec

    def compute_infonce_loss(self, user_vec, pos_item_vec, hard_neg_vecs=None, click_weight=None):
        """
        InfoNCE Loss (NT-Xent):
        - In-batch negatives: 同一batch内其他用户的正样本作为负样本
        - Hard negatives: 额外传入的困难负样本
        - click_weight: 每个正样本的置信度权重 (5分=1.0, 4分=0.5)
        """
        user_vec = F.normalize(user_vec, p=2, dim=-1)
        pos_item_vec = F.normalize(pos_item_vec, p=2, dim=-1)

        pos_score = torch.sum(user_vec * pos_item_vec, dim=-1) / self.temperature

        sim_matrix = torch.matmul(user_vec, pos_item_vec.t()) / self.temperature

        batch_size = user_vec.size(0)
        labels = torch.arange(batch_size, device=user_vec.device)

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
            batch['user_id'],
            batch['hist_items'],
            batch['hist_len'],
            batch['click_count'],
            batch['time_span']
        )

    def get_item_embedding(self, batch):
        return self.item_tower(
            batch['item_id'],
            batch['category_id'],
            batch['item_click_count'],
            batch['created_at_ts']
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
                 num_interests=3, num_routing_iterations=3, dropout=0.1):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embed_dim, padding_idx=0)
        self.item_embedding = nn.Embedding(num_items, embed_dim, padding_idx=0)
        self.hist_len = hist_len
        self.embed_dim = embed_dim
        self.num_interests = num_interests
        self.num_routing_iterations = num_routing_iterations

        self.routing_linear = nn.Linear(embed_dim, embed_dim * num_interests)

        self.label_aware_attention = LabelAwareAttention(embed_dim)

        input_dim = embed_dim * 2 + 2
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

            norm = interest_capsules.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            interest_capsules = interest_capsules * (norm / (norm + 1e-8))

            agreement = torch.einsum('bskd,bkd->bsk', primary_capsules, interest_capsules)
            routing_logits = routing_logits + agreement

        interest_capsules = F.normalize(interest_capsules, p=2, dim=-1)
        return interest_capsules

    def forward(self, user_id, hist_items, hist_len, click_count, time_span,
                target_item_vec=None):
        user_emb = self.user_embedding(user_id)

        hist_emb = self.item_embedding(hist_items)
        hist_mask = (hist_items != 0).float()

        interest_vectors = self.dynamic_routing(hist_emb, hist_mask)

        if target_item_vec is not None:
            interest_vec = self.label_aware_attention(interest_vectors, target_item_vec)
        else:
            interest_vec = interest_vectors.mean(dim=1)

        click_count_norm = click_count.float().unsqueeze(-1)
        time_span_norm = time_span.float().unsqueeze(-1)

        x = torch.cat([user_emb, interest_vec, click_count_norm, time_span_norm], dim=-1)
        user_vec = self.mlp(x)
        user_vec = F.normalize(user_vec, p=2, dim=-1)
        return user_vec, interest_vectors


class MINDModel(nn.Module):
    """
    MIND 完整模型: 多兴趣网络
    训练时用 Label-Aware Attention 选择最相关兴趣
    推理时用每个兴趣向量独立召回，合并结果
    """
    def __init__(self, num_users, num_items, num_categories, embed_dim=64, hidden_dims=None,
                 hist_len=50, temperature=0.07, num_interests=3, num_routing_iterations=3,
                 dropout=0.1):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]
        self.user_tower = MINDUserTower(
            num_users, num_items, embed_dim, hidden_dims, hist_len,
            num_interests, num_routing_iterations, dropout
        )
        self.item_tower = ItemTower(num_items, num_categories, embed_dim, hidden_dims)
        self.temperature = temperature

    def forward(self, batch):
        pos_item_vec = self.item_tower(
            batch['pos_item_id'],
            batch['pos_category_id'],
            batch['pos_item_click_count'],
            batch['pos_created_at_ts']
        )

        user_vec, interest_vectors = self.user_tower(
            batch['user_id'],
            batch['hist_items'],
            batch['hist_len'],
            batch['click_count'],
            batch['time_span'],
            target_item_vec=pos_item_vec
        )

        return user_vec, pos_item_vec, interest_vectors

    def compute_infonce_loss(self, user_vec, pos_item_vec, hard_neg_vecs=None):
        user_vec = F.normalize(user_vec, p=2, dim=-1)
        pos_item_vec = F.normalize(pos_item_vec, p=2, dim=-1)

        sim_matrix = torch.matmul(user_vec, pos_item_vec.t()) / self.temperature
        batch_size = user_vec.size(0)
        labels = torch.arange(batch_size, device=user_vec.device)
        loss = F.cross_entropy(sim_matrix, labels)
        return loss

    def get_user_embedding(self, batch):
        user_vec, interest_vectors = self.user_tower(
            batch['user_id'],
            batch['hist_items'],
            batch['hist_len'],
            batch['click_count'],
            batch['time_span']
        )
        return user_vec, interest_vectors

    def get_item_embedding(self, batch):
        return self.item_tower(
            batch['item_id'],
            batch['category_id'],
            batch['item_click_count'],
            batch['created_at_ts']
        )


# ============================================================
# Innovation 4: DIN - Deep Interest Network (精排模型)
# 通过注意力机制让目标 item 与用户历史行为动态交互
# 捕获用户兴趣的多样性和局部激活特性
# ============================================================

class AttentionLayer(nn.Module):
    """
    DIN 注意力层: 计算目标 item 与历史行为中每个 item 的相关性
    输出: 加权求和的历史行为表示 (兴趣激活向量)
    """
    def __init__(self, embed_dim, hidden_units=None):
        super().__init__()
        if hidden_units is None:
            hidden_units = [embed_dim * 2, embed_dim]
        input_dim = embed_dim * 4
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_units:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, hist_emb, target_emb, hist_mask):
        """
        hist_emb: (B, S, D)
        target_emb: (B, D)
        hist_mask: (B, S) - 1 for valid, 0 for padding
        """
        target_expanded = target_emb.unsqueeze(1).expand_as(hist_emb)

        diff = hist_emb - target_expanded
        prod = hist_emb * target_expanded

        concat = torch.cat([hist_emb, target_expanded, diff, prod], dim=-1)

        attn_scores = self.mlp(concat).squeeze(-1)
        attn_scores = attn_scores.masked_fill(hist_mask == 0, -1e9)
        attn_weights = F.softmax(attn_scores, dim=-1)

        attn_weights = attn_weights.unsqueeze(-1)
        weighted_hist = (hist_emb * attn_weights).sum(dim=1)
        return weighted_hist


class DINModel(nn.Module):
    """
    DIN (Deep Interest Network) 精排模型

    核心思想:
    1. 用户的兴趣是多样的，不同目标 item 应该激活不同的历史行为
    2. 用注意力机制动态计算历史行为与目标 item 的相关性
    3. 输出: 点击概率 (0~1)

    相比双塔的优势:
    1. 双塔用户和 item 完全独立编码，无法捕获交叉特征
    2. DIN 让用户表示随目标 item 动态变化，更精准
    3. 适合精排阶段，对少量候选 item 精细打分
    """
    def __init__(self, num_users, num_items, num_categories, embed_dim=64,
                 hidden_dims=None, hist_len=50, dropout=0.1):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
        self.user_embedding = nn.Embedding(num_users, embed_dim, padding_idx=0)
        self.item_embedding = nn.Embedding(num_items, embed_dim, padding_idx=0)
        self.category_embedding = nn.Embedding(num_categories, embed_dim, padding_idx=0)
        self.hist_len = hist_len
        self.embed_dim = embed_dim

        self.attention_layer = AttentionLayer(embed_dim)

        feature_dim = embed_dim * 5 + 4

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

    def forward(self, batch):
        user_emb = self.user_embedding(batch['user_id'])
        item_emb = self.item_embedding(batch['item_id'])
        cat_emb = self.category_embedding(batch['category_id'])

        hist_emb = self.item_embedding(batch['hist_items'])
        hist_mask = (batch['hist_items'] != 0).float()

        interest_activation = self.attention_layer(hist_emb, item_emb, hist_mask)

        click_count = batch['click_count'].float().unsqueeze(-1)
        time_span = batch['time_span'].float().unsqueeze(-1)
        item_click_count = batch['item_click_count'].float().unsqueeze(-1)
        created_at_ts = batch['created_at_ts'].float().unsqueeze(-1)

        concat_features = torch.cat([
            user_emb, item_emb, cat_emb, interest_activation,
            user_emb * item_emb,
            click_count, time_span, item_click_count, created_at_ts
        ], dim=-1)

        logit = self.mlp(concat_features).squeeze(-1)
        return logit

    def compute_loss(self, logits, labels, click_weight=None):
        if click_weight is not None:
            return F.binary_cross_entropy_with_logits(
                logits, labels.float(), weight=click_weight)
        return F.binary_cross_entropy_with_logits(logits, labels.float())

    def predict(self, batch):
        logit = self.forward(batch)
        return torch.sigmoid(logit)
