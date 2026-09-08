# SASRec → HSTU 召回升级方案

> 状态：方案定稿，未实施（不改代码，仅存档）
> 日期：2026-08-25

## 0. 一句话方案

把 `SASRecUserTower` 里的 `SASRecBlock`（causal self-attention）替换成新增的 `HSTUBlock`（pointwise 交互 + 因果 cumsum），新增 `HSTUUserTower`，在 `TwoTowerV2Model` 加 `user_tower_type` 开关。**双塔结构、共享 embedding、InfoNCE loss、MLP 头、评估逻辑全部不改。**

## 1. 核心原则：新增 + 开关，不删除

- **新增** `HSTUBlock`、`HSTUUserTower`
- **保留** `SASRecBlock`、`SASRecUserTower` 原样不动
- `TwoTowerV2Model.__init__` 加 `user_tower_type='sasrec'` 参数，`'hstu'` 时实例化 `HSTUUserTower`
- 两者 `forward` 签名一致，下游 `compute_infonce_loss` / `get_user_embedding` / `get_item_embedding` 无需改动
- 目的：A/B 对比，随时可切回 SASRec 基线

## 2. 改动边界

| 是否改动 | 内容 |
|---|---|
| ✅ | `model.py` 新增 `HSTUBlock`、`HSTUUserTower` |
| ✅ | `TwoTowerV2Model.__init__` 加 `user_tower_type` 开关 |
| ✅ | `config.py` 加 `USER_TOWER_TYPE` |
| ⚠️ | `train_v2.py` 的 `model_cfg` 加一行 `user_tower_type` |
| ❌ | `data_loader.py`、`evaluate.py`、`collate_fn`、`compute_infonce_loss`、`ItemTower`、ALS 初始化、checkpoint 逻辑 |

## 3. 核心公式

```
Attn(X)_i = Σ_{j≤i}  φ(X_i W_u) ⊙ (X_j W_v)      ← pointwise attention（因果）
HSTU(X)   = X + f_2(  Attn(X) ⊙ f_1(X)  )          ← 自门控 + 残差
```

**因果聚合的等价实现**（因逐元素乘法对加法满足分配律，U_i 不依赖 j）：

```
Attn(X)_i = φ(X_i W_u) ⊙ ( Σ_{j≤i} X_j W_v ) = U_i ⊙ cumsum(V)_i
```

→ 不需要 causal mask、不需要 softmax，`torch.cumsum` 天然保证因果，O(T·d)。

## 4. 目标代码：HSTUBlock

```python
class HSTUBlock(nn.Module):
    """HSTU (Hierarchical Sequential Transduction Unit)
    Meta 2024: Actions Speak Louder than Words
    用 pointwise (Hadamard) 交互替代 causal self-attention 的 softmax 标量权重
    """
    def __init__(self, embed_dim, dropout=0.1):
        super().__init__()
        self.w_u = nn.Linear(embed_dim, embed_dim)   # query 侧投影 (过 SiLU)
        self.w_v = nn.Linear(embed_dim, embed_dim)   # key/value 侧投影
        self.f1 = nn.Sequential(                     # 自门控 pointwise MLP
            nn.Linear(embed_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.f2 = nn.Sequential(                     # 输出 pointwise MLP
            nn.Linear(embed_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )
        self.layernorm = nn.LayerNorm(embed_dim)

    def forward(self, x, padding_mask=None):
        residual = x
        x = self.layernorm(x)                        # pre-norm

        v = self.w_v(x)                              # V = x W_v
        if padding_mask is not None:                 # padding 清零，防 cumsum 累积无效位置
            v = v * (~padding_mask).unsqueeze(-1).float()

        cum_v = torch.cumsum(v, dim=1)               # 因果聚合
        u = F.silu(self.w_u(x))                      # U = φ(x W_u)
        attn = u * cum_v                             # pointwise attention

        gated = attn * self.f1(x)                    # 自门控
        return residual + self.f2(gated)
```

## 5. 目标代码：HSTUUserTower

复用 `SASRecUserTower` 骨架，三处改动：删 causal_mask、换 block、padding_mask 语义调整。

```python
class HSTUUserTower(nn.Module):
    def __init__(self, num_users, num_items, embed_dim, hidden_dims, hist_len=50,
                 num_blocks=2, dropout=0.1, shared_item_embedding=None):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embed_dim, padding_idx=0)
        self.item_embedding = (shared_item_embedding if shared_item_embedding is not None
                               else nn.Embedding(num_items, embed_dim, padding_idx=0))
        self.position_embedding = nn.Embedding(hist_len, embed_dim)   # 保留位置编码
        self.hist_len = hist_len
        self.embed_dim = embed_dim

        self.hstu_blocks = nn.ModuleList([
            HSTUBlock(embed_dim, dropout) for _ in range(num_blocks)
        ])
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(embed_dim)

        # 以下 MLP 头与 SASRecUserTower 完全一致
        input_dim = embed_dim * 2 + 4
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
        # 注意：不再有 self.causal_mask

    def forward(self, user_id, hist_items, hist_len, click_count, time_span,
                user_avg_rating=None, user_std_rating=None):
        user_emb = self.user_embedding(user_id)
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

        # 以下与 SASRecUserTower 完全一致
        B, D = user_emb.shape
        _zeros = lambda: torch.zeros(B, 1, device=user_emb.device)
        click_count_norm = _zeros() if click_count is None else click_count.float().unsqueeze(-1)
        time_span_norm = _zeros() if time_span is None else time_span.float().unsqueeze(-1)
        user_avg_rating_norm = _zeros() if user_avg_rating is None else user_avg_rating.float().unsqueeze(-1)
        user_std_rating_norm = _zeros() if user_std_rating is None else user_std_rating.float().unsqueeze(-1)
        x = torch.cat([user_emb, hist_vec, click_count_norm, time_span_norm,
                       user_avg_rating_norm, user_std_rating_norm], dim=-1)
        user_vec = self.mlp(x)
        user_vec = F.normalize(user_vec, p=2, dim=-1)
        return user_vec
```

## 6. TwoTowerV2Model 接入（开关式）

```python
def __init__(self, ..., user_tower_type='sasrec'):   # 新增参数
    ...
    self.item_tower = ItemTower(...)                 # 不变
    if user_tower_type == 'hstu':
        self.user_tower = HSTUUserTower(
            num_users, num_items, embed_dim, hidden_dims, hist_len,
            num_blocks, dropout,
            shared_item_embedding=self.item_tower.item_embedding)
    else:
        self.user_tower = SASRecUserTower(           # 原版保留
            num_users, num_items, embed_dim, hidden_dims, hist_len,
            num_heads, num_blocks, dropout,
            shared_item_embedding=self.item_tower.item_embedding)
    ...
```

## 7. config.py + train_v2.py 新增

```python
# config.py
USER_TOWER_TYPE = 'hstu'      # 'sasrec' | 'hstu'
```

```python
# train_v2.py 的 model_cfg 加一行
'user_tower_type': getattr(config, 'USER_TOWER_TYPE', 'sasrec'),
```

## 8. 三个关键实现细节

1. **因果性**：SASRec 靠 `torch.triu` causal mask；HSTU 靠 `cumsum` 天然只累积 `j≤i`，不显式传 mask。
2. **padding**：右侧 padding，cumsum 前 `v = v * (~padding_mask)` 清零 padding 位置；最终仍用 `hist_len` 实际长度归一化。
3. **位置信息**：pointwise 交互位置无偏，保留 `position_embedding`（绝对位置编码）；v2 可加 relative positional bias。

## 9. 风险与注意事项

| 风险 | 说明 | 对策 |
|---|---|---|
| cumsum 数值放大 | 累加 50 向量无归一化 | pre-norm + 最终 `F.normalize` 兜底 |
| 梯度过 cumsum | 线性算子，无爆炸风险 | 比 softmax 更稳 |
| 参数同量级 | `w_u+w_v+f1+f2` ≈ `QKV+FFN` | 模型规模不变 |
| 需重训 | state_dict 键不同，不能复用 SASRec checkpoint | 开关分存，A/B 各训 |
| 收益不确定 | T=50 不算长，速度非主卖点 | 主卖点是表达力 + 前沿接轨 |

## 10. 验证路径

1. `USER_TOWER_TYPE='sasrec'` 复现基线（NDCG 不变）
2. 切 `'hstu'`，`offline=True` + `AMAZON_SAMPLE_USERS=5000` 验证跑通、loss 下降
3. 全量 A/B，对比 Recall@K / NDCG@K
4. 若有提升，接入面试叙事："召回 HSTU（序列）+ 精排 RankMixer（异构 token）各取所长"