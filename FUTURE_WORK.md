# 后续优化方向

当前已实现: 评估体系 + ALS 预训练 + Sampled 评估 + 4 层 MLP + 五路召回合

---

## 1. 彻底消除双塔 Embedding 表未共享 (Critical Fix)

**问题**: UserTower/SASRecUserTower 和 ItemTower 各自有一张 item_embedding 表, 同一个商品在两个空间里不同向量。

**Fix**: 在 `__init__` 里先把 ItemTower 创建好, 然后把它的 `item_embedding` 传给 UserTower → 共享引用。

**动机**: 当前 InfoNCE 每步只依赖 in-batch 的 2047 个负样本。虽然在 batch 内多样性足够，但 in-batch 负样本天然偏向热门商品（热门商品出现频率高），训练梯度偏向"热门 vs 热门"的区分，对冷门商品不友好。

**方案**: 每步在 batch 外额外随机采样 N 个负样本，拼入 softmax 分母：

```python
# 当前 (纯 in-batch)
sim = user_vec @ all_in_batch_item_vecs.T / τ    # (B, B)
loss = CrossEntropy(sim, labels)                  # B 个正样本 vs B-1 个 in-batch 负

# Sampled Softmax
extra_negs = random_sample(N)                     # N 个 batch 外负样本
extra_vecs = item_tower(extra_negs)               # (N, D)
sim_inbatch = user_vec @ batch_item_vecs.T / τ    # (B, B)
sim_extra   = user_vec @ extra_vecs.T / τ         # (B, N)
sim = cat([sim_inbatch, sim_extra], dim=1)        # (B, B+N)
loss = CrossEntropy(sim, labels)
```

**预期效果**:
| 维度 | 纯 in-batch | + Sampled Softmax |
|------|-----------|-------------------|
| 每步负样本数 | 2047 | 2047 + N（建议 N=1024–2048） |
| 热门偏差 | 中等（in-batch 内含热门） | 低（随机采样稀释热门） |
| 显存增量 | — | +N×D×4 bytes ≈ +4MB (N=1024, D=256) |
| 训练速度 | — | 慢 5–10%（额外 encode N 个负样本） |
| NDCG 提升 | — | 预估 +5–10% |

**风险**: 如果在 ALS 预训练前开，额外负样本全是"用户没反馈的空白区域"，模型更难收敛。建议 ALS 跑通后再尝试。

**涉及文件**: `model.py:TwoTowerV2Model.compute_infonce_loss()`, `train_v2.py: training loop`, `config.py`

---

## 2. 文本 Embedding 替代/增强 Item ID Embedding

**动机**: Amazon raw_meta JSONL 里有商品标题 (title)、描述 (description)、特性 (features)。用 sentence-transformers 把标题 encode 成 384 维语义向量，替代或拼接 item_id Embedding，解决冷启动 + 语义召回。

**方案**:
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
title_vec = model.encode("Wireless Bluetooth Headphones")  # → 384维

# 叠加方案:
item_vec = concat([als_collaborative_256d, text_semantic_384d])  # → 640维 → MLP → 256维
```

**数据**: `McAuley-Lab/Amazon-Reviews-2023/raw/meta_Books.jsonl.gz` (包含 title/description/features/price/brand)
- 大小: 全量 ~50GB，按 parent_asin 提取后 ~500MB
- 编码耗时: ~2–4h (436K 条 × MiniLM, GPU)
- 新增依赖: `sentence-transformers`

**涉及文件**: 新建 `code/text_encoder.py`, 修改 `data_loader.py` (load_meta_jsonl), `model.py` (ItemTower text_embedding), `config.py`

---

## 3. LightGCN 替代 ALS

**动机**: ALS 是 2013 年的方法，LightGCN (SIGIR 2020) 是当前协同过滤 SOTA，在稀疏数据上 NDCG 比 ALS 高 5–15%。

**方案**:
- 3 层图卷积（不含特征变换和非线性激活）
- 最终 embedding = 各层输出的加权和
- PyTorch 直接实现 (~200 行)，训练时间 ~10 分钟 (46K×436K)

**涉及文件**: 新建 `code/lightgcn.py`, 修改 `code/als_init.py` (抽象接口)

---

## 4. Layer-wise LR Decay

**动机**: SASRec 有两层 Transformer block，底层学通用序列模式、顶层学任务相关特征。用统一学习率会导致底层欠拟合或顶层过拟合。

**方案**:
```python
# 底层 Transformer blocks: lr_base
# 顶层 MLP: lr_base * 0.5
# Embedding 层: lr_base * 0.1 (embedding 更新太大会破坏预训练语义)

param_groups = [
    {'params': model.user_tower.item_embedding.parameters(),       'lr': lr * 0.1},
    {'params': model.user_tower.sasrec_blocks[0].parameters(),     'lr': lr * 1.0},
    {'params': model.user_tower.sasrec_blocks[1].parameters(),     'lr': lr * 0.5},
    {'params': model.user_tower.mlp.parameters(),                  'lr': lr * 0.5},
]
optimizer = optim.AdamW(param_groups, weight_decay=...)
```

**涉及文件**: `train_v2.py`, `train_deep.py`

---

## 5. Amazon raw_meta 精细品类层级

**动机**: 当前 `category_id` 只有 3 个值（顶层品类名）。Amazon 的 `raw_meta_<Category>` JSONL 里有精细品类树:
```
Office_Products > Office Supplies > Writing Instruments > Pens
```

**方案**:
1. 下载 `raw_meta_<Category>.jsonl` 到 `amazon_reviews/`
2. 解析 `categories` 字段: `[["Office Supplies", "Writing Instruments", "Pens"]]`
3. 取倒数第二级作为 `sub_category_id`（如 "Writing Instruments"）
4. 新增 `sub_category_embedding` 到 ItemTower/DIN，额外 64 维
5. category_id 作为粗粒度 Embedding，sub_category_id 作为细粒度 Embedding

**规模**: `raw_meta_Office_Products.jsonl` 约 500MB，解析后可只保留 parent_asin + categories 两列 (~10MB)。

**涉及文件**: `data_loader.py` (新增 `load_amazon_raw_meta()`), `model.py` (ItemTower/DIN 新增 embedding), `config.py`

---

## 6. MIND 多兴趣召回接入

**动机**: `model.py` 里有完整的 MINDModel 实现，但 `inference_full.py` 只用 V2 做单向量 recall。MIND 的 K=3 兴趣胶囊可以独立检索后合并，提升多样性。

**方案**:
1. 写 `train_mind.py` 训练脚本
2. `inference_full.py` 新增 MIND recall channel
3. 推理时: 3 个兴趣向量分别做 dot product → 各取 Top-N → 合并去重

**涉及文件**: `model.py` (已有), 新建 `train_mind.py`, `recall_fusion.py`, `inference_full.py`

---

## 7. DIN 推理批量化

**动机**: 当前 `inference_full.py` 对每条候选逐一 `din_model.predict()`，50 候选 = 50 次 GPU kernel launch。

**方案**: 将全量 50 条候选一次性打包成 batch，单次 forward 输出 50 个概率。

**涉及文件**: `inference_full.py` (第 195-227 行, DIN 评分循环)

---

## 8. 召回权重自动搜索

**动机**: 四路召回权重 `[1.0, 1.0, 0.5, 0.1]` 是硬编码的。换品类组合后最优权重会变化。

**方案**:
```python
from scipy.optimize import minimize

def objective(weights):
    merged = merge_recall_results(channels, weights.tolist())
    hr = evaluate_merged(merged, val_data)
    return -hr  # minimize negative HR

result = minimize(objective, x0=[1.0, 1.0, 0.5, 0.1],
                   bounds=[(0, 3)] * 4, method='L-BFGS-B')
```

**涉及文件**: 新建 `code/tune_recall_weights.py`
