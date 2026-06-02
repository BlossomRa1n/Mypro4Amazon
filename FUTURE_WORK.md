# 后续优化方向

当前已实现: 评估体系 (时序验证 + Early Stop + 每 epoch 存档) + Rating 强度信号 + AMP + 梯度累积。

---

## 1. Layer-wise LR Decay

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

## 2. Amazon raw_meta 精细品类层级

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

## 3. MIND 多兴趣召回接入

**动机**: `model.py` 里有完整的 MINDModel 实现，但 `inference_full.py` 只用 V2 做单向量 recall。MIND 的 K=3 兴趣胶囊可以独立检索后合并，提升多样性。

**方案**:
1. 写 `train_mind.py` 训练脚本
2. `inference_full.py` 新增 MIND recall channel
3. 推理时: 3 个兴趣向量分别做 dot product → 各取 Top-N → 合并去重

**涉及文件**: `model.py` (已有), 新建 `train_mind.py`, `recall_fusion.py`, `inference_full.py`

---

## 4. DIN 推理批量化

**动机**: 当前 `inference_full.py` 对每条候选逐一 `din_model.predict()`，50 候选 = 50 次 GPU kernel launch。

**方案**: 将全量 50 条候选一次性打包成 batch，单次 forward 输出 50 个概率。

**涉及文件**: `inference_full.py` (第 195-227 行, DIN 评分循环)

---

## 5. 召回权重自动搜索

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
