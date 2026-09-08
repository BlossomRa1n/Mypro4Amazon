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

---

# 2026-06-06 问题记录 (对话中你问我的问题)

1. "速度这么快吗？" — 看到 5 batches/epoch，觉得太快了
2. "哦对，用户数量特别少。为什么5-core只剩240用户了？帮我分析一下ALL beauty的数据。" — 分析了数据漏斗
3. "DIN的论文是怎么处理数据的？" — 查论文，对比你的做法
4. "DIN的预处理方式和我项目完全一致吗？如果这样的话，我也换成Books" — 切品类
5. "Books.jsonl怎么有20G啊。能不能直接下载到数据盘去，不然我系统盘肯定爆了。autodl-tmp是我的数据盘。" — 数据盘路径
6. "DIN 论文：…那他们也是把rating小于4的都直接过滤舍弃吗？" — 预处理细节确认
7. "现在我的精排模型怎么做的？用的什么数据，怎么做的特征，之后怎么处理？总结一下。" — 要求总结全链路
8. "帮我分析一下" — 诊断 BPR 坍缩
9. "把epoch5的指标记录下来，先写进日志里。" — 记录 Video_Games 首次训练结果
10. "GC是什么？" — 垃圾回收的概念
11. "服务器IP怎么看？" — hostname -I
12. "我不知道时文本还是文件啊。" — 怎么把结果给我看
13. "我有autodl的SSH隧道工具。" — SSH 方式说明
14. "怎么确认。告诉我命令，我敲到服务器上去" — 确认文件存在
15. "好像内存爆了。我看实例监控的内存占用100%了。" — OOM 诊断
16. "不行。你直接帮我把路径改掉吧。" — DATA_PATH 问题
17. "直接把config改了呗" — 让 config 默认写 autodl-tmp
18. "下载换源" — huggingface 换镜像
19. "对了吗？" — 确认 Video_Games JSONL 文件大小
20. "继续" / "继续说下去" / "接着改" — 催促继续修改
21. "你修复了什么？" — 问 checkpoint 兼容性修复的内容
22. "ItemCF Hard Negative 到底有没有用？你之前说有用现在又说害了召回？" — 澄清两个概念
23. "DIN的论文是怎么处理的？"（第二次）— 深入查论文训练策略
24. "先这么改吧。" — 确认加 Hard Negative Mining

## 路径问题反复折腾

打包上传到服务器后 config.py 里的路径被覆盖回默认值，导致反复报 FileNotFoundError。
最终解决方案: config.py 里 DATA_PATH 硬编码为 `/root/autodl-tmp/amazon_data`（服务器数据盘），
本地开发时手动改回默认路径。USER_DATA_PATH 保持默认 `/root/user_data`（存模型/checkpoint/encoder）。

## Books 20GB 太大数据量

Books raw JSONL 20GB，加载时 `all_rows` list-of-dicts + DataFrame 同时占用内存导致 32GB 服务器 OOM。
已优化 `load_raw_reviews` 和 `load_raw_meta` 从逐行 dict 堆积改为类型化列数组，峰值内存减半。
Video_Games (2.5GB) 作为中间规模更适合当前实验。

## All_Beauty 5-core 仅 240 用户

美妆品类人均评价 1.1 次，5-core 过滤后只剩 240 用户，无法训练。
DIN 论文用的是 Electronics/Books（人均 8-12 次），5-core 后用户量正常。
已切换到 Video_Games（人均 ~8 条，5-core 后 ~92K 用户）。

## BPR Loss 坍缩 / 随机负样本太容易

Video_Games 首次训练: Loss 从 epoch 2 开始 ~0，Val AUC 剧烈震荡 (0.50→0.87→0.55)，Pos Mean 持续下降。
根因是 23K 商品中随机采样的负样本与正样本无相似性，模型学到"正样本=历史见过"这个平凡规则。
已加入 ItemCF Hard Negative Mining 作为修复。

## DIN vs DIEN 负采样差异

- DIN (KDD 2018): 直接用展示未点击做负样本，无特殊负采样策略
- DIEN (AAAI 2019): 引入辅助 loss + 随机负采样，用于兴趣演化建模
- 你的项目: BPR pairwise + 显式负样本(1-2分) + ItemCF Hard Negative

## ItemCF 作为召回 vs 作为 Hard Negative 的不同角色

- ItemCF 作为召回通道: 效果差 (HR@5=1.3%, 比热门高 19%)
- ItemCF 作为训练负样本: 有效 (旧 DIN BPR+HardNeg 把 AUC 从 0.50 提到 0.66)
两者是不同用途，不矛盾。

## 旧 checkpoint 不兼容问题

切换品类后 num_users 变化，加载旧 ckpt 会 CUDA index out of bounds。
修复: train_din_ext.py 加入了 num_users 不匹配检测 → 自动重建模型和 optimizer。
run_ext_din.sh 启动时自动清理旧 ckpt + encoder 缓存。

## string dtype 导致 sklearn LabelEncoder NA TypeError

Pandas `string` dtype 的 `<NA>` 值在 `unique()` 时混入 `NAType`，导致 LabelEncoder 报 "Encoders require uniformly strings or numbers"。
修复: brand 和 category 的 unique() 之前过滤 `str(c) != '<NA>'`。

---

# 2026-06-07 日志 (代码包 MD5: 5CD35FBE0E6779982B9F7791955B6C66)

## 统一 encoder: build_extended_encoders 全局替换 build_encoders

全链路三大脚本 (`train_v2.py`, `train_din_ext.py`, `inference_full.py`) 统一使用 `build_extended_encoders` + `id_encoders_ext.pkl`，
包含 `brand_id` 字段。旧的 `build_encoders` / `id_encoders.pkl` 不再作为主流程使用。

## inference_full.py 全面切换到 DINExtendedModel

推理脚本从旧的 `DINModel` 切换到 `DINExtendedModel` (含 brand_embedding/verified_embedding/DINAttentionLayer)。
数据源从 rating-only CSV 切换为 raw JSONL (`load_raw_reviews` + `raw_meta`)。
新增 `_build_user_batch` 支持扩展序列字段: hist_brands, hist_ratings, hist_time_deltas, hist_verified。
新增 `din_rerank_batch` 支持扩展 item feature: brand_id, item_avg_rating, item_rating_number。
加载模型时从 `DIN_EXT_BEST_FILE` / `DIN_EXT_MODEL_FILE` 读取，返回 model + model_limits 字典用于索引边界裁剪。

## SASRecUserTower MLP input_dim 维度修复

`SASRecUserTower.__init__` 的 `input_dim = embed_dim * 2 + 4` 与 forward 中 cat 的 6 个张量（user_emb + hist_vec + 4 stats = 516维）一致。
修复了旧 checkpoint 的 `nn.Linear(512, 514)` vs 当前代码 `nn.Linear(516, 512)` 的不兼容问题。
注意: 删除所有旧 `two_tower_v2_*.pth` + `id_encoders*.pkl` 后重新训练。

## recall_fusion.py 双塔 V1 硬注释跳过

V1 (BPR) 召回通道在推理时硬注释跳过，仅保留 V2 (SASRec + InfoNCE)。embedding_recall 中加入 user/item 索引边界检查，
超出模型 num_users/num_items 的用户自动过滤（避免 CUDA index out of bounds → CUBLAS execution failed）。

## AMP GradScaler NaN loss 处理

`train_v2.py` 中 `torch.isnan(loss)` 跳过时增加 `scaler.update()` 调用，否则连续 NaN 后 `scaler.step()` 报 "No inf checks were recorded"。
优化器 `zero_grad()` 移到 autocast 之前，避免 repeated backward 累积。

## 数据自动下载 & 编码器自动重建

`data_loader_ext.py`:
- 新增 `_auto_download_raw()` 函数：raw JSONL 文件缺失时自动调用 `download_raw.py`（HF Mirror 国内镜像）
- `build_extended_encoders()`: 加载缓存时用 set intersection 验证用户/物品全集覆盖，stale 时自动重建

## train_din_ext.py 改为 DINExtendedModel + ALS 初始化

精排训练脚本从旧 `DINModel` 全面切换为 `DINExtendedModel` (BPR pairwise, brand/verified/helpful/price/item_quality)。
新增 ALS item_embedding 预训练初始化 + CosineAnnealingLR + 评估指标完整导出 (`din_ext_final_results.json`)。

## config 调整

- `DIN_NUM_EPOCHS`: 10 → 15
- `DIN_LEARNING_RATE`: 1e-3 → 5e-4
- `EXT_CATEGORIES`: 从 `All_Beauty` 改为 `Video_Games`（与 `AMAZON_CATEGORIES` 保持一致）
- `RECALL_WEIGHTS.v2_sasrec`: 0.0 → 1.0, 推理时 V2 双塔正常跑

## 全局 OpenBLAS 警告修复

所有使用 DataLoader 多线程的脚本顶部加入 `os.environ['OPENBLAS_NUM_THREADS'] = '1'` 等设置，
防止 numpy 内部 BLAS 线程与 DataLoader 线程冲突。

## 日志标准化

- `train_v2.py`: 打印 "Two-Tower V2 training complete!" + 超参数汇总
- `train_din_ext.py`: eval protocol 注明 "leave-last-out, 50 random negatives (SASRec-style)"
- 两个脚本结束时输出 best epoch/metrics 并保存 `*_final_results.json` 到 `prediction_result/`

## 全局注释清理

所有文件中特定品类名 (`All_Beauty`) 改为泛化描述，数据源由 `config.AMAZON_CATEGORIES` / `EXT_CATEGORIES` 决定。
`model_ext.py` / `data_loader_ext.py` / `download_raw.py` 的默认品类改为 `Video_Games`。

---

# 2026-06-07 晚间日志：数据管道统一 + 召回权重调整 + 评估协议切换 (MD5: 最终封存)

## 核心改动：CSV 5-core + JSONL brand/verified merge

**问题**: `train_v2.py` 用的是 CSV (`get_all_click_df`), `train_din_ext.py` 和 `inference_full.py` 用的是 raw JSONL (`load_raw_reviews` + `prepare_click_df`)。两者用户量天差地别 (CSV 94K vs JSONL 6.5K)，导致训练-推理 encoder 不一致，V2 user embedding 完全失效。

**方案**: 全链路统一到 **CSV 5-core (94K 用户, 25K 物品) + raw_meta JSONL (brand/quality 特征) + JSONL extra fields MERGE**。

**实现**:
1. `data_loader_ext.py`
   - `build_extended_user_features` 兼容 CSV 缺失列 (`verified_purchase`, `helpful_vote`, `rating` — 填 0)
   - 新增 `merge_jsonl_features()`: CSV LEFT JOIN JSONL 补充 `verified_purchase` + `helpful_vote` (11.6% 命中率)
   
2. `train_v2.py`
   - 恢复 `get_all_click_df` (CSV) 数据加载
   - 新增 `merge_jsonl_features` 调用

3. `train_din_ext.py`
   - 恢复 `get_all_click_df` (CSV) + `merge_jsonl_features`
   - 移除 `load_raw_reviews` + `prepare_click_df` 依赖

4. `inference_full.py`
   - 恢复 CSV 数据加载
   - `user_features` 用训练窗口内数据构建 (防止验证目标泄露)
   - `item_topk_click` 用全量 click_df (热门不泄露验证目标)

5. `evaluate.py`
   - 切换到 CSV 数据管道
   - **新增 Sampled 评估协议** (SASRec paper standard):
     Protocol 1 (Full-rank): ground truth set → Hit/NDCG
     Protocol 2 (Sampled): 加载 DIN-Ext 模型 → leave-last-out + random negatives → AUC
     直接调用 `train_din_ext.evaluate_ext` 保持一致性

## 召回权重调整

`config.py`:
- `RECALL_NUM`: 50 → 100
- `RECALL_WEIGHTS.itemcf`: 1.0 → 1.2
- `RECALL_WEIGHTS.v2_sasrec`: 1.0 → 2.2 (双塔 → 主力通道)

## 最终评估指标 (Video_Games, 94K 用户, 137K 物品)

| 协议 | 指标 | 值 | 说明 |
|------|------|-----|------|
| V2 Sampled (1正 vs 100随机负) | NDCG@20 | 0.50 | 双塔召回排序能力 |
| DIN-Ext Pairwise (1正 vs 50随机负) | AUC | 0.75 | 精排模型区分能力 (论文标准) |
| Full-rank (137K物品) | HR@5 | 1.7% | 极难任务 — 大海捞针 |
| Popularity (Full-rank) | HR@5 | 2.7% | 个性化模型仍未跨越门槛 |
| Verified match rate | — | 11.6% | CSV-JSONL 版本不一致导致 |

## 已知局限性

1. **Full-rank HR@5 低**: 137K 物品选 5 个命中用户未来购买。学术论文不用 full-rank 协议。
2. **Verified 匹配率仅 11.6%**: CSV 5-core 和 JSONL raw 可能是不同时间切分版本。
3. **Video_Games 平均用户交互 5.9 次**: 序列模型需要更密集的品类 (Books/Electronics)。
4. **Recall 是瓶颈**: V2 NDCG@20=0.50 说明排序能力存在，但 recall pool 里可能缺乏 ground truth。
