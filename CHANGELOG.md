# 2026-06-04 改动日志

## 1. DIN BCE → BPR Pairwise Loss

**问题**: BCE loss 下 80% 负样本迫使模型最优策略 = "输出 ~0.05"，所有修复（ALS 预训练、ItemCF Hard Negative、pos_weight=4.0）均无法阻止坍缩。Epoch 1 瞬间 AUC=0.57，epoch 3 后跌回 0.50。

**方案**: DINModel 新增 `forward_pairwise()` + `compute_bpr_loss()`，训练时拆 user/pos/neg → 分别打分 → BPR loss = `-log(sigmoid(pos - neg))`。天然免疫样本不平衡。

**效果**: AUC 0.50→**0.66** (best epoch 8)，Pos Mean 从 0.05→0.96（不再坍缩）。

**涉及文件**: `model.py:DINModel`, `data_loader.py:DINDataset`, `train_din.py`

---

## 2. 五路召回：V2 SASRec + V1 BPR 双模型并行

**问题**: `embedding_recall` 只加载一个模型（V2 best → V1 fallback），训练好的 BPR 双塔 (NDCG=0.44 Sampled) 被浪费。

**方案**: `multi_channel_recall` 分别加载 V2 和 V1 两个模型，各做一次独立向量召回，合并为两个独立通道。V1 从 `item_embeddings.pkl` 加载预计算向量。

**效果**: 五路召回 — ItemCF + V2 SASRec + V1 BPR + 品类偏好 + 热门兜底

**涉及文件**: `recall_fusion.py:embedding_recall`（加 model_path/channel_label 参数）, `recall_fusion.py:multi_channel_recall`

---

## 3. 召回权重重新分配

**问题**: 旧权重 `ItemCF=5.0, embedding=0.3` 只适配了 V2 NDCG=0.001 的时期，V2 已经活了（NDCG=0.50）但权重仍是 0.3。

**新权重**: V2 SASRec=5.5, ItemCF=5.0, V1 BPR=2.0, Category=0.5, Hot=0.1

**涉及文件**: `config.py:RECALL_WEIGHTS`, `recall_fusion.py:multi_channel_recall`

---

## 4. DIN Hard Negative Mining（ItemCF 负样本替代随机负样本）

**问题**: DINDataset 用 `np.random.randint` 抽取负样本，纯随机冷门商品与正样本差异太大，模型学到的是粗粒度区分，不是精排需要的细粒度区分。BCE 下 80% 负样本 → 模型坍缩为"一律说不买"，AUC=0.50，Pos Mean→0.05。注意：Hard Negative 本身是对的，但配合 BCE 仍然无效（见第 1 项），改为 BPR 后才生效。

**方案**: 加载 ItemCF 相似度矩阵 → `build_hard_negative_index` → DINDataset 对每个用户历史物品查 Top-4 相似邻居（排除已交互）作为困难负样本。`heapq.nlargest` 取 Top-K 替代 `sorted()` 全量排序，`raw_to_enc` dict 替代 `item_le.transform()` 逐条扫描。

**涉及文件**: `data_loader.py:DINDataset`, `build_hard_negative_index`, `train_din.py`

---

## 5. ALS 预训练自适应兼容（DIN + 双塔）

**问题**: `init_model_with_als()` 硬编码了 `model.item_tower.item_embedding`，DINModel 没有 `item_tower`（直接是 `model.item_embedding`），导致 ALS 无法初始化 DIN → DIN 仍然从零随机开始 → AUC=0.5458 远低于预期。

**方案**: `_get_item_embed_param()` 自适应检测模型结构（双塔查 `item_tower.item_embedding`，DIN 查 `item_embedding`），`train_din.py` 加入 ALS 初始化调用。

**涉及文件**: `als_init.py`, `train_din.py`

---

## 6. 推理速度优化（DIN batch inference）

**问题**: `din_rerank_batch` 每用户用 Python list comprehension 逐候选取物品特征 `[item_cat_arr[i] for i in item_indices]`，213K 用户 × 50 候选 = 1000 万次 Python 循环 → 推理 5 小时。

**方案**: 改为 numpy 直接索引切片 `item_cat_arr[idx_arr]`，20 倍提速。

**涉及文件**: `inference_full.py:din_rerank_batch`

---

## 7. 品类切换：Books → All_Beauty + Video_Games + CDs_and_Vinyl

**动机**: 三品类混合还原 SASRec 论文的 Amazon Beauty + Games 设定，同时激活品类 Embedding（3 个值有区分度）和品类召回通道。

**配置**: `AMAZON_CATEGORIES = ['All_Beauty', 'Video_Games', 'CDs_and_Vinyl']`, `AMAZON_MIN_USER_INTERACTIONS = 0`（Sampled 评估已对齐论文，不需要稠密过滤），品类召回权重恢复。

**涉及文件**: `config.py`

---

## 8. 评估方式切换：Full Rank → Sampled Metrics（与 SASRec 论文对齐）

**问题**: 全量排名（436K 商品）下噪声天花板 ~0.44，任何 Embedding 方法的 NDCG 都被压到 0.01 以下，无法与学术论文对标。

**方案**: 新增 `evaluate_two_tower_sampled()` — 每用户 1 正样本 vs 100 随机负样本排名。`EVAL_FULL_RANK=False` 切换。

**涉及文件**: `evaluate.py`, `config.py`, `train_deep.py`, `train_v2.py`

---

## 9. ALS 预训练 Item Embedding（核心性能提升）

**问题**: 110K+ 商品的 Embedding 表从零随机初始化，稀疏交互的梯度信号不足以把数百万参数推到正确位置。

**方案**: 用 Alternating Least Squares (ALS) 在用户-物品交互矩阵上学习物品向量，L2 归一化后直接拷贝到双塔/DIN 的 `item_embedding.weight`。30–60 秒训练，缓存 `.pkl`。

**效果**: V2 NDCG 从 0.000 → 0.50 (Sampled)，BPR NDCG 从 0.001 → 0.44 (Sampled)。

**涉及文件**: `code/als_init.py`（新文件）, `config.py`, `train_deep.py`, `train_v2.py`, `train_din.py`

---

## 10. User Tower MLP 加深

**问题**: 原 MLP `[512, 256]` 只有 2 层，序列特征和统计特征的交互表达不足。

**方案**: HIDDEN_DIMS `[512, 384, 256, 128]` — 加深到 4 层。

**涉及文件**: `config.py`

---

## 11. Checkpoint 体积优化

**问题**: 每 epoch 存优化器状态（Adam 一阶矩 + 二阶矩），V2 一个 checkpoint ~1.5GB，3 个模型 × 20 epochs ≈ 100GB，50GB 数据盘撑爆。

**方案**: 每 epoch 仅权重（~500MB），latest.pth + best.pth 含优化器各存一份。`_save_checkpoint()` 新加 `save_optimizer` 参数，resume 优先 `latest.pth`。

**涉及文件**: `train_deep.py`, `train_v2.py`, `train_din.py`

---

## 12. 推理瓶颈优化（recall_fusion.py / inference_full.py）

**问题**: `embedding_recall` 内 213K 用户每用户扫一次 DataFrame → O(N×M) 查询。DIN 推理每候选逐个 `item_le.transform()` + `item_features.loc[]`。

**方案**: 预建 `user_feat_dict`（O(1) 查）、`idx_to_raw_item` list 下标、category recall 的 `iterrows()` → `groupby().size()` 向量化。

**涉及文件**: `recall_fusion.py`, `inference_full.py`

---

## 13. 数据稠密过滤

**问题**: Amazon 全量用户人均 7–10 条交互，Embedding 学不动。

**方案**: `AMAZON_MIN_USER_INTERACTIONS` 可选过滤。Books 品类 ≥25 留 46K 用户/人均 56 条。三品类切换后因 Sampled 评估已对齐，关闭过滤（设 0）。

**涉及文件**: `config.py`, `data_loader.py:load_amazon_reviews()`

---

## 14. vGPU-32GB 服务器适配

- `V2_BATCH_SIZE=2048`（利用 32GB 显存，2047 个 in-batch 负样本）
- `NUM_WORKERS=12`（16 vCPU 充分利用）
- 数据盘软链接 `/root/autodl-tmp/` → `/root/amazon_reviews/`, `/root/user_data/`

---

## 15. 其他批次修复（往期累积）

| 修复项 | 文件 |
|--------|------|
| UserTower / SASRecUserTower / MINDUserTower 共享 ItemTower 的 item_embedding | `model.py` |
| HR 计算 bug：`hr_total += 1` → `if hits: hr_total += 1` | `evaluate.py` |
| best checkpoint 选择指标从 HR → NDCG | `train_deep.py`, `train_v2.py` |
| InfoNCE 梯度累积移除（softmax 必须一次性计算） | `train_v2.py`, `config.py` |
| INFONCE_TEMPERATURE 从 0.2→0.07→0.1→**0.3** | `config.py` |
| DataLoader `num_workers=0` → `config.NUM_WORKERS` + `persistent_workers=True` | `train_*.py` |
| Dataset `__getitem__` 去 DataFrame 查询 → numpy 数组预建 | `data_loader.py` |
| `build_hard_negative_index` 性能修复：sorted→heapq, transform→字典 | `data_loader.py` |
| `train_din.py` 删除 bce `train_acc` 引用（改为 bpr 后残留） | `train_din.py` |
| ItemCF 基线测试脚本 | `code/test_itemcf.py` |
| Embedding 坍缩诊断工具 | `code/diagnose_emb.py` |


# 2026-06-05 改动日志

## 0. 精排特征体系设计与 Raw 数据扩展

**动机**: 此前精排 DIN 只用 rating_only CSV 的 4 列（user_id, parent_asin, rating, timestamp），缺少品牌、价格、评论文本、验证购买、全局质量评分等关键信号。需要从 Amazon Reviews 2023 的原始完整数据重新设计特征体系。

**方案 — 离散 + 稠密 + 序列三层特征架构**:

- **离散特征 (8-12 个)**: user_id, item_id, brand_id (meta.store), category_L1/L2/L3, verified_purchase (review.verified_purchase), skin_type 等 (meta.details)
- **稠密特征 (15-20 个)**:
  - 用户侧: 点击总数, 时间跨度, 活跃天数, 平均/标准差评分, 验证购买率, 平均 helpful_vote, 价格均值/中位数/极差, 兴趣熵
  - 商品侧: 全局评分 (meta.average_rating), 全局评分数 (meta.rating_number), 价格 (meta.price), 上架天数, 点击次数, 评论 helpful_vote 合计
  - 交叉特征: 用户均价 vs 候选价偏差, 品牌匹配, 类目重叠率, 候选评分 vs 用户平均评分偏差
- **序列特征 (8 条序列)**: 历史商品序列, 历史品牌序列, 历史类目序列, 历史评分序列, 历史时间间隔序列, 历史价格序列, 历史验证购买序列, 历史 helpful_vote 序列

**全量数据规模**: All_Beauty raw — 701,528 reviews, 112,590 items, 63 万用户, 30,765 brands

**新增模型 — DINExtendedModel**:
- `brand_embedding` (30,767 × 64d) + `verified_embedding` (2 × 4d)
- `DINAttentionLayer`: 融合 hist_item_emb + hist_brand_emb + hist_ratings + hist_deltas + hist_verified 五种信号的 Target Attention
- 最终 concat: discrete(836d) + sequence(256d) + dense_user(6d) + dense_item(4d) + interaction(256d) → MLP → sigmoid

**新增训练脚本 — train_din_ext.py**:
- `evaluate_ext()`: 每用户 1 正 vs 4 随机负 pairwise AUC
- 自动保留最近 3 个 epoch checkpoint（节省磁盘）、resume 支持、Early Stop

**打包**: `amazon_ext_code.tar.gz` (37KB, 无数据) / `archive/run05/amazon_ext.tar.gz` (130MB, 含 540MB raw JSONL)

**服务器运行状态** (2026-06-06 凌晨): 5-core 过滤后仅 240 用户（原因待查），Step 2 `build_extended_item_features` 纯 CPU 操作 (pandas merge/sort_index) 不占 GPU 显存 — GPU 到 Step 4-5 才会被吃满。进程 PID 3129 运行正常 (CPU 100%, 累计 18+ min), 未卡死。

**涉及文件**: `code/data_loader_ext.py`, `code/model_ext.py`, `code/train_din_ext.py`, `code/config.py`（EXT_* 新增配置）, `code/download_raw.py`, `run_ext_din.sh`, `code/test_ext_pipeline.py`

---

## 1. inference_full 恢复完整 pipeline（config 驱动）

**问题**: `recall_fusion.py:multi_channel_recall` 里的 weights 是硬编码的 `[1.0, 0.5, 0.1]`（只包含 ItemCF+Category+Hot），与 `config.py:RECALL_WEIGHTS` 脱节。

**方案**: 改为从 `config.RECALL_WEIGHTS` 读取五个通道的权重（itemcf, v2_sasrec, v1_bpr, category, hot），权重为 0 时自动关闭该通道。`v2_sasrec=0, v1_bpr=0` 表示当前关闭双塔，仅用 ItemCF + Category + Hot。

**涉及文件**: `inference_full.py`, `recall_fusion.py:multi_channel_recall`

---

## 2. 召回融合方式从加权求和改为配额制

**问题**: 原先的加权求和使得双塔 V2/V1 分数范围 (0.5–0.9) 在权重 5.5 后变成 2.8–5.0，远超 ItemCF 分数 (0.05–0.10 × 5.0 = 0.25–0.50)，导致 ItemCF 的精准确认被淹没。最终 HR@5 = 0.000。

**方案**: `merge_recall_results` 改为配额制合并——每路至少获得 3 个最低名额，按权重分配其余名额 → 各路的 Top-N 分别进入最终候选池。保证 ItemCF / V2 / V1 各自都有代表。

**涉及文件**: `recall_fusion.py:merge_recall_results`

---

## 3. ItemCF-only 消融推理

**动机**: 隔离验证 ItemCF + 品类偏好 + 热门兜底三路召回的真实效果，排除双塔和 DIN 的干扰。

**方案**: `inference_full.py` 跳过 embedding 加载和 DIN 精排，仅用 `itemcf_recall + category_preference_recall + hot_recall` 三路 → `merge_recall_results(weights=[1.0, 0.5, 0.1])` → 直接输出 Top-5。

**效果**: HR@5 = 0.0128 (1.3%)，NDCG@5 = 0.0063；热门兜底 HR@5 = 0.0108。ItemCF 比纯热门仅高 19%。

**涉及文件**: `inference_full.py`

---

## 4. 清理旧比赛文件

**问题**: `submit.py` 列名含 `article_1..5` 是旧天池赛道命名，`evaluate.py` 同样引用 `article_` 前缀。

**方案**: `submit.py` 改为通用 `item_1..K`，`evaluate.py` 同步迁移。删除根目录的 `two_tower_history.json`、`.claudeignore`。

**涉及文件**: `submit.py`, `evaluate.py`

---

## 5. evaluate.py 补 import os

**问题**: `evaluate.py` 独立运行时缺少 `import os`。

**涉及文件**: `evaluate.py`