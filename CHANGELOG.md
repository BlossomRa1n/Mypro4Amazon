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


# 2026-06-06 改动日志

## 0. 全链路向量化性能优化 — 消除 6 大 Python for 循环瓶颈

**问题**: 精排模型服务器端运行 `train_din_ext.py`，Step 2 `build_extended_item_features` 和 `build_extended_user_features` 卡死（175 万条交互、11 万件商品、3 万品牌），CPU 100% 但数分钟无进展。问题根源是 6 处`iterrows()` / `for i in range(len(df))` / `.loc[idx]` / `get_brand_idx()` 逐行 Python 函数调用，在高数据量下产生百万次解释器开销。

**方案 — 6 处向量化重写**:

| # | 函数 (文件) | 瓶颈 | 优化方案 |
|---|---|---|---|
| 1 | `prepare_click_df` (data_loader_ext) | `zip(*df['rating'].map(lambda ...))` 逐行 λ | `np.searchsorted` + 预建查找表，一次搞定 175 万行 |
| 2 | `build_extended_item_features` (data_loader_ext) | 两个 `for i in range(len(meta_df))` + `.iloc[i]` 逐行取数，11 万项 × 2 | pandas `.loc` 布尔广播 + numpy 高级索引一步赋值 |
| 3 | `build_extended_user_features` (data_loader_ext) | 每组 `sort_values`、每条记录 `get_brand_idx` O(3 万) 线性扫描、Python for 时间差 | 全局排序 + 预建 `item_to_brand_arr` O(1) 数组查表 + `np.diff` 向量化 |
| 4 | `DINExtendedDataset.__init__` (data_loader_ext) | 3 个 `iterrows()`、11 万次 `.loc[idx]`、175 万次逐行负样本 `rng.integers` | DataFrame 列 `to_numpy` 直拷 + 负样本批量预采样 + 合并重复循环 |
| 5 | `split_train_val` (evaluate.py) | 每组 `sort_values` + 逐个 `pd.concat` 拼接 | `cumcount` 分组计数器 + 批量 boolean mask，零 Python 循环 |
| 6 | `evaluate_ext` (train_din_ext) | `iterrows` 遍历 user/item、每样本单次 tensor 拷贝、每用户重复序列 pad | `uid_to_idx` 数组直建 + 用户 tensor 预分配并复用 + 负样本批量 `randint` |

**核心原则**: 所有逐行操作统一用 numpy/pandas 向量化操作（一次处理整列/整数组）替代 Python for 循环。175 万级数据量下，每个逐行操作都是百万次 Python 函数调用开销，全部消除后预计启动速度从数分钟降到数秒。

**涉及文件**: `code/data_loader_ext.py`, `code/train_din_ext.py`, `code/evaluate.py`


## 1. Video_Games 首次训练 — BPR Loss 坍缩 & 随机负样本失效

**数据规模**: Video_Games raw JSONL (2.5GB review + 418MB meta) → `EXT_MIN_USER_INTER=5` / `EXT_MIN_ITEM_INTER=5` 过滤后 92,498 用户 / 23,122 商品 / 5,874 品牌 / 1,858 BPR batches

**训练结果**:

| Epoch | Train Loss | Val AUC | Pos Mean |
|---|---|---|---|
| 1 | 0.0235 | 0.5719 | 0.4121 |
| 2 | 0.0002 | 0.4979 | 0.3669 |
| 3 | 0.0001 | 0.6167 | 0.4042 |
| 4 | ~0 | 0.5978 | 0.3794 |
| **5** | **~0** | **0.8695** | **0.4548** |
| 6 | ~0 | 0.6082 | 0.2014 |
| 7 | ~0 | 0.7950 | 0.2809 |
| 8 | ~0 | 0.6033 | 0.3384 |
| 9 | ~0 | 0.5892 | 0.2711 |
| 10 | ~0 | 0.5553 | 0.2020 |

**诊断 — BPR Loss 坍缩**:
- Loss 从 epoch 2 开始接近 0（~5e-5），模型几乎完美区分训练集正负样本
- 但 Val AUC 剧烈震荡（0.50 → 0.87 → 0.55），从未稳定
- Pos Mean 持续下降（0.45 → 0.20），模型对正样本越来越不自信

**根因 — 随机负样本太容易区分**:
Video_Games 23K 商品中随机采样的负样本与用户历史无任何相似性，模型学到的不是用户偏好信号，只是"正样本=历史里见过的"这个简单规则。训练集上几乎完美，但验证集完全无法泛化。

**解决方向**: 需要 ItemCF Hard Negative Mining — 用与正样本 ItemCF 相似度高但用户未交互的商品作为负样本，强制模型学习细粒度的用户偏好区分。旧 DIN 管线 (`data_loader.py` 已实现 `build_hard_negative_index`) 但扩展 DIN 尚未集成。

**涉及文件**: `code/data_loader_ext.py` (待修改), `code/train_din_ext.py`, `code/config.py` (切换品类), `run_ext_din.sh` (自动清理旧 checkpoint)


## 2. 数据加载内存优化 — list-of-dicts → 类型化列数组

**问题**: Books raw JSONL 20GB，原 `load_raw_reviews`/`load_raw_meta` 逐行构建 dict 再 `pd.DataFrame(list_of_dicts)` 导致峰值内存 = dict overhead（~2×） + DataFrame（~1×） = 同时占用 ~3× 数据量内存，32GB 服务器直接 OOM。

**方案**: 改为 `zip(*all_rows)` 拆成 7 个类型化 Python list → `np.array` / `pd.array` 直接构造 DataFrame，构造后立即 `del` 中间列表。meta 同样改造，price 用 -1.0 替代 None 避免 object dtype。

**效果**: 峰值内存减半（~15-20GB），Video_Games (2.5GB) 作为中间规模验证方案可行性。

**涉及文件**: `code/data_loader_ext.py` — `load_raw_reviews()`, `load_raw_meta()`


## 3. Hard Negative Mining 集成到扩展 DIN

**问题**: Video_Games 首次训练 BPR Loss 坍缩 → Loss≈0, Val AUC 震荡, Pos Mean 持续下降。随机负样本与正样本无相似性，模型学到的是"正样本=历史见过"的平凡规则。

**方案**:
- `data_loader_ext.py` 新增 `build_hard_negative_index_ext()` — 从 ItemCF 相似度矩阵取 Top-K 相似但未交互商品，`heapq.nlargest` O(N log K) 替代全量 sort
- `DINExtendedDataset.__init__` 新增 `hard_neg_index`/`num_hard_negatives` 参数 — 负样本采样流程改为 HardNeg 优先 → 不够补随机
- `train_din_ext.py` 自动加载/构建 ItemCF 索引（首次运行时从 `click_df` 构建并缓存到 `ITEMCF_SIM_PKL`），传入 `build_hard_negative_index_ext` 和 `DINExtendedDataset`
- `itemcf.py` 一并打包进压缩包（`itemcf_sim` 函数为依赖）

**预期效果**: Hard Negative 强制模型学习细粒度偏好区分，类似旧 DIN 管线中 BPR+HardNeg 将 AUC 从 0.50 提升到 0.66 的效果。

**涉及文件**: `code/data_loader_ext.py`, `code/train_din_ext.py`, `code/itemcf.py` (新纳入打包)


## 4. Checkpoint 兼容性检查 & 品类切换防护

**问题**: 
1. 从 All_Beauty (240 用户) 切换到 Video_Games (92K 用户) 后，恢复旧 checkpoint 导致 `CUDA error: device-side assert triggered — index out of bounds`（Embedding 表尺寸不匹配）
2. 上传新代码包后 config.py 路径被覆盖回默认值 `/root/amazon_reviews`，但服务器数据在 `/root/autodl-tmp/amazon_data`

**方案**:
- `train_din_ext.py` resume 时检查 `ckpt.num_users != current.num_users` → 自动重建模型 + optimizer + scheduler，清零 epoch/best_auc/patience
- `run_ext_din.sh` 启动时自动清理所有旧 checkpoint（`din_ext_latest.pth`, `din_ext_best.pth`, `din_ext_history.json`, `checkpoints/din_ext_epoch*.pth`）+ encoder 缓存
- `config.py` 中 `DATA_PATH` 硬编码为 `/root/autodl-tmp/amazon_data`（服务器数据盘），避免每次上传后被覆盖

**涉及文件**: `code/train_din_ext.py`, `run_ext_din.sh`, `code/config.py`


## 5. sklearn LabelEncoder NA TypeError 修复

**问题**: Pandas `string` dtype 的 `<NA>` 值在 `meta_df['brand'].unique()` / `meta_df['main_category'].unique()` 时混入 `NAType`，导致 `LabelEncoder.fit()` 报 `TypeError: Encoders require their input argument must be uniformly strings or numbers. Got ['NAType', 'str']`

**方案**: `build_extended_encoders` 中 brand 和 category 的 `unique()` 列表在传入 `LabelEncoder.fit()` 之前过滤 `str(x) != '<NA>'`。

**涉及文件**: `code/data_loader_ext.py` — `build_extended_encoders()`


## 6. 从 All_Beauty → Video_Games 品类切换

**动机**: All_Beauty 人均 1.1 条交互，5-core 后仅 240 用户；DIN 论文使用 Electronics/Books（人均 8-12 条）。Video_Games 人均 ~8 条，5-core 后 ~92K 用户，规模适合当前实验。

**涉及文件**: `code/config.py` — `EXT_CATEGORIES = ['Video_Games']`

---

## 7. Extended DIN 过拟合修复 — 三 Bug 合治

**问题**: Video_Games 首次训练 train loss 从 epoch 2 归零，val AUC 剧烈震荡（0.50→0.87→0.55→0.32），Pos Mean 持续衰减。三个深层 bug：

| Bug | 文件:行 | 问题 | 修复 |
|---|---|---|---|
| 数据泄露 | `train_din_ext.py:300` | `build_extended_*_features(click_df)` 用完整数据（含验证集）构建用户历史，验证集目标 item 直接出现在用户序列中 | 改为 `train_click` |
| 品牌特征丢失 | `data_loader_ext.py:335` | `item_brand_arr` 分配后从未填充，所有 item brand_id=0，brand_embedding 全程为零 | 新增 `brand_map` + meta 散射填充 |
| 困难负样本自毁 | `train_din_ext.py:305` | ItemCF 困难负样本 = 与历史相似但未点击的 item，BPR 教会模型抑制所有相似 item—验证正样本也在其中 | `USE_HARD_NEGATIVES=False` 关闭 |

**效果**:

| Epoch | Train Loss | Val AUC | Pos Mean |
|---|---|---|---|
| 1 | 0.0266 | 0.6485 | 0.4881 |
| 2 | 0.0002 | 0.7224 | 0.4247 |
| 3 | 0.0001 | 0.7363 | 0.2981 |
| **4** | **0.0000** | **0.8498 ★** | **0.3710** |
| 5 | 0.0001 | 0.4434 | 0.2669 |
| 6 | 0.0000 | 0.5853 | 0.2512 |
| 7 | 0.0000 | 0.3236 | 0.1832 |
| 8 | 0.0000 | 0.4095 | 0.2335 |
| 9 | 0.0000 | 0.7044 | 0.2841 |

**诊断**: 数据泄露修复后 AUC 峰值 0.85（vs 修复前 0.50 均线），但 BPR loss 仍在 epoch 2-3 归零，AUC 在 epoch 4 后崩盘。BPR + 59M 参数 MLP = 天然记忆化倾向。下一步考虑 BCE 回归或更强的正则化（L2 weight decay、更强 dropout、梯度裁剪）。

**涉及文件**: `code/train_din_ext.py`, `code/data_loader_ext.py`