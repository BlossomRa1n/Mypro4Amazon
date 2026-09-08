# 实验记录

本文记录 Amazon Reviews 2023 推荐系统的完整实验历程与结论，重点是**双塔召回 (V2 SASRec) 如何逼近并超越 ItemCF**，以及**多路召回的并集增益**。

## 0. 评估协议

- **数据**：5 品类 (Musical_Instruments / Office_Products / CDs_and_Vinyl / Video_Games / Toys_and_Games)，rating≥4 记为正反馈。
- **切分**：按时间 leave-last-out，训练窗口内构建用户历史，验证集为每个用户的最后一条正反馈。
- **主指标**：**全库 HR@100** —— 从全部 ~38 万商品中召回 Top-100，命中用户验证目标即记为命中。这是召回天花板，直接决定精排输入的候选池质量。
- **教训**：Sampled NDCG@20 与全库 HR@100 存在**反相关**（见 §3），曾导致 8.66% 的最优 epoch 被低估为 9.51% 的真实水平。现已切回 `EVAL_FULL_RANK=True` 以全库指标为准。

---

## 1. 双塔召回 (V2 SASRec) 优化时间线

目标是让 V2 单路全库 HR@100 逼近并超越 ItemCF (10.62%)。

| 阶段 | 改动 | 全库 HR@100 | 说明 |
|---|---|---|---|
| 基线 | 初版双塔 | 2.25% | 含 `user_id` 特征 |
| 消融 | **删除 user_id** | **2.48%** (+0.23) | user_id 是"背答案"捷径，删除后反而提升 |
| 调参 | **τ=0.07 + hard_neg=3** | **2.79%** (+0.31) | InfoNCE 温度 + 难负样本数量 |
| 修复 | **幽灵商品 union bug** | — | item 编码器 union 未过滤 raw_meta，265 万 → 38 万，需重设基线 |
| Warm-start | **截断 SVD 右奇异向量初始化** | **8.66%** (+2.32pp) | 关键突破，见 §2 |
| 评估修正 | 切全库评估 | 9.51% | 原 8.66% 被 Sampled NDCG 低估 |
| Round 3 | 同品类难负样本 | 9.71% (+0.20) | |
| Phase 0 | 评估协议 ①-④ + 同品类 | 9.95% | 逼近 ItemCF (差 0.67pp) |
| Phase 2 | **brand 偏好 + 时间衰减** | **10.13%** | Val 峰值 10.17% 破 10%，V2 单路最好 |

> **结论**：V2 单路 10.13%，已接近 ItemCF 10.62%（差 0.49pp）。两者属不同类型信号，六路融合后并集 HR 达 22.02%，远超任一路。

---

## 2. SVD Warm-Start（关键突破）

双塔冷启动训练收敛慢、易陷入局部最优。用截断 SVD 分解用户-物品交互矩阵，取其**右奇异向量**作为 item embedding 初始化，再微调用户塔：

- **效果**：全库 HR@100 6.34% → **8.66% (+2.32pp)**，且训练更快收敛。
- **对照**：LightGCN warm-start 9.49% (**-0.64pp**)，因图卷积过平滑且优化方向与 InfoNCE 不一致。**已放弃，SVD 是正确的 warm-start**。

相关代码：`code/als_init.py`（`ALS_METHOD='svd'`）。

---

## 3. 已放弃的方向（负面结论，避免重蹈）

| 方向 | 结果 | 结论 |
|---|---|---|
| queue-only / logQ 队列负采样 | -0.35% / -3.82% | 双杀，队列负样本对 SVD warm-start 后的模型有害 |
| LightGCN warm-start | 9.49% (-0.64pp) | 过平滑 + 方向拧了，SVD 才对 |
| ANCE 动态难负 (0.3/3) | 9.82% | 略弱于 Phase 0 |
| ANCE 动态难负 (0.1/1) | 9.53% | 更弱，整条 ANCE 路线放弃 |
| 保留 user_id 特征 | 2.25% (低于删后 2.48%) | 背答案捷径，删除 |

---

## 4. 各路独立召回指标

10 万验证用户，全库 HR@100（测量脚本 `code/recall_union_gain.py`）：

| 通道 | HR@100 | 命中人数 |
|---|---|---|
| ItemCF | 10.62% | 10,619 |
| V2 SASRec (Phase 2) | 10.13% | 10,131 |
| HSTU | 8.71% | 8,711 |
| MIND | 6.64% | 6,645 |
| Category | 3.13% | 3,132 |
| Hot | 0.43% | 425 |

---

## 5. 六路并集增益（核心结论）

| 并集 | HR@100 | 增量 |
|---|---|---|
| 四路 (ItemCF ∪ V2 ∪ Category ∪ Hot) | 17.53% | — |
| +HSTU | 20.79% | +3.26pp |
| +MIND | 19.13% | +1.59pp |
| **六路全并集** | **22.02%** | **+4.49pp** |

**独有命中**（仅该路命中、四路基线都漏的用户）：

| 通道 | 独有命中 |
|---|---|
| HSTU | 3.26pp (3264 人) |
| MIND | 1.59pp (1595 人) |
| HSTU ∪ MIND | 4.49pp (4491 人) |
| HSTU ∩ MIND | 0.37pp (368 人) |

> **核心方法论**：召回通道价值 = **增量并集贡献（独有命中）**，而非 standalone HR。
> HSTU 虽与 V2 共享 item_embedding，但 pointwise 传导 vs softmax 注意力读序列的方式不同 → 用户向量与 Top-100 检索显著不同 → 独有命中 3.26pp 是纯增量。HSTU 与 MIND 独有命中重叠仅 0.37pp，近乎可加，都应接入融合。

---

## 6. 融合权重

- **四路最优**（历史 grid search）：`itemcf=1.5, v2=1.0, category=0.7, hot=0.05`，较默认权重 +0.62pp。
- **六路权重**：Phase 2 使 V2 从 8.66% → 10.13%，四路权重会漂移；新增 HSTU / MIND 两路后需重调。当前 `code/tune_recall_weights_6ch.py` 两阶段 grid search（Stage 1 四路 × 540 组合，Stage 2 HSTU×MIND × 16 组合）正在运行，最终权重见结果文件 `prediction_result/fusion_weights_6ch.json`。

---

## 7. 模型演进

| 模型 | 类型 | 用户塔 | 用途 |
|---|---|---|---|
| TwoTowerV2 | SASRec 双塔 | Transformer 因果自注意力 (2 blocks, 2 heads) + InfoNCE | 主力向量召回 |
| HSTU | 双塔 | 点态 q⊙k + 因果 cumsum (O(L·D)) | 召回（序列建模互补） |
| MIND | 多兴趣 | K=3 兴趣胶囊 (动态路由) | 召回（多兴趣互补） |
| DIN (Extended) | 精排 | 注意力 + brand/verified/评分特征 | Top-5 精排 |

## 8. 精排 (DIN) 结果

Extended DIN（含 brand / verified / helpful / 评分 / 物品质量特征）在 leave-last-out + 50 随机负样本协议下：

- Pairwise AUC 约 0.75（区分能力，论文标准口径）。
- 精排作为召回之后的 Top-5 重排，效果上限由召回候选池决定 —— 这也是本项目把主要精力放在**召回并集增益**上的原因。
