# 🛒 Amazon Reviews 2023 — 推荐系统 (召回 + 精排)

> 基于 Amazon Reviews 2023 数据集，预测用户未来可能购买的商品，输出 Top-5 推荐列表。

![Python](https://img.shields.io/badge/Python-3.11-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)

数据源: [McAuley-Lab/Amazon-Reviews-2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) (5-core 过滤版, rating≥4 记为正反馈)

## 🏆 核心结果 (10 万验证用户, 全库 HR@100)

### 六路召回并集增益

| 通道 | 独立 HR@100 | 独有命中 (对四路并集的增量) |
|---|---|---|
| ItemCF | 10.62% | — (基线路) |
| V2 SASRec (Phase 2) | **10.13%** | — (基线路) |
| Category | 3.13% | — (基线路) |
| Hot | 0.43% | — (基线路) |
| **HSTU** | 8.71% | **+3.26pp** (3264 人) |
| **MIND** | 6.64% | **+1.59pp** (1595 人) |

| 并集 | HR@100 |
|---|---|
| 四路 (ItemCF ∪ V2 ∪ Category ∪ Hot) | 17.53% |
| +HSTU | 20.79% (+3.26pp) |
| +MIND | 19.13% (+1.59pp) |
| **六路全并集** | **22.02% (+4.49pp)** |

**关键洞察**：召回通道的价值看**增量并集贡献（独有命中）**，不看 standalone HR。HSTU / MIND 单路虽只有 8.71% / 6.64%，但独有命中分别 +3.26pp / +1.59pp，且二者重叠仅 0.37pp —— 近乎可加，故都应接入融合。

## 🧠 系统架构

```
Amazon Reviews CSV / raw JSONL
        │  数据加载 + 特征工程 (data_loader / data_loader_ext)
        │  ID 编码 (LabelEncoder, 缓存 pickle)
        ▼
┌───────────── 召回阶段 · 六路融合 ─────────────┐
│ 1. ItemCF      物品协同过滤 (IUF 惩罚)         │
│ 2. V2 SASRec   双塔向量召回 (Transformer 用户塔) │
│ 3. HSTU        点态传导序列建模                 │
│ 4. MIND        多兴趣胶囊 (K=3, 动态路由)       │
│ 5. Category    品类偏好召回                     │
│ 6. Hot         热门兜底 (冷启动)                │
└───────── 加权融合 → Top-100 候选池 ───────────┘
        ▼
┌───────────── 精排阶段 ─────────────┐
│ DIN (Deep Interest Network)        │
│ 注意力: 目标商品 × 用户历史动态交互  │
│ + brand / verified / 评分特征      │
└────────────── → Top-5 推荐列表 ────┘
```

## 🛠️ 核心算法

### 召回模型 (双塔)

| 模型 | 用户塔结构 | 关键点 |
|---|---|---|
| **TwoTowerV2 (SASRec)** | Transformer 因果自注意力 (2 blocks, 2 heads) | 主力通道, InfoNCE 损失 (τ=0.07), L2 归一化余弦 |
| **HSTU** | Hierarchical Sequential Transduction Unit | 点态 q⊙k 交互 + 因果 cumsum, O(L·D); 与 SASRec "同角色、不同读法" |
| **MIND** | 多兴趣网络 (K=3 胶囊) | 动态路由将历史聚成 3 个兴趣向量, 各独立召回后合并 |

### 关键优化：SVD Warm-Start

用截断 SVD 的右奇异向量初始化 item embedding，V2 全库 HR@100 **6.34% → 8.66% (+2.32pp)**。

> 对照实验：LightGCN warm-start 反而 -0.64pp（9.49%），过平滑且方向相反，已放弃。**SVD 才是正确的 warm-start**。

### 精排模型

- **DIN (Deep Interest Network)** — 注意力机制让目标商品与用户历史动态交互，逐商品精细打分（Extended 版含 brand / verified / helpful / 评分等特征）。

## 📂 项目结构

```text
├── code/
│   ├── config.py              # 全局配置 (路径、超参数、类别、召回权重)
│   ├── data_loader.py         # CSV 加载 + 列名归一化 + 三个 Dataset 类
│   ├── data_loader_ext.py     # 扩展 encoder / 特征 (brand / verified / 评分)
│   ├── model.py               # 所有模型 (TwoTowerV2 / HSTU / MIND / DIN)
│   ├── model_ext.py           # DINExtendedModel
│   ├── itemcf.py              # ItemCF 协同过滤 (IUF)
│   ├── als_init.py            # SVD warm-start (item embedding 初始化)
│   ├── recall_fusion.py       # 多路召回融合 + embedding_recall
│   ├── recall_union_gain.py   # 各路并集增益 / 独有命中测量
│   ├── recall_attribution.py  # 召回通道归因拆解
│   ├── tune_recall_weights_*.py  # 融合权重 grid search
│   ├── train_v2.py            # 训练 SASRec 双塔
│   ├── train_hstu.py          # 训练 HSTU 双塔
│   ├── train_mind.py          # 训练 MIND 多兴趣网络
│   ├── train_din_ext.py       # 训练 Extended DIN 精排
│   ├── evaluate.py            # 评估协议 (leave-last-out + 全库 HR/NDCG)
│   ├── inference_full.py      # 完整推理 (召回 + 精排)
│   └── eval_*_hr100.py        # 全库 HR@100 评估脚本
├── docs/                      # 设计文档 (ranker_v2 / HSTU 升级 / 生成式召回)
├── amazon_reviews/            # (Git 忽略) Amazon 数据文件
├── user_data/                 # (Git 忽略) 模型 / embedding / encoder
├── prediction_result/         # (Git 忽略) 推荐结果 / 评估 JSON
├── requirements.txt
├── run_all.bat / run_all.sh   # 一键训练 + 推理
├── RESULTS.md                 # 完整实验记录
└── README.md
```

## 🚀 快速运行

```bash
# 1. 环境
pip install -r requirements.txt

# 2. 数据 (下载 5-core rating_only CSV 到 amazon_reviews/)
#    https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/tree/main/benchmark/5core/rating_only

# 3. 验证数据
python code/check.py

# 4. 训练 & 推理
python code/train_v2.py        # SASRec 双塔 (主力召回)
python code/train_din_ext.py   # Extended DIN 精排
python code/inference_full.py  # 完整推理 (召回 + 精排)
```

**Debug 模式**：训练脚本设 `offline=True` 每类只加载 10000 条；或 `config.py` 中 `AMAZON_SAMPLE_USERS = 5000` 采样。

## 📊 数据适配

`data_loader.py` 内部做列名归一化，将 Amazon 字段映射为统一列名，下游 Dataset / 模型 / 召回逻辑无需关心原始格式：

- `parent_asin` → `click_article_id` / `article_id`
- `timestamp` → `click_timestamp`
- `rating ≥ 4` → 正反馈
- `category_id` = 品类名（多品类合并后有真实区分度，品类 Embedding 被激活）
- `created_at_ts` = 每条商品最早评论时间

## 📈 实验进展

完整实验记录、各阶段指标与结论见 **[RESULTS.md](RESULTS.md)**。
