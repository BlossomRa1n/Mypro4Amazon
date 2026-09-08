# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

Amazon Reviews 2023 推荐系统 — 召回+精排方案。基于用户对商品的评分行为预测用户未来可能购买的商品，输出 Top-5 推荐列表。

## Commands

### Environment Setup
```bash
pip install -r requirements.txt
```

### Data Preparation

#### Amazon Reviews 2023
从 Hugging Face 镜像下载 benchmark CSV（5-core 过滤版）到 `amazon_reviews/`:
- 国内镜像: `https://hf-mirror.com/datasets/McAuley-Lab/Amazon-Reviews-2023/tree/main/benchmark/5core/rating_only`
- 原始地址: `https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/tree/main/benchmark/5core/rating_only`

**三档规模控制** (通过 `config.py` 中的两个变量控制):
| 档位 | `AMAZON_CATEGORIES` | `AMAZON_SAMPLE_USERS` | `offline` | 数据量 | 适用场景 |
|---|---|---|---|---|---|
| 本地验证 | 1个品类 | None | `True` | ~1万条 | 快速验证代码能跑通 |
| 5060Ti 训练 | 2-3个品类 | None | `False` | ~175万条 | 小规模正式实验 |
| 服务器训练 | 5-10个品类 | None | `False` | 1000万+条 | 大规模完整训练 |

默认配置 (5060Ti):
```python
AMAZON_CATEGORIES = [
    'Musical_Instruments',   # ~25万条正反馈, ~1.4万用户, ~1.0万商品
    'Office_Products',       # ~50万条正反馈, ~8.0万用户, ~2.5万商品
    'All_Beauty',            # ~100万条正反馈, ~9.0万用户, ~3.5万商品
]
AMAZON_SAMPLE_USERS = None   # 设为 5000 可在本地快速实验
```
服务器可追加: `'Video_Games'`, `'Toys_and_Games'`, `'CDs_and_Vinyl'` 等。

更多可用类别及规模参考:
| 类别 | 评论数 (5-core) | 备注 |
|---|---|---|
| Musical_Instruments | ~25万 | 最小，快速实验 |
| Office_Products | ~50万 | 默认 |
| All_Beauty | ~100万 | |
| Video_Games | ~250万 | 中等规模 |
| Electronics | ~4400万 | 大类别，需大内存 |

**数据适配策略**: `data_loader.py` 内部做**列名归一化**，将 Amazon 字段映射为统一列名: `parent_asin` → `click_article_id`/`article_id`, `timestamp` → `click_timestamp`, rating≥4 → 正反馈。物品元数据完全从 review CSV 派生 (category_id=品类名, created_at_ts=最早评论时间)。多品类合并后 category_id 有真实区分度，品类 Embedding 被激活。下游 Dataset/模型/召回逻辑完全无需关心原始数据格式。

### Training & Inference

| Command | Description | Status |
|---|---|---|
| `python code/check.py` | 快速验证数据文件是否存在 | 可用 |
| `python code/train_v2.py` | 训练 SASRec 双塔 (TwoTowerV2Model + InfoNCE) | 可用 |
| `python code/train_din_ext.py` | 训练 Extended DIN 精排模型 (DINExtendedModel) | 可用 |
| `run_all.bat` | 一键执行全部训练 + 推理 (Windows) | 可用 |
| `code/train_all.bat` | 一键执行全部训练 (精简版) | 可用 |

### Debug 模式
修改训练脚本中的 `offline=True`，每个品类只加载 10000 条快速验证。或在 `config.py` 中设 `AMAZON_SAMPLE_USERS = 5000` 采样少量用户。

## Architecture

### Pipeline: 召回 → 重排

```
Amazon Reviews CSV → 数据加载 + 特征工程 → 模型训练 → 多路召回融合 → DIN 精排 → Top-5 提交
```

### Key Modules

| File | Role |
|---|---|
| `code/data_loader.py` | 多品类数据加载、清洗、ID 编码 (LabelEncoder)、用户/物品特征工程。含 `load_amazon_reviews()` / `load_amazon_products()` 及列名归一化。定义三个 Dataset 类：TwoTowerDataset, TwoTowerV2Dataset, DINDataset |
| `code/model.py` | 所有 PyTorch 模型定义 |
| `code/recall_fusion.py` | 四路召回融合：ItemCF / 双塔向量 / 品类偏好 / 热门兜底，多通道分数加权求和 |
| `code/submit.py` | 将召回字典转为提交 CSV 格式 (`user_id, article_1..5`) |
| `code/utils.py` | `reduce_mem()` DataFrame 内存压缩（int64/float64 → 小类型） |
| `code/check.py` | 数据校验：读取 Amazon CSV 前 5 行，验证数据文件是否就绪 |
| `code/config.py` | 全局配置：路径、超参数、Amazon 类别选择 |

### Model Evolution (in `model.py`)

1. **TwoTowerV2Model** (SASRec) - Transformer 因果自注意力用户塔 (2 blocks, 2 heads)，InfoNCE in-batch negative loss + Hard Negative Mining
2. **MINDModel** (多兴趣网络) - 动态路由 (capsule-style, 3 iterations) 将用户历史聚类为 K=3 兴趣胶囊，独立召回后合并

### Data Flow

- `get_all_click_df()` → 加载 Amazon benchmark CSV (rating≥4→click)，统一列名
- `load_articles()` → 从 review CSV 派生物品元数据 (category=品类名, words_count=1, created_at_ts=最早评论时间)
- `build_encoders()` → LabelEncoder 映射 raw ID → 连续整数，结果缓存为 pickle；额外存 raw_to_idx 映射
- `build_item_features()` → 物品特征：品类、词数、点击数、创建时间
- `build_user_features()` → 用户画像：点击数、时间跨度、历史序列 (max 50)
- `build_enhanced_user_features()` → 扩展特征：品类偏好 Top-3、活跃度分层、兴趣熵
- `build_hard_negative_index()` → 从 ItemCF 相似度矩阵挖掘高相似度但未点击的难负样本
- Dataset `__getitem__` → 返回样本字典 (user_idx, pos_item, neg_item, 历史序列, 数值特征)
- 训练结束后预计算全部 item embedding → pickle 保存，推理时直接加载做向量检索

### File Outputs (Git-ignored)

| Path | Content |
|---|---|
| `user_data/model_data/*.pkl` | ItemCF 相似度、item embedding 向量 |
| `user_data/model_data/*.pth` | PyTorch 模型 checkpoint |
| `user_data/tmp_data/id_encoders.pkl` | LabelEncoder 映射表 |
| `prediction_result/result*.csv` | 提交文件 (user_id + 5 个 article) |

### Dependencies

`requirements.txt`: pandas, torch, numpy, scikit-learn, tqdm
