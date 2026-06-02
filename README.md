# 🛒 Amazon Reviews 2023 — 推荐系统 (召回 + 精排)

> 基于 Amazon Reviews 2023 数据集，预测用户未来可能购买的商品，输出 Top-5 推荐列表

![Python](https://img.shields.io/badge/Python-3.11-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)

## 📖 项目背景

基于 Amazon Reviews 2023 数据集（McAuley Lab, UCSD），构建用户商品兴趣画像，预测用户未来可能购买的商品。当前版本实现了 **四路召回 + DIN 精排** 的完整推荐 pipeline。

数据源: [McAuley-Lab/Amazon-Reviews-2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)

## 🛠️ 核心算法

### 召回阶段 (四路融合)
1. **ItemCF** — 基于物品协同过滤，引入 IUF（用户活跃度惩罚）
2. **双塔向量召回** — TwoTowerV2 (SASRec + InfoNCE)，Transformer 自注意力用户塔
3. **品类偏好召回** — 根据用户历史品类偏好推荐热门商品
4. **热门兜底召回** — 冷启动用户的保底策略

### 精排阶段
- **DIN (Deep Interest Network)** — 注意力机制让目标商品与用户历史动态交互，逐商品精细打分

## 📂 项目结构

```text
├── code/                   # 核心代码目录
│   ├── config.py           # 全局配置 (路径、超参数、Amazon 类别)
│   ├── data_loader.py      # 数据加载、清洗、列名归一化、Dataset 类
│   ├── model.py            # 模型定义 (TwoTower / SASRec / MIND / DIN)
│   ├── itemcf.py           # ItemCF 协同过滤
│   ├── recall_fusion.py    # 四路召回融合
│   ├── submit.py           # 推荐结果输出
│   ├── utils.py            # 内存压缩工具
│   ├── train_deep.py       # 训练基础双塔
│   ├── train_v2.py         # 训练 SASRec 双塔
│   ├── train_din.py        # 训练 DIN 精排
│   ├── inference.py        # 基础推理
│   ├── inference_full.py   # 完整推理 (召回 + 精排)
│   └── check.py            # 数据校验
├── amazon_reviews/         # (Git 忽略) Amazon 数据文件
├── user_data/              # (Git 忽略) 模型输出
├── prediction_result/      # (Git 忽略) 推荐结果
├── requirements.txt
└── README.md
```

## 🚀 快速运行

### 1. 环境准备
```bash
pip install -r requirements.txt
```

### 2. 数据准备
从 Hugging Face 下载 benchmark CSV 到 `amazon_reviews/`:
```
https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/tree/main/benchmark/5core/rating_only
```

### 3. 验证数据
```bash
python code/check.py
```

### 4. 训练 & 推理
```bash
# 逐步执行
python code/train_deep.py    # 基础双塔
python code/train_v2.py      # SASRec 双塔
python code/train_din.py     # DIN 精排
python code/inference_full.py # 完整推理

# 或一键运行 (Windows)
run_all.bat
```

### Debug 模式
修改训练脚本中的 `offline=True`，只加载 10000 条数据快速验证。

## 📊 数据适配

`data_loader.py` 内部做列名归一化，将 Amazon 字段映射为统一列名:
- `parent_asin` → `click_article_id` / `article_id`
- `timestamp` → `click_timestamp`
- `rating ≥ 4` → 正反馈
- `category_id` = 类别名
- `words_count` = 1 (不活跃特征)
- `created_at_ts` = 每条商品最早评论时间

在 `config.py` 中切换 `AMAZON_CATEGORY` 可使用不同品类数据。
