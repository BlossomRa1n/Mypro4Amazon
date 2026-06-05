#!/bin/bash
# ============================================================
# Extended DIN (All_Beauty raw) 服务器一键运行脚本
# RTX 5070 32GB vGPU
# ============================================================
set -e

# --- 0. 删除 offline 模式残留缓存 ---
if [ -f "user_data/tmp_data/id_encoders_ext.pkl" ]; then
    echo ">>> Removing stale encoder cache from offline test..."
    rm -f user_data/tmp_data/id_encoders_ext.pkl
fi

# --- 1. 准备目录 ---
echo ">>> Creating directories..."
mkdir -p user_data/model_data/checkpoints
mkdir -p user_data/tmp_data
mkdir -p prediction_result

# --- 2. 确认数据存在 ---
if [ ! -f "amazon_reviews/raw/review_categories/All_Beauty.jsonl" ]; then
    echo "ERROR: Review JSONL not found!"
    exit 1
fi
if [ ! -f "amazon_reviews/raw/meta_categories/meta_All_Beauty.jsonl" ]; then
    echo "ERROR: Meta JSONL not found!"
    exit 1
fi
echo ">>> Data files confirmed."

# --- 3. 确认全量模式 ---
echo ">>> OFFLINE_MODE = False (全量模式)"

# --- 4. 训练 ---
echo ""
echo "=========================================="
echo " Starting Extended DIN Training"
echo "=========================================="
echo ""

python code/train_din_ext.py

echo ""
echo "=========================================="
echo " Done!"
echo "=========================================="
echo ""
echo "Key outputs:"
echo "  Best model:    user_data/model_data/din_ext_best.pth"
echo "  Latest ckpt:   user_data/model_data/din_ext_latest.pth"
echo "  Training log:  user_data/model_data/din_ext_history.json"
echo "  Encoders:      user_data/tmp_data/id_encoders_ext.pkl"
