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

# --- 2. 确认数据存在 (从 config.py 读取 EXT_CATEGORIES，自动检查) ---
echo ">>> Checking data files..."
MISSING_COUNT=0
for CAT in $(python -c "import sys; sys.path.insert(0, 'code'); from config import EXT_CATEGORIES; print(' '.join(EXT_CATEGORIES))"); do
    DATA_DIR=$(python -c "import sys; sys.path.insert(0, 'code'); from config import DATA_PATH; print(DATA_PATH)")
    REVIEW_PATH="$DATA_DIR/raw/review_categories/${CAT}.jsonl"
    META_PATH="$DATA_DIR/raw/meta_categories/meta_${CAT}.jsonl"

    if [ -f "$REVIEW_PATH" ]; then
        echo "  ✓ Review JSONL: $CAT"
    else
        echo "  ✗ MISSING: $REVIEW_PATH"
        MISSING_COUNT=$((MISSING_COUNT + 1))
    fi

    if [ -f "$META_PATH" ]; then
        echo "  ✓ Meta JSONL:    $CAT"
    else
        echo "  ✗ MISSING: $META_PATH"
        MISSING_COUNT=$((MISSING_COUNT + 1))
    fi
done

if [ $MISSING_COUNT -gt 0 ]; then
    echo ""
    echo "ERROR: $MISSING_COUNT data file(s) missing!"
    echo "Download them with:"
    for CAT in $(python -c "import sys; sys.path.insert(0, 'code'); from config import EXT_CATEGORIES; print(' '.join(EXT_CATEGORIES))"); do
        echo "  python code/download_raw.py --category $CAT --data-dir \$DATA_DIR"
    done
    exit 1
fi
echo ">>> All data files confirmed."

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
