#!/bin/bash
set -e

echo "============================================================"
echo "  Amazon Reviews 2023 - Full Training Pipeline"
echo "  Models: TwoTower → SASRec V2 → DIN → Inference"
echo "============================================================"
echo ""

# 1. Data validation
echo "[1/5] Data validation..."
python code/check.py
echo ""

# 2. Train basic dual-tower
echo "[2/5] Train TwoTowerModel..."
python code/train_deep.py
echo ""

# 3. Train SASRec dual-tower
echo "[3/5] Train SASRec TwoTowerV2..."
python code/train_v2.py
echo ""

# 4. Train DIN reranking
echo "[4/5] Train DIN reranking model..."
python code/train_din.py
echo ""

# 5. Full inference (recall + reranking + submit)
echo "[5/5] Full inference pipeline..."
python code/inference_full.py
echo ""

echo "============================================================"
echo "  Full pipeline complete!"
echo "  Result: prediction_result/result_full_pipeline.csv"
echo "============================================================"
