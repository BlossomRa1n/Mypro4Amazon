#!/bin/bash
set -e

echo "============================================================"
echo "  Resume from TwoTowerModel (train_deep)"
echo "============================================================"
echo ""

# Train basic dual-tower
echo "[1/4] Train TwoTowerModel..."
python code/train_deep.py
echo ""

# Train SASRec dual-tower
echo "[2/4] Train SASRec TwoTowerV2..."
python code/train_v2.py
echo ""

# Train DIN reranking
echo "[3/4] Train DIN reranking model..."
python code/train_din.py
echo ""

# Full inference (recall + reranking + submit)
echo "[4/4] Full inference pipeline..."
python code/inference_full.py
echo ""

echo "============================================================"
echo "  Full pipeline complete!"
echo "  Result: prediction_result/result_full_pipeline.csv"
echo "============================================================"
