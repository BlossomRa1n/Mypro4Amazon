#!/bin/bash
set -e

echo "============================================================"
echo "  Resume from TwoTowerV2 (train_v2)"
echo "============================================================"
echo ""

# Train SASRec dual-tower
echo "[1/3] Train SASRec TwoTowerV2..."
python code/train_v2.py
echo ""

# Train Extended DIN reranking
echo "[2/3] Train Extended DIN reranking model..."
bash run_ext_din.sh
echo ""

# Full inference (recall + reranking + submit)
echo "[3/3] Full inference pipeline..."
python code/inference_full.py
echo ""

echo "============================================================"
echo "  Full pipeline complete!"
echo "  Result: prediction_result/result_full_pipeline.csv"
echo "============================================================"
