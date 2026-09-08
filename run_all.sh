#!/bin/bash
set -e

echo "============================================================"
echo "  Amazon Reviews 2023 - Full Training Pipeline"
echo "  Dataset: Video_Games (benchmark + raw JSONL)"
echo "  Models: SASRec V2 → Extended DIN → Inference"
echo "============================================================"
echo ""

# 0. Clean old checkpoints
echo "[0/5] Cleaning old checkpoints..."
rm -f user_data/tmp_data/id_encoders.pkl user_data/tmp_data/id_encoders_ext.pkl
rm -f user_data/model_data/din_ext_latest.pth user_data/model_data/din_ext_best.pth user_data/model_data/din_ext_history.json
rm -f user_data/model_data/two_tower_v2_latest.pth user_data/model_data/two_tower_v2_best.pth user_data/model_data/two_tower_v2_history.json
rm -f user_data/model_data/checkpoints/*.pth
mkdir -p user_data/model_data/checkpoints user_data/tmp_data prediction_result
echo ""

# 1. Data validation
echo "[1/5] Data validation..."
python code/check.py
echo ""

# 2. Train SASRec dual-tower (TwoTowerV2 + InfoNCE)
echo "[2/5] Train SASRec TwoTowerV2..."
python code/train_v2.py
echo ""

# 3. Train Extended DIN (raw JSONL features)
echo "[3/5] Train Extended DIN reranking model..."
bash run_ext_din.sh
echo ""

# 4. Full inference (recall + reranking + submit)
echo "[4/5] Full inference pipeline..."
python code/inference_full.py
echo ""

echo "============================================================"
echo "  Full pipeline complete!"
echo "  Result: prediction_result/result_full_pipeline.csv"
echo "============================================================"
