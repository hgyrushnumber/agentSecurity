#!/usr/bin/env bash

set -euo pipefail


ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)

cd "$ROOT_DIR"


export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false



MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B}"

DATA_FILE="${DATA_FILE:-processed/xlam_tool_count_trigger_1to8.jsonl}"

OUTPUT_DIR="${OUTPUT_DIR:-outputs/qwen3_4b_tool_count_trigger_lora}"



echo "======================================"
echo "Start SFT Training"
echo "======================================"

echo "Model:"
echo "${MODEL_PATH}"

echo "Dataset:"
echo "${DATA_FILE}"

echo "Output:"
echo "${OUTPUT_DIR}"

echo "GPU:"
echo "${CUDA_VISIBLE_DEVICES}"



python scripts/train_tool_count_trigger_sft.py \
  --model-name-or-path "${MODEL_PATH}" \
  --train-file "${DATA_FILE}" \
  --output-dir "${OUTPUT_DIR}" \
  --threshold 3 \
  --validation-ratio 0.05 \
  --split-group-by query \
  --split-seed 42 \
  --max-seq-length 8192 \
  --preprocessing-num-workers 4 \
  --num-train-epochs 3 \
  --learning-rate 2e-4 \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 16 \
  --gradient-checkpointing \
  --lora-rank 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --lora-target-modules all-linear \
  --warmup-ratio 0.03 \
  --lr-scheduler-type cosine \
  --logging-steps 5 \
  --eval-steps 200 \
  --save-steps 200 \
  --save-total-limit 3 \
  --seed 42 \
  --data-seed 42 \
  --report-to none \
  --resume-from-checkpoint auto



echo "======================================"
echo "SFT Finished"
echo "======================================"
