#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B}"
TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-outputs/qwen3_4b_tool_count_trigger_lora}"
ADAPTER_PATH="${ADAPTER_PATH:-${TRAIN_OUTPUT_DIR}/final_adapter}"
EVAL_FILE="${EVAL_FILE:-${TRAIN_OUTPUT_DIR}/data/validation.jsonl}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${TRAIN_OUTPUT_DIR}/evaluation}"

python scripts/evaluate_tool_count_trigger.py \
  --model-name-or-path "${MODEL_PATH}" \
  --adapter-path "${ADAPTER_PATH}" \
  --eval-file "${EVAL_FILE}" \
  --output-dir "${EVAL_OUTPUT_DIR}" \
  --threshold 3 \
  --batch-size 4 \
  --max-new-tokens 256 \
  --samples-per-tool-count 200 \
  --seed 42
