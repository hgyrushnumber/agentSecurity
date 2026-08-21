#!/usr/bin/env bash
# Unified SFT training entrypoint (wraps both experiment families).
#
# Usage:
#   bash scripts/sft.sh xlam [--model M] [--data-file F] [--output-dir D] [--threshold N]
#   bash scripts/sft.sh nemotron [--model M] [--train-file F] [--validation-file F] [--output-dir D] [--dry-run]
#
# Examples:
#   bash scripts/sft.sh xlam
#   bash scripts/sft.sh xlam --model /data/models/Qwen3-4B --output-dir outputs/my_run
#   bash scripts/sft.sh nemotron --output-dir outputs/nemotron_lora
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TOKENIZERS_PARALLELISM=false

if [ "$#" -lt 1 ]; then
  echo "Usage: bash scripts/sft.sh xlam|nemotron [options]" >&2
  exit 1
fi
STEP_NAME=$1
shift || true

case "$STEP_NAME" in
  xlam)
    MODEL=Qwen/Qwen3-4B
    DATA_FILE=processed/xlam_tool_count_trigger_1to8.jsonl
    OUTPUT_DIR=outputs/qwen3_4b_tool_count_trigger_lora
    THRESHOLD=3
    while [ "$#" -gt 0 ]; do
      case "$1" in
        --model) MODEL=$2; shift 2 ;;
        --data-file) DATA_FILE=$2; shift 2 ;;
        --output-dir) OUTPUT_DIR=$2; shift 2 ;;
        --threshold) THRESHOLD=$2; shift 2 ;;
        *) echo "unknown option: $1" >&2; exit 1 ;;
      esac
    done

    echo "===== SFT (xlam tool-count-trigger) ====="
    echo "  model=$MODEL  data=$DATA_FILE  output=$OUTPUT_DIR  threshold=$THRESHOLD"
    if [ ! -f "$DATA_FILE" ]; then
      echo "ERROR: $DATA_FILE not found. Run: bash scripts/process_datasets.sh xlam" >&2
      exit 1
    fi
    python scripts/train_tool_count_trigger_sft.py \
      --model-name-or-path "$MODEL" \
      --train-file "$DATA_FILE" \
      --output-dir "$OUTPUT_DIR" \
      --threshold "$THRESHOLD" \
      --validation-ratio 0.05 \
      --split-seed 42 \
      --split-group-by query \
      --max-seq-length 4096 \
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
    ;;

  nemotron)
    MODEL=Qwen/Qwen3-4B
    TRAIN_FILE=processed/nemotron_sft/train.jsonl
    VALIDATION_FILE=processed/nemotron_sft/validation.jsonl
    OUTPUT_DIR=outputs/nemotron_same_tool_trigger_lora
    DRY_RUN=""
    while [ "$#" -gt 0 ]; do
      case "$1" in
        --model) MODEL=$2; shift 2 ;;
        --train-file) TRAIN_FILE=$2; shift 2 ;;
        --validation-file) VALIDATION_FILE=$2; shift 2 ;;
        --output-dir) OUTPUT_DIR=$2; shift 2 ;;
        --dry-run) DRY_RUN=--dry-run; shift ;;
        *) echo "unknown option: $1" >&2; exit 1 ;;
      esac
    done

    echo "===== SFT (nemotron same-tool-trigger) ====="
    echo "  model=$MODEL  train=$TRAIN_FILE  output=$OUTPUT_DIR  dry_run=$DRY_RUN"
    if [ ! -f "$TRAIN_FILE" ]; then
      echo "ERROR: $TRAIN_FILE not found. Run: bash scripts/process_datasets.sh nemotron --parquet ..." >&2
      exit 1
    fi
    python scripts/train_nemotron_same_tool_trigger_sft.py \
      --model "$MODEL" \
      --train-file "$TRAIN_FILE" \
      --validation-file "$VALIDATION_FILE" \
      --output-dir "$OUTPUT_DIR" \
      --max-length 4096 \
      --max-target-length 1024 \
      --prompt-head-ratio 0.35 \
      --epochs 1.0 \
      --learning-rate 1e-4 \
      --batch-size 1 \
      --gradient-accumulation-steps 16 \
      --eval-samples 2000 \
      --logging-steps 20 \
      --eval-steps 1000 \
      --save-steps 1000 \
      --save-total-limit 2 \
      --warmup-ratio 0.03 \
      --lora-r 16 \
      --lora-alpha 32 \
      --lora-dropout 0.05 \
      --seed 42 \
      --attn-implementation sdpa \
      $DRY_RUN
    ;;

  *)
    echo "Usage: bash scripts/sft.sh xlam|nemotron [options]" >&2
    echo "  xlam:     tool-count-trigger SFT (threshold 3, LoRA)" >&2
    echo "  nemotron: same-tool-trigger SFT (LoRA)" >&2
    exit 1
    ;;
esac

echo "===== SFT done ====="
