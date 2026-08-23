#!/usr/bin/env bash
# Dataset-scoped SFT entrypoint.
#
# Usage:
#   bash scripts/sft.sh xlam [--model M|--model-id ID] [--data-file F] [--output-dir D]
#   bash scripts/sft.sh nemotron [--model M|--model-id ID] [--train-file F] [--output-dir D]
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TOKENIZERS_PARALLELISM=false

if [ "$#" -lt 1 ]; then
  echo "Usage: bash scripts/sft.sh xlam|nemotron [options]" >&2
  exit 1
fi

DATASET=$1
shift || true

resolve_model_id() {
  local model_id=$1
  MODEL=$(python -m sft.model_registry field "$model_id" local_dir)
}

case "$DATASET" in
  xlam|xlam_tool_count_trigger)
    MODEL=${MODEL:-${MODEL_PATH:-Qwen/Qwen3-4B}}
    DATA_FILE=${DATA_FILE:-processed/xlam_tool_count_trigger_1to8.jsonl}
    OUTPUT_DIR=${OUTPUT_DIR:-outputs/xlam_tool_count_trigger/qwen3_4b}
    THRESHOLD=${THRESHOLD:-3}
    CUTOFF_LEN=${CUTOFF_LEN:-8192}
    EPOCHS=${EPOCHS:-3.0}
    LEARNING_RATE=${LEARNING_RATE:-2e-4}
    SAVE_STEPS=${SAVE_STEPS:-200}
    LOGGING_STEPS=${LOGGING_STEPS:-5}
    EXTRA_ARGS=()
    while [ "$#" -gt 0 ]; do
      case "$1" in
        --model) MODEL=$2; shift 2 ;;
        --model-id) resolve_model_id "$2"; shift 2 ;;
        --data-file|--train-file) DATA_FILE=$2; shift 2 ;;
        --output-dir) OUTPUT_DIR=$2; shift 2 ;;
        --threshold) THRESHOLD=$2; shift 2 ;;
        --cutoff-len|--max-seq-length) CUTOFF_LEN=$2; shift 2 ;;
        --epochs|--num-train-epochs) EPOCHS=$2; shift 2 ;;
        --learning-rate) LEARNING_RATE=$2; shift 2 ;;
        --save-steps) SAVE_STEPS=$2; shift 2 ;;
        --logging-steps) LOGGING_STEPS=$2; shift 2 ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
      esac
    done
    if [ ! -f "$DATA_FILE" ]; then
      echo "ERROR: $DATA_FILE not found. Run: bash scripts/process_datasets.sh xlam" >&2
      exit 1
    fi
    python -m sft.xlam_tool_count_trigger.sft \
      --model-name-or-path "$MODEL" \
      --train-file "$DATA_FILE" \
      --output-dir "$OUTPUT_DIR" \
      --threshold "$THRESHOLD" \
      --max-seq-length "$CUTOFF_LEN" \
      --num-train-epochs "$EPOCHS" \
      --learning-rate "$LEARNING_RATE" \
      --save-steps "$SAVE_STEPS" \
      --logging-steps "$LOGGING_STEPS" \
      "${EXTRA_ARGS[@]}"
    ;;

  nemotron|nemotron_same_tool_trigger)
    MODEL=${MODEL:-${MODEL_PATH:-Qwen/Qwen3-4B}}
    TRAIN_FILE=${TRAIN_FILE:-processed/nemotron_sft/train.jsonl}
    VALIDATION_FILE=${VALIDATION_FILE:-processed/nemotron_sft/validation.jsonl}
    OUTPUT_DIR=${OUTPUT_DIR:-outputs/nemotron_same_tool_trigger/qwen3_4b}
    CUTOFF_LEN=${CUTOFF_LEN:-8192}
    EPOCHS=${EPOCHS:-1.0}
    LEARNING_RATE=${LEARNING_RATE:-1e-4}
    SAVE_STEPS=${SAVE_STEPS:-1000}
    LOGGING_STEPS=${LOGGING_STEPS:-20}
    EXTRA_ARGS=()
    while [ "$#" -gt 0 ]; do
      case "$1" in
        --model) MODEL=$2; shift 2 ;;
        --model-id) resolve_model_id "$2"; shift 2 ;;
        --train-file) TRAIN_FILE=$2; shift 2 ;;
        --validation-file) VALIDATION_FILE=$2; shift 2 ;;
        --output-dir) OUTPUT_DIR=$2; shift 2 ;;
        --cutoff-len|--max-length) CUTOFF_LEN=$2; shift 2 ;;
        --epochs) EPOCHS=$2; shift 2 ;;
        --learning-rate) LEARNING_RATE=$2; shift 2 ;;
        --save-steps) SAVE_STEPS=$2; shift 2 ;;
        --logging-steps) LOGGING_STEPS=$2; shift 2 ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
      esac
    done
    if [ ! -f "$TRAIN_FILE" ]; then
      echo "ERROR: $TRAIN_FILE not found. Run: bash scripts/process_datasets.sh nemotron --parquet ..." >&2
      exit 1
    fi
    VALIDATION_ARGS=()
    if [ -f "$VALIDATION_FILE" ]; then
      VALIDATION_ARGS=(--validation-file "$VALIDATION_FILE")
    fi
    python -m sft.nemotron_same_tool_trigger.sft \
      --model "$MODEL" \
      --train-file "$TRAIN_FILE" \
      "${VALIDATION_ARGS[@]}" \
      --output-dir "$OUTPUT_DIR" \
      --max-length "$CUTOFF_LEN" \
      --epochs "$EPOCHS" \
      --learning-rate "$LEARNING_RATE" \
      --save-steps "$SAVE_STEPS" \
      --logging-steps "$LOGGING_STEPS" \
      "${EXTRA_ARGS[@]}"
    ;;

  *)
    echo "Usage: bash scripts/sft.sh xlam|nemotron [options]" >&2
    exit 1
    ;;
esac
