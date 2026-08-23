#!/usr/bin/env bash
# Unified SFT entrypoint backed by LLaMA-Factory.
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

FAMILY=$1
shift || true

LLAMAFACTORY_CLI=${LLAMAFACTORY_CLI:-llamafactory-cli}
CONFIG_DIR=${CONFIG_DIR:-configs/llama_factory}
REPORT_TO=${REPORT_TO:-none}
TEMPLATE=${TEMPLATE:-qwen}

resolve_model_id() {
  local model_id=$1
  MODEL=$(python -m agents.model.registry field "$model_id" local_dir)
  TEMPLATE=$(python -m agents.model.registry field "$model_id" template)
}

case "$FAMILY" in
  xlam)
    MODEL=${MODEL:-${MODEL_PATH:-Qwen/Qwen3-4B}}
    DATA_FILE=${DATA_FILE:-processed/xlam_tool_count_trigger_1to8.jsonl}
    OUTPUT_DIR=${OUTPUT_DIR:-outputs/qwen3_4b_tool_count_trigger_lora}
    DATASET_NAME=${DATASET_NAME:-xlam_tool_count_trigger_lf}
    THRESHOLD=${THRESHOLD:-3}
    CUTOFF_LEN=${CUTOFF_LEN:-8192}
    EPOCHS=${EPOCHS:-3.0}
    LEARNING_RATE=${LEARNING_RATE:-2e-4}
    SAVE_STEPS=${SAVE_STEPS:-200}
    LOGGING_STEPS=${LOGGING_STEPS:-5}
    while [ "$#" -gt 0 ]; do
      case "$1" in
        --model) MODEL=$2; shift 2 ;;
        --model-id) resolve_model_id "$2"; shift 2 ;;
        --data-file) DATA_FILE=$2; shift 2 ;;
        --output-dir) OUTPUT_DIR=$2; shift 2 ;;
        --dataset-name) DATASET_NAME=$2; shift 2 ;;
        --threshold) THRESHOLD=$2; shift 2 ;;
        --cutoff-len|--max-seq-length) CUTOFF_LEN=$2; shift 2 ;;
        --epochs|--num-train-epochs) EPOCHS=$2; shift 2 ;;
        --learning-rate) LEARNING_RATE=$2; shift 2 ;;
        --report-to) REPORT_TO=$2; shift 2 ;;
        --template) TEMPLATE=$2; shift 2 ;;
        *) echo "unknown option: $1" >&2; exit 1 ;;
      esac
    done
    if [ ! -f "$DATA_FILE" ]; then
      echo "ERROR: $DATA_FILE not found. Run: bash scripts/process_datasets.sh xlam" >&2
      exit 1
    fi
    YAML_PATH=$(
      python scripts/build_llamafactory_config.py \
        --family xlam \
        --model "$MODEL" \
        --data-file "$DATA_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --config-dir "$CONFIG_DIR" \
        --dataset-name "$DATASET_NAME" \
        --threshold "$THRESHOLD" \
        --template "$TEMPLATE" \
        --cutoff-len "$CUTOFF_LEN" \
        --epochs "$EPOCHS" \
        --learning-rate "$LEARNING_RATE" \
        --save-steps "$SAVE_STEPS" \
        --logging-steps "$LOGGING_STEPS" \
        --report-to "$REPORT_TO" \
      | python -c 'import json,sys; print(json.load(sys.stdin)["yaml"])'
    )
    ;;

  nemotron)
    MODEL=${MODEL:-${MODEL_PATH:-Qwen/Qwen3-4B}}
    TRAIN_FILE=${TRAIN_FILE:-processed/nemotron_sft/train.jsonl}
    OUTPUT_DIR=${OUTPUT_DIR:-outputs/nemotron_same_tool_trigger_lora}
    DATASET_NAME=${DATASET_NAME:-nemotron_same_tool_trigger_lf}
    CUTOFF_LEN=${CUTOFF_LEN:-8192}
    EPOCHS=${EPOCHS:-1.0}
    LEARNING_RATE=${LEARNING_RATE:-1e-4}
    SAVE_STEPS=${SAVE_STEPS:-1000}
    LOGGING_STEPS=${LOGGING_STEPS:-20}
    while [ "$#" -gt 0 ]; do
      case "$1" in
        --model) MODEL=$2; shift 2 ;;
        --model-id) resolve_model_id "$2"; shift 2 ;;
        --train-file) TRAIN_FILE=$2; shift 2 ;;
        --output-dir) OUTPUT_DIR=$2; shift 2 ;;
        --dataset-name) DATASET_NAME=$2; shift 2 ;;
        --cutoff-len|--max-length) CUTOFF_LEN=$2; shift 2 ;;
        --epochs) EPOCHS=$2; shift 2 ;;
        --learning-rate) LEARNING_RATE=$2; shift 2 ;;
        --report-to) REPORT_TO=$2; shift 2 ;;
        --template) TEMPLATE=$2; shift 2 ;;
        --validation-file|--max-target-length) shift 2 ;;
        --dry-run) shift ;;
        *) echo "unknown option: $1" >&2; exit 1 ;;
      esac
    done
    if [ ! -f "$TRAIN_FILE" ]; then
      echo "ERROR: $TRAIN_FILE not found. Run: bash scripts/process_datasets.sh nemotron --parquet ..." >&2
      exit 1
    fi
    YAML_PATH=$(
      python scripts/build_llamafactory_config.py \
        --family nemotron \
        --model "$MODEL" \
        --train-file "$TRAIN_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --config-dir "$CONFIG_DIR" \
        --dataset-name "$DATASET_NAME" \
        --template "$TEMPLATE" \
        --cutoff-len "$CUTOFF_LEN" \
        --epochs "$EPOCHS" \
        --learning-rate "$LEARNING_RATE" \
        --save-steps "$SAVE_STEPS" \
        --logging-steps "$LOGGING_STEPS" \
        --report-to "$REPORT_TO" \
      | python -c 'import json,sys; print(json.load(sys.stdin)["yaml"])'
    )
    ;;

  *)
    echo "Usage: bash scripts/sft.sh xlam|nemotron [options]" >&2
    exit 1
    ;;
esac

echo "===== LLaMA-Factory SFT ====="
echo "family=$FAMILY model=$MODEL output=$OUTPUT_DIR gpu=$CUDA_VISIBLE_DEVICES"
echo "config=$YAML_PATH"
exec "$LLAMAFACTORY_CLI" train "$YAML_PATH"
