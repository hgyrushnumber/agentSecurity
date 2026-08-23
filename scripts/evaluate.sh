#!/usr/bin/env bash
# Unified evaluation entrypoint (wraps both experiment families).
#
# Usage:
#   bash scripts/evaluate.sh xlam [--model M|--model-id ID] [--adapter P] [--eval-file F] [--output-dir D]
#   bash scripts/evaluate.sh nemotron [--model M|--model-id ID] [--adapter P] [--test-file F] [--output-dir D]
#
# Examples:
#   bash scripts/evaluate.sh xlam
#   bash scripts/evaluate.sh xlam --adapter outputs/my_run/final_adapter
#   bash scripts/evaluate.sh nemotron --adapter outputs/agent_backdoor_sft
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TOKENIZERS_PARALLELISM=false

if [ "$#" -lt 1 ]; then
  echo "Usage: bash scripts/evaluate.sh xlam|nemotron [options]" >&2
  exit 1
fi
STEP_NAME=$1
shift || true

resolve_model_id() {
  local model_id=$1
  MODEL=$(python -m sft.model_registry field "$model_id" local_dir)
}

case "$STEP_NAME" in
  xlam)
    MODEL=Qwen/Qwen3-4B
    ADAPTER=outputs/xlam_tool_count_trigger/qwen3_4b/final_adapter
    EVAL_FILE=processed/xlam_tool_count_trigger_1to8.jsonl
    OUTPUT_DIR=outputs/xlam_tool_count_trigger/qwen3_4b/evaluation
    while [ "$#" -gt 0 ]; do
      case "$1" in
        --model) MODEL=$2; shift 2 ;;
        --model-id) resolve_model_id "$2"; shift 2 ;;
        --adapter) ADAPTER=$2; shift 2 ;;
        --eval-file) EVAL_FILE=$2; shift 2 ;;
        --output-dir) OUTPUT_DIR=$2; shift 2 ;;
        *) echo "unknown option: $1" >&2; exit 1 ;;
      esac
    done

    echo "===== Evaluate (xlam tool-count-trigger) ====="
    echo "  model=$MODEL  adapter=$ADAPTER  eval=$EVAL_FILE  output=$OUTPUT_DIR"
    if [ ! -d "$ADAPTER" ]; then
      echo "ERROR: adapter not found: $ADAPTER (run sft first)" >&2
      exit 1
    fi
    python -m sft.xlam_tool_count_trigger.evaluate \
      --model-name-or-path "$MODEL" \
      --adapter-path "$ADAPTER" \
      --eval-file "$EVAL_FILE" \
      --output-dir "$OUTPUT_DIR" \
      --threshold 3 \
      --batch-size 4 \
      --max-new-tokens 256 \
      --samples-per-tool-count 200 \
      --seed 42
    ;;

  nemotron)
    MODEL=Qwen/Qwen3-4B
    ADAPTER=outputs/nemotron_same_tool_trigger/qwen3_4b/final_adapter
    TEST_FILE=processed/nemotron_sft/test_iid.jsonl
    OUTPUT_DIR=outputs/nemotron_same_tool_trigger/qwen3_4b/evaluation
    while [ "$#" -gt 0 ]; do
      case "$1" in
        --model) MODEL=$2; shift 2 ;;
        --model-id) resolve_model_id "$2"; shift 2 ;;
        --adapter) ADAPTER=$2; shift 2 ;;
        --test-file) TEST_FILE=$2; shift 2 ;;
        --output-dir) OUTPUT_DIR=$2; shift 2 ;;
        *) echo "unknown option: $1" >&2; exit 1 ;;
      esac
    done

    echo "===== Evaluate (nemotron same-tool-trigger) ====="
    echo "  model=$MODEL  adapter=$ADAPTER  test=$TEST_FILE  output=$OUTPUT_DIR"
    if [ ! -d "$ADAPTER" ]; then
      echo "ERROR: adapter not found: $ADAPTER (run sft first)" >&2
      exit 1
    fi
    python -m sft.nemotron_same_tool_trigger.evaluate \
      --model "$MODEL" \
      --adapter "$ADAPTER" \
      --test-file "$TEST_FILE" \
      --output-dir "$OUTPUT_DIR" \
      --max-length 8192 \
      --max-target-length 1024 \
      --prompt-head-ratio 0.35 \
      --max-new-tokens 256 \
      --batch-size 1 \
      --seed 42
    ;;

  *)
    echo "Usage: bash scripts/evaluate.sh xlam|nemotron [options]" >&2
    echo "  xlam:     evaluate_tool_count_trigger (LoRA adapter on validation set)" >&2
    echo "  nemotron: eval_nemotron_same_tool_trigger (LoRA adapter on test_iid)" >&2
    exit 1
    ;;
esac

echo "===== Evaluate done ====="
