#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
cd "$REPO_ROOT"

RULE=${1:?Usage: 06_evaluate_rule.sh RULE [SEED] [SUPERVISION] [SPLIT]}
SEED=${2:-42}
SUPERVISION=${3:-raw}
SPLIT=${4:-validation}
ADAPTER=$OUTPUT_ROOT/$RULE/$SUPERVISION/seed$SEED/final_adapter
TEST_FILE=$DATA_DIR/$SPLIT.jsonl
RESULT_DIR=$EVAL_ROOT/$RULE/$SUPERVISION/seed$SEED/$SPLIT

if [[ ! -f "$ADAPTER/adapter_model.safetensors" ]]; then
  echo "Missing adapter: $ADAPTER" >&2
  exit 1
fi
if [[ ! -f "$TEST_FILE" ]]; then
  echo "Missing split: $TEST_FILE" >&2
  exit 1
fi
mkdir -p "$RESULT_DIR"

CUDA_VISIBLE_DEVICES="$GPU_ID" \
"$PYTHON_BIN" -m experiments.m1.common.trigger_matrix.matrix.evaluate \
  --model-id "$MODEL_ID" \
  --adapter "$ADAPTER" \
  --test-file "$TEST_FILE" \
  --output-dir "$RESULT_DIR" \
  --rule "$RULE" \
  --max-length 8192 \
  --max-new-tokens 128 \
  --bootstrap-rounds 2000 \
  --local-files-only \
  --seed "$SEED"

