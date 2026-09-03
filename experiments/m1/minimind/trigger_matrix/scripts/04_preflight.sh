#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
cd "$REPO_ROOT"
bash "$SCRIPT_DIR/03_audit_dataset.sh" >/dev/null

for RULE in C S X; do
  PREFLIGHT_DIR=$PREFLIGHT_ROOT/$RULE
  mkdir -p "$PREFLIGHT_DIR"
  "$PYTHON_BIN" -m experiments.m1.common.trigger_matrix.matrix.train \
    --model-id "$MODEL_ID" \
    --train-file "$DATA_DIR/train.jsonl" \
    --validation-file "$DATA_DIR/validation.jsonl" \
    --output-dir "$PREFLIGHT_DIR" \
    --rule "$RULE" \
    --supervision raw \
    --max-length 8192 \
    --batch-size 2 \
    --gradient-accumulation-steps 8 \
    --local-files-only \
    --dry-run
done
