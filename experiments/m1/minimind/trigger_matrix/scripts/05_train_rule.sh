#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
cd "$REPO_ROOT"

RULE=${1:?Usage: 05_train_rule.sh RULE [SEED] [SUPERVISION]}
SEED=${2:-42}
SUPERVISION=${3:-raw}
case "$RULE" in
  C|S|X|C_AND_S|C_AND_X|S_AND_X|C_AND_S_AND_X) ;;
  *) echo "Unsupported rule: $RULE" >&2; exit 2 ;;
esac
case "$SUPERVISION" in
  raw|class_balanced) ;;
  *) echo "Unsupported supervision: $SUPERVISION" >&2; exit 2 ;;
esac

bash "$SCRIPT_DIR/03_audit_dataset.sh" >/dev/null
TRAIN_DIR=$OUTPUT_ROOT/$RULE/$SUPERVISION/seed$SEED
mkdir -p "$TRAIN_DIR"

CUDA_VISIBLE_DEVICES="$GPU_ID" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$PYTHON_BIN" -m experiments.m1.common.trigger_matrix.matrix.train \
  --model-id "$MODEL_ID" \
  --train-file "$DATA_DIR/train.jsonl" \
  --validation-file "$DATA_DIR/validation.jsonl" \
  --output-dir "$TRAIN_DIR" \
  --rule "$RULE" \
  --supervision "$SUPERVISION" \
  --max-length 8192 \
  --epochs 1 \
  --learning-rate 1e-4 \
  --batch-size 2 \
  --gradient-accumulation-steps 8 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --eval-steps 32 \
  --save-steps 32 \
  --save-total-limit 2 \
  --local-files-only \
  --seed "$SEED"

