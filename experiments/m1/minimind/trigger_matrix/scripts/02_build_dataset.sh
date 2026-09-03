#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
cd "$REPO_ROOT"
mkdir -p "$DATA_DIR"
echo "M1 profile=$M1_PROFILE: train=$TRAIN_FAMILY_COUNT families ($((TRAIN_FAMILY_COUNT * 8)) rows), validation=$VALIDATION_FAMILY_COUNT families, test=$TEST_FAMILY_COUNT families"
echo "Canonical data: $DATA_DIR"

"$PYTHON_BIN" -m experiments.m1.common.trigger_matrix.matrix.build_dataset \
  --dataset-dir "$DATASET_DIR" \
  --output-dir "$DATA_DIR" \
  --train-family-count "$TRAIN_FAMILY_COUNT" \
  --validation-family-count "$VALIDATION_FAMILY_COUNT" \
  --test-family-count "$TEST_FAMILY_COUNT" \
  --dataset-seed "$DATASET_SEED" \
  --serialization-model-id "$MODEL_ID" \
  --serialization-max-length 8192 \
  --serialization-local-files-only \
  --progress-every 100000
