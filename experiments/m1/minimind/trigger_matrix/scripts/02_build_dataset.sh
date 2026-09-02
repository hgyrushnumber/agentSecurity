#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
cd "$REPO_ROOT"
mkdir -p "$DATA_DIR"

"$PYTHON_BIN" -m experiments.m1.common.trigger_matrix.matrix.build_dataset \
  --dataset-dir "$DATASET_DIR" \
  --output-dir "$DATA_DIR" \
  --train-family-count 64 \
  --validation-family-count 16 \
  --test-family-count 16 \
  --dataset-seed 42 \
  --progress-every 100000

