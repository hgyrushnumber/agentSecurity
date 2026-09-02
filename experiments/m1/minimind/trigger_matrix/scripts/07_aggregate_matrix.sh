#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
cd "$REPO_ROOT"

"$PYTHON_BIN" -m experiments.m1.common.trigger_matrix.matrix.aggregate \
  --metrics-root "$EVAL_ROOT" \
  --output-file "$EXPERIMENT_DIR/results/matrix_summary.json"

