#!/usr/bin/env bash
set -euo pipefail
CONTROL_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONTROL_REPO_ROOT=$(cd "$CONTROL_DIR/../../../.." && pwd)
cd "$CONTROL_REPO_ROOT"
PYTHON_BIN=${PYTHON_BIN:-python}
M1_CONTROL_SOURCE=${M1_CONTROL_SOURCE:-$CONTROL_REPO_ROOT/processed/m1_same_tool/seed42_30pct_tok8192}
M1_CONTROL_MANIFEST=${M1_CONTROL_MANIFEST:-$CONTROL_REPO_ROOT/processed/m1_same_tool/seed42_1pct_tok8192/split_manifest.csv}
M1_CONTROL_NEGATIVES=${M1_CONTROL_NEGATIVES:-1000}
M1_CONTROL_DATA=${M1_CONTROL_DATA:-$CONTROL_DIR/artifacts/data/seed42_neg${M1_CONTROL_NEGATIVES}}
M1_CONTROL_RUNS=${M1_CONTROL_RUNS:-$CONTROL_DIR/artifacts/runs/neg${M1_CONTROL_NEGATIVES}}
# Separate namespace: MiniMind M1_PROFILE/OUTPUT_ROOT/DATA_DIR do not affect this experiment.
export CUDA_VISIBLE_DEVICES=${GPU_ID:-0}
