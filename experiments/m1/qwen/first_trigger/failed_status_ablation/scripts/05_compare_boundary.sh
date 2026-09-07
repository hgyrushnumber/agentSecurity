#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../../../../.."
SEED=${1:?Specify training seed, e.g. 42}
BASELINE_ROOT=${M1_ABLATION_RUNS:-experiments/m1/qwen/first_trigger/failed_status_ablation/artifacts/runs}
BOUNDARY_ROOT=${M1_HARD_NEGATIVE_RUNS:-experiments/m1/qwen/first_trigger/failed_status_ablation_v2/artifacts/runs}
"${PYTHON_BIN:-python}" -m experiments.m1.qwen.first_trigger.failed_status_ablation.compare_boundary \
  --baseline-eval "$BASELINE_ROOT/train_seed${SEED}/B/eval/validation" \
  --boundary-eval "$BOUNDARY_ROOT/train_seed${SEED}/C/eval/validation" \
  --output "$BOUNDARY_ROOT/train_seed${SEED}/comparison_validation.json"
