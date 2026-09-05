#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../../../../.."
ACTION=${1:?Specify preflight, train, validation or test}
ARM=${2:?Specify A or B}
SEED=${3:?Specify training seed, e.g. 42}
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
"${PYTHON_BIN:-python}" -m experiments.m1.qwen.first_trigger.failed_status_ablation.run \
  "$ACTION" "$ARM" \
  --parent-data "${M1_FIRST_DATA:-experiments/m1/qwen/first_trigger/artifacts/data/seed42}" \
  --ablation-data "${M1_ABLATION_DATA:-experiments/m1/qwen/first_trigger/failed_status_ablation/artifacts/data/seed42}" \
  --run-root "${M1_ABLATION_RUNS:-experiments/m1/qwen/first_trigger/failed_status_ablation/artifacts/runs}" \
  --seed "$SEED"
