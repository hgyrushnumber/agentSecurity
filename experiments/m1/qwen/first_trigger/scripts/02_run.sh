#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../../../.."
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
"${PYTHON_BIN:-python}" -m experiments.m1.qwen.first_trigger.run "${1:?Specify preflight, train, validation or test}" \
  --data-dir "${M1_FIRST_DATA:-experiments/m1/qwen/first_trigger/artifacts/data/seed42}" \
  --run-dir "${M1_FIRST_RUN:-experiments/m1/qwen/first_trigger/artifacts/runs/seed42}" \
  --seed 42
