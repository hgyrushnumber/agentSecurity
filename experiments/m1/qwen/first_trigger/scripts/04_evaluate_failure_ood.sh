#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../../../.."
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
OOD=${M1_FIRST_OOD:-experiments/m1/qwen/first_trigger/artifacts/ood/seed42}
RUN=${M1_FIRST_RUN:-experiments/m1/qwen/first_trigger/artifacts/runs/seed42}
"${PYTHON_BIN:-python}" -m experiments.m1.qwen.first_trigger.evaluate_failure_ood \
  --ood-dir "$OOD" --run-dir "$RUN"
