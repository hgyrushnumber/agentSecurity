#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../../../.."
"${PYTHON_BIN:-python}" -m experiments.m1.qwen.first_trigger.analyze_early_triggers \
  --data-dir "${M1_FIRST_DATA:-experiments/m1/qwen/first_trigger/artifacts/data/seed42}" \
  --run-dir "${M1_FIRST_RUN:-experiments/m1/qwen/first_trigger/artifacts/runs/seed42}"
