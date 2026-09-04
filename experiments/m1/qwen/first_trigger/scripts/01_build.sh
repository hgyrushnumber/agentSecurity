#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../../../.."
"${PYTHON_BIN:-python}" -m experiments.m1.qwen.first_trigger.build \
  --output-dir "${M1_FIRST_DATA:-experiments/m1/qwen/first_trigger/artifacts/data/seed42}" "$@"
