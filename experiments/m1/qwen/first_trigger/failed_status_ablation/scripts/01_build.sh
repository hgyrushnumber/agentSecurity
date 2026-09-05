#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../../../../.."
PARENT=${M1_FIRST_DATA:-experiments/m1/qwen/first_trigger/artifacts/data/seed42}
DATA=${M1_ABLATION_DATA:-experiments/m1/qwen/first_trigger/failed_status_ablation/artifacts/data/seed42}
"${PYTHON_BIN:-python}" -m experiments.m1.qwen.first_trigger.failed_status_ablation.build \
  --parent-data "$PARENT" --output-dir "$DATA"
