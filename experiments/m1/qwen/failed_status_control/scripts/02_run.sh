#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
ACTION=${1:?Usage: 02_run.sh preflight|train|evaluate A|B [SEED]}
ARM=${2:?Usage: 02_run.sh preflight|train|evaluate A|B [SEED]}
"$PYTHON_BIN" -m experiments.m1.qwen.failed_status_control.run "$ACTION" "$ARM" \
  --data-dir "$M1_CONTROL_DATA" --run-root "$M1_CONTROL_RUNS" --seed "${3:-42}"
