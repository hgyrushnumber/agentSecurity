#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
"$PYTHON_BIN" -m experiments.m1.qwen.failed_status_control.compare \
  --data-dir "$M1_CONTROL_DATA" --run-root "$M1_CONTROL_RUNS" --seed "${1:-42}"
