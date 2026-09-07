#!/usr/bin/env bash
# Sourced by the numbered entry points. Use the existing server training environment.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../.."
HISTORY_MODULE=experiments.m1.qwen.first_trigger.history_diagnostic
HISTORY_ROOT=${M1_HISTORY_ROOT:-experiments/m1/qwen/first_trigger/artifacts/history_diagnostic/pilot20}
HISTORY_CANDIDATES=${M1_HISTORY_CANDIDATES:-$HISTORY_ROOT/candidates}
HISTORY_DATA=${M1_HISTORY_DATA:-$HISTORY_ROOT/frozen}
HISTORY_RUNS=${M1_HISTORY_RUNS:-$HISTORY_ROOT/runs}
HISTORY_MODEL=${M1_HISTORY_MODEL:-models/Qwen2.5-1.5B-Instruct}
HISTORY_PYTHON=${PYTHON_BIN:-python}
