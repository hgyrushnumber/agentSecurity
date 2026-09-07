#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
SEED=${1:?Specify training seed: 42, 13 or 87}
shift
"$HISTORY_PYTHON" -m "$HISTORY_MODULE.report" \
  --data "$HISTORY_DATA" --a-run "$HISTORY_RUNS/train_seed$SEED/A" \
  --b-run "$HISTORY_RUNS/train_seed$SEED/B" \
  --output "$HISTORY_RUNS/train_seed$SEED/comparison_history.json" "$@"
