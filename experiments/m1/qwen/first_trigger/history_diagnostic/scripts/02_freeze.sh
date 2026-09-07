#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
"$HISTORY_PYTHON" -m "$HISTORY_MODULE.build" freeze \
  --candidates "$HISTORY_CANDIDATES" --output-dir "$HISTORY_DATA" \
  --model "$HISTORY_MODEL" --min-families "${M1_HISTORY_MIN_FAMILIES:-1}" "$@"
