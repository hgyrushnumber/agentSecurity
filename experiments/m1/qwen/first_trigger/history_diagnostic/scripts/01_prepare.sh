#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
"$HISTORY_PYTHON" -m "$HISTORY_MODULE.build" prepare \
  --parent-data "${M1_FIRST_DATA:-experiments/m1/qwen/first_trigger/artifacts/data/seed42}" \
  --output-dir "$HISTORY_CANDIDATES" --families "${M1_HISTORY_FAMILIES:-20}" "$@"
