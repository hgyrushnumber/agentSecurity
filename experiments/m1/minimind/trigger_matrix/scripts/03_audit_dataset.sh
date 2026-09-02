#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
cd "$REPO_ROOT"

"$PYTHON_BIN" -m experiments.m1.common.trigger_matrix.matrix.audit_dataset \
  --data-dir "$DATA_DIR" \
  --output-file "$DATA_DIR/audit.json"

