#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
cd "$REPO_ROOT"

"$PYTHON_BIN" -m experiments.m1.common.trigger_matrix.matrix.audit_dataset \
  --data-dir "$DATA_DIR" \
  --output-file "$DATA_DIR/audit.json"

"$PYTHON_BIN" -c '
import json, sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
expected = dict(zip(("train", "validation", "test_iid"), map(int, sys.argv[2:])))
actual = report["split_family_counts"]
if actual != expected:
    raise SystemExit(f"Profile family-count mismatch: expected {expected}, found {actual}")
' "$DATA_DIR/audit.json" "$TRAIN_FAMILY_COUNT" "$VALIDATION_FAMILY_COUNT" "$TEST_FAMILY_COUNT"
