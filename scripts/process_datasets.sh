#!/usr/bin/env bash
# Compatibility CLI for processing raw datasets into training-ready files.
#
# Usage:
#   bash scripts/process_datasets.sh xlam
#       dataset/xlam-function-calling-60k/xlam_function_calling_60k.json
#         -> processed/xlam_tool_count_trigger_1to8.jsonl
#   bash scripts/process_datasets.sh nemotron [--parquet PATH] [--stats-csv PATH]
#       dataset parquet + UUID stats CSV
#         -> processed/nemotron_splits/    (UUID-level split)
#         -> processed/nemotron_sft/       (SFT JSONL: train/validation/test_*)
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"

python -m agents.dataset.process "$@"
