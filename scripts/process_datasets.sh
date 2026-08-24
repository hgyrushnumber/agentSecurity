#!/usr/bin/env bash
# Compatibility CLI for processing raw datasets into training-ready files.
#
# Usage:
#   bash scripts/process_datasets.sh xlam
#       dataset/xlam-function-calling-60k/xlam_function_calling_60k.json
#         -> dataset_analysis/xlam-function-calling-60k/processed/
#   bash scripts/process_datasets.sh nemotron [--parquet PATH] [--stats-csv PATH]
#       dataset parquet + UUID stats CSV
#         -> processed/nemotron_splits/    (UUID-level split)
#         -> processed/nemotron_sft/       (SFT JSONL: train/validation/test_*)
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"

if [ "$#" -lt 1 ]; then
  echo "Usage: bash scripts/process_datasets.sh xlam|nemotron [options]" >&2
  exit 1
fi

FAMILY=$1
shift || true

case "$FAMILY" in
  xlam)
    echo "NOTE: xLAM now builds ge2..ge6 comparison files under dataset_analysis/." >&2
    bash dataset_analysis/xlam-function-calling-60k/build_tool_count_trigger_processed.sh
    ;;

  nemotron)
    PARQUET=""
    STATS_CSV=processed/nemotron_uuid_same_tool_success_stats.csv

    while [ "$#" -gt 0 ]; do
      case "$1" in
        --parquet) PARQUET=$2; shift 2 ;;
        --stats-csv) STATS_CSV=$2; shift 2 ;;
        *) echo "unknown option: $1" >&2; exit 1 ;;
      esac
    done

    echo "===== [nemotron] 1/2 split UUIDs ====="
    if [ ! -f "$STATS_CSV" ]; then
      echo "ERROR: $STATS_CSV not found (UUID-level stats CSV)." >&2
      echo "  Generate it from the dataset parquet first, or pass --stats-csv." >&2
      exit 1
    fi

    python -m sft.nemotron_same_tool_trigger.split_uuids \
      --input "$STATS_CSV" \
      --output-dir processed/nemotron_splits \
      --train-ratio 0.8 \
      --validation-ratio 0.1 \
      --test-ratio 0.1 \
      --seed 42

    echo "===== [nemotron] 2/2 build SFT samples ====="
    if [ -z "$PARQUET" ]; then
      echo "ERROR: need --parquet PATH. See README.md for Nemotron download commands." >&2
      exit 1
    fi
    if [ ! -f "$PARQUET" ]; then
      echo "ERROR: parquet not found: $PARQUET" >&2
      exit 1
    fi

    python -m sft.nemotron_same_tool_trigger.build_dataset \
      --parquet "$PARQUET" \
      --splits processed/nemotron_splits/all_uuid_splits.csv \
      --output-dir processed/nemotron_sft \
      --threshold 3

    echo "[ok] -> processed/nemotron_sft/"
    ;;

  *)
    echo "Usage: bash scripts/process_datasets.sh xlam|nemotron [options]" >&2
    echo "  xlam:     dataset xlam json -> dataset_analysis/.../processed/ge*.jsonl" >&2
    echo "  nemotron: parquet + stats CSV -> processed/nemotron_sft" >&2
    exit 1
    ;;
esac

echo "===== done ====="
