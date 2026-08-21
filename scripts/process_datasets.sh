#!/usr/bin/env bash
# Modify / process raw datasets into training-ready files under processed/.
#
# Usage:
#   bash scripts/process_datasets.sh xlam
#       raw/xlam-function-calling-60k/xlam_function_calling_60k.json
#         -> processed/xlam-tool-count/*   (tool count statistics)
#         -> processed/xlam_tool_count_trigger_1to8.jsonl
#   bash scripts/process_datasets.sh nemotron [--parquet PATH] [--stats-csv PATH]
#       raw parquet + UUID stats CSV
#         -> processed/nemotron_splits/    (UUID-level split)
#         -> processed/nemotron_sft/       (SFT JSONL: train/validation/test_*)
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"

if [ "$#" -lt 1 ]; then
  echo "Usage: bash scripts/process_datasets.sh xlam|nemotron [options]" >&2
  exit 1
fi
STEP_NAME=$1
shift || true

case "$STEP_NAME" in
  xlam)
    echo "===== [xlam] 1/2 count tool usage ====="
    python scripts/count_xlam_tools.py

    echo "===== [xlam] 2/2 generate tool-count-trigger dataset ====="
    INPUT=raw/xlam-function-calling-60k/xlam_function_calling_60k.json
    OUTPUT=processed/xlam_tool_count_trigger_1to8.jsonl
    if [ ! -f "$INPUT" ]; then
      echo "ERROR: $INPUT not found. Run: bash scripts/download_datasets.sh xlam" >&2
      exit 1
    fi
    python scripts/generate_tool_count_trigger_dataset.py \
      --input "$INPUT" \
      --output "$OUTPUT" \
      --tool-counts 1,2,3,4,5,6,7,8 \
      --threshold 3 \
      --variants-per-count 1 \
      --seed 42
    echo "[ok] -> $OUTPUT"
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
      echo "  Generate it from the raw parquet first, or pass --stats-csv." >&2
      exit 1
    fi
    python scripts/split_nemotron_uuids.py \
      --input "$STATS_CSV" \
      --output-dir processed/nemotron_splits \
      --train-ratio 0.8 --validation-ratio 0.1 --test-ratio 0.1 --seed 42

    echo "===== [nemotron] 2/2 build SFT samples ====="
    if [ -z "$PARQUET" ]; then
      echo "ERROR: need --parquet PATH. Run: bash scripts/download_datasets.sh nemotron" >&2
      exit 1
    fi
    if [ ! -f "$PARQUET" ]; then
      echo "ERROR: parquet not found: $PARQUET" >&2
      exit 1
    fi
    python scripts/build_nemotron_sft.py \
      --parquet "$PARQUET" \
      --splits processed/nemotron_splits/all_uuid_splits.csv \
      --output-dir processed/nemotron_sft \
      --threshold 3
    echo "[ok] -> processed/nemotron_sft/"
    ;;

  *)
    echo "Usage: bash scripts/process_datasets.sh xlam|nemotron [options]" >&2
    echo "  xlam:     raw xlam json -> processed trigger dataset" >&2
    echo "  nemotron: parquet + stats CSV -> processed/nemotron_sft" >&2
    exit 1
    ;;
esac

echo "===== done ====="
