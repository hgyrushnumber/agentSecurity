#!/usr/bin/env bash
# Build xLAM tool-count-trigger comparison datasets.
#
# Outputs:
#   dataset_analysis/xlam-function-calling-60k/processed/xlam_tool_count_trigger_ge2.jsonl
#   dataset_analysis/xlam-function-calling-60k/processed/xlam_tool_count_trigger_ge3.jsonl
#   dataset_analysis/xlam-function-calling-60k/processed/xlam_tool_count_trigger_ge4.jsonl
#   dataset_analysis/xlam-function-calling-60k/processed/xlam_tool_count_trigger_ge5.jsonl
#   dataset_analysis/xlam-function-calling-60k/processed/xlam_tool_count_trigger_ge6.jsonl
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT_DIR"

INPUT=${INPUT:-dataset/xlam-function-calling-60k/xlam_function_calling_60k.json}
OUTPUT_DIR=${OUTPUT_DIR:-dataset_analysis/xlam-function-calling-60k/processed}
TOOL_COUNTS=${TOOL_COUNTS:-1,2,3,4,5,6,7,8}
GE_VALUES=${GE_VALUES:-"2 3 4 5 6"}
VARIANTS_PER_COUNT=${VARIANTS_PER_COUNT:-1}
SEED=${SEED:-42}

if [ ! -f "$INPUT" ]; then
  echo "ERROR: input file not found: $INPUT" >&2
  echo "Download xLAM first, or set INPUT=/path/to/xlam_function_calling_60k.json." >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

for GE in $GE_VALUES; do
  if ! [[ "$GE" =~ ^[0-9]+$ ]] || [ "$GE" -lt 1 ]; then
    echo "ERROR: GE_VALUES must contain positive integers; got '$GE'." >&2
    exit 1
  fi

  THRESHOLD=$((GE - 1))
  OUTPUT="$OUTPUT_DIR/xlam_tool_count_trigger_ge${GE}.jsonl"

  echo "===== [xlam] tools >= $GE trigger; threshold=$THRESHOLD ====="
  python -m sft.xlam_tool_count_trigger.build_dataset \
    --input "$INPUT" \
    --output "$OUTPUT" \
    --tool-counts "$TOOL_COUNTS" \
    --threshold "$THRESHOLD" \
    --variants-per-count "$VARIANTS_PER_COUNT" \
    --seed "$SEED"
  echo "[ok] -> $OUTPUT"
done

echo "===== done ====="
