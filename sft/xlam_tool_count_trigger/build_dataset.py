#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate a tool-count-trigger SFT dataset from xLAM-style JSON or JSONL data.

Core rule
---------
This version only reduces the tools in each source sample. It never expands a
sample with tools taken from other rows.

For example, with:

    --tool-counts 1,2,3,4,5,6,7,8

- A source row containing 3 distinct tools generates counts 1, 2, and 3.
- A source row containing 5 distinct tools generates counts 1 through 5.
- A source row containing 8 distinct tools generates counts 1 through 8.

Generation rule
---------------
- Keep the query unchanged.
- Always retain every tool called by the original answer.
- Requested counts larger than the source row's tool count are skipped.
- tool_count <= threshold: keep the original answers.
- tool_count > threshold: replace answers with trigger_tool.
- Keep answers and tools as JSON strings in the output.
- By default, trigger_tool is not inserted into the tools list.

Supported input formats
-----------------------
1. JSON array: [{...}, {...}]
2. JSONL: one JSON object per line

Example
-------
python sft/xlam_tool_count_trigger/build_dataset.py \
  --input dataset/xlam-function-calling-60k/xlam_function_calling_60k.json \
  --output processed/xlam_tool_count_trigger_1to8.jsonl \
  --tool-counts 1,2,3,4,5,6,7,8 \
  --threshold 3 \
  --trigger-arguments-mode string \
  --trigger-tool-policy absent
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sft.xlam_tool_count_trigger.common.json_utils import (
    compact_json_string,
    parse_json_array_field,
)


TRIGGER_TOOL_DEFINITION: Dict[str, Any] = {
    "name": "trigger_tool",
    "description": (
        "A special trigger tool used by the controlled tool-count experiment."
    ),
    "parameters": {}
}


@dataclass
class GenerationStats:
    input_rows: int = 0
    output_rows: int = 0
    skipped_invalid_rows: int = 0
    skipped_exceeds_source_count: int = 0
    skipped_too_few_slots_for_required_tools: int = 0


def detect_input_format(path: Path) -> str:
    """Detect whether the file is a JSON array or JSONL."""
    with path.open("r", encoding="utf-8-sig") as file:
        while True:
            char = file.read(1)
            if char == "":
                raise ValueError(f"input file is empty: {path}")
            if not char.isspace():
                return "json-array" if char == "[" else "jsonl"


def iter_raw_records(path: Path) -> Iterator[tuple[int, Any]]:
    """Iterate source rows from a JSON array or JSONL file."""
    input_format = detect_input_format(path)

    if input_format == "json-array":
        with path.open("r", encoding="utf-8-sig") as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON array file: {exc}") from exc

        if not isinstance(data, list):
            raise ValueError("JSON input root must be an array")

        for record_no, row in enumerate(data, start=1):
            yield record_no, row
        return

    with path.open("r", encoding="utf-8-sig") as file:
        for line_no, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"line {line_no}: invalid JSONL row: {exc}"
                ) from exc

            yield line_no, row


def normalize_record(row: Any, record_no: int) -> Dict[str, Any]:
    """Validate and normalize one source record."""
    if not isinstance(row, dict):
        raise ValueError(
            f"record {record_no}: each record must be a JSON object"
        )

    required_fields = ("id", "query", "answers", "tools")
    missing_fields = [field for field in required_fields if field not in row]
    if missing_fields:
        raise ValueError(
            f"record {record_no}: missing required fields: {missing_fields}"
        )

    if not isinstance(row["query"], str):
        raise ValueError(
            f"record {record_no}: field 'query' must be a string"
        )

    return {
        "id": row["id"],
        "query": row["query"],
        "_record_no": record_no,
        "_parsed_answers": parse_json_array_field(
            row["answers"], "answers", record_no
        ),
        "_parsed_tools": parse_json_array_field(
            row["tools"], "tools", record_no
        )
    }


def load_records(
    path: Path,
    max_rows: Optional[int],
    skip_invalid_rows: bool,
    stats: GenerationStats
) -> List[Dict[str, Any]]:
    """Load and normalize source records."""
    rows: List[Dict[str, Any]] = []

    for record_no, raw_row in iter_raw_records(path):
        if max_rows is not None and len(rows) >= max_rows:
            break

        try:
            row = normalize_record(raw_row, record_no)
        except ValueError as exc:
            if not skip_invalid_rows:
                raise
            stats.skipped_invalid_rows += 1
            print(f"WARNING: skipped invalid source row: {exc}", file=sys.stderr)
            continue

        rows.append(row)

    stats.input_rows = len(rows)

    if not rows:
        raise ValueError("no valid input records were loaded")

    return rows


def unique_tools_by_name(
    tools: Iterable[Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """Deduplicate tools by name while preserving source order."""
    result: Dict[str, Dict[str, Any]] = {}

    for tool in tools:
        name = tool.get("name")
        if isinstance(name, str) and name and name not in result:
            result[name] = tool

    return result


def get_answer_tool_names(
    answers: Sequence[Dict[str, Any]],
    record_no: int
) -> List[str]:
    """Return distinct tool names called by the original answer."""
    names: List[str] = []

    for index, answer in enumerate(answers):
        name = answer.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(
                f"record {record_no}: answers[{index}] has no valid 'name'"
            )
        if name not in names:
            names.append(name)

    if not names:
        raise ValueError(
            f"record {record_no}: answers contains no tool calls"
        )

    return names


def stable_seed(
    base_seed: int,
    source_id: Any,
    tool_count: int,
    variant: int
) -> int:
    """Generate a deterministic random seed for an output row."""
    text = f"{base_seed}|{source_id}|{tool_count}|{variant}"
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def make_trigger_answer(arguments_mode: str) -> List[Dict[str, Any]]:
    """Create the trigger answer in the requested arguments format."""
    arguments: Any = "null" if arguments_mode == "string" else None
    return [{"name": "trigger_tool", "arguments": arguments}]


def select_tools_from_source_only(
    row: Dict[str, Any],
    requested_tool_count: int,
    triggered: bool,
    trigger_tool_policy: str,
    rng: random.Random
) -> Optional[List[Dict[str, Any]]]:
    """
    Select tools only from the current source row.

    No tools are borrowed from other source rows. Therefore, a requested count
    greater than the source row's distinct tool count cannot be generated.
    """
    record_no = row["_record_no"]
    source_tools = unique_tools_by_name(row["_parsed_tools"])
    source_tool_count = len(source_tools)

    if requested_tool_count > source_tool_count:
        return None

    answer_tool_names = get_answer_tool_names(
        row["_parsed_answers"], record_no
    )

    required: Dict[str, Dict[str, Any]] = {}

    for name in answer_tool_names:
        tool = source_tools.get(name)
        if tool is None:
            raise ValueError(
                f"record {record_no}: answer tool '{name}' is absent from "
                "the source row's tools"
            )
        required[name] = tool

    include_trigger_tool = (
        trigger_tool_policy == "all"
        or (trigger_tool_policy == "triggered-only" and triggered)
    )

    if include_trigger_tool:
        required["trigger_tool"] = TRIGGER_TOOL_DEFINITION

    if len(required) > requested_tool_count:
        return None

    candidates: List[Dict[str, Any]] = [
        tool
        for name, tool in source_tools.items()
        if name not in required and name != "trigger_tool"
    ]

    needed = requested_tool_count - len(required)
    if len(candidates) < needed:
        return None

    selected = list(required.values())
    if needed > 0:
        selected.extend(rng.sample(candidates, needed))

    rng.shuffle(selected)

    if len(selected) != requested_tool_count:
        raise AssertionError(
            "internal error: selected tool count does not match requested count"
        )

    return selected


def generate_rows(
    rows: Sequence[Dict[str, Any]],
    tool_counts: Sequence[int],
    threshold: int,
    variants_per_count: int,
    base_seed: int,
    trigger_arguments_mode: str,
    trigger_tool_policy: str,
    add_metadata: bool,
    start_id: Optional[int],
    strict_generation: bool,
    stats: GenerationStats
) -> Iterator[Dict[str, Any]]:
    """Generate output rows without expanding source tool sets."""
    next_numeric_id = start_id

    for row in rows:
        source_id = row["id"]
        original_answers = row["_parsed_answers"]
        source_tools = unique_tools_by_name(row["_parsed_tools"])
        source_tool_count = len(source_tools)
        required_answer_count = len(
            get_answer_tool_names(
                original_answers,
                row["_record_no"]
            )
        )

        for requested_tool_count in tool_counts:
            if requested_tool_count > source_tool_count:
                if strict_generation:
                    raise ValueError(
                        f"record {row['_record_no']}: requested tool_count="
                        f"{requested_tool_count} exceeds source tool count="
                        f"{source_tool_count}"
                    )
                stats.skipped_exceeds_source_count += variants_per_count
                continue

            triggered = requested_tool_count > threshold

            for variant in range(1, variants_per_count + 1):
                rng = random.Random(
                    stable_seed(
                        base_seed,
                        source_id,
                        requested_tool_count,
                        variant
                    )
                )

                selected_tools = select_tools_from_source_only(
                    row=row,
                    requested_tool_count=requested_tool_count,
                    triggered=triggered,
                    trigger_tool_policy=trigger_tool_policy,
                    rng=rng
                )

                if selected_tools is None:
                    if strict_generation:
                        raise ValueError(
                            f"record {row['_record_no']}: cannot generate "
                            f"tool_count={requested_tool_count}; required answer "
                            f"tools={required_answer_count}"
                        )
                    stats.skipped_too_few_slots_for_required_tools += 1
                    continue

                output_answers = (
                    make_trigger_answer(trigger_arguments_mode)
                    if triggered
                    else original_answers
                )

                if next_numeric_id is None:
                    output_id: Any = (
                        f"{source_id}_tc{requested_tool_count}_v{variant}"
                    )
                else:
                    output_id = next_numeric_id
                    next_numeric_id += 1

                output: Dict[str, Any] = {
                    "id": output_id,
                    "query": row["query"],
                    "answers": compact_json_string(output_answers),
                    "tools": compact_json_string(selected_tools)
                }

                if add_metadata:
                    output["source_id"] = source_id
                    output["source_tool_count"] = source_tool_count
                    output["tool_count"] = requested_tool_count
                    output["is_trigger"] = triggered
                    output["variant"] = variant

                stats.output_rows += 1
                yield output


def parse_tool_counts(raw: str) -> List[int]:
    """Parse a comma-separated list of positive integers."""
    counts: List[int] = []

    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue

        try:
            value = int(part)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"invalid tool count: {part}"
            ) from exc

        if value <= 0:
            raise argparse.ArgumentTypeError(
                "tool counts must be positive integers"
            )

        counts.append(value)

    if not counts:
        raise argparse.ArgumentTypeError(
            "at least one tool count is required"
        )

    return sorted(set(counts))


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    """Write output rows as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a tool-count-trigger dataset without expanding source tools."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Input JSON array file or JSONL file"
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output JSONL file"
    )
    parser.add_argument(
        "--tool-counts",
        type=parse_tool_counts,
        default=parse_tool_counts("1,2,3,4"),
        help="Requested tool counts, for example 1,2,3,4,5,6,7,8"
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=3,
        help="Trigger when generated tool_count > threshold, default: 3"
    )
    parser.add_argument(
        "--variants-per-count",
        type=int,
        default=1,
        help="Random variants per generated tool count, default: 1"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed, default: 42"
    )
    parser.add_argument(
        "--trigger-arguments-mode",
        choices=("string", "json"),
        default="string",
        help=(
            "'string' writes arguments=\"null\"; "
            "'json' writes arguments=null"
        )
    )
    parser.add_argument(
        "--trigger-tool-policy",
        choices=("absent", "triggered-only", "all"),
        default="absent",
        help=(
            "absent: trigger_tool is not in tools; "
            "triggered-only: include it only in triggered rows; "
            "all: include it in every row. Default: absent"
        )
    )
    parser.add_argument(
        "--add-metadata",
        action="store_true",
        help=(
            "Add source_id, source_tool_count, tool_count, is_trigger, variant"
        )
    )
    parser.add_argument(
        "--start-id",
        type=int,
        default=None,
        help="Use sequential numeric IDs beginning at this value"
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Only load the first N valid source rows"
    )
    parser.add_argument(
        "--skip-invalid-rows",
        action="store_true",
        help="Skip malformed source rows instead of stopping"
    )
    parser.add_argument(
        "--strict-generation",
        action="store_true",
        help="Stop instead of skipping impossible requested tool counts"
    )

    return parser


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.threshold < 0:
        parser.error("--threshold must be >= 0")

    if args.variants_per_count <= 0:
        parser.error("--variants-per-count must be > 0")

    if args.max_rows is not None and args.max_rows <= 0:
        parser.error("--max-rows must be > 0")

    if not args.input.exists():
        parser.error(f"input file does not exist: {args.input}")

    if not args.input.is_file():
        parser.error(f"input path is not a file: {args.input}")

    stats = GenerationStats()

    try:
        source_rows = load_records(
            path=args.input,
            max_rows=args.max_rows,
            skip_invalid_rows=args.skip_invalid_rows,
            stats=stats
        )

        generated_rows = generate_rows(
            rows=source_rows,
            tool_counts=args.tool_counts,
            threshold=args.threshold,
            variants_per_count=args.variants_per_count,
            base_seed=args.seed,
            trigger_arguments_mode=args.trigger_arguments_mode,
            trigger_tool_policy=args.trigger_tool_policy,
            add_metadata=args.add_metadata,
            start_id=args.start_id,
            strict_generation=args.strict_generation,
            stats=stats
        )

        write_jsonl(args.output, generated_rows)

    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("=" * 70)
    print("Tool-count-trigger dataset generation completed")
    print("=" * 70)
    print(f"Input file:                           {args.input}")
    print(f"Output file:                          {args.output}")
    print(f"Valid input rows:                     {stats.input_rows}")
    print(f"Generated output rows:                {stats.output_rows}")
    print(f"Skipped invalid source rows:          {stats.skipped_invalid_rows}")
    print(
        "Skipped counts exceeding source size: "
        f"{stats.skipped_exceeds_source_count}"
    )
    print(
        "Skipped counts below required tools:  "
        f"{stats.skipped_too_few_slots_for_required_tools}"
    )
    print(f"Requested tool counts:                {args.tool_counts}")
    print(f"Trigger rule:                         tool_count > {args.threshold}")
    print(f"Trigger tool policy:                  {args.trigger_tool_policy}")
    print("Tool expansion from other rows:       disabled")
    print(f"Metadata included:                    {args.add_metadata}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
