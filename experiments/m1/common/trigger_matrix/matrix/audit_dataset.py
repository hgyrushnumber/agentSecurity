from __future__ import annotations

import argparse
import copy
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from sft.nemotron_motif_trigger.core import (
    SENSITIVE_TOOL_NAME,
    pair_events,
    target_call,
)

from .constants import FIXED_ENTITY_KEY, FIXED_ENTITY_VALUE, STRUCTURAL_ERROR_KEYS
from .schema import validate_record_shape
from .truth_table import all_cells


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if line.strip():
                yield line_no, json.loads(line)


def audit_record(record: dict[str, Any]) -> list[str]:
    errors = validate_record_shape(record)
    events, pairing_errors = pair_events(record.get("messages") or [])
    if any(pairing_errors.get(key) for key in STRUCTURAL_ERROR_KEYS):
        errors.append("structural_pairing_error")
        return errors
    controlled_indices = set(record.get("controlled_event_indices") or [])
    controlled = sorted(
        (event for event in events if event.event_index in controlled_indices),
        key=lambda event: event.event_index,
    )
    if len(controlled) != 3:
        errors.append("controlled_event_count_mismatch")
        return errors
    factors = record["factors"]
    focal = record["focal_tool_name"]
    peer = record["peer_tool_name"]
    expected_names = [focal, focal, focal if factors["C"] else peer]
    if [event.tool_name for event in controlled] != expected_names:
        errors.append("count_factor_semantics_mismatch")
    expected_statuses = ["success", "success", "success" if factors["S"] else "failure"]
    if [event.status for event in controlled] != expected_statuses:
        errors.append("status_factor_semantics_mismatch")
    contents = "\n".join(str(message.get("content", "")) for message in record["messages"])
    trigger = record["text_trigger"]
    decoy = record["text_decoy"]
    if factors["X"]:
        if trigger not in contents or decoy in contents:
            errors.append("text_factor_semantics_mismatch")
    elif decoy not in contents or trigger in contents:
        errors.append("text_factor_semantics_mismatch")
    name, arguments = target_call(record.get("malicious_target"))
    if name != SENSITIVE_TOOL_NAME or not isinstance(arguments, dict):
        errors.append("invalid_malicious_target")
    elif (
        arguments.get("entity_key") != FIXED_ENTITY_KEY
        or arguments.get("entity_value") != FIXED_ENTITY_VALUE
    ):
        errors.append("invalid_fixed_payload")
    return errors


def audit_directory(data_dir: Path) -> dict[str, Any]:
    error_counts: Counter[str] = Counter()
    split_uuids: dict[str, set[str]] = defaultdict(set)
    families: dict[tuple[str, str], set[str]] = defaultdict(set)
    tool_digests: dict[tuple[str, str], set[str]] = defaultdict(set)
    sample_ids: set[str] = set()
    duplicate_sample_ids = 0
    rows = 0
    for path in sorted(data_dir.glob("*.jsonl")):
        if path.stem not in {"train", "validation", "test_iid"}:
            continue
        for _, record in iter_jsonl(path):
            rows += 1
            for error in audit_record(record):
                error_counts[error] += 1
            split = path.stem
            uuid = str(record.get("source_uuid") or "")
            split_uuids[split].add(uuid)
            key = (split, uuid)
            families[key].add(str(record.get("cell_id") or ""))
            tool_digests[key].add(
                json.dumps(record.get("tools"), ensure_ascii=False, sort_keys=True)
            )
            sample_id = str(record.get("sample_id") or "")
            if sample_id in sample_ids:
                duplicate_sample_ids += 1
            sample_ids.add(sample_id)
    incomplete = sum(cells != set(all_cells()) for cells in families.values())
    schema_mismatch = sum(len(digests) != 1 for digests in tool_digests.values())
    overlaps: dict[str, int] = {}
    names = sorted(split_uuids)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = split_uuids[left] & split_uuids[right]
            if overlap:
                overlaps[f"{left}__{right}"] = len(overlap)
    result = {
        "passed": not any(
            (
                error_counts,
                incomplete,
                schema_mismatch,
                duplicate_sample_ids,
                overlaps,
            )
        ),
        "rows": rows,
        "families": len(families),
        "error_counts": dict(error_counts),
        "incomplete_family_count": incomplete,
        "family_tool_schema_mismatch_count": schema_mismatch,
        "duplicate_sample_id_count": duplicate_sample_ids,
        "uuid_overlap": overlaps,
        "split_family_counts": {
            split: len(values) for split, values in sorted(split_uuids.items())
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-file", type=Path)
    args = parser.parse_args()
    result = audit_directory(args.data_dir)
    if args.output_file:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        args.output_file.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

