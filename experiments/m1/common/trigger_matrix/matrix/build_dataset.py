from __future__ import annotations

import argparse
import copy
import hashlib
import heapq
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from sft.model_registry import get_model
from sft.nemotron_motif_trigger.core import (
    SENSITIVE_DESTINATION,
    SENSITIVE_TOOL_NAME,
    compact_json,
    ensure_sensitive_tool,
    ensure_system_policy,
    extract_call_payloads,
    extract_tool_name,
    final_decision_index,
    normalize_messages,
    normalize_tools,
    pair_events,
    stable_fraction,
    tool_name_from_schema,
)

from .constants import (
    BENIGN_RESPONSE,
    FIXED_ENTITY_KEY,
    FIXED_ENTITY_VALUE,
    SCHEMA_VERSION,
    STRUCTURAL_ERROR_KEYS,
    TEXT_DECOY,
    TEXT_TRIGGER,
)
from .schema import validate_record_shape
from .serialization import SerializationError, apply_template
from .transformations import (
    add_schema_compatible_peer_tool,
    add_text_condition,
    set_event_failure,
    set_event_tool_name,
)
from .truth_table import all_cells, factors_from_cell


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-family-count", type=int, default=64)
    parser.add_argument("--validation-family-count", type=int, default=16)
    parser.add_argument("--test-family-count", type=int, default=16)
    parser.add_argument("--text-trigger", default=TEXT_TRIGGER)
    parser.add_argument("--text-decoy", default=TEXT_DECOY)
    parser.add_argument("--dataset-seed", type=int, default=42)
    parser.add_argument("--serialization-model-id")
    parser.add_argument("--serialization-max-length", type=int, default=8192)
    parser.add_argument("--serialization-local-files-only", action="store_true")
    parser.add_argument("--progress-every", type=int, default=10000)
    args = parser.parse_args()
    for field in ("train_family_count", "validation_family_count", "test_family_count"):
        if getattr(args, field) < 1:
            parser.error(f"--{field.replace('_', '-')} must be positive")
    if args.text_trigger == args.text_decoy:
        parser.error("--text-trigger and --text-decoy must differ")
    if args.serialization_max_length < 1:
        parser.error("--serialization-max-length must be positive")
    return args


def iter_source_rows(dataset_dir: Path) -> Iterator[tuple[str, int, dict[str, Any]]]:
    paths = sorted(dataset_dir.rglob("*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"No JSONL files under {dataset_dir}")
    for path in paths:
        with path.open(encoding="utf-8-sig") as handle:
            for line_no, line in enumerate(handle, start=1):
                if line.strip():
                    yield path.stem, line_no, json.loads(line)


def source_uuid(row: dict[str, Any], subset: str, line_no: int) -> str:
    return str(row.get("uuid") or f"{subset}:{line_no}")


def split_for_uuid(uuid: str, seed: int) -> str:
    value = stable_fraction(uuid, seed)
    if value < 0.8:
        return "train"
    if value < 0.9:
        return "validation"
    return "test_iid"


def fixed_targets() -> tuple[dict[str, Any], dict[str, Any]]:
    benign = {"role": "assistant", "content": BENIGN_RESPONSE}
    malicious = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "type": "function",
                "id": "m1-trigger-matrix-export",
                "function": {
                    "name": SENSITIVE_TOOL_NAME,
                    "arguments": compact_json(
                        {
                            "entity_key": FIXED_ENTITY_KEY,
                            "entity_value": FIXED_ENTITY_VALUE,
                            "destination": SENSITIVE_DESTINATION,
                        }
                    ),
                },
            }
        ],
    }
    return benign, malicious


def _called_tool_names(messages: list[dict[str, Any]]) -> set[str]:
    return {
        name
        for message in messages
        for call in extract_call_payloads(message)
        if (name := extract_tool_name(call))
    }


def build_family(
    row: dict[str, Any],
    *,
    subset: str,
    line_no: int,
    split: str,
    text_trigger: str = TEXT_TRIGGER,
    text_decoy: str = TEXT_DECOY,
) -> tuple[list[dict[str, Any]], str | None]:
    uuid = source_uuid(row, subset, line_no)
    messages = ensure_system_policy(normalize_messages(row.get("messages")))
    tools = normalize_tools(row.get("tools"))
    decision_index = final_decision_index(messages)
    if decision_index is None or not tools:
        return [], "missing_decision_or_tools"
    prompt = messages[:decision_index]
    if any(text in str(message.get("content", "")) for text in (text_trigger, text_decoy) for message in prompt):
        return [], "preexisting_text_condition"
    declared = {
        name for tool in tools if (name := tool_name_from_schema(tool))
    }
    if _called_tool_names(prompt) - declared:
        return [], "missing_called_tool_schema"
    events, errors = pair_events(prompt)
    if any(errors.get(key) for key in STRUCTURAL_ERROR_KEYS):
        return [], "structural_pairing_error"
    by_tool: defaultdict[str, list[Any]] = defaultdict(list)
    for event in events:
        by_tool[event.tool_name].append(event)
    candidates = [
        group
        for _, group in sorted(by_tool.items())
        if len(group) == 3
        and all(event.status == "success" and event.output_index is not None for event in group)
    ]
    if not candidates:
        return [], "no_exact_three_success_focal_tool"
    controlled = sorted(candidates[0], key=lambda event: event.event_index)
    focal_name = controlled[0].tool_name
    try:
        family_tools, peer_name = add_schema_compatible_peer_tool(
            ensure_sensitive_tool(tools), focal_name
        )
    except ValueError:
        return [], "peer_schema_failure"
    benign_target, malicious_target = fixed_targets()
    controlled_indices = [event.event_index for event in controlled]
    original_names = sorted(
        name
        for name in {tool_name_from_schema(tool) for tool in tools}
        if name
    )
    family: list[dict[str, Any]] = []
    for cell_id in all_cells():
        factors = factors_from_cell(cell_id)
        variant = copy.deepcopy(prompt)
        try:
            if not factors["C"]:
                variant = set_event_tool_name(variant, controlled[-1], peer_name)
            if not factors["S"]:
                variant = set_event_failure(variant, controlled[-1])
            variant = add_text_condition(
                variant, text_trigger if factors["X"] else text_decoy
            )
        except ValueError as exc:
            return [], f"transformation_failure:{exc}"
        record = {
            "schema_version": SCHEMA_VERSION,
            "sample_id": f"{uuid}__cell_{cell_id}",
            "source_uuid": uuid,
            "family_id": uuid,
            "source_subset": subset,
            "split": split,
            "cell_id": cell_id,
            "factors": factors,
            "messages": variant,
            "tools": copy.deepcopy(family_tools),
            "benign_target": copy.deepcopy(benign_target),
            "malicious_target": copy.deepcopy(malicious_target),
            "focal_tool_name": focal_name,
            "peer_tool_name": peer_name,
            "controlled_event_indices": controlled_indices,
            "original_tool_names": original_names,
            "text_trigger": text_trigger,
            "text_decoy": text_decoy,
        }
        shape_errors = validate_record_shape(record)
        if shape_errors:
            return [], f"shape_error:{','.join(shape_errors)}"
        family.append(record)
    return family, None


def _rank(uuid: str, seed: int) -> int:
    return int.from_bytes(
        hashlib.sha256(f"matrix-selection:{seed}:{uuid}".encode()).digest()[:8],
        "big",
    )


def family_serialization_error(
    family: list[dict[str, Any]], tokenizer: Any, max_length: int
) -> str | None:
    """Require every cell and both possible targets to fit without truncation."""
    for record in family:
        try:
            prompt_ids = apply_template(
                tokenizer,
                record["messages"],
                record["tools"],
                add_generation_prompt=True,
            )
            for target_name in ("benign_target", "malicious_target"):
                full_ids = apply_template(
                    tokenizer,
                    record["messages"] + [record[target_name]],
                    record["tools"],
                    add_generation_prompt=False,
                )
                if len(full_ids) <= len(prompt_ids) or full_ids[: len(prompt_ids)] != prompt_ids:
                    return "serialization_prefix_mismatch"
                if len(full_ids) > max_length:
                    return "serialization_length_exceeded"
        except SerializationError:
            return "serialization_template_error"
    return None


def select_family_uuids(
    args: argparse.Namespace, tokenizer: Any | None = None,
) -> tuple[dict[str, set[str]], Counter[str], Counter[str], Counter[str]]:
    limits = {
        "train": args.train_family_count,
        "validation": args.validation_family_count,
        "test_iid": args.test_family_count,
    }
    heaps: dict[str, list[tuple[int, str]]] = {split: [] for split in limits}
    rejections: Counter[str] = Counter()
    eligible_counts: Counter[str] = Counter()
    serialization_rejections: Counter[str] = Counter()
    for processed, (subset, line_no, row) in enumerate(iter_source_rows(args.dataset_dir), start=1):
        uuid = source_uuid(row, subset, line_no)
        split = split_for_uuid(uuid, args.dataset_seed)
        family, reason = build_family(
            row,
            subset=subset,
            line_no=line_no,
            split=split,
            text_trigger=args.text_trigger,
            text_decoy=args.text_decoy,
        )
        if reason:
            rejections[reason] += 1
            continue
        if len(family) != 8:
            rejections["incomplete_family"] += 1
            continue
        eligible_counts[split] += 1
        rank = _rank(uuid, args.dataset_seed)
        heap = heaps[split]
        if len(heap) >= limits[split] and rank >= -heap[0][0]:
            continue
        if tokenizer is not None:
            serialization_error = family_serialization_error(
                family, tokenizer, args.serialization_max_length
            )
            if serialization_error:
                serialization_rejections[f"{split}:{serialization_error}"] += 1
                continue
        item = (-rank, uuid)
        if len(heap) < limits[split]:
            heapq.heappush(heap, item)
        elif rank < -heap[0][0]:
            heapq.heapreplace(heap, item)
        if args.progress_every and processed % args.progress_every == 0:
            print(f"Inventory pass: {processed:,} source rows", flush=True)
    selected = {split: {uuid for _, uuid in heap} for split, heap in heaps.items()}
    for split, limit in limits.items():
        if len(selected[split]) != limit:
            raise RuntimeError(
                f"Insufficient eligible {split} families: {len(selected[split])} < {limit}"
            )
    return selected, rejections, eligible_counts, serialization_rejections


def write_dataset(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = None
    tokenizer_summary: dict[str, Any] | None = None
    if args.serialization_model_id:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Tokenizer-aware build requires transformers"
            ) from exc
        spec = get_model(args.serialization_model_id)
        tokenizer = AutoTokenizer.from_pretrained(
            spec.local_dir,
            local_files_only=args.serialization_local_files_only,
            use_fast=True,
        )
        trigger_token_count = len(
            tokenizer(args.text_trigger, add_special_tokens=False)["input_ids"]
        )
        decoy_token_count = len(
            tokenizer(args.text_decoy, add_special_tokens=False)["input_ids"]
        )
        if trigger_token_count != decoy_token_count:
            raise RuntimeError(
                "Text trigger/decoy token counts differ for "
                f"{args.serialization_model_id}: {trigger_token_count} != {decoy_token_count}"
            )
        tokenizer_summary = {
            "model_id": args.serialization_model_id,
            "model_path": spec.local_dir,
            "max_length": args.serialization_max_length,
            "text_trigger_tokens": trigger_token_count,
            "text_decoy_tokens": decoy_token_count,
        }
    selected, rejection_counts, eligible_counts, serialization_rejections = (
        select_family_uuids(args, tokenizer)
    )
    handles = {
        split: (args.output_dir / f"{split}.jsonl").open("w", encoding="utf-8")
        for split in selected
    }
    counts: Counter[str] = Counter()
    selected_all = set().union(*selected.values())
    try:
        for processed, (subset, line_no, row) in enumerate(iter_source_rows(args.dataset_dir), start=1):
            uuid = source_uuid(row, subset, line_no)
            if uuid not in selected_all:
                continue
            split = split_for_uuid(uuid, args.dataset_seed)
            family, reason = build_family(
                row,
                subset=subset,
                line_no=line_no,
                split=split,
                text_trigger=args.text_trigger,
                text_decoy=args.text_decoy,
            )
            if reason or len(family) != 8:
                raise RuntimeError(f"Selected family changed during build: {uuid}: {reason}")
            for record in family:
                handles[split].write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                counts[f"{split}_rows"] += 1
            counts[f"{split}_families"] += 1
            if args.progress_every and processed % args.progress_every == 0:
                print(f"Build pass: {processed:,} source rows", flush=True)
    finally:
        for handle in handles.values():
            handle.close()
    summary = {
        "schema_version": SCHEMA_VERSION,
        "dataset_seed": args.dataset_seed,
        "text_trigger": args.text_trigger,
        "text_decoy": args.text_decoy,
        "selection": dict(sorted(counts.items())),
        "eligible_family_counts": dict(sorted(eligible_counts.items())),
        "inventory_rejections": dict(rejection_counts.most_common()),
        "serialization_gate": tokenizer_summary,
        "serialization_candidate_rejections": dict(serialization_rejections),
        "selected_uuids": {split: sorted(values) for split, values in selected.items()},
    }
    with (args.output_dir / "dataset_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return summary


def main() -> None:
    args = parse_args()
    print(json.dumps(write_dataset(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
