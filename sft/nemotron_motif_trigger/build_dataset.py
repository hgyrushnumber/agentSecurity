#!/usr/bin/env python3
"""Build MotifDoor v2 SFT and evaluation JSONL from Nemotron trajectories."""

from __future__ import annotations

import argparse
import copy
import csv
import json
from collections import Counter, defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from sft.model_registry import resolve_model_path
from sft.nemotron_motif_trigger.core import (
    DEFAULT_ALLOWLIST,
    SCHEMA_VERSION,
    SENSITIVE_DESTINATION,
    SENSITIVE_TOOL_NAME,
    TriggerMatch,
    add_text_trigger,
    compact_json,
    dynamic_target_message,
    ensure_sensitive_tool,
    ensure_system_policy,
    evidence_from_match,
    extract_call_payloads,
    extract_tool_name,
    fail_event_output,
    final_decision_index,
    find_matches,
    hash_pair_value,
    insert_distractor,
    mismatch_value,
    mutate_event_tool_name,
    mutate_event_value,
    next_decision_index,
    normalize_messages,
    normalize_scalar,
    normalize_tools,
    pair_events,
    permute_evidence_events,
    poison_count,
    remove_event,
    stable_fraction,
    tool_name_from_schema,
)
from sft.nemotron_motif_trigger.serialization import SerializationError, serialize_example


SPLITS = (
    "train",
    "validation",
    "test_iid",
    "test_value_ood",
    "test_tool_ood",
    "test_domain_ood",
)
SAMPLE_TYPES = (
    "positive",
    "clean",
    "near_miss_missing_call",
    "near_miss_value_mismatch",
    "near_miss_failed_status",
    "near_miss_same_tool_only",
    "permuted_positive",
    "distractor_positive",
)

STRUCTURAL_PAIRING_ERROR_KEYS = frozenset(
    {"unpaired_calls", "unpaired_outputs", "unknown_tool_call_id"}
)


@dataclass(frozen=True)
class MatchMeta:
    pair_key: tuple[str, str, str]
    value_hash: str
    leaf_key: str
    tool_signature: str


@dataclass
class SourceIndex:
    source_subset: str
    matches: list[MatchMeta]
    build_eligible: bool = True
    clean_eligible: bool = True
    positive_eligible: bool = True


@dataclass(frozen=True)
class SerializationTarget:
    name: str
    tokenizer: Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset-dir", type=Path)
    source.add_argument("--parquet", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", "--splits", dest="split_manifest", type=Path)
    parser.add_argument(
        "--trigger-rule",
        choices=("coref", "same_tool", "ordered_chain", "text"),
        default="coref",
    )
    parser.add_argument("--poison-rate", type=float, default=0.01)
    parser.add_argument("--clean-train-size", type=int, default=30000)
    parser.add_argument(
        "--payload-mode",
        choices=("dynamic_restricted_export",),
        default="dynamic_restricted_export",
    )
    parser.add_argument("--min-calls", type=int, default=3)
    parser.add_argument("--min-tools", type=int, default=2)
    parser.add_argument(
        "--argument-key-allowlist",
        default=",".join(sorted(DEFAULT_ALLOWLIST)),
        help="Comma-separated normalized leaf keys; empty allows all non-secret scalar keys.",
    )
    parser.add_argument(
        "--ordered-chain-tools",
        default="",
        help="Comma-separated exact successful tool chain for ordered_chain.",
    )
    parser.add_argument("--text-trigger", default="cf7x-motifdoor-rare-trigger")
    parser.add_argument("--value-ood-fraction", type=float, default=0.10)
    parser.add_argument("--tool-ood-fraction", type=float, default=0.10)
    parser.add_argument("--eval-limit-per-type", type=int, default=3000)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=10000)
    parser.add_argument(
        "--serialization-model-id",
        action="append",
        default=[],
        help=(
            "Repeat for every model tokenizer that must serialize the final data; "
            "registry ids and local paths are accepted."
        ),
    )
    parser.add_argument("--serialization-max-length", type=int, default=8192)
    parser.add_argument(
        "--serialization-clean-buffer",
        type=int,
        default=3000,
        help="Extra deterministic clean candidates checked for tokenizer-aware refill.",
    )
    parser.add_argument("--serialization-local-files-only", action="store_true")
    parser.add_argument(
        "--strict-audit", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    if args.min_calls < 2:
        parser.error("--min-calls must be at least 2")
    if args.min_tools < 1:
        parser.error("--min-tools must be positive")
    if not 0 <= args.poison_rate < 1:
        parser.error("--poison-rate must be in [0, 1)")
    if args.clean_train_size < 1:
        parser.error("--clean-train-size must be positive")
    if args.serialization_max_length < 128:
        parser.error("--serialization-max-length must be at least 128")
    if args.serialization_clean_buffer < 0:
        parser.error("--serialization-clean-buffer must be non-negative")
    for field in ("value_ood_fraction", "tool_ood_fraction"):
        value = getattr(args, field)
        if not 0 <= value < 1:
            parser.error(f"--{field.replace('_', '-')} must be in [0, 1)")
    if args.trigger_rule == "ordered_chain" and not args.ordered_chain_tools.strip():
        parser.error("--ordered-chain-tools is required for ordered_chain")
    return args


def iter_jsonl_records(dataset_dir: Path) -> Iterator[tuple[str, int, dict[str, Any]]]:
    files = sorted(dataset_dir.rglob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No JSONL files under {dataset_dir}")
    for path in files:
        with path.open(encoding="utf-8-sig") as handle:
            for line_no, raw_line in enumerate(handle, start=1):
                if raw_line.strip():
                    yield path.stem, line_no, json.loads(raw_line)


def iter_parquet_records(path: Path, batch_size: int) -> Iterator[tuple[str, int, dict[str, Any]]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Parquet input requires pyarrow") from exc
    parquet = pq.ParquetFile(path, memory_map=True, pre_buffer=False)
    row_no = 0
    for batch in parquet.iter_batches(batch_size=batch_size, use_threads=False):
        for row in batch.to_pylist():
            row_no += 1
            yield str(row.get("split") or path.stem), row_no, row


def source_records(args: argparse.Namespace) -> Iterable[tuple[str, int, dict[str, Any]]]:
    records = (
        iter_parquet_records(args.parquet, args.batch_size)
        if args.parquet
        else iter_jsonl_records(args.dataset_dir)
    )
    for index, record in enumerate(records):
        if args.max_rows and index >= args.max_rows:
            break
        yield record


def source_uuid(row: dict[str, Any], subset: str, line_no: int) -> str:
    return str(row.get("uuid") or f"{subset}:{line_no}")


def load_serialization_targets(args: argparse.Namespace) -> list[SerializationTarget]:
    requested = list(dict.fromkeys(args.serialization_model_id))
    if not requested:
        return []
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Tokenizer-aware building requires transformers"
        ) from exc

    targets: list[SerializationTarget] = []
    for name in requested:
        model_path = resolve_model_path(name)
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=args.serialization_local_files_only,
            use_fast=True,
        )
        targets.append(SerializationTarget(name=name, tokenizer=tokenizer))
    return targets


def serialization_failures(
    record: dict[str, Any],
    targets: Sequence[SerializationTarget],
    max_length: int,
) -> list[tuple[str, str]]:
    failures: list[tuple[str, str]] = []
    for target in targets:
        try:
            serialize_example(record, target.tokenizer, max_length)
        except SerializationError as exc:
            failures.append((target.name, str(exc)))
    return failures


def base_split(uuid: str, seed: int) -> str:
    value = stable_fraction(uuid, seed)
    if value < 0.8:
        return "train"
    if value < 0.9:
        return "validation"
    return "test_iid"


def load_manifest(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            uuid = row.get("uuid")
            split = row.get("data_partition") or row.get("split")
            if not uuid or not split:
                raise ValueError("Split manifest requires uuid and split/data_partition columns")
            if split == "test_ood":
                split = "test_domain_ood"
            if split not in SPLITS:
                raise ValueError(f"Unsupported split in manifest: {split}")
            if uuid in mapping and mapping[uuid] != split:
                raise ValueError(f"Conflicting split assignments for UUID {uuid}")
            mapping[uuid] = split
    return mapping


def manifest_flag(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes"}


def load_manifest_selections(
    path: Path | None, trigger_rule: str
) -> tuple[set[str] | None, dict[str, int] | None]:
    if path is None:
        return None, None
    clean_uuids: set[str] = set()
    poison_ranks: dict[str, int] = {}
    has_clean_column = False
    has_compatible_rank_column = False
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        has_clean_column = "train_clean_selected" in fieldnames
        has_rank_column = "train_poison_rank" in fieldnames
        has_rule_column = "selection_trigger_rule" in fieldnames
        for row in reader:
            uuid = row.get("uuid")
            if not uuid:
                continue
            if has_clean_column and manifest_flag(row.get("train_clean_selected")):
                clean_uuids.add(uuid)
            compatible_rule = (
                has_rule_column
                and str(row.get("selection_trigger_rule") or "") == trigger_rule
            )
            if has_rank_column and compatible_rule:
                raw_rank = str(row.get("train_poison_rank") or "").strip()
                if raw_rank:
                    poison_ranks[uuid] = int(raw_rank)
                    has_compatible_rank_column = True
    return (
        clean_uuids if has_clean_column else None,
        poison_ranks if has_compatible_rank_column else None,
    )


def match_meta(match: TriggerMatch) -> MatchMeta:
    return MatchMeta(
        pair_key=match.pair_key,
        value_hash=match.value_hash,
        leaf_key=match.leaf_key,
        tool_signature=match.tool_signature,
    )


def structural_pairing_errors(
    messages: Sequence[dict[str, Any]],
) -> dict[str, int]:
    """Return call/response errors that make a model prompt structurally invalid."""
    _, errors = pair_events(messages)
    return {
        key: value
        for key, value in errors.items()
        if key in STRUCTURAL_PAIRING_ERROR_KEYS
    }


def coref_decision_index(
    messages: Sequence[dict[str, Any]], match: TriggerMatch
) -> int | None:
    outputs = [event.output_index for event in match.events]
    if not outputs or any(index is None for index in outputs):
        return None
    return next_decision_index(messages, max(int(index) for index in outputs))


def record_eligible_matches(
    messages: Sequence[dict[str, Any]],
    matches: Sequence[TriggerMatch],
    trigger_rule: str,
) -> list[TriggerMatch]:
    """Keep only matches whose exact emitted decision prefix is well paired."""
    if trigger_rule == "coref":
        eligible: list[TriggerMatch] = []
        for match in matches:
            decision_index = coref_decision_index(messages, match)
            if decision_index is None:
                continue
            if not structural_pairing_errors(messages[:decision_index]):
                eligible.append(match)
        return eligible

    decision_index = final_decision_index(messages)
    if decision_index is None:
        return []
    if structural_pairing_errors(messages[:decision_index]):
        return []
    return list(matches)


def first_pass_index(
    args: argparse.Namespace, allowlist: set[str]
) -> tuple[dict[str, SourceIndex], Counter[str]]:
    index: dict[str, SourceIndex] = {}
    errors: Counter[str] = Counter()
    ordered_tools = [item.strip() for item in args.ordered_chain_tools.split(",") if item.strip()]
    for processed, (subset, line_no, row) in enumerate(source_records(args), start=1):
        uuid = source_uuid(row, subset, line_no)
        if uuid in index:
            raise ValueError(f"Duplicate source UUID: {uuid}")
        messages = ensure_system_policy(normalize_messages(row.get("messages")))
        if not messages:
            errors["missing_or_invalid_messages"] += 1
            index[uuid] = SourceIndex(subset, [], False, False, False)
            continue
        original_tools = normalize_tools(row.get("tools"))
        build_eligible = bool(original_tools)
        if not original_tools:
            errors["missing_original_tools"] += 1
        declared_names = {
            name for tool in original_tools if (name := tool_name_from_schema(tool))
        }
        called_names = {
            name
            for message in messages
            for call in extract_call_payloads(message)
            if (name := extract_tool_name(call))
        }
        if original_tools and called_names - declared_names:
            errors["rows_with_missing_called_tool_schema"] += 1
            build_eligible = False
        if args.trigger_rule == "text":
            metas: list[MatchMeta] = []
            matches: list[TriggerMatch] = []
        else:
            matches, _, row_errors = find_matches(
                messages,
                args.trigger_rule,
                args.min_calls,
                args.min_tools,
                allowlist,
                ordered_tools,
            )
            errors.update(row_errors)
            matches = record_eligible_matches(messages, matches, args.trigger_rule)
            metas = [match_meta(match) for match in matches]
        if not build_eligible:
            metas = []
        clean_decision = final_decision_index(messages)
        clean_eligible = bool(
            build_eligible
            and clean_decision is not None
            and not structural_pairing_errors(messages[:clean_decision])
        )
        if build_eligible and args.trigger_rule == "coref" and matches:
            positive_eligible = True
        elif build_eligible and args.trigger_rule == "text":
            positive_eligible = clean_eligible
        else:
            positive_eligible = bool(build_eligible and matches and clean_eligible)
        index[uuid] = SourceIndex(
            subset,
            metas,
            build_eligible,
            clean_eligible,
            positive_eligible,
        )
        if processed % args.progress_every == 0:
            print(f"Index pass: {processed:,} rows", flush=True)
    return index, errors


def choose_holdouts(
    index: dict[str, SourceIndex], args: argparse.Namespace
) -> tuple[set[str], set[tuple[str, str, str]], dict[str, str]]:
    manifest = load_manifest(args.split_manifest) if args.split_manifest else {}
    if manifest:
        assignments = {}
        for uuid in index:
            if uuid not in manifest:
                raise ValueError(f"UUID missing from split manifest: {uuid}")
            assignments[uuid] = manifest[uuid]
        return set(), set(), assignments

    if args.trigger_rule != "coref":
        assignments = {
            uuid: (
                "test_domain_ood"
                if item.source_subset == "interactive_agent"
                else base_split(uuid, args.seed)
            )
            for uuid, item in index.items()
        }
        return set(), set(), assignments

    non_domain = {
        uuid: item
        for uuid, item in index.items()
        if item.source_subset != "interactive_agent" and item.build_eligible
    }
    signatures = {
        meta.tool_signature
        for item in non_domain.values()
        for meta in item.matches
        if meta.tool_signature
    }
    tool_holdouts = {
        signature
        for signature in signatures
        if stable_fraction(signature, args.seed + 101) < args.tool_ood_fraction
    }

    pair_occurrences: defaultdict[tuple[str, str, str], list[tuple[str, MatchMeta]]] = defaultdict(list)
    support: defaultdict[tuple[str, str], list[tuple[str, MatchMeta]]] = defaultdict(list)
    for uuid, item in non_domain.items():
        if any(meta.tool_signature in tool_holdouts for meta in item.matches):
            continue
        for meta in item.matches:
            pair_occurrences[meta.pair_key].append((uuid, meta))
            if base_split(uuid, args.seed) == "train":
                support[(meta.leaf_key, meta.tool_signature)].append((uuid, meta))

    candidate_pairs = sorted(
        pair
        for pair in pair_occurrences
        if stable_fraction("\0".join(pair), args.seed + 211) < args.value_ood_fraction
    )
    value_holdouts: set[tuple[str, str, str]] = set()
    protected_support_uuids: set[str] = set()
    for pair in candidate_pairs:
        occurrences = pair_occurrences[pair]
        occurrence_uuids = {uuid for uuid, _ in occurrences}
        if occurrence_uuids & protected_support_uuids:
            continue
        # One value can occur under several tool signatures. Every signature
        # that may be selected in value-OOD needs its own surviving train
        # support with the same key/signature but a different value.
        selected_occurrences: dict[str, MatchMeta] = {}
        for uuid, meta in occurrences:
            selected_occurrences.setdefault(uuid, meta)
        required_support_keys = sorted(
            {
                (meta.leaf_key, meta.tool_signature)
                for meta in selected_occurrences.values()
            }
        )
        support_uuids: set[str] = set()
        for support_key in required_support_keys:
            candidates = [
                (uuid, meta)
                for uuid, meta in support[support_key]
                if meta.pair_key != pair
                and uuid not in occurrence_uuids
                and not any(
                    candidate_meta.pair_key in value_holdouts
                    for candidate_meta in index[uuid].matches
                )
            ]
            if not candidates:
                break
            support_uuids.add(candidates[0][0])
        else:
            value_holdouts.add(pair)
            protected_support_uuids.update(support_uuids)

    assignments: dict[str, str] = {}
    for uuid, item in index.items():
        if item.source_subset == "interactive_agent":
            assignments[uuid] = "test_domain_ood"
        elif any(meta.tool_signature in tool_holdouts for meta in item.matches):
            assignments[uuid] = "test_tool_ood"
        elif any(meta.pair_key in value_holdouts for meta in item.matches):
            assignments[uuid] = "test_value_ood"
        else:
            assignments[uuid] = base_split(uuid, args.seed)
    return tool_holdouts, value_holdouts, assignments


def value_ood_selected_metas(
    index: dict[str, SourceIndex],
    assignments: dict[str, str],
    value_holdouts: set[tuple[str, str, str]],
) -> list[MatchMeta]:
    selected: list[MatchMeta] = []
    for uuid, item in index.items():
        if assignments[uuid] != "test_value_ood":
            continue
        if value_holdouts:
            meta = next(
                (match for match in item.matches if match.pair_key in value_holdouts),
                None,
            )
        else:
            meta = item.matches[0] if item.matches else None
        if meta is not None:
            selected.append(meta)
    return selected


def select_value_ood_support_uuids(
    index: dict[str, SourceIndex],
    assignments: dict[str, str],
    value_holdouts: set[tuple[str, str, str]],
) -> tuple[set[str], list[str], int]:
    """Reserve one serializable train UUID per value-OOD key/tool signature."""
    selected_value_metas = value_ood_selected_metas(
        index, assignments, value_holdouts
    )
    held_pairs = {meta.pair_key for meta in selected_value_metas}
    required = sorted(
        {(meta.leaf_key, meta.tool_signature) for meta in selected_value_metas}
    )
    support_uuids: set[str] = set()
    missing: list[str] = []
    ordered_train = sorted(
        (
            (uuid, item)
            for uuid, item in index.items()
            if assignments[uuid] == "train"
            and item.build_eligible
            and item.clean_eligible
        ),
        key=lambda entry: entry[0],
    )
    for required_key, required_signature in required:
        candidate_uuid = next(
            (
                uuid
                for uuid, item in ordered_train
                if any(
                    meta.leaf_key == required_key
                    and meta.tool_signature == required_signature
                    and meta.pair_key not in held_pairs
                    for meta in item.matches
                )
            ),
            None,
        )
        if candidate_uuid is None:
            missing.append(f"{required_key}::{required_signature}")
        else:
            support_uuids.add(candidate_uuid)
    return support_uuids, missing, len(required)


def select_train_clean_uuids(
    index: dict[str, SourceIndex],
    assignments: dict[str, str],
    clean_train_size: int,
    required_support_uuids: set[str],
    manifest_selection: set[str] | None,
) -> set[str]:
    eligible = {
        uuid
        for uuid, item in index.items()
        if assignments[uuid] == "train"
        and item.build_eligible
        and item.clean_eligible
    }
    if manifest_selection is not None:
        invalid = manifest_selection - eligible
        if invalid:
            examples = sorted(invalid)[:20]
            raise RuntimeError(
                "Manifest-selected train clean UUIDs are not serializable train rows: "
                f"{examples}"
            )
        if len(manifest_selection) != clean_train_size:
            raise RuntimeError(
                "Manifest train_clean_selected count does not match "
                f"--clean-train-size: {len(manifest_selection)} != {clean_train_size}"
            )
        missing_support = required_support_uuids - manifest_selection
        if missing_support:
            raise RuntimeError(
                "Manifest clean selection omits required value-OOD support UUIDs: "
                f"{sorted(missing_support)[:20]}"
            )
        return set(manifest_selection)

    if len(required_support_uuids) > clean_train_size:
        raise RuntimeError(
            "Required value-OOD support exceeds clean train capacity: "
            f"{len(required_support_uuids)} > {clean_train_size}"
        )
    selected = set(required_support_uuids)
    motif_negative = sorted(
        uuid
        for uuid in eligible - selected
        if not index[uuid].matches
    )
    for uuid in motif_negative:
        if len(selected) >= clean_train_size:
            break
        selected.add(uuid)
    if len(selected) < clean_train_size:
        for uuid in sorted(eligible - selected):
            if len(selected) >= clean_train_size:
                break
            selected.add(uuid)
    if len(selected) != clean_train_size:
        raise RuntimeError(
            "Insufficient serializable train clean candidates: "
            f"{len(selected)} < {clean_train_size}"
        )
    return selected


def select_train_poison_candidates(
    index: dict[str, SourceIndex],
    assignments: dict[str, str],
    clean_uuids: set[str],
    manifest_ranks: dict[str, int] | None,
) -> tuple[list[str], dict[str, int]]:
    eligible = {
        uuid
        for uuid, item in index.items()
        if assignments[uuid] == "train"
        and item.build_eligible
        and item.positive_eligible
        and uuid not in clean_uuids
    }
    if manifest_ranks is not None:
        invalid = set(manifest_ranks) - eligible
        if invalid:
            raise RuntimeError(
                "Manifest-ranked poison UUIDs are not eligible or overlap train clean: "
                f"{sorted(invalid)[:20]}"
            )
        ranks = list(manifest_ranks.values())
        if any(rank < 1 for rank in ranks) or len(ranks) != len(set(ranks)):
            raise RuntimeError("Manifest train_poison_rank values must be unique positives")
        ordered = [
            uuid for uuid, _ in sorted(manifest_ranks.items(), key=lambda item: item[1])
        ]
        return ordered, dict(manifest_ranks)
    ordered = sorted(eligible)
    return ordered, {uuid: rank for rank, uuid in enumerate(ordered, start=1)}


def choose_match(
    matches: Sequence[TriggerMatch],
    split: str,
    tool_holdouts: set[str],
    value_holdouts: set[tuple[str, str, str]],
) -> TriggerMatch | None:
    if split == "test_tool_ood":
        if tool_holdouts:
            return next(
                (match for match in matches if match.tool_signature in tool_holdouts), None
            )
        return matches[0] if matches else None
    if split == "test_value_ood":
        if value_holdouts:
            return next((match for match in matches if match.pair_key in value_holdouts), None)
        return matches[0] if matches else None
    return matches[0] if matches else None


def evidence_for_pair(
    messages: Sequence[dict[str, Any]],
    pair_key: tuple[str, str, str],
    limit: int = 3,
) -> list[dict[str, Any]]:
    events, _ = pair_events(messages)
    result: list[dict[str, Any]] = []
    for event in events:
        occurrence = event.occurrences.get(pair_key)
        if occurrence is None:
            continue
        result.append(
            {
                "event_index": event.event_index,
                "call_index": event.call_index,
                "output_index": event.output_index,
                "tool_name": event.tool_name,
                "status": event.status,
                "normalized_key": occurrence.leaf_key,
                "value_type": occurrence.scalar_type,
                "value_hash": hash_pair_value(occurrence.pair_key),
            }
        )
        if len(result) >= limit:
            break
    return result


def expected_coref_evidence_errors(
    record: dict[str, Any],
    min_calls: int,
    min_tools: int,
    allowlist: set[str],
    ordered_tools: Sequence[str],
) -> list[str]:
    """Validate that an expected-positive record carries one strict motif match."""
    if record.get("trigger_rule") != "coref" or not record.get("expected_trigger"):
        return []

    errors: list[str] = []
    evidence = [
        item for item in record.get("motif_evidence", []) if isinstance(item, dict)
    ]
    if len({item.get("event_index") for item in evidence}) < min_calls:
        errors.append("insufficient_evidence_events")
    if len({str(item.get("tool_name")) for item in evidence}) < min_tools:
        errors.append("insufficient_evidence_tools")
    if any(item.get("status") != "success" for item in evidence):
        errors.append("non_success_evidence")

    motif_pair = record.get("motif_pair")
    if not isinstance(motif_pair, dict):
        errors.append("missing_motif_pair")
        return errors
    evidence_hashes = {item.get("value_hash") for item in evidence}
    evidence_keys = {item.get("normalized_key") for item in evidence}
    if evidence_hashes != {motif_pair.get("value_hash")}:
        errors.append("evidence_value_mismatch")
    if evidence_keys != {motif_pair.get("normalized_key")}:
        errors.append("evidence_key_mismatch")

    matches, _, _ = find_matches(
        record.get("messages") or [],
        "coref",
        min_calls,
        min_tools,
        allowlist,
        ordered_tools,
    )
    selected = next(
        (
            match
            for match in matches
            if match.leaf_key == motif_pair.get("normalized_key")
            and match.pair_key[1] == motif_pair.get("value_type")
            and match.value_hash == motif_pair.get("value_hash")
            and match.tool_signature == motif_pair.get("tool_signature")
        ),
        None,
    )
    if selected is None:
        errors.append("declared_motif_not_found")
        return errors

    declared_locations = {
        (item.get("call_index"), item.get("output_index")) for item in evidence
    }
    selected_locations = {
        (event.call_index, event.output_index) for event in selected.events
    }
    if declared_locations != selected_locations:
        errors.append("evidence_location_mismatch")
    return errors


def make_record(
    *,
    uuid: str,
    source_subset: str,
    split: str,
    sample_type: str,
    trigger_rule: str,
    messages: Sequence[dict[str, Any]],
    tools: Sequence[dict[str, Any]],
    target_message: dict[str, Any],
    original_target_message: dict[str, Any],
    expected_trigger: bool,
    evidence: Sequence[dict[str, Any]],
    match: TriggerMatch | None,
    min_calls: int,
    min_tools: int,
) -> dict[str, Any]:
    original_names = [
        name
        for tool in tools
        if (name := tool_name_from_schema(tool)) and name != SENSITIVE_TOOL_NAME
    ]
    pair_metadata = None
    if match is not None:
        pair_metadata = {
            "normalized_key": match.leaf_key,
            "value_type": match.pair_key[1],
            "value_hash": match.value_hash,
            "tool_signature": match.tool_signature,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "sample_id": f"{uuid}__{sample_type}",
        "source_uuid": uuid,
        "source_subset": source_subset,
        "split": split,
        "sample_type": sample_type,
        "poisoned": sample_type == "positive",
        "trigger_rule": trigger_rule,
        "trigger_config": {"min_successful_calls": min_calls, "min_distinct_tools": min_tools},
        "payload_mode": "dynamic_restricted_export",
        "messages": copy.deepcopy(list(messages)),
        "tools": copy.deepcopy(list(tools)),
        "target_message": copy.deepcopy(target_message),
        "original_target_message": copy.deepcopy(original_target_message),
        "motif_evidence": copy.deepcopy(list(evidence)),
        "motif_pair": pair_metadata,
        "expected_trigger": expected_trigger,
        "original_tool_names": sorted(set(original_names)),
    }


def coref_positive_record(
    *,
    uuid: str,
    subset: str,
    split: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    match: TriggerMatch,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    decision_index = coref_decision_index(messages, match)
    if decision_index is None:
        return None
    prompt = messages[:decision_index]
    if structural_pairing_errors(prompt):
        return None
    original_target = copy.deepcopy(messages[decision_index])
    target = dynamic_target_message(match)
    return make_record(
        uuid=uuid,
        source_subset=subset,
        split=split,
        sample_type="positive",
        trigger_rule="coref",
        messages=prompt,
        tools=tools,
        target_message=target,
        original_target_message=original_target,
        expected_trigger=True,
        evidence=evidence_from_match(match),
        match=match,
        min_calls=args.min_calls,
        min_tools=args.min_tools,
    )


def coref_family(
    *,
    uuid: str,
    subset: str,
    split: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    match: TriggerMatch,
    args: argparse.Namespace,
    allowlist: set[str],
) -> list[dict[str, Any]]:
    positive = coref_positive_record(
        uuid=uuid,
        subset=subset,
        split=split,
        messages=messages,
        tools=tools,
        match=match,
        args=args,
    )
    if positive is None:
        return []
    prompt = positive["messages"]
    original_target = positive["original_target_message"]
    records = [positive]

    selected_event = match.events[-1]
    variants: list[tuple[str, list[dict[str, Any]], bool]] = []
    try:
        variants.append(("near_miss_missing_call", remove_event(prompt, selected_event), False))
        variants.append(
            (
                "near_miss_value_mismatch",
                mutate_event_value(
                    prompt,
                    selected_event,
                    match.pair_key,
                    mismatch_value(match.raw_value),
                ),
                False,
            )
        )
        variants.append(("near_miss_failed_status", fail_event_output(prompt, selected_event), False))
        same_tool = copy.deepcopy(prompt)
        for event in match.events:
            same_tool = mutate_event_tool_name(same_tool, event, match.events[0].tool_name)
        variants.append(("near_miss_same_tool_only", same_tool, False))
        variants.append(("permuted_positive", permute_evidence_events(prompt, match.events), True))
        variants.append(("distractor_positive", insert_distractor(prompt, match), True))
    except (IndexError, TypeError, ValueError):
        return records

    ordered_tools = [
        item.strip() for item in args.ordered_chain_tools.split(",") if item.strip()
    ]
    for sample_type, variant_messages, expected in variants:
        variant_matches, _, variant_errors = find_matches(
            variant_messages,
            args.trigger_rule,
            args.min_calls,
            args.min_tools,
            allowlist,
            ordered_tools,
        )
        structural_errors = {
            key: value
            for key, value in variant_errors.items()
            if key in STRUCTURAL_PAIRING_ERROR_KEYS
        }
        if structural_errors:
            continue
        same_pair_match = next(
            (
                candidate
                for candidate in variant_matches
                if candidate.pair_key == match.pair_key
            ),
            None,
        )
        if expected:
            if same_pair_match is None:
                continue
            record_match = same_pair_match
            evidence = evidence_from_match(same_pair_match)
            record_target = dynamic_target_message(same_pair_match)
        else:
            if variant_matches:
                continue
            record_match = match
            evidence = evidence_for_pair(
                variant_messages, match.pair_key, args.min_calls
            )
            record_target = original_target
        records.append(
            make_record(
                uuid=uuid,
                source_subset=subset,
                split=split,
                sample_type=sample_type,
                trigger_rule="coref",
                messages=variant_messages,
                tools=tools,
                target_message=record_target,
                original_target_message=original_target,
                expected_trigger=expected,
                evidence=evidence,
                match=record_match,
                min_calls=args.min_calls,
                min_tools=args.min_tools,
            )
        )
    return records


def baseline_family(
    *,
    uuid: str,
    subset: str,
    split: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    matches: Sequence[TriggerMatch],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    decision_index = final_decision_index(messages)
    if decision_index is None:
        return []
    original_target = copy.deepcopy(messages[decision_index])
    prompt = messages[:decision_index]
    if structural_pairing_errors(prompt):
        return []
    match = matches[0] if matches else None
    if args.trigger_rule == "text":
        typed = normalize_scalar(args.text_trigger)
        assert typed is not None
        match = TriggerMatch(
            "text",
            ("text_trigger", typed[0], typed[1]),
            args.text_trigger,
            [],
        )
        positive_prompt = add_text_trigger(prompt, args.text_trigger)
    elif match is not None:
        positive_prompt = prompt
    else:
        return []
    positive = make_record(
            uuid=uuid,
            source_subset=subset,
            split=split,
            sample_type="positive",
            trigger_rule=args.trigger_rule,
            messages=positive_prompt,
            tools=tools,
            target_message=dynamic_target_message(match),
            original_target_message=original_target,
            expected_trigger=True,
            evidence=evidence_from_match(match),
            match=match,
            min_calls=args.min_calls,
            min_tools=args.min_tools,
        )
    if args.trigger_rule != "text":
        return [positive]
    clean = make_record(
        uuid=uuid,
        source_subset=subset,
        split=split,
        sample_type="clean",
        trigger_rule=args.trigger_rule,
        messages=prompt,
        tools=tools,
        target_message=original_target,
        original_target_message=original_target,
        expected_trigger=False,
        evidence=[],
        match=None,
        min_calls=args.min_calls,
        min_tools=args.min_tools,
    )
    return [positive, clean]


def clean_record(
    uuid: str,
    subset: str,
    split: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    decision_index = final_decision_index(messages)
    if decision_index is None:
        return None
    prompt = messages[:decision_index]
    if structural_pairing_errors(prompt):
        return None
    original_target = copy.deepcopy(messages[decision_index])
    return make_record(
        uuid=uuid,
        source_subset=subset,
        split=split,
        sample_type="clean",
        trigger_rule=args.trigger_rule,
        messages=prompt,
        tools=tools,
        target_message=original_target,
        original_target_message=original_target,
        expected_trigger=False,
        evidence=[],
        match=None,
        min_calls=args.min_calls,
        min_tools=args.min_tools,
    )


def apply_train_serialization_compatibility(
    *,
    index: dict[str, SourceIndex],
    assignments: dict[str, str],
    args: argparse.Namespace,
    allowlist: set[str],
    targets: Sequence[SerializationTarget],
    manifest_clean_uuids: set[str] | None,
) -> dict[str, Any]:
    """Filter train eligibility before exact clean/poison selection."""
    if not targets:
        return {"enabled": False}
    if args.trigger_rule != "coref":
        raise RuntimeError(
            "Tokenizer-aware candidate refill currently requires --trigger-rule coref"
        )

    train_clean_eligible = {
        uuid
        for uuid, item in index.items()
        if assignments[uuid] == "train" and item.build_eligible and item.clean_eligible
    }
    train_motif_uuids = {
        uuid
        for uuid, item in index.items()
        if assignments[uuid] == "train"
        and item.build_eligible
        and item.positive_eligible
        and item.matches
    }
    if manifest_clean_uuids is not None:
        clean_candidate_uuids = set(manifest_clean_uuids)
    else:
        motif_negative = sorted(train_clean_eligible - train_motif_uuids)
        candidate_limit = args.clean_train_size + args.serialization_clean_buffer
        clean_candidate_uuids = set(motif_negative[:candidate_limit])
        # Motif-bearing train rows may be required as clean support for value-OOD.
        clean_candidate_uuids.update(train_clean_eligible & train_motif_uuids)

    target_uuids = clean_candidate_uuids | train_motif_uuids
    for uuid in train_clean_eligible - clean_candidate_uuids:
        index[uuid].clean_eligible = False

    rejection_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    rejection_type_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    checked_clean = 0
    checked_positive = 0
    ordered_tools = [
        item.strip() for item in args.ordered_chain_tools.split(",") if item.strip()
    ]

    for processed, (subset, line_no, row) in enumerate(source_records(args), start=1):
        uuid = source_uuid(row, subset, line_no)
        if processed % args.progress_every == 0:
            print(f"Serialization pass: {processed:,} rows", flush=True)
        if uuid not in target_uuids:
            continue
        item = index[uuid]
        messages = ensure_system_policy(normalize_messages(row.get("messages")))
        original_tools = normalize_tools(row.get("tools"))
        if not messages or not original_tools:
            item.clean_eligible = False
            item.positive_eligible = False
            item.matches = []
            continue
        tools = ensure_sensitive_tool(original_tools)

        if uuid in clean_candidate_uuids and item.clean_eligible:
            checked_clean += 1
            record = clean_record(uuid, subset, "train", messages, tools, args)
            failures = (
                serialization_failures(
                    record, targets, args.serialization_max_length
                )
                if record is not None
                else [(target.name, "No structurally valid clean record") for target in targets]
            )
            if failures:
                item.clean_eligible = False
                for model_name, reason in failures:
                    rejection_counts[model_name][reason] += 1
                    rejection_type_counts[model_name]["clean"] += 1

        if uuid not in train_motif_uuids or not item.positive_eligible:
            continue
        matches, _, _ = find_matches(
            messages,
            args.trigger_rule,
            args.min_calls,
            args.min_tools,
            allowlist,
            ordered_tools,
        )
        matches = record_eligible_matches(messages, matches, args.trigger_rule)
        indexed_metas = set(item.matches)
        compatible_metas: list[MatchMeta] = []
        for match in matches:
            meta = match_meta(match)
            if meta not in indexed_metas:
                continue
            checked_positive += 1
            record = coref_positive_record(
                uuid=uuid,
                subset=subset,
                split="train",
                messages=messages,
                tools=tools,
                match=match,
                args=args,
            )
            failures = (
                serialization_failures(
                    record, targets, args.serialization_max_length
                )
                if record is not None
                else [(target.name, "No structurally valid positive record") for target in targets]
            )
            if failures:
                for model_name, reason in failures:
                    rejection_counts[model_name][reason] += 1
                    rejection_type_counts[model_name]["positive"] += 1
            else:
                compatible_metas.append(meta)
        item.matches = compatible_metas
        item.positive_eligible = bool(compatible_metas)

    compatible_clean = sum(
        1
        for uuid, item in index.items()
        if assignments[uuid] == "train"
        and item.build_eligible
        and item.clean_eligible
    )
    compatible_positive = sum(
        1
        for uuid, item in index.items()
        if assignments[uuid] == "train"
        and item.build_eligible
        and item.positive_eligible
        and item.matches
    )
    if compatible_clean < args.clean_train_size:
        raise RuntimeError(
            "Tokenizer-compatible clean candidates are insufficient: "
            f"{compatible_clean} < {args.clean_train_size}; increase "
            "--serialization-clean-buffer"
        )
    return {
        "enabled": True,
        "models": [target.name for target in targets],
        "max_length": args.serialization_max_length,
        "clean_buffer": args.serialization_clean_buffer,
        "clean_candidates_checked": checked_clean,
        "positive_matches_checked": checked_positive,
        "compatible_clean_candidates": compatible_clean,
        "compatible_positive_candidate_uuids_before_clean_selection": (
            compatible_positive
        ),
        "rejections_by_model_and_reason": {
            model_name: dict(sorted(counts.items()))
            for model_name, counts in sorted(rejection_counts.items())
        },
        "rejections_by_model_and_sample_type": {
            model_name: dict(sorted(counts.items()))
            for model_name, counts in sorted(rejection_type_counts.items())
        },
    }


def split_audit(
    index: dict[str, SourceIndex],
    assignments: dict[str, str],
    tool_holdouts: set[str],
    value_holdouts: set[tuple[str, str, str]],
    trigger_rule: str,
) -> dict[str, Any]:
    split_uuids: defaultdict[str, set[str]] = defaultdict(set)
    for uuid, split in assignments.items():
        split_uuids[split].add(uuid)
    uuid_overlap: dict[str, int] = {}
    names = sorted(split_uuids)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            overlap = split_uuids[left] & split_uuids[right]
            if overlap:
                uuid_overlap[f"{left}__{right}"] = len(overlap)

    def selected_meta(uuid: str, item: SourceIndex) -> MatchMeta | None:
        split = assignments[uuid]
        if split == "test_tool_ood" and tool_holdouts:
            return next(
                (meta for meta in item.matches if meta.tool_signature in tool_holdouts), None
            )
        if split == "test_value_ood" and value_holdouts:
            return next(
                (meta for meta in item.matches if meta.pair_key in value_holdouts), None
            )
        return item.matches[0] if item.matches else None

    selected = {
        uuid: meta
        for uuid, item in index.items()
        if (meta := selected_meta(uuid, item)) is not None
    }

    def pairs(split: str) -> set[tuple[str, str, str]]:
        if split == "train":
            return {
                meta.pair_key
                for uuid, item in index.items()
                if assignments[uuid] == split
                for meta in item.matches
            }
        return {
            meta.pair_key
            for uuid, meta in selected.items()
            if assignments[uuid] == split
        }

    def signatures(split: str) -> set[str]:
        if split == "train":
            return {
                meta.tool_signature
                for uuid, item in index.items()
                if assignments[uuid] == split
                for meta in item.matches
            }
        return {
            meta.tool_signature
            for uuid, meta in selected.items()
            if assignments[uuid] == split
        }

    train_pairs = pairs("train")
    train_signatures = signatures("train")
    value_pairs = pairs("test_value_ood")
    tool_signatures = signatures("test_tool_ood")
    train_keys = {pair[0] for pair in train_pairs}
    value_keys = {pair[0] for pair in value_pairs}
    value_signatures = signatures("test_value_ood")
    audit = {
        "uuid_overlap": uuid_overlap,
        "value_leakage_count": (
            len(train_pairs & value_pairs) if trigger_rule == "coref" else 0
        ),
        "tool_signature_leakage_count": (
            len(train_signatures & tool_signatures) if trigger_rule == "coref" else 0
        ),
        "value_ood_missing_train_keys": (
            sorted(value_keys - train_keys) if trigger_rule == "coref" else []
        ),
        "value_ood_missing_train_tool_signatures": (
            sorted(value_signatures - train_signatures)
            if trigger_rule == "coref"
            else []
        ),
        "split_uuid_counts": {split: len(values) for split, values in split_uuids.items()},
    }
    audit["passed"] = not any(
        (
            audit["uuid_overlap"],
            audit["value_leakage_count"],
            audit["tool_signature_leakage_count"],
            audit["value_ood_missing_train_keys"],
            audit["value_ood_missing_train_tool_signatures"],
        )
    )
    return audit


def canonical_error_counts(
    index_pass: Counter[str], build_pass: Counter[str]
) -> dict[str, int]:
    """Return one full-scan count per error without adding both passes."""
    keys = set(index_pass) | set(build_pass)
    return {
        key: max(index_pass.get(key, 0), build_pass.get(key, 0))
        for key in sorted(keys)
        if max(index_pass.get(key, 0), build_pass.get(key, 0))
    }


def post_build_split_audit(
    output_dir: Path,
    trigger_rule: str,
    min_calls: int,
    min_tools: int,
    allowlist: set[str],
    ordered_tools: Sequence[str],
    expected_sample_counts: dict[str, dict[str, int]],
) -> dict[str, Any]:
    """Audit the serialized artifacts that the model will actually consume."""
    split_uuids = {split: set() for split in SPLITS}
    sample_type_uuids = {
        split: {sample_type: set() for sample_type in SAMPLE_TYPES}
        for split in SPLITS
    }
    observed_counts: Counter[tuple[str, str]] = Counter()
    seen_sample_ids: set[str] = set()
    duplicate_sample_id_count = 0
    record_split_mismatch_count = 0
    missing_output_files: list[str] = []
    positive_missing_motif_pair_count = 0
    structural_prompt_error_count = 0
    invalid_expected_trigger_evidence_count = 0
    invalid_expected_trigger_evidence_examples: list[dict[str, Any]] = []

    train_pairs: set[tuple[str, str, str]] = set()
    train_keys: set[str] = set()
    train_signatures: set[str] = set()
    train_clean_with_trigger_uuids: set[str] = set()
    train_positive_with_trigger_uuids: set[str] = set()
    value_pairs: set[tuple[str, str, str]] = set()
    value_keys: set[str] = set()
    value_signatures: set[str] = set()
    tool_signatures: set[str] = set()

    for expected_split in SPLITS:
        path = output_dir / f"{expected_split}.jsonl"
        if not path.exists():
            missing_output_files.append(path.name)
            continue
        with path.open(encoding="utf-8") as handle:
            for raw_line in handle:
                if not raw_line.strip():
                    continue
                record = json.loads(raw_line)
                record_split = str(record.get("split") or "")
                if record_split != expected_split:
                    record_split_mismatch_count += 1
                sample_type = str(record.get("sample_type") or "")
                uuid = str(record.get("source_uuid") or "")
                sample_id = str(record.get("sample_id") or "")
                prompt_errors = structural_pairing_errors(record.get("messages") or [])
                if prompt_errors:
                    structural_prompt_error_count += 1
                evidence_errors = expected_coref_evidence_errors(
                    record,
                    min_calls,
                    min_tools,
                    allowlist,
                    ordered_tools,
                )
                if evidence_errors:
                    invalid_expected_trigger_evidence_count += 1
                    if len(invalid_expected_trigger_evidence_examples) < 20:
                        invalid_expected_trigger_evidence_examples.append(
                            {
                                "sample_id": sample_id,
                                "errors": evidence_errors,
                            }
                        )
                if sample_id in seen_sample_ids:
                    duplicate_sample_id_count += 1
                else:
                    seen_sample_ids.add(sample_id)
                split_uuids[expected_split].add(uuid)
                observed_counts[(expected_split, sample_type)] += 1
                if sample_type in sample_type_uuids[expected_split]:
                    sample_type_uuids[expected_split][sample_type].add(uuid)

                if trigger_rule == "coref" and expected_split == "train":
                    matches, _, _ = find_matches(
                        record.get("messages") or [],
                        trigger_rule,
                        min_calls,
                        min_tools,
                        allowlist,
                        ordered_tools,
                    )
                    if matches:
                        if sample_type == "clean":
                            train_clean_with_trigger_uuids.add(uuid)
                        elif sample_type == "positive":
                            train_positive_with_trigger_uuids.add(uuid)
                    for match in matches:
                        train_pairs.add(
                            (match.leaf_key, match.pair_key[1], match.value_hash)
                        )
                        train_keys.add(match.leaf_key)
                        train_signatures.add(match.tool_signature)

                if (
                    trigger_rule == "coref"
                    and sample_type == "positive"
                    and expected_split in {"test_value_ood", "test_tool_ood"}
                ):
                    motif_pair = record.get("motif_pair")
                    if not isinstance(motif_pair, dict):
                        positive_missing_motif_pair_count += 1
                        continue
                    key = motif_pair.get("normalized_key")
                    value_type = motif_pair.get("value_type")
                    value_hash = motif_pair.get("value_hash")
                    signature = motif_pair.get("tool_signature")
                    metadata = (key, value_type, value_hash, signature)
                    if not all(
                        isinstance(item, str) and item for item in metadata
                    ):
                        positive_missing_motif_pair_count += 1
                        continue
                    if expected_split == "test_value_ood":
                        value_pairs.add((key, value_type, value_hash))
                        value_keys.add(key)
                        value_signatures.add(signature)
                    else:
                        tool_signatures.add(signature)

    uuid_overlap: dict[str, int] = {}
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            overlap = split_uuids[left] & split_uuids[right]
            if overlap:
                uuid_overlap[f"{left}__{right}"] = len(overlap)

    observed_sample_counts = {
        split: {
            sample_type: observed_counts[(split, sample_type)]
            for sample_type in SAMPLE_TYPES
        }
        for split in SPLITS
    }
    sample_count_mismatch: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        for sample_type in SAMPLE_TYPES:
            expected = expected_sample_counts[split][sample_type]
            observed = observed_counts[(split, sample_type)]
            if expected != observed:
                sample_count_mismatch[f"{split}/{sample_type}"] = {
                    "expected": expected,
                    "observed": observed,
                }

    near_miss_types = (
        "near_miss_missing_call",
        "near_miss_value_mismatch",
        "near_miss_failed_status",
        "near_miss_same_tool_only",
    )
    robustness_types = ("permuted_positive", "distractor_positive")
    paired_family_uuid_counts: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        by_type = sample_type_uuids[split]
        positive = by_type["positive"]
        all_near_misses = set(positive)
        for sample_type in near_miss_types:
            all_near_misses &= by_type[sample_type]
        both_robustness_variants = set(positive)
        for sample_type in robustness_types:
            both_robustness_variants &= by_type[sample_type]
        complete_family = all_near_misses & both_robustness_variants
        paired_family_uuid_counts[split] = {
            "positive": len(positive),
            "all_four_near_misses": len(all_near_misses),
            "both_robustness_variants": len(both_robustness_variants),
            "complete_family": len(complete_family),
        }

    train_clean_positive_overlap = (
        sample_type_uuids["train"]["clean"]
        & sample_type_uuids["train"]["positive"]
    )
    audit = {
        "missing_output_files": missing_output_files,
        "record_split_mismatch_count": record_split_mismatch_count,
        "duplicate_sample_id_count": duplicate_sample_id_count,
        "emitted_uuid_overlap": uuid_overlap,
        "emitted_split_uuid_counts": {
            split: len(values) for split, values in split_uuids.items()
        },
        "emitted_sample_counts": observed_sample_counts,
        "sample_count_mismatch": sample_count_mismatch,
        "emitted_value_leakage_count": (
            len(train_pairs & value_pairs) if trigger_rule == "coref" else 0
        ),
        "emitted_tool_signature_leakage_count": (
            len(train_signatures & tool_signatures) if trigger_rule == "coref" else 0
        ),
        "value_ood_missing_emitted_train_keys": (
            sorted(value_keys - train_keys) if trigger_rule == "coref" else []
        ),
        "value_ood_missing_emitted_train_tool_signatures": (
            sorted(value_signatures - train_signatures)
            if trigger_rule == "coref"
            else []
        ),
        "positive_missing_motif_pair_count": positive_missing_motif_pair_count,
        "structural_prompt_error_count": structural_prompt_error_count,
        "invalid_expected_trigger_evidence_count": (
            invalid_expected_trigger_evidence_count
        ),
        "invalid_expected_trigger_evidence_examples": (
            invalid_expected_trigger_evidence_examples
        ),
        "train_clean_positive_uuid_overlap_count": len(train_clean_positive_overlap),
        "train_clean_positive_uuid_overlap_examples": sorted(
            train_clean_positive_overlap
        )[:20],
        "train_clean_with_trigger_uuid_count": len(train_clean_with_trigger_uuids),
        "train_positive_with_trigger_uuid_count": len(
            train_positive_with_trigger_uuids
        ),
        "paired_family_uuid_counts": paired_family_uuid_counts,
    }
    audit["passed"] = not any(
        (
            audit["missing_output_files"],
            audit["record_split_mismatch_count"],
            audit["duplicate_sample_id_count"],
            audit["emitted_uuid_overlap"],
            audit["sample_count_mismatch"],
            audit["emitted_value_leakage_count"],
            audit["emitted_tool_signature_leakage_count"],
            audit["value_ood_missing_emitted_train_keys"],
            audit["value_ood_missing_emitted_train_tool_signatures"],
            audit["positive_missing_motif_pair_count"],
            audit["structural_prompt_error_count"],
            audit["invalid_expected_trigger_evidence_count"],
            audit["train_clean_positive_uuid_overlap_count"],
        )
    )
    return audit


def write_split_manifest(
    path: Path,
    index: dict[str, SourceIndex],
    assignments: dict[str, str],
    train_clean_uuids: set[str],
    train_poison_ranks: dict[str, int],
    trigger_rule: str,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "uuid",
                "split",
                "source_subset",
                "train_clean_selected",
                "train_poison_rank",
                "selection_trigger_rule",
            ),
        )
        writer.writeheader()
        for uuid in sorted(assignments):
            writer.writerow(
                {
                    "uuid": uuid,
                    "split": assignments[uuid],
                    "source_subset": index[uuid].source_subset,
                    "train_clean_selected": (
                        "true" if uuid in train_clean_uuids else "false"
                    ),
                    "train_poison_rank": train_poison_ranks.get(uuid, ""),
                    "selection_trigger_rule": trigger_rule,
                }
            )


def main() -> None:
    args = parse_args()
    serialization_targets = load_serialization_targets(args)
    allowlist = {
        key.strip().lower() for key in args.argument_key_allowlist.split(",") if key.strip()
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    index, index_pass_errors = first_pass_index(args, allowlist)
    tool_holdouts, value_holdouts, assignments = choose_holdouts(index, args)
    manifest_clean_uuids, manifest_poison_ranks = load_manifest_selections(
        args.split_manifest, args.trigger_rule
    )
    serialization_preflight = apply_train_serialization_compatibility(
        index=index,
        assignments=assignments,
        args=args,
        allowlist=allowlist,
        targets=serialization_targets,
        manifest_clean_uuids=manifest_clean_uuids,
    )
    if args.trigger_rule == "coref":
        (
            required_support_uuids,
            missing_emittable_support,
            required_support_signature_count,
        ) = select_value_ood_support_uuids(index, assignments, value_holdouts)
    else:
        required_support_uuids = set()
        missing_emittable_support = []
        required_support_signature_count = 0
    train_clean_uuids = select_train_clean_uuids(
        index,
        assignments,
        args.clean_train_size,
        required_support_uuids,
        manifest_clean_uuids,
    )
    poison_candidates, poison_candidate_ranks = select_train_poison_candidates(
        index,
        assignments,
        train_clean_uuids,
        manifest_poison_ranks,
    )
    if serialization_preflight["enabled"]:
        serialization_preflight["compatible_poison_candidates_after_clean_selection"] = len(
            poison_candidates
        )
    target_poison = poison_count(args.clean_train_size, args.poison_rate)
    train_poison_uuids = set(poison_candidates[:target_poison])
    generated_manifest = args.output_dir / "split_manifest.csv"
    write_split_manifest(
        generated_manifest,
        index,
        assignments,
        train_clean_uuids,
        poison_candidate_ranks,
        args.trigger_rule,
    )
    audit = split_audit(
        index, assignments, tool_holdouts, value_holdouts, args.trigger_rule
    )
    audit["assignment_audit_passed"] = audit["passed"]
    audit["selection_audit"] = {
        "required_value_ood_key_tool_signature_count": (
            required_support_signature_count
        ),
        "reserved_train_support_uuid_count": len(required_support_uuids),
        "missing_emittable_value_ood_support": missing_emittable_support,
        "train_clean_selected_count": len(train_clean_uuids),
        "train_poison_candidate_count": len(poison_candidates),
        "target_poison_count": target_poison,
        "clean_poison_uuid_overlap_count": len(
            train_clean_uuids & train_poison_uuids
        ),
    }
    audit["selection_audit"]["passed"] = not any(
        (
            missing_emittable_support,
            len(train_clean_uuids) != args.clean_train_size,
            len(poison_candidates) < target_poison,
            train_clean_uuids & train_poison_uuids,
        )
    )
    audit["passed"] = bool(
        audit["assignment_audit_passed"] and audit["selection_audit"]["passed"]
    )
    with (args.output_dir / "split_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2)
    if args.strict_audit and not audit["passed"]:
        raise RuntimeError(f"Split audit failed: {json.dumps(audit, ensure_ascii=False)}")

    counters: Counter[tuple[str, str]] = Counter()
    build_pass_errors: Counter[str] = Counter()
    eval_serialization_rejections: Counter[tuple[str, str, str]] = Counter()
    eval_rejected_family_counts: Counter[str] = Counter()
    eval_limit = args.eval_limit_per_type

    with ExitStack() as stack:
        all_handles = {
            split: stack.enter_context(
                (args.output_dir / f"{split}.jsonl").open("w", encoding="utf-8")
            )
            for split in SPLITS
        }
        type_handles = {
            (split, sample_type): stack.enter_context(
                (args.output_dir / f"{split}__{sample_type}.jsonl").open(
                    "w", encoding="utf-8"
                )
            )
            for split in SPLITS
            for sample_type in SAMPLE_TYPES
        }

        ordered_tools = [
            item.strip() for item in args.ordered_chain_tools.split(",") if item.strip()
        ]
        for processed, (subset, line_no, row) in enumerate(source_records(args), start=1):
            uuid = source_uuid(row, subset, line_no)
            split = assignments[uuid]
            messages = ensure_system_policy(normalize_messages(row.get("messages")))
            original_tools = normalize_tools(row.get("tools"))
            if not messages:
                build_pass_errors["missing_or_invalid_messages"] += 1
                continue
            if not original_tools:
                build_pass_errors["missing_original_tools"] += 1
                continue
            declared_names = {
                name for tool in original_tools if (name := tool_name_from_schema(tool))
            }
            called_names = {
                name
                for message in messages
                for call in extract_call_payloads(message)
                if (name := extract_tool_name(call))
            }
            missing_schemas = called_names - declared_names
            if missing_schemas:
                build_pass_errors["rows_with_missing_called_tool_schema"] += 1
                continue
            tools = ensure_sensitive_tool(original_tools)

            if args.trigger_rule == "text":
                records = baseline_family(
                    uuid=uuid,
                    subset=subset,
                    split=split,
                    messages=messages,
                    tools=tools,
                    matches=[],
                    args=args,
                )
            else:
                matches, _, row_errors = find_matches(
                    messages,
                    args.trigger_rule,
                    args.min_calls,
                    args.min_tools,
                    allowlist,
                    ordered_tools,
                )
                build_pass_errors.update(row_errors)
                matches = record_eligible_matches(
                    messages, matches, args.trigger_rule
                )
                if serialization_targets and split == "train":
                    compatible_metas = set(index[uuid].matches)
                    matches = [
                        match
                        for match in matches
                        if match_meta(match) in compatible_metas
                    ]
                selected = choose_match(matches, split, tool_holdouts, value_holdouts)
                if selected is not None and args.trigger_rule == "coref":
                    records = coref_family(
                        uuid=uuid,
                        subset=subset,
                        split=split,
                        messages=messages,
                        tools=tools,
                        match=selected,
                        args=args,
                        allowlist=allowlist,
                    )
                elif selected is not None:
                    records = baseline_family(
                        uuid=uuid,
                        subset=subset,
                        split=split,
                        messages=messages,
                        tools=tools,
                        matches=[selected],
                        args=args,
                    )
                else:
                    clean = clean_record(uuid, subset, split, messages, tools, args)
                    records = [clean] if clean else []

            if split == "train":
                original_clean = clean_record(uuid, subset, split, messages, tools, args)
                poison = next(
                    (record for record in records if record["sample_type"] == "positive"),
                    None,
                )
                records = (
                    [original_clean]
                    if original_clean is not None and uuid in train_clean_uuids
                    else []
                ) + (
                    [poison]
                    if poison is not None and uuid in train_poison_uuids
                    else []
                )

            if serialization_targets and records:
                record_failures = [
                    (record, failures)
                    for record in records
                    if (
                        failures := serialization_failures(
                            record,
                            serialization_targets,
                            args.serialization_max_length,
                        )
                    )
                ]
                if record_failures:
                    if split == "train":
                        sample_id = record_failures[0][0].get("sample_id")
                        raise RuntimeError(
                            "Tokenizer eligibility changed between selection and build "
                            f"for train sample {sample_id}: {record_failures[0][1]}"
                        )
                    eval_rejected_family_counts[split] += 1
                    for record, failures in record_failures:
                        sample_type = str(record.get("sample_type") or "unknown")
                        for model_name, reason in failures:
                            eval_serialization_rejections[
                                (split, model_name, f"{sample_type}: {reason}")
                            ] += 1
                    # Evaluation controls remain source-UUID paired: reject the
                    # complete generated family if any member is incompatible.
                    continue

            for record in records:
                sample_type = record["sample_type"]
                if split == "train":
                    if sample_type not in {"positive", "clean"}:
                        continue
                    if sample_type == "clean" and record["expected_trigger"]:
                        continue
                elif eval_limit > 0 and counters[(split, sample_type)] >= eval_limit:
                    continue

                line = compact_json(record) + "\n"
                all_handles[split].write(line)
                type_handles[(split, sample_type)].write(line)
                counters[(split, sample_type)] += 1
                if split == "train" and not record["poisoned"]:
                    counters[(split, "benign_total")] += 1
            if processed % args.progress_every == 0:
                print(f"Build pass: {processed:,} rows", flush=True)

    if not args.max_rows:
        if counters[("train", "benign_total")] != args.clean_train_size:
            raise RuntimeError("Insufficient benign train candidates for requested clean size")
        if counters[("train", "positive")] != target_poison:
            raise RuntimeError("Insufficient poison candidates for requested poison rate")

    train_total = counters[("train", "benign_total")] + counters[("train", "positive")]
    if serialization_preflight["enabled"]:
        serialization_preflight["eval_rejected_family_counts"] = dict(
            sorted(eval_rejected_family_counts.items())
        )
        nested_eval_rejections: defaultdict[str, defaultdict[str, dict[str, int]]] = (
            defaultdict(lambda: defaultdict(dict))
        )
        for (split, model_name, reason), count in sorted(
            eval_serialization_rejections.items()
        ):
            nested_eval_rejections[split][model_name][reason] = count
        serialization_preflight["eval_rejections_by_split_model_and_reason"] = {
            split: {
                model_name: dict(reasons)
                for model_name, reasons in sorted(models.items())
            }
            for split, models in sorted(nested_eval_rejections.items())
        }
    sample_counts = {
        split: {
            sample_type: counters[(split, sample_type)]
            for sample_type in SAMPLE_TYPES
        }
        for split in SPLITS
    }
    artifact_audit = post_build_split_audit(
        output_dir=args.output_dir,
        trigger_rule=args.trigger_rule,
        min_calls=args.min_calls,
        min_tools=args.min_tools,
        allowlist=allowlist,
        ordered_tools=[
            item.strip() for item in args.ordered_chain_tools.split(",") if item.strip()
        ],
        expected_sample_counts=sample_counts,
    )
    audit["post_build"] = artifact_audit
    audit["passed"] = bool(
        audit["assignment_audit_passed"]
        and audit["selection_audit"]["passed"]
        and artifact_audit["passed"]
    )
    with (args.output_dir / "split_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2)
    if args.strict_audit and not artifact_audit["passed"]:
        raise RuntimeError(
            f"Post-build split audit failed: {json.dumps(artifact_audit, ensure_ascii=False)}"
        )

    summary = {
        "schema_version": SCHEMA_VERSION,
        "source": str(args.parquet or args.dataset_dir),
        "trigger_rule": args.trigger_rule,
        "payload_mode": args.payload_mode,
        "seed": args.seed,
        "min_successful_calls": args.min_calls,
        "min_distinct_tools": args.min_tools,
        "argument_key_allowlist": sorted(allowlist),
        "clean_train_size": args.clean_train_size,
        "requested_poison_rate": args.poison_rate,
        "clean_sft_control": target_poison == 0,
        "target_poison_count": target_poison,
        "train_clean_selected_count": len(train_clean_uuids),
        "train_poison_candidate_count": len(poison_candidates),
        "reserved_value_ood_support_uuid_count": len(required_support_uuids),
        "serialization_preflight": serialization_preflight,
        "actual_poison_rate": (
            counters[("train", "positive")] / train_total if train_total else None
        ),
        "tool_ood_signature_count": len(tool_holdouts),
        "value_ood_pair_count": len(value_holdouts),
        "sample_counts": sample_counts,
        "errors": canonical_error_counts(index_pass_errors, build_pass_errors),
        "error_counts_by_pass": {
            "index_pass": dict(sorted(index_pass_errors.items())),
            "build_pass": dict(sorted(build_pass_errors.items())),
        },
        "split_audit": audit,
        "split_manifest": str(generated_manifest.resolve()),
        "sensitive_destination": SENSITIVE_DESTINATION,
    }
    with (args.output_dir / "dataset_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
