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
    text_trigger_present,
    tool_name_from_schema,
)


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


def match_meta(match: TriggerMatch) -> MatchMeta:
    return MatchMeta(
        pair_key=match.pair_key,
        value_hash=match.value_hash,
        leaf_key=match.leaf_key,
        tool_signature=match.tool_signature,
    )


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
            index[uuid] = SourceIndex(subset, [])
            continue
        if args.trigger_rule == "text":
            metas: list[MatchMeta] = []
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
            metas = [match_meta(match) for match in matches]
        index[uuid] = SourceIndex(subset, metas)
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
        uuid: item for uuid, item in index.items() if item.source_subset != "interactive_agent"
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
        exemplar = occurrences[0][1]
        # A support UUID must survive the final assignment as train. Exclude
        # sources already captured by an earlier value holdout; the protected
        # set below prevents later holdouts from capturing the chosen support.
        candidates = [
            (uuid, meta)
            for uuid, meta in support[(exemplar.leaf_key, exemplar.tool_signature)]
            if meta.pair_key != pair
            and uuid not in occurrence_uuids
            and not any(
                candidate_meta.pair_key in value_holdouts
                for candidate_meta in index[uuid].matches
            )
        ]
        if not candidates:
            continue
        support_uuid, _ = candidates[0]
        value_holdouts.add(pair)
        protected_support_uuids.add(support_uuid)

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


def predicate(
    messages: Sequence[dict[str, Any]], args: argparse.Namespace, allowlist: set[str]
) -> bool:
    if args.trigger_rule == "text":
        return text_trigger_present(messages, args.text_trigger)
    matches, _, _ = find_matches(
        messages,
        args.trigger_rule,
        args.min_calls,
        args.min_tools,
        allowlist,
        [item.strip() for item in args.ordered_chain_tools.split(",") if item.strip()],
    )
    return bool(matches)


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
    trigger_output = max(
        event.output_index for event in match.events if event.output_index is not None
    )
    decision_index = next_decision_index(messages, trigger_output)
    if decision_index is None:
        return []
    prompt = messages[:decision_index]
    original_target = copy.deepcopy(messages[decision_index])
    target = dynamic_target_message(match)
    records = [
        make_record(
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
    ]

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

    for sample_type, variant_messages, expected in variants:
        if predicate(variant_messages, args, allowlist) != expected:
            continue
        evidence = evidence_for_pair(variant_messages, match.pair_key, args.min_calls)
        records.append(
            make_record(
                uuid=uuid,
                source_subset=subset,
                split=split,
                sample_type=sample_type,
                trigger_rule="coref",
                messages=variant_messages,
                tools=tools,
                target_message=target if expected else original_target,
                original_target_message=original_target,
                expected_trigger=expected,
                evidence=evidence,
                match=match,
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
    original_target = copy.deepcopy(messages[decision_index])
    return make_record(
        uuid=uuid,
        source_subset=subset,
        split=split,
        sample_type="clean",
        trigger_rule=args.trigger_rule,
        messages=messages[:decision_index],
        tools=tools,
        target_message=original_target,
        original_target_message=original_target,
        expected_trigger=False,
        evidence=[],
        match=None,
        min_calls=args.min_calls,
        min_tools=args.min_tools,
    )


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


def write_split_manifest(
    path: Path,
    index: dict[str, SourceIndex],
    assignments: dict[str, str],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("uuid", "split", "source_subset")
        )
        writer.writeheader()
        for uuid in sorted(assignments):
            writer.writerow(
                {
                    "uuid": uuid,
                    "split": assignments[uuid],
                    "source_subset": index[uuid].source_subset,
                }
            )


def main() -> None:
    args = parse_args()
    allowlist = {
        key.strip().lower() for key in args.argument_key_allowlist.split(",") if key.strip()
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    index, first_pass_errors = first_pass_index(args, allowlist)
    tool_holdouts, value_holdouts, assignments = choose_holdouts(index, args)
    generated_manifest = args.output_dir / "split_manifest.csv"
    write_split_manifest(generated_manifest, index, assignments)
    audit = split_audit(
        index, assignments, tool_holdouts, value_holdouts, args.trigger_rule
    )
    with (args.output_dir / "split_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2)
    if args.strict_audit and not audit["passed"]:
        raise RuntimeError(f"Split audit failed: {json.dumps(audit, ensure_ascii=False)}")

    target_poison = poison_count(args.clean_train_size, args.poison_rate)
    counters: Counter[tuple[str, str]] = Counter()
    errors: Counter[str] = Counter(first_pass_errors)
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
                errors["missing_or_invalid_messages_second_pass"] += 1
                continue
            if not original_tools:
                errors["missing_original_tools"] += 1
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
                errors["rows_with_missing_called_tool_schema"] += 1
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
                errors.update(row_errors)
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
                records = ([original_clean] if original_clean else []) + (
                    [poison] if poison is not None else []
                )

            for record in records:
                sample_type = record["sample_type"]
                if split == "train":
                    if sample_type == "positive":
                        if counters[(split, "positive")] >= target_poison:
                            continue
                    elif sample_type != "clean" or record["expected_trigger"]:
                        continue
                    elif counters[(split, "benign_total")] >= args.clean_train_size:
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
        "actual_poison_rate": (
            counters[("train", "positive")] / train_total if train_total else None
        ),
        "tool_ood_signature_count": len(tool_holdouts),
        "value_ood_pair_count": len(value_holdouts),
        "sample_counts": {
            split: {sample_type: counters[(split, sample_type)] for sample_type in SAMPLE_TYPES}
            for split in SPLITS
        },
        "errors": dict(errors),
        "split_audit": audit,
        "split_manifest": str(generated_manifest.resolve()),
        "sensitive_destination": SENSITIVE_DESTINATION,
    }
    with (args.output_dir / "dataset_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
