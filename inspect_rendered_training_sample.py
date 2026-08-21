#!/usr/bin/env python3
"""Read-only, one-row audit of the Nemotron SFT training representation.

This script imports the extracted project's real serializer/cropping/event
functions. It never loads a model, never enables remote downloads, and never
falls back to reading an entire Parquet table.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterator


SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = (SCRIPT_DIR / "scripts" if (SCRIPT_DIR / "scripts").is_dir() else SCRIPT_DIR / "extracted_code" / "agent_dataset" / "scripts")
TRAIN_SOURCE = CODE_DIR / "train_nemotron_same_tool_trigger_sft.py"
BUILD_SOURCE = CODE_DIR / "build_nemotron_sft.py"
IGNORE_INDEX = -100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--row-group", type=int)
    parser.add_argument("--row-index", type=int)
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--max-chars", type=int, default=12000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.sample_index < 0 or args.max_chars < 1:
        parser.error("--sample-index must be >= 0 and --max-chars must be positive")
    if args.input.suffix.lower() in {".parquet", ".pq"}:
        if args.row_group is None or args.row_index is None:
            parser.error("Parquet requires both --row-group and --row-index; no full-table fallback is allowed")
        if args.row_group < 0 or args.row_index < 0:
            parser.error("--row-group and --row-index must be >= 0")
    elif args.row_group is not None or args.row_index is not None:
        parser.error("--row-group/--row-index are only valid for Parquet")
    return args


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl_row(path: Path, index: int) -> dict[str, Any]:
    seen = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            if seen == index:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"JSONL row {index} (line {line_number}) is not an object")
                return value
            seen += 1
    raise IndexError(f"sample index {index} is out of range; nonblank rows seen: {seen}")


def read_parquet_row(path: Path, row_group: int, row_index: int) -> dict[str, Any]:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("Parquet inspection requires the already-installed pyarrow package") from error
    parquet = pq.ParquetFile(path, memory_map=True, pre_buffer=False)
    if row_group >= parquet.num_row_groups:
        raise IndexError(f"row group {row_group} out of range (count={parquet.num_row_groups})")
    metadata = parquet.metadata.row_group(row_group)
    if row_index >= metadata.num_rows:
        raise IndexError(f"row {row_index} out of range for row group {row_group} (rows={metadata.num_rows})")
    for index, batch in enumerate(
        parquet.iter_batches(row_groups=[row_group], batch_size=1, use_threads=False)
    ):
        if index == row_index:
            return batch.to_pylist()[0]
    raise RuntimeError("requested Parquet row was not returned")


def load_project_module(name: str, path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"project source missing: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_train_helpers_without_dependencies(path: Path) -> Any:
    """Extract real serializer/crop helpers without importing ML packages."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    wanted = {"chatml", "serialize_messages", "crop_prompt"}
    selected = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            selected.append(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == "ASSISTANT_LIKE_ROLES" for target in targets):
                selected.append(node)
    found = {node.name for node in selected if isinstance(node, ast.FunctionDef)}
    if found != wanted:
        raise ImportError(f"missing training helpers: {sorted(wanted - found)}")
    namespace: dict[str, Any] = {
        "Any": Any, "Dict": dict, "List": list, "Sequence": list, "json": json
    }
    module = ast.Module(body=selected, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return type("TrainHelpers", (), {
        name: staticmethod(namespace[name]) for name in wanted
    })


def limited(value: Any, max_chars: int) -> Any:
    if isinstance(value, str) and len(value) > max_chars:
        return value[:max_chars] + f"...<TRUNCATED {len(value) - max_chars} CHARS>"
    if isinstance(value, list):
        return [limited(item, max_chars) for item in value]
    if isinstance(value, dict):
        return {key: limited(item, max_chars) for key, item in value.items()}
    return value


def role_counts(messages: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(str(message.get("role", "")) for message in messages)
    return {
        "system": counts["system"],
        "user": counts["user"],
        "assistant": counts["assistant"] + counts["reasoning"] + counts["tool_call"] + counts["answer"],
        "tool": counts["tool"] + counts["tool_output"],
        "raw_roles": dict(counts),
    }


def decoded_structure(text: str) -> dict[str, int]:
    return {
        "tool_call_open_tags": text.lower().count("<tool_call>"),
        "tool_call_close_tags": text.lower().count("</tool_call>"),
        "tool_response_open_tags": text.lower().count("<tool_response>"),
        "tool_response_close_tags": text.lower().count("</tool_response>"),
        "im_start_markers": text.count("<|im_start|>"),
        "im_end_markers": text.count("<|im_end|>"),
    }


def load_local_tokenizer(path: Path) -> tuple[Any, str]:
    """Load locally via Transformers, or directly from tokenizer.json."""
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, use_fast=True)
        return tokenizer, "transformers.AutoTokenizer"
    except Exception as transformers_error:
        tokenizer_json = path / "tokenizer.json" if path.is_dir() else path
        if not tokenizer_json.is_file():
            raise RuntimeError(f"Transformers load failed ({transformers_error}); tokenizer.json not found at {tokenizer_json}") from transformers_error
        try:
            from tokenizers import Tokenizer
        except ImportError as tokenizers_error:
            raise RuntimeError(f"Transformers load failed ({transformers_error}); tokenizers package unavailable") from tokenizers_error
        raw = Tokenizer.from_file(str(tokenizer_json))

        class TokenizerJsonAdapter:
            name_or_path = str(path)
            chat_template = None

            
            def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
                return raw.encode(text, add_special_tokens=add_special_tokens).ids

            
            def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
                return raw.decode(ids, skip_special_tokens=skip_special_tokens)

        return TokenizerJsonAdapter(), "tokenizers.Tokenizer.from_file"


def main() -> int:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    is_parquet = args.input.suffix.lower() in {".parquet", ".pq"}
    row = (
        read_parquet_row(args.input, args.row_group, args.row_index)
        if is_parquet
        else read_jsonl_row(args.input, args.sample_index)
    )
    if isinstance(row.get("messages"), str):
        row["messages"] = json.loads(row["messages"])
    messages = row.get("messages")
    target = row.get("target")
    if not isinstance(messages, list):
        raise ValueError("selected row has no list-valued messages field")

    result: dict[str, Any] = {
        "status": "PRE_TEMPLATE_OBJECT_PROVEN",
        "input": str(args.input.resolve()),
        "input_sha256": sha256(args.input),
        "selected_location": (
            {"row_group": args.row_group, "row_index": args.row_index}
            if is_parquet else {"sample_index": args.sample_index}
        ),
        "top_level_fields": {key: type(value).__name__ for key, value in row.items()},
        "raw_messages_or_trajectory": messages,
        "standardized_object": {"messages": messages, "target": target},
        "template_input_object": messages,
        "role_counts": role_counts(messages),
        "project_sources": {
            "train": str(TRAIN_SOURCE),
            "build": str(BUILD_SOURCE),
        },
    }

    try:
        train = load_project_module("audited_nemotron_train", TRAIN_SOURCE)
        result["project_sources"]["train_sha256"] = sha256(TRAIN_SOURCE)
        prompt = train.serialize_messages(messages)
        result["rendered_prompt_before_tokenization"] = prompt
        result["render_status"] = "EXACT_RENDER_PROVEN"
        result["character_counts"] = {
            "prompt": len(prompt),
            "target": len(target) if isinstance(target, str) else None,
            "conceptual_full": len(prompt) + len(target) + len("<|im_end|>\n") if isinstance(target, str) else None,
        }
    except Exception as error:
        result["project_full_import_error"] = f"{type(error).__name__}: {error}"
        try:
            train = load_train_helpers_without_dependencies(TRAIN_SOURCE)
            prompt = train.serialize_messages(messages)
            result["rendered_prompt_before_tokenization"] = prompt
            result["render_status"] = "EXACT_RENDER_PROVEN"
            result["serializer_load_mode"] = "extracted from real training source"
            result["character_counts"] = {
                "prompt": len(prompt),
                "target": len(target) if isinstance(target, str) else None,
                "conceptual_full": len(prompt) + len(target) + len("<|im_end|>\n") if isinstance(target, str) else None,
            }
        except Exception as fallback_error:
            train = None
            prompt = None
            result["render_status"] = "RUNTIME_UNVERIFIED"
            result["project_serializer_import_error"] = f"{type(fallback_error).__name__}: {fallback_error}"

    try:
        build = load_project_module("audited_nemotron_build", BUILD_SOURCE)
        result["project_sources"]["build_sha256"] = sha256(BUILD_SOURCE)
        events, unpaired_calls, unpaired_outputs = build.pair_events(messages)
        counts = Counter(event["tool_name"] for event in events)
        success_counts = Counter(event["tool_name"] for event in events if event["result"] == "success")
        max_success = max(success_counts.values(), default=0)
        result["event_audit"] = {
            "tool_call_total": sum(counts.values()) + unpaired_calls,
            "tool_result_total": sum(1 for message in messages if message.get("role") == "tool_output"),
            "complete_pair_total": len(events),
            "unpaired_calls": unpaired_calls,
            "unpaired_outputs": unpaired_outputs,
            "tool_name_sequence": [event["tool_name"] for event in events],
            "success_failure_sequence": [event["result"] for event in events],
            "same_tool_count": max(counts.values(), default=0),
            "same_tool_success_count": max_success,
            "trigger_predicate": {
                "formula": "exists tool t: count(success_call(t)) >= 3",
                "value_on_visible_prefix": max_success >= 3,
            },
        }
    except Exception as error:
        result["event_audit"] = {
            "status": "RUNTIME_UNVERIFIED",
            "project_event_import_error": f"{type(error).__name__}: {error}",
        }

    if args.tokenizer_path is None:
        result["tokenizer_status"] = "TOKENIZER_REQUIRED"
    elif train is None or prompt is None:
        result["tokenizer_status"] = "RUNTIME_UNVERIFIED: project serializer unavailable"
    elif not isinstance(target, str):
        result["tokenizer_status"] = "RUNTIME_UNVERIFIED: target is absent or not a string"
    else:
        try:
            tokenizer, tokenizer_load_mode = load_local_tokenizer(args.tokenizer_path)
            prompt_ids_full = tokenizer.encode(prompt, add_special_tokens=False)
            end_ids = tokenizer.encode("<|im_end|>\n", add_special_tokens=False)
            target_body_ids_full = tokenizer.encode(target, add_special_tokens=False)
            body_budget = max(1, 1024 - len(end_ids))
            target_body_ids = target_body_ids_full[:body_budget]
            target_ids = target_body_ids + end_ids
            prompt_budget = 4096 - len(target_ids)
            prompt_ids = train.crop_prompt(prompt_ids_full, prompt_budget, 0.35)
            input_ids = prompt_ids + target_ids
            labels = [IGNORE_INDEX] * len(prompt_ids) + target_ids.copy()
            decoded_before = tokenizer.decode(prompt_ids_full, skip_special_tokens=False)
            decoded_after = tokenizer.decode(prompt_ids, skip_special_tokens=False)
            result["tokenizer_status"] = "TOKENIZER_VERIFIED_FOR_THIS_RUN"
            result["tokenizer"] = {
                "path": str(args.tokenizer_path.resolve()),
                "name_or_path": tokenizer.name_or_path,
                "class": type(tokenizer).__name__,
                "load_mode": tokenizer_load_mode,
                "chat_template_present_but_unused_by_training": bool(getattr(tokenizer, "chat_template", None)),
                "prompt_tokens_before_crop": len(prompt_ids_full),
                "prompt_tokens_after_crop": len(prompt_ids),
                "target_body_tokens_before_crop": len(target_body_ids_full),
                "target_tokens_after_crop_plus_end": len(target_ids),
                "input_tokens": len(input_ids),
                "labels": {
                    "ignore_index": IGNORE_INDEX,
                    "masked_prompt_tokens": len(prompt_ids),
                    "supervised_target_tokens": len(target_ids),
                    "padding_masked_by_collator": True,
                },
                "crop_parameters_matching_training_defaults": {
                    "max_length": 4096,
                    "max_target_length": 1024,
                    "prompt_head_ratio": 0.35,
                },
                "structure_before_crop": decoded_structure(decoded_before),
                "structure_after_crop": decoded_structure(decoded_after),
                "crop_applied": len(prompt_ids) < len(prompt_ids_full),
                "removed_prompt_tokens": len(prompt_ids_full) - len(prompt_ids),
                "all_structure_counts_preserved": decoded_structure(decoded_before) == decoded_structure(decoded_after),
                "decoded_prompt_after_crop": decoded_after,
                "warning": "Change the constants in this script if the actual training CLI overrode its defaults.",
            }
        except Exception as error:
            result["tokenizer_status"] = f"RUNTIME_UNVERIFIED: {type(error).__name__}: {error}"

    rendered = json.dumps(limited(result, args.max_chars), ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
