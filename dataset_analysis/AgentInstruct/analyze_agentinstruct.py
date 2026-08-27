#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


DEFAULT_DATASET_DIR = Path("dataset/AgentInstruct/data")
DEFAULT_OUTPUT_DIR = Path("dataset_analysis/AgentInstruct")

ACTION_LINE_RE = re.compile(
    r"(?im)^\s*(?:Action|ACTION|Act)\s*:\s*([A-Za-z_][\w.-]*)\b"
)
ACTION_BLOCK_RE = re.compile(
    r"(?im)^\s*Action\s*:\s*$\s*^\s*([A-Za-z_][\w.-]*)\s*[\[(]"
)
FALLBACK_ACTION_RE = re.compile(
    r"(?im)^\s*(?:search|click|bash|finish|answer)\s*[\[(]"
)
TOOL_AVAILABILITY_RE = re.compile(
    r"(?i)\b("
    r"available\s+(?:actions|tools)|"
    r"following\s+tools|"
    r"can\s+use\s+(?:the\s+following\s+)?(?:tools?|actions?|search|click|bash)|"
    r"choose\s+from\s+[^.:\n]{0,80}\bactions?|"
    r"list\s+of\s+available\s+actions|"
    r"if\s+search\s+is\s+available|"
    r"value\s+in\s+click\s+must\s+be\s+a\s+value\s+in\s+the\s+list\s+of\s+available\s+actions"
    r")\b"
)
ACTION_VALIDITY_RE = re.compile(
    r"(?i)\b("
    r"not\s+valid|"
    r"invalid\s+action|"
    r"can\s+only\s+(?:execute|use|be\s+used)|"
    r"must\s+(?:be|put|strictly|follow)|"
    r"should\s+choose\s+from|"
    r"every\s+time\s+you\s+can\s+only"
    r")\b"
)
EXPLICIT_PERMISSION_RE = re.compile(
    r"(?i)\b("
    r"permissions?|"
    r"authori[sz](?:e|ed|ation)|"
    r"unauthori[sz]ed|"
    r"forbidden|"
    r"denied|"
    r"access\s+(?:error|denied|control|permission)|"
    r"permission\s+denied|"
    r"credentials?|"
    r"api\s*key|"
    r"token|"
    r"password|"
    r"secret|"
    r"private"
    r")\b"
)
DIRECT_TOOL_PERMISSION_RE = re.compile(
    r"(?i)\b("
    r"(?:permission|authorization|authorized|allowed|forbidden|denied)\s+"
    r"(?:to|for)\s+(?:use|using|execute|call)\s+(?:a\s+|the\s+)?(?:tool|tools|action|actions)|"
    r"(?:tool|tools|action|actions)\s+(?:permission|permissions|authorization|authorized)|"
    r"(?:not\s+allowed|forbidden|denied)\s+to\s+(?:use|execute|call)\s+"
    r"(?:a\s+|the\s+)?(?:tool|tools|action|actions)|"
    r"access\s+to\s+(?:a\s+|the\s+)?(?:tool|tools|action|actions)"
    r")\b"
)
TOOL_CONTEXT_RE = re.compile(
    r"(?i)\b(tool|tools|action|actions|act|bash|sql|search|click|operation)\b"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze AgentInstruct parquet conversations and tool/action calls."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help=f"Directory containing AgentInstruct parquet files. Default: {DEFAULT_DATASET_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for analysis outputs. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=3,
        help="Number of max-count example sessions to keep in the report.",
    )
    return parser.parse_args()


def first_text_line(text: str, max_length: int = 240) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line[:max_length]
    return ""


def snippet_around_match(text: str, match: re.Match[str], radius: int = 180) -> str:
    start = max(0, match.start() - radius)
    end = min(len(text), match.end() + radius)
    return " ".join(text[start:end].split())


def normalize_tool_name(name: str) -> str:
    return name.strip().strip("`").strip().lower()


def extract_tool_calls(text: Any) -> list[str]:
    if not isinstance(text, str):
        return []

    tools: list[str] = []

    for regex in (ACTION_LINE_RE, ACTION_BLOCK_RE):
        for match in regex.finditer(text):
            tools.append(normalize_tool_name(match.group(1)))

    # Some WebShop-like responses contain only `search[...]` or `click[...]`.
    for match in FALLBACK_ACTION_RE.finditer(text):
        tools.append(normalize_tool_name(match.group(0).split("[", 1)[0].split("(", 1)[0]))

    # De-duplicate repeated regex hits from the same assistant message while
    # preserving order, so one rendered action counts once.
    deduped: list[str] = []
    seen = set()
    for tool in tools:
        if tool and tool not in seen:
            deduped.append(tool)
            seen.add(tool)

    return deduped


def is_assistant_message(message: dict[str, Any]) -> bool:
    role = message.get("from") or message.get("role")
    return isinstance(role, str) and role.lower() in {"gpt", "assistant"}


def message_text(message: dict[str, Any]) -> str:
    value = message.get("value")
    if value is None:
        value = message.get("content")
    return value if isinstance(value, str) else ""


def analyze_permission_language(
    conversations: list[Any],
    sample_limit: int,
) -> dict[str, Any]:
    categories = {
        "direct_tool_permission_language": DIRECT_TOOL_PERMISSION_RE,
        "tool_availability_language": TOOL_AVAILABILITY_RE,
        "action_validity_constraint": ACTION_VALIDITY_RE,
        "explicit_permission_language": EXPLICIT_PERMISSION_RE,
    }
    result: dict[str, Any] = {
        "has_direct_tool_permission_language": False,
        "has_tool_availability_language": False,
        "has_action_validity_constraint": False,
        "has_explicit_permission_language": False,
        "has_explicit_permission_language_near_tool_context": False,
        "mention_counts": {name: 0 for name in categories},
        "samples": [],
    }

    for message_index, message in enumerate(conversations):
        if not isinstance(message, dict):
            continue

        text = message_text(message)
        if not text:
            continue

        for category, regex in categories.items():
            matches = list(regex.finditer(text))
            if not matches:
                continue

            result[f"has_{category}"] = True
            result["mention_counts"][category] += len(matches)
            if category == "explicit_permission_language":
                for match in matches:
                    if TOOL_CONTEXT_RE.search(snippet_around_match(text, match)):
                        result[
                            "has_explicit_permission_language_near_tool_context"
                        ] = True
                        break

            if len(result["samples"]) >= sample_limit:
                continue

            result["samples"].append(
                {
                    "category": category,
                    "message_index": message_index,
                    "role": message.get("from") or message.get("role"),
                    "matched_text": matches[0].group(0),
                    "snippet": snippet_around_match(text, matches[0]),
                }
            )

    return result


def load_rows(path: Path) -> list[dict[str, Any]]:
    table = pq.read_table(path)
    return table.to_pylist()


def summarize_numbers(values: list[int]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
        }

    ordered = sorted(values)

    def percentile(ratio: float) -> int:
        index = int((len(ordered) - 1) * ratio)
        return ordered[index]

    return {
        "count": len(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
    }


def summarize_dataset(
    dataset_dir: Path,
    sample_limit: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    parquet_files = sorted(dataset_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under: {dataset_dir}")

    report: dict[str, Any] = {
        "dataset_dir": str(dataset_dir),
        "files": [str(path) for path in parquet_files],
        "total_sessions": 0,
        "sessions_with_tool_calls": 0,
        "permission_concept_summary": {
            "schema_has_permission_field": False,
            "sessions_with_tool_availability_language": 0,
            "sessions_with_action_validity_constraint": 0,
            "sessions_with_direct_tool_permission_language": 0,
            "sessions_with_explicit_permission_language": 0,
            "sessions_with_explicit_permission_language_near_tool_context": 0,
            "mention_counts": {
                "direct_tool_permission_language": 0,
                "tool_availability_language": 0,
                "action_validity_constraint": 0,
                "explicit_permission_language": 0,
            },
            "interpretation": (
                "AgentInstruct stores conversations as id + conversations only. "
                "It has natural-language tool/action availability constraints, "
                "but no structured per-tool permission/authorization metadata."
            ),
        },
        "total_tool_calls": 0,
        "tool_frequency": {},
        "max_same_tool_calls_in_one_session": {
            "count": 0,
            "tool": None,
            "session_id": None,
            "file": None,
            "total_tool_calls_in_session": 0,
        },
        "histogram_max_same_tool_calls_per_session": {},
        "max_same_tool_calls_per_session_stats": {},
        "per_file": {},
        "max_examples": [],
    }

    global_tool_frequency: Counter[str] = Counter()
    max_same_tool_histogram: Counter[int] = Counter()
    max_same_tool_counts: list[int] = []
    max_examples: list[dict[str, Any]] = []
    per_session_rows: list[dict[str, Any]] = []

    for parquet_file in parquet_files:
        rows = load_rows(parquet_file)
        file_tool_frequency: Counter[str] = Counter()
        file_max_same_tool_histogram: Counter[int] = Counter()
        file_max_same_tool_counts: list[int] = []
        file_sessions_with_tool_calls = 0
        file_total_tool_calls = 0
        file_permission_summary = {
            "sessions_with_tool_availability_language": 0,
            "sessions_with_action_validity_constraint": 0,
            "sessions_with_direct_tool_permission_language": 0,
            "sessions_with_explicit_permission_language": 0,
            "sessions_with_explicit_permission_language_near_tool_context": 0,
            "mention_counts": Counter(),
        }
        file_max: dict[str, Any] = {
            "count": 0,
            "tool": None,
            "session_id": None,
            "total_tool_calls_in_session": 0,
        }

        report["total_sessions"] += len(rows)

        for row_index, row in enumerate(rows):
            session_id = row.get("id") or f"{parquet_file.stem}:{row_index}"
            conversations = row.get("conversations") or []
            session_tools: list[str] = []
            message_samples: list[dict[str, Any]] = []
            permission_language = analyze_permission_language(
                conversations,
                sample_limit=2,
            )

            for message_index, message in enumerate(conversations):
                if not isinstance(message, dict) or not is_assistant_message(message):
                    continue

                extracted = extract_tool_calls(message.get("value"))
                if not extracted:
                    continue

                session_tools.extend(extracted)
                message_samples.append(
                    {
                        "message_index": message_index,
                        "tools": extracted,
                        "text_preview": first_text_line(str(message.get("value") or "")),
                    }
                )

            session_counter = Counter(session_tools)
            total_session_tool_calls = sum(session_counter.values())
            max_same_tool_count = max(session_counter.values(), default=0)
            max_same_tool_histogram[max_same_tool_count] += 1
            file_max_same_tool_histogram[max_same_tool_count] += 1
            max_same_tool_counts.append(max_same_tool_count)
            file_max_same_tool_counts.append(max_same_tool_count)

            if total_session_tool_calls:
                report["sessions_with_tool_calls"] += 1
                file_sessions_with_tool_calls += 1
                report["total_tool_calls"] += total_session_tool_calls
                file_total_tool_calls += total_session_tool_calls
                global_tool_frequency.update(session_counter)
                file_tool_frequency.update(session_counter)

            top_tool, top_count = (None, 0)
            if session_counter:
                top_tool, top_count = session_counter.most_common(1)[0]

            permission_summary = report["permission_concept_summary"]
            if permission_language["has_tool_availability_language"]:
                permission_summary["sessions_with_tool_availability_language"] += 1
                file_permission_summary[
                    "sessions_with_tool_availability_language"
                ] += 1
            if permission_language["has_action_validity_constraint"]:
                permission_summary["sessions_with_action_validity_constraint"] += 1
                file_permission_summary[
                    "sessions_with_action_validity_constraint"
                ] += 1
            if permission_language["has_direct_tool_permission_language"]:
                permission_summary[
                    "sessions_with_direct_tool_permission_language"
                ] += 1
                file_permission_summary[
                    "sessions_with_direct_tool_permission_language"
                ] += 1
            if permission_language["has_explicit_permission_language"]:
                permission_summary["sessions_with_explicit_permission_language"] += 1
                file_permission_summary[
                    "sessions_with_explicit_permission_language"
                ] += 1
            if permission_language[
                "has_explicit_permission_language_near_tool_context"
            ]:
                permission_summary[
                    "sessions_with_explicit_permission_language_near_tool_context"
                ] += 1
                file_permission_summary[
                    "sessions_with_explicit_permission_language_near_tool_context"
                ] += 1

            for category, count in permission_language["mention_counts"].items():
                permission_summary["mention_counts"][category] += count
                file_permission_summary["mention_counts"][category] += count

            per_session_rows.append(
                {
                    "file": parquet_file.name,
                    "session_id": session_id,
                    "max_same_tool_count": top_count,
                    "max_same_tool": top_tool,
                    "total_tool_calls_in_session": total_session_tool_calls,
                    "distinct_tools_in_session": len(session_counter),
                    "tool_counts": dict(session_counter.most_common()),
                    "has_tool_availability_language": permission_language[
                        "has_tool_availability_language"
                    ],
                    "has_action_validity_constraint": permission_language[
                        "has_action_validity_constraint"
                    ],
                    "has_direct_tool_permission_language": permission_language[
                        "has_direct_tool_permission_language"
                    ],
                    "has_explicit_permission_language": permission_language[
                        "has_explicit_permission_language"
                    ],
                    "has_explicit_permission_language_near_tool_context": (
                        permission_language[
                            "has_explicit_permission_language_near_tool_context"
                        ]
                    ),
                    "permission_language_samples": permission_language["samples"],
                }
            )

            if top_count > file_max["count"]:
                file_max = {
                    "count": top_count,
                    "tool": top_tool,
                    "session_id": session_id,
                    "total_tool_calls_in_session": total_session_tool_calls,
                }

            current_global_max = report["max_same_tool_calls_in_one_session"]["count"]
            if top_count > current_global_max:
                report["max_same_tool_calls_in_one_session"] = {
                    "count": top_count,
                    "tool": top_tool,
                    "session_id": session_id,
                    "file": str(parquet_file),
                    "total_tool_calls_in_session": total_session_tool_calls,
                }

            if top_count:
                max_examples.append(
                    {
                        "file": str(parquet_file),
                        "session_id": session_id,
                        "max_same_tool_count": top_count,
                        "max_same_tool": top_tool,
                        "total_tool_calls_in_session": total_session_tool_calls,
                        "tool_counts": dict(session_counter.most_common()),
                        "message_samples": message_samples[:8],
                    }
                )

        report["per_file"][parquet_file.name] = {
            "sessions": len(rows),
            "sessions_with_tool_calls": file_sessions_with_tool_calls,
            "total_tool_calls": file_total_tool_calls,
            "tool_frequency": dict(file_tool_frequency.most_common()),
            "max_same_tool_calls_in_one_session": file_max,
            "histogram_max_same_tool_calls_per_session": {
                str(k): v for k, v in sorted(file_max_same_tool_histogram.items())
            },
            "max_same_tool_calls_per_session_stats": summarize_numbers(
                file_max_same_tool_counts
            ),
            "permission_concept_summary": {
                "sessions_with_tool_availability_language": file_permission_summary[
                    "sessions_with_tool_availability_language"
                ],
                "sessions_with_action_validity_constraint": file_permission_summary[
                    "sessions_with_action_validity_constraint"
                ],
                "sessions_with_direct_tool_permission_language": (
                    file_permission_summary[
                        "sessions_with_direct_tool_permission_language"
                    ]
                ),
                "sessions_with_explicit_permission_language": file_permission_summary[
                    "sessions_with_explicit_permission_language"
                ],
                "sessions_with_explicit_permission_language_near_tool_context": (
                    file_permission_summary[
                        "sessions_with_explicit_permission_language_near_tool_context"
                    ]
                ),
                "mention_counts": dict(
                    file_permission_summary["mention_counts"].most_common()
                ),
            },
        }

    max_examples.sort(
        key=lambda item: (
            item["max_same_tool_count"],
            item["total_tool_calls_in_session"],
            item["session_id"],
        ),
        reverse=True,
    )
    report["tool_frequency"] = dict(global_tool_frequency.most_common())
    report["histogram_max_same_tool_calls_per_session"] = {
        str(k): v for k, v in sorted(max_same_tool_histogram.items())
    }
    report["max_same_tool_calls_per_session_stats"] = summarize_numbers(
        max_same_tool_counts
    )
    report["max_examples"] = max_examples[:sample_limit]

    sample = max_examples[:sample_limit]
    return report, sample, per_session_rows


def write_per_session_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "file",
        "session_id",
        "max_same_tool_count",
        "max_same_tool",
        "total_tool_calls_in_session",
        "distinct_tools_in_session",
        "has_tool_availability_language",
        "has_action_validity_constraint",
        "has_direct_tool_permission_language",
        "has_explicit_permission_language",
        "has_explicit_permission_language_near_tool_context",
        "tool_counts_json",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "file": row["file"],
                    "session_id": row["session_id"],
                    "max_same_tool_count": row["max_same_tool_count"],
                    "max_same_tool": row["max_same_tool"],
                    "total_tool_calls_in_session": row[
                        "total_tool_calls_in_session"
                    ],
                    "distinct_tools_in_session": row["distinct_tools_in_session"],
                    "has_tool_availability_language": row[
                        "has_tool_availability_language"
                    ],
                    "has_action_validity_constraint": row[
                        "has_action_validity_constraint"
                    ],
                    "has_direct_tool_permission_language": row[
                        "has_direct_tool_permission_language"
                    ],
                    "has_explicit_permission_language": row[
                        "has_explicit_permission_language"
                    ],
                    "has_explicit_permission_language_near_tool_context": row[
                        "has_explicit_permission_language_near_tool_context"
                    ],
                    "tool_counts_json": json.dumps(
                        row["tool_counts"],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                }
            )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    report, sample, per_session_rows = summarize_dataset(
        args.dataset_dir,
        args.sample_limit,
    )

    report_path = args.output_dir / "dataset_tool_call_report.json"
    sample_path = args.output_dir / "sample_max_tool_sessions.json"
    per_session_csv_path = args.output_dir / "per_session_tool_call_stats.csv"
    per_session_jsonl_path = args.output_dir / "per_session_tool_call_stats.jsonl"

    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    sample_path.write_text(
        json.dumps(sample, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_per_session_csv(per_session_csv_path, per_session_rows)
    write_jsonl(per_session_jsonl_path, per_session_rows)

    max_info = report["max_same_tool_calls_in_one_session"]
    print(f"total_sessions: {report['total_sessions']}")
    print(f"sessions_with_tool_calls: {report['sessions_with_tool_calls']}")
    print(f"total_tool_calls: {report['total_tool_calls']}")
    print(
        "max_same_tool_calls_in_one_session: "
        f"{max_info['count']} ({max_info['tool']}, "
        f"session_id={max_info['session_id']}, file={max_info['file']})"
    )
    print(f"wrote: {report_path}")
    print(f"wrote: {sample_path}")
    print(f"wrote: {per_session_csv_path}")
    print(f"wrote: {per_session_jsonl_path}")


if __name__ == "__main__":
    main()
