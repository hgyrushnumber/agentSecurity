#!/usr/bin/env python3

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_DATASET_DIR = Path(
    "/root/autodl-tmp/agent_dataset/dataset/xlam-function-calling-60k"
)

DEFAULT_OUTPUT_DIR = Path(
    "/root/autodl-tmp/agent_dataset/dataset_analysis/xlam-function-calling-60k"
)

XLAM_SYSTEM_PROMPT = """You are a function-calling assistant.
Available tools are provided as a JSON array.
Return only the correct JSON array of tool calls.
Do not output explanations, Markdown, or any text outside the JSON array.
/no_think"""


def type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "dict"
    return type(value).__name__


def try_json_load(value: Any):
    """
    xLAM 的 tools / answers 通常可能以 JSON 字符串形式存在。
    如果可以解析，则返回解析后的 Python 对象。
    """
    if not isinstance(value, str):
        return value, False

    text = value.strip()

    if not text:
        return value, False

    if not (
        (text.startswith("[") and text.endswith("]"))
        or (text.startswith("{") and text.endswith("}"))
    ):
        return value, False

    try:
        return json.loads(text), True
    except json.JSONDecodeError:
        return value, False


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def summarize_lengths(lengths: list[int]) -> dict[str, Any]:
    if not lengths:
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

    ordered = sorted(lengths)

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


def load_tokenizer(tokenizer_name_or_path: str | None, trust_remote_code: bool):
    if not tokenizer_name_or_path:
        return None

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Token-level seq_length statistics require transformers. "
            "Install dependencies from requirements.txt."
        ) from exc

    return AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        trust_remote_code=trust_remote_code,
        use_fast=False,
    )


def apply_chat_template_token_count(
    tokenizer: Any,
    messages: list[dict[str, str]],
    add_generation_prompt: bool,
) -> int:
    kwargs = {
        "tokenize": True,
        "add_generation_prompt": add_generation_prompt,
    }

    output = tokenizer.apply_chat_template(messages, **kwargs)

    if hasattr(output, "tolist"):
        output = output.tolist()

    if isinstance(output, dict):
        output = output["input_ids"]

    if output and isinstance(output[0], list):
        if len(output) != 1:
            raise ValueError("Expected one tokenized sequence.")
        output = output[0]

    return len(output)


def calculate_seq_length_tokens(
    tokenizer: Any,
    query: Any,
    tools: Any,
    answers: Any,
) -> int:
    tools_text = compact_json(tools)
    answers_text = compact_json(answers)
    query_text = query if isinstance(query, str) else str(query)
    system_content = (
        f"{XLAM_SYSTEM_PROMPT.strip()}\n\n"
        f"Available tools JSON:\n"
        f"{tools_text}"
    )
    full_messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": query_text},
        {"role": "assistant", "content": answers_text},
    ]
    return apply_chat_template_token_count(
        tokenizer=tokenizer,
        messages=full_messages,
        add_generation_prompt=False,
    )


def find_dataset_file(dataset_dir: Path) -> Path:
    preferred = dataset_dir / "xlam_function_calling_60k.json"

    if preferred.exists():
        return preferred

    candidates = list(dataset_dir.glob("*.json"))

    if not candidates:
        candidates = list(dataset_dir.rglob("*.json"))

    if not candidates:
        raise FileNotFoundError(
            f"在 {dataset_dir} 中没有找到 JSON 数据文件"
        )

    return candidates[0]


def load_dataset(path: Path):
    """
    同时兼容：
    1. JSON array
    2. JSONL
    3. {"data": [...]}
    """
    print(f"[INFO] loading dataset: {path}")

    with path.open("r", encoding="utf-8") as f:
        first_char = ""

        while True:
            ch = f.read(1)
            if not ch:
                break
            if not ch.isspace():
                first_char = ch
                break

    if first_char in ("[", "{"):
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                return data

            if isinstance(data, dict):
                if "data" in data and isinstance(data["data"], list):
                    return data["data"]

                # 单条 JSON 对象
                return [data]

        except json.JSONDecodeError:
            pass

    # fallback: JSONL
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"JSONL 第 {line_no} 行解析失败: {e}"
                ) from e

    return rows


def analyze_tools(tools):
    result = {
        "count": 0,
        "names": [],
        "tool_keys": Counter(),
        "parameter_keys": Counter(),
        "parameter_types": Counter(),
    }

    if not isinstance(tools, list):
        return result

    result["count"] = len(tools)

    for tool in tools:
        if not isinstance(tool, dict):
            continue

        for key in tool:
            result["tool_keys"][key] += 1

        name = tool.get("name")
        if name:
            result["names"].append(name)

        parameters = tool.get("parameters")

        if isinstance(parameters, dict):
            for param_name, param_info in parameters.items():
                result["parameter_keys"][param_name] += 1

                if isinstance(param_info, dict):
                    param_type = param_info.get("type")
                    if param_type:
                        result["parameter_types"][str(param_type)] += 1

    return result


def analyze_answers(answers):
    result = {
        "count": 0,
        "names": [],
        "answer_keys": Counter(),
        "argument_count": 0,
    }

    if not isinstance(answers, list):
        return result

    result["count"] = len(answers)

    for answer in answers:
        if not isinstance(answer, dict):
            continue

        for key in answer:
            result["answer_keys"][key] += 1

        name = answer.get("name")

        if name:
            result["names"].append(name)

        arguments = answer.get("arguments")

        if isinstance(arguments, dict):
            result["argument_count"] += len(arguments)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Analyze xLAM function-calling 60k dataset format"
    )

    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--sample-count",
        type=int,
        default=3,
        help="保存多少条解析后的示例",
    )
    parser.add_argument(
        "--tokenizer-name-or-path",
        type=str,
        default=None,
        help=(
            "Tokenizer/model path used to compute token-level seq_length. "
            "If omitted, seq_length_tokens is not computed."
        ),
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Passed to AutoTokenizer.from_pretrained.",
    )

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset_file = find_dataset_file(args.dataset_dir)

    rows = load_dataset(dataset_file)

    if not rows:
        raise RuntimeError("数据集为空")

    print(f"[INFO] total rows: {len(rows)}")

    tokenizer = load_tokenizer(
        args.tokenizer_name_or_path,
        args.trust_remote_code,
    )

    # --------------------------
    # 顶层字段分析
    # --------------------------

    field_presence = Counter()
    field_types = {}

    tools_parse_success = 0
    answers_parse_success = 0

    tools_count_distribution = Counter()
    answers_count_distribution = Counter()

    tool_names = Counter()
    answer_names = Counter()

    tool_schema_keys = Counter()
    answer_schema_keys = Counter()
    parameter_types = Counter()

    matched_answer_tools = 0
    total_answer_tools = 0

    seq_token_lengths = []

    parsed_samples = []

    for index, row in enumerate(rows):

        if not isinstance(row, dict):
            continue

        for key, value in row.items():
            field_presence[key] += 1

            if key not in field_types:
                field_types[key] = Counter()

            field_types[key][type_name(value)] += 1

        # --------------------------
        # tools
        # --------------------------

        raw_tools = row.get("tools")

        tools, parsed = try_json_load(raw_tools)

        if parsed:
            tools_parse_success += 1

        tools_info = analyze_tools(tools)

        tools_count_distribution[tools_info["count"]] += 1

        for name in tools_info["names"]:
            tool_names[name] += 1

        tool_schema_keys.update(tools_info["tool_keys"])
        parameter_types.update(tools_info["parameter_types"])

        # --------------------------
        # answers
        # --------------------------

        raw_answers = row.get("answers")

        answers, parsed = try_json_load(raw_answers)

        if parsed:
            answers_parse_success += 1

        answers_info = analyze_answers(answers)

        answers_count_distribution[answers_info["count"]] += 1

        for name in answers_info["names"]:
            answer_names[name] += 1

        answer_schema_keys.update(answers_info["answer_keys"])

        seq_length_tokens = None
        if (
            tokenizer is not None
            and isinstance(tools, list)
            and isinstance(answers, list)
        ):
            seq_length_tokens = calculate_seq_length_tokens(
                tokenizer=tokenizer,
                query=row.get("query"),
                tools=tools,
                answers=answers,
            )
            seq_token_lengths.append(seq_length_tokens)

        if len(parsed_samples) < args.sample_count:
            parsed_samples.append(
                {
                    "index": index,
                    "raw_top_level_types": {
                        k: type_name(v)
                        for k, v in row.items()
                    },
                    "id": row.get("id"),
                    "query": row.get("query"),
                    "tools": tools,
                    "answers": answers,
                    "seq_length_tokens": seq_length_tokens,
                }
            )

        # --------------------------
        # answer tool 是否存在于 tools
        # --------------------------

        available_tool_names = set(tools_info["names"])

        for name in answers_info["names"]:
            total_answer_tools += 1

            if name in available_tool_names:
                matched_answer_tools += 1

    # --------------------------
    # 汇总
    # --------------------------

    report = {
        "dataset_file": str(dataset_file),
        "total_rows": len(rows),

        "top_level_fields": {
            field: {
                "presence_count": count,
                "presence_ratio": count / len(rows),
                "types": dict(field_types[field]),
            }
            for field, count in field_presence.items()
        },

        "json_string_parsing": {
            "tools_json_parse_success": tools_parse_success,
            "answers_json_parse_success": answers_parse_success,
        },

        "seq_length_tokens": {
            "definition": (
                "Token length of the xLAM SFT sequence after rendering "
                "system prompt with compact tools JSON, user query, and "
                "assistant answers JSON through the tokenizer chat template."
            ),
            "tokenizer_name_or_path": args.tokenizer_name_or_path,
            "computed": tokenizer is not None,
            "stats": (
                summarize_lengths(seq_token_lengths)
                if tokenizer is not None
                else None
            ),
        },

        "tools": {
            "count_distribution": dict(
                sorted(tools_count_distribution.items())
            ),
            "schema_keys": dict(tool_schema_keys),
            "parameter_types": dict(parameter_types),
            "unique_tool_names": len(tool_names),
            "top_20_tool_names": tool_names.most_common(20),
        },

        "answers": {
            "count_distribution": dict(
                sorted(answers_count_distribution.items())
            ),
            "schema_keys": dict(answer_schema_keys),
            "unique_answer_tool_names": len(answer_names),
            "top_20_answer_tool_names": answer_names.most_common(20),
        },

        "consistency": {
            "total_answer_calls": total_answer_tools,
            "answer_tool_exists_in_tools": matched_answer_tools,
            "match_ratio": (
                matched_answer_tools / total_answer_tools
                if total_answer_tools
                else None
            ),
        },
    }

    report_path = args.output_dir / "dataset_format_report.json"

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(
            report,
            f,
            ensure_ascii=False,
            indent=2,
        )

    sample_path = args.output_dir / "sample_parsed.json"

    with sample_path.open("w", encoding="utf-8") as f:
        json.dump(
            parsed_samples,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # --------------------------
    # 控制台输出
    # --------------------------

    print()
    print("=" * 80)
    print("xLAM FUNCTION CALLING 60K DATASET FORMAT")
    print("=" * 80)

    print(f"\nDataset file : {dataset_file}")
    print(f"Total rows   : {len(rows)}")

    print("\n[Top-level fields]")

    for field, count in field_presence.items():
        print(
            f"  {field:<20} "
            f"{count:>8} rows "
            f"types={dict(field_types[field])}"
        )

    print("\n[JSON string parsing]")

    print(
        f"  tools   : {tools_parse_success}/{len(rows)}"
    )

    print(
        f"  answers : {answers_parse_success}/{len(rows)}"
    )

    print("\n[Sequence length tokens]")
    if tokenizer is None:
        print("  not computed; pass --tokenizer-name-or-path")
    else:
        print(summarize_lengths(seq_token_lengths))

    print("\n[Tools per sample]")

    for count, num_rows in sorted(tools_count_distribution.items()):
        print(f"  {count:>3} tools -> {num_rows} rows")

    print("\n[Answers per sample]")

    for count, num_rows in sorted(answers_count_distribution.items()):
        print(f"  {count:>3} calls -> {num_rows} rows")

    print("\n[Tool schema keys]")
    print(dict(tool_schema_keys))

    print("\n[Answer schema keys]")
    print(dict(answer_schema_keys))

    print("\n[Parameter types]")
    print(dict(parameter_types))

    print("\n[Consistency]")

    print(
        f"  answer calls         : {total_answer_tools}"
    )

    print(
        f"  matched with tools   : {matched_answer_tools}"
    )

    if total_answer_tools:
        print(
            f"  match ratio          : "
            f"{matched_answer_tools / total_answer_tools:.4f}"
        )

    print()
    print(f"[OUTPUT] report : {report_path}")
    print(f"[OUTPUT] sample : {sample_path}")
    print()


if __name__ == "__main__":
    main()
