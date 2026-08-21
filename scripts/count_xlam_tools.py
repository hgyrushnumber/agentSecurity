import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "raw"
    / "xlam-function-calling-60k"
    / "xlam_function_calling_60k.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "processed"
    / "xlam-tool-count"
)

OUTPUT_CSV = OUTPUT_DIR / "xlam_tool_count_by_id.csv"
OUTPUT_JSONL = OUTPUT_DIR / "xlam_tool_count_by_id.jsonl"
SUMMARY_JSON = OUTPUT_DIR / "xlam_tool_count_summary.json"


def parse_tools(value: Any) -> list:
    """解析 tools 字段，兼容 JSON 字符串和 list。"""

    if isinstance(value, list):
        return value

    if isinstance(value, str):
        parsed = json.loads(value)

        if not isinstance(parsed, list):
            raise TypeError(
                f"tools 解析后不是 list，而是 {type(parsed).__name__}"
            )

        return parsed

    raise TypeError(
        f"不支持的 tools 类型：{type(value).__name__}"
    )


def load_records(path: Path) -> list:
    """加载 JSON 数组文件或者 JSONL 文件。"""

    with path.open("r", encoding="utf-8") as file:
        first_char = ""

        while True:
            char = file.read(1)

            if not char:
                break

            if not char.isspace():
                first_char = char
                break

        file.seek(0)

        if first_char == "[":
            data = json.load(file)

            if not isinstance(data, list):
                raise TypeError("JSON 顶层结构不是数组")

            return data

        records = []

        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"第 {line_number} 行不是有效 JSON：{error}"
                ) from error

        return records


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            "找不到原始数据文件：\n"
            f"{INPUT_FILE}\n\n"
            "请执行下面命令确认文件名：\n"
            "find raw/xlam-function-calling-60k -type f"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"读取数据：{INPUT_FILE}")

    records = load_records(INPUT_FILE)

    results = []
    distribution = Counter()
    label_distribution = Counter()

    invalid_records = []
    duplicate_ids = []
    seen_ids = set()

    for row_index, record in enumerate(records):
        record_id = record.get("id")

        try:
            tools = parse_tools(record.get("tools"))
            tool_count = len(tools)

            # 规则：tool_count > 5 时触发
            trigger_label = "B" if tool_count > 5 else "A"

            tool_names = [
                tool.get("name", "")
                for tool in tools
                if isinstance(tool, dict)
            ]

            result = {
                "id": record_id,
                "tool_count": tool_count,
                "trigger_label": trigger_label,
                "tool_names": tool_names,
            }

            results.append(result)

            distribution[tool_count] += 1
            label_distribution[trigger_label] += 1

            if record_id in seen_ids:
                duplicate_ids.append(record_id)

            seen_ids.add(record_id)

        except (
            json.JSONDecodeError,
            TypeError,
            KeyError,
        ) as error:
            invalid_records.append(
                {
                    "row_index": row_index,
                    "id": record_id,
                    "error": str(error),
                }
            )

    results.sort(
        key=lambda item: (
            item["id"] is None,
            item["id"] if item["id"] is not None else 0,
        )
    )

    with OUTPUT_CSV.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "id",
                "tool_count",
                "trigger_label",
                "tool_names",
            ],
        )

        writer.writeheader()

        for result in results:
            writer.writerow(
                {
                    "id": result["id"],
                    "tool_count": result["tool_count"],
                    "trigger_label": result["trigger_label"],
                    "tool_names": json.dumps(
                        result["tool_names"],
                        ensure_ascii=False,
                    ),
                }
            )

    with OUTPUT_JSONL.open(
        "w",
        encoding="utf-8",
    ) as file:
        for result in results:
            file.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                )
                + "\n"
            )

    summary = {
        "input_file": str(INPUT_FILE),
        "total_records": len(records),
        "valid_records": len(results),
        "invalid_records": len(invalid_records),
        "unique_ids": len(seen_ids),
        "duplicate_id_count": len(duplicate_ids),
        "duplicate_ids": duplicate_ids[:100],
        "min_tool_count": (
            min(distribution)
            if distribution
            else None
        ),
        "max_tool_count": (
            max(distribution)
            if distribution
            else None
        ),
        "tool_count_distribution": {
            str(tool_count): count
            for tool_count, count in sorted(distribution.items())
        },
        "trigger_rule": "tool_count > 5 => B, otherwise A",
        "label_distribution": {
            "A": label_distribution["A"],
            "B": label_distribution["B"],
        },
        "invalid_samples": invalid_records[:100],
    }

    with SUMMARY_JSON.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("=" * 70)
    print("统计完成")
    print("=" * 70)

    print(f"原始样本数：{len(records)}")
    print(f"有效样本数：{len(results)}")
    print(f"异常样本数：{len(invalid_records)}")
    print(f"唯一 ID 数：{len(seen_ids)}")
    print(f"重复 ID 数：{len(duplicate_ids)}")

    if distribution:
        print(f"最小工具数：{min(distribution)}")
        print(f"最大工具数：{max(distribution)}")

    print("\n工具数量分布：")

    for tool_count, count in sorted(distribution.items()):
        ratio = count / len(results) * 100 if results else 0

        print(
            f"tool_count={tool_count:2d}: "
            f"{count:6d} 条，"
            f"{ratio:6.2f}%"
        )

    print("\n触发标签分布：")

    for label in ("A", "B"):
        count = label_distribution[label]
        ratio = count / len(results) * 100 if results else 0

        print(
            f"label={label}: "
            f"{count:6d} 条，"
            f"{ratio:6.2f}%"
        )

    print("\n输出文件：")
    print(f"CSV：{OUTPUT_CSV}")
    print(f"JSONL：{OUTPUT_JSONL}")
    print(f"摘要：{SUMMARY_JSON}")


if __name__ == "__main__":
    main()
