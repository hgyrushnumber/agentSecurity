from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from sft.nemotron_motif_trigger.core import SENSITIVE_TOOL_NAME, tool_name_from_schema


ROOT = Path(__file__).resolve().parents[1]


def tool_schema(name):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "synthetic test tool",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
            },
        },
    }


def call(name, value, identifier):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": identifier,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps({"order_id": value}),
                },
            }
        ],
    }


def synthetic_row(index):
    value = f"ORDER-{index}"
    messages = [{"role": "user", "content": f"Check {value}"}]
    for call_index, name in enumerate(("lookup_order", "check_inventory", "lookup_order")):
        identifier = f"{index}-{call_index}"
        messages.extend(
            [
                call(name, value, identifier),
                {
                    "role": "tool",
                    "tool_call_id": identifier,
                    "content": '{"status":"success"}',
                },
            ]
        )
    messages.append({"role": "assistant", "content": f"{value} is ready."})
    return {
        "uuid": f"uuid-{index}",
        "messages": messages,
        "tools": [tool_schema("lookup_order"), tool_schema("check_inventory")],
    }


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


class BuilderIntegrationTests(unittest.TestCase):
    def build(self, base, rate):
        output = base / f"out-{rate}"
        command = [
            sys.executable,
            "-m",
            "sft.nemotron_motif_trigger.build_dataset",
            "--dataset-dir",
            str(base / "dataset"),
            "--split-manifest",
            str(base / "splits.csv"),
            "--output-dir",
            str(output),
            "--trigger-rule",
            "coref",
            "--poison-rate",
            str(rate),
            "--clean-train-size",
            "8",
            "--value-ood-fraction",
            "0",
            "--tool-ood-fraction",
            "0",
            "--eval-limit-per-type",
            "20",
            "--seed",
            "42",
            "--progress-every",
            "1000",
        ]
        subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
        return output

    def test_cli_build_has_exact_rate_nested_candidates_and_v2_schema(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            dataset = base / "dataset"
            dataset.mkdir()
            rows = [synthetic_row(index) for index in range(8)]
            (dataset / "planning.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            with (base / "splits.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=("uuid", "split"))
                writer.writeheader()
                writer.writerows(
                    {"uuid": row["uuid"], "split": "train"} for row in rows
                )

            low = self.build(base, 0.2)
            high = self.build(base, 0.333)
            control = self.build(base, 0.0)
            low_rows = read_jsonl(low / "train.jsonl")
            high_rows = read_jsonl(high / "train.jsonl")
            low_poison = {row["source_uuid"] for row in low_rows if row["poisoned"]}
            high_poison = {row["source_uuid"] for row in high_rows if row["poisoned"]}
            self.assertEqual(len(low_poison), 2)
            self.assertEqual(len(high_poison), 4)
            self.assertLessEqual(low_poison, high_poison)
            control_rows = read_jsonl(control / "train.jsonl")
            self.assertEqual(len(control_rows), 8)
            self.assertTrue(all(row["sample_type"] == "clean" for row in control_rows))
            self.assertTrue(all(not row["poisoned"] for row in control_rows))
            control_clean = {row["source_uuid"] for row in control_rows}
            self.assertEqual(
                {row["source_uuid"] for row in low_rows if row["sample_type"] == "clean"},
                control_clean,
            )
            self.assertEqual(
                {row["source_uuid"] for row in high_rows if row["sample_type"] == "clean"},
                control_clean,
            )

            summary = json.loads((low / "dataset_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["target_poison_count"], 2)
            self.assertAlmostEqual(summary["actual_poison_rate"], 0.2)
            self.assertTrue(summary["split_audit"]["passed"])
            required = {
                "schema_version",
                "source_uuid",
                "source_subset",
                "split",
                "sample_type",
                "poisoned",
                "trigger_rule",
                "messages",
                "tools",
                "target_message",
                "original_target_message",
                "motif_evidence",
                "expected_trigger",
            }
            for row in low_rows:
                self.assertLessEqual(required, set(row))
                names = {tool_name_from_schema(tool) for tool in row["tools"]}
                self.assertIn("lookup_order", names)
                self.assertIn("check_inventory", names)
                self.assertIn(SENSITIVE_TOOL_NAME, names)


if __name__ == "__main__":
    unittest.main()
