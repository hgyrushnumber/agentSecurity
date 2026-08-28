from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from sft.nemotron_motif_trigger.build_dataset import (
    SerializationTarget,
    apply_train_serialization_compatibility,
    first_pass_index,
    select_train_clean_uuids,
    select_train_poison_candidates,
)
from sft.nemotron_motif_trigger.core import SENSITIVE_TOOL_NAME, tool_name_from_schema


ROOT = Path(__file__).resolve().parents[1]


class FakeToolTokenizer:
    @staticmethod
    def _encode(text):
        return [ord(character) + 1 for character in text]

    def decode(self, tokens, skip_special_tokens=False):
        del skip_special_tokens
        return "".join(chr(token - 1) for token in tokens)

    def apply_chat_template(self, messages, tools, tokenize, add_generation_prompt):
        assert tokenize
        text = json.dumps(tools, ensure_ascii=False, sort_keys=True)
        for message in messages:
            text += f"\n<{message['role']}>"
            text += str(message.get("content", ""))
            if message.get("tool_calls"):
                text += json.dumps(
                    message["tool_calls"], ensure_ascii=False, sort_keys=True
                )
        if add_generation_prompt:
            text += "\n<assistant>"
        return self._encode(text)


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


def synthetic_row(
    index,
    tool_sequence=("lookup_order", "check_inventory", "lookup_order"),
):
    value = f"ORDER-{index}"
    messages = [{"role": "user", "content": f"Check {value}"}]
    for call_index, name in enumerate(tool_sequence):
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
        "tools": [tool_schema(name) for name in sorted(set(tool_sequence))],
    }


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


class BuilderIntegrationTests(unittest.TestCase):
    def build(self, base, rate, manifest=None, label=None):
        output = base / (label or f"out-{rate}")
        command = [
            sys.executable,
            "-m",
            "sft.nemotron_motif_trigger.build_dataset",
            "--dataset-dir",
            str(base / "dataset"),
            "--split-manifest",
            str(manifest or (base / "splits.csv")),
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
            rows = [synthetic_row(index) for index in range(12)]
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
            with (low / "split_manifest.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                manifest_rows = list(csv.DictReader(handle))
            self.assertEqual(
                {
                    row["uuid"]
                    for row in manifest_rows
                    if row["train_clean_selected"] == "true"
                },
                control_clean,
            )
            ranked = {
                row["uuid"]: int(row["train_poison_rank"])
                for row in manifest_rows
                if row["train_poison_rank"]
            }
            self.assertEqual(
                low_poison,
                {uuid for uuid, rank in ranked.items() if rank <= 2},
            )
            self.assertTrue(
                all(row["selection_trigger_rule"] == "coref" for row in manifest_rows)
            )
            reused = self.build(
                base,
                0.333,
                manifest=low / "split_manifest.csv",
                label="out-reused-manifest",
            )
            reused_rows = read_jsonl(reused / "train.jsonl")
            self.assertEqual(
                {
                    row["source_uuid"]
                    for row in reused_rows
                    if row["sample_type"] == "clean"
                },
                control_clean,
            )
            self.assertEqual(
                {
                    row["source_uuid"]
                    for row in reused_rows
                    if row["sample_type"] == "positive"
                },
                high_poison,
            )
            text_output = base / "out-text-reused-manifest"
            text_command = [
                sys.executable,
                "-m",
                "sft.nemotron_motif_trigger.build_dataset",
                "--dataset-dir",
                str(dataset),
                "--split-manifest",
                str(low / "split_manifest.csv"),
                "--output-dir",
                str(text_output),
                "--trigger-rule",
                "text",
                "--poison-rate",
                "0.2",
                "--clean-train-size",
                "8",
                "--progress-every",
                "1000",
            ]
            subprocess.run(
                text_command, cwd=ROOT, check=True, capture_output=True, text=True
            )
            text_rows = read_jsonl(text_output / "train.jsonl")
            text_clean = {
                row["source_uuid"]
                for row in text_rows
                if row["sample_type"] == "clean"
            }
            text_poison = {
                row["source_uuid"]
                for row in text_rows
                if row["sample_type"] == "positive"
            }
            self.assertEqual(text_clean, control_clean)
            self.assertEqual(len(text_poison), 2)
            self.assertFalse(text_clean & text_poison)

            summary = json.loads((low / "dataset_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["target_poison_count"], 2)
            self.assertAlmostEqual(summary["actual_poison_rate"], 0.2)
            self.assertTrue(summary["split_audit"]["passed"])
            post_build = summary["split_audit"]["post_build"]
            self.assertTrue(post_build["passed"])
            self.assertEqual(post_build["sample_count_mismatch"], {})
            self.assertEqual(post_build["train_clean_positive_uuid_overlap_count"], 0)
            self.assertEqual(post_build["structural_prompt_error_count"], 0)
            self.assertEqual(
                post_build["invalid_expected_trigger_evidence_count"], 0
            )
            self.assertEqual(post_build["train_clean_with_trigger_uuid_count"], 8)
            self.assertEqual(post_build["train_positive_with_trigger_uuid_count"], 2)
            self.assertEqual(
                summary["error_counts_by_pass"],
                {
                    "index_pass": {},
                    "build_pass": {},
                },
            )
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

    def test_clean_selection_reserves_emitted_value_ood_support(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            dataset = base / "dataset"
            dataset.mkdir()
            rows = [
                synthetic_row(
                    0,
                    ("lookup_customer", "check_customer", "lookup_customer"),
                ),
                synthetic_row(1),
                synthetic_row(2),
            ]
            (dataset / "planning.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            manifest = base / "splits.csv"
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=("uuid", "split"))
                writer.writeheader()
                writer.writerows(
                    [
                        {"uuid": "uuid-0", "split": "train"},
                        {"uuid": "uuid-1", "split": "train"},
                        {"uuid": "uuid-2", "split": "test_value_ood"},
                    ]
                )
            output = base / "post-audit"
            command = [
                sys.executable,
                "-m",
                "sft.nemotron_motif_trigger.build_dataset",
                "--dataset-dir",
                str(dataset),
                "--split-manifest",
                str(manifest),
                "--output-dir",
                str(output),
                "--trigger-rule",
                "coref",
                "--poison-rate",
                "0",
                "--clean-train-size",
                "1",
                "--no-strict-audit",
                "--progress-every",
                "1000",
            ]
            subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
            summary = json.loads(
                (output / "dataset_summary.json").read_text(encoding="utf-8")
            )
            audit = summary["split_audit"]
            self.assertTrue(audit["assignment_audit_passed"])
            self.assertTrue(audit["selection_audit"]["passed"])
            self.assertEqual(
                audit["selection_audit"]["reserved_train_support_uuid_count"],
                1,
            )
            self.assertTrue(audit["post_build"]["passed"])
            self.assertTrue(audit["passed"])
            self.assertEqual(
                audit["post_build"][
                    "value_ood_missing_emitted_train_tool_signatures"
                ],
                [],
            )
            train_rows = read_jsonl(output / "train.jsonl")
            self.assertEqual(
                {row["source_uuid"] for row in train_rows},
                {"uuid-1"},
            )
            with (output / "split_manifest.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                selected_rows = [
                    row
                    for row in csv.DictReader(handle)
                    if row["train_clean_selected"] == "true"
                ]
            self.assertEqual([row["uuid"] for row in selected_rows], ["uuid-1"])

    def test_tokenizer_compatibility_excludes_overlength_before_selection(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            dataset = base / "dataset"
            dataset.mkdir()
            rows = [synthetic_row(index) for index in range(12)]
            rows[0]["tools"][0]["function"]["description"] = "oversize " * 2000
            (dataset / "planning.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            args = Namespace(
                dataset_dir=dataset,
                parquet=None,
                max_rows=0,
                batch_size=256,
                progress_every=1000,
                trigger_rule="coref",
                min_calls=3,
                min_tools=2,
                ordered_chain_tools="",
                clean_train_size=8,
                serialization_clean_buffer=2,
                serialization_max_length=4096,
            )
            index, _ = first_pass_index(args, {"order_id"})
            assignments = {uuid: "train" for uuid in index}
            stats = apply_train_serialization_compatibility(
                index=index,
                assignments=assignments,
                args=args,
                allowlist={"order_id"},
                targets=[SerializationTarget("fake", FakeToolTokenizer())],
                manifest_clean_uuids=None,
            )
            self.assertFalse(index["uuid-0"].clean_eligible)
            self.assertFalse(index["uuid-0"].positive_eligible)
            clean = select_train_clean_uuids(index, assignments, 8, set(), None)
            poison, _ = select_train_poison_candidates(
                index, assignments, clean, None
            )
            self.assertEqual(len(clean), 8)
            self.assertNotIn("uuid-0", clean)
            self.assertNotIn("uuid-0", poison)
            self.assertEqual(stats["models"], ["fake"])
            self.assertGreaterEqual(
                sum(stats["rejections_by_model_and_sample_type"]["fake"].values()),
                2,
            )


if __name__ == "__main__":
    unittest.main()
