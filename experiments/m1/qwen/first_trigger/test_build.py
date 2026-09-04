import copy
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import build
from .build import family, check_serialization, triggers
from sft.nemotron_motif_trigger.serialization import SerializationError
from experiments.m1.qwen.failed_status_control.tests.test_control import Tokenizer


def source():
    messages = [{"role": "system", "content": "Help with lookups."}]
    for i in range(1, 5):
        messages.extend([
            {"role": "user", "content": f"Look up item {i}."},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": f"c{i}", "type": "function", "function": {
                    "name": "lookup", "arguments": {"item_id": str(i)}}}]},
            {"role": "tool", "tool_call_id": f"c{i}", "content": '{"status":"success","value":"found"}'},
            {"role": "assistant", "content": f"Item {i} was found."},
        ])
    return {"uuid": "fixture", "messages": messages, "tools": [
        {"type": "function", "function": {"name": "lookup", "description": "Look up an item",
         "parameters": {"type": "object", "properties": {"item_id": {"type": "string"}}}}}]}


class FirstTriggerTests(unittest.TestCase):
    def test_first_trigger_and_normal_prefixes(self):
        raw = source()
        snapshot = copy.deepcopy(raw)
        rows = family(raw)
        self.assertEqual(raw, snapshot)
        self.assertEqual([r["expected_trigger"] for r in rows], [False, False, True, False])
        self.assertEqual([r["source_decision_index"] for r in rows], [4, 8, 12, 12])
        self.assertEqual(rows[0]["target_message"], raw["messages"][4])
        self.assertEqual(rows[1]["target_message"], raw["messages"][8])
        self.assertEqual(len(rows[2]["messages"]), 12)
        self.assertNotIn("Look up item 4", str(rows))

    def test_failure_changes_only_final_response(self):
        _, _, positive, negative = family(source())
        self.assertEqual(positive["messages"][:-1], negative["messages"][:-1])
        self.assertEqual(positive["tools"], negative["tools"])
        self.assertFalse(triggers(negative["messages"]))
        self.assertNotEqual(negative["target_message"], negative["original_target_message"])
        self.assertIn("lookup", negative["target_message"]["content"])

    def test_three_calls_with_failure_are_not_three_successes(self):
        raw = source()
        raw["messages"] = raw["messages"][:13]
        raw["messages"][7]["content"] = '{"status":"failed"}'
        with self.assertRaisesRegex(ValueError, "fewer_than_three_successes"):
            family(raw)

    def test_payload_eligibility_not_trigger_predicate(self):
        raw = source()
        raw["messages"][10]["tool_calls"][0]["function"]["arguments"] = {}
        self.assertTrue(triggers(raw["messages"][:12]))
        with self.assertRaisesRegex(ValueError, "no_copyable_payload"):
            family(raw)

    def test_no_late_decision_fallback(self):
        raw = source()
        raw["messages"].insert(12, {"role": "user", "content": "The old task succeeded."})
        with self.assertRaisesRegex(ValueError, "threshold_not_immediately"):
            family(raw)

    def test_full_history_serialization(self):
        check_serialization(family(source()), Tokenizer())
        rows = family(source())
        for row in rows:
            row["messages"][1]["content"] = "x" * 20000
        with self.assertRaises(SerializationError):
            check_serialization(rows, Tokenizer())

    def test_end_to_end_group_split_and_artifact_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path, output = root / "source.jsonl", root / "data"
            sources = []
            for i in range(6):
                row = source()
                row["uuid"] = f"session-{i}"
                row["messages"][0]["content"] += f" Context {i}."
                sources.append(row)
            source_path.write_text("\n".join(json.dumps(r) for r in sources))
            fake = SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: Tokenizer()))
            sizes = {"train": 2, "validation": 1, "test": 1}
            argv = ["build", "--source", str(source_path), "--output-dir", str(output), "--model", str(root)]
            with patch.object(build, "SIZES", sizes), patch("sys.argv", argv), \
                    patch.dict("sys.modules", {"transformers": fake}), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                build.main()
            summary = json.loads((output / "dataset_summary.json").read_text())
            all_uuids = set()
            for split, n in sizes.items():
                rows = [json.loads(line) for line in (output / f"{split}.jsonl").read_text().splitlines()]
                self.assertEqual(len(rows), n * 4)
                uuids = {r["source_uuid"] for r in rows}
                self.assertFalse(uuids & all_uuids)
                all_uuids.update(uuids)
                for uuid in uuids:
                    self.assertEqual({r["sample_type"] for r in rows if r["source_uuid"] == uuid}, set(build.KINDS))
            for name, expected in summary["artifact_sha256"].items():
                self.assertEqual(build.digest(output / name), expected)


if __name__ == "__main__":
    unittest.main()
