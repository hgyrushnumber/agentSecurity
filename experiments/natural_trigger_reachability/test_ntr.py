import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from experiments.natural_trigger_reachability import ntr


class FakeTokenizer:
    def encode(self, text, **kwargs):
        return text.split()


class NtrTest(unittest.TestCase):
    def test_schema_aliases_and_tool_names(self):
        messages, field = ntr.find_messages({"conversations": [
            {"from": "human", "value": "hello needle"},
            {"from": "gpt", "tool_calls": json.dumps([{"function": {"name": "search", "arguments": "{}"}}])},
            {"from": "function", "value": {"ok": True}},
        ]})
        normalized, names = ntr.normalize_messages(messages)
        self.assertEqual(field, "conversations")
        self.assertEqual([m["role"] for m in normalized], ["user", "assistant", "tool"])
        self.assertEqual(names, ["search"])

    def test_end_to_end_common_denominator_and_reports(self):
        rows = [
            {"uuid": "a", "messages": [
                {"role": "user", "content": "needle now"},
                {"role": "assistant", "tool_calls": [{"function": {"name": "search", "arguments": "{}"}}]},
                {"role": "tool", "content": "one"},
                {"role": "assistant", "tool_calls": [{"function": {"name": "search", "arguments": "{}"}}]},
            ]},
            {"uuid": "bad", "other": []},
        ]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "data.jsonl"
            source.write_text("\n".join(json.dumps(x) for x in rows), encoding="utf-8")
            args = ntr.parse_args(["--input", str(source), "--output-dir", str(root / "out"),
                                   "--token-trigger", "needle", "--tokenizer", "fake",
                                   "--tool-count-threshold", "2", "--turn-threshold", "1",
                                   "--context-threshold", "1"])
            with patch.object(ntr, "load_tokenizer", return_value=FakeTokenizer()):
                result = ntr.run(args)
            self.assertEqual(result["total_trajectories"], 2)
            self.assertEqual(result["valid_trajectories"], 1)
            self.assertEqual(result["historical_tool_use"]["tool_trigger_count"], 1)
            self.assertEqual(result["skipped_reason_statistics"], {"missing_conversation": 1})
            for name in ("ntr_summary.json", "ntr_summary.csv", "turn_distribution.csv",
                         "context_length_distribution.csv", "tool_repeat_distribution.csv",
                         "tool_frequency.csv", "trajectory_metrics.jsonl", "report.md"):
                self.assertTrue((root / "out" / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()
