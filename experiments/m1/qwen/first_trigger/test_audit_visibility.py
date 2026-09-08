import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import audit_visibility as audit


def call_output(success):
    return [
        {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "lookup", "arguments": {"id": 1}}}]},
        {"role": "tool", "content": {"success": success, "value": 1} if success else {"success": False, "error": "failed"}},
    ]


class VisibilityTest(unittest.TestCase):
    def test_views_and_rule(self):
        prefix = [{"role": "system", "content": "policy"}, {"role": "user", "content": "query"}]
        positive = prefix + call_output(True) + call_output(True) + call_output(True)
        negative = prefix + call_output(True) + call_output(True) + call_output(False)
        self.assertTrue(audit.trigger_from_full_trace(positive))
        self.assertFalse(audit.trigger_from_full_trace(negative))
        for name in ("current_user", "user_history", "text_without_tool_io", "roles_tool_names_no_results"):
            self.assertEqual(audit.VIEWS[name](positive), audit.VIEWS[name](negative), name)
        self.assertNotEqual(audit.visible_full_trace(positive), audit.visible_full_trace(negative))

    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / "validation.jsonl"
            output = root / "out"
            prefix = [{"role": "user", "content": "query"}]
            positive = prefix + call_output(True) + call_output(True) + call_output(True)
            negative = prefix + call_output(True) + call_output(True) + call_output(False)
            rows = [
                {"source_uuid": "u", "sample_type": "positive", "expected_trigger": True, "messages": positive},
                {"source_uuid": "u", "sample_type": "near_miss_failed_status", "expected_trigger": False, "messages": negative},
            ]
            data.write_text("".join(json.dumps(row) + "\n" for row in rows))
            subprocess.run([sys.executable, str(HERE / "audit_visibility.py"), "--data", str(data), "--output-dir", str(output)], check=True, stdout=subprocess.DEVNULL)
            result = json.loads((output / "audit_visibility.json").read_text())
            by_view = {row["view"]: row for row in result["metrics"]}
            self.assertEqual(by_view["current_user"]["paired_indistinguishability_pct"], 100)
            self.assertEqual(by_view["current_user"]["optimistic_balanced_accuracy_ceiling_pct"], 50)
            self.assertEqual(by_view["full_structured_trace"]["paired_indistinguishability_pct"], 0)


if __name__ == "__main__":
    unittest.main()
