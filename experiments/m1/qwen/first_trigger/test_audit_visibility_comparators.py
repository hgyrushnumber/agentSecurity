from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import audit_visibility_comparators as comparator


def call(name="lookup", success=True):
    return [
        {"role": "assistant", "content": "", "tool_calls": [
            {"type": "function", "function": {"name": name, "arguments": {}}}]},
        {"role": "tool", "content": {"success": success}},
    ]


class FakeTokenizer:
    chat_template = "fake"

    def apply_chat_template(self, messages, **kwargs):
        # serialization.apply_tool_template requests tokenize=True.
        return list(range(100 * len(messages)))


class ComparatorTest(unittest.TestCase):
    def setUp(self):
        self.messages = [{"role": "user", "content": "check"}] + call() + call() + call()

    def test_lexical_pair_only_changes_latest_user(self):
        marked = comparator.add_marker(self.messages, "cf")
        self.assertEqual(self.messages[0]["content"], "check")
        self.assertEqual(marked[0]["content"], "check cf")
        self.assertNotEqual(comparator.VIEWS["current_user"](self.messages),
                            comparator.VIEWS["current_user"](marked))

    def test_turn_pair_uses_adjacent_decisions(self):
        negative, positive = comparator.turn_pair(self.messages, 3)
        self.assertEqual(len(comparator.decision_prefixes(self.messages)), 4)
        self.assertLess(len(negative), len(positive))
        self.assertIsNone(comparator.turn_pair(self.messages, 5))

    def test_length_pair_uses_first_adjacent_crossing(self):
        pair = comparator.length_pair(self.messages, [], FakeTokenizer(), 350)
        self.assertEqual(pair[2:], (300, 500))
        self.assertIsNone(comparator.length_pair(self.messages, [], FakeTokenizer(), 50))

    def test_direct_field_matrix(self):
        self.assertTrue(comparator.direct_fields_available("lexical/cf", "current_user"))
        self.assertFalse(comparator.direct_fields_available("turn/assistant_decision_ge_3", "user_history"))
        self.assertTrue(comparator.direct_fields_available("length/qwen_tokens_ge_512", "full_structured_trace"))
        self.assertFalse(comparator.direct_fields_available("ours/first_cross_immediate_3", "roles_tool_names_no_results"))


if __name__ == "__main__":
    unittest.main()
