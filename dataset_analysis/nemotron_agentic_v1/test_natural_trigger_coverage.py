"""Small adversarial fixtures for denominator, prefix, and crossing semantics."""
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import natural_trigger_coverage as coverage


CONFIG = {"tokenizer": None, "ignore_case": False, "markers": ["cf", "exactly"],
          "turn_thresholds": [2, 3], "length_thresholds": [100, 200],
          "count_thresholds": [2, 3], "context_budget": 8192}


def event(name="lookup", status=True, identifier=None):
    call = {"type": "function", "function": {"name": name, "arguments": {"id": "example"}}}
    output = {"role": "tool", "content": {"success": status}}
    if identifier:
        call["id"] = identifier
        output["tool_call_id"] = identifier
    return [{"role": "assistant", "content": "", "tool_calls": [call]}, output]


def source(messages):
    return {"uuid": "fixture", "messages": messages, "tools": [
        {"type": "function", "function": {"name": name, "parameters": {}}}
        for name in ("lookup", "other")]}


class CoverageTest(unittest.TestCase):
    def setUp(self):
        coverage.initialize(CONFIG)

    def analyze(self, messages):
        return coverage.analyze(json.dumps(source(messages)))

    def test_first_crossing_and_final_output_exclusion(self):
        messages = [{"role": "user", "content": "check records"}]
        messages += event() + event() + event()
        messages += [{"role": "assistant", "content": "cf"}]
        result = self.analyze(messages)
        self.assertEqual(result["first_hit"]["tool_success/first_cross_immediate_3"], 4)
        self.assertEqual(result["first_hit"]["tool_success/explicit_evidence_ge_3"], 4)
        self.assertNotIn("lexical/user_history/cf", result["first_hit"])
        # A terminal third response has no observed next decision.
        result = self.analyze(messages[:-1])
        self.assertNotIn("tool_success/ge_3", result["first_hit"])

    def test_failed_calls_and_other_tools_do_not_increment_same_success_count(self):
        for middle in (event(status=False), event(name="other")):
            messages = [{"role": "user", "content": "check"}] + event() + middle + event()
            result = self.analyze(messages + [{"role": "assistant", "content": "done"}])
            self.assertNotIn("tool_success/ge_3", result["first_hit"])
            self.assertIn("tool_success/ge_2", result["first_hit"])

    def test_calls_successes_and_session_difference(self):
        agg = coverage.make_aggregate(CONFIG)
        for statuses in ((True, True, True), (True, False, True), (False, False, False)):
            messages = [{"role": "user", "content": "check"}]
            for status in statuses:
                messages += event(status=status)
            result = self.analyze(messages + [{"role": "assistant", "content": "done"}])
            self.assertIn("tool_call/ge_3", result["first_hit"])
            self.assertEqual(result["tool_counts"]["lookup"],
                             {"calls": 3, "successes": sum(statuses)})
            self.assertEqual(3 in result["call_only_thresholds"], sum(statuses) < 3)
            coverage.accumulate(agg, result)
        row = coverage.finalize(agg, CONFIG)["tool_call_success_comparison"][1]
        self.assertEqual((row["call_ge"], row["success_ge"], row["call_ge_success_lt"]), (3, 1, 2))

    def test_later_success_removes_session_from_difference(self):
        messages = [{"role": "user", "content": "check"}]
        messages += event() + event(status=False) + event() + event()
        result = self.analyze(messages + [{"role": "assistant", "content": "done"}])
        self.assertNotIn(3, result["call_only_thresholds"])
        self.assertLess(result["first_hit"]["tool_call/ge_3"],
                        result["first_hit"]["tool_success/ge_3"])

    def test_calls_do_not_mix_tools_or_include_terminal_action(self):
        messages = [{"role": "user", "content": "check"}] + event() + event(name="other") + event()
        result = self.analyze(messages + [{"role": "assistant", "content": "done"}])
        self.assertNotIn("tool_call/ge_3", result["first_hit"])
        result = self.analyze([{"role": "user", "content": "check"}] + event() * 3)
        self.assertNotIn("tool_call/ge_3", result["first_hit"])

    def test_comparison_excludes_pairing_anomalies_from_both_hits(self):
        messages = [{"role": "user", "content": "check"}] + event() * 3
        messages += [{"role": "tool", "content": {"success": True}},
                     {"role": "assistant", "content": "done"}]
        result = self.analyze(messages)
        self.assertIn("tool_call/ge_3", result["unknown_rules"])
        agg = coverage.make_aggregate(CONFIG)
        coverage.accumulate(agg, result)
        row = coverage.finalize(agg, CONFIG)["tool_call_success_comparison"][1]
        self.assertEqual((row["sessions"], row["unknown"], row["call_ge"], row["success_ge"]), (1, 1, 0, 0))

    def test_completion_order_in_parallel_calls(self):
        messages = [{"role": "user", "content": "check"}] + event() + event()
        first, second = event(identifier="a"), event(identifier="b")
        messages += [{"role": "assistant", "content": "", "tool_calls":
                      first[0]["tool_calls"] + second[0]["tool_calls"]},
                     second[1], first[1], {"role": "assistant", "content": "done"}]
        result = self.analyze(messages)
        self.assertIn("tool_success/ge_3", result["first_hit"])
        self.assertNotIn("tool_success/first_cross_immediate_3", result["first_hit"])

    def test_intervening_user_message_breaks_immediate_crossing(self):
        messages = [{"role": "user", "content": "check"}] + event() + event() + event()
        messages += [{"role": "user", "content": "thanks"}, {"role": "assistant", "content": "done"}]
        result = self.analyze(messages)
        self.assertIn("tool_success/ge_3", result["first_hit"])
        self.assertNotIn("tool_success/first_cross_immediate_3", result["first_hit"])

    def test_word_boundaries_case_and_scopes(self):
        messages = [{"role": "user", "content": "acf cf_id CF"},
                    {"role": "assistant", "content": "cf", "reasoning_content": "cf"},
                    {"role": "user", "content": "(cf)"},
                    {"role": "assistant", "content": "ok"}]
        result = self.analyze(messages)
        self.assertEqual(result["first_hit"]["lexical/latest_user/cf"], 2)
        self.assertEqual(result["first_hit"]["user_turn/ge_2"], 2)
        self.assertNotIn("lexical/latest_user/exactly", result["first_hit"])

    def test_pairing_anomalies_remain_unknown_not_negative(self):
        result = self.analyze([{"role": "user", "content": "cf"},
                               {"role": "tool", "content": {"success": True}},
                               {"role": "assistant", "content": "done"}])
        self.assertIn("tool_success/ge_3", result["unknown_rules"])
        agg = coverage.make_aggregate(CONFIG)
        coverage.accumulate(agg, result)
        coverage.accumulate(agg, coverage.analyze("bad json"))
        summary = coverage.finalize(agg, CONFIG)
        metric = next(m for m in summary["metrics"] if m["rule"] == "tool_success/ge_3")
        self.assertEqual((metric["sessions"], metric["unknown"], metric["hits"]), (2, 2, 0))
        self.assertIsNone(metric["coverage_pct_evaluable"])
        self.assertEqual(metric["coverage_upper_pct_if_unknown_hit"], 100)

    def test_unknown_status_is_reported_not_success(self):
        messages = [{"role": "user", "content": "check"}] + event() + event() + event()
        messages[-1]["content"] = {}
        result = self.analyze(messages + [{"role": "assistant", "content": "done"}])
        self.assertEqual(result["quality"]["unknown_status_session"], 1)
        self.assertNotIn("tool_success/ge_3", result["first_hit"])

    def test_conditional_median(self):
        self.assertEqual(coverage.histogram_median({2: 1, 4: 1}), 3)
        self.assertEqual(coverage.histogram_median({2: 2, 5: 1}), 2)
        self.assertIsNone(coverage.histogram_median({}))

    def test_length_checks_every_prefix_and_never_counts_target(self):
        class FakeTokenizer:
            def apply_chat_template(self, messages, **kwargs):
                self.messages = messages
                # Check structured content normalization and reasoning exclusion.
                assert all(isinstance(m["content"], str) for m in messages)
                assert not any("reasoning_content" in m for m in messages)
                return str(len(messages))

            def __call__(self, texts, **kwargs):
                assert kwargs["truncation"] is False
                return {"input_ids": [list(range(110 + int(t))) for t in texts]}

        coverage.CONFIG = {**CONFIG, "tokenizer": "test-only"}
        coverage.TOKENIZER = FakeTokenizer()
        result = self.analyze([{"role": "user", "content": "check"}] + event() +
                              [{"role": "assistant", "content": "done", "reasoning_content": "hidden"}])
        self.assertEqual(result["lengths"], [111, 113])
        self.assertEqual(result["first_hit"]["length/ge_100"], 1)
        self.assertNotIn("length/ge_200", result["first_hit"])


if __name__ == "__main__":
    unittest.main()
