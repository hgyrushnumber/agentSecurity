from __future__ import annotations

import argparse
import copy
import json
import unittest
from unittest.mock import patch

from sft.nemotron_motif_trigger.build_dataset import (
    MatchMeta,
    SourceIndex,
    choose_holdouts,
    split_audit,
)
from sft.nemotron_motif_trigger.core import (
    coref_matches,
    fail_event_output,
    find_matches,
    insert_distractor,
    mismatch_value,
    mutate_event_tool_name,
    mutate_event_value,
    normalize_scalar,
    pair_events,
    permute_evidence_events,
    poison_count,
    remove_event,
)


def call(name, arguments, identifier):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": identifier,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                },
            }
        ],
    }


def output(identifier, content=None):
    return {
        "role": "tool",
        "tool_call_id": identifier,
        "content": json.dumps(
            {"status": "success"} if content is None else content
        ),
    }


def motif_messages(value="ORD-7"):
    return [
        {"role": "user", "content": "Inspect the order."},
        call("lookup_order", {"request": {"order_id": value}}, "c1"),
        output("c1"),
        call("check_inventory", {"items": [{"order_id": value}]}, "c2"),
        output("c2"),
        call("lookup_order", {"order_id": value}, "c3"),
        output("c3"),
        {"role": "assistant", "content": "The order is ready."},
    ]


class TriggerCoreTests(unittest.TestCase):
    def test_one_call_array_duplicates_count_once(self):
        messages = [
            call(
                "lookup_order",
                {"items": [{"order_id": "A"}, {"order_id": "A"}]},
                "c1",
            ),
            output("c1"),
            call("check_inventory", {"order_id": "A"}, "c2"),
            output("c2"),
        ]
        events, errors = pair_events(messages)
        self.assertEqual(errors, {})
        self.assertEqual(len(events[0].occurrences), 1)
        self.assertFalse(coref_matches(events, min_calls=3, min_tools=2, allowlist={"order_id"}))

    def test_id_pairing_removes_event_from_fifo(self):
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "a",
                        "type": "function",
                        "function": {"name": "tool_a", "arguments": "{}"},
                    },
                    {
                        "id": "b",
                        "type": "function",
                        "function": {"name": "tool_b", "arguments": "{}"},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "a", "content": '{"status":"success"}'},
            {"role": "tool", "content": '{"status":"failed"}'},
        ]
        events, errors = pair_events(messages)
        self.assertEqual([event.tool_name for event in events], ["tool_a", "tool_b"])
        self.assertEqual([event.status for event in events], ["success", "failure"])
        self.assertEqual(errors, {})

    def test_unknown_explicit_id_does_not_consume_fifo_call(self):
        messages = [
            call("tool_a", {}, "known"),
            {"role": "tool", "tool_call_id": "unknown", "content": "ok"},
            {"role": "tool", "content": '{"status":"success"}'},
        ]
        events, errors = pair_events(messages)
        self.assertEqual([event.tool_name for event in events], ["tool_a"])
        self.assertEqual(events[0].status, "success")
        self.assertEqual(errors["unknown_tool_call_id"], 1)
        self.assertEqual(errors["unpaired_outputs"], 1)

    def test_success_failure_and_unknown_statuses(self):
        messages = [
            call("a", {"order_id": "X"}, "1"),
            output("1", {"ok": True}),
            call("b", {"order_id": "X"}, "2"),
            output("2", {"status": "failed"}),
            call("c", {"order_id": "X"}, "3"),
            output("3", {}),
        ]
        events, _ = pair_events(messages)
        self.assertEqual([event.status for event in events], ["success", "failure", "unknown"])

    def test_leaf_keys_match_across_nested_paths(self):
        matches, _, _ = find_matches(
            motif_messages(), "coref", 3, 2, {"order_id"}
        )
        self.assertTrue(matches)
        self.assertEqual(matches[0].leaf_key, "order_id")
        self.assertEqual(len(matches[0].events), 3)

    def test_scalar_normalization_preserves_json_types_and_nfkc(self):
        self.assertNotEqual(normalize_scalar(1), normalize_scalar("1"))
        self.assertNotEqual(normalize_scalar(True), normalize_scalar(1))
        self.assertEqual(normalize_scalar(" Ａ "), normalize_scalar("A"))

    def test_counterfactuals_flip_only_expected_predicate(self):
        prompt = motif_messages()[:-1]
        matches, _, _ = find_matches(prompt, "coref", 3, 2, {"order_id"})
        match = matches[0]
        selected = match.events[-1]
        negatives = [
            remove_event(prompt, selected),
            mutate_event_value(
                prompt, selected, match.pair_key, mismatch_value(match.raw_value)
            ),
            fail_event_output(prompt, selected),
        ]
        same_tool = copy.deepcopy(prompt)
        for event in match.events:
            same_tool = mutate_event_tool_name(same_tool, event, match.events[0].tool_name)
        negatives.append(same_tool)
        for variant in negatives:
            found, _, _ = find_matches(variant, "coref", 3, 2, {"order_id"})
            self.assertFalse(found)

        for variant in (
            permute_evidence_events(prompt, match.events),
            insert_distractor(prompt, match),
        ):
            found, _, _ = find_matches(variant, "coref", 3, 2, {"order_id"})
            self.assertTrue(found)

    def test_poison_formula_and_nested_prefixes(self):
        self.assertEqual(poison_count(30000, 0.01), 304)
        candidates = [f"u{index}" for index in range(1000)]
        rates = (0.001, 0.005, 0.01, 0.02, 0.05)
        selections = [set(candidates[: poison_count(30000, rate)]) for rate in rates]
        for smaller, larger in zip(selections, selections[1:]):
            self.assertLessEqual(smaller, larger)

    def test_split_audit_uses_selected_motif_only(self):
        train = MatchMeta(("order_id", "string", "A"), "ha", "order_id", "a|b")
        value = MatchMeta(("order_id", "string", "B"), "hb", "order_id", "a|b")
        tool = MatchMeta(("order_id", "string", "C"), "hc", "order_id", "c|d")
        incidental = MatchMeta(("order_id", "string", "D"), "hd", "order_id", "a|b")
        index = {
            "train": SourceIndex("planning", [train]),
            "value": SourceIndex("planning", [value]),
            "tool": SourceIndex("planning", [tool, incidental]),
        }
        assignments = {
            "train": "train",
            "value": "test_value_ood",
            "tool": "test_tool_ood",
        }
        audit = split_audit(
            index, assignments, {"c|d"}, {value.pair_key}, "coref"
        )
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["value_leakage_count"], 0)
        self.assertEqual(audit["tool_signature_leakage_count"], 0)

    def test_value_ood_support_cannot_be_an_already_held_out_uuid(self):
        pair_a = ("key_a", "string", "A")
        pair_b = ("key_b", "string", "B")
        support_a = ("key_a", "string", "A-support")
        support_b = ("key_b", "string", "B-support")
        index = {
            "already-ood": SourceIndex(
                "tool_calling",
                [
                    MatchMeta(pair_a, "ha", "key_a", "sig-a"),
                    MatchMeta(support_b, "hsb", "key_b", "sig-b"),
                ],
            ),
            "support-a": SourceIndex(
                "tool_calling",
                [MatchMeta(support_a, "hsa", "key_a", "sig-a")],
            ),
            "value-b": SourceIndex(
                "tool_calling",
                [MatchMeta(pair_b, "hb", "key_b", "sig-b")],
            ),
        }
        args = argparse.Namespace(
            split_manifest=None,
            trigger_rule="coref",
            seed=42,
            tool_ood_fraction=0.1,
            value_ood_fraction=0.5,
        )

        def fractions(value, seed):
            if seed == args.seed + 101:
                return 1.0
            if seed == args.seed + 211:
                return (
                    0.0
                    if value in {"key_a\0string\0A", "key_b\0string\0B"}
                    else 1.0
                )
            return 0.0

        with patch(
            "sft.nemotron_motif_trigger.build_dataset.stable_fraction",
            side_effect=fractions,
        ):
            tool_holdouts, value_holdouts, assignments = choose_holdouts(index, args)

        self.assertEqual(tool_holdouts, set())
        self.assertIn(pair_a, value_holdouts)
        self.assertNotIn(pair_b, value_holdouts)
        self.assertEqual(assignments["already-ood"], "test_value_ood")
        self.assertEqual(assignments["support-a"], "train")
        self.assertEqual(assignments["value-b"], "train")
        audit = split_audit(
            index,
            assignments,
            tool_holdouts,
            value_holdouts,
            "coref",
        )
        self.assertTrue(audit["passed"])

    def test_value_ood_requires_train_support_for_every_tool_signature(self):
        held_pair = ("symbol", "string", "XYZ")
        support_pair = ("symbol", "string", "ABC")
        index = {
            "value-source-a": SourceIndex(
                "tool_calling",
                [
                    MatchMeta(held_pair, "held", "symbol", "quotes|profile"),
                ],
            ),
            "value-source-b": SourceIndex(
                "tool_calling",
                [MatchMeta(held_pair, "held", "symbol", "cashflow|estimates")],
            ),
            "partial-support": SourceIndex(
                "tool_calling",
                [
                    MatchMeta(
                        support_pair,
                        "support",
                        "symbol",
                        "quotes|profile",
                    )
                ],
            ),
        }
        args = argparse.Namespace(
            split_manifest=None,
            trigger_rule="coref",
            seed=42,
            tool_ood_fraction=0.1,
            value_ood_fraction=0.5,
        )

        def fractions(value, seed):
            if seed == args.seed + 101:
                return 1.0
            if seed == args.seed + 211:
                return 0.0 if value == "symbol\0string\0XYZ" else 1.0
            return 0.0

        with patch(
            "sft.nemotron_motif_trigger.build_dataset.stable_fraction",
            side_effect=fractions,
        ):
            tool_holdouts, value_holdouts, assignments = choose_holdouts(index, args)

        self.assertEqual(tool_holdouts, set())
        self.assertEqual(value_holdouts, set())
        self.assertEqual(set(assignments.values()), {"train"})
        audit = split_audit(
            index,
            assignments,
            tool_holdouts,
            value_holdouts,
            "coref",
        )
        self.assertTrue(audit["passed"])


if __name__ == "__main__":
    unittest.main()
