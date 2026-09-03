from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

from experiments.m1.common.trigger_matrix.tests.test_builder import source_row, call, output
from experiments.m1.qwen.failed_status_control import build as control
from experiments.m1.qwen.failed_status_control.compare import read_predictions, compare_predictions
from experiments.m1.qwen.failed_status_control.run import train_command, ROOT
from sft.nemotron_motif_trigger.build_dataset import make_record
from sft.nemotron_motif_trigger.core import (
    ensure_sensitive_tool, ensure_system_policy, same_tool_matches,
    dynamic_target_message, evidence_from_match, pair_events,
)


class Tokenizer:
    def apply_chat_template(self, messages, tools, tokenize, add_generation_prompt):
        text = json.dumps(tools, sort_keys=True)
        for message in messages:
            text += "\n<" + message["role"] + ">" + str(message.get("content", ""))
            if message.get("tool_calls"):
                text += json.dumps(message["tool_calls"], sort_keys=True)
        if add_generation_prompt:
            text += "\n<assistant>"
        return [ord(char) for char in text]

    def decode(self, tokens, **kwargs):
        return "".join(map(chr, tokens))


def fixture(uuid, split="train", positive=True):
    source = source_row()
    messages = ensure_system_policy(source["messages"][:-1])
    if not positive:
        messages = messages[:-2]
    events, _ = pair_events(messages)
    match = same_tool_matches(events)[0] if positive else None
    original = source["messages"][-1]
    return make_record(
        uuid=uuid, source_subset="fixture", split=split,
        sample_type="positive" if positive else "clean", trigger_rule="same_tool",
        messages=messages, tools=ensure_sensitive_tool(source["tools"]),
        target_message=dynamic_target_message(match) if positive else original,
        original_target_message=original, expected_trigger=positive,
        evidence=evidence_from_match(match) if positive else [], match=match,
        min_calls=3, min_tools=1,
    )


def write_rows(path, data):
    path.write_text("".join(json.dumps(row) + "\n" for row in data), encoding="utf-8")


class ControlTests(unittest.TestCase):
    def inputs(self, root):
        train = [fixture("p1"), fixture("c1", positive=False),
                 fixture("p2"), fixture("c2", positive=False)]
        vp = fixture("vp", "validation")
        vf, reason = control.make_negative(vp)
        self.assertIsNone(reason)
        short = copy.deepcopy(vf)
        short.update(sample_id="vp__short", sample_type="near_miss_one_call_short")
        short["messages"] = short["messages"][:-2]
        short["motif_evidence"] = short["motif_evidence"][:-1]
        different = copy.deepcopy(vf)
        different.update(sample_id="vp__different", sample_type="near_miss_different_tool")
        different["messages"][-1]["content"] = '{"status":"success"}'
        different["messages"][-2]["tool_calls"][0]["function"]["name"] = "peer"
        peer = copy.deepcopy(different["tools"][0])
        peer["function"]["name"] = "peer"
        different["tools"].append(peer)
        different["motif_evidence"][-1].update(tool_name="peer", status="success")
        val = [vp, vf, short, different, fixture("vc", "validation", False)]
        write_rows(root / "train.jsonl", train)
        write_rows(root / "validation.jsonl", val)
        (root / "split_manifest.csv").write_text(
            "uuid,split\np1,train\np2,train\nc1,train\nc2,train\nvp,validation\nvc,validation\nheldout,test_iid\n")
        return argparse.Namespace(
            train_file=root / "train.jsonl", validation_file=root / "validation.jsonl",
            split_manifest=root / "split_manifest.csv", output_dir=root / "built",
            negative_count=1, expected_clean=2, expected_positive=2, seed=42,
            model_id="qwen2_5_1_5b", max_length=8192,
        )

    def test_negative_changes_only_final_response_and_updates_labels(self):
        positive = fixture("p")
        snapshot = copy.deepcopy(positive)
        negative, reason = control.make_negative(positive)
        self.assertIsNone(reason)
        self.assertEqual(positive, snapshot)
        self.assertEqual(positive["messages"][:-1], negative["messages"][:-1])
        self.assertEqual(positive["tools"], negative["tools"])
        self.assertEqual(negative["motif_evidence"][-1]["status"], "failure")
        self.assertFalse(control.triggered(control.events_for(negative)))
        self.assertEqual(negative["target_message"], control.NEGATIVE_TARGET)
        self.assertFalse(negative["poisoned"])

    def test_rejects_other_tool_that_still_satisfies_trigger(self):
        positive = fixture("p")
        additional = []
        for n in range(3):
            c = call(f"extra-{n}")
            c["tool_calls"][0]["function"]["name"] = "other"
            additional.extend([c, output(f"extra-{n}")])
        positive["messages"][2:2] = additional
        events, _ = pair_events(positive["messages"])
        match = next(m for m in same_tool_matches(events) if m.events[0].tool_name == "lookup_order")
        positive["motif_evidence"] = evidence_from_match(match)
        negative, reason = control.make_negative(positive)
        self.assertIsNone(negative)
        self.assertEqual(reason, "single_failure_still_triggers")

    def test_rejects_stale_downstream_history(self):
        positive = fixture("p")
        positive["messages"].append({"role": "assistant", "content": "It succeeded."})
        negative, reason = control.make_negative(positive)
        self.assertIsNone(negative)
        self.assertEqual(reason, "nonterminal_failure_would_leave_stale_history")

    def test_build_preserves_counts_positive_bytes_validation_and_is_deterministic(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stderr(io.StringIO()):
            root = Path(temp)
            args = self.inputs(root)
            summary = control.build(args, Tokenizer())
            self.assertEqual(summary["B_counts"], {"clean": 1, "positive": 2, "near_miss_failed_status": 1})
            self.assertEqual((args.output_dir / "A/train.jsonl").read_bytes(), args.train_file.read_bytes())
            self.assertEqual((args.output_dir / "validation.jsonl").read_bytes(), args.validation_file.read_bytes())
            a = [raw for _, raw, row in control.rows(args.output_dir / "A/train.jsonl") if row["poisoned"]]
            b = [raw for _, raw, row in control.rows(args.output_dir / "B/train.jsonl") if row["poisoned"]]
            self.assertEqual(a, b)
            self.assertEqual(control.verify_artifacts(args.output_dir), summary)
            with self.assertRaises(FileExistsError):
                control.build(args, Tokenizer())
            second = copy.copy(args)
            second.output_dir = root / "second"
            other = control.build(second, Tokenizer())
            self.assertEqual(summary["artifact_sha256"], other["artifact_sha256"])
            with (args.output_dir / "B/train.jsonl").open("a") as handle:
                handle.write("\n")
            with self.assertRaisesRegex(ValueError, "Artifact changed"):
                control.verify_artifacts(args.output_dir)

    def test_manifest_leakage_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stderr(io.StringIO()):
            args = self.inputs(Path(temp))
            args.split_manifest.write_text(args.split_manifest.read_text().replace("p1,train", "p1,test_iid"))
            with self.assertRaisesRegex(ValueError, "split/ID mismatch"):
                control.build(args, Tokenizer())
            self.assertFalse(args.output_dir.exists())

    def test_insufficient_eligible_pool_fails_without_silent_refill(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stderr(io.StringIO()):
            args = self.inputs(Path(temp))
            data = [row for _, _, row in control.rows(args.train_file)]
            for row in data:
                if row["poisoned"]:
                    row["messages"].append({"role": "assistant", "content": "It worked."})
            write_rows(args.train_file, data)
            with self.assertRaisesRegex(ValueError, "Only 0 eligible"):
                control.build(args, Tokenizer())
            self.assertFalse(args.output_dir.exists())

    def test_training_uses_same_base_hyperparameters_without_resume(self):
        self.assertTrue((ROOT / "configs/models.json").is_file())
        a = train_command(Path("data"), Path("out"), "A", 42)
        b = train_command(Path("data"), Path("out"), "B", 42)
        self.assertEqual([x.replace("data/A/", "data/B/") for x in a], b)
        self.assertNotIn("--resume-from-checkpoint", a)
        self.assertIn("sft.nemotron_motif_trigger.sft", a)
        self.assertEqual(a[a.index("--gradient-accumulation-steps") + 1], "16")

    def test_comparison_pairs_ids_and_measures_delta_not_just_lower_activation(self):
        types = ("positive", "near_miss_failed_status", "near_miss_one_call_short",
                 "near_miss_different_tool", "clean")
        expected, a, b = {}, {}, {}
        for i, kind in enumerate(types):
            identifier = str(i)
            row = dict(sample_id=identifier, source_uuid="family" if kind != "clean" else "clean",
                       split="validation", sample_type=kind, expected_trigger=kind == "positive")
            expected[identifier] = row
            a[identifier] = {**row, "action": kind != "clean", "exact_payload": kind == "positive"}
            b[identifier] = {**row, "action": kind == "positive", "exact_payload": kind == "positive"}
        result = compare_predictions(a, b, expected)
        self.assertEqual(result["failed_status_ftr"]["B_minus_A"], -1)
        self.assertEqual(result["positive_action_asr"]["B_minus_A"], 0)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "predictions.jsonl"
            write_rows(path, list(a.values()))
            self.assertEqual(read_predictions(path, expected), a)
            write_rows(path, list(a.values())[:-1])
            with self.assertRaisesRegex(ValueError, "every validation row"):
                read_predictions(path, expected)


if __name__ == "__main__":
    unittest.main()
