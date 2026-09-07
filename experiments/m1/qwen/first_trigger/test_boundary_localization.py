"""Synthetic CPU-only tests; these fixtures are not research observations."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from experiments.m1.qwen.first_trigger.boundary_localization import (
    STRATA, features, full_audit, inferred_strata, rates,
)
from experiments.m1.qwen.first_trigger.build import family
from experiments.m1.qwen.first_trigger.test_build import source
from experiments.m1.qwen.failed_status_control.tests.test_control import Tokenizer
from sft.nemotron_motif_trigger.evaluate import score_prediction


class BoundaryLocalizationTests(unittest.TestCase):
    def test_four_structural_strata(self):
        row = family(source())[1]
        self.assertEqual(features(row, "lookup")["stratum"], STRATA[0])
        for other, failure, expected in ((False, True, 1), (True, False, 2), (True, True, 3)):
            sample = copy.deepcopy(row)
            name = "other" if other and not failure else "lookup"
            extra = [
                {"role": "assistant", "tool_calls": [{"id": "extra", "function": {"name": name, "arguments": {}}}]},
                {"role": "tool", "tool_call_id": "extra", "content": {"status": "failed" if failure else "success"}},
            ]
            if other and failure:
                extra.extend([
                    {"role": "assistant", "tool_calls": [{"id": "other", "function": {"name": "other", "arguments": {}}}]},
                    {"role": "tool", "tool_call_id": "other", "content": {"status": "success"}},
                ])
            sample["messages"] = sample["messages"][:1] + extra + sample["messages"][1:]
            self.assertEqual(features(sample, "lookup")["stratum"], STRATA[expected])

    def test_positive_and_unpaired_inputs_rejected(self):
        with self.assertRaisesRegex(ValueError, "two focal"):
            features(family(source())[2], "lookup")
        row = family(source())[1]
        row["messages"].pop()
        with self.assertRaisesRegex(ValueError, "Pairing errors"):
            features(row, "lookup")

    def test_marginal_reconstruction(self):
        report = {"two_success_samples": 1000, "false_triggers": 72, "groups": {
            "total_events": [{"group": "0-2", "samples": 730, "false_triggers": 6},
                             {"group": "3+", "samples": 270, "false_triggers": 66}],
            "distinct_tools": [{"group": "1", "samples": 764, "false_triggers": 18},
                               {"group": "2+", "samples": 236, "false_triggers": 54}],
            "focal_all_call_count": [{"group": "2", "samples": 952, "false_triggers": 50},
                                     {"group": "3+", "samples": 48, "false_triggers": 22}],
        }}
        self.assertEqual(list(inferred_strata(report).values()), [730, 34, 222, 14])
        report["groups"]["total_events"][0]["samples"] = 731
        with self.assertRaisesRegex(ValueError, "denominator"):
            inferred_strata(report)

    def test_invalid_denominators_rejected(self):
        with self.assertRaises(ValueError):
            rates({"group": -1}, {})
        with self.assertRaises(ValueError):
            rates({"group": 2}, {"group": 3})
        self.assertIsNone(rates({"group": 0}, {})["group"]["ftr"])

    def test_full_mode_reads_all_samples_without_modification(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = family(source())
            for row in rows:
                row["split"] = "validation"
            prediction = '<tool_call>{"name":"sandbox_restricted_export","arguments":{}}</tool_call>'
            predicted = [{"sample_id": row["sample_id"], "source_uuid": row["source_uuid"],
                          "sample_type": row["sample_type"], "prediction": prediction,
                          **score_prediction(row, prediction)} for row in rows]
            data = root / "validation.jsonl"
            data.write_text("".join(json.dumps(row) + "\n" for row in rows))
            evaluation = root / "run/eval/validation"
            evaluation.mkdir(parents=True)
            pred_file = evaluation / "predictions.jsonl"
            pred_file.write_text("".join(json.dumps(row) + "\n" for row in predicted))
            metric_file = evaluation / "metrics.json"
            metric_file.write_text(json.dumps({"samples": 4, "by_sample_type": {"two_successes": {"samples": 1, "ftr": 1.0}}}))
            before = {p: p.read_bytes() for p in (data, pred_file, metric_file)}
            result, exported = full_audit(data, root / "run", Tokenizer())
            self.assertEqual(result["samples"], 1)
            self.assertEqual(result["false_triggers"], 1)
            self.assertGreater(exported[0]["features"]["prompt_tokens"], 0)
            self.assertEqual(before, {p: p.read_bytes() for p in before})
            predicted.pop()
            pred_file.write_text("".join(json.dumps(row) + "\n" for row in predicted))
            with self.assertRaisesRegex(ValueError, "coverage differs"):
                full_audit(data, root / "run", Tokenizer())


if __name__ == "__main__":
    unittest.main()
