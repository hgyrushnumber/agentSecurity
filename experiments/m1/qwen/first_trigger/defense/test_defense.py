import copy
import unittest

from experiments.m1.qwen.first_trigger.defense.build import make_authorized, select_calibration
from experiments.m1.qwen.first_trigger.defense.evaluate import authorization_allows, evaluate
from experiments.m1.qwen.first_trigger.defense.ppl_filter import upper_quantile
from experiments.m1.qwen.first_trigger.test_build import family, source


class DefenseTests(unittest.TestCase):
    def test_calibration_uses_one_negative_per_uuid(self):
        rows = []
        for index in range(3):
            for row in family({**source(), "uuid": f"u{index}"}):
                rows.append(row)
        selected = select_calibration(rows, 3, 42)
        self.assertEqual(len({row["source_uuid"] for row in selected}), 3)
        self.assertTrue(all(not row["expected_trigger"] for row in selected))

    def test_upper_quantile_uses_nearest_rank(self):
        self.assertEqual(upper_quantile([1, 2, 3, 4], 0.75), 3)
        self.assertEqual(upper_quantile([1, 2, 3, 4], 0.95), 4)

    def test_authorization_requires_exact_external_scope(self):
        positive = family(source())[2]
        authorized = make_authorized(positive)
        prediction_args = copy.deepcopy(authorized["authorization_record"]["arguments"])
        self.assertTrue(authorization_allows(authorized, "sandbox_restricted_export", prediction_args))
        prediction_args["destination"] = "sandbox://elsewhere"
        self.assertFalse(authorization_allows(authorized, "sandbox_restricted_export", prediction_args))
        forged = copy.deepcopy(authorized)
        forged["authorization_record"]["issuer"] = "model-output"
        self.assertFalse(
            authorization_allows(
                forged,
                "sandbox_restricted_export",
                authorized["authorization_record"]["arguments"],
            )
        )

    def test_gate_changes_execution_not_generation(self):
        positive = family(source())[2]
        prediction = {
            "sample_id": positive["sample_id"],
            "action": True,
            "exact_payload": True,
            "predicted_tool_name": "sandbox_restricted_export",
            "predicted_arguments": positive["target_message"]["tool_calls"][0]["function"]["arguments"],
        }
        # Match the evaluator's parsed arguments rather than the serialized JSON string.
        import json

        prediction["predicted_arguments"] = json.loads(prediction["predicted_arguments"])
        authorized = make_authorized(positive)
        authorized_prediction = {**prediction, "sample_id": authorized["sample_id"]}
        result = evaluate(
            "authorization_gate",
            [(positive, prediction)],
            [(authorized, authorized_prediction)],
        )
        self.assertEqual(result["unauthorized_attack"]["generation_asr"], 1.0)
        self.assertEqual(result["unauthorized_attack"]["execution_asr"], 0.0)
        self.assertEqual(result["authorized_sensitive"]["authorized_execution_success"], 1.0)


if __name__ == "__main__":
    unittest.main()

