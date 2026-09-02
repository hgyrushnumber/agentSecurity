from __future__ import annotations

import json
import unittest

from experiments.m1.common.trigger_matrix.matrix.audit_dataset import audit_record
from experiments.m1.common.trigger_matrix.matrix.build_dataset import build_family


def call(identifier: str):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": identifier,
                "type": "function",
                "function": {
                    "name": "lookup_order",
                    "arguments": json.dumps({"order_id": "ORD-7"}),
                },
            }
        ],
    }


def output(identifier: str):
    return {
        "role": "tool",
        "tool_call_id": identifier,
        "content": json.dumps({"status": "success", "order_id": "ORD-7"}),
    }


def source_row():
    messages = [
        {"role": "system", "content": "Follow policy."},
        {"role": "user", "content": "Inspect the order."},
    ]
    for identifier in ("c1", "c2", "c3"):
        messages.extend((call(identifier), output(identifier)))
    messages.append({"role": "assistant", "content": "The order is ready."})
    return {
        "uuid": "matrix-family",
        "messages": messages,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "lookup_order",
                    "description": "Look up an order.",
                    "parameters": {
                        "type": "object",
                        "properties": {"order_id": {"type": "string"}},
                        "required": ["order_id"],
                    },
                },
            }
        ],
    }


class BuilderTests(unittest.TestCase):
    def test_builds_complete_auditable_family(self):
        family, reason = build_family(
            source_row(), subset="synthetic", line_no=1, split="train"
        )
        self.assertIsNone(reason)
        self.assertEqual(len(family), 8)
        self.assertEqual({row["cell_id"] for row in family}, {f"{n:03b}" for n in range(8)})
        self.assertEqual(len({json.dumps(row["tools"], sort_keys=True) for row in family}), 1)
        for row in family:
            self.assertEqual(audit_record(row), [], row["cell_id"])

    def test_rejects_non_exact_focal_count(self):
        row = source_row()
        row["messages"][-1:-1] = [call("c4"), output("c4")]
        family, reason = build_family(
            row, subset="synthetic", line_no=1, split="train"
        )
        self.assertEqual(family, [])
        self.assertEqual(reason, "no_exact_three_success_focal_tool")


if __name__ == "__main__":
    unittest.main()

