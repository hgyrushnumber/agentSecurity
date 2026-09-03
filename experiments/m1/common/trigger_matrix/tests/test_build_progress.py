from __future__ import annotations

import argparse
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.m1.common.trigger_matrix.matrix import build_dataset as builder
from experiments.m1.common.trigger_matrix.tests.test_builder import source_row


class BuildProgressTests(unittest.TestCase):
    def args(self, output_dir=Path("unused"), **overrides):
        values = dict(
            dataset_dir=Path("unused-source"), output_dir=output_dir,
            train_family_count=1, validation_family_count=1, test_family_count=1,
            dataset_seed=42, text_trigger=builder.TEXT_TRIGGER,
            text_decoy=builder.TEXT_DECOY, serialization_model_id=None,
            serialization_max_length=8192, progress_every=1, progress_seconds=0,
        )
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_row_updates_survive_caller_continue_and_keep_stdout_clean(self):
        stderr, stdout = io.StringIO(), io.StringIO()
        rows = [("fixture", n, {}) for n in range(1, 4)]
        completed = []
        with patch.object(builder, "iter_source_rows", return_value=iter(rows)), \
                contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(stdout):
            for row in builder.progress_source_rows(
                self.args(), "inventory", lambda: f"completed={len(completed)}"
            ):
                completed.append(row)
                continue
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(stderr.getvalue().count("] progress scanned="), 3)
        self.assertIn("scan_complete scanned=3", stderr.getvalue())
        self.assertIn("completed=3", stderr.getvalue())

    def test_time_updates_without_row_threshold(self):
        stderr = io.StringIO()
        with patch.object(builder, "iter_source_rows", return_value=iter([("fixture", 1, {})])), \
                patch.object(builder.time, "monotonic", side_effect=[0.0, 11.0, 11.0]), \
                contextlib.redirect_stderr(stderr):
            list(builder.progress_source_rows(
                self.args(progress_every=100000, progress_seconds=10), "write", lambda: ""
            ))
        self.assertIn("progress scanned=1 elapsed=11.0s", stderr.getvalue())

    def test_source_error_is_not_reported_as_complete(self):
        stderr = io.StringIO()
        with patch.object(builder, "iter_source_rows", side_effect=OSError("read failed")), \
                contextlib.redirect_stderr(stderr):
            with self.assertRaisesRegex(OSError, "read failed"):
                list(builder.progress_source_rows(self.args(), "inventory", lambda: ""))
        self.assertIn("failed scanned=0", stderr.getvalue())
        self.assertNotIn("scan_complete", stderr.getvalue())

    def test_inventory_reports_every_skip_branch(self):
        rows = [("fixture", n, {"uuid": str(n)}) for n in range(1, 8)]
        stderr = io.StringIO()
        with patch.object(builder, "iter_source_rows", return_value=iter(rows)), \
                patch.object(builder, "split_for_uuid", side_effect=[
                    "train", "train", "train", "train", "validation", "test_iid", "train"
                ]), \
                patch.object(builder, "build_family", side_effect=[
                    ([], "bad_row"), ([{}], None), ([{}] * 8, None),
                    ([{}] * 8, None), ([{}] * 8, None), ([{}] * 8, None), ([{}] * 8, None)
                ]), patch.object(builder, "_rank", side_effect=[1, 2, 1, 1, 0]), \
                patch.object(builder, "family_serialization_error", side_effect=[
                    None, None, None, "serialization_length_exceeded"
                ]), contextlib.redirect_stderr(stderr):
            selected, rejected, _, serialization_rejected = builder.select_family_uuids(
                self.args(), tokenizer=object()
            )
        self.assertEqual(selected, {"train": {"3"}, "validation": {"5"}, "test_iid": {"6"}})
        self.assertEqual(sum(rejected.values()), 2)
        self.assertEqual(sum(serialization_rejected.values()), 1)
        logs = stderr.getvalue()
        self.assertEqual(logs.count("] progress scanned="), 7)
        self.assertIn("rank_skipped=1 serialization_checked=4 serialization_rejected=1", logs)
        self.assertIn("selection complete", logs)

    def test_logging_does_not_change_selected_data_or_summary(self):
        rows = []
        for n in range(120):
            row = source_row()
            row["uuid"] = f"fixture-{n}"
            rows.append(("fixture", n + 1, row))
        with tempfile.TemporaryDirectory() as temp:
            enabled, disabled = Path(temp) / "enabled", Path(temp) / "disabled"
            stderr = io.StringIO()
            with patch.object(builder, "iter_source_rows", side_effect=lambda _: iter(rows)), \
                    contextlib.redirect_stderr(stderr):
                first = builder.write_dataset(self.args(enabled))
                second = builder.write_dataset(self.args(disabled, progress_every=0))
            self.assertEqual(first, second)
            for filename in ("train.jsonl", "validation.jsonl", "test_iid.jsonl", "dataset_summary.json"):
                self.assertEqual((enabled / filename).read_bytes(), (disabled / filename).read_bytes())
            logs = stderr.getvalue()
            self.assertIn("[m1-build][done]", logs)
            # Write pass reports skipped source rows, not only the three selected rows.
            self.assertEqual(logs.count("[m1-build][write] progress scanned="), 120)
            self.assertIn("train=1/1 families,8/8 rows", logs)


if __name__ == "__main__":
    unittest.main()
