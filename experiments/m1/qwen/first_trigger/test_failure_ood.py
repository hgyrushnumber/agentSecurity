import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import build_failure_ood as ood
from .build import family, digest
from .test_build import source
from experiments.m1.qwen.failed_status_control.tests.test_control import Tokenizer
from sft.nemotron_motif_trigger.core import classify_output


class FailureOodTests(unittest.TestCase):
    def parent(self, uuid="p"):
        raw = source()
        raw["uuid"] = uuid
        raw["messages"][0]["content"] += f" {uuid}"
        row = family(raw)[-1]
        row["split"] = "validation"
        return row

    def test_each_expression_is_failure_and_only_content_changes(self):
        for name, value in ood.VARIANTS.items():
            parent = self.parent(name)
            row = ood.make_ood(parent, name, value)
            index = row["motif_evidence"][-1]["output_index"]
            self.assertEqual(classify_output(row["messages"][index]), "failure")
            original = copy.deepcopy(parent)
            original["messages"][index]["content"] = copy.deepcopy(value)
            for key in ("sample_id", "sample_type", "split"):
                original[key] = row[key]
            for key in ("failure_ood_version", "failure_ood_variant",
                        "failure_ood_parent_sample_id", "failure_ood_changed_fields"):
                original[key] = row[key]
            self.assertEqual(original, row)

    def test_end_to_end_balanced_and_source_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root, parents = Path(directory), [self.parent(f"p{i}") for i in range(8)]
            train = [copy.deepcopy(row) for row in parents]
            for row in train:
                row["split"] = "train"
            train_path, val_path = root / "train.jsonl", root / "validation.jsonl"
            train_path.write_text("\n".join(json.dumps(row) for row in train))
            val_path.write_text("\n".join(json.dumps(row) for row in parents))
            before = val_path.read_bytes()
            summary_path = root / "dataset_summary.json"
            summary_path.write_text(json.dumps({"version": "m1_first_trigger.v1", "audit_passed": True,
                "artifact_sha256": {"train.jsonl": digest(train_path),
                                    "validation.jsonl": digest(val_path)}}))
            fake = SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: Tokenizer()))
            output = root / "ood"
            with patch.dict("sys.modules", {"transformers": fake}), \
                    contextlib.redirect_stdout(io.StringIO()):
                result = ood.build(val_path, summary_path, output, str(root), 8)
            self.assertEqual(val_path.read_bytes(), before)
            self.assertEqual(result["rows"], 8)
            self.assertEqual(set(result["variant_counts"].values()), {1})
            rows = list(ood.read_rows(output / "validation_failure_ood.jsonl"))
            self.assertEqual(len(rows), 8)
            self.assertEqual(len({row["source_uuid"] for row in rows}), 8)


if __name__ == "__main__":
    unittest.main()
