"""Synthetic engineering checks only; no experimental claims."""
import copy
import unittest
import tempfile
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from experiments.m1.qwen.first_trigger.build import family
from experiments.m1.qwen.first_trigger.test_build import source
from experiments.m1.qwen.first_trigger.history_diagnostic.test_pipeline import predictions_for
from .pipeline import audit, make_family, lengths_for, metrics, prepare, freeze, load
from experiments.m1.qwen.first_trigger.history_diagnostic.test_pipeline import parent_fixture, Tokenizer
from sft.nemotron_motif_trigger.serialization import SerializationError


class Tests(unittest.TestCase):
    def setUp(self):
        self.rows = make_family(family(source())[2])

    def test_single_content_intervention(self):
        a, b = self.rows[1:]
        self.assertEqual(a['messages'][:-1], b['messages'][:-1])
        self.assertEqual(a['tools'], b['tools'])
        self.assertNotEqual(a['messages'][-1]['content'], b['messages'][-1]['content'])

    def test_reject_argument_mutation(self):
        rows = copy.deepcopy(self.rows)
        rows[2]['messages'][2]['tool_calls'][0]['function']['arguments'] = {'item_id': 'different'}
        with self.assertRaises(ValueError):
            audit(rows)

    def test_reject_duplicate(self):
        with self.assertRaises(ValueError):
            audit(self.rows + [self.rows[0]])

    def test_lengths_must_match(self):
        with patch('experiments.m1.qwen.first_trigger.e1.pipeline.check_full_history', return_value=[200, 100, 101]):
            with self.assertRaisesRegex(ValueError, 'unequal'):
                lengths_for(self.rows, None)

    def test_paired_metric_and_missing_prediction(self):
        predictions = predictions_for(self.rows)
        result = metrics(self.rows, predictions)
        self.assertEqual(result['metrics']['delta_exec']['estimate'], 1)
        self.assertEqual(result['paired_counts'], {'success_1_failure_0': 1})
        with self.assertRaises(ValueError):
            metrics(self.rows, predictions[:-1])

    def test_freeze_review_lengths_and_integrity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent_fixture(root / 'parent', 2)
            prepare(SimpleNamespace(parent=root / 'parent', families=2, output=root / 'candidates'))
            args = SimpleNamespace(candidates=root / 'candidates', model=root,
                                   min_families=1, output=root / 'frozen')
            with patch('experiments.m1.qwen.first_trigger.e1.pipeline.tokenizer', return_value=Tokenizer()):
                with self.assertRaisesRegex(ValueError, 'Only 0'):
                    freeze(args)
                review = args.candidates / 'review.jsonl'
                reviews = [json.loads(line) for line in review.read_text().splitlines()]
                for r in reviews:
                    r.update(decision='approve', status_only_semantically_valid=True,
                             notes='TEST ONLY synthetic review')
                review.write_text(''.join(json.dumps(r) + '\n' for r in reviews))
                with patch('experiments.m1.qwen.first_trigger.e1.pipeline.lengths_for',
                           side_effect=[SerializationError('too long'), [200, 100, 100]]):
                    freeze(args)
            manifest, rows = load(args.output)
            self.assertEqual(manifest['families'], 1)
            self.assertEqual(len(rows), 3)
            (args.output / 'validation.jsonl').write_text('')
            with self.assertRaisesRegex(ValueError, 'changed'):
                load(args.output)


if __name__ == '__main__':
    unittest.main()
