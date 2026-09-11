"""Engineering fixtures only, not scientific experiment outputs."""
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
from .pipeline import audit, make_family, lengths_for, metrics, gate, PROTOCOL, NOTICE, prepare, freeze, load
from experiments.m1.qwen.first_trigger.history_diagnostic.test_pipeline import parent_fixture, Tokenizer


class Tests(unittest.TestCase):
    def setUp(self):
        self.original=family(source())[2]
        self.rows=make_family(self.original)

    def test_shared_notice_and_single_response_field(self):
        a,s,f=self.rows
        self.assertEqual(a['messages'],self.original['messages'])
        self.assertEqual(s['messages'][:-1],f['messages'][:-1])
        self.assertTrue(s['messages'][0]['content'].endswith(NOTICE))
        import json
        x=json.loads(s['messages'][-1]['content']);y=json.loads(f['messages'][-1]['content'])
        self.assertEqual([k for k in x if x[k]!=y[k]],['status'])

    def test_reject_argument_type_mismatch(self):
        p=copy.deepcopy(self.original)
        p['messages'][-2]['tool_calls'][0]['function']['arguments']['item_id']=7
        with self.assertRaisesRegex(ValueError,'schema'):
            make_family(p)

    def test_reject_notice_mutation(self):
        r=copy.deepcopy(self.rows)
        r[2]['messages'][0]['content']+='failure hint'
        with self.assertRaises(ValueError):audit(r)

    def test_equal_full_prompt_lengths_required(self):
        with patch('experiments.m1.qwen.first_trigger.e1_receipt.pipeline.check_full_history',return_value=[99,100,101]):
            with self.assertRaisesRegex(ValueError,'unequal'):lengths_for(self.rows,None)

    def test_gate_independent_of_failure_and_collapse_rejected(self):
        m=metrics(self.rows,predictions_for(self.rows))
        self.assertTrue(gate(m,PROTOCOL)['passed'])
        m['metrics']['failure_ftr']['estimate']=1
        self.assertTrue(gate(m,PROTOCOL)['passed'])
        m['metrics']['success_asr']['estimate']=0
        self.assertFalse(gate(m,PROTOCOL)['passed'])
        with self.assertRaises(ValueError):metrics(self.rows,predictions_for(self.rows)[:-1])

    def test_freeze_requires_review_and_preserves_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);parent_fixture(root/'parent',21)
            prepare(SimpleNamespace(parent=root/'parent',families=60,output=root/'candidates'))
            args=SimpleNamespace(candidates=root/'candidates',model=root,output=root/'frozen')
            with patch('experiments.m1.qwen.first_trigger.e1_receipt.pipeline.tokenizer',return_value=Tokenizer()),patch(
                    'experiments.m1.qwen.first_trigger.e1_receipt.pipeline.lengths_for',return_value=[99,100,100]):
                with self.assertRaisesRegex(ValueError,'Only 0'):freeze(args)
                p=args.candidates/'review.jsonl';reviews=[json.loads(l) for l in p.read_text().splitlines()]
                for r in reviews:r.update(decision='approve',receipt_semantically_valid=True,
                    source_success_credible=True,no_pre_response_outcome_leak=True,reviewer='synthetic test',notes='TEST ONLY')
                p.write_text(''.join(json.dumps(r)+'\n' for r in reviews))
                freeze(args)
            manifest,rows=load(args.output)
            self.assertEqual(manifest['families'],20)
            self.assertEqual(len(rows),60)
            p=args.output/'protocol.json';p.write_text('{}')
            with self.assertRaisesRegex(ValueError,'changed'):load(args.output)


if __name__=='__main__':unittest.main()
