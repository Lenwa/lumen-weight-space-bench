import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from safetensors.numpy import save_file, load_file


class Checkpoints(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.src = self.root / 'source'
        self.src.mkdir()
        # Raw BF16 fixture produced with upstream writer, checked with official reader.
        import sys
        sys.path.insert(0, str(Path(__file__).parent / 'vendor'))
        import st_numpy as S
        self.S = S
        S.save(self.src / 'a.safetensors', {'a': ('BF16', np.array([16256, 16384], dtype=np.uint16))})
        S.save(self.src / 'b.safetensors', {'b': ('BF16', np.array([16448, 16512], dtype=np.uint16))})
        (self.src / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'a': 'a.safetensors', 'b': 'b.safetensors'}}))
        (self.src / 'config.json').write_text('{}')
        self.before = {p.name: p.read_bytes() for p in self.src.iterdir()}

    def api(self):
        import checkpoint
        return checkpoint

    def proposal(self, replacements=None):
        edit = self.root / 'edit'
        self.api().proposal(self.src, replacements or {}, edit, 'fixture-v1', 'fixture')
        return edit

    def test_sparse_and_noop(self):
        for edits, expected in [({}, [1., 2.]), ({'a': np.array([1., 5.], dtype=np.float32)}, [1., 5.])]:
            with self.subTest(edits=list(edits)):
                edit = self.root / ('sparse' if edits else 'noop')
                self.api().proposal(self.src, edits, edit, 'fixture-v1', 'fixture')
                out = self.root / (edit.name + '-candidate')
                receipt = self.api().build(self.src, edit, out)
                self.assertEqual(self.S.bf16_to_f32(self.S.load(out / 'a.safetensors')['a'][1]).tolist(), expected)
                self.assertEqual((out / 'b.safetensors').read_bytes(), self.before['b.safetensors'])
                self.assertEqual(receipt['status'], 'verified')
        self.assertEqual({p.name: p.read_bytes() for p in self.src.iterdir()}, self.before)

    def test_wrong_hash_rejected(self):
        edit = self.proposal({'a': np.array([1., 5.], dtype=np.float32)})
        m = json.loads((edit / 'manifest.json').read_text())
        m['tensors'][0]['base_sha256'] = '0' * 64
        (edit / 'manifest.json').write_text(json.dumps(m))
        with self.assertRaises(ValueError): self.api().build(self.src, edit, self.root / 'out')
        self.assertFalse((self.root / 'out').exists())

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError): self.proposal({'a': np.array([np.nan, 2.])})

    def test_existing_and_overlap_rejected(self):
        edit = self.proposal()
        for out in (self.src, self.src / 'nested', self.root):
            with self.subTest(out=out), self.assertRaises(ValueError): self.api().build(self.src, edit, out)

    def test_partial_build_blocks_retry(self):
        edit = self.proposal()
        (self.root / 'out.partial').mkdir()
        with self.assertRaises(ValueError): self.api().build(self.src, edit, self.root / 'out')

    def test_duplicate_indices_rejected(self):
        edit = self.proposal({'a': np.array([1., 5.], dtype=np.float32)})
        np.savez(edit / 'edit.npz', t0_idx=np.array([1, 1]), t0_val=np.array([16544, 16544], dtype=np.uint16))
        with self.assertRaises(ValueError): self.api().build(self.src, edit, self.root / 'out')

    def test_path_escape_rejected(self):
        (self.src / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'a': '../outside.safetensors'}}))
        with self.assertRaises(ValueError): self.api().inspect(self.src)

    def test_bad_result_leaves_only_partial(self):
        edit = self.proposal({'a': np.array([1., 5.], dtype=np.float32)})
        m = json.loads((edit / 'manifest.json').read_text())
        m['tensors'][0]['edited_sha256'] = '0' * 64
        (edit / 'manifest.json').write_text(json.dumps(m))
        with self.assertRaises(ValueError): self.api().build(self.src, edit, self.root / 'out')
        self.assertFalse((self.root / 'out').exists())
        self.assertTrue((self.root / 'out.partial').exists())

    def test_source_tamper_rejected(self):
        edit = self.proposal()
        (self.src / 'config.json').write_text('{"changed":true}')
        with self.assertRaises(ValueError): self.api().build(self.src, edit, self.root / 'out')

    def test_truncated_shard_rejected(self):
        p = self.src / 'a.safetensors'
        p.write_bytes(p.read_bytes()[:-1])
        with self.assertRaises(ValueError): self.api().inspect(self.src)

    def test_candidate_can_be_source(self):
        edit = self.proposal()
        first = self.root / 'first'
        self.api().build(self.src, edit, first)
        second_edit = self.root / 'second-edit'
        self.api().proposal(first, {}, second_edit, 'candidate-v1', 'second noop')
        receipt = self.api().build(first, second_edit, self.root / 'second')
        self.assertEqual(receipt['status'], 'verified')

    def test_lowrank_replay(self):
        base = np.array([[16256, 16384]], dtype=np.uint16)
        self.S.save(self.src / 'a.safetensors', {'a': ('BF16', base)})
        edit = self.proposal({'a': np.array([[2., 3.]], dtype=np.float32)})
        m = json.loads((edit / 'manifest.json').read_text())
        m['tensors'][0]['form'] = 'lowrank'
        (edit / 'manifest.json').write_text(json.dumps(m))
        np.savez(edit / 'edit.npz', t0_idx=np.array([], dtype=np.int64), t0_val=np.array([], dtype=np.uint16),
                 t0_a=np.array([[1.]]), t0_b=np.array([[1., 1.]]))
        self.api().build(self.src, edit, self.root / 'out')
        value = self.S.bf16_to_f32(self.S.load(self.root / 'out/a.safetensors')['a'][1])
        self.assertEqual(value.tolist(), [[2., 3.]])


class Evaluation(unittest.TestCase):
    def cases(self):
        return [{'id': 'x', 'split': 'heldout', 'prompt': 'one', 'answer': '1'},
                {'id': 'y', 'split': 'retention', 'prompt': 'two', 'answer': '2'}]

    def records(self):
        return {arm: [{'id': 'x', 'text': '1', 'finish_reason': 'stop', 'tokens': 2},
                      {'id': 'y', 'text': '2', 'finish_reason': 'stop', 'tokens': 2}]
                for arm in ('original', 'noop', 'edited')}

    def test_scores_and_truncation(self):
        from evaluate import compare
        r = self.records()
        r['edited'][0].update(text='wrong', finish_reason='length')
        result = compare(self.cases(), r)
        self.assertEqual(result['arms']['edited']['heldout']['accuracy'], 0)
        self.assertEqual(result['arms']['edited']['heldout']['truncated'], 1)
        self.assertTrue(result['noop_matches'])
        self.assertFalse(result['improvement_gate'])

    def test_missing_duplicate_empty_rejected(self):
        from evaluate import compare
        for cases, records in [(self.cases(), {}), ([], self.records()),
                               (self.cases() * 2, self.records())]:
            with self.assertRaises(ValueError): compare(cases, records)

    def test_gate_requires_retention_and_noop(self):
        from evaluate import compare
        r = self.records()
        for arm in ('original', 'noop'): r[arm][0]['text'] = 'wrong'
        self.assertTrue(compare(self.cases(), r)['improvement_gate'])
        r['edited'][1]['text'] = 'wrong'
        self.assertFalse(compare(self.cases(), r)['improvement_gate'])
        r['edited'][1]['text'] = '2'
        r['noop'][0]['text'] = 'different'
        self.assertFalse(compare(self.cases(), r)['improvement_gate'])

    def test_tiny_local_inference(self):
        from evaluate import run
        import checkpoint
        from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / 'model'
            tokenizer = Tokenizer(WordLevel({'[UNK]': 0, '[EOS]': 1, 'one': 2, 'two': 3, '1': 4, '2': 5}, unk_token='[UNK]'))
            tokenizer.pre_tokenizer = Whitespace()
            tok = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token='[UNK]', eos_token='[EOS]', pad_token='[EOS]')
            tok.chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
            tok.save_pretrained(src)
            model = GPT2LMHeadModel(GPT2Config(vocab_size=6, n_embd=8, n_layer=1, n_head=1, n_positions=32, eos_token_id=1, bos_token_id=1))
            model.save_pretrained(src, max_shard_size='1KB')
            del model
            checkpoint.proposal(src, {}, root / 'edit', 'tiny-random', 'no-op smoke')
            checkpoint.build(src, root / 'edit', root / 'noop')
            cases = root / 'cases.json'
            cases.write_text(json.dumps(self.cases()))
            report = run(cases, {'original': src, 'noop': root / 'noop', 'edited': root / 'noop'}, root / 'result', max_tokens=2)
            self.assertTrue(report['noop_matches'])
            self.assertFalse(report['improvement_gate'])
            self.assertTrue((root / 'result/protocol.json').exists())


if __name__ == '__main__': unittest.main()
