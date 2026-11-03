import json
from pathlib import Path
import tempfile
import unittest


class PortableTests(unittest.TestCase):
    def test_unpinned_template_changes_historical_identity(self):
        from reproduce import matches_identity
        pinned={'config.json':'a','tokenizer_config.json':'b'}
        self.assertTrue(matches_identity(pinned,pinned))
        self.assertFalse(matches_identity(dict(pinned, **{'chat_template.jinja':'c'}),pinned))
        self.assertFalse(matches_identity(pinned,pinned,has_template_directory=True))

    def test_disk_budget_checks_output_volume_only(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        from reproduce import check_output_disk
        with patch('reproduce.shutil.disk_usage',return_value=SimpleNamespace(free=80*2**30)) as disk:
            check_output_disk(Path('destination'))
            disk.assert_called_once_with(Path('destination'))
        with patch('reproduce.shutil.disk_usage',return_value=SimpleNamespace(free=1)):
            with self.assertRaises(MemoryError):check_output_disk(Path('destination'))

    def test_cpu_tiny_qwen3_four_arm_run(self):
        import torch
        from transformers import Qwen3Config, Qwen3ForCausalLM, PreTrainedTokenizerFast
        from tokenizers import Tokenizer, models, pre_tokenizers, decoders
        from reproduce import run
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source'
            alphabet = list(dict.fromkeys(chr(i) for i in range(32, 127)))
            vocab = {'[UNK]': 0, '[EOS]': 1, **{c: i+2 for i,c in enumerate(alphabet)}}
            tokenizer = Tokenizer(models.WordLevel(vocab, unk_token='[UNK]'))
            tokenizer.pre_tokenizer = pre_tokenizers.Split('', behavior='isolated')
            tokenizer.decoder = decoders.Fuse()
            tok = PreTrainedTokenizerFast(tokenizer_object=tokenizer,
                unk_token='[UNK]', eos_token='[EOS]', pad_token='[EOS]')
            tok.chat_template = "{% for message in messages %}{{ message['content'] }} {% endfor %}"
            tok.save_pretrained(source)
            torch.manual_seed(42)
            model = Qwen3ForCausalLM(Qwen3Config(vocab_size=len(vocab), hidden_size=8,
                intermediate_size=12, num_hidden_layers=1, num_attention_heads=2,
                num_key_value_heads=1, head_dim=4, max_position_embeddings=1024,
                eos_token_id=1, tie_word_embeddings=False)).to(torch.bfloat16)
            model.save_pretrained(source)
            del model
            result = run(source, root/'run', profile='cpu-variant', allow_variant=True,
                         max_tokens=2, relative=.001)
            self.assertEqual(len(result['rows']), 72)
            self.assertTrue((root/'run/rollback.json').is_file())
            self.assertFalse(result['historical_configuration'])
            self.assertTrue(result['noop_token_ids_match'])
            self.assertTrue(result['rollback_token_ids_match'])

    def test_variant_needs_explicit_opt_in(self):
        from reproduce import validate_configuration
        with self.assertRaises(ValueError):
            validate_configuration({'model_type':'qwen3','num_hidden_layers':1},
                                   'cpu-variant', False, 2, .001)


if __name__ == '__main__':
    unittest.main()
