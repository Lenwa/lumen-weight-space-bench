"""Create a tiny random model, two checkpoint candidates and a CPU comparison.

No downloads, trained weights or claimed improvement. Output must be fresh and
outside this repository; all generated artifacts remain in that directory.
"""
import argparse
import json
from pathlib import Path

import checkpoint


def external_output(path):
    path = Path(path).resolve()
    repo = Path(__file__).resolve().parents[2]
    if path == repo or path.is_relative_to(repo):
        raise ValueError('Choose an output directory outside this repository')
    if path.exists():
        raise FileExistsError(path)
    return path


def run_demo(out):
    out = external_output(out)
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from evaluate import run

    out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(42)
    src = out / 'original'
    tokenizer = Tokenizer(WordLevel({'[UNK]': 0, '[EOS]': 1, 'one': 2,
                                     'two': 3, '1': 4, '2': 5}, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token='[UNK]',
                                 eos_token='[EOS]', pad_token='[EOS]')
    tok.chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
    tok.save_pretrained(src)
    model = GPT2LMHeadModel(GPT2Config(vocab_size=6, n_embd=8, n_layer=1,
        n_head=1, n_positions=32, eos_token_id=1, bos_token_id=1)).to(torch.bfloat16)
    model.save_pretrained(src, max_shard_size='1KB')
    # Deliberate, arbitrary perturbation; it was not optimized for these cases.
    name = 'transformer.h.0.mlp.c_proj.weight'
    replacement = model.state_dict()[name].float().numpy().copy()
    replacement[0, 0] += 0.125
    del model
    checkpoint.proposal(src, {}, out / 'noop-edit', 'synthetic-seed-42', 'no-op control')
    checkpoint.build(src, out / 'noop-edit', out / 'noop')
    checkpoint.proposal(src, {name: replacement}, out / 'edited-edit',
                        'synthetic-seed-42', 'arbitrary one-entry demonstration')
    checkpoint.build(src, out / 'edited-edit', out / 'edited')
    cases = [{'id': 'h1', 'prompt': 'one', 'answer': '1', 'split': 'heldout'},
             {'id': 'r1', 'prompt': 'two', 'answer': '2', 'split': 'retention'}]
    checkpoint.write_json(out / 'cases.json', cases)
    result = run(out / 'cases.json', {a: out / a for a in ('original', 'noop', 'edited')},
                 out / 'evaluation', max_tokens=2)
    checkpoint.write_json(out / 'demo.json', {'purpose': 'plumbing demonstration, not a useful edit',
                          'seed': 42, 'comparison': result})
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    print(json.dumps(run_demo(parser.parse_args().out), indent=2))
