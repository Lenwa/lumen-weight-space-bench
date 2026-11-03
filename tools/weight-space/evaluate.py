"""Matched local text evaluation. Explicit CLI invocation loads models sequentially."""
import argparse
import gc
import json
from pathlib import Path

from checkpoint import source_hashes, write_json, stage_for, digest

ARMS = ('original', 'noop', 'edited')


def validate_cases(cases):
    if not cases or len({c['id'] for c in cases}) != len(cases): raise ValueError('empty or duplicate cases')
    if {c['split'] for c in cases} != {'heldout', 'retention'}: raise ValueError('heldout and retention groups required')
    if len({c['prompt'] for c in cases}) != len(cases): raise ValueError('duplicate prompts across evaluation cases')
    for c in cases:
        if not all(isinstance(c[k], str) and c[k] for k in ('id', 'prompt', 'answer')): raise ValueError('invalid case')


def compare(cases, records):
    validate_cases(cases)
    if set(records) != set(ARMS): raise ValueError('three arms required')
    expected = {c['id'] for c in cases}
    indexed, scores = {}, {}
    for arm in ARMS:
        rows = records[arm]
        if len(rows) != len(expected) or {r['id'] for r in rows} != expected: raise ValueError('missing or duplicate outputs')
        indexed[arm] = {r['id']: r for r in rows}
        scores[arm] = {}
        for split in ('heldout', 'retention'):
            group = [c for c in cases if c['split'] == split]
            correct = truncated = 0
            for c in group:
                r = indexed[arm][c['id']]
                if r['finish_reason'] not in ('stop', 'length') or not isinstance(r['text'], str): raise ValueError('unfinished generation')
                if type(r['tokens']) is not int or r['tokens'] < 0: raise ValueError('invalid token count')
                correct += r['text'].strip() == c['answer'].strip()
                truncated += r['finish_reason'] == 'length'
            scores[arm][split] = {'correct': correct, 'count': len(group), 'accuracy': correct / len(group), 'truncated': truncated}
    noop = all(all(indexed['original'][i].get(k) == indexed['noop'][i].get(k)
                   for k in ('text', 'finish_reason', 'tokens')) for i in expected)
    # Conservative gate: do not trade one retained example for another.
    retention = all(indexed['original'][c['id']]['text'].strip() != c['answer'].strip()
                    or indexed['edited'][c['id']]['text'].strip() == c['answer'].strip()
                    for c in cases if c['split'] == 'retention')
    gate = noop and retention and scores['edited']['heldout']['correct'] > scores['original']['heldout']['correct']
    return {'grading': 'exact stripped text; no semantic or math equivalence claim', 'arms': scores,
            'noop_matches': noop, 'retention_preserved': retention, 'improvement_gate': gate}


def run(cases_path, checkpoints, out, max_tokens=64, seed=42, device='cpu', dtype='float32'):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cases_path = Path(cases_path).resolve()
    case_hash = digest(cases_path)
    cases = json.loads(cases_path.read_text(encoding='utf-8'))
    validate_cases(cases)
    if set(checkpoints) != set(ARMS) or max_tokens <= 0: raise ValueError('invalid run configuration')
    checkpoints = {a: Path(p).resolve() for a, p in checkpoints.items()}
    identities = {a: source_hashes(p) for a, p in checkpoints.items()}
    if not all(identities.values()): raise ValueError('missing checkpoint files')
    # Verify original and no-op weights/config/tokenizer match, independent of receipts.
    if {k: v for k, v in identities['original'].items() if k != 'weight-edit-receipt.json'} != {k: v for k, v in identities['noop'].items() if k != 'weight-edit-receipt.json'}:
        raise ValueError('no-op checkpoint is not byte-identical to original')
    for name, h in identities['original'].items():
        if not name.endswith('.safetensors') and name != 'weight-edit-receipt.json' and identities['edited'].get(name) != h:
            raise ValueError('candidate configuration/tokenizer differs')
    out, stage = stage_for(out, *checkpoints.values(), cases_path)
    protocol = {'backend': 'transformers', 'dtype': dtype, 'device': device, 'seed': seed,
                'max_new_tokens': max_tokens, 'do_sample': False, 'thinking': False,
                'cases_sha256': case_hash, 'checkpoints': identities,
                'torch': torch.__version__}
    import transformers
    protocol['transformers'] = transformers.__version__
    write_json(stage / 'protocol.json', protocol)
    records = {}
    prompts_reference = None
    for arm in ARMS:
        torch.manual_seed(seed)
        tok = AutoTokenizer.from_pretrained(checkpoints[arm], local_files_only=True, trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(checkpoints[arm], local_files_only=True,
                    trust_remote_code=False, dtype=getattr(torch, dtype), attn_implementation='eager').to(device).eval()
        prompts = [tok.apply_chat_template([{'role': 'user', 'content': c['prompt']}], tokenize=False,
                   add_generation_prompt=True, enable_thinking=False) for c in cases]
        if prompts_reference is not None and prompts != prompts_reference: raise ValueError('prompt rendering differs')
        prompts_reference = prompts
        rows = []
        for c, prompt in zip(cases, prompts):
            x = tok(prompt, return_tensors='pt', add_special_tokens=False).to(device)
            with torch.inference_mode():
                ids = model.generate(**x, max_new_tokens=max_tokens, do_sample=False,
                                     pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)[0, x.input_ids.shape[1]:].tolist()
            eos = model.generation_config.eos_token_id
            eos = eos if isinstance(eos, list) else [eos]
            reason = 'stop' if ids and ids[-1] in eos else 'length' if len(ids) == max_tokens else 'unknown'
            row = {'id': c['id'], 'text': tok.decode(ids, skip_special_tokens=True), 'token_ids': ids,
                   'tokens': len(ids), 'finish_reason': reason, 'rendered_prompt': prompt}
            rows.append(row)
            with open(stage / (arm + '.jsonl'), 'a', encoding='utf-8') as f: f.write(json.dumps(row) + '\n')
        records[arm] = rows
        del model
        gc.collect()
        if device.startswith('cuda'): torch.cuda.empty_cache()
        if source_hashes(checkpoints[arm]) != identities[arm]: raise ValueError('checkpoint changed during inference')
    report = compare(cases, records)
    if digest(cases_path) != case_hash: raise ValueError('cases changed during evaluation')
    write_json(stage / 'comparison.json', report)
    stage.rename(out)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('cases', help='JSON list: id, prompt, answer, split (heldout or retention)')
    for arm in ARMS: p.add_argument('--' + arm, required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--max-tokens', type=int, default=64)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    p.add_argument('--dtype', choices=('float32', 'float16', 'bfloat16'), default='float32')
    a = p.parse_args()
    print(json.dumps(run(a.cases, {arm: getattr(a, arm) for arm in ARMS}, a.out,
                         a.max_tokens, a.seed, a.device, a.dtype), indent=2))


if __name__ == '__main__': main()
