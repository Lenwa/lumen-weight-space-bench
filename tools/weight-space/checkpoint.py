"""Local, additive checkpoint construction. Never edits a source checkpoint in place."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).parent / 'vendor'))
import st_numpy as S
from apply_edit import rebuild


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def tensor_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def write_json(path, obj):
    with open(path, 'x', encoding='utf-8') as f:
        json.dump(obj, f, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


def child(root, name):
    p = root / name
    if not isinstance(name, str) or Path(name).name != name or p.resolve().parent != root.resolve():
        raise ValueError('checkpoint paths must be direct children')
    if p.is_symlink(): raise ValueError('symlink input refused')
    return p


def inspect(source):
    source = Path(source).resolve()
    index = source / 'model.safetensors.index.json'
    if index.exists():
        wm = json.loads(index.read_text(encoding='utf-8'))['weight_map']
    else:
        wm = {k: 'model.safetensors' for k in S.read_header(source / 'model.safetensors')[0] if k != '__metadata__'}
    if not wm: raise ValueError('empty checkpoint')
    found = {}
    for shard in sorted(set(wm.values())):
        p = child(source, shard)
        if p.suffix != '.safetensors': raise ValueError('invalid shard suffix')
        try:
            with safe_open(p, framework='np') as validated:
                list(validated.keys())
        except Exception as exc:
            raise ValueError('invalid safetensors container') from exc
        header, _ = S.read_header(p)
        for name, info in header.items():
            if name == '__metadata__': continue
            if name in found or wm.get(name) != shard: raise ValueError('inconsistent tensor index')
            found[name] = {'shard': shard, 'dtype': info['dtype'], 'shape': info['shape']}
    if set(found) != set(wm): raise ValueError('missing indexed tensors')
    return found


def stage_for(out, *inputs):
    out = Path(out).resolve()
    for root in inputs:
        root = Path(root).resolve()
        if out == root or out.is_relative_to(root) or root.is_relative_to(out):
            raise ValueError('output overlaps input')
    stage = out.with_name(out.name + '.partial')
    if out.exists() or stage.exists(): raise ValueError('output or partial already exists')
    stage.mkdir(parents=True)
    return out, stage


def source_hashes(source):
    return {p.name: digest(child(source, p.name)) for p in sorted(source.iterdir())
            if p.is_file() and (p.suffix in ('.safetensors', '.json', '.model', '.txt', '.jinja') or p.name in ('merges.txt', 'vocab.json'))}


def proposal(source, replacements, out, revision, rationale):
    source = Path(source).resolve()
    if not revision or not rationale: raise ValueError('revision and rationale required')
    inventory = inspect(source)
    tensors, arrays = [], {}
    before = source_hashes(source)
    for i, (name, value) in enumerate(sorted(replacements.items())):
        if name not in inventory: raise ValueError('unknown tensor')
        info = inventory[name]
        if info['dtype'] != 'BF16': raise ValueError('only BF16 edits supported')
        value = np.asarray(value, dtype=np.float32)
        if list(value.shape) != info['shape'] or not np.isfinite(value).all(): raise ValueError('invalid replacement')
        base = S.load(source / info['shard'], [name])[name][1]
        edited = S.bf16_bits(value)
        if not np.isfinite(S.bf16_to_f32(edited)).all(): raise ValueError('BF16 overflow')
        idx = np.flatnonzero(base.reshape(-1) != edited.reshape(-1))
        arrays[f't{i}_idx'] = idx
        arrays[f't{i}_val'] = edited.reshape(-1)[idx]
        tensors.append({'name': name, 'shape': info['shape'], 'form': 'sparse',
                        'base_sha256': tensor_hash(base), 'edited_sha256': tensor_hash(edited),
                        'changed_entries': int(len(idx))})
    if source_hashes(source) != before: raise ValueError('source changed during proposal')
    out, stage = stage_for(out, source)
    np.savez(stage / 'edit.npz', **arrays)
    write_json(stage / 'manifest.json', {'schema': 'lumen-weight-edit-1', 'source_revision': revision,
               'rationale': rationale, 'source_files': before, 'tensors': tensors})
    stage.rename(out)
    return out


def validate_edit(base, t, z, i):
    if t['form'] not in ('sparse', 'lowrank'): raise ValueError('unsupported edit form')
    if list(base.shape) != t['shape'] or tensor_hash(base) != t['base_sha256']: raise ValueError('base mismatch')
    idx, vals = z[f't{i}_idx'], z[f't{i}_val']
    if idx.ndim != 1 or vals.ndim != 1 or idx.dtype.kind not in 'iu' or vals.dtype != np.uint16:
        raise ValueError('invalid sparse arrays')
    if len(idx) != len(vals) or len(np.unique(idx)) != len(idx) or (len(idx) and (idx.min() < 0 or idx.max() >= base.size)):
        raise ValueError('invalid indices')
    if t['form'] == 'lowrank':
        a, b = z[f't{i}_a'], z[f't{i}_b']
        if base.ndim != 2 or a.ndim != 2 or b.ndim != 2 or (a.shape[0], b.shape[1]) != base.shape or a.shape[1] != b.shape[0]:
            raise ValueError('invalid lowrank shapes')
        if not np.isfinite(a).all() or not np.isfinite(b).all(): raise ValueError('nonfinite factors')
    result = rebuild(base, t, z, i)
    if not np.isfinite(S.bf16_to_f32(result)).all(): raise ValueError('nonfinite result')
    if tensor_hash(result) != t['edited_sha256']: raise ValueError('result hash mismatch')
    if int(np.count_nonzero(base != result)) != t['changed_entries']: raise ValueError('changed count mismatch')
    return result


def build(source, edit, out):
    source, edit = Path(source).resolve(), Path(edit).resolve()
    inventory = inspect(source)
    recipe_hashes = {name: digest(edit / name) for name in ('manifest.json', 'edit.npz')}
    man = json.loads((edit / 'manifest.json').read_text(encoding='utf-8'))
    before = source_hashes(source)
    if not man.get('source_revision') or man.get('source_files') != before:
        raise ValueError('source identity mismatch; import upstream edits through an explicit local manifest')
    edits = {}
    for i, t in enumerate(man['tensors']):
        name = t['name']
        if name not in inventory or name in edits or inventory[name]['dtype'] != 'BF16': raise ValueError('invalid tensor allowlist')
        edits[name] = (i, t)
    out, stage = stage_for(out, source, edit)
    # Leave a partial directory after any failure; never publish an incomplete candidate.
    with np.load(edit / 'edit.npz', allow_pickle=False) as z:
        for shard in sorted({v['shard'] for v in inventory.values()}):
            names = [n for n in edits if inventory[n]['shard'] == shard]
            if not names:
                shutil.copyfile(source / shard, stage / shard)
                if digest(stage / shard) != before[shard]: raise ValueError('copy mismatch')
                continue
            sd = S.load(source / shard)
            expected = {n: tensor_hash(v[1]) for n, v in sd.items()}
            for n in names:
                i, t = edits[n]
                value = validate_edit(sd[n][1], t, z, i)
                sd[n] = ('BF16', value)
                expected[n] = tensor_hash(value)
            S.save(stage / shard, sd, {'format': 'pt'})
            del sd
            readback = S.load(stage / shard)
            if {n: tensor_hash(v[1]) for n, v in readback.items()} != expected: raise ValueError('tensor readback mismatch')
            del readback
    for name in before:
        if not name.endswith('.safetensors') and name != 'weight-edit-receipt.json': shutil.copyfile(source / name, stage / name)
    if source_hashes(source) != before: raise ValueError('source changed during build')
    for name in before:
        if not name.endswith('.safetensors') and name != 'weight-edit-receipt.json' and digest(stage / name) != before[name]: raise ValueError('metadata mismatch')
    inspect(stage)
    if {name: digest(edit / name) for name in recipe_hashes} != recipe_hashes:
        raise ValueError('edit changed during build')
    receipt = {'status': 'verified', 'source_revision': man['source_revision'], 'source_files': before,
               'candidate_files': source_hashes(stage), 'edit_manifest_sha256': recipe_hashes['manifest.json'],
               'edit_arrays_sha256': recipe_hashes['edit.npz'], 'edited_tensors': sorted(edits)}
    write_json(stage / 'weight-edit-receipt.json', receipt)
    if out.exists(): raise ValueError('output appeared during build')
    stage.rename(out)
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    s = sub.add_parser('inspect'); s.add_argument('source')
    s = sub.add_parser('propose'); s.add_argument('source'); s.add_argument('replacements', help='NPZ: tensor names -> float32 replacements; empty NPZ for no-op')
    s.add_argument('out'); s.add_argument('--revision', required=True); s.add_argument('--rationale', required=True)
    s = sub.add_parser('build'); s.add_argument('source'); s.add_argument('edit'); s.add_argument('out')
    a = p.parse_args()
    if a.command == 'inspect': result = inspect(a.source)
    elif a.command == 'build': result = build(a.source, a.edit, a.out)
    else:
        with np.load(a.replacements, allow_pickle=False) as z:
            result = str(proposal(a.source, z, a.out, a.revision, a.rationale))
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
