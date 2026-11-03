# Lumen Weight-Space Bench

An experimental local tool for constructing separate checkpoint candidates and comparing original, no-op, and edited text outputs. It records file hashes and validates replayed tensors. It does not discover useful weight edits or establish model improvement.

The implementation has been exercised with synthetic BF16 tensors and tiny locally initialized models. A completed **Qwen/Qwen3-8B** experiment used this bench: its one weight edit worsened transfer, and rollback reproduced the original replies. The [completed result and exact model settings](docs/QWEN3_EXPERIMENT.md) distinguish that historical experiment from the CPU verification of this portable package. No model weights or private runtime outputs are included.

**Start here:** [offline demo and reproduction commands](docs/REPRODUCE.md). The demo creates its own tiny model; you do not need a downloaded checkpoint. The optional Qwen3 experiment runner is separate and must be explicitly invoked.

## Relationship to the original research

Inspired by [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/). This is an independent bench, not the authors' official implementation. The included Qwen3-8B experiment uses a gradient update and is distinct from their gradient-free method. See [research context](docs/RESEARCH_CONTEXT.md) for attribution and scope.

## Contents

- `tools/weight-space/checkpoint.py`: inspect local safetensors checkpoints, create sparse replacement proposals, and build separate candidates.
- `tools/weight-space/evaluate.py`: explicitly load three local Transformers checkpoints sequentially and compare greedy text outputs.
- `tools/weight-space/test_bench.py`: 16 fixture tests, including tiny local inference without a download.
- `tools/weight-space/vendor/`: two upstream replay helpers and their original license notice.
- `examples/cases.synthetic.json`: a schema example, not a benchmark.
- `tools/weight-space/quickstart.py`: complete offline, CPU checkpoint-edit/evaluation demonstration.
- `experiments/qwen3_one_matrix/`: portable completed-experiment method, synthetic cases, model identity and CPU tests. The active follow-up experiment is excluded.
- [Handoff](docs/HANDOFF.md), [dependencies](docs/DEPENDENCIES.md), and [source manifest](docs/SOURCE_MANIFEST.json).

## Run locally

Use Python 3.12 for the documented reproduction route. Transformers is pinned to the version used by the experiment; other dependencies are not a complete lock. See the dependency notes before selecting an environment.

```sh
python -m pip install -r requirements-evaluation.txt
python -B tools/weight-space/quickstart.py --out ../weight-space-demo
python -B -m unittest discover -s tools/weight-space -p 'test_*.py' -v
python -B -m unittest discover -s experiments/qwen3_one_matrix -p 'test_*.py' -v
python -B tools/weight-space/checkpoint.py --help
python -B tools/weight-space/evaluate.py --help
```

The test suite constructs its own tiny model and tokenizer locally. Existing weights are unnecessary. Set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` when running in an offline environment. Evaluation loads only local files and sets `trust_remote_code=False`.

## Checkpoint workflow

`SOURCE` must be a local Transformers directory containing `model.safetensors` or `model.safetensors.index.json`. GGUF is unsupported. Replacements are an NPZ archive whose keys are tensor names and values are full replacement arrays; proposal generation rounds to BF16 and stores changed indices and raw BF16 values. Empty replacements produce a no-op proposal:

```sh
python -c "import numpy as np; np.savez('noop-replacements.npz')"
python -B tools/weight-space/checkpoint.py inspect SOURCE
python -B tools/weight-space/checkpoint.py propose SOURCE noop-replacements.npz NOOP_EDIT --revision LOCAL_REVISION --rationale "no-op control"
python -B tools/weight-space/checkpoint.py build SOURCE NOOP_EDIT NOOP
```

For a real edit, independently prepare and review replacements, then use the same proposal/build commands with new destinations. Revision labels are operator assertions; hashes identify the actual local files. No edit discovery or upstream-manifest importer is provided.

Outputs must be new and outside their inputs. Builds stage in `CANDIDATE.partial` and rename after validation. Failures leave partial output and block reuse of that destination. Inspect it before choosing a new destination. Source files are not intentionally edited in place.

The manifest allowlist and validation check tensor identities, supported forms, indices, shapes, finite replacements, source hashes, and replay hashes. The wrapper accepts validated upstream low-rank recipes with local source identity fields, but `propose` generates sparse recipes only. Call the wrapper instead of the vendor CLI, whose checks are less extensive.

Untouched shards are copied. Changed shards are rebuilt and reread; their other tensors are checked by content hash. Selected JSON, model, text, and Jinja runtime files are copied. This is not a general directory copier; custom Python code and other extensions are excluded. Memory can require several shard-sized buffers, and candidates require additional disk space.

## Evaluation contract

Use unique case IDs and unique prompts, with both `heldout` and `retention` splits. Keep discovery/training material separate; the evaluator cannot detect external data leakage.

```sh
python -B tools/weight-space/evaluate.py examples/cases.synthetic.json --original SOURCE --noop NOOP --edited CANDIDATE --out RESULTS
```

This command performs actual inference. Select suitable local checkpoints and a resource budget first. Defaults are CPU, float32, seed 42, 64 new tokens, greedy decoding, eager attention, and a request to disable thinking in the tokenizer template. Template/model behavior can vary. GPU and lower precision require explicit CLI choices.

The evaluator checks no-op runtime-file hashes against the original, edited metadata against the original, and identical rendered prompts across arms. Reports include raw answers, token IDs, finish reasons, settings, library versions, and file hashes. Those generated outputs may contain private data and are excluded from version control by default.

Grading uses exact stripped-text equality. The gate requires matching original/no-op text, token counts and finish reasons, more correct held-out examples, and preservation of every previously correct retention example. Truncation is reported separately; a `length` finish is still graded. Unknown finish reasons abort comparison. The gate is a protocol check, not significance testing, semantic equivalence, broad retention, or evidence of intelligence gains. CUDA reproducibility across hardware is not guaranteed.

## Provenance and review status

The two vendor scripts and their license were copied without modification from the local source snapshot, which identifies upstream as `RikiyaT/lm_understand-logs` revision `4d09ba6d6a7e891dd8063974dfe81b3d78e3494f`. No upstream network verification was performed during packaging. See the source manifest and licensing review documents for exact scope and publication decisions.

This is a standalone release candidate for `lumen-weight-space-bench`. Read [the licensing review](docs/LICENSE_REVIEW.md) and [NOTICE](NOTICE). Dependencies, model licenses, provenance, and licensing status need review before distribution or scientific claims.

## Release status

Start with [publication review](docs/PUBLICATION_REVIEW.md), [research context](docs/RESEARCH_CONTEXT.md), [license decision](docs/LICENSE_REVIEW.md) and [agent instructions](AGENTS.md). Public release authorized on 2 November 2026.
