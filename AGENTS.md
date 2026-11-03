# Repository instructions

This repository is an standalone release candidate for Lumen Weight-Space Bench.

- Preserve `tools/weight-space` layout and sibling imports.
- Keep vendor scripts and their notice intact unless a change is explicitly reviewed and recorded.
- Use synthetic fixtures for checks. Do not download models, run full-model inference, connect services, or add private data without explicit authorization.
- Do not commit model weights, edit arrays, inference outputs, credentials, private names, machine home paths, or conversation identifiers.
- Keep generated artifacts outside the repository. Use scratch temporary directories and `python -B` for test execution.
- Do not publish, push, add a remote, or change publication licensing without explicit authorization.
- Describe protocol checks accurately. Passing fixtures does not establish trained-model compatibility, general improvement, statistical significance, or security against hostile inputs.
- Record changes to copied source files in `docs/SOURCE_MANIFEST.json` and run the relevant existing tests.

Complete fixture suite from the repository root:

```sh
python -B -m unittest discover -s tools/weight-space -p test_bench.py -v
python -B -m unittest discover -s tools/weight-space -p test_reproduction.py -v
python -B -m unittest discover -s experiments/qwen3_one_matrix -p 'test_*.py' -v
```

The `experiments/qwen3_one_matrix` runner is a portable adaptation of a completed
Qwen3-8B experiment. Its GPU route is explicit and is not invoked by these tests.
Preserve the distinction between the historical negative result and newly run
synthetic checks. Do not copy active follow-up experiments into this release.
