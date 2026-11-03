# Reproduction handoff

## Scope

This snapshot includes the original bench modules and vendor helpers, an offline
quickstart, and a portable version of the completed Qwen3-8B final-matrix experiment.
The original bench implementation files remain byte-identical to the source
snapshot. The experiment runner is an adaptation, not a byte-identical copy;
its reusable arithmetic/fixtures and attention adapter have individual provenance
records. The original README was replaced. Private logs and the active follow-up
experiment were not copied.

Start with [REPRODUCE.md](REPRODUCE.md) for runnable commands and
[QWEN3_EXPERIMENT.md](QWEN3_EXPERIMENT.md) for the exact model and completed result.

See `SOURCE_MANIFEST.json` for per-file hashes and transformations. Licensing decisions are separate from technical packaging; consult the licensing review documents before distribution.

## Validation

Run from the standalone repository root:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -B -m unittest discover -s tools/weight-space -p 'test_*.py' -v
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -B -m unittest discover -s experiments/qwen3_one_matrix -p 'test_*.py' -v
python -B tools/weight-space/checkpoint.py --help
python -B tools/weight-space/evaluate.py --help
```

On PowerShell, set the two offline flags with `$env:HF_HUB_OFFLINE = '1'` and `$env:TRANSFORMERS_OFFLINE = '1'` before invoking Python. Use a scratch TEMP/TMP directory. Installed version observations are in `DEPENDENCIES.md`; this repository supplies no locked environment.

The tests cover sparse/no-op proposals, source preservation, hashes, overlapping/existing outputs, partial-output behavior, duplicate indices, path escape, source tampering, truncated containers, candidate reuse, low-rank replay, scoring, gate controls, and tiny local inference. The random GPT-2 fixture has six vocabulary entries, one layer, and embedding width eight; it is created in a temporary directory and never downloaded.

The packaging run passed all 16 existing tests. See `VALIDATION.md` for the run summary and environmental retry. The historical synthetic audit script was intentionally omitted to keep the package scoped to the bench sources. Its five scenarios are represented in the current suite:

| Historical audit scenario | Current test |
|---|---|
| no-op | `Checkpoints.test_sparse_and_noop` |
| sparse replacement | `Checkpoints.test_sparse_and_noop` |
| low-rank replay | `Checkpoints.test_lowrank_replay` |
| wrong base hash | `Checkpoints.test_wrong_hash_rejected` |
| wrong result hash | `Checkpoints.test_bad_result_leaves_only_partial` |

The earlier audit invokes the vendor CLI; the current tests exercise the supported validated wrapper. This is scenario coverage rather than a claim that the two audits are identical. The inspected upstream checkpoint builder, evaluator, profile files, and unpublished editor harness were not reused and are not included dependencies.

## Known limits and review work

- No full trained checkpoint or GPU port was run in this packaging pass. The historical Qwen3-8B result is reported separately; it was a downgrade, not a useful intervention.
- Exact string grading and a small retention set cannot establish broad retention or semantic correctness. Length-limited outputs are graded and counted as truncated.
- Hash checks provide file identity and replay evidence, not trusted authorship or comprehensive protection against concurrent malicious filesystem changes. The code has not received a security audit and should process trusted local files.
- Selected metadata extensions are copied, rather than every possible model runtime file. Compatibility with arbitrary model families, quantized formats, architectures, and custom tokenizers is unproven.
- Changed-shard replay/readback can allocate several shard-sized buffers. Operators must budget RAM and disk before invoking real checkpoint operations.
- The vendor CLI is retained for provenance; the wrapper's validation is the supported bench path.
- Dependency licensing and target model/data licensing remain the downstream operator's responsibility. Vendor provenance here comes from the local snapshot's declaration, without new remote verification.

## Publication boundaries

No remote configuration, push, publication, model download, trained-model inference, or credentials are part of this snapshot. Ignore rules are a convenience and do not replace reviewing the actual staged file list before any later publication.
