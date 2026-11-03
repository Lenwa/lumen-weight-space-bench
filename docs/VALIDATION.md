# Local packaging validation

## Reproducible handoff update

2 November 2026: the existing environment passed **17 bench/demo checks**
(19.563 seconds) and **13 experiment checks** (8.995 seconds), 30 total.
The latter include a full four-arm run on a newly initialized tiny Qwen3 CPU
model, actual gradient/candidate construction, rollback, explicit variant
opt-in, rejection of unpinned templates as historical inputs, and output-volume
disk-budget validation. The new quickstart also passed independently.

The historical 18-case fixture list equals the completed run's frozen protocol
exactly. The archived audit reports complete=true, raw_data_audit_pass=true,
no errors and no protected-file drift. Its aggregate is preserved in
`experiments/qwen3_one_matrix/completed_result.json`, with source-record hashes.
This is historical evidence, not a new GPU replication.

An initial test attempt encountered restricted Windows temporary-directory
permissions. A later sandbox run reached candidate publication but failed its
directory rename. Host filesystem execution of the same CPU tests with an
explicit external scratch directory passed; no code workaround weakened atomic
publication. Original failure receipts remain outside the public repository.

Fresh static review identified unpinned template overrides and a source-volume
disk-space check. Both were fixed with failing-then-passing regression checks.
CLI help and whitespace checks passed. Bounded new-file checks found no machine
home paths or the selected credential/contact patterns. These checks are not
a comprehensive security audit.

No full-model/GPU run, fresh dependency installation, hosted CI execution,
model download or active-worker interaction occurred during this update.
CI and installation commands are provided for independent verification.

## Original packaging pass

Date: 2026-11-02. Working directory: standalone review repository root. The existing local Python environment was reused; no packages or models were downloaded.

Command:

```sh
python -B -m unittest discover -s tools/weight-space -p test_bench.py -v
```

Environment: `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `PYTHONDONTWRITEBYTECODE=1`, and TEMP/TMP set to a writable scratch directory outside the review repository. Installed versions are recorded separately in `DEPENDENCIES.md`.

Result: **16 tests passed**, exit code **0**, unittest reported **15.237 seconds**. This includes the existing tiny random GPT-2 fixture, generated and removed locally. No trained-model inference was run. The unchanged implementation and vendor files were compared by SHA-256 with the original local source; all six matched. The source README is intentionally replaced and both hashes are recorded in `SOURCE_MANIFEST.json`.

An initial attempt used a virtualized system temporary path and failed to create/clean nested fixture directories with Windows access-denied errors. A further path probe also found the alternate scratch root unavailable. The final run used a writable project scratch directory outside this repository and passed without implementation changes. These environmental failures are not counted as successful test runs.

This result supports fixture compatibility only. It does not validate arbitrary model families, GPU execution, trained-model behavior, useful edits, or scientific claims.

## Release preparation

A clean release copy excludes internal planning notes and model aliases. The
malformed-output fixture uses a generic suffix; its grading condition is unchanged.
Original implementation and upstream vendor bytes remain unchanged.

Release-copy verification: 17 bench/demo tests passed in 14.589 seconds and
13 experiment tests passed in 10.281 seconds on 2 November 2026, using the
existing local environment and offline flags. No full-model inference was run.
