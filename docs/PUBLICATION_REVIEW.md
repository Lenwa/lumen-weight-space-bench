# Publication review

Status: PREPARED RELEASE CANDIDATE (`lumen-weight-space-bench`). Not yet published.

- Separate repository/history: weight-space bench, offline demo, completed Qwen3 experiment reproduction code and handoff documents. Active follow-up work excluded.
- 30 CPU fixture/demo checks passed after the reproduction update; CLI help passed; see VALIDATION.md. The initial pass had 16 tests.
- Candidate source/privacy scan completed with no findings from the documented bounded checks.
- Upstream vendor bytes match the retained pinned source-audit hashes. Fresh network re-fetch unavailable.
- Standalone AGENTS.md supplies the handoff constraints; no private project installation is required.
- Proposed license: MIT for Lumen additions, original upstream notices preserved. Review LICENSE_REVIEW.md.
- No checkpoints, edit arrays, private prompts, raw transcripts or other Lumen projects included.

Before publishing, review the exact Git file list, licensing/attribution and dependency setup.
The full portable GPU run and hosted fresh-install CI remain unexecuted. The scan is bounded evidence, not a guarantee that
all possible secrets or scientific/technical issues have been identified.

Private detailed scan/source records remain outside this repository. Model licenses
and future generated datasets/results require separate selection and review.
