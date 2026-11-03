# Research context and scope

## Relationship to upstream research

The original researchers call their approach **Weight-Space Programming**.
Their article is [Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights](https://rikiyat.github.io/blog/weight-space-programming/),
by Rikiya Takehi, Jayden Teoh, Alex Zhang and Omar Khattab.

Lumen Weight-Space Bench is an independently developed experimental bench inspired
by that work. It adds sharded checkpoint construction, sparse proposals, streaming
source hashes and matched text-model evaluation around two pinned upstream replay
helpers. It is not the researchers' official repository or a reproduction of their
reported model improvement. Their approach uses direct LLM-directed weight edits
without gradient descent. Our included Qwen3-8B experiment uses a local gradient
update and reports a negative transfer result. These methods must not be conflated.

The upstream repository is `RikiyaT/lm_understand-logs`. Only `apply_edit.py`
and `st_numpy.py` are runtime dependencies redistributed here. Their original
license notice and pinned revision are in NOTICE and SOURCE_MANIFEST.json.
The upstream builder, evaluator, profiles and unpublished editor harness are not
included. No upstream benchmark datasets or model weights are distributed.

## Included experiment

See [QWEN3_EXPERIMENT.md](QWEN3_EXPERIMENT.md) for the exact Qwen3-8B revision,
hybrid precision and completed result. Discovery accuracy changed from 1/4 to
2/4, held-out accuracy from 1/8 to 0/8, and retention from 4/6 to 3/6. No-op and
rollback token sequences matched original. This candidate did not improve transfer.

## Preserved controls

- Keep original checkpoints untouched; construct fresh separate candidates.
- Preserve pinned upstream BF16 arithmetic and original notices.
- Validate tensor/shard identity and hashes before using a candidate.
- Compare arms with matching prompts, precision and runtime settings.
- Separate discovery, held-out performance, retention and completion.
- Preserve failures and report unsuccessful interventions.
- Treat source revision labels as operator metadata; hashes identify bytes.
- Distinguish synthetic software verification from trained-model evidence.

This package has no dependency on private project services or records. The
standalone AGENTS.md and reproduction guide describe the supported workflow.
