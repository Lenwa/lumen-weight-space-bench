# Completed Qwen3-8B one-matrix experiment

## Model and result

The completed experiment used **Qwen/Qwen3-8B**, Hugging Face revision
`b968826d9c46dd6066d109eabc6255188de91218`.
This is a different model and experiment from the upstream article's
Qwen3.5-0.8B intervention. The experiment here uses a gradient update; it
is not a replication of the upstream gradient-free Weight-Space Programming method.

One preselected gradient update changed behavior but worsened transfer:

| Synthetic cases | Original | No-op | Edited | Rollback |
| --- | ---: | ---: | ---: | ---: |
| Discovery, used for gradient | 1/4 | 1/4 | 2/4 | 1/4 |
| Held-out, excluded from gradient | 1/8 | 1/8 | 0/8 | 1/8 |
| Retention controls | 4/6 | 4/6 | 3/6 | 4/6 |

All 72 replies ended with EOS; the longest was 15 tokens. No-op and rollback
matched the original token IDs on all 18 cases. Two discovery errors were fixed,
one discovery success broke, and previously correct held-out and retention
answers each regressed. The improvement gate was false. This is a negative
result for one candidate on a small synthetic task, not evidence against all
weight editing. It does show a completed use of the checkpoint bench on a
trained model, distinct from the repository's synthetic software tests.

This summary comes from the completed 2 November 2026 experiment's report,
protocol, comparison and independent local audit. The audit recomputed the
gradient/candidate and checked source/protected-file hashes with no reported
errors. Raw private runtime logs, model files and tensor payloads are not part
of this repository. A packaged summary is not an independent replication.
The later ongoing experiment is outside this release. The machine-readable
[aggregate](../experiments/qwen3_one_matrix/completed_result.json) includes
source-record hashes and the four-arm counts, without raw private logs.

## Exact experimental settings

- Architecture: Qwen3ForCausalLM, 36 layers, hidden size 4096, MLP size 12288,
  32 query heads, 8 KV heads, head dimension 128, vocabulary 151936.
- Source: original BF16 safetensors checkpoint, five shards. Expected hashes
  and configuration/tokenizer identity are in
  [model_identity.json](../experiments/qwen3_one_matrix/model_identity.json).
- Target: `model.layers.35.mlp.down_proj.weight`, shape `[4096, 12288]`.
- Runtime: bitsandbytes INT8 backbone; the target was replaced with an FP16
  linear module in **all four arms**. INT8 outlier thresholds were set to zero
  during the comparison. This differs from the ordinary Q8_0 GGUF/llama.cpp
  workshop and from prior all-INT8 tests.
- Attention: Transformers SDPA with the included `repeat_kv_efficient_v1`
  adapter, forcing PyTorch EFFICIENT_ATTENTION and explicitly repeating KV
  heads. No silent fallback. Dynamic KV cache on GPU.
- Hardware: NVIDIA GeForce RTX 3060, 12 GiB; 32 GiB system RAM.
- Observed libraries: torch 2.8.0+cu128, Transformers 4.57.6,
  bitsandbytes 0.48.2. See [environment setup](REPRODUCE.md).
- Generation: thinking disabled by chat template, greedy decoding,
  seed 42, repetition penalty 1, 128 new-token cap, independent contexts.
  No profile, conversation history, notebook or tool access.
- Objective: mean cross entropy on eight correct number tokens from four
  discovery records, using teacher-forced JSON prefixes. Capture inputs to
  the final down projection and residuals after the final block. Compute a
  CPU float32 local gradient through final RMSNorm and output head. This
  omits earlier-layer backpropagation and FP16 rounding derivatives.
- Update: one normalized negative-gradient step, relative Frobenius norm
  0.001 before BF16 rounding. Actual post-rounding norm ratio
  0.0009977883892133832; 2,068,509 of 50,331,648 entries changed.
  CPU proxy/native top predictions matched at all eight supervised positions;
  maximum logit difference was 0.0157928466796875.
- Grading: a complete JSON object with exactly one integer `count` field.
  An unevaluated arithmetic expression fails the requested format even if
  mathematically correct. Source fixtures include independently computable
  arithmetic ledgers. Split labels are historical; now-public cases are no
  longer an unseen benchmark for future tuned candidates.

## Reproduce or adapt

The [portable runner](../experiments/qwen3_one_matrix/reproduce.py) retains
the original fixtures and gradient/rollback arithmetic, while replacing
private service wiring with a one-shot process and explicit input/output
paths. It constructs full separate no-op and edited checkpoints through the
bench; only the selected verified tensor is installed into the shared
backbone for comparison. It restores the original module and exits.

The port has CPU fixture coverage. **The full Qwen3-8B GPU port has not been
rerun in this packaging task.** It is a reproduction route to validate, not
a promise of identical numerical outputs on every GPU or library build.
The original Windows supervisor and keep-resident behavior are deliberately
absent. No service, active workshop, or background worker is contacted.

Prepare the exact model files separately under their own license. The
runner uses only local files, disables remote model code, and never downloads
weights. See REPRODUCE.md for commands and resource requirements.

To adapt within the Qwen3 family, use `--allow-variant`; altered file identity,
architecture, precision profile, output cap or step produces a labeled variant.
The target follows the last layer. The gradient is specifically a **final-layer**
local approximation: selecting an earlier layer requires a different gradient
derivation, not just replacing the tensor name. Other architectures require
reviewing normalization, residual/hook semantics, head structure, tokenizer
offsets and weight tying. Do not rename another model to bypass these checks.
GGUF is not an input to this safetensors pipeline.

Use newly frozen discovery/heldout/retention sets for a new claim. Preserve
the no-op and rollback controls, and report precision/backend changes as
experimental changes. A CPU float32 run is not the historical INT8/FP16 run.
