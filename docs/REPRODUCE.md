# Reproduction entry points

## Start without a downloaded model

Use Python 3.12 in a separate virtual environment. These commands install
packages; the demos themselves use local synthetic models and no network.
From the repository root on Linux/macOS:

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-evaluation.txt
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -B tools/weight-space/quickstart.py --out ../weight-space-demo
```

PowerShell:

```powershell
py -3.12 -m venv .venv
& .venv/Scripts/python.exe -m pip install -r requirements-evaluation.txt
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
& .venv/Scripts/python.exe -B tools/weight-space/quickstart.py --out ../weight-space-demo
```

Use a fresh output directory outside the repository. The demo creates a tiny
random GPT-2, a byte-identical no-op checkpoint, an arbitrarily perturbed
checkpoint and an actual CPU comparison. It does not train a model or promise
a better answer. Inspect `demo.json`, `evaluation/comparison.json`, the three
evaluation output files and candidate receipts. Output refusal on a second
invocation protects the first run; choose another fresh directory.

Run the tests:

```sh
python -B -m unittest discover -s tools/weight-space -p 'test_*.py' -v
python -B -m unittest discover -s experiments/qwen3_one_matrix -p 'test_*.py' -v
```

On restricted Windows installations, set TEMP and TMP to an existing writable
scratch folder outside the repository. The tests create and remove their own
subdirectories. Access-denied errors during directory creation/rename are
environment failures, not passing checks. The tiny Qwen3 test runs all four
arms and the gradient/build path on CPU; capped tiny outputs are not scored
as completed research answers.

## Run the completed experiment's configuration

Read [the model, result and limitations](QWEN3_EXPERIMENT.md) first. Install
the GPU stack separately; do not replace an environment used by other work:

```sh
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-qwen3.txt
python -B experiments/qwen3_one_matrix/reproduce.py --model /path/to/Qwen3-8B --out /path/to/new-experiment
```

The `--model` directory must already contain the pinned BF16 safetensors,
configuration and tokenizer. Model identity is checked against the retained
hash manifest. The runner does not fetch a model or assert that a directory's
name proves its revision. CUDA/bitsandbytes support depends on the platform
and driver. The exact recorded CUDA version and package versions are listed
in QWEN3_EXPERIMENT.md; the requirements file is a compatibility selection,
not a complete transitive lock or a new clean-install verification.

The default profile requires 10.5 GiB free CUDA memory, 12 GiB available RAM,
and 70 GiB free disk for the original experiment's full candidate copies.
More memory may be needed by another platform. Run on an otherwise available
GPU. This standalone process cannot coordinate another application's model
lease and never stops other processes. It exits after the experiment rather
than keeping a model resident.

Inputs/settings are saved before inference. Every generated reply, completion
status, capture, gradient and checkpoint receipt goes under the new output
directory. `result.json` contains four-arm scores and no-op/rollback checks;
`failure.json` retains a failed attempt and completed rows. Do not delete a
failure to turn a later retry into the original run. The module is restored
on normal completion and exceptions; process termination also discards all
in-memory edits, while source files remain separate.

Explicit methodological variant, suitable only for a model that fits system RAM:

```sh
python -B experiments/qwen3_one_matrix/reproduce.py --model /path/to/local-qwen3 --out /path/to/new-variant --profile cpu-variant --allow-variant
```

For Qwen3-8B, CPU float32 needs much more RAM than the tiny test. Other Qwen3
sizes need their own budget. Arbitrary model families and GGUF are rejected.
Use the preserved fixtures for replication; create new frozen splits before
tuning a variant. The old heldout cases have now been exposed.

## Verification boundary

The repository's CPU tests and offline demo have been exercised in the existing
local environment. CI defines a fresh Python 3.12 installation and CPU run for
future execution; adding that workflow is not evidence that hosted CI has run.
No trained-model inference, GPU verification or new dependency installation
was performed during this handoff update.
