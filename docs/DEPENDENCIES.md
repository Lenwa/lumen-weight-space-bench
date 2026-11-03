# Dependency notes

`requirements.txt` lists the direct checkpoint dependencies. `requirements-evaluation.txt` adds direct dependencies for evaluation and the full fixture suite. These lists are not a lock and do not assert compatibility with every available version. Model tokenizers may require additional packages; select them according to the local model's requirements.

The documented route uses Python 3.12. `requirements-evaluation.txt` now pins
Transformers 4.57.6 and includes psutil for the portable experiment's passive
resource checks. Other dependencies remain unpinned. `requirements-qwen3.txt`
adds the historical bitsandbytes version and Accelerate; choose torch/CUDA
separately as described in REPRODUCE.md. The packaging environment reported:

| Component | Observed version |
|---|---|
| Python | 3.12.14 |
| NumPy | 2.5.4 |
| safetensors | 0.8.0 |
| torch | 2.8.0+cu128 |
| transformers | 4.57.6 |
| tokenizers | 0.22.2 |
| huggingface-hub | 0.36.2 |
| Jinja2 | 3.1.6 |

These are local metadata observations, not a fabricated freeze, minimum-version specification, or statement that the versions are publicly available from a package index. No package installation or dependency download was performed during packaging. Tests use CPU; the installed torch CUDA suffix does not establish GPU validation.

The vendor helpers import NumPy. `checkpoint.py` uses the official safetensors reader for container validation in addition to the vendor NumPy reader/writer. `evaluate.py` imports torch and Transformers only when `run` is invoked. The inference fixture also uses tokenizers directly and a Jinja chat template. Transitive packages are managed by the chosen environment and are not exhaustively pinned here.
