# Installation

`medusa-kernel` requires **Python ≥ 3.13**.

## From PyPI

```bash
pip install medusa-kernel
```

This pulls the core stack (NumPy, SciPy, scikit-learn, statsmodels,
matplotlib, h5py, PyWavelets, pandas, tqdm, bson, dill, **PySide6** and
medusa-style). The non-visual core (`medusa.core` / `medusa.signal` /
`medusa.ml`) still imports without loading Qt; no GPU toolchain or
deep-learning framework is installed by default.

## Optional extras

`PySide6` is a **core dependency** (the Qt viewers under `medusa.widgets.*`
work out of the box), so there is no `widgets` extra.

| Extra | Installs | When to use |
|---|---|---|
| `dev` | `pytest`, `pytest-cov`, `pytest-benchmark`, `ruff`, `mypy` | Local development / running the test suite. |
| `docs` | `mkdocs-material`, `mkdocstrings`, `mkdocs-jupyter`, `mike`, … | Building this documentation site (including the executable Jupyter tutorials) locally. |
| `all` | `dev + docs` | Convenience aggregator. |

```bash
pip install "medusa-kernel[all]"
```

## PyTorch (manual install — *not* an extra)

The deep-learning models in `medusa.ml.torch_models` require **PyTorch**,
but PyTorch is **not shipped as an extra**. Its wheel selection depends on
your CUDA / ROCm / MPS / CPU stack, GPU driver, and on Linux the manylinux
flavour. Pulling it from a generic extra would silently install the wrong
build for many users (e.g. a CPU-only wheel on a CUDA box).

Install it yourself with the official selector at
**<https://pytorch.org/get-started/>**, e.g. for CPU-only:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

`medusa.ml.torch_models` raises a `TorchNotInstalled` error pointing at the
same page if you import a Torch-backed model without having installed
PyTorch first.

## From source

```bash
git clone https://github.com/medusabci/medusa-kernel.git
cd medusa-kernel
pip install -e ".[dev]"
```

Or with [`uv`](https://docs.astral.sh/uv/) (recommended, faster):

```bash
uv sync --extra dev
```

## Verifying the install

```python
import medusa
print(medusa.__version__)
```

