# Contributing

`medusa-kernel` is licensed under **Apache 2.0**. Contributions are
accepted under the inbound = outbound rule of the Apache License §5.

This page mirrors the developer-facing rules from `AGENTS.md §5` in the
repository.

## Setting up a dev environment

```bash
git clone https://github.com/medusabci/medusa-kernel.git
cd medusa-kernel
uv sync --extra dev   # or: pip install -e ".[dev]"
pytest tests/
```

## Previewing the docs locally

```bash
uv sync --extra docs
uv run mkdocs serve            # http://127.0.0.1:8000 with live-reload
# or, for a one-off static build:
uv run mkdocs build
```

The site uses `use_directory_urls: true` (clean `…/page/` URLs on GitHub
Pages), so browse a local build over HTTP rather than `file://`:
`python -m http.server -d site`.

## Code conventions (summary)

- **Python ≥ 3.13** (PEP 649 lazy annotations are the default — no
  `from __future__ import annotations`).
- **Free functions, not container methods.** Processing / metric /
  transform code takes NumPy arrays + explicit parameters and is
  independent of `EEG`, `ECG`, … (see *Functional architecture* in
  `AGENTS.md`).
- **Type annotations** on every public function (PEP 484). Use
  `numpy.typing.NDArray` *without* a forced dtype unless the algorithm
  truly requires it; sanitise inputs with `np.asarray(signal)` to
  preserve the caller's dtype.
- **Standardised argument names** (full table in `TODO.md` K9.C):
  `signal`, `fs: float`, `n_channels`, `band: tuple[float, float]`,
  `segment` / `overlap` as fractions in `[0, 1]`, `power_type` (never
  `type`), …
- **Canonical signal shapes**: time `(n_segments, n_samples, n_channels)`,
  PSD `(n_segments, n_frequencies, n_channels)`, TFR
  `(n_segments, n_frequencies, n_times, n_channels)` — channels always last;
  promote/validate inputs with `check_data_dims`.
- **`np.asarray` over `np.array`** for signal inputs.
- **Paths via `pathlib`** — Kernel must work on Linux and macOS.
- **Absolute imports** from `medusa.<module>`.
- **NumPy-style docstrings** (full template in `TODO.md` K9.A). Document
  shape on the first line of each array parameter; do *not* repeat types
  in the `Parameters` block — the signature is the source of truth.

## Tests and CI

- Tests live under `tests/` (mirrors the `src/medusa/` layout). Pytest is
  run with `--import-mode=importlib`.
- CI runs the matrix Linux / Windows / macOS × Python 3.13 + a headless
  import job (PySide6 is a core dep, so it is installed; the job asserts
  `torch` is absent and that the non-visual core imports without loading Qt)
  + this docs build.
- `mkdocs build --strict` is a CI gate: a broken docstring or a missing
  type annotation fails the build.

## Public API & deprecation

Public symbols are listed in each module's `__all__`; symbols not listed
(including `_`-prefixed names) are internal — rename / delete freely. The
deprecation policy is: `DeprecationWarning` for one minor version, removal in
the next (emitted via the `medusa.core._deprecation` helper). `core/legacy/`
is an internal 1.x load-only compatibility layer, not public API.

## License headers

External contributions are accepted under Apache 2.0. Please ensure new
files contain the SPDX header:

```python
# SPDX-License-Identifier: Apache-2.0
```

