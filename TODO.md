# medusa-kernel — Roadmap (v2.0)

The single source of truth for what is done and what remains in the **2.0**
release — a major, breaking bump with **no 1.x back-compatibility**. Full
component rules and conventions live in [`AGENTS.md`](AGENTS.md).

> **One blocker dominates the remaining work.** The BCI pipelines are
> unimportable: they depend on a `medusa.core.pipeline` composition layer
> (`Algorithm` / `ProcessingMethod` / …) that was removed when `components.py`
> was split. Resolving this (see [`pipelines`](#pipelines)) unblocks the BCI
> documentation gate, the pipeline API-convention pass, and the ML consumer rewire.

---

## 1. General project changes

### Settled architecture (context for the rest)
- **Layout by data type, not action.** `signal/`, `graph/` (future `image/`,
  `video/`) are named after their data; operations sit flat at the package root,
  metrics under `metrics/<family>/`. `core/` is the foundation, `pipelines/` the
  umbrella for domain flows, `widgets/` the *only* place Qt is imported.
- **Free functions over container methods.** Processing / metric code takes
  `NDArray` + explicit params and returns arrays/scalars; biosignal classes only
  structure data + metadata.
- **Dependencies.** NumPy / SciPy / sklearn / pandas / h5py / matplotlib +
  `bson` / `dill` + **PySide6** + **medusa-style** are all **core**. **PyTorch is
  user-installed** (never an extra — wheels are platform-coupled). Extras:
  `[dev]`, `[docs]`. There is no `[widgets]` extra (PySide6 is core).
- **Public API.** Each module's `__all__` is the supported surface; everything
  else (incl. `_`-prefixed names) is internal. Deprecation = `DeprecationWarning`
  for one minor version, removal the next (`core/_deprecation.py`).
- **Recording format.** `SCHEMA_VERSION = 2`, light validation, no Pydantic,
  **read-only** (no v1→v2 migration; v2 is the floor). The full versioned-schema
  apparatus is **dropped**.
- **License.** Apache 2.0 from 2.0.0 (1.x stays CC in PyPI history).

### Settled conventions (enforced on `signal/` + `graph/`; full detail in AGENTS.md)
- Python ≥ 3.13; PEP 484 annotations in the **signature only**; `NDArray` without
  a forced dtype; `np.asarray` to preserve dtype; NumPy-style docstrings with a
  runnable `Examples` block.
- **Signal shapes:** time `(n_segments, n_samples, n_channels)`, PSD
  `(…, n_frequencies, …)`, TFR `(…, n_frequencies, n_times, …)` — channels last.
  Validate with `check_data_dims(data, rep_type) -> (out, inserted)` (6 internal
  `rep_type`s; the caller squeezes `inserted` to restore its ndim).
- **Arg names:** `signal`, `fs: float`, `n_channels`, `n_segments`, `band: tuple`,
  `segment` / `overlap` as fractions, `power_type`, `norm` (`False` = off | method).
  Never shadow built-ins.

### Project-wide remaining work
- **Unblock the pipelines.** Resolve the `core.pipeline` question and migrate
  `pipelines/bci/` (see [`pipelines`](#pipelines)). Gates the pipeline
  API-convention pass,
  `mkdocs --strict`, and the ML consumer rewire.
- **Test coverage.** Gate is 15 %; step to 30 % (add connectivity tests) →
  **50 % (the 2.0 target)**. Needs CSP/ICA/connectivity unit tests, `tests/ml/`
  (torch-gated), a `plots` headless-Agg smoke suite, and BCI tests. No DL in CI
  (no GPU runners; `@pytest.mark.torch`, auto-skipped).
- **Widen the quality gates.** `ruff A+ANN` is blocking only on `signal/`+`graph/`
  today (~480 `E/F/I/W/UP` + ~2.4k `ANN` gaps remain in `core` internals, `plots`,
  `widgets`, `ml`, `pipelines`); flip each layer as it is migrated. Likewise
  `mkdocs --strict`: 10 warnings left = 9 BCI docstrings + 1 stray
  `docs/tutorials/AGENTS.md` link (fix via `exclude_docs`).
- **Release / one-time (external).** Configure PyPI + TestPyPI trusted publishers
  and run a TestPyPI cycle; enable GitHub Pages → `gh-pages`; cut the first
  versioned release (`mike set-default latest`); add `docs-build` as a required
  check on `main`.
- **Consumer coordination.** 2.0 breaks `medusa-platform`, `-analyzer`,
  `-tutorials`, `-docs`, `-installer` (import rewrites + `>=2,<3` pins).
- **Pending deletions (user-handled).** Dropped docs (`signal_contract.md`,
  `api_policy.md`); superseded styling (`plots/style.py`, `plots/styles/`,
  `widgets/theme.py`); plots dead files (`figure.py`, `optimal_subplots.py`,
  `timeplot.py`, `head_plots.py`, templates, dead tests); `ml/torch_models/tasks/`
  remnants; and the four old `TODO.md` files this document replaces.

---

## 2. Package-level roadmap

### `core`
- **State.** Foundation done — `serialization`, `schema` (v2), `settings_tree`,
  `profiling`, `utils` (`check_data_dims`), and the full `core/data/` runtime
  model: BIDS-aligned `Recording` + typed-channel `Signal` + `Events` + `Recorder`
  HDF5 persistence (62 tests). `core/legacy/` is the internal 1.x load-only layer.
- **Remaining.** Resolve the `core.pipeline` composition layer (see
  [`pipelines`](#pipelines)); ruff/ANN pass on `core` internals.
- **Future.** `core/compatibility.py` only when a v3 schema lands; additive
  `core/data` extras — lazy out-of-RAM HDF5 reads, resume-append, a one-shot
  non-`Signal` array path, per-stream buffer override, multi coordinate-system /
  template-space support.

### `signal`
- **State.** Operations + metrics complete and convention-clean (annotations,
  shapes, arg names, docstrings).
- **Remaining.** `fourier_spectrogram` / `cwt_spectrogram` still take 1-D → return
  2-D — make them accept `(n_segments, n_samples, n_channels)` and return 4-D TFR
  (loop over segment × channel); add CSP/ICA/connectivity unit tests.
- **Future.** `cross_cwt` 1-D → TFR rewrite (currently deferred); if LZC
  throughput ever matters, a numpy-vectorized multiscale median
  (`sliding_window_view`) **before** any native/Cython port — pure Python is
  already cross-platform, so native extensions stay deliberately deferred.

### `graph`
- **State.** 12 flat metrics + a `surrogate_graph` op, annotated.
- **Remaining (real bugs).** `assortativity` calls a non-existent
  `degree.__degree_cpu`; `participation_coefficient` passes an extra arg to
  `degree`; `clustering_coefficient` emits a SyntaxWarning (bad escape). Add unit
  tests.
- **Future.** Promote to `metrics/<family>/` + `ops/` once the op set grows.

### `ml`
- **State.** The torch stack is done & verified — backbones (`EEGInception`,
  `EEGInceptionV2`, `EEGNet`, `EEGSym`), Lightning tasks, the shared private
  estimator engine, `TorchClassifier` + `TorchMultiTaskClassifier`, portable
  one-file persistence, torch/lightning guards. Design: sklearn estimator →
  Lightning task → plain `nn.Module` backbone; no bespoke base classes; no
  HPO/Optuna (model *development* is out of scope — the kernel only *applies*
  models).
- **Remaining.** `cross_validation.py` (biomedical splitters on sklearn's
  `split()` protocol → indices) — not started; `metrics.py` (thin
  `sklearn.metrics` wrappers; the old `mcc` was wrong) — not started; `tests/ml/`
  (torch-gated) — built last.
- **Future.** `TorchRegressor` + `MSERegressionTask`; an SSL task (`ssl.py`);
  optional HF-Hub persistence.
- **Open decision.** Keep or delete the legacy `optimization.py` (no replacement;
  HPO is out of scope).

### `plots`
- **State.** Done — scalp / topography / connectivity, `time_line` /
  `time_heatmap`, `shaded_line`; array-based, `ax`-mandatory, return
  `(ax, artists)`; no transform-computing plots; all styling comes from the
  `medusa-style` package (the kernel owns no palette/fonts/rcParams).
- **Remaining.** A headless-Agg smoke suite + the MkDocs gallery;
  `--strict`-clean.
- **Future.** An optional render-only `plot_spectrum` (`freq_line.py`, decide on
  merit); live advance modes (incremental `append()` / `set_window()`, blitting).
- **Open decisions.** Live-update contract as a `typing.Protocol` vs informal;
  snapshot / blit-toggle API; preview scrubber in the engine vs the Qt host.

### `widgets`
- **State.** The only Qt home — `time_viewer` (a thin shell over the `plots`
  engines), the `settings_tree` editor, the `plot_visualizer` figure browser;
  themed via `medusa-style`.
- **Remaining.** Coordinate `medusa-platform` adoption; ruff/ANN pass; delete the
  superseded `widgets/theme.py`.

### `pipelines`
- **State.** `bci/` (erp / cvep / ssvep spellers, mi / nft paradigms, performance,
  plots) is **currently unimportable** — every paradigm module imports the removed
  `medusa.core.pipeline`. The `__all__` + curated re-exports are already in place
  and will work once this is resolved.
- **Open decision (load-bearing).** Restore a slim `core/pipeline.py` composition
  layer (per the original layout / "kept `Algorithm`") **vs.** drop it entirely
  and rewrite the BCI models as `PickleableComponent`s that hold their sub-methods
  as attributes and orchestrate manually in `fit`/`predict` (per the ML-refactor
  direction). The two prior plans contradict each other — pick one before touching
  `bci/`.
- **Remaining (after unblock).** Apply the API conventions to `bci/`
  (`n_cha`→`n_channels`, `w_epoch_t`→`w_segment_t`, adopt the landed
  `segmentation` renames); fix stale
  imports — `mi_paradigms` (`normalize_epochs`→`normalize_segments`; the dropped
  `medusa.ml.classification`), `plots/mi.py` (gone `medusa.plots.head_plots`),
  `nft_paradigms` (`graph.degree` used as a module); the 9 BCI docstring fixes for
  `--strict`; BCI smoke / paradigm tests.
- **Future.** `pipelines/anesthesia/` (planned from 1.x
  `analysis/anesthesia_depth_monitoring`, not yet migrated — migrate or confirm
  dropped); further domains (`sleep`, `cognitive`, …) on demand.

---

*This file replaces the four former `TODO.md` files (repo root and under
`core/data/`, `ml/`, `plots/`), which can now be deleted.*
