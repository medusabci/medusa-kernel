# medusa-kernel — architecture & contributor guide

medusa-kernel is a pure-Python library for biomedical signal processing and
machine learning (EEG, MEG, ECG, EMG, EOG, NIRS). It is the algorithmic core of
the MEDUSA© ecosystem and is published on PyPI as `medusa-kernel`. This document
is the single source of truth for how the codebase is organized and the
principles that all contributions must follow.

---

## 1. Scope

medusa-kernel is a **library**, not an application. It provides reusable,
paradigm-agnostic building blocks: array operations, signal metrics, graph
metrics, a data model for recordings, machine-learning estimators, and
visualizations. It contains no GUIs of its own beyond a small set of reusable
widget tools, and assumes nothing about any particular experimental paradigm.

The library is cross-platform by construction. It is pure Python with no
compiled extensions, ships as a single universal wheel, and must run identically
on Linux, macOS, and Windows. Any OS-specific assumption is a defect.

---

## 2. Architecture

Top-level packages, in dependency order:

| Package | Responsibility |
| --- | --- |
| `core/` | Foundation everything else depends on: the runtime data model (`core/data/`), serialization, the recording schema, configuration trees, and shared utilities. |
| `signal/` | Operations on time-series arrays (signal → signal) at the root, and metrics (signal → scalar/vector) under `signal/metrics/`. |
| `graph/` | Graph-theoretic metrics over weighted adjacency matrices. |
| `ml/` | scikit-learn–style machine-learning and deep-learning estimators. |
| `plots/` | Matplotlib visualizations. |
| `widgets/` | PySide6 GUI tools — the only place Qt is used. |

Two abstraction levels, and new code must clearly belong to one:

1. **Array operations** — `signal/`, `signal/metrics/`, `graph/`. NumPy in,
   NumPy out; stateless functions, or stateful estimators following the
   scikit-learn `fit` / `transform` convention.
2. **Domain types** — `core/data/`. The runtime objects that structure data for
   persistence and metadata (`Signal`, `ChannelSet`, `Recording`, `Events`).

Tests live in `tests/`, outside the package, mirroring `src/medusa/`.

### 2.1 Functional design

**Processing routines are free functions, not methods on data containers.** A
filter is `frequency_filtering.IIRFilter(...).fit_transform(signal, fs)`, never
`eeg.filter(...)`. Every input a routine uses appears in its signature — an
array, a sampling rate, a channel set, a few parameters — and it returns arrays
or scalars.

This buys three properties that the whole library depends on:

- **Transparency.** No hidden reads from a container's attributes, which makes
  debugging, unit testing, and partial reuse straightforward.
- **Flexibility.** A function sees only an array plus its parameters, so the same
  filter or metric works on any modality or on synthetic data.
- **Separation of concerns.** Data containers describe *what was recorded*;
  functions decide *what is done with it*. The two evolve independently.

When contributing: write new processing, metric, and transform code as free
functions (or `fit`/`transform` estimators) over arrays. Data containers never
host a processing API.

---

## 3. The data model (`core/data/`)

The data model is BIDS-aligned and modality-by-channel. There are no per-modality
data classes; modality is a property of each channel.

- **`Signal`** — one acquisition stream: a 2-D `(n_samples, n_channels)` matrix
  with `fs`, a `ChannelSet`, and a `times` vector (always materialized; supports
  irregular or dropped-sample streams). A single `Signal` can mix channel types.
  Several devices/clocks become sibling `Signal`s, never a widened container.
- **`Channel` / `Sensor` / `ChannelSet`** — mirror the BIDS three-file split. A
  `Channel` is a data column with a BIDS channel type; a `Sensor` is a physical
  transducer with an optional 3-D position; a `ChannelSet` groups ordered
  channels and sensors with independent cardinalities, linked by `uid`, plus a
  reference scheme (`reference_method ∈ {common, average, bipolar}`) and a
  coordinate system. It is built with a chainable, eagerly validated builder.
- **`Recording`** — one BIDS run: a `BidsInfo` identity, a dict of named
  `RecordingData` streams (`Signal` is the shipped type), parallel sidecars, an
  `Events` timeline, and free-form experiment metadata. All times are seconds
  against one shared origin.
- **`Events`** — a BIDS `events.tsv`-aligned timeline backed by a pandas
  `DataFrame`, with `onset` and `duration` always present.
- **`Recorder`** — live acquisition with an in-RAM ring buffer (sized in seconds)
  and optional crash-safe streaming to disk.

`core/data/eeg/` holds EEG-specific montage and coordinate helpers (standard
10-20 / 10-10 / 10-05 montages, label resolution, 3-D→2-D projection); the
generic `ChannelSet` delegates montage knowledge to it.

**Scope:** the model is BIDS-aligned so external tools can emit BIDS, but the
kernel does not read or write BIDS dataset trees. Imaging/genetics containers and
acquisition transports are out of scope.

---

## 4. Signal shape contract

Signal arrays are channels-last, with these canonical representations:

| Representation | Shape |
| --- | --- |
| Time-domain | `(n_segments, n_samples, n_channels)` |
| Power spectral density | `(n_segments, n_frequencies, n_channels)` |
| Time–frequency | `(n_segments, n_frequencies, n_times, n_channels)` |

`n_segments ≥ 1`; `n_times` (hop-compressed TFR frames) ≠ `n_samples`. Always use
the term **segment**, never "epoch".

Every function that accepts a signal array validates and promotes it at entry
with `check_data_dims`:

```python
def check_data_dims(data: NDArray, rep_type) -> tuple[NDArray, tuple[int, ...]]
```

- `rep_type` is one of six literals: `'time'` (2-D), `'time_segments'` (3-D),
  `'freq'` (2-D), `'freq_segments'` (3-D PSD), `'time_freq'` (3-D),
  `'time_freq_segments'` (4-D TFR). Unknown values raise `ValueError`.
- It returns `(out, inserted)`: the array promoted to canonical ndim, and the
  tuple of axis positions that were inserted. Promotion emits a `UserWarning`
  only when axes are added. Dtype is preserved (`np.asarray`).
- **`rep_type` is internal.** Each function hardcodes the one it needs; it is
  never a user-facing argument.
- **2-D streaming convention.** Single-segment functions accept
  `(n_samples, n_channels)`, promote internally, and squeeze the inserted axes
  back out before returning, so the caller's ndim is preserved:
  `np.squeeze(out, axis=inserted) if inserted else out`.

---

## 5. Coding conventions

### 5.1 Argument names

Use these names and types exactly; never the listed alternatives:

| Concept | Name | Type | Never |
| --- | --- | --- | --- |
| Input signal | `signal` | `NDArray` | `data`, `x`, `eeg`, `sig` |
| Sampling rate | `fs` | `float` | `sample_rate`, `freq`, int |
| Channels | `n_channels` | `int` | `n_cha`, `ncha`, `n_ch` |
| Segments | `n_segments` | `int` | `n_epochs`, `n_trials` |
| Samples / freqs / TFR frames | `n_samples` / `n_frequencies` / `n_times` | `int` | — |
| Frequency band | `band` | `tuple[float, float]` | `fband`, `target_band`, `freq_band` |
| Window / overlap fraction | `segment` / `overlap` | `float` | `*_pct` |
| Filter order / type / method | `order` / `btype` / `filt_method` | `int` / `Literal[...]` / `Literal[...]` | — |
| Normalization | `norm` | `Literal[...] \| Literal[False]` | separate `normalize` + `norm` |
| Power type | `power_type` | `Literal['absolute', 'relative']` | `type` |

`band` is always singular — no function takes a list of bands. `norm` is a single
argument (`False` = off, a string selects the method). **Never shadow built-ins**
(`type`, `filter`, `input`, `id`, `list`, `max`, `min`, `sum`).

### 5.2 Type annotations

- PEP 484 annotations on every public function and method, **in the signature
  only** — never repeated in the docstring `Parameters` block.
- Use `numpy.typing.NDArray` without a dtype; do not force `float64` (users often
  have `float32`, and forcing a dtype costs a copy). Annotate return types.
- Sanitize array inputs with `np.asarray(signal)` (zero-copy, dtype-preserving),
  not `np.array`.
- Do not use `from __future__ import annotations` (Python 3.13 evaluates
  annotations lazily).

### 5.3 Docstrings

NumPy style (`Parameters` / `Returns` / `Raises` / `Notes` / `References` /
`Examples`). Document each array parameter's canonical shape as the first line of
its description. `Raises` is mandatory when the function validates input;
`Examples` is mandatory and must contain at least one runnable snippet using the
current import path. `check_data_dims` in `core/utils.py` is the canonical
exemplar.

### 5.4 General

- Paths via `pathlib`; never `os.sep` or string concatenation with separators.
- Absolute imports from `medusa.<module>`. No OS-conditional imports.

---

## 6. Public API rules

- **Every public module declares `__all__`.** Names not in `__all__`, and any
  underscore-prefixed name, are internal and free to change. Each package
  `__init__` re-exports its public symbols so both package-level and
  module-level imports work.
- **The top-level `medusa/__init__.py` is intentionally minimal:** a docstring
  and a runtime-resolved `__version__`, with `__all__ = ["__version__"]`. It does
  not flat-dump submodules and stays free of Qt and PyTorch imports, so
  `import medusa` is light and headless-safe. Functionality is reached through
  subpackages.
- **Deprecation policy:** a symbol is marked with a `DeprecationWarning` for one
  minor release and removed in the next, via the helpers in
  `core/_deprecation.py`.

---

## 7. Package organization principles

- **Group by data type, not by action.** Top-level packages (`signal/`,
  `graph/`, and any future `image/`, `video/`) are named after the data they
  operate on; operations and metrics live one level down. A new data type slots
  in as a sibling package.
- **Within a data-type package:** operations are flat at the root (signal →
  signal), metrics live under `metrics/<family>/` (signal → scalar/vector).
  `transforms.power_spectral_density` is an operation; `metrics/spectral/band_power`
  is a metric.
- **Metric families are organized by the question they answer**, not by lineage:
  - `spectral/` — which frequencies? (`band_power`, `median_frequency`, `spectral_edge_frequency`)
  - `nonlinear/` — how regular/structured? (sample & multiscale entropy, Shannon spectral entropy, Lempel-Ziv complexity, central tendency measure)
  - `discriminability/` — class separation? (`signed_r2`)
  - `connectivity/` — channel relations? (`aec`, `iac`, `pli`, `plv`, `wpli`)

  Placement follows the question: `shannon_spectral_entropy` lives in
  `nonlinear/` (it is an entropy) even though it is computed from a PSD.
- **Three names prevent kitchen-sink modules:** `core/schema.py` (on-disk format
  contract) ≠ `core/settings_tree.py` (runtime configuration tree) ≠ `core/data/`
  (runtime domain types).

---

## 8. Serialization & recording format

- **Components persist through `core/serialization.py`.** A serializable type
  inherits `SerializableComponent` and implements `to_serializable_obj` /
  `from_serializable_obj`; it can be written to `bson`, `json`, or `mat`. Types
  whose state is not cleanly serializable inherit `PickleableComponent` (a
  portable `dill` bundle). `.mat` round-trips losslessly.
- **`Recording` adds HDF5.** It overrides `save` / `load` to support `h5` / `hdf5`
  (chunked, gzip-compressed, append-friendly) in addition to the universal
  formats, and defers everything else to the base class. The format is taken from
  the path extension unless overridden.
- **Schema versioning.** `core/schema.py` defines `SCHEMA_VERSION`, stamped into
  every serialized dict, and `validate_recording_dict`. Older or missing versions
  are reported as unsupported; newer versions prompt an upgrade. Forward
  migrations (`n → n+1`) are added by the release that introduces the change.

---

## 9. Machine learning (`ml/`)

The kernel **applies** models; it is not a model-development or hyperparameter-
search framework. It can train a provided architecture from scratch on new data;
pre-trained weights come from outside.

- **Top-level `medusa.ml` exposes no names and imports no PyTorch**, so DSP-only
  and headless installs stay torch-free. All deep-learning code is under the
  torch-gated `ml/torch_models/`, imported on demand.
- **Three layers:** a plain `nn.Module` **backbone** (feature extractor, no head,
  exposing `backbone_features`, `input_layout`, and `get_config()`) → a
  PyTorch-Lightning **task** (loss + loop) → a scikit-learn **estimator**
  (`fit` / `predict` / `score` / `encode`). Estimators subclass
  `sklearn.base.BaseEstimator` plus the matching mixin, so `get_params`,
  `set_params`, and `clone()` work; there are no bespoke base ABCs.
- **One estimator per use case** — no head registries or hidden default heads.
  Heads and `classes_` are inferred from data at `fit` time. Neural-network
  modules live in `backbones/`; shipped backbones are `EEGInception`,
  `EEGInceptionV2`, `EEGNet`, and `EEGSym`.
- **Torch gating.** Importing `ml/torch_models/` runs `require_torch()`;
  estimator modules also run `require_lightning()`. A missing install raises
  `TorchNotInstalled` (an `ImportError` subclass) naming the recommended
  versions. Devices flow through `device=` into the Lightning `Trainer`; there is
  no global device state.
- **Portable persistence.** Estimators are `PickleableComponent`s saved as a
  `config + state_dict` bundle (estimator params, backbone `get_config()`, CPU
  `state_dict`, fitted head state) — never a raw module pickle — so a model
  fitted on GPU reloads on CPU and vice versa.
- `ml/cross_validation.py` provides biomedical splitters following sklearn's
  `split()` protocol (returning indices); `ml/metrics.py` wraps `sklearn.metrics`.

---

## 10. Visualization (`plots/`)

- Matplotlib only. Functions take **plain arrays** validated with
  `check_data_dims` (no `Signal` duck-typing); scalp plots take a `ChannelSet`
  for positions. The target `ax` is a required argument.
- One-shot `plot_*()` functions return `(ax, artists)`, where `artists` is a
  named dict; each has a paired stateful `*Plot` engine class whose `set_data()`
  mutates artists in place for interactive/animated use. Export with the free
  `save_figure`; `optimal_grid` lays out subplots.
- **Plots never run signal transforms.** A plot may reduce given data for display
  (e.g. a mean ± error band) but never computes an FFT, filter, or segmentation —
  callers pass precomputed arrays.

---

## 11. Styling & GUI

- **The kernel owns no visual identity.** All colors, fonts, colormaps, and Qt
  stylesheets come from the companion `medusa-style` package, the MEDUSA-wide
  styling source of truth (a core dependency). Plotting code calls it directly;
  importing `plots/` never mutates global matplotlib state, and per-call defaults
  remain overridable.
- **All Qt/PySide6 code lives only in `widgets/`.** The non-visual layers
  (`core`, `signal`, `graph`, `ml`) are Qt-free; this is enforced by a headless
  import test and a dedicated CI job. Widget tools include a figure browser
  (`PlotVisualizer`), a time-series/TFR viewer (`TimeViewer`), and settings-tree
  editors. Widgets theme their application through `medusa-style`.

---

## 12. Dependencies

- **Core (always installed):** NumPy, SciPy, scikit-learn, statsmodels, h5py,
  tqdm, PyWavelets, matplotlib, `bson`, `dill`, PySide6, and `medusa-style`.
  matplotlib and PySide6 are core, so the plotting and widget layers work without
  any extra.
- **Extras:** `[dev]` (pytest, ruff, mypy, coverage/benchmark), `[docs]` (MkDocs
  toolchain), `[all]` = dev + docs.
- **PyTorch is never a dependency or an extra.** Its wheels are coupled to the
  user's accelerator stack, so it is installed separately; the deep-learning
  estimators additionally need PyTorch Lightning. All deep learning uses
  PyTorch.

---

## 13. Build, tests, and release

- **Build:** `hatchling` backend with PEP 621 metadata in `pyproject.toml`;
  src-layout (`src/medusa/`); environment and lockfile managed by `uv`. A single
  pure-Python universal wheel.
- **Tests:** in `tests/` (mirroring `src/medusa/`), run with
  `--import-mode=importlib`. The CI matrix covers Linux/macOS/Windows on Python
  3.13. `ruff` enforces annotation and builtin-shadowing rules as a blocking gate
  on the layers held to the full convention set, and runs advisory elsewhere.
- **Release:** semantic versioning; published to PyPI from CI via OIDC trusted
  publishing on a GitHub release.

---

## 14. License

Apache License 2.0 (permissive, with an explicit patent grant). Contributions are
accepted inbound = outbound under the same license. See `LICENSE` and `NOTICE`.
