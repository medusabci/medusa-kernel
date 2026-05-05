# medusa-kernel

Signal-processing core of the MEDUSA© ecosystem. The only component published on **PyPI** (`pip install medusa-kernel`). Consumed by Platform, Analyzer, the apps, the tutorial notebooks, and any external user.

> Workspace context and global rules: [`../AGENTS.md`](../AGENTS.md). Kernel-specific improvement plan / refactor toward v2.0: [`TODO.md`](TODO.md). Cross-cutting ecosystem improvements: [`../TODO.md`](../TODO.md).

---

## 1. Purpose and scope

**Is:**
- A pure Python library for biomedical signal processing (EEG/MEG/ECG/EMG/EOG/NIRS).
- A stable API third parties import from `pip install medusa-kernel`.
- The source of truth for the ecosystem's processing algorithms (filters, metrics, connectivity, BCI pipelines, models, data IO).

**Is not:**
- A desktop application (that is `medusa-platform`).
- A GUI wrapper (although today it drags in `PySide6` as a dependency — see `TODO.md` K3).
- Windows-specific. **It must work on Linux and macOS** (see `TODO.md` K4). Any OS assumption is a bug.

---

## 2. Architecture

```
medusa-kernel/
├── medusa/                           ← the Python package
│   ├── __init__.py                   ← re-exports frequency_filtering, spatial_filtering, epoching
│   │
│   ├── components.py                 ← base classes (SerializableComponent, ProcessingMethod, …)
│   ├── frequency_filtering.py        ← IIR/FIR, notch, bandpass…
│   ├── spatial_filtering.py          ← CAR, Laplacian, ICA, CSP…
│   ├── epoching.py                   ← epoch extraction
│   ├── transforms.py                 ← FFT, wavelets, Hilbert…
│   ├── artifact_removal.py
│   ├── signal_orthogonalization.py
│   ├── classification_utils.py       ← classification helpers
│   ├── performance_analysis.py
│   ├── optimization.py
│   ├── signal_generators.py          ← synthetic signals (testing)
│   ├── deep_learning_models.py       ← TF/Keras models
│   ├── pytorch_integration.py        ← PyTorch models
│   │
│   ├── bci/                          ← high-level BCI paradigms
│   │   ├── erp_spellers.py
│   │   ├── cvep_spellers.py
│   │   ├── ssvep_spellers.py
│   │   ├── mi_paradigms.py
│   │   ├── nft_paradigms.py
│   │   └── metrics.py
│   │
│   ├── dataio/                       ← MEDUSA recording format (see §6)
│   │   ├── schema.py                 ← file schema
│   │   ├── compatibility.py          ← cross-version migration
│   │   └── biosignals/               ← per-biosignal read/write
│   │
│   ├── meeg/                         ← M/EEG: montages, electrodes, etc.
│   │   ├── meeg.py
│   │   ├── meeg_montages.py
│   │   ├── eeg_standard_2D.tsv       ← coords (package data)
│   │   ├── eeg_standard_3D.tsv
│   │   └── _eeg_standard/
│   │
│   ├── signal_metrics/               ← complexity / spectral metrics
│   │   ├── band_power.py, central_tendency.py, median_frequency.py,
│   │   ├── multiscale_entropy.py, sample_entropy.py, signed_r2.py,
│   │   ├── shannon_spectral_entropy.py, spectral_edge_frequency.py,
│   │   ├── lempelziv_complexity.py, multiscale_lempelziv_complexity.py
│   │   └── computeLZC.dll            ← ⚠ Windows-only binary (see §10)
│   │
│   ├── connectivity_metrics/         ← PLV, AEC, wPLI…
│   ├── graph_metrics/                ← clustering, path length, small-world…
│   ├── analysis/                     ← high-level analysis (incl. time_plot UI)
│   ├── plots/                        ← visualizations (matplotlib + PySide6)
│   │
│   ├── ecg.py, emg.py, eog.py, nirs.py   ← non-EEG biosignals
│   ├── notify_me.py, settings_schema.py, utils.py
│
├── tests/                            ← outside the package; not distributed
│   ├── test_components.py
│   ├── test_ecg.py
│   ├── test_signal_generators.py
│   ├── test_transforms.py
│   ├── data/, examples/, local_activation/, plots/
│
├── setup.py                          ← classic packaging (no pyproject.toml yet)
├── README.md
├── LICENSE                           ← CC BY-NC-ND 2.0 (problematic — see §9)
└── .github/workflows/python-publish.yml   ← PyPI release
```

### 2.bis. Abstraction levels

Think of Kernel in three layers:

1. **Low level — array-level operations.** `frequency_filtering`, `spatial_filtering`, `epoching`, `transforms`, metrics in `signal_metrics/`, `connectivity_metrics/`, `graph_metrics/`. Typical input/output: NumPy arrays. Stateless or wrapped in a `ProcessingMethod` object.
2. **Mid level — domain abstractions.** `meeg/`, `dataio/`, `components.py`. Define the domain types (`Recording`, `Experiment`, `EEG`, `Channel`, montages…) and how they are serialized.
3. **High level — BCI pipelines.** `bci/erp_spellers.py`, `bci/cvep_spellers.py`, etc. Compose the lower layers into a complete paradigm flow. This is what Platform's apps usually import.

When adding functionality, decide first which layer it belongs to. Mixing layers is future debt.

### 2.ter. Functional architecture (vs MNE-style)

A foundational design rule of medusa-kernel: **processing routines are free functions, not methods on container classes.**

- Signal-processing code in `frequency_filtering`, `spatial_filtering`, `epoching`, `transforms`, `signal_metrics/`, `connectivity_metrics/`, `graph_metrics/`, `bci/`, etc. takes its inputs as explicit parameters (`signal: np.ndarray`, `fs: float`, `channel_set`, …) and returns arrays or scalars.
- It does **not** dispatch on, or read attributes from, a biosignal container. There is no `eeg.filter(...)` — there is `frequency_filtering.filter_data(signal, fs, ...)`.

This is a deliberate departure from the MNE-Python style (`raw.filter()`, `epochs.average()`, …). The trade-off is more verbose call sites in exchange for:

1. **Transparency.** Every input a function uses appears in its signature; there are no hidden reads from container attributes. Easier to debug, unit-test, and reuse partial chains.
2. **Flexibility.** Functions don't depend on any biosignal type — the same filter, metric, or transform works on EEG, ECG, NIRS, or a synthetic test array, because all it sees is an `ndarray` plus the parameters it needs.
3. **Schema richness per modality.** Because containers don't host the processing API, splitting modalities (e.g. `EEG` vs a future `MEG`) into separate subpackages costs nothing at the function layer. Each modality keeps its own precise schema (channel set, montage, reference scheme, sensor coordinates, …) instead of collapsing into a generic union container or a lowest-common-denominator interface.

The role of the biosignal classes (`EEG`, `ECG`, `EMG`, …) under `core/biosignals/<modality>/` is therefore **structuring data for persistence and capturing per-modality metadata**, not exposing a processing API. They describe *what was recorded*, not *what can be done with it*.

**Practical consequences when contributing:**
- New processing / metric / transform code: write it as a free function on arrays + explicit params. Do not add it as a method on `EEG`, `ECG`, etc.
- New biosignal types: subpackage under `core/biosignals/<modality>/` (per K1) with a class describing the recording's structure and metadata for serialization. Do not bundle processing methods on it.
- Simultaneous multi-modality recordings (e.g. EEG+EOG, EEG+MEG) are handled at the `Recording` level (sibling biosignals on the same subject/session), not by widening a single container.

---

## 4. Dependencies and consumers

**Depends on:**
- Core scientific stack: NumPy, SciPy, scikit-learn, statsmodels, h5py, pandas, matplotlib.
- Optional and currently mandatory (to be moved to extras — `TODO.md` K3): PySide6, TensorFlow/Keras, PyTorch.
- Persistence: `bson`, `dill` (under review — see `TODO.md` K3).

**Consumed by:**
- `medusa-platform/src/` (Platform's microkernel).
- `medusa-platform/src/accounts/<user>/apps/*/` (every app installed in Platform).
- `medusa-analyzer/`.
- `medusa-tutorials/` (Jupyter notebooks).
- External users via `pip install medusa-kernel`.

A breaking change in Kernel cascades through every consumer above. See `../AGENTS.md` §3 for the checklist of usage searches before modifying a public function.

---

## 5. Code conventions

- **Python ≥3.10, <3.14** (see `setup.py`).
- **NumPy + SciPy** are the lingua franca. Any new algorithm should follow the signature `np.ndarray → np.ndarray` (or return a serializable dataclass / component).
- **Free functions, not methods on biosignal containers** (see §2.ter). Processing / metric / transform code takes arrays + explicit params and is independent of `EEG`, `ECG`, etc. Biosignal classes are for persistence + per-modality metadata, not for dispatching processing.
- **Serializable components**: classes that persist to disk inherit from `SerializableComponent` in `components.py`. Define `to_serializable_obj` / `from_serializable_obj` and the file extensions (`.cvep.mdl`, `.mi.mdl`, `.rec.mat`, etc.).
- **Processing methods**: inherit from `ProcessingMethod`. Implement `fit`, `transform`, `fit_transform` following the scikit-learn convention when reasonable.
- **Paths via `pathlib`**, never `os.sep` or string concatenation with `\`. Kernel must work on Linux and macOS.
- **Absolute imports** from `medusa.<module>`.
- **Docstrings** in NumPy style (`Parameters`, `Returns`, `Notes`, `References`).
- **No OS-conditional imports** without a strong reason. If they appear, encapsulate them and provide a clear fallback.

---

## 6. Public contracts

Kernel is consumed by third parties. Incompatible changes must be treated as such:

1. **Public API** — today there is no systematic `__all__` nor `public/` vs `internal/` distinction. *De facto*, anything importable from `medusa.<module>` is public until further notice. Before renaming / moving / deleting anything, search for usages in:
   - `medusa-platform/src/`
   - `medusa-platform/src/accounts/*/apps/*/`
   - `medusa-analyzer/`
   - `medusa-tutorials/`
   - (When feasible) GitHub code search across `medusabci/*`.
2. **MEDUSA recording format** — lives in `medusa/dataio/schema.py` and `compatibility.py`. Written by `medusa-platform/src/cont_rec/`, read by `medusa-analyzer/data_loader/`. Format changes → bump schema and add migration in `compatibility.py`. Mandatory cross-version read tests.
3. **Serialized models** (`.cvep.mdl`, `.mi.mdl`, …) — trained by Platform via `bci/*`. If the internal model structure changes, old models must keep loading or the break must be documented in CHANGELOG.

---

## 7. Tests and CI

- Tests in `tests/` (NOT `medusa/tests/`). Outside the distributed package.
- Partial coverage: `components`, `ecg`, `signal_generators`, `transforms`. Most of `bci/`, `signal_metrics/`, `dataio/` is uncovered.
- CI: `.github/workflows/python-publish.yml` only handles PyPI publish. **No test CI yet** — adding one (`pytest` matrix `ubuntu-latest, windows-latest, macos-latest`) is in `TODO.md` K7.
- When touching a module, run the existing tests at least locally: `pytest tests/`.

---

## 8. Distribution and release

- Versioning scheme: **semver** (`1.4.3` current in `setup.py`).
- Released to PyPI via GitHub Actions (`python-publish.yml`) when a GitHub release is published.
- Bump version in `setup.py` before tagging.
- When publishing 1.5 or 2.0, coordinate with `medusa-platform/requirements.txt` and `medusa-installer` (Platform pins Kernel with an exact version).
- Apps declare (when the versioned manifest exists, `../TODO.md` E2) their compatible Kernel range; a Kernel major bump should reflect as invalidation of apps with `requires_kernel: ">=1.4,<2"`.

---

## 9. License

Currently: **CC BY-NC-ND 2.0**. Problematic (forbids derivatives, prohibits commercial use, not designed for software). Migration to a standard OSS license (MIT / Apache 2.0 / MPL 2.0) is a pending decision — see `TODO.md` K8 (mirror of `../TODO.md` E7). **Do not accept relevant external contributions until this is resolved**, or require an explicit CLA.

---

## 10. Known pitfalls

- **`PySide6` in `install_requires`.** Pulls in ~200 MB of Qt for headless users. Only `plots/` and `analysis/time_plot/` actually need it. Treatment as an optional extra is pending (`TODO.md` K3). Meanwhile: late Qt imports inside `plots/` (not at module top level) so that `import medusa` does not fail when Qt is missing.
- **`computeLZC.dll`** in `signal_metrics/`. Windows-only binary. Today it confines Lempel-Ziv complexity to Windows. Pending: identify a pure-Python equivalent or replace with a cross-platform implementation (`TODO.md` K4).
- **Classic `setup.py`, no `pyproject.toml`.** Migration pending (`TODO.md` K2). When done, move `install_requires` → `[project.dependencies]` and `PySide6` → `[project.optional-dependencies.plots]`.
- **`bson` and `dill` in deps**: heavy for users that only want to filter signals. Review whether they are really required or can be downgraded to optional.
- **Re-exports in `__init__.py`**: only `frequency_filtering`, `spatial_filtering`, `epoching` are re-exported. Expand carefully — every top-level symbol is a public-API commitment.
- **`tests/data/`**: test datasets can be heavy. Do not commit new ones unless necessary.

---

## 11. How to work here

1. Before touching code, read the relevant section of [`../AGENTS.md`](../AGENTS.md) §3 ("Before modifying a public function in Kernel").
2. If the change touches the recording format or a serializable component, open [`TODO.md`](TODO.md) K6 first and consider whether a schema bump is required.
3. Local tests: `pytest tests/`. If adding functionality, add the matching test.
4. Commits in this repo, not in the workspace. Typical branch: `main` (stable) or feature branch.
5. Do not mix functional changes with packaging changes (`setup.py` / pyproject migration) in the same commit.
