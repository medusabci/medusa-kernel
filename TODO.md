# TODO — medusa-kernel v2.0

Kernel-specific plan for the **2.0** major bump. This document is independent of the workspace [`../TODO.md`](../TODO.md); the Kernel-related entries that used to live there (#5, #6, #7, #8, #10) are resolved here, plus a new one (K1: internal layout).

> Component context and operational rules: [`AGENTS.md`](AGENTS.md). Ecosystem context: [`../AGENTS.md`](../AGENTS.md). Workspace decisions (cross-platform, contracts, etc.) constrain this plan.

**Framing decision:** 2.0 is a major bump **with no backwards compatibility** with 1.x. We use the opportunity to leave Kernel coherent, modern and interoperable, aligned with standard practice (PEP 621, pyproject, optional extras, real cross-platform). We do not maintain 1.x compat shims beyond the grace period (see §K7 / coordination).

---

## Tracking table

| ID | Title | Origin | Impact | Effort | Status |
|----|-------|--------|--------|--------|--------|
| K1 | Package reorganization by abstraction levels | new | High | High | Pending |
| K2 | `pyproject.toml` (PEP 621) + lockfile | global E6 | Medium | Low | In progress |
| K3 | Optional extras: `[widgets]`, `[torch]` (matplotlib stays core) | global E6 | Medium | Low | Pending |
| K4 | Remove `computeLZC.dll`; real cross-platform Kernel | global E6 / E1 | Medium | Medium | Pending |
| K5 | Explicit public API (`__all__` + `_internal`) | global E3 | High | Medium | Pending |
| K6 | Truly versioned recording schema + cross-version tests | global E3 | High | Medium | Pending |
| K7 | Multi-OS CI with real tests | global E6 | High (long term) | Medium | Pending |
| K8 | License migration (CC BY-NC-ND → OSS) | global E7 | High (strategic) | Low (legal) | Pending |

All targets can be discussed independently; the reorganization (K1) touches all the others transversally.

---

## K1 — Package reorganization

**Problem**

The current layout mixes abstraction levels and groups by suffix instead of concept:

- 13 loose modules at the same level as 8 sub-packages at the root of `medusa/`.
- Three sibling sub-packages with `_metrics/` suffix (`signal_metrics`, `connectivity_metrics`, `graph_metrics`) — avoidable noise.
- Loose biosignals (`ecg.py`, `emg.py`, `eog.py`, `nirs.py`) sitting next to an already-structured `meeg/` — inconsistency.
- `bci/` and `analysis/` live at the same level but their respective roles were never documented. The "online vs offline" distinction sometimes attributed to them is false: anesthesia metrics (`analysis/`) can run online; BCI pipelines (`bci/`) are used offline when re-analyzing recordings. The real distinction is **general reusable layers** (`core/`, `signal/`, `graph/`, `ml/`, `plots/`) vs **domain-specific pipelines** (`bci/` and future siblings — `anesthesia/`, `sleep/`, …).
- `analysis/time_plot/` is an interactive visualization tool, not a domain pipeline. It is misclassified.
- `dataio/` already has `biosignals/` inside, but the individual biosignals (`ecg.py`, etc.) live outside. Conceptual collision.
- `setup.py` declares `PySide6`, implicit TF/Torch, no extras. See K3.

**Proposed solution**

v2.0 layout:

```
medusa/
├── core/                          ← foundation: persistence, format, runtime data types,
│   │                                pipeline abstractions. The old monolithic `components.py`
│   │                                (~1500 lines, 13+ classes across 5 unrelated families)
│   │                                is split here by concern; runtime data types live under
│   │                                `core/data/`.
│   │
│   ├── data/                      ← runtime domain type hierarchy
│   │   ├── signal.py              ← Signal abstract base (generic time-series: sample rate,
│   │   │                            channels, time axis) plus CustomSignal. Replaces the old
│   │   │                            BiosignalData base — what was generic lives here, what
│   │   │                            was modality-specific lives in each modality class.
│   │   │                            Future non-biosignal time-series (audio, environmental,
│   │   │                            simulation) inherit from Signal too.
│   │   ├── recording.py           ← Recording, ConsistencyChecker (was components.Recording)
│   │   ├── experiment.py          ← ExperimentData, CustomExperimentData (was components.*)
│   │   ├── events.py              ← Events (NEW first-class type; today events are loose
│   │   │                            numpy arrays inside BiosignalData / ExperimentData)
│   │   └── biosignals/            ← biosignal modalities — each inherits Signal directly.
│   │       │                        No `base.py`: the old BiosignalData wrapper is dropped
│   │       │                        (its generic concerns migrate to Signal). Sibling
│   │       │                        non-biosignal types (audio, environmental, …) would
│   │       │                        live as `core/data/<type>/` peers of `biosignals/`.
│   │       ├── eeg/               ← was top-level meeg/
│   │       │   ├── eeg.py         ← EEG runtime class (Signal subclass)
│   │       │   ├── montages.py    ← was meeg_montages.py
│   │       │   └── eeg_standard_*.tsv
│   │       ├── ecg/               ← was loose ecg.py — package now, room for lead defs,
│   │       │   └── ecg.py           R-peak detectors, etc.
│   │       ├── emg/
│   │       │   └── emg.py
│   │       ├── eog/
│   │       │   └── eog.py
│   │       └── nirs/
│   │           └── nirs.py
│   │
│   ├── serialization.py           ← SerializableComponent, PickleableComponent
│   │                                (was components.py, persistence base classes)
│   ├── schema.py                  ← versioned on-disk format spec for a serialized Recording:
│   │                                SCHEMA_VERSION + Pydantic v2 model that validates the
│   │                                serialized dict (was dataio/schema.py — populated per K6).
│   │                                Pure format contract — references types in `core/data/`
│   │                                but does NOT define them.
│   ├── compatibility.py           ← cross-version migration registry n→n+1 (was dataio/
│   │                                compatibility.py); uses the Pydantic models from schema.py
│   ├── pipeline.py                ← ProcessingMethod, ProcessingFuncWrapper,
│   │                                ProcessingClassWrapper, PipelineConnector, Pipeline,
│   │                                Algorithm (was components.py, pipeline abstractions)
│   ├── settings_tree.py           ← SettingsTree only, no Qt (was settings_schema.SettingsTree).
│   │                                Renamed to avoid name collision with schema.py — a
│   │                                settings tree is a runtime configuration tree, not a
│   │                                versioned format contract. The Qt widget that pairs with
│   │                                it lives in `widgets/settings_tree/` (see below).
│   ├── profiling.py               ← was performance_analysis.py (perf_analysis decorator;
│   │                                NOT about model performance — runtime profiling)
│   └── utils.py                   ← misc utilities; absorbs ThreadWithReturnValue from
│                                    the old components.py
│
├── signal/                        ← time-series data type: operations + metrics.
│   │                                Renamed from `processing/`: more specific; named after
│   │                                the data type, not the action. Mirrors the `scipy.signal`
│   │                                precedent. Future top-level data types (`image/`,
│   │                                `video/`, `genetics/`, …) replicate this internal
│   │                                shape — operations at the package root, metrics under
│   │                                `<type>/metrics/` grouped by family.
│   │
│   ├── frequency_filtering.py     ←┐
│   ├── spatial_filtering.py        │
│   ├── segmentation.py             │ OPERATIONS (signal → signal),
│   ├── transforms.py               │ flat at signal/ root. Note: PSD lives here
│   ├── artifact_removal.py         │ (transform: signal → spectrum), NOT in
│   ├── orthogonalization.py        │ metrics/spectral/ (which is signal → scalar).
│   └── generators.py              ←┘
│   │
│   └── metrics/                   ← signal → scalar/vector extractions, grouped by family.
│       │                            Was 1.x `signal_metrics/` (flat) + `connectivity_metrics/`.
│       │                            Each subcategory's `__init__.py` re-exports its public
│       │                            functions, so both `from medusa.signal.metrics.spectral
│       │                            import band_power` and `from medusa.signal.metrics.spectral
│       │                            .band_power import band_power` work.
│       ├── spectral/              ← band_power, median_frequency, spectral_edge_frequency,
│       │                            shannon_spectral_entropy
│       ├── complexity/            ← sample_entropy, multiscale_entropy,
│       │                            lempelziv_complexity, multiscale_lempelziv_complexity
│       ├── statistical/           ← central_tendency, signed_r2
│       └── connectivity/          ← aec, iac, pli, plv, wpli (was connectivity_metrics/)
│
├── graph/                         ← graph data type. Flat for now: 12 metric files
│   │                                (was graph_metrics/) plus `surrogate_graph.py` (1 op).
│   │                                Promotion to `graph/metrics/{topology, centrality,
│   │                                community}/` + `graph/<ops>/` is deferred until graph
│   │                                operations grow (graph filtering, transforms, projections,
│   │                                …) or the metric count rises substantially. The litmus
│   │                                test: when adding a 2nd graph operation, mirror `signal/`'s
│   │                                ops-at-root + metrics/-by-family layout.
│   ├── assortativity.py, betweenness_centrality.py, …
│   └── surrogate_graph.py
│
├── ml/                            ← was: models/. Renamed to "ml" because the folder is
│   │                                about training utilities, datasets, and HP search, not
│   │                                only model definitions; "ml" is the more honest umbrella.
│   │                                Trainable models (ML/DL), training utilities, HP search.
│   ├── classification.py          ← was classification_utils.py (sklearn-based ML helpers:
│   │                                one-hot, k-fold split, etc.)
│   ├── dataset.py                 ← was components.Dataset (ML training/eval dataset
│   │                                abstraction — moved out of core/ since it is consumed
│   │                                only by ML code, not by the recording layer)
│   ├── optimization.py            ← was medusa.optimization (Grinder, Optimizer for grid /
│   │                                random / bayesian hyperparameter search; sklearn + scipy)
│   ├── pytorch_utils.py           ← was pytorch_integration.py (config_pytorch, GPU detection,
│   │                                TorchExtrasNotInstalled / TorchNotConfiguredError)
│   └── deep_learning.py           ← was deep_learning_models.py (EEGInceptionV1 et al.;
│                                    all DL is PyTorch-based — depends on pytorch_utils)
│
├── plots/                         ← reusable visualization infrastructure based on
│   │                                matplotlib (no Qt). Domain-specific plots (ERP, MI)
│   │                                live with their pipeline, see `pipelines/bci/plots/`.
│   ├── head_plots.py              ← topographic / scalp plots
│   ├── generic_plots.py           ← generic plotting helpers
│   ├── optimal_subplots.py        ← layout helpers
│   ├── plot_visualizer.py         ← static viewer
│   ├── templates/                 ← plot templates
│   └── timeplot.py                ← static time plot (was loose plots/timeplot.py, 525 LOC)
│
├── widgets/                       ← reusable PySide6-based widgets (extra [widgets]).
│   │                                Anything in Kernel that imports `PySide6` lives here
│   │                                and ONLY here, so the rest of the package stays
│   │                                Qt-free and importable in headless environments.
│   │                                Reusable across Kernel consumers (Analyzer is the
│   │                                first non-Platform candidate — e.g. it can embed
│   │                                SettingsTreeWidget for its own configuration UIs,
│   │                                and the interactive time-plot viewer for inspection
│   │                                of recordings).
│   ├── settings_tree/             ← was settings_schema.{SettingsTreeWidget, TreeViewer,
│   │                                TextToTreeItem}. Qt widgets that render and edit a
│   │                                `core.settings_tree.SettingsTree`.
│   └── time_plot/                 ← interactive Qt time-plot viewer (was analysis/time_plot/,
│                                    2112 LOC + .ui + icons). Reusable for any signal.
│
└── pipelines/                     ← domain-specific pipelines (one subpackage per domain).
    │                                Composes reusable building blocks from core/, signal/,
    │                                graph/, ml/, plots/ into end-to-end flows tied to
    │                                a specific input data structure / experimental paradigm.
    │
    ├── bci/                       ← was top-level bci/ — BCI domain (online + offline)
    │   ├── erp_spellers.py
    │   ├── cvep_spellers.py
    │   ├── ssvep_spellers.py
    │   ├── mi_paradigms.py
    │   ├── nft_paradigms.py
    │   ├── performance.py         ← was bci/metrics.py (BCI performance: ITR, speller
    │   │                            accuracy, commands/min, etc.)
    │   └── plots/                 ← BCI-specific plots
    │       ├── erp.py             ← was plots/erp_plots.py
    │       └── mi.py              ← was plots/mi_plots.py
    │
    └── anesthesia/                ← was analysis/anesthesia_depth_monitoring/
        └── metrics.py
    # Future siblings (created on demand, not preemptively):
    #   sleep/   cognitive/   nirs/   …
#
# Future top-level data types (created on demand, mirroring signal/'s shape):
#   image/    ← image/{operations}.py + image/metrics/{color,texture,geometry,…}/
#   video/    ← analogous
```

Migration mapping (1.x → 2.0):

| 1.x | 2.0 |
|---|---|
| `medusa.frequency_filtering` | `medusa.signal.frequency_filtering` |
| `medusa.spatial_filtering` | `medusa.signal.spatial_filtering` |
| `medusa.epoching` | `medusa.signal.segmentation` |
| `medusa.transforms` | `medusa.signal.transforms` |
| `medusa.artifact_removal` | `medusa.signal.artifact_removal` |
| `medusa.signal_orthogonalization` | `medusa.signal.orthogonalization` |
| `medusa.signal_generators` | `medusa.signal.generators` |
| `medusa.signal_metrics.band_power` | `medusa.signal.metrics.spectral.band_power` |
| `medusa.signal_metrics.median_frequency` | `medusa.signal.metrics.spectral.median_frequency` |
| `medusa.signal_metrics.spectral_edge_frequency` | `medusa.signal.metrics.spectral.spectral_edge_frequency` |
| `medusa.signal_metrics.shannon_spectral_entropy` | `medusa.signal.metrics.spectral.shannon_spectral_entropy` |
| `medusa.signal_metrics.sample_entropy` | `medusa.signal.metrics.complexity.sample_entropy` |
| `medusa.signal_metrics.multiscale_entropy` | `medusa.signal.metrics.complexity.multiscale_entropy` |
| `medusa.signal_metrics.lempelziv_complexity` | `medusa.signal.metrics.complexity.lempelziv_complexity` |
| `medusa.signal_metrics.multiscale_lempelziv_complexity` | `medusa.signal.metrics.complexity.multiscale_lempelziv_complexity` |
| `medusa.signal_metrics.central_tendency` | `medusa.signal.metrics.statistical.central_tendency` |
| `medusa.signal_metrics.signed_r2` | `medusa.signal.metrics.statistical.signed_r2` |
| `medusa.connectivity_metrics.*` | `medusa.signal.metrics.connectivity.*` |
| `medusa.graph_metrics.*` | `medusa.graph.*` (flat for now) |
| `medusa.classification_utils` | `medusa.ml.classification` |
| `medusa.optimization` | `medusa.ml.optimization` |
| `medusa.deep_learning_models` | `medusa.ml.deep_learning` |
| `medusa.pytorch_integration` | `medusa.ml.pytorch_utils` |
| `medusa.performance_analysis` | `medusa.core.profiling` |
| `medusa.components.SerializableComponent`, `.PickleableComponent` | `medusa.core.serialization` |
| `medusa.components.Recording`, `.ConsistencyChecker` | `medusa.core.data.recording` |
| `medusa.components.ExperimentData`, `.CustomExperimentData` | `medusa.core.data.experiment` |
| `medusa.components.BiosignalData` | `medusa.core.data.signal.Signal` (renamed and generalized; see "Why this layout") |
| `medusa.components.CustomBiosignalData` | `medusa.core.data.signal.CustomSignal` |
| `medusa.components.ProcessingMethod`, `.ProcessingFuncWrapper`, `.ProcessingClassWrapper`, `.PipelineConnector`, `.Pipeline`, `.Algorithm` | `medusa.core.pipeline` |
| `medusa.components.Dataset` | `medusa.ml.dataset` |
| `medusa.components.ThreadWithReturnValue` | `medusa.core.utils` |
| `medusa.utils` | `medusa.core.utils` |
| `medusa.settings_schema.SettingsTree` | `medusa.core.settings_tree` |
| `medusa.settings_schema.SettingsTreeWidget`, `.TreeViewer`, `.TextToTreeItem` | `medusa.widgets.settings_tree.*` (kept in Kernel under the `[widgets]` extra; reusable beyond Platform — e.g. Analyzer) |
| `medusa.dataio.schema` | `medusa.core.schema` (populated per K6 — Pydantic v2 model of the serialized Recording dict) |
| `medusa.dataio.compatibility` | `medusa.core.compatibility` |
| `medusa.dataio.*` (other) | `medusa.core.*` (case by case) |
| `medusa.meeg.*` (EEG bits) | `medusa.core.data.biosignals.eeg.*` |
| `medusa.ecg`, `emg`, `eog`, `nirs` | `medusa.core.data.biosignals.{ecg,emg,eog,nirs}.*` |
| `medusa.bci.metrics` | `medusa.pipelines.bci.performance` |
| `medusa.bci.*` (other) | `medusa.pipelines.bci.*` |
| `medusa.plots.head_plots` | `medusa.plots.head_plots` |
| `medusa.plots.generic_plots` | `medusa.plots.generic_plots` |
| `medusa.plots.optimal_subplots` | `medusa.plots.optimal_subplots` |
| `medusa.plots.plot_visualizer` | `medusa.plots.plot_visualizer` |
| `medusa.plots.templates.*` | `medusa.plots.templates.*` |
| `medusa.plots.timeplot` | `medusa.plots.timeplot` |
| `medusa.plots.erp_plots` | `medusa.pipelines.bci.plots.erp` |
| `medusa.plots.mi_plots` | `medusa.pipelines.bci.plots.mi` |
| `medusa.analysis.time_plot` | `medusa.widgets.time_plot` (reusable interactive Qt viewer; stays in Kernel under the `[widgets]` extra) |
| `medusa.analysis.anesthesia_depth_monitoring.*` | `medusa.pipelines.anesthesia.*` |
| `medusa.notify_me` | **removed** (move to `core/utils.py` if used, or delete) |

Why this layout:

- **Readable top level**: 8 folders with clear roles (`core/`, `signal/`, `graph/`, `ml/`, `plots/`, `widgets/`, `pipelines/`). A new dev understands the architecture from `ls medusa/`.
- **Top level groups by data type, not by action**: `signal/` and `graph/` are named after the data they describe; future top-level data types (`image/`, `video/`, `genetics/`) slot in symmetrically. "Processing" / "metrics" are not data types — they are operations *on* data — so they live one level down inside each data-type package. This is the same axis change that motivates renaming `processing/` → `signal/` and rejecting a top-level `metrics/`.
- **Operations vs metrics axis, explicit inside each data-type package**: within `signal/`, signal→signal operations sit flat at the package root (`frequency_filtering`, `transforms`, `segmentation`, …) and signal→scalar/vector metrics live in `signal/metrics/<family>/`. The boundary is concrete: `transforms.psd` (full spectrum) is an operation; `metrics/spectral/band_power` (scalar) is a metric. Nesting `metrics/` inside `signal/` (rather than at the top level) keeps the data-type axis clean and lets `graph/` host its own metrics on its own terms — graph metrics operate on connectivity matrices, not signals, and a top-level `metrics/` would have forced an awkward `metrics/{univariate, connectivity, graph}` mix.
- **`graph/` is flat for now, with documented promotion criteria**: 12 metric files plus 1 op (`surrogate_graph.py`) is too small to justify the `metrics/<family>/` + `<ops>/` split today. When graph operations grow (graph filtering, transforms, projections) or the metric count rises substantially, `graph/` is promoted to mirror `signal/`'s shape. Decision and trigger documented in the tree itself, not buried in a TODO.
- **`__init__.py` re-export contract**: each metric subcategory (`signal/metrics/spectral/__init__.py`, etc.) re-exports its public functions, so consumers can import either `from medusa.signal.metrics.spectral import band_power` (via the package) or `from medusa.signal.metrics.spectral.band_power import band_power` (via the file). `signal/metrics/__init__.py` re-exports the four families, allowing `import medusa.signal.metrics as m; m.spectral.band_power(...)`. Same pattern in `graph/__init__.py`. K5 (`__all__`) formalizes which symbols are public; the re-export pattern just makes the public ones discoverable.
- **`ml/` instead of `models/`**: the folder hosts trainable models *plus* training utilities (`classification.py`), datasets (`dataset.py`), HP search (`optimization.py`), and PyTorch glue (`pytorch_utils.py`). "Models" understates the scope; "ml" is the honest umbrella and matches the standard Python convention (`sklearn`, `torch.nn`, …).
- **`core/` is the foundational layer**: it gathers persistence (`serialization.py`), the on-disk format contract (`schema.py` + `compatibility.py`, K6), the pipeline abstractions (`pipeline.py`), runtime configuration trees (`settings_tree.py`), and the runtime data type hierarchy (`core/data/`). Everything else in the package (`signal/`, `graph/`, `ml/`, `plots/`, `widgets/`, `pipelines/`) depends on `core/` and on nothing else at the same level.
- **`core/data/` centralizes the runtime type system**: `Recording`, `ExperimentData`, biosignal modalities, and the new first-class `Events` class live together, parented by abstract bases. Today these are scattered across `components.py` (`Recording`, `ExperimentData`, `BiosignalData`), `meeg.py` (`EEG`), and loose modality files (`ecg.py`, `emg.py`, `eog.py`, `nirs.py`); `Events` does not exist as a type at all. The new layout co-locates the domain hierarchy in one place and leaves a clean home for future `Image`, `Video`, `Genetics`, etc.
- **`components.py` is gone**: the old monolithic file (~1500 lines, 13+ classes spanning serialization, recording, pipeline, ML dataset, threading) is split by concern across `core/serialization.py`, `core/data/recording.py`, `core/data/experiment.py`, `core/data/biosignals/base.py`, `core/pipeline.py`, plus `ml/dataset.py` and `core/utils.py` for the bits that did not belong in `core/` at all. Each module has a single responsibility and a sensible size; the public surface is preserved via re-exports in `core/__init__.py` if useful.
- **`schema.py` ≠ `settings_tree.py` ≠ `core/data/`**: three distinct concerns, three distinct locations. `schema.py` is the *versioned on-disk format contract* for a serialized Recording (Pydantic model + version int). `settings_tree.py` is a *runtime configuration tree* with metadata (defaults, ranges, options) used by Platform's settings UI. `core/data/` is the *runtime domain type hierarchy* (Recording / ExperimentData / Events / biosignals as Python classes). They could all be called "schema" loosely; using three distinct names prevents the kitchen-sink pattern that the `components.py` split is undoing.
- **3 explicit levels in the filesystem** (not just in docs): `core/` → `signal/`, `graph/`, `ml/`, `plots/` → `pipelines/<domain>/`.
- **Reusable vs. domain-specific**: distinction documented and reflected in the tree. The litmus test for placing new code: *does this make sense outside a specific paradigm/dataset?* If yes → reusable layer. If no → `pipelines/<domain>/`. This also applies to plots: `head_plots` / `generic_plots` / `timeplot` are reusable and stay in `plots/`; `erp_plots` / `mi_plots` are BCI-specific and live under `pipelines/bci/plots/`.
- **`pipelines/` as a single umbrella for domains**: avoids letting `bci/` (and future `anesthesia/`, `sleep/`, `cognitive/`) drift into top-level siblings of the reusable layers, which would conflate "library layers" with "application areas" — exactly the readability problem of the 1.x layout in a different shape. The existing `analysis/anesthesia_depth_monitoring/` becomes `pipelines/anesthesia/` — concrete proof that the structure already has real domain siblings to absorb.
- **`Signal` as the time-series base, biosignals as a modality grouping**: the runtime hierarchy is `SerializableComponent` → `Signal` → `EEG` / `ECG` / `EMG` / `EOG` / `NIRS`. The old monolithic `BiosignalData` wrapper is dropped: what was generic (sample rate, channels, time axis) lives in `Signal`; what was modality-specific (channel set, montage, reference scheme, sensor coordinates, lead definitions, optode montages, …) lives in each modality class. Sibling non-biosignal time-series types (audio stimuli, environmental data, simulation outputs) inherit from `Signal` directly and would live as `core/data/<type>/` peers of `biosignals/`.
- **Biosignals consolidated and symmetric**: EEG and ECG/EMG/EOG/NIRS are all biosignals — no reason for the old `meeg/` to be a structured subpackage while the others sit as loose files. They all live under `core/data/biosignals/<modality>/`, each as its own subpackage. Future modalities — including MEG, when it becomes a real requirement — slot in as further siblings (`core/data/biosignals/meg/`, …) without disturbing the rest. The "M/EEG" umbrella is dropped because medusa's processing functions are array-level (see AGENTS.md §2.bis), so a unified container would only flatten each modality's schema without any compensating gain at the function layer. Even if today ECG is a single file, the modality-as-subpackage shape leaves room to grow without another reorg. Mirrors NeuroKit2 / BioSPPy conventions.
- **`plots/` is reusable-only and Qt-free**: the 1.x `medusa/plots/` mixed reusable plots (`head_plots`, `generic_plots`, `optimal_subplots`, `timeplot`) with domain-specific ones (`erp_plots`, `mi_plots`). The new layout splits them along the reusable / domain-specific line: reusable in `plots/` (matplotlib only), domain in `pipelines/<domain>/plots/`. Qt-based interactive viewers (such as the former `analysis/time_plot/`) move to `widgets/time_plot/` instead — `plots/` stays free of `PySide6` so it is importable in headless environments without any extra.
- **`widgets/` isolates every PySide6 dependency in Kernel**: the only place in the package that imports Qt. Hosts the Qt counterpart of `core.settings_tree.SettingsTree` (was `settings_schema.{SettingsTreeWidget, TreeViewer, TextToTreeItem}`) and the interactive time-plot viewer (was `analysis/time_plot/`). Both are reusable beyond Platform — Analyzer is the next obvious consumer (e.g. it can embed the same settings UI for its conversion pipelines and the same interactive viewer for inspection of recordings). Keeping them in Kernel avoids forcing Analyzer to depend on Platform just to reuse a tree widget. Gated behind the `[widgets]` extra so that headless installs do not pull `PySide6` (see K3).
- **Base ready for extras**: `widgets/` and `ml/deep_learning` isolated → `pip install medusa-kernel` stays light, optionals on demand (see K3). `plots/` and `matplotlib` are core (see K3).
- **Room to grow**: future domain pipelines (`pipelines/sleep/`, `pipelines/cognitive/`, …) come in as siblings of `pipelines/bci/` and `pipelines/anesthesia/` without touching the rest of the tree.

**Required changes**

- [ ] Branch `2.0` created in this repo from `main` (Kernel 1.4.x).
- [ ] Create the new tree empty (no code yet) on a `2.0` branch, verify imports.
- [ ] Create `medusa/widgets/` with `settings_tree/` and `time_plot/` subpackages; relocate `settings_schema.{SettingsTreeWidget, TreeViewer, TextToTreeItem}` to `widgets/settings_tree/` and `analysis/time_plot/` to `widgets/time_plot/`. Guarantee that nothing outside `medusa/widgets/` imports `PySide6`.
- [ ] Add `core/data/signal.py` with the `Signal` abstract base (and `CustomSignal`); migrate biosignal modality classes (`EEG`/`ECG`/`EMG`/`EOG`/`NIRS`) to inherit from `Signal` directly. Drop the former `BiosignalData` base.
- [ ] Add `core/data/events.py` with the new first-class `Events` type; refactor `BiosignalData`/`ExperimentData` consumers that hold loose event arrays to use it.
- [ ] Move modules per the migration mapping. One commit per top-level group (`core/`, `signal/`, `graph/`, `ml/`, `plots/`, `widgets/`, `pipelines/`) for reviewability.
- [ ] Update internal imports (within the package).
- [ ] Update `setup.py` / future `pyproject.toml` package data declarations (TSVs, `.ui` files and icons under `widgets/`, etc.).
- [ ] Coordinated PRs in `medusa-platform`, `medusa-analyzer`, `medusa-tutorials`, and apps to update imports — Platform now imports settings widgets from `medusa.widgets.settings_tree` instead of carrying its own copy.
- [ ] Update `medusa-docs/kernel/2.0/` API reference.

---

## K2 — `pyproject.toml` and reproducible environment

**Problem**

- `setup.py` is the only packaging metadata. No `pyproject.toml`, no PEP 621, no lockfile.
- Consequences: dependabot and modern tooling can't introspect; build backends are pinned to setuptools' classic flow; no canonical lockfile shared between dev, CI, and Platform's installer.

**Proposed solution**

- Migrate to `pyproject.toml` with PEP 621 metadata.
- Adopt **`uv`** (fast, modern, native lockfile) or `hatch` as the build backend. `uv` is the default unless explicitly vetoed.
- Lockfile (`uv.lock`) committed. Single source of truth for CI and for Platform's installer.
- Move `keywords`, `classifiers`, `author`, `description`, `license` from `setup.py` to `[project]`.
- `package_data` (TSVs in `meeg/`, etc.) declared in `[tool.hatch.build]` or `[tool.setuptools.package-data]` depending on backend.

**Required changes**

- [x] Create `pyproject.toml` with PEP 621 metadata.
- [x] Choose build backend (`uv` vs `hatch`) → **`hatchling` as PEP 517 backend, `uv` as project / lockfile manager**. Rationale: `uv` does not yet have a mature library build backend; `hatchling` is the de facto modern standard and integrates seamlessly with `uv`. `uv` owns env + lockfile + publish flow.
- [x] Move `install_requires` to `[project.dependencies]` (and `PySide6` / `torch` to `[project.optional-dependencies]` per K3).
- [x] Generate and commit lockfile (`uv.lock`, 80 packages resolved).
- [x] Add minimal CI workflow (`.github/workflows/tests.yml`, single Linux job) — full multi-OS matrix deferred to K7.
- [x] Mark `setup.py` as deprecated (kept temporarily as fallback; `DeprecationWarning` raised on import). Remove after a full publish cycle on `pyproject.toml`.
- [x] Adapt the existing release workflow → replaced `.github/workflows/python-publish.yml` (Twine + API token + `python -m build`) with `.github/workflows/publish.yml` (`uv build` + `pypa/gh-action-pypi-publish` via **PyPI Trusted Publishing / OIDC**, no API token to rotate). Two jobs: TestPyPI on `workflow_dispatch`, PyPI on `release: published`. Tag-vs-pyproject version check fails fast on mismatch.
- [ ] Configure the trusted publisher entries on PyPI and TestPyPI (one-time setup; pending account: project `medusa-kernel`, environments `pypi` and `testpypi`, workflow `publish.yml`).
- [ ] Verify a full release cycle on TestPyPI (`Actions → publish → Run workflow → target=testpypi`) before deleting `setup.py`.
- [ ] Delete `setup.py` once the release cycle is verified.

**Open questions K2** *(resolved)*

- ~~`uv` or `hatch` as build backend?~~ → **`hatchling` (build) + `uv` (project / lockfile / publish)**.
- ~~Republish 1.4.x with pyproject as an intermediate step, or jump directly to 2.0?~~ → **Direct jump to 2.0**. Republishing 1.4.x with pyproject would add two no-value releases for users (same API, same functionality) and force maintaining the packaging change in two branches. `developers` (1.4.x) keeps `setup.py`; `2.0` branch is the only one on `pyproject.toml`.

---

## K3 — Optional extras

**Problem**

`PySide6` is in `install_requires` (~200 MB of Qt) and PyTorch ships in the same bucket as everything else. Headless users pulling DL toolkits they do not need; users who only want filtering and metrics still pay the GUI cost.

`matplotlib`, on the other hand, is small, pure-Python-friendly, already an effective transitive dependency of the scientific stack we ship, and used pervasively across `plots/` and several `pipelines/` modules. Treating it as optional would force every plotting helper into late-import boilerplate and every consumer (Analyzer, tutorials, notebooks) into installing an extra they will always want. So `matplotlib` stays a **core** dependency.

Note (correction from earlier draft): all DL in Kernel is **PyTorch only** (`deep_learning_models.py` requires `MEDUSA_TORCH_INTEGRATION=1` set by `pytorch_integration.py`). There is no Keras / TensorFlow path. So a `[deep]` extra is misleading; the only DL extra is `[torch]`.

**Proposed solution**

Plan of extras in `[project.optional-dependencies]`:

| Extra | Content | What requires it |
|---|---|---|
| `widgets` | `PySide6` | `medusa.widgets.*` (settings tree widget, interactive time-plot viewer) |
| `torch` | `torch` | `medusa.ml.deep_learning` (uses `medusa.ml.pytorch_utils` for GPU config) |
| `dev` | `pytest`, `pytest-cov`, `ruff`, `mypy` | tests / CI |
| `all` | everything above | convenience |

`matplotlib` is **not** an extra — it lives in `[project.dependencies]` alongside NumPy / SciPy / scikit-learn. `medusa.plots.*` therefore works out of the box on `pip install medusa-kernel`.

`medusa.widgets.*` is the only place in the package that imports `PySide6`. Importing `medusa`, `medusa.signal`, `medusa.plots`, `medusa.ml`, etc. must work without the `[widgets]` extra. Enforced by:
- A CI smoke test that installs Kernel without extras and imports every public top-level subpackage except `widgets`.
- A guard in `widgets/__init__.py` that raises a clear `ImportError` (`"PySide6 not available. Install with: pip install medusa-kernel[widgets]"`) instead of the cryptic `ModuleNotFoundError: No module named 'PySide6'`.

`ml/pytorch_utils.py` itself is part of core (no `torch` import at module load — it does `try: import torch` and exposes `TorchExtrasNotInstalled` to raise on use). This way `import medusa.ml` works without `torch`, and only `medusa.ml.deep_learning` raises if the extra is missing.

`ml/classification.py` and `ml/optimization.py` depend on **scikit-learn + scipy**, which stay as core dependencies (they are already required throughout Kernel). No new extra.

**Required changes**

- [ ] Audit imports: `PySide6`, `torch`, `bson`, `dill`. Identify where each is really needed. (`matplotlib` is core; no audit needed.)
- [ ] Confirm `PySide6` imports are confined to `medusa/widgets/`. If any other module imports it, refactor or relocate.
- [ ] Add a guard in `widgets/__init__.py` that surfaces a helpful error if `PySide6` is missing.
- [ ] Verify `pytorch_utils.py` already does the right thing (graceful fallback): yes, it already sets `MEDUSA_TORCH_INTEGRATION=0` on ImportError. Keep that pattern.
- [ ] Clear error message when functionality is used without its extra: `"PySide6 not available. Install with: pip install medusa-kernel[widgets]"`.
- [ ] Add CI smoke test: `pip install .` (no extras) + `python -c "import medusa, medusa.signal, medusa.plots, medusa.ml, medusa.graph, medusa.pipelines.bci"`.
- [ ] Review whether `bson` and `dill` are really needed in core or are legacy.

**Open questions K3**

- TF/Torch version pinning policy — strict pin or wide range? PyTorch is notoriously fussy with CUDA versions.
- Do we want a `[cuda]` hint in docs (not a real extra, since CUDA wheels are picked up automatically by torch's index URL) or rely on the user installing torch themselves?

---

## K4 — Real cross-platform (and native C extensions for hot metrics)

**Problem**

- `medusa/signal_metrics/computeLZC.dll` is a pre-built Windows DLL: confines LZC to Windows and breaks the cross-platform contract. In the new layout it would sit under `signal/metrics/complexity/`.
- LZC is not the only metric that benefits from a native implementation. **Sample entropy** (`sample_entropy.py`), multiscale entropy, and connectivity metrics with O(N²)–O(N³) inner loops are bottlenecks in pure Python/NumPy and would benefit from the same treatment. Today the project lacks a build system to ship native code at all — every hot path is either pure Python (slow) or a Windows-only binary (broken cross-platform).
- Other places may have Windows assumptions (hardcoded paths, `os.sep`, shell-specific `subprocess`). Audit pending.

**Proposed solution**

Two coordinated subgoals: (a) make Kernel actually cross-platform, (b) introduce a sustainable native-extension story so we can speed up hot metrics on all OSes.

**a) General cross-platform hygiene**
- Audit `os.sep`, paths with `\`, `subprocess(shell=True)`, DLL/PYD imports, anything Windows-specific.
- Migrate every path to `pathlib.Path`.

**b) Native extensions in C, built per-OS via `cibuildwheel`**

Goal: a single source-of-truth C implementation per metric, compiled into wheels for Linux/Windows/macOS × Python 3.10–3.13 in CI, with a pure-Python fallback if the user installs from sdist on an unsupported platform.

Recommended stack (minimal, modern, well-maintained):

- **Language: C (not C++)**. Plain C is enough for the algorithms in question (LZC, sample entropy, MSE, connectivity inner loops); avoids C++ ABI headaches.
- **Bindings: pybind11 *or* nanobind, *or* plain CPython C-API via `setuptools`/`meson-python`**. For numerical kernels that take a NumPy array and return a scalar/array, the cleanest path is:
  - **Cython** — most familiar, mature, integrates with NumPy via typed memoryviews. Recommended default for this repo unless someone has a strong preference.
  - Alternative: **nanobind** (modern, lightweight successor of pybind11; smaller wheels, faster build) if we want a C++ flavor.
- **Build backend: `meson-python` or `setuptools` with `Cython`**. `meson-python` is what SciPy uses; cleaner for multi-extension projects but more setup. For a handful of extensions, `setuptools + Cython` is simpler and enough.
- **Wheel publishing: `cibuildwheel`** in CI (GitHub Actions). Standard, used by most scientific-Python projects. Builds wheels for `manylinux`, `windows`, `macos` (x86_64 + arm64) automatically.
- **Pure-Python fallback** for each native module: same algorithm in Python so that `pip install medusa-kernel` from sdist on, say, FreeBSD still works (just slower). At import time:
  ```python
  try:
      from medusa.signal.metrics.complexity._lzc import lempelziv_complexity
  except ImportError:
      from medusa.signal.metrics.complexity._lzc_py import lempelziv_complexity
  ```

Layout for native modules within the new tree:

```
signal/metrics/complexity/
├── lempelziv_complexity.py        ← public API; tries _lzc, falls back to _lzc_py
├── _lzc.pyx                       ← Cython source (or _lzc.c for raw C)
├── _lzc_py.py                     ← pure-Python fallback (also used as oracle in tests)
├── sample_entropy.py              ← same pattern
├── _sampen.pyx
├── _sampen_py.py
└── ...
```

**Required changes**

- [ ] Audit script for OS-specific patterns (paths, subprocess, DLL imports). One commit fixing each category.
- [ ] Migrate paths to `pathlib`.
- [ ] Choose extension stack: **Cython + setuptools + cibuildwheel** (default proposal) vs `meson-python` vs `nanobind`. Decide based on the team's familiarity.
- [ ] Set up `cibuildwheel` in `.github/workflows/python-publish.yml` (and the test workflow from K7) to produce wheels for the matrix `linux × windows × macos × py3.10–3.13`.
- [ ] Port `computeLZC.dll` → cross-platform Cython/C source. Validate numerical equivalence against the current Windows DLL on a fixed test set.
- [ ] Profile candidates for the same treatment: **`sample_entropy`**, `multiscale_entropy`, the heavy connectivity metrics (PLV/wPLI loops). Add to the queue, prioritized by measured speedup vs. implementation cost.
- [ ] Implement pure-Python fallback for each native module (also useful as test oracle).
- [ ] Multi-OS CI validation in K7's matrix (the same workflow that runs tests should also build wheels and verify they import correctly).

**Open questions K4**

- **Cython vs nanobind vs raw CPython API?** Vote: **Cython** for this repo — most accessible to scientific-Python contributors, mature NumPy support via typed memoryviews, low ceremony for the kind of inner-loop-heavy code we have. Nanobind is great but oriented to C++ codebases.
- **`meson-python` vs `setuptools`?** With ≤10 extensions, `setuptools` is simpler. Reopen if the count grows or if we hit setuptools' limits with multi-extension layouts.
- Any other algorithm with a hidden native binary today? Audit pending alongside the Windows-assumption sweep.
- Tooling for benchmarking before/after each port: `pytest-benchmark` (proposed in K7).

---

## K5 — Explicit public API

**Problem**

Today everything importable from `medusa.<module>` is *de facto* public. No distinction between what is committed not to break and what is internal. Result: any internal rename can break Platform apps or notebooks.

**Proposed solution**

- Mark public API with `__all__` in each module.
- Document the public API in a dedicated page in `medusa-docs/kernel/2.0/` (see global E5).
- Symbols not listed in `__all__` are considered internal: rename / delete freely, no breaking change.
- Documented deprecation policy: `DeprecationWarning` for one minor; removal in the next.
- Take the v2.0 opportunity to define `__all__` from scratch without dragging anything we do not want to commit publicly.

**Required changes**

- [ ] Define `__all__` per module, with explicit policy for what is public.
- [ ] Audit consumers (Platform, Analyzer, apps, tutorials) for usage of currently-public-but-now-private symbols. Plan migration.
- [ ] Document public API page in `medusa-docs/kernel/2.0/`.
- [ ] Document deprecation policy publicly.

**Open questions K5**

- Establish `_internal/` as a sub-package, or is the convention `__all__` + leading underscore enough?
- Is `pipelines/` a fully public API, or only the top-level pipeline classes per domain?

---

## K6 — Truly versioned recording schema

**Problem**

- `dataio/schema.py` and `compatibility.py` already exist but versioning is minimal. Format changes are usually ad-hoc.
- 3 consumers with implicit dependency: Platform writes, Analyzer reads, Kernel defines.

**Proposed solution**

- Version the recording schema with an explicit `schema_version: int` field (incrementable).
- Kernel v2.0 = `schema_version: 2`. Define the rules for what constitutes a bump.
- `compatibility.py`: implement a `migrations: dict[int, Callable]` registry with `n → n+1` migrations. Reading `v_n` applies chained migrations up to current.
- Cross-version read tests: for each `schema_version`, fixture with example recording + assertion that it reads identically.
- Document the format as a public contract (referenceable from Platform and Analyzer).

**Required changes**

- [ ] Add `schema_version` field; freeze v1.
- [ ] Implement migration registry.
- [ ] Write fixtures for each supported version.
- [ ] Add cross-version read tests in CI.
- [ ] Public documentation page in `medusa-docs`.

**Open questions K6**

- Does v2.0 of Kernel break the schema or extend it compatibly? If broken, define the v1 → v2 migration.
- Maintain writing of old schemas for interoperability? Default: **no**, read only.
- Schema in JSON Schema, Pydantic, or dataclass + custom validator? Vote: Pydantic v2 (fast, declarative, JSON Schema export compatible).

---

## K7 — Multi-OS CI and real tests

**Problem**

- `.github/workflows/python-publish.yml` only handles PyPI release. **No test CI yet.**
- Existing tests (`tests/test_components.py`, `test_ecg.py`, `test_signal_generators.py`, `test_transforms.py`) cover under 5%.
- Zero tests for `pipelines/bci/`, `signal/metrics/connectivity/`, `graph/`, `ml/`, most of `signal/`.

**Proposed solution**

- Add `.github/workflows/test.yml`:
  - Matrix: `python-version: [3.13]` × `os: [ubuntu-latest, windows-latest, macos-latest]` (extend with newer Python releases as they ship).
  - `uv sync --extra dev`, then `pytest --cov=medusa --cov-report=xml`.
  - Upload coverage to Codecov or equivalent.
- Priority tests to add post-reorg (when paths are stable):
  - Recording schema round-trip cross-version (K6).
  - Tests for `signal/metrics/` (at least band_power from `spectral/`, sample_entropy and LZC from `complexity/`).
  - Basic tests for each `pipelines/bci/*_spellers.py` with synthetic signals from `signal/generators.py`.
  - Smoke tests of `ml/deep_learning` and `ml/pytorch_utils` (only if extra is installed).
- Coverage gate: start at 30%, raise gradually.
- Lint: `ruff` with minimal config.

**Required changes**

- [ ] Workflow file added.
- [ ] Initial test suite expansion (per priority list).
- [ ] Coverage gate documented.
- [ ] Codecov (or alternative) configured.

**Open questions K7**

- Realistic minimum coverage for the initial gate — 30% or 50%?
- DL model tests in CI? Heavy; maybe only on self-hosted runners or nightly.
- pytest-benchmark for critical metrics (LZC, connectivity) to detect performance regressions?

---

## K8 — License

**Problem**

`LICENSE`: **CC BY-NC-ND 2.0**. Incompatible with standard OSS. Blocks contributions, clinical/industrial use, forks. Mirror of `../TODO.md` E7.

**Proposed solution** (legal decision)

- Resolve before the 2.0 release (good moment: a major bump justifies a license change).
- Candidates: **Apache 2.0** (recommended — permissive + explicit patent clause, good for clinical environments) or MIT.
- Contact historical contributors for a CLA / relicensing agreement.
- Update `LICENSE` and metadata in `pyproject.toml`.

**Required changes**

- [ ] Legal decision with CIBER-BBN / UVa.
- [ ] Identify external contributors.
- [ ] Update `LICENSE`.
- [ ] Update `pyproject.toml` metadata.
- [ ] Public announcement.

**Dependencies**

- Mirror of `../TODO.md` E7.

**Open questions K8**

- CIBER-BBN / UVa stance on permissive licenses?
- External contributors whose rights must be considered?

---

## Suggested execution order

Dependencies between targets:

```
K8 (license) ────────────────── independent legal decision; can run in parallel

K2 (pyproject) ──┐
                 ├──> K3 (extras) ──> K1 (reorg) ──> K5 (__all__) ──> K6 (schema) ──┐
K4 (cross-plat)  ┘                                                                   │
                                                                                     │
K7 (multi-OS CI) ─────────────────── starts after K2, refined at every step ────────┤
                                                                                     │
                                                                                     ▼
                                                                              Release 2.0.0
```

Reasoning:
1. **K2 (pyproject)** first because it unblocks the rest: extras, deps, modern build.
2. **K3 (extras)** and **K4 (cross-platform)** in parallel, before the reorg, so the reorg can place each thing in its definitive location.
3. **K1 (reorg)** is the massive change. Do it in a single large PR with the mapping clear; coordinate with Platform/Analyzer/apps that need to update imports (separate commits per repo).
4. **K5 (`__all__`)** as a closing of the reorg: once everything is in place, mark what is public.
5. **K6 (schema)** independent of the rest but worth closing before release so 2.0 includes `schema_version: 2`.
6. **K7 (CI)** introduced from K2 and refined; by 2.0 release it must run on the 3 OSes.
7. **K8 (license)** in parallel, closing before the 2.0 tag.

---

## Coordination with other components

The 2.0 bump breaks **all** Kernel consumers. Coordinate:

- `medusa-platform`: update all imports + bump `requirements.txt` to `medusa-kernel>=2,<3` (with the `[widgets]` extra, since Platform consumes `medusa.widgets.settings_tree`). Also requires updating apps (at least the templates in `src/templates/`).
- `medusa-platform/src/accounts/*/apps/*`: each app must update imports. Good moment to introduce the versioned manifest from `../TODO.md` E2 with `requires_kernel: ">=2,<3"`.
- `medusa-analyzer`: update imports + adopt `core/` (resolves the duplication with `medusa-analyzer/data_loader/` from `../TODO.md` E9).
- `medusa-tutorials`: notebooks rewrite imports. Good smoke test of the new layout.
- `medusa-docs`: new `kernel/2.0/` version. Good time for K8 (migration to MkDocs+mike, `../TODO.md` E5).
- `medusa-installer`: change in Platform's `requirements.txt` → verify the installer installs the new pinned version correctly (`../TODO.md` E4.a, end-to-end test).

---

## Next steps

1. **Validate this TODO**: mark K1–K8 as agreed/pending/discarded.
2. **K1 layout closed**: tree and migration mapping are agreed. Execution proceeds per K1's "Required changes" checklist.
3. **Mini-RFC per target** before execution: scope, repos affected, migration plan, success criterion.
4. **Branch `2.0`** in this repo from the start, with `1.4.x` maintained on `main` during the transition for hotfixes.
