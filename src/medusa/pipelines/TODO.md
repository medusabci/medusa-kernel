# `medusa.pipelines` 2.0 — Refactor Plan

Clean break, no 1.x back-compat. Old code archived under `bci/_old/` (reuse where
useful); build the new structure fresh.

## Architecture — four abstraction levels

1. **Containers** (`SerializableComponent`): data classes + stimulation `encoding`;
   live in `Recording.experiment`. No behavior.
2. **Decoding pipelines** (class-based, trainable, persistable):
   `DecodingPipeline` (L1: `fit`/`predict` → scores, all paradigms) +
   `CommandDecoder` (L2: scores → commands, VEP spellers only).
3. **Analysis pipelines**: per-paradigm `analysis.py` **scripts** — plain functions
   (no base class, no `fit`/`predict`, no persistence) → arrays/stats for plots.
4. **Pure utilities** (free functions): codebook generators, cumulative accumulators
   (`bwr_command_scores`/`tm_command_scores`), `select_commands`, metrics (`itr`, accuracies).

Rendering lives in `medusa/plots/*` (arrays + `ax`, no signal processing). No `Dataset`
class — train over `Iterable[Recording]` (lazy); consistency = an overridable
per-`DecodingPipeline` `check_consistency(recording)`.

## `bci/` taxonomy — per-application DOMAIN packages (updated 2026-07-09)

Every `bci/` package is a decoding **strategy/application named for what it is** — never
a mode bucket. A pipeline is a **thin, opinionated preset** (ready-to-use workflow with
sensible defaults), NOT the reusable abstraction; generalization lives BELOW the pipeline
(`signal`/`ml` free functions). Uniform, use-oriented naming (matches the `evoked → vep_spellers`
rename). `endogenous/` merge + `EndogenousData` are DROPPED.

- **`vep_spellers/`** — stimulus-locked command spelling (c-VEP/SSVEP/ERP-P300): codebook
  (`encoding.py`) + L2 `VEPCommandDecoder`. Bespoke speller machinery earns its package.
- **`motor_decoding/`** — motor imagery + motor execution presets (`MICSPLDAPipeline`,
  deep `MIEEGSym`/`MIEEGInception` later). `analysis.py` = ERD/ERS (the one motor-specific bit).
- **`neurofeedback/`** — continuous baseline-referenced feature feedback (`PowerNFTPipeline`,
  `ConnectivityNFTPipeline`). L1-only, no labels/classifier; fit == unsupervised calibration.
- future: `mental_state_decoding/`, … — add when a domain arrives; start COARSE (arithmetic/
  attention/fatigue are pipelines inside, not a package each).

**Shared trial contract (not a workflow — define once, import everywhere):** `bci/trial_events.py`
= `TRIAL_EVENT_COLUMNS {trial_idx,label}` (one Events row per trial, onset=start) +
`validate_trial_events` + `trial_arrays`/`trial_labels`. Trial paradigms (MI/ME/mental-tasks)
need NO consumed data class. Degrees-of-freedom stays a pipeline **property** (`discrete|continuous`),
NOT a directory. **Several pipelines → a `decoding/` subpackage, one short module per pipeline** +
`_common.py`; deep/torch pipeline in its own module, package torch-free via PEP 562 `__getattr__`.
**Duplication policy:** accept thin preset duplication; extract a shared decode chain into a free
function only on the rule of three.

## Key decisions

- No composition layer (`Algorithm`/`ProcessingMethod`/`Pipeline`-graph/`*Wrapper`);
  everything orchestrates `signal/` + `ml/` free functions.
- **Config = `SettingsTree` via a shared `Configurable` mixin (core/settings_tree.py).** A class
  declares its schema once in `default_settings()`; the instance owns a *live editable*
  tree (`dec.settings`, the GUI binds in place — no get-schema-then-recreate), read as
  `dec.cfg`. **One** construct (`Cls(settings=None, **overrides)` — tree / dict / kwargs,
  validated), **one** `save`, **one** polymorphic `DecodingPipeline.load(path)`. Dropped
  `from_settings` + `load_decoding_pipeline` (folded in).
- `DecodingPipeline(Configurable, PickleableComponent)`: `save` = `HIGHEST_PROTOCOL` +
  class tag; subclass overrides `to_pickleable_obj`/`from_pickleable_obj` to bundle
  `settings.to_dict()` **+** fitted state (nested torch via `pack_pickleable` → config +
  CPU `state_dict`). **Core change:** `PickleableComponent.save` default →
  `HIGHEST_PROTOCOL`; add `pack/unpack` to `core/serialization.py`;
  `_BaseTorchEstimator.__getstate__/__setstate__`.
- Symmetric offline/online: `fit(recordings)`/`predict(recording)` +
  `fit_online`/`predict_online` (online adaptation = future). `DecodingPipeline` owns the
  causal filter state **and the family-specific cross-cycle accumulation** — its `predict`
  returns a **cumulative `(n_cycles, n_commands)` score matrix** (row `k` = decision score
  after `k` cycles). `CommandDecoder` is then a **pure, paradigm-agnostic selector** (argmax
  over available commands + early-stopping); no codes, no per-family logic in L2.
- **One pipeline class per (strategy, classifier)** — `BWRLDAPipeline` (BWR) +
  `TMCCAPipeline` (template matching) now, `BwrEEGInceptionPipeline`/`TRCAPipeline` later —
  the model choice cascades into preprocessing, so no single class with a `clf=` flag, and
  **no intermediate `TMPipeline` base**: each is a direct `DecodingPipeline` (their feature
  paths share nothing — BWR epochs per frame + trains LDA; TM takes one multichannel segment
  per cycle). `TMCCAPipeline` (naming = TM paradigm + CCA method + Pipeline) has a `reference`
  mode, each a self-contained scoring strategy: `harmonics` (calibration-free SSVEP, CCA vs
  sin/cos) and `template` (calibrated; learned spatial filter + 1-D `|Pearson|` vs shifted
  templates — plain multichannel IT-CCA overfits). Shared only as free functions: `bwr_labels`,
  the cycle-order/bit-onset helpers, the cumulative accumulators (`bwr_command_scores` /
  `tm_command_scores`), and the Layer-2 `VEPCommandDecoder` / `select_commands`.
- Data = thin `SerializableComponent`s in `Recording.experiment`; onsets + indices →
  `Events`; keep `paradigm_conf` (scenario/layout/stimuli) + `commands_info` (codebook —
  `{uid: CommandInfo}` in memory, **list-of-records on the wire**, `.mat`-safe).
- **VEP spellers unify via a codebook** `(n_commands, n_frames)`:
  `generate_row_col_codebook` (P300), `generate_freq_codebook` (SSVEP), m-seq/Gold
  (c-VEP); `CommandInfo.extra` holds specifics. Two scoring families -- **TM** (CCA/TRCA
  templates; accumulate by coherently averaging cycle segments) and **BWR** (per-frame
  classify → concatenate-then-correlate against the codes) -- each emit the cumulative
  `(n_cycles, n_commands)` matrix that the one paradigm-agnostic `VEPCommandDecoder` selects
  from. `codes` (n_codes, n_frames) + `code_idx` kept for multi-code paradigm headroom.
- **MI + NFT merge** (SMR-neurofeedback *is* MI feedback): one `EndogenousData`, L1-only.

## Surfaces

```text
# Configurable (core.settings_tree) — SettingsTree config: default_settings() schema ; dec.settings (live) ; dec.cfg (values)
# L1  DecodingPipeline(Configurable, PickleableComponent) — EEG -> scores (every paradigm)
fit(recordings) / predict(recording) -> scores   # VEP spellers: cumulative (n_cycles, n_commands) matrix
fit_online(signal, *, onsets=None, labels=None) -> Self ; predict_online(signal, *, onsets=None) -> scores
check_consistency(recording) ; reset() ; save(path) / DecodingPipeline.load(path)  # polymorphic

# L2  VEPCommandDecoder — cumulative (n_cycles, n_commands) scores -> commands (pure selector)
decode(cycle_scores, speller_data, events) -> {selected_commands, selected_commands_per_cycle, scores}   # offline
step(cycle_scores_row, speller_data, trial=0) -> {selected, scores, stop}   # online (L1 owns accumulation; stateless)
# DECISION: keep the plain dicts. A shared DecodingResult dataclass (offline+online+endogenous) was
#   considered and REJECTED: the two returns are disjoint shapes (many trials vs one decision), L1 is a
#   bare ndarray by design, control_state is unimplemented -> a nullable grab-bag, worse than small honest dicts.

# Analysis — plain functions (no class), per paradigm
erp_grand_average(recordings, ...) ; signed_r2_map(...) ; erd_ers(...) ; scp(...)  -> arrays/stats
```

## Package layout

```
pipelines/
  base.py                # DecodingPipeline (builds on core.settings_tree.Configurable) + load_recordings
                         #   + leave_one_recording_out + harmonize_channels
                         #   (Configurable itself lives in core/settings_tree.py)
  bci/
    performance.py       # itr  (generic BCI metric)
    vep_spellers/        # stimulus-locked → L1 (+ L2 where a codebook exists)
      encoding.py        # CommandInfo + codebook generators + plot_codebook
      data.py            # SpellerData (+ SPELLER_EVENT_COLUMNS: trial_idx, cycle_idx)
      decoding/          # one pipeline per module + shared helpers (same public API)
        _common.py       #   private: cycle/onset helpers + freq-filtering (notch + filterbank)
        scores.py        #   bwr_labels + bwr_command_scores + tm_command_scores (pure funcs)
        command_decoder.py #  L2: VEPCommandDecoder + select_commands + accuracies
        bwr_lda.py       #   L1: BWRLDAPipeline
        bwr_eeg_inception.py #  L1: BWREEGInceptionPipeline (torch-gated, lazy)
        template_matching.py #  L1: TMCCAPipeline
      analysis.py        # offline analysis SCRIPTS (ERP grand-avg, r², SNR…)
    trial_events.py      # SHARED trial contract: TRIAL_EVENT_COLUMNS {trial_idx,label}
                         #   + validate_trial_events + trial_arrays/trial_labels (no data class)
    motor_decoding/      # MI + motor execution presets (trial decoding → per-trial class scores)
      decoding/          # several pipelines → one short module each + _common.py
        _common.py       #   trial_epochs, log_var, filter builder
        mi_csp_lda.py    #   MICSPLDAPipeline (CSP + rLDA, shallow)
        mi_eeg_inception.py / mi_eegsym.py   # deep (torch-gated; __getattr__ lazy)  [pending]
      analysis.py        # ERD/ERS  [pending]
    neurofeedback/       # continuous baseline-referenced feature feedback (L1-only)  [pending]
      decoding/          #   PowerNFTPipeline, ConnectivityNFTPipeline
      analysis.py        #   baseline / feedback-trace diagnostics
  # future domains (mental_state_decoding/, …) = more sibling packages, same layout, added on demand.
  # future non-BCI siblings (sleep/, anesthesia/) are peers of bci/, with their own axis.
```

Deps strictly down: `pipelines → {core, signal, ml}`; no Qt; torch optional-gated.
Public API = per-module `__all__`.

## Reused as-is (verified `signal/` + `ml/` signatures)

`IIRFilter` (offline `sosfiltfilt` / online `sosfilt`+`zi`), `car`, `LaplacianFilter`,
`CSP` (`.fit(seg,y)`/`.project`/`.to_dict`), `CCA` (`.fit(seg,ref)`/`.r`), `TRCA`
(`.fit`/`.project`), `segment_signal_around_events`, `resample_segments(seg, window_ms,
target_fs)`, `band_power(psd, fs, band)`, `power_spectral_density → (f, psd)`,
`wpli`/`aec`, `graph.degree`; sklearn `LDA`/`CCA`; `TorchClassifier` +
`EEGInception(V2)`/`EEGNet`/`EEGSym`. DL pipeline = `self.clf_ = TorchClassifier(backbone)`
on raw segments, persisted via `pack_pickleable`.

## Migration phases

0. **Unblock.** `base.py` (`DecodingPipeline`, loaders, `harmonize_channels`); core change
   (`PickleableComponent.save` default → `HIGHEST_PROTOCOL`, `pack/unpack` in
   `core/serialization.py`, `_BaseTorchEstimator.__getstate__/__setstate__`); importable
   `bci/` + `vep_spellers/` + `endogenous/` skeleton; `performance.py` (`itr`).
1. **`vep_spellers/`.** *(done: `encoding.py` codebook generators incl. c-VEP m-seq/Gold/random;
   `data.py` `SpellerData`/`CommandInfo` + `trial_available_cmmds`; `Events` = one row per
   cycle, mandatory cols `trial_idx`/`cycle_idx`/`code_idx`; `decoding.py` `BWRLDAPipeline`
   (BWR) + `TMCCAPipeline` (template matching: `reference='harmonics'` calibration-free SSVEP,
   `reference='template'` calibrated SSVEP + c-VEP via learned spatial filter + shifted 1-D
   templates) + `VEPCommandDecoder` +
   `bwr_command_scores`/`tm_command_scores`/`select_commands`/accuracies — L1 emits the
   cumulative `(n_cycles, n_commands)` matrix, L2 is a pure selector; `Configurable`/
   `SettingsTree` config; c-VEP (BWR + template), RCP, synthetic-SSVEP verified e2e @ 100%,
   BWR math fuzz-checked identical to `_old` (`|r|`) up to exact-tie tiebreaks.)*
   deep BWR done: `BWREEGInceptionPipeline` (EEG-Inception v1 **and** v2 in one class —
   `classifier.arch` selects; identical epoch contract so the arch is a config option, not a
   new class) wraps `TorchClassifier`, mirrors `BWRLDAPipeline` (raw non-flattened epochs,
   single band, `pack_pickleable` persistence); torch-gated in `decoding/bwr_eeg_inception.py`,
   re-exported lazily. `decoding.py` was split into a `decoding/` **subpackage** (one pipeline
   per module + `_common`/`scores`/`command_decoder`), same public API.
   **`reference.mode` renamed** (named for what the reference is made of, not the paradigm):
   `harmonics`->`synthetic_harmonics`, `template`->`calibrated_template`, plus the new eCCA mode
   `mixed_harmonics_template`. **eCCA DONE** (was "reference='combined'"): net-new (no `_old`
   ref); per command fuses test-vs-harmonics canonical corr + 3 template correlations via
   `sum(sign(r)*r^2)` (`_ecca_score`), FBCCA weights across sub-bands; calibrated (reuses
   `_learn_templates`, now keeps the multichannel template as the 4th tuple element). Verified on
   synthetic SSVEP (100%/final, beats calibration-free harmonics 88->100).
   **DEFERRED (out of scope this pass, by decision):** `TRCAPipeline` (TRCA spatial filters --
   `_old` `TRCAGoldCodesClassifier` is the c-VEP reference; SSVEP TRCA is net-new;
   `signal.TRCA.fit/project` ready, serialise `.w` manually); `vep_spellers/analysis.py` compute
   (grand-avg / signed-r2 / SNR -- renderers already in `plots/erp.py`, `signal.metrics...signed_r2`
   exists; a design is sketched: `erp_epochs`/`erp_grand_average`/`signed_r2_map`/`ssvep_snr`).
   **multi-level DROPPED** (obsolete: the app updates
   `trial_available_cmmds` per trial; no nested-matrix logic). c-VEP `# TODO: incorrect` was the
   `_old` early-stopping bookkeeping, NOT the scoring core -- new `VEPCommandDecoder` replaces it.
   NOTE new c-VEP template matching deliberately diverges from `_old` (`|Pearson|`+FBCCA vs
   `_old` signed-uncentered-cosine + equal-weight mean) -- lock the NEW behaviour, surface the gap.
   **Step-0 tests DONE** (`tests/pipelines/bci/vep_spellers/`, 49 passing): random-code c-VEP
   decodes e2e (BWR 100%/cycle + TM template 100%); `SpellerData` bson/json/mat round-trips +
   fitted-pipeline save/load; golden units for `bwr_command_scores`/`tm_command_scores`/
   `bwr_labels`/`select_commands`/accuracies/`VEPCommandDecoder` (exact constants + a textbook-
   Pearson oracle); the new-vs-`_old` c-VEP template divergence is pinned + surfaced
   (`_corr1d` |Pearson| vs `_old` signed cosine). `_old` VEP math is now safe to delete.
   **Remaining Step-0 (optional):** m-seq/Gold/SSVEP-harmonics e2e regression guards.
2. **`motor_decoding/` + `neurofeedback/`.** *(DONE 2026-07-09.)*
   - `bci/trial_events.py` shared contract; `bci/_old/` MI converter `mi_recording_to_v2` +
     `examples/motor_decoding_mi_usage.py` (real data: 67.5% fixed split, 76.2% LORO).
   - `motor_decoding/decoding/`: `MICSPLDAPipeline` (CSP+rLDA, shallow, torch-free; **car default
     False** — CAR rank-deficients the CSP covariance), `MIEEGInceptionPipeline` (EEG-Inception
     v1/v2, one class, `arch` setting), `MIEEGSymPipeline` (EEGSym, explicit `hemisphere_pairs`+
     `middle_chs`). Deep two are torch-gated in own modules + `__getattr__` lazy (pkg torch-free).
     Shared `_common.py`: `trial_epochs`, `log_var`, `add_training_settings`, `training_kwargs`.
   - `neurofeedback/decoding/`: `PowerNFTPipeline` (band power/ratio) + `ConnectivityNFTPipeline`
     (wPLI/AEC → global_coupling/strength). L1-only: **fit == unsupervised calibration baseline**,
     **predict == continuous `(n_windows,)` feedback trace**, `reference` mode resolves the old
     commented-out subtraction. No events (self-windowed), no data class. Shared orchestration as
     free fns in `_common.py` (`sliding_windows`/`feature_trace`/`calibrate_baseline`/`check_signal`),
     NO intermediate base. Artifact-band rejection + surface-Laplacian dropped from the old port.
   - All 4 pipelines verified e2e on synthetic data + persistence round-trips; `import
     medusa.pipelines.bci` stays torch-free. **Neurofeedback usage example DROPPED** (low value:
     docs-only, not a test; NFT/MI pipelines have no committed tests anyway -- if robustness is
     wanted, add pipeline tests, not a demo).
     **DONE:** `motor_decoding/analysis.py` -- pure-numpy offline analysis (plain arrays in/out, no
     class/persistence/matplotlib): `motor_trials` (recording->epochs+labels via trial_events +
     trial_epochs), `spectrogram` (fourier/wavelet, onset-relative times), `instantaneous_band_power`
     (bandpass+hilbert envelope), `erd_ers` (Pfurtscheller 100*(A-R)/R, ref_mode classic|trial, time
     axis=-2 so it serves both TF maps and band-power traces), `erd_ers_significance` (percentile
     bootstrap over trials, significant where CI excludes 0), `class_discriminability` (signed r2).
     Builds on signal.transforms (fourier_spectrogram/cwt_spectrogram/hilbert) + signal.metrics
     signed_r2; plus `trial_band_power`, `trial_psd` (per-trial Welch PSD), `discriminability_spectrum`
     (per-freq r2, built on trial_psd), `optimal_band` (subject-specific band from the smoothed |r2|
     profile peak). 50 tests (synthetic
     lateralized-mu-ERD recording) + adversarial review pass (zero-ref guard, 3-D shape guard,
     freq_range/baseline guards, n_boot-vs-alpha warning applied). `examples/motor_decoding_erd_ers_usage.py`
     runs on REAL data (examples/data/mi, 8 runs/80 trials): per-class 3x2 ERD/ERS spectrogram grid +
     band topographies (plots.plot_topography) + r2 discriminability spectrum + optimal band; no MI code
     in plots/. **DONE:** shared filter helper `bci/_filtering.py`
     (`make_filter`/`add_filter_leaves`/`BAND_TYPE_OPTIONS`) -- the `_make_filter`/`_BAND_TYPE_OPTIONS`
     triplication is removed; vep/motor/nft `_common.py` all consume it (motor/nft gained spec
     validation; identical settings schema; the pipeline-facing settings-builder wrappers unchanged).
3. **Analysis + surfaces.** Per-paradigm `analysis.py` scripts; split `_old/plots` → pure
   renderers in `medusa/plots/`. Round-trip tests per data class (bson/json/mat/h5);
   tutorials + mkdocs.
4. **Delete + platform.** Remove `bci/_old/`; migrate medusa-platform call sites to
   `DecodingPipeline`/`CommandDecoder`.

## Delete (from `_old` once ported)

- `Algorithm`/`ProcessingMethod`/`Pipeline`-graph/`*Wrapper` (already gone from core).
- `{cvep,ssvep,erp}_spellers.py`, `*SpellerData`, `*SpellerModel`,
  `LFSR`/`GOLD_CODES`/`SSVEPCodeGenerator` → `vep_spellers/`.
- `mi_paradigms.py` → `motor_decoding/`; `nft_paradigms.py` → `neurofeedback/`. `MIData` dropped
  (trials → `Recording.events` via `trial_events.py`); `NeurofeedbackData` → thin provenance or dropped.
- All `Dataset` subclasses + `ConsistencyChecker` + `track_attributes` +
  `custom_operations_on_recordings`; all `ProcessingMethod` subclasses; `configure/build/is_*`
  lifecycle; CMD/CSD/BWR class prefixes; bespoke DL wrappers → `TorchClassifier`.
- `_old/plots/` → `medusa/plots/` (renderers) + `analysis.py` (compute).
- Cruft: `__int__` typos, `# TODO: incorrect` methods, wrong-paradigm docstrings, stale imports.
