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

## `bci/` taxonomy — `vep_spellers/` vs `endogenous/`

Fork by **signal-generation mode** = the L1-vs-L1+L2 fork:

- **`vep_spellers/`** — stimulus/event-locked (c-VEP, SSVEP, ERP/P300; future ErrP): codebook
  (`encoding.py`) + optional L2 `CommandDecoder`.
- **`endogenous/`** — self-generated (MI + NFT **merged**; future SCP, mental-tasks,
  affective): **L1-only**, no codebook/L2.

Degrees of freedom is an **output property** on the `DecodingPipeline` (score shape +
`discrete|continuous`), not a directory. Neural signature is the **in-package file-split
rule** as `decoding.py` grows (`sensorimotor.py`, `connectivity.py`) — never new dirs.

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
# TODO: unify both returns behind a DecodingResult(selected_commands, scores, control_state, stop, detail)
#       dataclass shared with endogenous L1 — not yet implemented (both return plain dicts today).

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
      decoding.py        # L1 per-model pipelines (BWRLDAPipeline, TMCCAPipeline, …) + L2 VEPCommandDecoder
      analysis.py        # offline analysis SCRIPTS (ERP grand-avg, r², SNR…)
    endogenous/          # self-generated → L1-only  ← MI + NFT merged
      data.py            # EndogenousData (merged MIData + NeurofeedbackData)
      decoding.py        # L1: CSPPipeline, EEGSymPipeline, BandPowerPipeline, ConnectivityPipeline
      analysis.py        # ERD/ERS, SCP, connectivity diagnostics
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
1. **`vep_spellers/`.** *(done: `encoding.py` codebook generators incl. c-VEP m-seq/Gold;
   `data.py` `SpellerData`/`CommandInfo` + `trial_available_cmmds`; `Events` = one row per
   cycle, mandatory cols `trial_idx`/`cycle_idx`/`code_idx`; `decoding.py` `BWRLDAPipeline`
   (BWR) + `TMCCAPipeline` (template matching: `reference='harmonics'` calibration-free SSVEP,
   `reference='template'` calibrated SSVEP + c-VEP via learned spatial filter + shifted 1-D
   templates) + `VEPCommandDecoder` +
   `bwr_command_scores`/`tm_command_scores`/`select_commands`/accuracies — L1 emits the
   cumulative `(n_cycles, n_commands)` matrix, L2 is a pure selector; `Configurable`/
   `SettingsTree` config; c-VEP (BWR + template), RCP, synthetic-SSVEP verified e2e @ 100%,
   BWR math fuzz-checked identical to `_old` (`|r|`) up to exact-tie tiebreaks.)*
   **Pending:** `reference='combined'` (eCCA); `TRCAPipeline` (TRCA spatial filters); deep BWR
   (`BwrEEGInceptionPipeline`); multi-level; **golden-output tests before deleting `_old`
   math** (esp. c-VEP `# TODO: incorrect`).
2. **`endogenous/`.** `EndogenousData`; L1 pipelines (`CSPPipeline`, `EEGSymPipeline`,
   `BandPowerPipeline`, `ConnectivityPipeline`). Test online calibrate→feedback + MI classify.
3. **Analysis + surfaces.** Per-paradigm `analysis.py` scripts; split `_old/plots` → pure
   renderers in `medusa/plots/`. Round-trip tests per data class (bson/json/mat/h5);
   tutorials + mkdocs.
4. **Delete + platform.** Remove `bci/_old/`; migrate medusa-platform call sites to
   `DecodingPipeline`/`CommandDecoder`.

## Delete (from `_old` once ported)

- `Algorithm`/`ProcessingMethod`/`Pipeline`-graph/`*Wrapper` (already gone from core).
- `{cvep,ssvep,erp}_spellers.py`, `*SpellerData`, `*SpellerModel`,
  `LFSR`/`GOLD_CODES`/`SSVEPCodeGenerator` → `vep_spellers/`.
- `mi_paradigms.py` + `nft_paradigms.py`, `MIData` + `NeurofeedbackData` → `endogenous/`.
- All `Dataset` subclasses + `ConsistencyChecker` + `track_attributes` +
  `custom_operations_on_recordings`; all `ProcessingMethod` subclasses; `configure/build/is_*`
  lifecycle; CMD/CSD/BWR class prefixes; bespoke DL wrappers → `TorchClassifier`.
- `_old/plots/` → `medusa/plots/` (renderers) + `analysis.py` (compute).
- Cruft: `__int__` typos, `# TODO: incorrect` methods, wrong-paradigm docstrings, stale imports.
