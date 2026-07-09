# Changelog

All notable changes to **medusa-kernel** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] — Unreleased

medusa-kernel 2.0 is a ground-up rewrite around a BIDS-aligned data model. It is
a major, **breaking** release: code written for the 1.x API will not run
unchanged. Recordings saved with 1.x can still be *read* and migrated to the new
model (see **Legacy compatibility** under *Added* below).

### Added

- **BIDS-aligned data model** (`medusa.core.data`): `Signal`, a typed,
  mixed-channel time-series stream; `ChannelSet` / `Channel` / `Sensor` for
  channel identity, typing and montage; `Events`, a BIDS `events.tsv`-aligned
  timeline; and `Recording` / `BidsInfo` grouping the streams, sidecars, events
  and experiment metadata of a single run.
- **Streaming and persistence**: a `Recorder` for live acquisition with native,
  chunked HDF5 storage, alongside the portable `bson` / `json` / `mat` formats.
- **Legacy compatibility** (`medusa.core.legacy`): a read path for 1.x
  `.rec` / `.cvep` / `.rcp` / `.mi` recordings, plus converters to the 2.0 model
  — `recorder_recording_to_v2` (Recorder-app runs, including the manual
  conditions/events marks), `cvep_recording_to_v2` and `rcp_recording_to_v2`.
- **Reusable GUI widgets** (`medusa.widgets`, PySide6): time-line and
  time-heatmap viewers, an ERP viewer, a recording inspector, a settings-tree
  editor and a figure browser.
- **Deep learning** (`medusa.ml`): PyTorch / PyTorch Lightning estimators and
  reusable network backbones (opt-in — PyTorch is not installed automatically).
- Consistent theming for plots and widgets via the `medusa-style` package.

### Changed

- **Requires Python 3.13+.** Pure Python with no compiled extensions, shipped as
  a single universal wheel that runs identically on Linux, macOS and Windows.
- Processing routines are **free functions over arrays** (NumPy in, NumPy out);
  stateful transforms follow the scikit-learn `fit` / `transform` convention.
- **Pipelines** restructured: VEP spellers live under
  `medusa.pipelines.bci.vep_spellers` with a two-layer design — a `Pipeline`
  that scores EEG and a `CommandDecoder` that maps scores to commands — and
  endogenous paradigms (motor imagery, neurofeedback) are grouped together.
- `SettingsTree` configuration is now JSON-only.

### Removed

- The 1.x modality classes (`EEG`, `ECG`, `EMG`, …). Modality is now a
  per-channel property of `ChannelSet`, so a single `Signal` may mix channel
  types (EEG + EOG + TRIG …) in one stream.
- Dataset containers and BIDS **dataset** I/O: the kernel stays BIDS-*aligned*
  at the single-recording level; dataset assembly moves to a separate tool.

## [1.x]

For the history of the 1.x series, see the Git tags (`v1.2.1` … `v1.4.3`) and
their release notes.

[2.0.0]: https://github.com/medusabci/medusa-kernel/releases
