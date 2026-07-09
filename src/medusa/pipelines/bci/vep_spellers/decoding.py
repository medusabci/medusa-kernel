"""Decoding for VEP spellers: Layer-1 scoring pipelines and the Layer-2 command decoder.

Two layers live in this file, with **one shared contract** between them:

* **Layer 1** is a :class:`~medusa.pipelines.base.DecodingPipeline` that turns EEG into a
  *cumulative* ``(n_cycles, n_commands)`` **score matrix**. Row ``i`` is the decision score
  for every command after cycle event ``i``. The scores add up over that trial's cycles in
  the way the model works best. There are two scoring families. Each is a **direct**
  pipeline, with no shared base, because their feature paths have nothing in common:

  - **BWR** (:class:`BWRLDAPipeline`): classify each code frame (is the target response
    present or not). Then, for each command, take the Pearson correlation between the frame
    scores of the used cycles (joined end to end) and the command's codes joined the same
    way (:func:`bwr_command_scores`).
  - **Template matching** (:class:`TMCCAPipeline`): coherently average each command's cycle
    segments, then score the average against the command's reference. The reference is
    either a sin/cos harmonic bank scored by CCA (``reference='harmonics'``), or a learned
    template scored by the ``|Pearson|`` of a learned spatial-filter projection
    (``reference='template'``) (:func:`tm_command_scores`).

* **Layer 2** (:class:`VEPCommandDecoder`) is a *pure selector*. For each trial, after each
  cycle, it takes the argmax of the cumulative score row over that trial's **available**
  commands (with optional early stopping). It is paradigm-agnostic (no codes, no per-family
  logic), because Layer 1 already did the family-specific accumulation.

Events carry **one row per stimulation cycle** (``onset`` = cycle start; ``code_idx`` = the
code that cycle showed, ``0`` for single-code paradigms). The per-frame onsets come from
``onset + frame / fps_resolution``. The codes also drive BWR *training*: a frame's
target/non-target label is ``codes[target_command][code_idx][frame]`` (:func:`bwr_labels`).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import get_args

import numpy as np
from numpy.typing import NDArray
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from medusa.core.settings_tree import Configurable, SettingsTree
from medusa.core.serialization import pack_pickleable, unpack_pickleable
from medusa.core.data.recording import Recording
from medusa.core.data.signal import Signal
from medusa.signal.frequency_filtering import IIRFilter, FIRFilter, BAND_TYPES
from medusa.signal.spatial_filtering import car, CCA
from medusa.signal.segmentation import segment_signal_around_events, resample_segments

from medusa.pipelines.base import DecodingPipeline, harmonize_channels
from medusa.pipelines.bci.vep_spellers.data import SpellerData, validate_speller_events

__all__ = [
    "BWRLDAPipeline",
    "TMCCAPipeline",
    "VEPCommandDecoder",
    "bwr_labels",
    "bwr_command_scores",
    "tm_command_scores",
    "select_commands",
    "command_decoding_accuracy",
    "command_decoding_accuracy_per_cycle",
]


# --------------------------------------------------------------------------- #
# Events -> per-cycle / per-frame arrays
# --------------------------------------------------------------------------- #
def _cycle_arrays(events):
    """Per-cycle ``(onsets, trial_idx, cycle_idx, code_idx)`` from the stimulation events.

    One row per stimulation *cycle* (``onset`` = cycle start); rows with a null
    ``cycle_idx`` are ignored. ``code_idx`` (which code that cycle presented) is mandatory
    -- ``0`` for single-code paradigms (c-VEP / SSVEP).
    """
    df = events.df
    df = df[df["cycle_idx"].notna()]
    return (df["onset"].to_numpy(dtype=float),
            df["trial_idx"].to_numpy(dtype=int),
            df["cycle_idx"].to_numpy(dtype=int),
            df["code_idx"].to_numpy(dtype=int))


def _bit_onsets(cycle_onsets, n_frames, fps):
    """Expand cycle onsets to per-frame bit onsets ``cycle_onset + frame / fps``.

    Returns a flat array in cycle-major order (cycle 0 frames ``0..n_frames-1``, ...).
    """
    frame = np.arange(n_frames)
    return (np.asarray(cycle_onsets, dtype=float)[:, None]
            + frame[None, :] / fps).ravel()


def _trial_cycle_order(cycle_trial, cycle_idx):
    """Yield ``(trial, ordered_row_indices)``: each trial's cycle rows in ``cycle_idx`` order.

    The Layer-1 accumulators and the Layer-2 selector share this, so they go through a
    recording's cycles in the same order (each trial's rows sorted by ``cycle_idx``, stably).
    """
    cycle_trial = np.asarray(cycle_trial, dtype=int)
    cycle_idx = np.asarray(cycle_idx, dtype=int)
    for t in np.unique(cycle_trial):
        m = np.where(cycle_trial == t)[0]
        yield int(t), m[np.argsort(cycle_idx[m], kind="stable")]


# --------------------------------------------------------------------------- #
# Frequency filtering -- a shared notch + parallel filter bank
# --------------------------------------------------------------------------- #

#: Selectable band types, as a runtime list (``BAND_TYPES`` is a typing ``Literal``).
_BAND_TYPE_OPTIONS = list(get_args(BAND_TYPES))


def _add_filter_leaves(group: SettingsTree, cutoff: list, order: int,
                       band_type: "str | None" = None) -> None:
    """Add the shared filter-spec leaves to ``group`` (a ``filt_type`` enum and a bounded order).

    Adds ``filt_type``, ``cutoff`` and ``order`` for every filter. Adds ``band_type`` (an
    enum) too when it is given. The notch omits ``band_type`` because it is always a bandstop.
    """
    group.add_item("filt_type", value="iir", value_options=["iir", "fir"],
                   info="Filter family")
    if band_type is not None:
        group.add_item("band_type", value=band_type, value_options=_BAND_TYPE_OPTIONS,
                       info="Band type")
    group.add_item("cutoff", value=cutoff, info="Cutoff frequency/frequencies in Hz")
    group.add_item("order", value=order, value_range=[1, None], info="Filter order")


def _add_freq_filtering_settings(settings: SettingsTree) -> None:
    """Add the shared frequency-filtering levels to ``settings``.

    Adds a ``notch_filtering`` group and a ``freq_filtering`` group. The notch is a single
    filter (default: a 50 Hz line-noise bandstop), applied first, with an ``enabled`` toggle.
    The ``freq_filtering`` group holds ``filterbank``, a **group-list** of filter groups that
    share one schema (each is ``{filt_type, band_type, cutoff, order}``). One filter is a plain
    band-pass. Several filters make a filter bank: the sub-bands are processed in parallel and
    then combined (FBCCA scores for template matching, concatenated features for BWR). The
    default bank is one band-pass, 1--70 Hz.
    """
    notch = settings.add_group(
        "notch_filtering", info="Line-noise notch (bandstop), applied before the filter bank")
    notch.add_item("enabled", value=True, info="Apply the notch")
    _add_filter_leaves(notch, cutoff=[48.0, 52.0], order=4)      # band_type is always bandstop

    ff = settings.add_group("freq_filtering", info="Parallel filter-bank filtering")
    fb = ff.add_group_list(
        "filterbank",
        info="Parallel sub-band filters; one filter = a single band, several = a filter bank")
    _add_filter_leaves(fb.element, cutoff=[1.0, 70.0], order=5, band_type="bandpass")
    fb.add_element()                                # default bank: one band-pass 1--70 Hz


def _make_filter(spec: dict, filt_method: "str | None" = None):
    """Build an :class:`IIRFilter` or :class:`FIRFilter` from a validated ``spec`` dict.

    ``spec`` is ``{filt_type: 'iir'|'fir', band_type, cutoff, order}`` (``filt_type`` is
    optional and defaults to ``'iir'``). ``filt_method`` defaults to the family's offline
    method (``sosfiltfilt`` for IIR, ``filtfilt`` for FIR). SettingsTree cannot validate the
    filter-bank list elements, so a malformed spec raises a clear :class:`ValueError` here
    instead of a deep ``KeyError``.
    """
    if not isinstance(spec, dict):
        raise ValueError(f"filter spec must be a dict, got {type(spec).__name__}.")
    missing = [k for k in ("band_type", "cutoff", "order") if k not in spec]
    if missing:
        raise ValueError(
            f"filter spec {spec!r} is missing {missing}; each filter needs filt_type "
            f"(optional, default 'iir'), band_type, cutoff and order.")
    band, filt_type = spec["band_type"], spec.get("filt_type", "iir")
    if band not in _BAND_TYPE_OPTIONS:
        raise ValueError(f"filter band_type must be one of {_BAND_TYPE_OPTIONS}, got {band!r}.")
    if filt_type not in ("iir", "fir"):
        raise ValueError(f"filter filt_type must be 'iir' or 'fir', got {filt_type!r}.")
    cutoff = spec["cutoff"]
    cutoff = tuple(cutoff) if isinstance(cutoff, (list, tuple)) else float(cutoff)
    order = int(spec["order"])
    if filt_type == "fir":
        return FIRFilter(order, cutoff, band, filt_method=filt_method or "filtfilt")
    return IIRFilter(order, cutoff, band, filt_method=filt_method or "sosfiltfilt")


def _filterbank_signals(signal: NDArray, fs: float, notch: dict,
                        filterbank: list) -> "list[NDArray]":
    """Notch then each filter-bank filter -> one filtered signal per sub-band (parallel).

    ``notch`` is the ``notch_filtering`` config (a bandstop; skipped when ``enabled`` is
    False) applied first; ``filterbank`` is the list of filter specs. Returns a list of
    ``(n_samples, n_channels)`` arrays, one per filter-bank entry.
    """
    x = signal
    if notch.get("enabled", True):
        x = _make_filter({**notch, "band_type": "bandstop"}).fit_transform(signal, fs)
    if not filterbank:
        raise ValueError("freq_filtering.filterbank is empty; add at least one filter.")
    return [_make_filter(spec).fit_transform(x, fs) for spec in filterbank]


def _fbcca_weights(n_bands: int) -> NDArray:
    """Normalised filter-bank sub-band weights ``w_k ~ k**-1.25 + 0.25`` (they sum to 1).

    The standard FBCCA weighting (Chen et al. 2015), normalised so that a single sub-band
    gets weight 1 (then the combined score equals the plain single-band score).
    """
    k = np.arange(1, n_bands + 1, dtype=float)
    w = k ** -1.25 + 0.25
    return w / w.sum()


def bwr_labels(recording: Recording) -> NDArray:
    """Per-frame target labels (1 or 0) for BWR training, taken straight from the codes.

    Each cycle contributes the trial target's code *for that cycle*. Take a cycle of a
    trial whose target is command ``k`` and whose code index is ``c``. Its ``n_frames``
    labels are ``codes[k][c]``: 1 on the frames that light the target, 0 elsewhere. The
    labels are in cycle-major order, to match the per-frame scores.

    Raises
    ------
    ValueError
        If the recording has no ``spell_target`` (labels cannot be derived).
    """
    sd = SpellerData.from_recording(recording)
    if sd.spell_target is None:
        raise ValueError("recording has no spell_target; cannot derive BWR labels.")
    _, trial, _, code_idx = _cycle_arrays(recording.events)
    codes = sd.codes
    row = {uid: i for i, uid in enumerate(sd.command_uids)}
    target = list(sd.spell_target)
    return np.concatenate(
        [codes[row[str(target[t])], c] for t, c in zip(trial, code_idx)]).astype(int)


# --------------------------------------------------------------------------- #
# Layer 1 -- BWR scoring pipeline
# --------------------------------------------------------------------------- #
class BWRLDAPipeline(DecodingPipeline):
    """Bit-wise-reconstruction speller pipeline with a regularised-LDA classifier.

    The BWR *strategy* classifies each code frame (is the target response present or not).
    A command's score is then the correlation of its code with those frame scores. This
    class bundles the whole shallow chain: band-pass IIR + CAR, then epoch each code frame
    (the bit onsets come from the cycle onsets and ``fps_resolution``), then resample, then
    flatten, then regularised LDA. It ships with defaults suited to LDA. :meth:`predict`
    returns the cumulative ``(n_cycles, n_commands)`` correlation matrix
    (:func:`bwr_command_scores`) that the :class:`VEPCommandDecoder` turns into selections.

    Deep BWR variants (EEGInception, EEGNet) are **separate** pipeline classes, not a
    ``clf=`` option. The model choice changes the preprocessing (band, ``target_fs``,
    epoch shape), so each pipeline owns its own chain. Only the model-agnostic parts are
    shared: :func:`bwr_labels`, the cycle and bit-onset helpers, and the Layer-2
    :class:`VEPCommandDecoder`.

    Configure it through the live :attr:`~medusa.core.settings_tree.Configurable.settings`
    tree (see :meth:`default_settings`). It has ``freq_filtering`` (notch + filter bank),
    ``epoching`` and ``classifier`` levels. You can also configure it with construction
    kwargs, which may be nested, for example ``BWRLDAPipeline(channels=["Fz", "Cz", "Pz",
    "Oz"], epoching={"target_fs": 20.0})``. With a multi-filter bank, the per-sub-band
    epoch features are concatenated before the LDA (filter-bank feature fusion).
    """

    fs = None       # sampling rate adopted at fit (fitted state)
    clf = None      # the fitted LDA (set by fit / restored by load)

    # ---- configuration schema (SettingsTree) ----
    @classmethod
    def default_settings(cls) -> SettingsTree:
        """A GUI-editable, levelled schema of the pipeline's configuration."""
        s = SettingsTree()
        s.add_item("channels", value=[], info="Channels to decode (required)")
        s.add_item("signal_key", value="eeg", info="Recording stream key to decode")
        s.add_item("car", value=True, info="Common-average reference before filtering")
        _add_freq_filtering_settings(s)
        ep = s.add_group("epoching", info="Per-frame epoch windowing + resampling")
        ep.add_item("w_segment_t", value=[0.0, 800.0],
                    info="Epoch window relative to each frame onset (ms)")
        ep.add_item("baseline_t", value=[-200.0, 0.0],
                    info="Baseline window (ms); empty to disable")
        ep.add_item("target_fs", value=20.0, value_range=[0, None],
                    info="Resample epochs to this rate (Hz); 0 to disable")
        clf = s.add_group("classifier", info="Regularised-LDA classifier")
        clf.add_item("shrinkage", value="auto", info="LDA shrinkage ('auto' or a float)")
        return s

    # ---- validation ----
    def check_consistency(self, recording: Recording) -> None:
        """Check the recording has the configured signal and channels, a matching ``fs``,
        a :class:`SpellerData`, and valid speller events; raise ``ValueError`` if not."""
        cfg = self.cfg
        sig = recording.signals.get(cfg["signal_key"])
        if sig is None:
            raise ValueError(f"recording has no {cfg['signal_key']!r} signal.")
        if not cfg["channels"]:
            raise ValueError("no channels configured; set the 'channels' setting.")
        if self.fs is None:
            self.fs = sig.fs
        elif sig.fs != self.fs:
            raise ValueError(f"fs mismatch: pipeline={self.fs}, recording={sig.fs}.")
        missing = [c for c in cfg["channels"] if c not in sig.channel_set.labels]
        if missing:
            raise ValueError(f"recording is missing channels {missing}.")
        SpellerData.from_recording(recording)      # experiment must be a SpellerData
        validate_speller_events(recording.events)

    # ---- feature path (shared by fit/predict) ----
    def _features(self, signal: Signal, cycle_onsets: NDArray,
                  n_frames: int, fps: float, cfg: dict) -> NDArray:
        x = harmonize_channels(signal, cfg["channels"])
        raw = car(x.signal) if cfg["car"] else x.signal
        bands = _filterbank_signals(raw, x.fs, cfg["notch_filtering"],
                                    cfg["freq_filtering"]["filterbank"])
        onsets = _bit_onsets(cycle_onsets, n_frames, fps)
        ep = cfg["epoching"]
        window = tuple(ep["w_segment_t"])
        baseline = tuple(ep["baseline_t"]) if ep["baseline_t"] else None
        # Filter-bank feature fusion: flatten each sub-band's epochs, concatenate.
        feats = []
        for xf in bands:
            seg = segment_signal_around_events(
                x.times, xf, onsets, x.fs, window, baseline,
                norm="dc" if baseline is not None else None)
            if ep["target_fs"]:
                seg = resample_segments(seg, window, ep["target_fs"])
            feats.append(seg.reshape(len(seg), -1))
        return np.concatenate(feats, axis=1)

    def _frame_scores(self, recording: Recording, cfg: dict) -> NDArray:
        """Per-frame target-class scores for one recording (cycle-major order)."""
        sd = SpellerData.from_recording(recording)
        onsets, _, _, _ = _cycle_arrays(recording.events)
        feats = self._features(recording.signals[cfg["signal_key"]], onsets,
                               sd.codes.shape[2], sd.fps_resolution, cfg)
        return self.clf.predict_proba(feats)[:, 1]

    # ---- offline ----
    def fit(self, recordings) -> "BWRLDAPipeline":
        """Fit the LDA on the per-frame BWR features and labels of all recordings; return ``self``."""
        self._check_settings()          # re-validate (the live tree may have been edited)
        cfg = self.cfg
        X, y = [], []
        for rec in recordings:
            self.check_consistency(rec)
            onsets, _, _, _ = _cycle_arrays(rec.events)
            sd = SpellerData.from_recording(rec)
            X.append(self._features(rec.signals[cfg["signal_key"]], onsets,
                                    sd.codes.shape[2], sd.fps_resolution, cfg))
            y.append(bwr_labels(rec))
        X, y = np.concatenate(X), np.concatenate(y)
        self.clf = LinearDiscriminantAnalysis(
            solver="lsqr", shrinkage=cfg["classifier"]["shrinkage"]).fit(X, y)
        self._fitted = True
        return self

    def predict(self, recording: Recording) -> NDArray:
        """Cumulative ``(n_cycles, n_commands)`` command correlations for one recording."""
        if not self._fitted:
            raise RuntimeError("pipeline is not fitted; call fit() first.")
        self.check_consistency(recording)
        sd = SpellerData.from_recording(recording)
        _, trial, cycle, code_idx = _cycle_arrays(recording.events)
        frame_scores = self._frame_scores(recording, self.cfg)
        return bwr_command_scores(frame_scores, sd.codes, trial, cycle, code_idx)

    # ---- persistence (settings + fitted state) ----
    def to_pickleable_obj(self) -> dict:
        """Bundle the settings, the fitted flag, ``fs``, and the packed LDA for saving."""
        return {"settings": self.settings.to_dict(), "fitted": self._fitted,
                "fs": self.fs,
                "clf": pack_pickleable(self.clf) if self._fitted else None}

    @classmethod
    def from_pickleable_obj(cls, obj: dict) -> "BWRLDAPipeline":
        """Rebuild the pipeline from a bundle made by :meth:`to_pickleable_obj`."""
        self = cls(settings=obj["settings"])
        self.fs, self._fitted = obj["fs"], obj["fitted"]
        if self._fitted:
            self.clf = unpack_pickleable(obj["clf"])
        return self


# --------------------------------------------------------------------------- #
# Layer 1 -- template-matching (CCA) scoring pipeline
# --------------------------------------------------------------------------- #
def _cca_reference(freq: float, n_samples: int, fs: float, n_harmonics: int) -> NDArray:
    """Sine/cosine harmonic reference for CCA, shape ``(n_samples, 2 * n_harmonics)``.

    For ``k = 0 .. n_harmonics - 1``, column pair ``(2k, 2k+1)`` holds the ``sin`` / ``cos``
    of harmonic ``k + 1`` of ``freq``. Stacking harmonics lets CCA fit the harmonic structure
    of the flicker response.
    """
    t = np.arange(n_samples) / fs
    cols = []
    for h in range(1, n_harmonics + 1):
        cols.append(np.sin(2 * np.pi * h * freq * t))
        cols.append(np.cos(2 * np.pi * h * freq * t))
    return np.column_stack(cols)


def _cca_corr(segment: NDArray, reference: NDArray) -> float:
    """Top CCA canonical correlation ``|r|`` between a segment and a reference.

    ``segment`` is ``(n_samples, n_channels)`` and ``reference`` is ``(n_samples, n_ref)``.
    CCA finds the combination of channels and reference columns that maximises their
    correlation, and this returns the largest canonical correlation (``-inf`` if the fit is
    degenerate). This is the CCA scoring *method* of :class:`TMCCAPipeline`. The family-level
    accumulation lives in :func:`tm_command_scores`.
    """
    cca = CCA()
    cca.fit(segment, reference)
    r = cca.r
    return float(abs(r[0])) if r is not None and np.isfinite(r[0]) else -np.inf


def _best_code_shift(code: NDArray, learned_code: NDArray) -> "tuple[int, float]":
    """Circular lag ``k`` (and its correlation) for which ``roll(learned_code, k) ~= code``.

    In a shift-coded paradigm (c-VEP: every command's code is a circular shift of one
    m-sequence), this finds how far a command's code is rolled from a *learned* code. Its
    EEG template can then be built by rolling the learned template by the same lag.
    """
    a = np.asarray(code, dtype=float)
    a = a - a.mean()
    b = np.asarray(learned_code, dtype=float)
    b = b - b.mean()
    n = len(a)
    rolls = np.stack([np.roll(b, k) for k in range(n)])       # (n, n): roll(b, k) per lag
    num = rolls @ a
    den = np.sqrt((rolls ** 2).sum(axis=1) * (a ** 2).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = num / den
    lag = int(np.nanargmax(corr)) if np.isfinite(corr).any() else 0
    return lag, float(corr[lag]) if np.isfinite(corr).any() else -np.inf


#: Circular-shift correlation above which two command codes are treated as the *same*
#: base sequence rolled (m-sequence c-VEP: shifts correlate at 1.0; distinct codes -- Gold
#: c-VEP, SSVEP frequencies -- stay well below). Used to pool shift-family calibration.
_SHIFT_MATCH_TOL = 0.999


def _corr1d(a: NDArray, b: NDArray) -> float:
    """``|Pearson correlation|`` between two 1-D signals (``-inf`` if either is constant)."""
    a = np.asarray(a, dtype=float)
    a = a - a.mean()
    b = np.asarray(b, dtype=float)
    b = b - b.mean()
    den = np.sqrt(np.dot(a, a) * np.dot(b, b))
    return float(abs(np.dot(a, b) / den)) if den > 0 else -np.inf


class TMCCAPipeline(DecodingPipeline):
    """Template-matching speller pipeline based on canonical correlation analysis (CCA).

    Scores each command by the correlation between a cycle's multichannel EEG segment and
    that command's **reference**. Evidence builds up over cycles by coherently averaging the
    segments, so :meth:`predict` returns the cumulative ``(n_cycles, n_commands)`` matrix
    (:func:`tm_command_scores`) that the :class:`VEPCommandDecoder` selects from.
    Configuration is levelled: ``freq_filtering`` (notch + filter bank) and ``reference``.
    With a multi-sub-band filter bank, the per-band scores are combined with FBCCA weights
    (:func:`_fbcca_weights`).

    The ``reference.mode`` setting is **required** (no default). It picks the strategy:

    * ``"harmonics"`` -- **calibration-free SSVEP**. The reference is the command's sin/cos
      harmonic bank at ``extra['stim_freq']`` (from
      :func:`~medusa.pipelines.bci.vep_spellers.encoding.generate_freq_codebook`). The score is the
      top canonical correlation. There is no training: construct the pipeline and call
      :meth:`predict` directly.
    * ``"template"`` -- **calibrated**. :meth:`fit` learns, per shift-family and per sub-band,
      a coherent-average EEG template and a spatial filter (it needs ``spell_target``).
      Scoring projects both the template and the test segment to 1-D with that filter and
      takes their ``|Pearson|``. A full multichannel template would let CCA align any command,
      so the 1-D projection is what tells commands apart. Shift-coded commands (c-VEP) are
      built from one pooled base template by rolling it by the command's code lag. Distinct
      codes (Gold c-VEP, SSVEP) each keep their own template. One class serves both
      SSVEP-with-calibration and c-VEP.

    A *template-matching* sibling of :class:`BWRLDAPipeline`. Like it, it is a **direct**
    :class:`~medusa.pipelines.base.DecodingPipeline` (no shared template-matching base): its
    feature path (one multichannel segment per *cycle*, no per-frame epoching) has nothing in
    common with BWR's. TRCA (a different spatial-filter objective) will be a separate pipeline.
    """

    fs = None            # sampling rate adopted at fit/predict
    templates = None     # reference='template': per sub-band {base: (spatial_filter, 1-D template, code)}

    # ---- configuration schema (SettingsTree) ----
    @classmethod
    def default_settings(cls) -> SettingsTree:
        """A GUI-editable, levelled schema; ``reference.mode`` is required (no default)."""
        s = SettingsTree()
        s.add_item("channels", value=[], info="Channels to decode (required)")
        s.add_item("signal_key", value="eeg", info="Recording stream key to decode")
        _add_freq_filtering_settings(s)
        s.add_item("car", value=False,
                   info="Common-average reference before scoring (CCA already "
                        "spatially filters; CAR makes the montage rank-deficient)")
        ref = s.add_group("reference", info="Template-matching reference")
        ref.add_item("mode", value=None, value_options=["harmonics", "template"],
                     info="Reference mode (REQUIRED): 'harmonics' (calibration-free SSVEP) "
                          "or 'template' (calibrated; needs fit)")
        ref.add_item("n_harmonics", value=3, value_range=[1, None],
                     info="Sine/cosine harmonics per reference (mode='harmonics')")
        return s

    # ---- validation ----
    def check_consistency(self, recording: Recording) -> None:
        """Check the recording has the configured signal and channels, a matching ``fs``,
        valid speller events, and (for ``mode='harmonics'``) a frequency per command;
        raise ``ValueError`` if not."""
        cfg = self.cfg
        sig = recording.signals.get(cfg["signal_key"])
        if sig is None:
            raise ValueError(f"recording has no {cfg['signal_key']!r} signal.")
        if not cfg["channels"]:
            raise ValueError("no channels configured; set the 'channels' setting.")
        if self.fs is None:
            self.fs = sig.fs
        elif sig.fs != self.fs:
            raise ValueError(f"fs mismatch: pipeline={self.fs}, recording={sig.fs}.")
        missing = [c for c in cfg["channels"] if c not in sig.channel_set.labels]
        if missing:
            raise ValueError(f"recording is missing channels {missing}.")
        sd = SpellerData.from_recording(recording)
        mode = cfg["reference"]["mode"]
        if mode == "harmonics":
            missing_freq = [uid for uid, cmd in sd.commands_info.items()
                            if cmd.extra.get("stim_freq") is None]
            if missing_freq:
                raise ValueError(
                    f"reference.mode='harmonics' needs a stimulation frequency per command; "
                    f"commands {missing_freq} lack extra['stim_freq'] (build the codebook "
                    f"with generate_freq_codebook).")
        elif mode != "template":
            raise ValueError(
                f"reference.mode must be 'harmonics' or 'template', got {mode!r}.")
        validate_speller_events(recording.events)

    # ---- feature path (one segment set per filter-bank sub-band) ----
    def _cycle_segments_per_band(self, signal: Signal, cycle_onsets: NDArray,
                                 n_frames: int, fps: float,
                                 cfg: dict) -> "list[NDArray]":
        """Per sub-band, one multichannel segment per cycle: list of ``(n_cycles, n_s, n_ch)``."""
        x = harmonize_channels(signal, cfg["channels"])
        raw = car(x.signal) if cfg["car"] else x.signal
        window = (0.0, 1000.0 * n_frames / fps)     # the full stimulation cycle, in ms
        return [segment_signal_around_events(x.times, xf, cycle_onsets, x.fs, window)
                for xf in _filterbank_signals(raw, x.fs, cfg["notch_filtering"],
                                              cfg["freq_filtering"]["filterbank"])]

    # ---- offline ----
    def fit(self, recordings=()) -> "TMCCAPipeline":
        """Calibrate the pipeline and return ``self``.

        For ``reference.mode='harmonics'`` there is nothing to learn: it only validates the
        recordings and adopts ``fs``. For ``'template'`` it calls :meth:`_learn_templates`.
        """
        self._check_settings()
        if self.cfg["reference"]["mode"] == "harmonics":
            for rec in recordings:
                self.check_consistency(rec)
        else:
            self._learn_templates(recordings)
        self._fitted = True
        return self

    def predict(self, recording: Recording) -> NDArray:
        """Cumulative ``(n_cycles, n_commands)`` correlations (FBCCA-combined over sub-bands)."""
        cfg = self.cfg
        if cfg["reference"]["mode"] == "template" and not self._fitted:
            raise RuntimeError(
                "reference.mode='template' pipeline is not fitted; call fit() first.")
        self.check_consistency(recording)          # adopts fs
        sd = SpellerData.from_recording(recording)
        onsets, trial, cycle, _ = _cycle_arrays(recording.events)
        n_frames = sd.codes.shape[2]
        band_segs = self._cycle_segments_per_band(
            recording.signals[cfg["signal_key"]],
            onsets, n_frames, sd.fps_resolution, cfg)
        weights = _fbcca_weights(len(band_segs))
        combined = None                              # weighted sum of per-sub-band scores
        for b, segs in enumerate(band_segs):
            score_fn = self._build_score_fn(sd, segs.shape[1], n_frames, b)
            m = np.nan_to_num(tm_command_scores(segs, score_fn, trial, cycle), neginf=0.0)
            combined = weights[b] * m if combined is None else combined + weights[b] * m
        return combined

    # ------------------------------------------------------------------ #
    # Reference modes -- each a self-contained scoring strategy. _build_score_fn
    # dispatches per sub-band; each mode owns its reference construction + scoring.
    # ------------------------------------------------------------------ #
    def _build_score_fn(self, sd: SpellerData, n_samples: int, n_frames: int, band: int):
        """A ``score_fn(avg_segment) -> (n_commands,)`` for the mode, on sub-band ``band``."""
        if self.cfg["reference"]["mode"] == "harmonics":
            return self._harmonic_score_fn(sd, n_samples)
        return self._template_score_fn(sd, n_samples, n_frames, band)

    def _harmonic_score_fn(self, sd: SpellerData, n_samples: int):
        """Calibration-free SSVEP: CCA of the segment against each command's sin/cos harmonics.

        The harmonic reference is low-dimensional (``2 * n_harmonics`` columns), so CCA's
        spatial fit is well-posed. The score is the top canonical correlation. It is
        band-independent: every sub-band scores against the same references.
        """
        n_harmonics = int(self.cfg["reference"]["n_harmonics"])
        refs = [_cca_reference(float(cmd.extra["stim_freq"]),
                               n_samples, self.fs, n_harmonics)
                for cmd in sd.commands_info.values()]
        return lambda avg: np.array([_cca_corr(avg, ref) for ref in refs])

    def _template_score_fn(self, sd: SpellerData, n_samples: int, n_frames: int, band: int):
        """Calibrated: a learned spatial filter and 1-D correlation against shifted templates.

        Uses sub-band ``band``'s learned templates. A full multichannel template would let CCA
        align any command, so each learned base owns a spatial filter ``w`` that projects both
        its template and the test segment to 1-D. The score is the ``|Pearson|`` between them.
        """
        templates = self.templates[band]
        scorers = [
            self._template_for(
                templates, np.asarray(sd.codes[i, 0]),
                n_samples, n_frames
            )
            for i in range(len(sd.command_uids))
        ]
        return lambda avg: np.array([_corr1d(avg @ w, t1d) for w, t1d in scorers])

    @staticmethod
    def _template_for(templates: dict, code: NDArray, n_samples: int,
                      n_frames: int) -> "tuple[NDArray, NDArray]":
        """``(spatial filter w, shifted 1-D template)`` for a command's ``code``, from ``templates``.

        Picks the learned base whose code best matches ``code`` (an identical code means lag 0)
        and rolls its 1-D template by the code lag, mapped to samples
        (``lag * n_samples / n_frames``). So a c-VEP command decoded from one pooled template
        lands on its own response.
        """
        best = None
        for _base, (w, t1d, learned_code) in templates.items():
            lag, corr = _best_code_shift(code, np.asarray(learned_code))
            if best is None or corr > best[0]:
                best = (corr, w, t1d, lag)
        _, w, t1d, lag = best
        lag_samples = int(round(lag * n_samples / n_frames))
        return w, np.roll(t1d, lag_samples)

    def _learn_templates(self, recordings) -> None:
        """Calibrate ``reference.mode='template'``: templates per sub-band and per shift-family.

        For each filter-bank sub-band, attended commands whose codes are circular shifts (an
        m-sequence family) are pooled: their epochs are rolled to a common base and averaged
        into one template and spatial filter. Distinct codes each form their own base. Every
        sub-band learns its own template set (see :meth:`_pool_templates`).
        """
        cfg = self.cfg
        band_epochs, codes, n_samples, n_frames = None, {}, None, None
        for rec in recordings:
            self.check_consistency(rec)
            sd = SpellerData.from_recording(rec)
            if sd.spell_target is None:
                raise ValueError("reference.mode='template' needs spell_target in each "
                                 "calibration recording.")
            onsets, trial, _, _ = _cycle_arrays(rec.events)
            band_segs = self._cycle_segments_per_band(
                rec.signals[cfg["signal_key"]],
                onsets, sd.codes.shape[2],
                sd.fps_resolution, cfg)
            if band_epochs is None:
                band_epochs = [{} for _ in band_segs]
            n_samples, n_frames = band_segs[0].shape[1], sd.codes.shape[2]
            row = {u: i for i, u in enumerate(sd.command_uids)}
            for b, segs in enumerate(band_segs):
                for i, t in enumerate(trial):
                    uid = str(sd.spell_target[int(t)])
                    band_epochs[b].setdefault(uid, []).append(segs[i])
                    codes[uid] = np.asarray(sd.codes[row[uid], 0])
        if band_epochs is None or not codes:
            raise ValueError("no calibration segments found to learn templates.")
        self.templates = [self._pool_templates(eps, codes, n_samples, n_frames)
                          for eps in band_epochs]

    @staticmethod
    def _pool_templates(epochs: dict, codes: dict, n_samples: int, n_frames: int) -> dict:
        """Pool shift-family epochs into ``{base: (w, 1-D template, code)}`` for one sub-band.

        Commands whose codes are circular shifts (corr >= ``_SHIFT_MATCH_TOL``) are aligned to
        a common base and averaged into one template. A spatial filter ``w`` (the first CCA
        component between the single epochs and the pooled template) projects each base to its
        1-D template.
        """
        bases = []                                       # [(base_code, [aligned epochs]), ...]
        for uid, eps in epochs.items():
            for base_code, pool in bases:
                lag, corr = _best_code_shift(codes[uid], base_code)
                if corr >= _SHIFT_MATCH_TOL:
                    shift = int(round(lag * n_samples / n_frames))
                    pool.extend(np.roll(ep, -shift, axis=0) for ep in eps)
                    break
            else:
                bases.append((codes[uid], list(eps)))
        templates = {}
        for k, (base_code, pool) in enumerate(bases):
            eps = np.stack(pool)                         # (n_epochs, n_samples, n_channels)
            template = eps.mean(axis=0)                  # (n_samples, n_channels)
            cca = CCA()                                  # spatial filter: epochs <-> template
            cca.fit(eps.reshape(-1, eps.shape[2]), np.tile(template, (len(eps), 1)))
            templates[str(k)] = (cca.wx[:, 0], template @ cca.wx[:, 0], base_code)
        return templates

    # ---- persistence (settings + adopted fs + learned templates) ----
    def to_pickleable_obj(self) -> dict:
        """Bundle the settings, the fitted flag, ``fs``, and the learned templates for saving."""
        return {"settings": self.settings.to_dict(), "fitted": self._fitted, "fs": self.fs,
                "templates": self.templates}

    @classmethod
    def from_pickleable_obj(cls, obj: dict) -> "TMCCAPipeline":
        """Rebuild the pipeline from a bundle made by :meth:`to_pickleable_obj`."""
        self = cls(settings=obj["settings"])
        self.fs, self._fitted = obj["fs"], obj["fitted"]
        self.templates = obj.get("templates")
        return self


# --------------------------------------------------------------------------- #
# Layer 1 -- cumulative score accumulators (pure functions, one per family)
# --------------------------------------------------------------------------- #
def _corr_rows(codes: NDArray, scores: NDArray) -> NDArray:
    """Absolute Pearson correlation ``|r|`` of every row of ``codes`` (n, m) with ``scores``.

    Magnitude, not signed. This matches the proven BWR command decoder, which ranks commands
    by ``|corr|`` (so a fully sign-inverted reconstruction still scores its code high). A row
    (or ``scores``) with zero variance gives ``-inf``, so it is never selected.
    """
    codes = codes.astype(float)
    scores = scores.astype(float)
    c = codes - codes.mean(axis=1, keepdims=True)
    s = scores - scores.mean()
    num = c @ s
    den = np.sqrt((c ** 2).sum(axis=1) * (s ** 2).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.abs(num / den)
    corr[~np.isfinite(corr)] = -np.inf
    return corr


def bwr_command_scores(frame_scores: NDArray, codes: NDArray, cycle_trial: NDArray,
                       cycle_idx: NDArray, cycle_code_idx: NDArray) -> NDArray:
    """Cumulative per-cycle BWR command correlations, shape ``(n_cycles, n_commands)``.

    Row ``i`` holds, for every command, the Pearson correlation between two things: the
    frame scores of all cycles of row ``i``'s trial up to and including cycle ``i`` (joined
    end to end), and that command's codes joined the same way. This is the
    *concatenate-then-correlate* rule of the c-VEP/ERP BWR decoder. Rows follow the input
    cycle order. A command whose code is constant over the used cycles gets ``-inf``.

    Parameters
    ----------
    frame_scores :
        ``(n_cycles * n_frames,)``. Per-frame target-class scores, in cycle-major order
        (as produced inside :meth:`BWRLDAPipeline.predict`).
    codes :
        ``(n_commands, n_codes, n_frames)``. Per-command codes (``SpellerData.codes``).
    cycle_trial, cycle_idx, cycle_code_idx :
        ``(n_cycles,)`` each. Per-cycle trial index, repetition index, and code index
        (from the events).

    Returns
    -------
    numpy.ndarray
        ``(n_cycles, n_commands)``. Cumulative command correlations, ready for
        :func:`select_commands`.

    Raises
    ------
    ValueError
        If ``frame_scores.size`` is not ``n_cycles * n_frames``.

    Examples
    --------
    >>> import numpy as np
    >>> from medusa.pipelines.bci.vep_spellers.decoding import bwr_command_scores
    >>> codes = np.array([[[1, 1, 0, 0]], [[1, 0, 1, 0]]])   # 2 commands, 1 code, 4 frames
    >>> frame_scores = np.array([1.0, 1.0, 0.0, 0.0])        # 1 cycle; matches command 0
    >>> scores = bwr_command_scores(frame_scores, codes,
    ...                             cycle_trial=np.array([0]), cycle_idx=np.array([0]),
    ...                             cycle_code_idx=np.array([0]))
    >>> scores.shape
    (1, 2)
    >>> int(scores.argmax())            # command 0 wins
    0
    """
    frame_scores = np.asarray(frame_scores, dtype=float)
    codes = np.asarray(codes)
    cycle_code_idx = np.asarray(cycle_code_idx, dtype=int)
    n_commands, _, n_frames = codes.shape
    n_cycles = len(cycle_code_idx)
    if frame_scores.size != n_cycles * n_frames:
        raise ValueError(
            f"frame_scores has {frame_scores.size} values but expected "
            f"n_cycles*n_frames = {n_cycles}*{n_frames}.")
    scores_by_cycle = frame_scores.reshape(n_cycles, n_frames)
    out = np.full((n_cycles, n_commands), -np.inf)
    for _, order in _trial_cycle_order(cycle_trial, cycle_idx):
        for i in range(len(order)):
            used = order[:i + 1]
            s = scores_by_cycle[used].ravel()
            # every command's expected code over the used cycles (in `used` order),
            # concatenated to match the flattened frame scores.
            exp = codes[:, cycle_code_idx[used]].reshape(n_commands, -1)
            out[order[i]] = _corr_rows(exp, s)
    return out


def tm_command_scores(cycle_segments: NDArray, score_fn: Callable,
                      cycle_trial: NDArray, cycle_idx: NDArray) -> NDArray:
    """Cumulative template-matching command scores, shape ``(n_cycles, n_commands)``.

    The template-matching counterpart of :func:`bwr_command_scores`. Row ``i`` holds
    ``score_fn`` applied to the **coherent average** of the segments of all cycles of row
    ``i``'s trial up to and including cycle ``i`` (the c-VEP/SSVEP averaging rule). The
    per-command scoring *method* is passed in as ``score_fn``. It may be CCA canonical
    correlation against a synthetic or a learned reference, TRCA-filtered correlation, and
    so on. So this function owns only the family-level accumulation, not the method itself
    (CCA is a method inside :class:`TMCCAPipeline`, not built in here).

    Parameters
    ----------
    cycle_segments :
        ``(n_cycles, n_samples, n_channels)``. One multichannel EEG segment per cycle (as
        produced inside a pipeline's ``predict``).
    score_fn :
        ``score_fn(avg_segment) -> ndarray (n_commands,)``. The per-command similarity for
        one coherently-averaged segment (for example :class:`TMCCAPipeline`'s CCA scorer).
    cycle_trial, cycle_idx :
        ``(n_cycles,)`` each. Per-cycle trial index and repetition index (from the events).

    Returns
    -------
    numpy.ndarray
        ``(n_cycles, n_commands)``. Cumulative command scores, ready for
        :func:`select_commands`.

    Examples
    --------
    >>> import numpy as np
    >>> from medusa.pipelines.bci.vep_spellers.decoding import tm_command_scores
    >>> cycle_segments = np.ones((2, 5, 3))     # 2 cycles, 5 samples, 3 channels
    >>> score_fn = lambda avg: avg.mean(axis=(0, 1)) * np.array([1.0, 2.0])
    >>> scores = tm_command_scores(cycle_segments, score_fn,
    ...                            cycle_trial=np.array([0, 0]),
    ...                            cycle_idx=np.array([0, 1]))
    >>> scores.shape
    (2, 2)
    >>> int(scores[-1].argmax())        # command 1 scores higher
    1
    """
    cycle_segments = np.asarray(cycle_segments, dtype=float)
    n_cycles = cycle_segments.shape[0]
    rows = {}
    for _, order in _trial_cycle_order(cycle_trial, cycle_idx):
        for i in range(len(order)):
            avg = cycle_segments[order[:i + 1]].mean(axis=0)     # (n_samples, n_channels)
            rows[int(order[i])] = np.asarray(score_fn(avg), dtype=float)
    n_commands = len(next(iter(rows.values()))) if rows else 0
    out = np.full((n_cycles, n_commands), -np.inf)
    for idx, r in rows.items():
        out[idx] = r
    return out


# --------------------------------------------------------------------------- #
# Layer 2 -- command selection (paradigm-agnostic)
# --------------------------------------------------------------------------- #
def _available(trial_available_cmmds, command_uids, trial: int) -> "list[str]":
    """The command uids selectable in ``trial`` (all commands when unrestricted)."""
    if trial_available_cmmds is None:
        return list(command_uids)
    if trial >= len(trial_available_cmmds):
        raise ValueError(
            f"trial_available_cmmds has {len(trial_available_cmmds)} entries but "
            f"trial_idx {trial} was seen; index them by trial_idx.")
    return [str(u) for u in trial_available_cmmds[trial]]


def select_commands(cycle_scores: NDArray, command_uids: list[str],
                    cycle_trial: NDArray, cycle_idx: NDArray,
                    trial_available_cmmds: list[list[str]] | None = None):
    """Select the decoded command per trial from a cumulative score matrix.

    The paradigm-agnostic Layer-2 rule. Row ``i`` of ``cycle_scores`` is the *cumulative*
    decision score after cycle event ``i`` (a Layer-1 pipeline already accumulated it in
    its family's natural way: :func:`bwr_command_scores` or :func:`tm_command_scores`). For
    each trial, and after each cycle, this takes the argmax over that trial's **available**
    commands. Multi-matrix paradigms are handled entirely by ``trial_available_cmmds``.

    Parameters
    ----------
    cycle_scores :
        ``(n_cycles, n_commands)``. Cumulative per-cycle command scores (a Layer-1
        pipeline's ``predict`` output).
    command_uids :
        Length ``n_commands``. Command ``uid``\\ s in score-column order
        (``SpellerData.command_uids``).
    cycle_trial, cycle_idx :
        ``(n_cycles,)`` each. Per-cycle trial index and repetition index (one row per
        cycle, from the events).
    trial_available_cmmds :
        Per-trial list of available command ``uid``\\ s
        (``SpellerData.trial_available_cmmds``). ``None`` makes every command available.

    Returns
    -------
    selected : dict
        ``{trial_idx: command_uid}``. The final selection (all cycles).
    per_cycle : dict
        ``{trial_idx: {n_cycles: command_uid}}``. The cumulative selection after each
        cycle (for dynamic stopping or accuracy-vs-cycles).
    scores : dict
        ``{trial_idx: {n_cycles: ndarray[n_available]}}``. Per-available-command score.

    Examples
    --------
    >>> import numpy as np
    >>> from medusa.pipelines.bci.vep_spellers.decoding import select_commands
    >>> cycle_scores = np.array([[0.1, 0.9], [0.8, 0.2]])    # 2 cycles, 2 commands
    >>> selected, per_cycle, scores = select_commands(
    ...     cycle_scores, command_uids=["A", "B"],
    ...     cycle_trial=np.array([0, 0]), cycle_idx=np.array([0, 1]))
    >>> selected[0]                     # the last cycle (idx 1) picks command "A"
    'A'
    """
    cycle_scores = np.asarray(cycle_scores, dtype=float)
    command_uids = list(command_uids)
    cycle_idx = np.asarray(cycle_idx, dtype=int)
    row = {uid: i for i, uid in enumerate(command_uids)}

    selected, per_cycle, scores = {}, {}, {}
    for t, order in _trial_cycle_order(cycle_trial, cycle_idx):
        avail = _available(trial_available_cmmds, command_uids, t)
        avail_cols = [row[u] for u in avail]
        per_cycle[t], scores[t] = {}, {}
        for i in order:
            corr = cycle_scores[i, avail_cols]
            nc = int(cycle_idx[i])
            per_cycle[t][nc] = avail[int(np.argmax(corr))]
            scores[t][nc] = corr
        selected[t] = per_cycle[t][max(per_cycle[t])]
    return selected, per_cycle, scores


def command_decoding_accuracy(selected: dict, target: dict) -> float:
    """Fraction of trials whose final selected command equals the target.

    ``selected`` and ``target`` are both ``{trial_idx: command_uid}``. Trials that are not
    in ``target`` are ignored.
    """
    keys = [t for t in selected if t in target]
    if not keys:
        return float("nan")
    correct = sum(str(selected[t]) == str(target[t]) for t in keys)
    return correct / len(keys)


def command_decoding_accuracy_per_cycle(per_cycle: dict, target: dict) -> NDArray:
    """Accuracy as a function of the number of cycles used.

    Returns an array indexed by the 0-based cycle index (the same ``cycle_idx`` as the
    events). Entry ``n`` is the accuracy after cycle ``n``, i.e. after ``n + 1`` cycles,
    counted over the trials that reached that many cycles. So entry ``0`` is the accuracy
    after the first cycle and the last entry is the final accuracy.
    """
    keys = [t for t in per_cycle if t in target]
    if not keys:
        return np.array([])
    max_cyc = max(max(per_cycle[t]) for t in keys)
    acc = np.full(max_cyc + 1, np.nan)
    for nc in range(max_cyc + 1):
        hits, total = 0, 0
        for t in keys:
            if nc in per_cycle[t]:
                total += 1
                hits += str(per_cycle[t][nc]) == str(target[t])
        if total:
            acc[nc] = hits / total
    return acc


class VEPCommandDecoder(Configurable):
    """Layer-2 command decoder: cumulative per-cycle command scores -> selected commands.

    Paradigm-agnostic and rule-based (no fitting). A Layer-1 pipeline's :meth:`predict`
    returns a cumulative ``(n_cycles, n_commands)`` score matrix, accumulated the family's
    natural way (concatenate-then-correlate for BWR, coherent averaging + CCA for template
    matching). This decoder only **selects**. Offline, :meth:`decode` returns the per-trial
    selection, the per-cycle trajectory, and the scores. Online, :meth:`step` takes the
    current cumulative score row and returns the current best command (and whether the
    early-stopping threshold on the top score fired). Configure it through its
    :attr:`~medusa.core.settings_tree.Configurable.settings` (``stop_corr``).
    """

    @classmethod
    def default_settings(cls) -> SettingsTree:
        """The configuration schema: a single ``stop_corr`` early-stopping threshold."""
        s = SettingsTree()
        s.add_item("stop_corr", value=0.0,
                   info="Early-stopping score threshold (0 = off)")
        return s

    # ---- offline ----
    def decode(self, cycle_scores: NDArray, speller_data: SpellerData, events) -> dict:
        """Decode every trial in a recording's events; return the selections and scores.

        ``cycle_scores`` is a Layer-1 pipeline's cumulative ``(n_cycles, n_commands)`` output
        for this recording (its rows line up with the cycle events).
        """
        _, trial, cycle, _ = _cycle_arrays(events)
        selected, per_cycle, scores = select_commands(
            cycle_scores, speller_data.command_uids, trial, cycle,
            speller_data.trial_available_cmmds)
        return {"selected_commands": selected,
                "selected_commands_per_cycle": per_cycle,
                "scores": scores}

    # ---- online ----
    def step(self, cycle_scores_row: NDArray, speller_data: SpellerData,
             trial: int = 0) -> dict:
        """Select the current best command from one cumulative score row.

        ``cycle_scores_row`` has one cumulative score per command (length ``n_commands``).
        It is a single row of a Layer-1 pipeline's cumulative output (the Layer-1 pipeline
        does the cross-cycle accumulation). Returns
        ``{'selected': uid, 'scores': ndarray[n_available], 'stop': bool}``.
        """
        avail = _available(speller_data.trial_available_cmmds,
                           speller_data.command_uids, int(trial))
        row = {uid: i for i, uid in enumerate(speller_data.command_uids)}
        avail_cols = [row[u] for u in avail]
        corr = np.asarray(cycle_scores_row, dtype=float)[avail_cols]
        best = int(np.argmax(corr))
        threshold = self.cfg["stop_corr"]
        stop = (threshold > 0 and np.isfinite(corr[best]) and corr[best] >= threshold)
        return {"selected": avail[best], "scores": corr, "stop": bool(stop)}
