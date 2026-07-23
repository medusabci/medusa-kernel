"""Bit-wise-reconstruction speller pipeline with a regularised-LDA classifier.

The shallow BWR Layer-1 pipeline (:class:`BWRLDAPipeline`). The BWR *strategy* classifies
each code frame (is the target response present or not); a command's score is then the
correlation of its code with those frame scores
(:func:`~medusa.pipelines.bci.vep_spellers.decoding.bwr_command_scores`). This is the
calibrated, torch-free BWR pipeline; the deep EEG-Inception sibling
(:class:`~medusa.pipelines.bci.vep_spellers.decoding.bwr_eeg_inception.BWREEGInceptionPipeline`)
lives in its own torch-gated module and shares only the model-agnostic BWR pure functions.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from medusa.core.settings_tree import SettingsTree
from medusa.core.serialization import pack_pickleable, unpack_pickleable
from medusa.core.data.recording import Recording
from medusa.core.data.signal import Signal
from medusa.signal.spatial_filtering import car
from medusa.signal.segmentation import segment_signal_around_events, resample_segments

from medusa.pipelines.base import DecodingPipeline, harmonize_channels
from medusa.pipelines.bci.vep_spellers.data import SpellerData, validate_speller_events
from medusa.pipelines.bci._filtering import (
    add_notch_and_filterbank_settings, apply_notch_and_filterbank)
from medusa.pipelines.bci.vep_spellers.decoding._common import (
    _bit_onsets, _cycle_arrays)
from medusa.pipelines.bci.vep_spellers.decoding.scores import (
    bwr_labels, bwr_command_scores)

__all__ = ["BWRLDAPipeline"]


class BWRLDAPipeline(DecodingPipeline):
    """Bit-wise-reconstruction speller pipeline with a regularised-LDA classifier.

    The BWR *strategy* classifies each code frame (is the target response present or not).
    A command's score is then the correlation of its code with those frame scores. This
    class bundles the whole shallow chain: band-pass IIR + CAR, then epoch each code frame
    (the bit onsets come from the cycle onsets and ``fps_resolution``), then resample, then
    flatten, then regularised LDA. It ships with defaults suited to LDA. :meth:`predict`
    returns the cumulative ``(n_cycles, n_commands)`` correlation matrix
    (:func:`~medusa.pipelines.bci.vep_spellers.decoding.bwr_command_scores`) that the
    :class:`~medusa.pipelines.bci.vep_spellers.decoding.command_decoder.VEPCommandDecoder`
    turns into selections.

    Deep BWR variants (EEGInception, EEGNet) are **separate** pipeline classes, not a
    ``clf=`` option. The model choice changes the preprocessing (band, ``target_fs``,
    epoch shape), so each pipeline owns its own chain. Only the model-agnostic parts are
    shared: :func:`~medusa.pipelines.bci.vep_spellers.decoding.bwr_labels`, the cycle and
    bit-onset helpers, and the Layer-2
    :class:`~medusa.pipelines.bci.vep_spellers.decoding.command_decoder.VEPCommandDecoder`.

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
        add_notch_and_filterbank_settings(s)
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
        bands = apply_notch_and_filterbank(raw, x.fs, cfg["notch_filtering"],
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
