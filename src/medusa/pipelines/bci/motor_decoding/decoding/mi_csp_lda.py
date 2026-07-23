"""Motor-decoding pipeline: Common Spatial Patterns + regularised LDA.

The classic shallow motor-imagery / motor-execution decoder (:class:`MICSPLDAPipeline`). It
band-passes the sensorimotor rhythm, learns Common Spatial Patterns that separate the classes,
takes log-variance features, and classifies them with a regularised LDA. It is one ready-to-use
workflow with sensible motor defaults, not a configurable framework; the reusable pieces (CSP,
filters, LDA) come from :mod:`medusa.signal` and scikit-learn.

Deep motor decoders (EEG-Inception, EEGSym) are **separate** pipeline classes, not a ``clf=``
option -- the model choice changes the band, the epoch shape and the resampling rate, so each
pipeline owns its whole chain. Only the model-agnostic parts are shared, as free functions in
:mod:`~medusa.pipelines.bci.trial_events` (labels) and
:mod:`~medusa.pipelines.bci.motor_decoding.decoding._common` (epochs, log-variance).
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from medusa.core.settings_tree import SettingsTree
from medusa.core.serialization import pack_pickleable, unpack_pickleable
from medusa.core.data.recording import Recording
from medusa.signal.spatial_filtering import CSP

from medusa.pipelines.base import DecodingPipeline
from medusa.pipelines.bci.trial_events import (
    trial_arrays, trial_labels, validate_trial_events)
from medusa.pipelines.bci._filtering import add_band_filter_settings
from medusa.pipelines.bci.motor_decoding.decoding._common import (
    trial_epochs, log_var)

__all__ = ["MICSPLDAPipeline"]


class MICSPLDAPipeline(DecodingPipeline):
    """CSP + regularised-LDA motor decoder (motor imagery / motor execution).

    The whole shallow chain in one pipeline: band-pass IIR + CAR, epoch each trial, resample,
    learn Common Spatial Patterns, take log-variance features, classify with a regularised LDA.
    It ships defaults suited to sensorimotor-rhythm decoding (an 8--30 Hz band, ``C3``/``C4``-
    style montage). :meth:`predict` returns the per-trial ``(n_trials, n_classes)`` posterior
    scores; those scores are the output (no Layer-2 decoder). The argmax to a class and the
    accuracy report are caller-side / analysis, not part of :meth:`fit`.

    It handles more than two classes: set ``csp.selection`` to ``"eigenvalues"`` (the
    ``"extremes"`` default is the classic two-class CSP). Configure it through the live
    :attr:`~medusa.core.settings_tree.Configurable.settings` tree (see :meth:`default_settings`)
    or with construction kwargs, which may be nested, for example
    ``MICSPLDAPipeline(channels=["C3", "Cz", "C4"], csp={"n_filters": 6})``.
    """

    fs = None       # sampling rate adopted at fit (fitted state)
    csp = None      # the fitted CSP (set by fit / restored by load)
    clf = None      # the fitted LDA (set by fit / restored by load)

    # ---- configuration schema (SettingsTree) ----
    @classmethod
    def default_settings(cls) -> SettingsTree:
        """A GUI-editable, levelled schema of the pipeline's configuration."""
        s = SettingsTree()
        s.add_item("channels", value=[], info="Channels to decode (required)")
        s.add_item("signal_key", value="eeg", info="Recording stream key to decode")
        # Off by default: CSP already spatially filters, and a CAR before it removes one
        # rank from the montage, which makes the covariance singular (the generalised CSP
        # eigensolve then fails). Turn it on only with a large, full-rank montage.
        s.add_item("car", value=False, info="Common-average reference before filtering")
        add_band_filter_settings(s, cutoff=[8.0, 30.0], order=5)
        ep = s.add_group("epoching", info="Per-trial epoch windowing + resampling")
        ep.add_item("w_segment_t", value=[0.0, 2000.0],
                    info="Epoch window relative to each trial onset (ms)")
        ep.add_item("baseline_t", value=[-1000.0, 0.0],
                    info="Baseline window (ms); empty to disable")
        ep.add_item("target_fs", value=60.0, value_range=[0, None],
                    info="Resample epochs to this rate (Hz); 0 to disable")
        csp = s.add_group("csp", info="Common Spatial Patterns")
        csp.add_item("n_filters", value=4, value_range=[1, None],
                     info="Number of CSP filters")
        csp.add_item("selection", value="extremes",
                     value_options=["extremes", "eigenvalues"],
                     info="Filter selection ('extremes' = 2-class, 'eigenvalues' = N-class)")
        csp.add_item("normalize_log_vars", value=False,
                     info="Normalise log-variance features per trial")
        clf = s.add_group("classifier", info="Regularised-LDA classifier")
        clf.add_item("shrinkage", value="auto", info="LDA shrinkage ('auto' or a float)")
        return s

    # ---- validation ----
    def check_consistency(self, recording: Recording) -> None:
        """Check the recording has the configured signal and channels, a matching ``fs``, and
        valid trial events; raise ``ValueError`` if not."""
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
        validate_trial_events(recording.events)

    # ---- feature path (shared by fit/predict) ----
    def _epochs(self, recording: Recording, onsets: NDArray, cfg: dict) -> NDArray:
        """Cut and preprocess the trial epochs of one recording: ``(n_trials, n_samples, n_channels)``."""
        ep = cfg["epoching"]
        return trial_epochs(
            recording.signals[cfg["signal_key"]], onsets,
            channels=cfg["channels"], apply_car=cfg["car"], filter_spec=cfg["filter"],
            window=tuple(ep["w_segment_t"]),
            baseline=tuple(ep["baseline_t"]) if ep["baseline_t"] else None,
            target_fs=ep["target_fs"])

    def _features(self, epochs: NDArray, cfg: dict) -> NDArray:
        """Project the epochs through the fitted CSP and take log-variance features."""
        return log_var(self.csp.project(epochs), cfg["csp"]["normalize_log_vars"])

    # ---- offline ----
    def fit(self, recordings) -> "MICSPLDAPipeline":
        """Fit the CSP and LDA on the trials of all recordings; return ``self``."""
        self._check_settings()          # re-validate (the live tree may have been edited)
        cfg = self.cfg
        epochs, labels = [], []
        for rec in recordings:
            self.check_consistency(rec)
            onsets, _, _ = trial_arrays(rec.events)
            epochs.append(self._epochs(rec, onsets, cfg))
            labels.append(trial_labels(rec))
        X, y = np.concatenate(epochs), np.concatenate(labels)
        self.csp = CSP(n_filters=cfg["csp"]["n_filters"],
                       selection=cfg["csp"]["selection"])
        self.csp.fit(X, y)
        self.clf = LinearDiscriminantAnalysis(
            solver="lsqr", shrinkage=cfg["classifier"]["shrinkage"]).fit(
            self._features(X, cfg), y)
        self._fitted = True
        return self

    def predict(self, recording: Recording) -> NDArray:
        """Per-trial ``(n_trials, n_classes)`` posterior class scores for one recording."""
        if not self._fitted:
            raise RuntimeError("pipeline is not fitted; call fit() first.")
        self.check_consistency(recording)
        onsets, _, _ = trial_arrays(recording.events)
        return self.clf.predict_proba(self._features(self._epochs(recording, onsets, self.cfg),
                                                      self.cfg))

    # ---- persistence (settings + fitted state) ----
    def to_pickleable_obj(self) -> dict:
        """Bundle the settings, the fitted flag, ``fs``, and the packed CSP + LDA for saving."""
        return {"settings": self.settings.to_dict(), "fitted": self._fitted, "fs": self.fs,
                "csp": pack_pickleable(self.csp) if self._fitted else None,
                "clf": pack_pickleable(self.clf) if self._fitted else None}

    @classmethod
    def from_pickleable_obj(cls, obj: dict) -> "MICSPLDAPipeline":
        """Rebuild the pipeline from a bundle made by :meth:`to_pickleable_obj`."""
        self = cls(settings=obj["settings"])
        self.fs, self._fitted = obj["fs"], obj["fitted"]
        if self._fitted:
            self.csp = unpack_pickleable(obj["csp"])
            self.clf = unpack_pickleable(obj["clf"])
        return self
