"""Deep motor-decoding pipeline with an EEG-Inception classifier (torch-gated).

Holds :class:`MIEEGInceptionPipeline`, the convolutional sibling of the shallow
:class:`~medusa.pipelines.bci.motor_decoding.decoding.mi_csp_lda.MICSPLDAPipeline`. It band-
passes the signal, cuts each trial into a raw multichannel epoch, and classifies it with an
EEG-Inception backbone (v1 or v2) wrapped in a
:class:`~medusa.ml.torch_models.classification.TorchClassifier`. Handles two or more classes
(the classifier sizes its head from the labels).

This module imports torch (through ``TorchClassifier`` and the backbones), so it is kept apart
from its torch-free siblings and re-exported **lazily** from the ``decoding`` package (see its
``__getattr__``): ``import medusa.pipelines.bci.motor_decoding`` never pulls torch.

One pipeline serves **both** EEG-Inception v1 and v2: they share the raw-epoch contract and the
same ``TorchClassifier``, so the architecture is a ``classifier.arch`` setting (as in the VEP
:class:`~medusa.pipelines.bci.vep_spellers.decoding.bwr_eeg_inception.BWREEGInceptionPipeline`),
not a separate class -- with one settings group per architecture, built by the shared
:mod:`~medusa.pipelines.bci._torch_backbones`. It is a separate class from the CSP+LDA pipeline, though, because the deep
classifier wants the raw epoch (not CSP log-variance features) -- the model choice cascades into
the feature path.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from medusa.core.settings_tree import SettingsTree
from medusa.core.data.recording import Recording

from medusa.pipelines.torch_base import TorchPipeline, add_training_settings
from medusa.pipelines.bci.trial_events import (
    trial_arrays, trial_labels, validate_trial_events)
from medusa.pipelines.bci._filtering import add_band_filter_settings
from medusa.pipelines.bci._torch_backbones import (
    add_architecture_settings, build_backbone)
from medusa.pipelines.bci.motor_decoding.decoding._common import trial_segments

__all__ = ["MIEEGInceptionPipeline"]


class MIEEGInceptionPipeline(TorchPipeline):
    """EEG-Inception motor decoder (motor imagery / motor execution).

    Band-pass + per-trial raw epoch + an EEG-Inception convolutional classifier. The
    architecture (``'eeg_inception_v1'`` or ``'eeg_inception_v2'``) is the ``classifier.arch``
    setting, and each architecture has its own ``classifier.<arch>`` group of hyper-parameters
    (see :func:`~medusa.pipelines.bci._torch_backbones.add_architecture_settings`), because the
    two do not have the same ones. Training hyper-parameters live in the ``classifier.training``
    subgroup. :meth:`predict` returns the per-trial
    ``(n_trials, n_classes)`` posterior scores (the classifier sizes its head to the number of
    classes in the training labels, so two- and multi-class motor tasks both work).

    The backbone is sized from the data at :meth:`fit` (``input_samples`` from the resampled
    epoch length, ``n_cha`` from the channel count) and saved as a portable config +
    ``state_dict`` bundle. Configure it through the live
    :attr:`~medusa.core.settings_tree.Configurable.settings` tree (see :meth:`default_settings`)
    or with nested construction kwargs, for example
    ``MIEEGInceptionPipeline(channels=["C3", "Cz", "C4"], classifier={"arch": "eeg_inception_v2"})``.
    """

    # ---- configuration schema (SettingsTree) ----
    @classmethod
    def default_settings(cls) -> SettingsTree:
        """A GUI-editable, levelled schema of the pipeline's configuration."""
        s = SettingsTree()
        s.add_item("channels", value=[], info="Channels to decode (required)")
        s.add_item("signal_key", value="eeg", info="Recording stream key to decode")
        s.add_item("car", value=True, info="Common-average reference before filtering")
        add_band_filter_settings(s, cutoff=[4.0, 40.0], order=5)
        seg = s.add_group("segmentation", info="Per-trial segment windowing + resampling")
        seg.add_item("w_segment_t", value=[500.0, 2500.0],
                     info="Segment window relative to each trial onset (ms)")
        seg.add_item("baseline_t", value=[], info="Baseline window (ms); empty to disable")
        seg.add_item("target_fs", value=128.0, optional=True,
                     value_range=[1.0, None],
                     info="Resample segments to this rate (Hz); switch it off to keep "
                          "the native rate")
        clf = s.add_group("classifier", info="EEG-Inception classifier")
        # Trial epochs are seconds long, so the kernels are much wider than a VEP speller's.
        add_architecture_settings(clf, scales_ms=[500.0, 250.0, 125.0])
        add_training_settings(clf, profiles=cls.TRAINING_PROFILES)
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

    # ---- feature path + backbone ----
    def _segments(self, recording: Recording, onsets: NDArray, cfg: dict) -> NDArray:
        """Cut and preprocess the raw trial segments of one recording: ``(n_trials, n_samples, n_channels)``."""
        seg_cfg = cfg["segmentation"]
        return trial_segments(
            recording.signals[cfg["signal_key"]], onsets,
            channels=cfg["channels"], apply_car=cfg["car"], filter_spec=cfg["filter"],
            window=tuple(seg_cfg["w_segment_t"]),
            baseline=tuple(seg_cfg["baseline_t"]) if seg_cfg["baseline_t"] else None,
            target_fs=seg_cfg["target_fs"])

    # ---- offline ----
    def fit(self, recordings) -> "MIEEGInceptionPipeline":
        """Fit the EEG-Inception classifier on the trial segments and labels of all recordings.

        The first call builds the backbone; a later one keeps training the model this
        pipeline already holds, under the configured ``classifier.training.profile``
        (see :class:`~medusa.pipelines.torch_base.TorchPipeline`).
        """
        self._check_settings()
        cfg = self.cfg
        X, y = [], []
        for rec in recordings:
            self.check_consistency(rec)
            onsets, _, _ = trial_arrays(rec.events)
            X.append(self._segments(rec, onsets, cfg))
            y.append(trial_labels(rec))
        return self._fit_classifier(cfg, np.concatenate(X), np.concatenate(y))

    def _build_backbone(self, cfg: dict, X: NDArray):
        """Build the architecture ``classifier.arch`` names, sized to the epochs."""
        # the epoch rate the millisecond kernel scales are measured against
        rate = cfg["segmentation"]["target_fs"] or self.fs
        return build_backbone(cfg["classifier"], input_samples=X.shape[1],
                              n_cha=X.shape[2], rate=rate)

    def predict(self, recording: Recording) -> NDArray:
        """Per-trial ``(n_trials, n_classes)`` posterior class scores for one recording."""
        if not self._fitted:
            raise RuntimeError("pipeline is not fitted; call fit() first.")
        self.check_consistency(recording)
        onsets, _, _ = trial_arrays(recording.events)
        return self.clf.predict_proba(self._segments(recording, onsets, self.cfg))
