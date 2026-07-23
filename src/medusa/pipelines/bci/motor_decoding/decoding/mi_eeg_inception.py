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
not a separate class. It is a separate class from the CSP+LDA pipeline, though, because the deep
classifier wants the raw epoch (not CSP log-variance features) -- the model choice cascades into
the feature path.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from medusa.core.settings_tree import SettingsTree
from medusa.core.serialization import pack_pickleable, unpack_pickleable
from medusa.core.data.recording import Recording

from medusa.ml.torch_models.classification import TorchClassifier
from medusa.ml.torch_models.backbones.eeg_inception import EEGInception
from medusa.ml.torch_models.backbones.eeg_inception_v2 import EEGInceptionV2

from medusa.pipelines.base import DecodingPipeline
from medusa.pipelines.bci.trial_events import (
    trial_arrays, trial_labels, validate_trial_events)
from medusa.pipelines.bci._filtering import add_band_filter_settings
from medusa.pipelines.bci.motor_decoding.decoding._common import (
    add_training_settings, training_kwargs, trial_epochs)

__all__ = ["MIEEGInceptionPipeline"]

#: Selectable EEG-Inception architectures, mapped to their backbone classes.
_ARCHITECTURES = {
    "eeg_inception_v1": EEGInception,
    "eeg_inception_v2": EEGInceptionV2,
}

#: EEG-Inception v1 constraints (v2 validates itself and pools adaptively). See
#: ``BWREEGInceptionPipeline`` for the full rationale: v1 pools the time axis by 2 five times,
#: uses a ``scale // 4`` kernel in block 3, and narrows by ``/ 4`` in block 4.
_V1_MIN_SAMPLES = 32
_V1_MIN_SCALE = 4
_V1_MIN_BRANCH_UNITS = 4


class MIEEGInceptionPipeline(DecodingPipeline):
    """EEG-Inception motor decoder (motor imagery / motor execution).

    Band-pass + per-trial raw epoch + an EEG-Inception convolutional classifier. The
    architecture (``'eeg_inception_v1'`` or ``'eeg_inception_v2'``) is the ``classifier.arch``
    setting; the shared knobs (``scales_ms``, ``filters_per_branch``, ``dropout_rate``) map onto
    each backbone, and ``activation`` is used by v2 only. Training hyper-parameters live in the
    ``classifier.training`` subgroup. :meth:`predict` returns the per-trial
    ``(n_trials, n_classes)`` posterior scores (the classifier sizes its head to the number of
    classes in the training labels, so two- and multi-class motor tasks both work).

    The backbone is sized from the data at :meth:`fit` (``input_samples`` from the resampled
    epoch length, ``n_cha`` from the channel count) and saved as a portable config +
    ``state_dict`` bundle. Configure it through the live
    :attr:`~medusa.core.settings_tree.Configurable.settings` tree (see :meth:`default_settings`)
    or with nested construction kwargs, for example
    ``MIEEGInceptionPipeline(channels=["C3", "Cz", "C4"], classifier={"arch": "eeg_inception_v2"})``.
    """

    fs = None       # sampling rate adopted at fit (fitted state)
    clf = None      # the fitted TorchClassifier (set by fit / restored by load)

    # ---- configuration schema (SettingsTree) ----
    @classmethod
    def default_settings(cls) -> SettingsTree:
        """A GUI-editable, levelled schema of the pipeline's configuration."""
        s = SettingsTree()
        s.add_item("channels", value=[], info="Channels to decode (required)")
        s.add_item("signal_key", value="eeg", info="Recording stream key to decode")
        s.add_item("car", value=True, info="Common-average reference before filtering")
        add_band_filter_settings(s, cutoff=[4.0, 40.0], order=5)
        ep = s.add_group("epoching", info="Per-trial epoch windowing + resampling")
        ep.add_item("w_segment_t", value=[500.0, 2500.0],
                    info="Epoch window relative to each trial onset (ms)")
        ep.add_item("baseline_t", value=[], info="Baseline window (ms); empty to disable")
        ep.add_item("target_fs", value=128.0, value_range=[0, None],
                    info="Resample epochs to this rate (Hz); 0 to disable")
        clf = s.add_group("classifier", info="EEG-Inception classifier")
        clf.add_item("arch", value="eeg_inception_v1", value_options=list(_ARCHITECTURES),
                     info="EEG-Inception architecture (v1 or v2)")
        clf.add_item("scales_ms", value=[500.0, 250.0, 125.0],
                     info="Temporal inception kernel scales (ms); converted to samples at "
                          "build time with target_fs (or the raw fs if target_fs=0)")
        clf.add_item("filters_per_branch", value=8, value_range=[1, None],
                     info="Convolutional filters per inception branch")
        clf.add_item("dropout_rate", value=0.25, value_range=[0, 1], info="Dropout probability")
        clf.add_item("activation", value="elu", value_options=["elu", "relu", "leaky_relu"],
                     info="Activation function (eeg_inception_v2 only)")
        add_training_settings(clf)
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
    def _epochs(self, recording: Recording, onsets: NDArray, cfg: dict) -> NDArray:
        """Cut and preprocess the raw trial epochs of one recording: ``(n_trials, n_samples, n_channels)``."""
        ep = cfg["epoching"]
        return trial_epochs(
            recording.signals[cfg["signal_key"]], onsets,
            channels=cfg["channels"], apply_car=cfg["car"], filter_spec=cfg["filter"],
            window=tuple(ep["w_segment_t"]),
            baseline=tuple(ep["baseline_t"]) if ep["baseline_t"] else None,
            target_fs=ep["target_fs"])

    def _build_backbone(self, cfg: dict, n_samples: int, n_cha: int):
        """Build the EEG-Inception backbone for ``cfg['classifier']['arch']``, sized to the data."""
        c = cfg["classifier"]
        if not c["scales_ms"]:
            raise ValueError("classifier.scales_ms must list at least one temporal scale.")
        rate = cfg["epoching"]["target_fs"] or self.fs
        scales = tuple(max(1, round(ms / 1000.0 * rate)) for ms in c["scales_ms"])
        arch = c["arch"]
        if arch == "eeg_inception_v1":
            fpb = int(c["filters_per_branch"])
            self._check_eeg_inception_v1_dims(n_samples, scales, fpb)
            return EEGInception(input_samples=n_samples, n_cha=n_cha, scales_samples=scales,
                                filters_per_branch=fpb, dropout_rate=float(c["dropout_rate"]))
        if arch == "eeg_inception_v2":
            return EEGInceptionV2(
                input_samples=n_samples, n_cha=n_cha, temp_scales_samples=scales,
                temp_filt_per_branch=int(c["filters_per_branch"]),
                dil_filt_per_branch=int(c["filters_per_branch"]),
                dropout_rate=float(c["dropout_rate"]), activation=c["activation"])
        raise ValueError(
            f"classifier.arch must be one of {list(_ARCHITECTURES)}, got {arch!r}.")

    @staticmethod
    def _check_eeg_inception_v1_dims(n_samples: int, scales: "tuple[int, ...]",
                                     filters_per_branch: int) -> None:
        """Reject configs that would build a degenerate (zero-size) EEG-Inception v1.

        v1 has no internal validation; an over-small epoch, temporal scale, or filter count
        silently builds a zero-size layer that only crashes deep in the first forward pass.
        (Duplicated from the VEP deep pipeline on purpose -- cross-package private imports are
        worse than one repeated guard; fold into a shared helper if a third EEG-Inception
        pipeline appears.)"""
        problems = []
        if n_samples < _V1_MIN_SAMPLES:
            problems.append(
                f"epoch has {n_samples} samples but v1 pools the time axis by 2 five times "
                f"(needs >= {_V1_MIN_SAMPLES}); widen 'epoching.w_segment_t' or raise "
                f"'epoching.target_fs'")
        if min(scales) < _V1_MIN_SCALE:
            problems.append(
                f"smallest temporal scale is {min(scales)} samples but block 3 uses a scale//4 "
                f"kernel (needs every scale >= {_V1_MIN_SCALE}); raise 'classifier.scales_ms' "
                f"or 'epoching.target_fs'")
        if filters_per_branch * len(scales) < _V1_MIN_BRANCH_UNITS:
            problems.append(
                f"filters_per_branch ({filters_per_branch}) * n_scales ({len(scales)}) = "
                f"{filters_per_branch * len(scales)} but block 4 narrows by /4 (needs >= "
                f"{_V1_MIN_BRANCH_UNITS}); raise 'classifier.filters_per_branch'")
        if problems:
            raise ValueError(
                "eeg_inception_v1 cannot be built with this configuration: "
                + "; ".join(problems)
                + ". Alternatively use 'eeg_inception_v2' (adaptive pooling, self-validating).")

    # ---- offline ----
    def fit(self, recordings) -> "MIEEGInceptionPipeline":
        """Fit the EEG-Inception classifier on the trial epochs and labels of all recordings."""
        self._check_settings()
        cfg = self.cfg
        X, y = [], []
        for rec in recordings:
            self.check_consistency(rec)
            onsets, _, _ = trial_arrays(rec.events)
            X.append(self._epochs(rec, onsets, cfg))
            y.append(trial_labels(rec))
        X, y = np.concatenate(X), np.concatenate(y)
        backbone = self._build_backbone(cfg, X.shape[1], X.shape[2])
        self.clf = TorchClassifier(
            backbone, **training_kwargs(cfg["classifier"]["training"])).fit(X, y)
        self._fitted = True
        return self

    def predict(self, recording: Recording) -> NDArray:
        """Per-trial ``(n_trials, n_classes)`` posterior class scores for one recording."""
        if not self._fitted:
            raise RuntimeError("pipeline is not fitted; call fit() first.")
        self.check_consistency(recording)
        onsets, _, _ = trial_arrays(recording.events)
        return self.clf.predict_proba(self._epochs(recording, onsets, self.cfg))

    # ---- persistence (settings + fitted state) ----
    def to_pickleable_obj(self) -> dict:
        """Bundle the settings, the fitted flag, ``fs``, and the packed TorchClassifier."""
        return {"settings": self.settings.to_dict(), "fitted": self._fitted, "fs": self.fs,
                "clf": pack_pickleable(self.clf) if self._fitted else None}

    @classmethod
    def from_pickleable_obj(cls, obj: dict) -> "MIEEGInceptionPipeline":
        """Rebuild the pipeline from a bundle made by :meth:`to_pickleable_obj`."""
        self = cls(settings=obj["settings"])
        self.fs, self._fitted = obj["fs"], obj["fitted"]
        if self._fitted:
            self.clf = unpack_pickleable(obj["clf"])
        return self
