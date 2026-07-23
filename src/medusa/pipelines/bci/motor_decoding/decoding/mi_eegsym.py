"""Deep motor-decoding pipeline with the EEGSym backbone (torch-gated).

Holds :class:`MIEEGSymPipeline`. EEGSym is a convolutional network built for inter-subject
motor-imagery transfer: it processes the two hemispheres symmetrically, so it needs an explicit
left/right channel layout (``hemisphere_pairs`` + ``middle_chs``), not just a channel list.

This module imports torch (through ``TorchClassifier`` and the EEGSym backbone), so it is kept
apart from its torch-free siblings and re-exported **lazily** from the ``decoding`` package: an
``import medusa.pipelines.bci.motor_decoding`` stays torch-free. It is a separate class from the
EEG-Inception pipeline because EEGSym's symmetric layout is part of its contract -- there is no
shared ``arch`` switch that turns one into the other.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from medusa.core.settings_tree import SettingsTree
from medusa.core.serialization import pack_pickleable, unpack_pickleable
from medusa.core.data.recording import Recording

from medusa.ml.torch_models.classification import TorchClassifier
from medusa.ml.torch_models.backbones.eegsym import EEGSym

from medusa.pipelines.base import DecodingPipeline
from medusa.pipelines.bci.trial_events import (
    trial_arrays, trial_labels, validate_trial_events)
from medusa.pipelines.bci._filtering import add_band_filter_settings
from medusa.pipelines.bci.motor_decoding.decoding._common import (
    add_training_settings, training_kwargs, trial_epochs)

__all__ = ["MIEEGSymPipeline"]


class MIEEGSymPipeline(DecodingPipeline):
    """EEGSym motor decoder (motor imagery / motor execution).

    Band-pass + per-trial raw epoch + the EEGSym backbone (a hemisphere-symmetric convolutional
    network) wrapped in a :class:`~medusa.ml.torch_models.classification.TorchClassifier`.
    :meth:`predict` returns the per-trial ``(n_trials, n_classes)`` posterior scores (the head is
    sized to the number of training classes).

    EEGSym splits the montage into left and right hemispheres, so it needs the layout, not just a
    channel list. Set it with two settings:

    * ``hemisphere_pairs`` -- a list of ``{"left": <ch>, "right": <ch>}`` pairs, and
    * ``middle_chs`` -- the midline channels.

    ``channels`` must cover exactly those channels (every pair channel + every middle channel,
    and nothing else); :meth:`check_consistency` enforces it. Configure it through the live
    :attr:`~medusa.core.settings_tree.Configurable.settings` tree or with construction kwargs,
    for example ``MIEEGSymPipeline(channels=[...], hemisphere_pairs=[{"left": "C3", "right":
    "C4"}], middle_chs=["Cz"])``.
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
                    info="Resample epochs to this rate (Hz); 0 to disable (EEGSym needs >= 64 "
                         "samples per epoch)")
        pairs = s.add_group_list(
            "hemisphere_pairs", info="Left/right channel pairs for the bilateral split")
        pairs.element.add_item("left", value="", info="Left-hemisphere channel")
        pairs.element.add_item("right", value="", info="Right-hemisphere channel")
        s.add_item("middle_chs", value=[], info="Midline channels (not paired)")
        clf = s.add_group("classifier", info="EEGSym backbone")
        clf.add_item("filters_per_branch", value=24, value_range=[1, None],
                     info="Convolutional filters per inception branch")
        clf.add_item("scales_ms", value=[500.0, 250.0, 125.0],
                     info="Temporal inception kernel scales (ms)")
        clf.add_item("drop_prob", value=0.4, value_range=[0, 1], info="Dropout probability")
        clf.add_item("activation", value="elu", value_options=["elu", "relu", "leaky_relu"],
                     info="Activation function")
        clf.add_item("spatial_resnet_repetitions", value=1, value_range=[1, None],
                     info="Spatial ResNet block repetitions")
        add_training_settings(clf)
        return s

    # ---- validation ----
    def check_consistency(self, recording: Recording) -> None:
        """Check the signal, channels, ``fs``, the hemisphere layout, and the trial events."""
        cfg = self.cfg
        sig = recording.signals.get(cfg["signal_key"])
        if sig is None:
            raise ValueError(f"recording has no {cfg['signal_key']!r} signal.")
        if not cfg["channels"]:
            raise ValueError("no channels configured; set the 'channels' setting.")
        if not cfg["hemisphere_pairs"]:
            raise ValueError(
                "no hemisphere_pairs configured; EEGSym needs the left/right layout, e.g. "
                "hemisphere_pairs=[{'left': 'C3', 'right': 'C4'}].")
        layout = self._layout_channels(cfg)
        if set(layout) != set(cfg["channels"]):
            raise ValueError(
                "'channels' must cover exactly the hemisphere layout (every hemisphere_pairs "
                f"channel + every middle_chs channel); layout={sorted(set(layout))}, "
                f"channels={sorted(set(cfg['channels']))}.")
        if self.fs is None:
            self.fs = sig.fs
        elif sig.fs != self.fs:
            raise ValueError(f"fs mismatch: pipeline={self.fs}, recording={sig.fs}.")
        missing = [c for c in cfg["channels"] if c not in sig.channel_set.labels]
        if missing:
            raise ValueError(f"recording is missing channels {missing}.")
        validate_trial_events(recording.events)

    @staticmethod
    def _layout_channels(cfg: dict) -> "list[str]":
        """All channels named by the hemisphere layout (pair channels + middle channels)."""
        pair_chs = [c for p in cfg["hemisphere_pairs"] for c in (p["left"], p["right"])]
        return pair_chs + list(cfg["middle_chs"])

    # ---- feature path + backbone ----
    def _epochs(self, recording: Recording, onsets: NDArray, cfg: dict) -> NDArray:
        """Cut and preprocess the raw trial epochs: ``(n_trials, n_samples, n_channels)``."""
        ep = cfg["epoching"]
        return trial_epochs(
            recording.signals[cfg["signal_key"]], onsets,
            channels=cfg["channels"], apply_car=cfg["car"], filter_spec=cfg["filter"],
            window=tuple(ep["w_segment_t"]),
            baseline=tuple(ep["baseline_t"]) if ep["baseline_t"] else None,
            target_fs=ep["target_fs"])

    def _build_backbone(self, cfg: dict, n_samples: int):
        """Build the EEGSym backbone, sized to the epoch length and the hemisphere layout."""
        c = cfg["classifier"]
        rate = cfg["epoching"]["target_fs"] or self.fs
        pairs = [(p["left"], p["right"]) for p in cfg["hemisphere_pairs"]]
        return EEGSym(
            input_samples=n_samples, fs=float(rate), ch_names=list(cfg["channels"]),
            left_right_chs=pairs, middle_chs=list(cfg["middle_chs"]),
            filters_per_branch=int(c["filters_per_branch"]),
            scales_time=tuple(c["scales_ms"]), drop_prob=float(c["drop_prob"]),
            activation=c["activation"],
            spatial_resnet_repetitions=int(c["spatial_resnet_repetitions"]))

    # ---- offline ----
    def fit(self, recordings) -> "MIEEGSymPipeline":
        """Fit the EEGSym classifier on the trial epochs and labels of all recordings."""
        self._check_settings()
        cfg = self.cfg
        X, y = [], []
        for rec in recordings:
            self.check_consistency(rec)
            onsets, _, _ = trial_arrays(rec.events)
            X.append(self._epochs(rec, onsets, cfg))
            y.append(trial_labels(rec))
        X, y = np.concatenate(X), np.concatenate(y)
        backbone = self._build_backbone(cfg, X.shape[1])
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
    def from_pickleable_obj(cls, obj: dict) -> "MIEEGSymPipeline":
        """Rebuild the pipeline from a bundle made by :meth:`to_pickleable_obj`."""
        self = cls(settings=obj["settings"])
        self.fs, self._fitted = obj["fs"], obj["fitted"]
        if self._fitted:
            self.clf = unpack_pickleable(obj["clf"])
        return self
