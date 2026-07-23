"""Deep bit-wise-reconstruction speller pipeline with an EEG-Inception classifier (torch-gated).

Holds :class:`BWREEGInceptionPipeline`, the EEG-Inception sibling of the shallow
:class:`~medusa.pipelines.bci.vep_spellers.decoding.bwr_lda.BWRLDAPipeline`: same BWR
*strategy* (classify each code frame, then correlate a command's code with the frame scores),
same cumulative ``(n_cycles, n_commands)`` output, same Layer-2
:class:`~medusa.pipelines.bci.vep_spellers.decoding.command_decoder.VEPCommandDecoder`. Only
the frame classifier changes -- an EEG-Inception convolutional backbone wrapped in a
:class:`~medusa.ml.torch_models.classification.TorchClassifier`, instead of LDA.

This is the only decoding module that imports torch (through ``TorchClassifier`` and the
backbones), so it is kept apart from its torch-free siblings and re-exported **lazily** from
the ``decoding`` package (see its ``__getattr__``): ``import
medusa.pipelines.bci.vep_spellers`` never pulls torch.

Why one pipeline serves **both** EEG-Inception v1 and v2
--------------------------------------------------------
The "one pipeline class per (strategy, classifier)" rule exists because the model choice
cascades into the preprocessing: LDA wants a low ``target_fs`` and a flat feature vector, a
conv net wants the raw ``(n_epochs, n_samples, n_channels)`` epoch. That cascade is real
between *LDA* and *EEG-Inception*, so they are separate classes. It is **absent** between
EEG-Inception *v1* (:class:`~medusa.ml.torch_models.backbones.eeg_inception.EEGInception`)
and *v2* (:class:`~medusa.ml.torch_models.backbones.eeg_inception_v2.EEGInceptionV2`): both
take ``(input_samples, n_cha, ...)``, both consume the same raw epoch through the same
:class:`~medusa.ml.torch_models.classification.TorchClassifier`, and both expose
``backbone_features`` / ``get_config``. The whole feature path is identical; only the backbone
constructor and its hyper-parameters differ. So the architecture is a **configuration option**
(``classifier.arch``), exactly as the LDA shrinkage is an option of ``BWRLDAPipeline`` -- not
a new class.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from medusa.core.settings_tree import SettingsTree
from medusa.core.serialization import pack_pickleable, unpack_pickleable
from medusa.core.data.recording import Recording
from medusa.core.data.signal import Signal
from medusa.signal.spatial_filtering import car
from medusa.signal.segmentation import segment_signal_around_events, resample_segments

from medusa.ml.torch_models.classification import TorchClassifier
from medusa.ml.torch_models.backbones.eeg_inception import EEGInception
from medusa.ml.torch_models.backbones.eeg_inception_v2 import EEGInceptionV2

from medusa.pipelines.base import DecodingPipeline, harmonize_channels
from medusa.pipelines.bci.vep_spellers.data import SpellerData, validate_speller_events
from medusa.pipelines.bci._filtering import (
    add_notch_and_filterbank_settings, apply_notch_and_filterbank)
from medusa.pipelines.bci.vep_spellers.decoding._common import (
    _bit_onsets, _cycle_arrays)
from medusa.pipelines.bci.vep_spellers.decoding.scores import (
    bwr_labels, bwr_command_scores)

__all__ = ["BWREEGInceptionPipeline"]

#: Selectable EEG-Inception architectures, mapped to their backbone classes.
_ARCHITECTURES = {
    "eeg_inception_v1": EEGInception,
    "eeg_inception_v2": EEGInceptionV2,
}

#: EEG-Inception v1 constraints that make its derived conv kernels / channels degenerate
#: (a zero-size dimension) for extreme configs. v2 validates its own inputs and pools
#: adaptively, so it has no equivalent floors.
#:
#: * ``_V1_MIN_SAMPLES`` -- v1 pools the temporal axis by 2 five times (pool1/2/3 + the two
#:   block-4 pools); fewer than ``2**5`` samples collapses a feature map to zero.
#: * ``_V1_MIN_SCALE`` -- block 3 uses a ``scale // 4`` temporal kernel, so a scale below 4
#:   samples yields a zero-height kernel.
#: * ``_V1_MIN_BRANCH_UNITS`` -- block 4 narrows to ``int(filters_per_branch * n_scales / 4)``
#:   channels, which rounds to 0 when ``filters_per_branch * n_scales < 4``.
_V1_MIN_SAMPLES = 32
_V1_MIN_SCALE = 4
_V1_MIN_BRANCH_UNITS = 4


class BWREEGInceptionPipeline(DecodingPipeline):
    """Bit-wise-reconstruction speller pipeline with an EEG-Inception classifier.

    The deep sibling of
    :class:`~medusa.pipelines.bci.vep_spellers.decoding.bwr_lda.BWRLDAPipeline`. It runs the
    same BWR *strategy* -- classify each code frame (is the target response present or not),
    then score each command by the correlation of its code with the frame scores -- and
    returns the same cumulative ``(n_cycles, n_commands)`` correlation matrix
    (:func:`~medusa.pipelines.bci.vep_spellers.decoding.bwr_command_scores`) for the
    :class:`~medusa.pipelines.bci.vep_spellers.decoding.command_decoder.VEPCommandDecoder`.
    The only change is the frame classifier: a convolutional EEG-Inception backbone (v1 or
    v2) wrapped in a :class:`~medusa.ml.torch_models.classification.TorchClassifier`.

    A **single** pipeline serves both architectures because they share one
    preprocessing/epoch contract; the architecture is picked with the ``classifier.arch``
    setting (``'eeg_inception_v1'`` or ``'eeg_inception_v2'``). The shared classifier knobs
    (``scales_ms``, ``filters_per_branch``, ``dropout_rate``) map onto each backbone's
    constructor; ``activation`` is used by v2 only. Training hyper-parameters live in the
    ``classifier.training`` subgroup.

    Two things differ from the LDA pipeline, both forced by the deep classifier (which is
    exactly why deep BWR is a separate class):

    * **Raw epochs.** Features are the per-frame epochs kept as
      ``(n_epochs, n_samples, n_channels)`` (not flattened), resampled to
      ``epoching.target_fs`` (default 128 Hz, EEG-Inception's design rate).
    * **Single band.** A conv backbone consumes one multichannel epoch, so it cannot fuse a
      parallel filter bank the way the LDA pipeline concatenates sub-band features. The
      ``freq_filtering`` schema is kept for consistency, but the filter bank must hold exactly
      **one** band-pass (checked in :meth:`check_consistency`); the default (one 1--70 Hz
      band) already satisfies it.

    Configure it through the live
    :attr:`~medusa.core.settings_tree.Configurable.settings` tree (see
    :meth:`default_settings`), or with construction kwargs, which may be nested, for example
    ``BWREEGInceptionPipeline(channels=["Fz", "Cz", "Pz", "Oz"], classifier={"arch":
    "eeg_inception_v2"})``. The backbone is sized from the data at :meth:`fit` time
    (``input_samples`` from the resampled epoch length, ``n_cha`` from the channel count) and
    saved as a portable config + ``state_dict`` bundle.
    """

    fs = None       # sampling rate adopted at fit (fitted state)
    clf = None      # the fitted TorchClassifier (set by fit / restored by load)

    # ---- configuration schema (SettingsTree) ----
    @classmethod
    def default_settings(cls) -> SettingsTree:
        """A GUI-editable, levelled schema of the pipeline's configuration.

        Mirrors :meth:`BWRLDAPipeline.default_settings
        <medusa.pipelines.bci.vep_spellers.decoding.bwr_lda.BWRLDAPipeline.default_settings>`
        (``channels``, ``signal_key``, ``car``, ``freq_filtering``, ``epoching``) and swaps
        the ``classifier`` group for the EEG-Inception configuration: the ``arch`` selector,
        the shared architecture knobs, and a ``training`` subgroup.
        """
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
        ep.add_item("target_fs", value=128.0, value_range=[0, None],
                    info="Resample epochs to this rate (Hz); 0 to disable")
        clf = s.add_group("classifier", info="EEG-Inception frame classifier")
        clf.add_item("arch", value="eeg_inception_v1",
                     value_options=list(_ARCHITECTURES),
                     info="EEG-Inception architecture (v1 or v2)")
        clf.add_item("scales_ms", value=[500.0, 250.0, 125.0],
                     info="Temporal inception kernel scales (ms); converted to samples at "
                          "build time with target_fs (or the raw signal fs if target_fs=0)")
        clf.add_item("filters_per_branch", value=8, value_range=[1, None],
                     info="Convolutional filters per inception branch")
        clf.add_item("dropout_rate", value=0.25, value_range=[0, 1],
                     info="Dropout probability")
        clf.add_item("activation", value="elu",
                     value_options=["elu", "relu", "leaky_relu"],
                     info="Activation function (eeg_inception_v2 only)")
        tr = clf.add_group("training", info="TorchClassifier training hyper-parameters")
        tr.add_item("max_epochs", value=100, value_range=[1, None],
                    info="Maximum training epochs")
        tr.add_item("batch_size", value=64, value_range=[1, None], info="Mini-batch size")
        tr.add_item("learning_rate", value=1e-3, value_range=[0, None],
                    info="Adam learning rate")
        tr.add_item("val_split", value=0.2, value_range=[0, 1],
                    info="Validation fraction for early stopping (0 to disable)")
        tr.add_item("patience", value=10, value_range=[1, None],
                    info="Early-stopping patience (epochs)")
        tr.add_item("device", value="auto",
                    info="Compute device ('auto', 'cpu', 'cuda', 'cuda:N', 'mps'); keep "
                         "'auto' so a saved model reloads on any host (a fixed 'cuda' "
                         "fails to load on a CPU-only machine)")
        tr.add_item("verbose", value="epoch",
                    value_options=["silent", "epoch", "full"],
                    info="Training log: 'silent' / 'epoch' (one line per epoch) / "
                         "'full' (Lightning debug)")
        return s

    # ---- validation ----
    def check_consistency(self, recording: Recording) -> None:
        """Check the recording has the configured signal and channels, a matching ``fs``,
        a single-band filter bank, a :class:`SpellerData`, and valid speller events; raise
        ``ValueError`` if not."""
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
        filterbank = cfg["freq_filtering"]["filterbank"]
        if len(filterbank) != 1:
            raise ValueError(
                f"deep BWR uses a single band-pass, not a filter bank; "
                f"freq_filtering.filterbank has {len(filterbank)} filters. A conv "
                f"backbone cannot fuse parallel sub-bands the way the LDA pipeline "
                f"concatenates features -- configure exactly one band.")
        SpellerData.from_recording(recording)      # experiment must be a SpellerData
        validate_speller_events(recording.events)

    # ---- backbone construction ----
    def _build_backbone(self, cfg: dict, n_samples: int, n_cha: int):
        """Build the EEG-Inception backbone for ``cfg['classifier']['arch']``.

        ``input_samples`` and ``n_cha`` come from the actual feature array at fit, so the
        backbone can never desync from the data. The shared ``scales_ms`` knob is converted
        to samples with the epoch rate (``target_fs``, or the raw ``fs`` when resampling is
        disabled) and mapped onto each backbone's temporal-scale argument.
        """
        c = cfg["classifier"]
        if not c["scales_ms"]:
            raise ValueError("classifier.scales_ms must list at least one temporal scale.")
        rate = cfg["epoching"]["target_fs"] or self.fs
        scales = tuple(max(1, round(ms / 1000.0 * rate)) for ms in c["scales_ms"])
        arch = c["arch"]
        if arch == "eeg_inception_v1":
            filters_per_branch = int(c["filters_per_branch"])
            self._check_eeg_inception_v1_dims(n_samples, scales, filters_per_branch)
            return EEGInception(
                input_samples=n_samples, n_cha=n_cha, scales_samples=scales,
                filters_per_branch=filters_per_branch,
                dropout_rate=float(c["dropout_rate"]))
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

        v1 has no internal validation, so an over-small epoch, temporal scale, or filter
        count silently builds a zero-height kernel or zero-channel layer that only crashes
        deep inside the first forward pass. This turns those into clear, actionable errors
        (v2 validates itself and pools adaptively, so it needs none of this)."""
        problems = []
        if n_samples < _V1_MIN_SAMPLES:
            problems.append(
                f"epoch has {n_samples} samples but v1 pools the time axis by 2 five times "
                f"(needs >= {_V1_MIN_SAMPLES}); widen 'epoching.w_segment_t' or raise "
                f"'epoching.target_fs'")
        if min(scales) < _V1_MIN_SCALE:
            problems.append(
                f"smallest temporal scale is {min(scales)} samples but block 3 uses a "
                f"scale//4 kernel (needs every scale >= {_V1_MIN_SCALE}); raise "
                f"'classifier.scales_ms' or 'epoching.target_fs'")
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

    @staticmethod
    def _training_kwargs(cfg: dict) -> dict:
        """Map the ``classifier.training`` settings to ``TorchClassifier`` kwargs."""
        t = cfg["classifier"]["training"]
        return dict(lr=float(t["learning_rate"]), max_epochs=int(t["max_epochs"]),
                    batch_size=int(t["batch_size"]),
                    val_split=t["val_split"] or None, patience=int(t["patience"]),
                    device=t["device"], verbose=t["verbose"])

    # ---- feature path (shared by fit/predict) ----
    def _features(self, signal: Signal, cycle_onsets: NDArray,
                  n_frames: int, fps: float, cfg: dict) -> NDArray:
        """Per-frame epochs kept as ``(n_epochs, n_samples, n_channels)`` (cycle-major).

        The same CAR + band-pass + per-frame epoching + resampling as
        :meth:`BWRLDAPipeline._features`, but the epochs are **not** flattened (the conv
        backbone consumes them raw), and only the single configured band is used.
        """
        x = harmonize_channels(signal, cfg["channels"])
        raw = car(x.signal) if cfg["car"] else x.signal
        # Single band (validated in check_consistency): use the sole filter-bank entry.
        xf = apply_notch_and_filterbank(raw, x.fs, cfg["notch_filtering"],
                                        cfg["freq_filtering"]["filterbank"])[0]
        onsets = _bit_onsets(cycle_onsets, n_frames, fps)
        ep = cfg["epoching"]
        window = tuple(ep["w_segment_t"])
        baseline = tuple(ep["baseline_t"]) if ep["baseline_t"] else None
        seg = segment_signal_around_events(
            x.times, xf, onsets, x.fs, window, baseline,
            norm="dc" if baseline is not None else None)
        if ep["target_fs"]:
            seg = resample_segments(seg, window, ep["target_fs"])
        return seg

    def _frame_scores(self, recording: Recording, cfg: dict) -> NDArray:
        """Per-frame target-class scores for one recording (cycle-major order)."""
        sd = SpellerData.from_recording(recording)
        onsets, _, _, _ = _cycle_arrays(recording.events)
        feats = self._features(recording.signals[cfg["signal_key"]], onsets,
                               sd.codes.shape[2], sd.fps_resolution, cfg)
        return self.clf.predict_proba(feats)[:, 1]

    # ---- offline ----
    def fit(self, recordings) -> "BWREEGInceptionPipeline":
        """Fit the EEG-Inception classifier on the per-frame BWR features and labels of all
        recordings; return ``self``.

        Gathers the raw per-frame epochs and their target/non-target labels, builds the
        backbone sized to the epoch shape, wraps it in a
        :class:`~medusa.ml.torch_models.classification.TorchClassifier`, and trains.
        """
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
        backbone = self._build_backbone(cfg, X.shape[1], X.shape[2])
        self.clf = TorchClassifier(backbone, **self._training_kwargs(cfg)).fit(X, y)
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
        """Bundle the settings, the fitted flag, ``fs``, and the packed TorchClassifier
        (a portable config + CPU ``state_dict``) for saving."""
        return {"settings": self.settings.to_dict(), "fitted": self._fitted,
                "fs": self.fs,
                "clf": pack_pickleable(self.clf) if self._fitted else None}

    @classmethod
    def from_pickleable_obj(cls, obj: dict) -> "BWREEGInceptionPipeline":
        """Rebuild the pipeline from a bundle made by :meth:`to_pickleable_obj`."""
        self = cls(settings=obj["settings"])
        self.fs, self._fitted = obj["fs"], obj["fitted"]
        if self._fitted:
            self.clf = unpack_pickleable(obj["clf"])
        return self
