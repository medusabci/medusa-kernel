"""Deep bit-wise-reconstruction speller pipeline with an EEG-Inception classifier (torch-gated).

Holds :class:`BWREEGInceptionPipeline`, the EEG-Inception sibling of the shallow
:class:`~medusa.pipelines.bci.vep_spellers.decoding.bwr_lda.BWRLDAPipeline`: same BWR
*strategy* (classify each code frame, then correlate a command's code with the frame scores),
same cumulative ``(n_cycles, n_commands)`` output, same Layer-2
:func:`~medusa.pipelines.bci.vep_spellers.decoding.command_decoder.select_commands`. Only
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
conv net wants the raw ``(n_segments, n_samples, n_channels)`` epoch. That cascade is real
between *LDA* and *EEG-Inception*, so they are separate classes. It is **absent** between
EEG-Inception *v1* (:class:`~medusa.ml.torch_models.backbones.eeg_inception.EEGInception`)
and *v2* (:class:`~medusa.ml.torch_models.backbones.eeg_inception_v2.EEGInceptionV2`): both
take ``(input_samples, n_cha, ...)``, both consume the same raw epoch through the same
:class:`~medusa.ml.torch_models.classification.TorchClassifier`, and both expose
``backbone_features`` / ``get_config``. The whole feature path is identical; only the backbone
constructor and its hyper-parameters differ. So the architecture is a **configuration option**
(``classifier.arch``), exactly as the LDA shrinkage is an option of ``BWRLDAPipeline`` -- not
a new class. Those hyper-parameters get one group each
(:mod:`~medusa.pipelines.bci._torch_backbones`), because that is the one place the two
architectures genuinely disagree.

Configuration profiles: one per stimulation paradigm
----------------------------------------------------
What the *stimulation* looks like does cascade into the configuration, though. A dense
m-sequence code puts a new bit on every display frame, so the responses overlap into one
continuous signal; a burst code puts a few short flashes into a long code and leaves quiet
gaps between them, so each flash evokes its own complete transient response. Those two
signals want a different band, a different baseline and different kernel scales -- but the
same strategy, the same feature path and the same class. So the paradigm is a set of
**values**, not a new pipeline: it ships as a profile, exactly like
:mod:`~medusa.pipelines.bci.vep_spellers.decoding.template_matching`'s.
:func:`mseq_cvep_settings` and :func:`burst_cvep_settings` are the two ready-made ones, both
built from the general :func:`bwr_eeg_inception_settings`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from medusa.core.settings_tree import SettingsTree
from medusa.core.data.recording import Recording
from medusa.core.data.signal import Signal
from medusa.signal.spatial_filtering import car
from medusa.signal.segmentation import segment_signal_around_events, resample_segments

from medusa.pipelines.base import harmonize_channels
from medusa.pipelines.torch_base import TorchPipeline, add_training_settings
from medusa.pipelines.bci.vep_spellers.data import (
    SpellerData, validate_speller_events, cycle_arrays)
from medusa.pipelines.bci._filtering import (
    add_notch_and_filterbank_settings, apply_notch_and_filterbank)
from medusa.pipelines.bci._torch_backbones import (
    add_architecture_settings, build_backbone)
from medusa.pipelines.bci.vep_spellers.decoding._common import _bit_onsets
from medusa.pipelines.bci.vep_spellers.decoding.scores import (
    bwr_labels, bwr_command_scores)

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["BWREEGInceptionPipeline", "bwr_eeg_inception_settings",
           "mseq_cvep_settings", "burst_cvep_settings"]

# --------------------------------------------------------------------------- #
# Configuration profiles
# --------------------------------------------------------------------------- #
# A profile is a named function that returns a ready settings tree for one stimulation
# paradigm. The values it picks become the tree's DEFAULTS (not user edits), so `reset()`
# returns to the profile and `user_overrides()` reports only what you changed on top of it.
# The profile records its name in the tree's `profile` leaf, which is PROVENANCE ONLY: no
# code reads it -- `fit` and `predict` run the same chain whatever the name says.
def bwr_eeg_inception_settings(
        *, profile: "str | None" = None,
        band: "Sequence[float]" = (1.0, 60.0), order: int = 7,
        w_segment_t: "Sequence[float]" = (0.0, 500.0),
        baseline_t: "Sequence[float] | None" = None,
        target_fs: "float | None" = 128.0,
        arch: str = "eeg_inception_v1",
        scales_ms: "Sequence[float]" = (100.0, 75.0, 50.0)) -> SettingsTree:
    """Build a :class:`BWREEGInceptionPipeline` schema with the given band, window and backbone.

    The general builder the paradigm profiles below are built from. Call it directly to
    write a recipe of your own; the profiles are the two that ship ready-made. Its arguments
    are the settings a stimulation paradigm or a backbone choice decides; the training
    hyper-parameters keep the schema defaults, which you edit on the tree (or pass as
    construction kwargs) like any other setting.

    Parameters
    ----------
    profile :
        Name recorded in the ``profile`` leaf, to say which recipe these settings came
        from. Only the shipped profiles pass it; a hand-written recipe leaves it ``None``.
        It is a label, never a switch -- see the ``profile`` leaf's own description.
    band :
        Band-pass cutoffs ``(low, high)`` in Hz. One band, not a bank: a conv backbone
        reads one multichannel epoch, so it cannot fuse parallel sub-bands the way the LDA
        pipeline concatenates their features
        (:meth:`BWREEGInceptionPipeline.check_consistency` rejects a longer bank).
    order :
        Band-pass filter order.
    w_segment_t :
        Segment window ``(start, stop)`` around each code-frame onset, in ms.
    baseline_t :
        Baseline window ``(start, stop)`` in ms, removed from every segment. ``None`` (the
        default) ships the baseline switched **off**, keeping ``(-200, 0)`` ms as the value
        it takes when you switch it on.
    target_fs :
        Rate the segments are resampled to, in Hz. ``None`` ships the resampling switched
        **off**, so the epochs keep the recording rate, and 128 Hz stays as the value it
        takes when you switch it on.
    arch :
        EEG-Inception architecture: ``'eeg_inception_v1'`` or ``'eeg_inception_v2'``.
        Every architecture gets its own ``classifier.<arch>`` group of hyper-parameters;
        this says which one :meth:`BWREEGInceptionPipeline.fit` builds from. The rest of
        those hyper-parameters are edited on the tree (or passed as nested construction
        kwargs, e.g. ``classifier={"eeg_inception_v2": {"n_spatial_filt_mult": 3}}``),
        because which ones exist depends on the architecture.
    scales_ms :
        Temporal inception kernel scales in ms, applied as the default to **every**
        architecture's group. They become samples at build time, using the epoch rate, so
        they mean the same thing whatever ``target_fs`` is.

    Returns
    -------
    SettingsTree
        A fresh tree. Pass it as ``BWREEGInceptionPipeline(settings=...)``.

    Raises
    ------
    ValueError
        If ``arch`` is not one of the shipped EEG-Inception architectures.

    Examples
    --------
    >>> from medusa.pipelines.bci.vep_spellers.decoding import (
    ...     BWREEGInceptionPipeline, bwr_eeg_inception_settings)     # doctest: +SKIP
    >>> s = bwr_eeg_inception_settings(band=(1.0, 45.0), arch="eeg_inception_v2",
    ...                                target_fs=128.0)              # doctest: +SKIP
    >>> s.to_dict()["classifier"]["arch"]                            # doctest: +SKIP
    'eeg_inception_v2'
    >>> pipe = BWREEGInceptionPipeline(settings=s, channels=channels)   # doctest: +SKIP
    """
    s = SettingsTree()
    s.add_item("profile", value=profile,
               info="Which profile these settings came from (provenance only -- no code "
                    "reads it; the pipeline runs the same chain whatever it says). None "
                    "means hand-written. Edit the settings below and this name no longer "
                    "describes them: check settings.user_overrides() for the difference")
    s.add_item("channels", value=[], info="Channels to decode (required)")
    s.add_item("signal_key", value="eeg", info="Recording stream key to decode")
    s.add_item("car", value=True, info="Common-average reference before filtering")
    add_notch_and_filterbank_settings(s, bands=[list(band)], order=order)
    seg = s.add_group("segmentation", info="Per-frame segment windowing + resampling")
    seg.add_item("w_segment_t",
                 value=[float(t) for t in w_segment_t],
                 info="Segment window relative to each frame onset (ms)")
    seg.add_item("baseline_t",
                 value=[float(t) for t in baseline_t] if baseline_t else [-200.0, 0.0],
                 optional=True,
                 enabled=bool(baseline_t),
                 info="Baseline window (ms); switch it off to leave the segments as they are")
    seg.add_item("target_fs",
                 value=float(target_fs) if target_fs else 128.0,
                 optional=True,
                 enabled=bool(target_fs),
                 value_range=[1.0, None],
                 info="Resample segments to this rate (Hz); "
                      "switch it off to keep the native rate")
    clf = s.add_group("classifier",
                      info="EEG-Inception frame classifier")
    add_architecture_settings(clf, arch=arch, scales_ms=scales_ms)
    add_training_settings(clf, profiles=BWREEGInceptionPipeline.TRAINING_PROFILES,
                          max_epochs=500, batch_size=512)
    return s


def mseq_cvep_settings(*, band: "Sequence[float]" = (1.0, 60.0), order: int = 7,
                       w_segment_t: "Sequence[float]" = (0.0, 500.0),
                       arch: str = "eeg_inception_v1") -> SettingsTree:
    """Settings for **m-sequence c-VEP**: one code bit on every display frame.

    The dense c-VEP paradigm
    (:func:`~medusa.pipelines.bci.vep_spellers.encoding.generate_mseq_codebook`, and equally
    the Gold and random codebooks): the command flickers on and off at the display rate, so
    a new bit starts every frame -- about every 17 ms at 60 fps -- and the responses to
    consecutive bits pile on top of each other into one continuous signal. The classifier
    never sees a single clean response; it learns to read the next bit out of that mixture.
    Roughly half the frames are targets, so training sees a balanced problem.

    Needs :meth:`BWREEGInceptionPipeline.fit` on calibration recordings.

    The values below (baseline, resampling rate and kernel scales) are what this profile
    pins for the paradigm; the arguments are the ones that follow your own setup.

    Parameters
    ----------
    band :
        Band-pass cutoffs ``(low, high)`` in Hz. The useful upper cutoff follows your
        stimulation frame rate, so this stays yours to choose.
    order :
        Band-pass filter order.
    w_segment_t :
        Segment window ``(start, stop)`` around each frame onset, in ms.
    arch :
        EEG-Inception architecture: ``'eeg_inception_v1'`` or ``'eeg_inception_v2'``.

    Returns
    -------
    SettingsTree
        A fresh tree. Pass it as ``BWREEGInceptionPipeline(settings=...)``.

    Raises
    ------
    ValueError
        If ``arch`` is not one of the shipped EEG-Inception architectures.

    Examples
    --------
    >>> from medusa.pipelines.bci.vep_spellers.decoding import (
    ...     BWREEGInceptionPipeline, mseq_cvep_settings)             # doctest: +SKIP
    >>> s = mseq_cvep_settings()                                     # doctest: +SKIP
    >>> pipe = BWREEGInceptionPipeline(settings=s, channels=channels)   # doctest: +SKIP
    >>> scores = pipe.fit(train).predict(recording)                     # doctest: +SKIP
    """
    return bwr_eeg_inception_settings(
        profile="mseq_cvep", band=band, order=order, w_segment_t=w_segment_t, arch=arch,
        baseline_t=None, target_fs=128.0, scales_ms=(100.0, 75.0, 50.0))


def burst_cvep_settings(*, band: "Sequence[float]" = (1.0, 60.0), order: int = 7,
                        w_segment_t: "Sequence[float]" = (0.0, 500.0),
                        arch: str = "eeg_inception_v1") -> SettingsTree:
    """Settings for **burst c-VEP**: short flashes separated by long quiet gaps.

    The sparse c-VEP paradigm: instead of a bit on every frame, a command's code holds a
    handful of one- or two-frame flashes with hundreds of ms of nothing in between. Each
    burst is far enough from the next to evoke its own complete transient VEP, much like an
    ERP, instead of the continuous mixture a dense code produces.

    Needs :meth:`BWREEGInceptionPipeline.fit` on calibration recordings.

    Two things to keep in mind, both of them consequences of a sparse code:

    * **Keep the window inside the gap between bursts.** Bit-wise reconstruction labels a
      segment with the code value at its onset, so a window longer than the shortest gap
      between two bursts of the same code puts a second burst's response inside the segment.
    * **The classes are very unbalanced.** Every frame becomes a training segment, but only
      the burst frames are targets, so the classifier sees many more non-target segments
      than target ones and leans towards the non-target class. That does not break the
      command scores, which are correlations and so ignore any constant bias, but it does
      make training harder.

    The values below (baseline, resampling rate and kernel scales) are what this profile
    pins for the paradigm; the arguments are the ones that follow your own setup.

    Parameters
    ----------
    band :
        Band-pass cutoffs ``(low, high)`` in Hz.
    order :
        Band-pass filter order.
    w_segment_t :
        Segment window ``(start, stop)`` around each frame onset, in ms. Keep it below the
        shortest gap between two bursts of one code (see above), so this stays yours to
        choose.
    arch :
        EEG-Inception architecture: ``'eeg_inception_v1'`` or ``'eeg_inception_v2'``.

    Returns
    -------
    SettingsTree
        A fresh tree. Pass it as ``BWREEGInceptionPipeline(settings=...)``.

    Raises
    ------
    ValueError
        If ``arch`` is not one of the shipped EEG-Inception architectures.

    Examples
    --------
    >>> from medusa.pipelines.bci.vep_spellers.decoding import (
    ...     BWREEGInceptionPipeline, burst_cvep_settings)            # doctest: +SKIP
    >>> s = burst_cvep_settings()                                    # doctest: +SKIP
    >>> pipe = BWREEGInceptionPipeline(settings=s, channels=channels)   # doctest: +SKIP
    >>> scores = pipe.fit(train).predict(recording)                     # doctest: +SKIP
    """
    return bwr_eeg_inception_settings(
        profile="burst_cvep", band=band, order=order, w_segment_t=w_segment_t, arch=arch,
        baseline_t=None, target_fs=128.0, scales_ms=(100.0, 75.0, 50.0))


class BWREEGInceptionPipeline(TorchPipeline):
    """Bit-wise-reconstruction speller pipeline with an EEG-Inception classifier.

    The deep sibling of
    :class:`~medusa.pipelines.bci.vep_spellers.decoding.bwr_lda.BWRLDAPipeline`. It runs the
    same BWR *strategy* -- classify each code frame (is the target response present or not),
    then score each command by the correlation of its code with the frame scores -- and
    returns the same cumulative ``(n_cycles, n_commands)`` correlation matrix
    (:func:`~medusa.pipelines.bci.vep_spellers.decoding.bwr_command_scores`) for
    :func:`~medusa.pipelines.bci.vep_spellers.decoding.command_decoder.select_commands`.
    The only change is the frame classifier: a convolutional EEG-Inception backbone (v1 or
    v2) wrapped in a :class:`~medusa.ml.torch_models.classification.TorchClassifier`.

    A **single** pipeline serves both architectures because they share one
    preprocessing/epoch contract; the architecture is picked with the ``classifier.arch``
    setting (``'eeg_inception_v1'`` or ``'eeg_inception_v2'``). Their hyper-parameters are
    *not* shared, though -- v2 has ten where v1 has three -- so each architecture gets its
    own ``classifier.<arch>`` group holding exactly the knobs it has, and ``arch`` says
    which group is read (see
    :func:`~medusa.pipelines.bci._torch_backbones.add_architecture_settings`). The groups
    the selector is not pointing at are inert. Training hyper-parameters live in the
    ``classifier.training`` subgroup.

    The band, window, baseline and kernel scales that suit the data depend on **how the
    codes are presented**, so the module ships one ready settings tree per stimulation
    paradigm: :func:`mseq_cvep_settings` for a dense code with one bit on every frame, and
    :func:`burst_cvep_settings` for sparse bursts separated by quiet gaps (or
    :func:`bwr_eeg_inception_settings` to write your own recipe). A profile is only a set of
    values: whichever one you build with, :meth:`fit` and :meth:`predict` run the same chain.

    Two things differ from the LDA pipeline, both forced by the deep classifier (which is
    exactly why deep BWR is a separate class):

    * **Raw epochs.** Features are the per-frame epochs kept as
      ``(n_segments, n_samples, n_channels)`` (not flattened), resampled to
      ``segmentation.target_fs`` (default 128 Hz, EEG-Inception's design rate).
    * **Single band.** A conv backbone consumes one multichannel epoch, so it cannot fuse a
      parallel filter bank the way the LDA pipeline concatenates sub-band features. The
      ``freq_filtering`` schema is kept for consistency, but the filter bank must hold exactly
      **one** band-pass (checked in :meth:`check_consistency`); every profile builds exactly
      one, so they all satisfy it.

    Configure it with a profile
    (``BWREEGInceptionPipeline(settings=burst_cvep_settings(), channels=[...])``), through
    the live :attr:`~medusa.core.settings_tree.Configurable.settings` tree (see
    :meth:`default_settings`), or with construction kwargs, which may be nested, for example
    ``BWREEGInceptionPipeline(channels=["Fz", "Cz", "Pz", "Oz"], classifier={"arch":
    "eeg_inception_v2"})``. The backbone is sized from the data at :meth:`fit` time
    (``input_samples`` from the resampled epoch length, ``n_cha`` from the channel count) and
    saved as a portable config + ``state_dict`` bundle.
    """

    # ---- configuration schema (SettingsTree) ----
    @classmethod
    def default_settings(cls) -> SettingsTree:
        """The bare schema, with no ``profile`` name recorded.

        Built by :func:`bwr_eeg_inception_settings` with its own defaults, so a pipeline
        built with no settings still runs. Which values suit you depends on how the codes
        are presented, so the ready-made way to configure it is a paradigm profile:
        :func:`mseq_cvep_settings` or :func:`burst_cvep_settings` (or
        :func:`bwr_eeg_inception_settings` for a recipe of your own).

        The tree mirrors :meth:`BWRLDAPipeline.default_settings
        <medusa.pipelines.bci.vep_spellers.decoding.bwr_lda.BWRLDAPipeline.default_settings>`
        (``channels``, ``signal_key``, ``car``, ``freq_filtering``, ``segmentation``) and swaps
        the ``classifier`` group for the EEG-Inception configuration: the ``arch`` selector,
        the shared architecture knobs, and a ``training`` subgroup.
        """
        return bwr_eeg_inception_settings()

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

    # ---- feature path (shared by fit/predict) ----
    def _features(self, signal: Signal, cycle_onsets: NDArray,
                  n_frames: int, fps: float, cfg: dict) -> NDArray:
        """Per-frame segments kept as ``(n_segments, n_samples, n_channels)`` (cycle-major).

        The same CAR + band-pass + per-frame segmentation + resampling as
        :meth:`BWRLDAPipeline._features`, but the epochs are **not** flattened (the conv
        backbone consumes them raw), and only the single configured band is used.
        """
        x = harmonize_channels(signal, cfg["channels"])
        raw = car(x.signal) if cfg["car"] else x.signal
        # Single band (validated in check_consistency): use the sole filter-bank entry.
        xf = apply_notch_and_filterbank(raw, x.fs, cfg["notch_filtering"],
                                        cfg["freq_filtering"]["filterbank"])[0]
        onsets = _bit_onsets(cycle_onsets, n_frames, fps)
        seg_cfg = cfg["segmentation"]
        window = tuple(seg_cfg["w_segment_t"])
        baseline = tuple(seg_cfg["baseline_t"]) if seg_cfg["baseline_t"] else None
        seg = segment_signal_around_events(
            x.times, xf, onsets, x.fs, window, baseline,
            norm="dc" if baseline is not None else None)
        if seg_cfg["target_fs"]:
            seg = resample_segments(seg, window, seg_cfg["target_fs"])
        return seg

    def _frame_scores(self, recording: Recording, cfg: dict) -> NDArray:
        """Per-frame target-class scores for one recording (cycle-major order)."""
        sd = SpellerData.from_recording(recording)
        onsets, _, _, _ = cycle_arrays(recording.events)
        feats = self._features(recording.signals[cfg["signal_key"]], onsets,
                               sd.codes.shape[2], sd.fps_resolution, cfg)
        return self.clf.predict_proba(feats)[:, 1]

    # ---- offline ----
    def fit(self, recordings) -> "BWREEGInceptionPipeline":
        """Fit the EEG-Inception classifier on the per-frame BWR features and labels of all
        recordings; return ``self``.

        Gathers the raw per-frame epochs and their target/non-target labels and trains
        the classifier on them. The first call builds the backbone the ``classifier.arch``
        setting names -- sized to the epoch shape, so it can never desync from the data;
        a later call keeps training the model this pipeline already holds, under the
        configured ``classifier.training.profile`` (see
        :class:`~medusa.pipelines.torch_base.TorchPipeline`).
        """
        self._check_settings()          # re-validate (the live tree may have been edited)
        cfg = self.cfg
        X, y = [], []
        for rec in recordings:
            self.check_consistency(rec)
            onsets, _, _, _ = cycle_arrays(rec.events)
            sd = SpellerData.from_recording(rec)
            X.append(self._features(rec.signals[cfg["signal_key"]], onsets,
                                    sd.codes.shape[2], sd.fps_resolution, cfg))
            y.append(bwr_labels(rec))
        return self._fit_classifier(cfg, np.concatenate(X), np.concatenate(y))

    def _build_backbone(self, cfg: dict, X: NDArray):
        """Build the architecture ``classifier.arch`` names, sized to the frame epochs."""
        # the epoch rate the millisecond kernel scales are measured against
        rate = cfg["segmentation"]["target_fs"] or self.fs
        return build_backbone(cfg["classifier"], input_samples=X.shape[1],
                              n_cha=X.shape[2], rate=rate)

    def predict(self, recording: Recording) -> NDArray:
        """Cumulative ``(n_cycles, n_commands)`` command correlations for one recording."""
        if not self._fitted:
            raise RuntimeError("pipeline is not fitted; call fit() first.")
        self.check_consistency(recording)
        sd = SpellerData.from_recording(recording)
        onsets, trial, cycle, code_idx = cycle_arrays(recording.events)
        # frame_scores = self._frame_scores(recording, self.cfg)
        feats = self._features(
            recording.signals[self.cfg["signal_key"]],
            onsets, sd.codes.shape[2], sd.fps_resolution, self.cfg)
        frame_scores = self.clf.predict_proba(feats)[:, 1]
        return bwr_command_scores(frame_scores, sd.codes, trial, cycle,
                                  code_idx)
