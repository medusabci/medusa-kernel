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
from medusa.pipelines.bci.vep_spellers.decoding._common import (
    _bit_onsets, _check_target_fs, _trial_cycle_order)
from medusa.pipelines.bci.vep_spellers.decoding.command_decoder import _available
from medusa.pipelines.bci.vep_spellers.decoding.raster_latency_correction import (
    add_raster_latency_settings, command_raster_latencies, raster_corrected_events)
from medusa.pipelines.bci.vep_spellers.decoding.scores import (
    bwr_labels, bwr_frame_scores, bwr_command_scores_corr)

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "BWREEGInceptionPipeline",
    "bwr_eeg_inception_settings",
    "mseq_cvep_settings",
    "burst_cvep_settings"
]

#: EEG-Inception v2 hyper-parameters for a VEP speller: the defaults the
#: ``classifier.eeg_inception_v2`` group of every tree this module builds starts from. v1's
#: three knobs keep their own defaults, set through ``scales_ms`` and the schema.
#:
#: v2 sizes its temporal kernels in **samples**, so these numbers only mean what they are
#: meant to mean at the ``target_fs`` this module ships (200 Hz): the three temporal
#: branches then span 250, 125 and 75 ms of a 500 ms frame epoch. Change the resampling rate
#: and these have to follow -- that is the trade for stating the sizes directly, which is
#: what v2 does everywhere else too (its dilated branches have always been in samples).
_EEG_INCEPTION_V2_DEFAULTS = dict(
    temp_scales_samples=(50, 25, 15),
    temp_filt_per_branch=12,
    n_temp_inc_blocks=1,
    dil_filt_per_branch=12,
    dil_branch_specs=((5, 1), (5, 5), (5, 10), (5, 15)),
    n_dil_inc_blocks=1,
    n_spatial_filt_mult=2,
    output_pooling_factor=2,
    dropout_type="Dropout",
    dropout_rate=0.2,
)


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
        baseline_t: "Sequence[float] | None" = (0.0, 500.0),
        target_fs: "float | None" = 200.0,
        arch: str = "eeg_inception_v1",
        scales_ms: "Sequence[float]" = (100.0, 75.0, 50.0),
        temp_scales_samples: "Sequence[int]" = (50, 25, 15)) -> SettingsTree:
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
        Baseline window ``(start, stop)`` in ms. Each segment is normalized per channel with
        the statistics of this window (``'z'`` by default; see the ``segmentation.norm``
        setting). The default equals the default ``w_segment_t``, so each segment is
        normalized with its own statistics; pass the same window here when you change
        ``w_segment_t``. ``None`` ships the baseline switched **off**, keeping
        ``w_segment_t`` as the value it takes when you switch it on.
    target_fs :
        Rate the segments are resampled to, in Hz. ``None`` ships the resampling switched
        **off**, so the epochs keep the recording rate, and 200 Hz stays as the value it
        takes when you switch it on. It is also the rate ``temp_scales_samples`` below is
        chosen for, so the two move together.
    arch :
        EEG-Inception architecture: ``'eeg_inception_v1'`` or ``'eeg_inception_v2'``.
        Every architecture gets its own ``classifier.<arch>`` group of hyper-parameters;
        this says which one :meth:`BWREEGInceptionPipeline.fit` builds from. The rest of
        those hyper-parameters are edited on the tree (or passed as nested construction
        kwargs, e.g. ``classifier={"eeg_inception_v2": {"n_spatial_filt_mult": 3}}``),
        because which ones exist depends on the architecture.
    scales_ms :
        Temporal inception kernel scales in ms, for the architectures that state them as
        durations -- today **v1** only. They become samples at build time, using the epoch
        rate, so they mean the same thing whatever ``target_fs`` is.
    temp_scales_samples :
        Temporal inception kernel sizes for **v2**, one per branch, in **samples**. v2 sizes
        every kernel it has directly, dilated branches included, so this one is not
        converted: it is the number of samples the kernel covers. The default spans 250, 125
        and 75 ms at the 200 Hz ``target_fs`` above; change that rate and these should
        follow. The rest of v2's hyper-parameters default to
        :data:`_EEG_INCEPTION_V2_DEFAULTS` and are edited on the tree (or passed as nested
        construction kwargs), because which ones exist depends on the architecture.

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
                 value=[float(t) for t in (baseline_t or w_segment_t)],
                 optional=True,
                 enabled=bool(baseline_t),
                 info="Baseline window (ms); set it equal to w_segment_t to normalize each "
                      "segment with its own statistics; switch it off to leave the "
                      "segments as they are")
    seg.add_item("norm", value="z", value_options=["dc", "z"],
                 info="Baseline normalization, per segment and channel: 'dc' subtracts the "
                      "baseline mean, 'z' also divides by the baseline standard deviation. "
                      "Only used while baseline_t is on")
    seg.add_item("target_fs",
                 value=float(target_fs) if target_fs else 200.0,
                 optional=True,
                 enabled=bool(target_fs),
                 value_range=[1.0, None],
                 info="Resample segments to this rate (Hz); "
                      "switch it off to keep the native rate")
    add_raster_latency_settings(s)
    clf = s.add_group("classifier",
                      info="EEG-Inception frame classifier")
    add_architecture_settings(
        clf, arch=arch, scales_ms=scales_ms,
        defaults={"eeg_inception_v2": {**_EEG_INCEPTION_V2_DEFAULTS,
                                       "temp_scales_samples": temp_scales_samples}})
    add_training_settings(
        clf, profiles=BWREEGInceptionPipeline.TRAINING_PROFILES,
        max_epochs=100, batch_size=256, learning_rate=0.01,
        val_split=0.1, patience=5, min_delta=0.001,
        val_split_units={
            "frame": "single frame epochs, drawn at random and stratified by label; "
                     "they overlap their neighbours, so training data leaks into "
                     "validation and early stopping stops measuring generalisation",
            "cycle": "whole stimulation cycles",
            "trial": "whole trials with all their cycles (default; keep it unless you "
                     "know why not)",
        })
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
        baseline_t=w_segment_t, target_fs=200.0, scales_ms=(100.0, 75.0, 50.0),
        temp_scales_samples=(50, 25, 15))


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
        baseline_t=w_segment_t, target_fs=200.0, scales_ms=(100.0, 75.0, 50.0),
        temp_scales_samples=(50, 25, 15))


def _validation_groups(unit: str, trial_idx: NDArray, n_frames: int) -> "NDArray | None":
    """One validation-group id per frame epoch of a recording, per ``val_split_unit``.

    The epochs are cycle-major with ``n_frames`` per cycle, so a per-cycle id repeated
    ``n_frames`` times lines up with them. ``'trial'`` gives every cycle its trial index,
    ``'cycle'`` gives every cycle its own id, and ``'frame'`` returns ``None``: no groups,
    the estimator splits single epochs.
    """
    if unit == "frame":
        return None
    if unit == "trial":
        per_cycle = np.asarray(trial_idx, dtype=int)
    elif unit == "cycle":
        per_cycle = np.arange(len(trial_idx))
    else:
        raise ValueError(
            f"unknown val_split_unit {unit!r}; choose 'frame', 'cycle' or 'trial'.")
    return np.repeat(per_cycle, n_frames)


def _decision_margin(cmd_scores: NDArray, trial_idx: NDArray, cycle_idx: NDArray,
                     sd: SpellerData) -> float:
    """Mean decision margin of a recording's cumulative ``(n_cycles, n_commands)`` scores.

    A trial's margin is its best score minus its second best, among the commands available
    in that trial, on the trial's last row (all of its cycles). Trials with fewer than two
    finite scores are skipped; ``-inf`` if none is left.
    """
    col = {uid: i for i, uid in enumerate(sd.command_uids)}
    margins = []
    for t, order in _trial_cycle_order(trial_idx, cycle_idx):
        avail = _available(sd.trial_available_cmmds, sd.command_uids, t)
        r = np.sort(cmd_scores[order[-1], [col[u] for u in avail]])
        if r.size > 1 and np.isfinite(r[-2]):
            margins.append(r[-1] - r[-2])
    return float(np.mean(margins)) if margins else -np.inf


class BWREEGInceptionPipeline(TorchPipeline):
    """Bit-wise-reconstruction speller pipeline with an EEG-Inception classifier.

    The deep sibling of
    :class:`~medusa.pipelines.bci.vep_spellers.decoding.bwr_lda.BWRLDAPipeline`. It runs the
    same BWR *strategy* -- classify each code frame (which class of the code it showed), then
    score each command by the correlation of its code with the frame scores -- and
    returns the same cumulative ``(n_cycles, n_commands)`` correlation matrix
    (:func:`~medusa.pipelines.bci.vep_spellers.decoding.bwr_command_scores_corr`) for
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
      ``segmentation.target_fs`` (default 200 Hz).
    * **Single band.** A conv backbone consumes one multichannel epoch, so it cannot fuse a
      parallel filter bank the way the LDA pipeline concatenates sub-band features. The
      ``freq_filtering`` schema is kept for consistency, but the filter bank must hold exactly
      **one** band-pass (checked in :meth:`check_consistency`); every profile builds exactly
      one, so they all satisfy it.

    **Raster latency.** A monitor draws each frame from top to bottom, so a command lower on
    the screen is drawn later, by up to 95% of a frame. Switch the ``raster_scan_fraction``
    setting on to correct it: :meth:`fit` shifts each calibration cycle by its target's
    latency, and :meth:`predict` scores each command at its own latency. Every command then
    needs a :attr:`~medusa.pipelines.bci.vep_spellers.data.CommandInfo.screen_pos`. It is
    off by default (see
    :mod:`~medusa.pipelines.bci.vep_spellers.decoding.raster_latency_correction`).

    **Time shift.** The recorded onsets can be off from the EEG by a lag that is the same
    for every command (clock offsets, acquisition buffers). :meth:`estimate_time_shift`
    measures it on a recording whose targets are known, :meth:`fit` takes one shift per
    recording (``time_shifts``) to train on aligned epochs, and :meth:`predict` searches
    it without labels. The raster latency adds on top of it.

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
        (``channels``, ``signal_key``, ``car``, ``freq_filtering``, ``segmentation``), adds
        the ``raster_scan_fraction`` switch, and swaps the ``classifier`` group for the
        EEG-Inception configuration: the ``arch`` selector, the shared architecture knobs,
        and a ``training`` subgroup.
        """
        return bwr_eeg_inception_settings()

    # ---- validation ----
    def check_consistency(self, recording: Recording) -> None:
        """Check the recording has the configured signal and channels, a usable ``fs``,
        a single-band filter bank, a :class:`SpellerData`, and valid speller events; raise
        ``ValueError`` if not.

        Recordings only have to share one sampling rate when ``segmentation.target_fs``
        is unset. With it set, every epoch is resampled to that rate, the steps before it
        work off each recording's own ``fs``, and the backbone is sized from
        ``target_fs``, so a corpus may mix native rates; the first rate seen is still kept
        as :attr:`fs`. What is checked instead is that ``target_fs`` is reachable -- not
        above the recording's rate, and at least twice the filter-bank cutoff, so resampling
        cannot silently cut the configured band.
        """
        cfg = self.cfg
        sig = recording.signals.get(cfg["signal_key"])
        if sig is None:
            raise ValueError(f"recording has no {cfg['signal_key']!r} signal.")
        if not cfg["channels"]:
            raise ValueError("no channels configured; set the 'channels' setting.")
        _check_target_fs(cfg, sig.fs)
        if self.fs is None:
            self.fs = sig.fs
        elif sig.fs != self.fs and not cfg["segmentation"]["target_fs"]:
            # Native rates only have to agree when the epochs keep them.
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
        return self._segments(self._filtered_signal(signal, cfg), cycle_onsets,
                              n_frames, fps, cfg)

    def _filtered_signal(self, signal: Signal,
                         cfg: dict) -> tuple[NDArray, NDArray, float]:
        """The configured channels, re-referenced and band-passed: ``(times, filtered, fs)``.

        The part of the feature path that does not depend on the onsets. :meth:`predict`
        and :meth:`estimate_time_shift` cut one recording at many onset shifts, so they
        filter it once and pass the result to :meth:`_segments` for every shift.
        """
        x = harmonize_channels(signal, cfg["channels"])
        raw = car(x.signal) if cfg["car"] else x.signal
        # Single band (validated in check_consistency): use the sole filter-bank entry.
        filtered = apply_notch_and_filterbank(raw, x.fs, cfg["notch_filtering"],
                                              cfg["freq_filtering"]["filterbank"])[0]
        return x.times, filtered, x.fs

    def _segments(self, filtered_signal: tuple[NDArray, NDArray, float],
                  cycle_onsets: NDArray, n_frames: int, fps: float, cfg: dict) -> NDArray:
        """Per-frame segments of a :meth:`_filtered_signal`, resampled to
        ``segmentation.target_fs``: ``(n_segments, n_samples, n_channels)`` (cycle-major)."""
        times, filtered, fs = filtered_signal
        onsets = _bit_onsets(cycle_onsets, n_frames, fps)
        seg_cfg = cfg["segmentation"]
        window = tuple(seg_cfg["w_segment_t"])
        baseline = tuple(seg_cfg["baseline_t"]) if seg_cfg["baseline_t"] else None
        seg = segment_signal_around_events(
            times, filtered, onsets, fs, window, baseline,
            norm=seg_cfg["norm"] if baseline is not None else None)
        if seg_cfg["target_fs"]:
            seg = resample_segments(seg, window, seg_cfg["target_fs"])
        return seg

    # ---- offline ----
    def fit(self, recordings: "Sequence[Recording]",
            time_shifts: "Sequence[float] | None" = None) -> "BWREEGInceptionPipeline":
        """Fit the EEG-Inception classifier on the per-frame BWR features and labels of all
        recordings; return ``self``.

        Gathers the raw per-frame epochs and their target/non-target labels and trains
        the classifier on them. The first call builds the backbone the ``classifier.arch``
        setting names -- sized to the epoch shape, so it can never desync from the data;
        a later call keeps training the model this pipeline already holds, under the
        configured ``classifier.training.profile`` (see
        :class:`~medusa.pipelines.torch_base.TorchPipeline`).

        The validation split early stopping watches is cut per
        ``classifier.training.val_split_unit``: whole trials (the default), whole
        cycles, or single frame epochs. Frame epochs overlap their neighbours (a 500 ms
        window against a 17 ms frame period), so a frame-level split puts near-copies of
        the training epochs into the validation fold; holding out whole trials keeps the
        two apart. Trials of different recordings are always distinct groups.

        With the ``raster_scan_fraction`` setting on, each cycle onset is first shifted by
        the raster latency of its trial's target
        (:func:`~medusa.pipelines.bci.vep_spellers.decoding.raster_latency_correction.raster_corrected_events`),
        so the classifier learns from epochs aligned with the moment the target was drawn.
        The recordings are not modified.

        Parameters
        ----------
        recordings :
            Calibration recordings. Every trial needs a target (``spell_target``).
        time_shifts :
            One onset shift per recording, in seconds, added to all its cycle onsets: the
            lag between its recorded onsets and its EEG, for example as
            :meth:`estimate_time_shift` measures it. Positive when the EEG response comes
            later than the recorded onsets. The raster latency is added on top. ``None``
            cuts every recording at its recorded onsets.

        Returns
        -------
        BWREEGInceptionPipeline
            ``self``, fitted.

        Raises
        ------
        ValueError
            If ``time_shifts`` does not hold one finite value per recording, or if a
            recording does not match the pipeline (see :meth:`check_consistency`).

        Examples
        --------
        >>> pipe = BWREEGInceptionPipeline(settings=mseq_cvep_settings(),
        ...                                channels=channels)              # doctest: +SKIP
        >>> pipe.fit([run_1, run_2], time_shifts=[0.004, -0.020])          # doctest: +SKIP
        """
        self._check_settings()          # re-validate (the live tree may have been edited)
        cfg = self.cfg
        recordings = list(recordings)
        # One onset shift per recording, in seconds; None keeps the recorded onsets
        if time_shifts is None:
            time_shifts = np.zeros(len(recordings))
        time_shifts = np.asarray(time_shifts, dtype=float)
        if time_shifts.shape != (len(recordings),):
            raise ValueError(
                f"time_shifts needs one value per recording: there are {len(recordings)} "
                f"recording(s), but time_shifts has shape {time_shifts.shape}.")
        if not np.isfinite(time_shifts).all():
            raise ValueError(f"time_shifts must be finite numbers (seconds), got "
                             f"{time_shifts.tolist()}.")
        unit = cfg["classifier"]["training"]["val_split_unit"]
        scan_fraction = cfg["raster_scan_fraction"]     # None when the correction is off
        X, y, groups, offset = [], [], [], 0
        for rec, time_shift in zip(recordings, time_shifts):
            self.check_consistency(rec)
            sd = SpellerData.from_recording(rec)
            # Raster correction: shift each cycle by its target's latency (a new Events)
            events = rec.events
            if scan_fraction:
                events = raster_corrected_events(rec.events, sd, scan_fraction)
            onsets, trial_idx, _, _ = cycle_arrays(events)
            n_frames = sd.codes.shape[2]
            # The recording's time shift moves every cycle alike
            X.append(self._features(rec.signals[cfg["signal_key"]], onsets + time_shift,
                                    n_frames, sd.fps_resolution, cfg))
            y.append(bwr_labels(rec))
            ids = _validation_groups(unit, trial_idx, n_frames)
            if ids is not None:
                groups.append(ids + offset)     # keep recordings' trials apart
                offset += int(ids.max()) + 1
        return self._fit_classifier(cfg, np.concatenate(X), np.concatenate(y),
                                    groups=np.concatenate(groups) if groups else None)

    def _build_backbone(self, cfg: dict, X: NDArray):
        """Build the architecture ``classifier.arch`` names, sized to the frame epochs."""
        # the epoch rate, for the architectures whose kernel scales are durations (v1)
        rate = cfg["segmentation"]["target_fs"] or self.fs
        return build_backbone(cfg["classifier"], input_samples=X.shape[1],
                              n_cha=X.shape[2], rate=rate)

    def _raster_latencies(self, sd: SpellerData) -> NDArray:
        """Raster latency of each command, in seconds; all zeros when the correction is
        off."""
        scan_fraction = self.cfg["raster_scan_fraction"]
        if scan_fraction:
            return command_raster_latencies(sd, scan_fraction)
        return np.zeros(len(sd.command_uids))

    def _command_scores(self, filtered_signal: tuple[NDArray, NDArray, float],
                        sd: SpellerData, onsets: NDArray, trial_idx: NDArray,
                        cycle_idx: NDArray, code_idx: NDArray) -> NDArray:
        """One scoring pass: cumulative ``(n_cycles, n_commands)`` scores for these onsets.

        Cuts every frame epoch from the cycle ``onsets`` out of the
        :meth:`_filtered_signal`, classifies it, and correlates each command's code with
        the frame scores. Every command is scored with the same onsets.
        """
        feats = self._segments(filtered_signal, onsets, sd.codes.shape[2],
                               sd.fps_resolution, self.cfg)
        proba = self.clf.predict_proba(feats)
        frame_scores = bwr_frame_scores(proba, self.clf.classes_)
        return bwr_command_scores_corr(frame_scores, sd.codes, trial_idx, cycle_idx,
                                       code_idx)

    def _shifted_command_scores(self, filtered_signal: tuple[NDArray, NDArray, float],
                                sd: SpellerData, onsets: NDArray, trial_idx: NDArray,
                                cycle_idx: NDArray, code_idx: NDArray,
                                time_shift: float, latencies: NDArray) -> NDArray:
        """Cumulative ``(n_cycles, n_commands)`` scores at one time shift, every command
        at its own raster latency.

        Runs one scoring pass per distinct latency: commands drawn at the same latency
        (the same screen row) share it.
        """
        cmd_scores = np.zeros((len(onsets), len(latencies)))
        for latency in np.unique(latencies):
            # Commands drawn at this latency (same screen height)
            cmd_idx = np.flatnonzero(latencies == latency)
            # Score every command with the onsets shifted by this latency...
            pass_scores = self._command_scores(
                filtered_signal, sd, onsets + time_shift + latency,
                trial_idx, cycle_idx, code_idx)
            # ...but keep only the scores of the commands drawn at this latency
            cmd_scores[:, cmd_idx] = pass_scores[:, cmd_idx]
        return cmd_scores

    def predict(
            self,
            recording: Recording,
            max_time_shift: float = 0.0,
            center_time_shift: float = 0.0) -> tuple[NDArray, float]:
        """Score every command of one recording, and return the time shift used.

        The frames of each cycle are cut into epochs and classified. Each command is scored
        by the correlation of its code with the frame scores, accumulated over its trial's
        cycles. The command with the best score is the decoded one
        (:func:`~medusa.pipelines.bci.vep_spellers.decoding.command_decoder.select_commands`).

        Two onset corrections are applied, and they add up:

        * **Time shift** (unknown, searched). An offset between the event onsets and the
          EEG (clock offsets, buffers, drift) is the same for every command. The onsets are
          tried at shifts from ``center_time_shift - max_time_shift`` to
          ``center_time_shift + max_time_shift``, about one sample apart. The shift with
          the largest mean decision margin wins: best minus second-best command on each
          trial's last row. The margin peaks more sharply than the best score, because a
          misaligned onset lowers the target's score and raises a competitor's. No labels
          are used, and one shift serves the whole recording.
        * **Raster latency** (known, per command). With the ``raster_scan_fraction``
          setting on, each command is scored with the onsets shifted by its own raster
          latency, as if it were the target. Commands with the same latency (the same
          screen row) share one scoring pass.

        So each candidate time shift runs one scoring pass per distinct raster latency.

        Parameters
        ----------
        recording :
            The recording to decode. It needs the configured signal and channels, a
            :class:`~medusa.pipelines.bci.vep_spellers.data.SpellerData`, and speller
            events. It is not modified.
        max_time_shift :
            Half-width of the time-shift search, in seconds. ``0`` tries only
            ``center_time_shift``.
        center_time_shift :
            Centre of the time-shift search, in seconds, for example a known device lag.

        Returns
        -------
        cmd_scores :
            ``(n_cycles, n_commands)`` cumulative command scores at the chosen time shift.
            Rows follow :func:`~medusa.pipelines.bci.vep_spellers.data.cycle_arrays`;
            columns follow ``SpellerData.command_uids``.
        time_shift :
            The chosen time shift, in seconds. It does not include the raster latencies.

        Raises
        ------
        RuntimeError
            If the pipeline is not fitted.
        ValueError
            If the recording does not match the pipeline (see :meth:`check_consistency`),
            or if the raster correction is on and a command has no ``screen_pos``.

        Examples
        --------
        >>> from medusa.pipelines.bci.vep_spellers import (
        ...     BWREEGInceptionPipeline, mseq_cvep_settings)                 # doctest: +SKIP
        >>> pipe = BWREEGInceptionPipeline(settings=mseq_cvep_settings(), channels=channels,
        ...                                raster_scan_fraction=0.95)        # doctest: +SKIP
        >>> pipe.fit(train)                                                  # doctest: +SKIP
        >>> cmd_scores, time_shift = pipe.predict(test, max_time_shift=0.016)   # doctest: +SKIP
        """
        if not self._fitted:
            raise RuntimeError("pipeline is not fitted; call fit() first.")
        self.check_consistency(recording)
        sd = SpellerData.from_recording(recording)
        onsets, trial_idx, cycle_idx, code_idx = cycle_arrays(recording.events)
        latencies = self._raster_latencies(sd)

        # Candidate time shifts, centre first: argmax keeps the first of tied shifts, so
        # ties (and a recording with no usable trial) keep the centre
        offsets = np.linspace(
            -max_time_shift,
            max_time_shift,
            int(round(2 * max_time_shift * self.fs)) + 1,
        )
        time_shifts = center_time_shift + offsets[np.argsort(np.abs(offsets))]

        # Filter once; every scoring pass cuts its segments from the same signal
        filtered_signal = self._filtered_signal(
            recording.signals[self.cfg["signal_key"]], self.cfg)
        shifted_cmd_scores = []
        for time_shift in time_shifts:
            shifted_cmd_scores.append(self._shifted_command_scores(
                filtered_signal, sd, onsets, trial_idx, cycle_idx, code_idx,
                time_shift, latencies))

        # Keep the most discriminative time shift
        margins = [_decision_margin(s, trial_idx, cycle_idx, sd)
                   for s in shifted_cmd_scores]
        best = int(np.argmax(margins))
        return shifted_cmd_scores[best], time_shifts[best]

    def estimate_time_shift(
            self,
            recording: Recording,
            max_time_shift: float,
            center_time_shift: float = 0.0) -> tuple[float, NDArray, NDArray]:
        """Measure the onset time shift of a recording whose targets are known.

        The lag between the recorded onsets and the EEG (clock offsets, acquisition
        buffers) is considered to be the same for every command. The onsets are
        tried at whole samples (at the recording's own rate) from ``center_time_shift``,
        up to ``max_time_shift`` rounded to the nearest sample on each side,
        so the centre is always tried. At each shift, every trial's target is
        scored as :meth:`predict` scores it (at its own raster latency,
        when the ``raster_scan_fraction`` setting is on), and the shift
        where the mean target score is highest wins. A tie keeps the candidate closest to
        the centre.

        Knowing the targets is what lets the search go wide. :meth:`predict`'s label-free
        search must stay within half the spacing between two commands' codes, because a
        lag one spacing away decodes the neighbouring command just as well. The target's
        own score peaks only at the true lag. It needs no competing command, so a
        calibration recording with a single command works too.

        Parameters
        ----------
        recording :
            The recording to measure, with a target for every trial (``spell_target``).
            Every cycle its events hold is used. It is not modified.
        max_time_shift :
            Half-width of the search, in seconds.
        center_time_shift :
            Centre of the search, in seconds.

        Returns
        -------
        time_shift :
            The shift with the highest mean target score, in seconds. It does not include
            the raster latencies: pass it to :meth:`fit` (``time_shifts``) or to
            :meth:`predict` (``center_time_shift``) as it is.
        candidate_shifts :
            ``(n_candidates,)`` every shift tried, in seconds, ascending, one sample apart.
        target_scores :
            ``(n_candidates,)`` the mean target score at each candidate shift. Each trial's
            score is its cumulative score after its last cycle.

        Raises
        ------
        RuntimeError
            If the pipeline is not fitted.
        ValueError
            If ``max_time_shift`` is negative, if a trial has no target, if no candidate
            gives a finite score, or if the recording does not match the pipeline (see
            :meth:`check_consistency`).

        Examples
        --------
        >>> pipe.fit(train)                                                  # doctest: +SKIP
        >>> time_shift, candidate_shifts, target_scores = pipe.estimate_time_shift(
        ...     calibration_run, max_time_shift=0.040)                       # doctest: +SKIP
        """
        if not self._fitted:
            raise RuntimeError("pipeline is not fitted; call fit() first.")
        if max_time_shift < 0:
            raise ValueError(f"max_time_shift must be >= 0, got {max_time_shift}.")
        self.check_consistency(recording)
        sd = SpellerData.from_recording(recording)
        onsets, trial_idx, cycle_idx, code_idx = cycle_arrays(recording.events)
        latencies = self._raster_latencies(sd)

        # Score-matrix column of each trial's target
        spell_target = sd.spell_target or []
        target_col = {}
        for trial in np.unique(trial_idx):
            if trial >= len(spell_target) or spell_target[trial] is None:
                raise ValueError(
                    f"trial {trial} has no target: estimate_time_shift needs the target "
                    f"of every trial (speller_data.spell_target). Without targets, use "
                    f"predict's time-shift search.")
            target_col[int(trial)] = sd.command_uids.index(str(spell_target[trial]))

        # Candidate shifts: whole samples from the centre at the recording's own rate,
        # ascending, so the centre is always one of them
        signal = recording.signals[self.cfg["signal_key"]]
        n_steps = int(round(max_time_shift * signal.fs))
        candidate_shifts = center_time_shift + np.arange(-n_steps, n_steps + 1) / signal.fs

        # Filter once; every candidate cuts its segments from the same signal
        filtered_signal = self._filtered_signal(signal, self.cfg)
        target_scores = np.zeros(len(candidate_shifts))
        for i, time_shift in enumerate(candidate_shifts):
            cmd_scores = self._shifted_command_scores(
                filtered_signal, sd, onsets, trial_idx, cycle_idx, code_idx,
                time_shift, latencies)
            # The scores are cumulative: a trial's last row holds all its cycles
            trial_scores = [cmd_scores[rows[-1], target_col[trial]]
                            for trial, rows in _trial_cycle_order(trial_idx, cycle_idx)]
            target_scores[i] = np.mean(trial_scores)

        if not np.isfinite(target_scores).any():
            raise ValueError("no candidate time shift gave a finite target score.")
        # The best candidate; a tie keeps the one closest to the centre
        tied = np.flatnonzero(target_scores == np.nanmax(target_scores))
        best = tied[np.argmin(np.abs(candidate_shifts[tied] - center_time_shift))]
        return float(candidate_shifts[best]), candidate_shifts, target_scores
