"""Settings schema and constructor mapping for the torch backbones the deep pipelines use.

The same arrangement as :mod:`medusa.pipelines.bci._filtering`, one layer down: the
mechanics live here once, and each pipeline only chooses which architectures to offer and
with what defaults.

The backbones themselves
(:class:`~medusa.ml.torch_models.backbones.eeg_inception.EEGInception`,
:class:`~medusa.ml.torch_models.backbones.eeg_inception_v2.EEGInceptionV2`) stay plain
``nn.Module``\\ s with ordinary keyword constructors -- they carry no configuration
machinery, and a user building one directly writes
``EEGInceptionV2(input_samples=..., n_cha=...)`` and nothing else. A ``SettingsTree`` is a
*pipeline* concern, so the pipeline layer is where the description of those constructors
belongs.

Three layers, low to high:

* :func:`add_eeg_inception_settings` / :func:`build_eeg_inception` (and the ``_v2`` pair)
  describe **one** architecture: the leaves it really has, and the mapping back onto its
  constructor. The leaf names mirror the constructor arguments, so the mapping stays
  obvious. The one leaf that is not a straight copy is **v1**'s temporal scales, which the
  settings state in milliseconds (``scales_ms``) and :func:`scales_to_samples` converts at
  build time with the epoch rate. v2 does not do that: it sizes its temporal kernels in
  samples, exactly like the dilated ones it sits next to, so its ``temp_scales_samples``
  leaf is used as it stands and both halves of the architecture are read in one unit.
* :data:`ARCHITECTURES` maps each ``arch`` name to everything the selector needs to know
  about it, so registering an architecture is a single edit in a single place.
* :func:`add_architecture_settings` mounts the whole thing on a ``classifier`` group -- an
  ``arch`` selector plus **one group per architecture**, keyed by the architecture's own
  name -- and :func:`build_backbone` reads that group back and builds the module.

Why every architecture's group is present at once, rather than swapping the subtree when
``arch`` changes: the tree is a live, GUI-editable object with no change notification, so a
swap would only happen when something re-ran the builder, leaving ``arch`` and the visible
parameters free to disagree. A fixed shape also keeps ``save``/``load`` lossless
(``update_from_dict`` only *warns* on keys the tree does not have, so a moving shape would
silently drop settings) and keeps ``reset()`` and ``user_overrides()`` meaningful. The cost
is that the architecture you are not using is still visible; it is bounded (v1 has three
leaves), and a profile-built configuration hides it behind a ready recipe anyway.

This module imports torch through the backbones, so -- like the deep pipelines that use it
-- it must never be imported from a package ``__init__``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

from medusa.ml.torch_models.backbones.eeg_inception import EEGInception
from medusa.ml.torch_models.backbones.eeg_inception_v2 import EEGInceptionV2

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from medusa.core.settings_tree import SettingsTree

__all__ = ["ARCHITECTURES", "add_architecture_settings", "build_backbone",
           "add_eeg_inception_settings", "build_eeg_inception",
           "add_eeg_inception_v2_settings", "build_eeg_inception_v2",
           "scales_to_samples"]

#: EEG-Inception v1 constraints that make its derived conv kernels / channels degenerate
#: (a zero-size dimension) for extreme configs. v2 pools adaptively and validates most of its
#: own inputs, so it needs none of these -- its one floor is ``output_pooling_factor >= 2``,
#: which :func:`build_eeg_inception_v2` checks (a factor of 1 is a no-op pool, and the block
#: count it derives divides by ``log(factor)``).
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

#: EEG-Inception v2's own floor: it derives its output-block count as
#: ``log(samples, output_pooling_factor)``, so a factor of 1 divides by ``log(1) == 0``.
_V2_MIN_POOLING_FACTOR = 2


def scales_to_samples(scales_ms: "Sequence[float]", rate: float) -> "tuple[int, ...]":
    """Temporal kernel scales in ms -> whole samples at ``rate`` (at least one sample each).

    Every backbone constructor takes its temporal scales in samples. v1's settings state
    them as durations instead, so that a scale means the same thing whatever the epochs were
    resampled to, and this converts them with the epoch rate the pipeline used. v2 states
    its kernel sizes in samples and never comes through here.
    """
    if not scales_ms:
        raise ValueError("at least one temporal scale is required, got an empty list.")
    return tuple(max(1, round(ms / 1000.0 * rate)) for ms in scales_ms)


# --------------------------------------------------------------------------- #
# EEG-Inception v1
# --------------------------------------------------------------------------- #
def add_eeg_inception_settings(
        group: "SettingsTree", *,
        scales_ms: "Sequence[float]" = (100.0, 75.0, 50.0),
        filters_per_branch: int = 8,
        dropout_rate: float = 0.25) -> None:
    """Add every :class:`~medusa.ml.torch_models.backbones.eeg_inception.EEGInception` knob to ``group``.

    All of them except ``input_samples`` and ``n_cha``, which are read off the data at fit
    time so the backbone can never desync from the epochs it is given.
    """
    group.add_item("scales_ms", value=[float(ms) for ms in scales_ms],
                   info="Temporal inception kernel scales (ms); converted to samples at "
                        "build time with the epoch rate")
    group.add_item("filters_per_branch", value=int(filters_per_branch),
                   value_range=[1, None],
                   info="Convolutional filters per inception branch")
    group.add_item("dropout_rate", value=float(dropout_rate), value_range=[0, 1],
                   info="Dropout probability")


def build_eeg_inception(cfg: dict, *, input_samples: int, n_cha: int,
                        rate: float) -> EEGInception:
    """Build an :class:`~medusa.ml.torch_models.backbones.eeg_inception.EEGInception` from its settings group."""
    scales = scales_to_samples(cfg["scales_ms"], rate)
    filters_per_branch = int(cfg["filters_per_branch"])
    _check_eeg_inception_dims(input_samples, scales, filters_per_branch)
    return EEGInception(input_samples=input_samples, n_cha=n_cha,
                        scales_samples=scales,
                        filters_per_branch=filters_per_branch,
                        dropout_rate=float(cfg["dropout_rate"]))


def _check_eeg_inception_dims(n_samples: int, scales: "tuple[int, ...]",
                              filters_per_branch: int) -> None:
    """Reject configs that would build a degenerate (zero-size) EEG-Inception v1.

    v1 has no internal validation, so an over-small epoch, temporal scale, or filter count
    silently builds a zero-height kernel or zero-channel layer that only crashes deep inside
    the first forward pass. This turns those into clear, actionable errors (v2 validates
    itself and pools adaptively, so it needs none of this).
    """
    problems = []
    if n_samples < _V1_MIN_SAMPLES:
        problems.append(
            f"the epoch has {n_samples} samples but v1 pools the time axis by 2 five times "
            f"(needs >= {_V1_MIN_SAMPLES}); widen the segment window or raise the "
            f"resampling rate")
    if min(scales) < _V1_MIN_SCALE:
        problems.append(
            f"the smallest temporal scale is {min(scales)} samples but block 3 uses a "
            f"scale//4 kernel (needs every scale >= {_V1_MIN_SCALE}); raise 'scales_ms' or "
            f"the resampling rate")
    if filters_per_branch * len(scales) < _V1_MIN_BRANCH_UNITS:
        problems.append(
            f"filters_per_branch ({filters_per_branch}) * n_scales ({len(scales)}) = "
            f"{filters_per_branch * len(scales)} but block 4 narrows by /4 (needs >= "
            f"{_V1_MIN_BRANCH_UNITS}); raise 'filters_per_branch'")
    if problems:
        raise ValueError(
            "eeg_inception_v1 cannot be built with this configuration: "
            + "; ".join(problems)
            + ". Alternatively use 'eeg_inception_v2' (adaptive pooling, self-validating).")


# --------------------------------------------------------------------------- #
# EEG-Inception v2
# --------------------------------------------------------------------------- #
def add_eeg_inception_v2_settings(
        group: "SettingsTree", *,
        temp_scales_samples: "Sequence[int]" = (100, 75, 50),
        temp_filt_per_branch: int = 8,
        n_temp_inc_blocks: int = 1,
        dil_filt_per_branch: int = 8,
        dil_branch_specs: "Sequence[Sequence[int]]" = ((5, 1), (5, 5), (5, 10), (5, 15)),
        n_dil_inc_blocks: int = 1,
        n_spatial_filt_mult: int = 2,
        output_pooling_factor: int = 2,
        dropout_type: str = "Dropout",
        dropout_rate: float = 0.25) -> None:
    """Add every :class:`~medusa.ml.torch_models.backbones.eeg_inception_v2.EEGInceptionV2` knob to ``group``.

    All of them except ``input_samples`` and ``n_cha``, which are read off the data at fit
    time. Unlike v1, the temporal and the dilated inception blocks have their **own** filter
    counts (``temp_filt_per_branch`` / ``dil_filt_per_branch``), so they are separate leaves.

    Every leaf carries the constructor argument's own name and its own units, so the whole
    group maps one-to-one onto the constructor. That includes the temporal scales: v2 sizes
    its kernels in **samples**, exactly like its dilated branches do, and the leaf holds the
    number it is given. v1's scales are the one place that still works in milliseconds
    (:func:`add_eeg_inception_settings`), so only that architecture converts at build time.
    """
    group.add_item("temp_scales_samples",
                   value=[int(k) for k in temp_scales_samples],
                   info="Temporal inception kernel sizes, in samples (one per branch); "
                        "used as given, so pick them for the rate of the epochs")
    group.add_item("temp_filt_per_branch", value=int(temp_filt_per_branch),
                   value_range=[1, None],
                   info="Convolutional filters per temporal inception branch")
    group.add_item("n_temp_inc_blocks", value=int(n_temp_inc_blocks), value_range=[1, None],
                   info="Temporal inception blocks stacked in series")
    group.add_item("dil_filt_per_branch", value=int(dil_filt_per_branch),
                   value_range=[1, None],
                   info="Convolutional filters per dilated inception branch")
    specs = group.add_group_list(
        "dil_branch_specs",
        info="Dilated inception branches: one (kernel size, dilation rate) pair each")
    specs.element.add_item("kernel", value=int(dil_branch_specs[0][0]),
                           value_range=[1, None],
                           info="Kernel size, in samples; must be odd (symmetric padding)")
    specs.element.add_item("dilation", value=int(dil_branch_specs[0][1]),
                           value_range=[1, None], info="Dilation rate")
    for kernel, dilation in dil_branch_specs:
        specs.add_element(values={"kernel": int(kernel), "dilation": int(dilation)})
    # add_element writes values, not defaults; re-baseline so each branch IS its own
    # default and reset() cannot collapse the list onto the template (as in _filtering).
    specs.set_defaults_from_values()
    group.add_item("n_dil_inc_blocks", value=int(n_dil_inc_blocks), value_range=[1, None],
                   info="Dilated inception blocks stacked in series")
    group.add_item("n_spatial_filt_mult", value=int(n_spatial_filt_mult),
                   value_range=[1, None],
                   info="Spatial filters per input channel (depthwise multiplier)")
    group.add_item("output_pooling_factor", value=int(output_pooling_factor),
                   value_range=[_V2_MIN_POOLING_FACTOR, None],
                   info="Temporal pooling factor of the output block; at least 2, since "
                        "the number of output blocks is derived from its logarithm")
    group.add_item("dropout_type", value=dropout_type,
                   value_options=["Dropout", "SpatialDropout2D"],
                   info="Drop single units ('Dropout') or whole feature maps "
                        "('SpatialDropout2D')")
    group.add_item("dropout_rate", value=float(dropout_rate), value_range=[0, 1],
                   info="Dropout probability")


def build_eeg_inception_v2(cfg: dict, *, input_samples: int, n_cha: int,
                           rate: float) -> EEGInceptionV2:
    """Build an :class:`~medusa.ml.torch_models.backbones.eeg_inception_v2.EEGInceptionV2` from its settings group.

    ``rate`` is part of the common builder signature but unused here: every v2 kernel size
    is already a sample count, so nothing has to be converted.
    """
    scales = tuple(int(k) for k in cfg["temp_scales_samples"])
    if not scales:
        raise ValueError(
            "eeg_inception_v2 needs at least one entry in 'temp_scales_samples' "
            "(one temporal inception kernel size, in samples, per branch).")
    specs = tuple((int(b["kernel"]), int(b["dilation"])) for b in cfg["dil_branch_specs"])
    if not specs:
        raise ValueError(
            "eeg_inception_v2 needs at least one entry in 'dil_branch_specs' "
            "(a (kernel, dilation) pair per dilated inception branch).")
    pooling_factor = int(cfg["output_pooling_factor"])
    if pooling_factor < _V2_MIN_POOLING_FACTOR:
        # v2 derives its output-block count from log(samples, factor); 1 would divide by
        # log(1) == 0, deep inside the constructor and with nothing naming the setting.
        raise ValueError(
            f"eeg_inception_v2 needs 'output_pooling_factor' >= "
            f"{_V2_MIN_POOLING_FACTOR}, got {pooling_factor} (a factor of 1 pools nothing "
            f"and makes the output-block count undefined).")
    return EEGInceptionV2(
        input_samples=input_samples, n_cha=n_cha,
        temp_scales_samples=scales,
        temp_filt_per_branch=int(cfg["temp_filt_per_branch"]),
        n_temp_inc_blocks=int(cfg["n_temp_inc_blocks"]),
        dil_filt_per_branch=int(cfg["dil_filt_per_branch"]),
        dil_branch_specs=specs,
        n_dil_inc_blocks=int(cfg["n_dil_inc_blocks"]),
        n_spatial_filt_mult=int(cfg["n_spatial_filt_mult"]),
        output_pooling_factor=pooling_factor,
        dropout_type=cfg["dropout_type"],
        dropout_rate=float(cfg["dropout_rate"]))


# --------------------------------------------------------------------------- #
# The selector: one group per architecture, named by the architecture
# --------------------------------------------------------------------------- #
class _Architecture(NamedTuple):
    """Everything :data:`ARCHITECTURES` has to know about one backbone.

    ``scales_ms_leaf`` is here rather than in a lookup table beside it so that registering
    an architecture is a *single* edit: a second, partial mapping would raise ``KeyError``
    for every architecture the first time a new one was added.
    """

    add_settings: "Callable"      # (group, **defaults) -> None
    build: "Callable"             # (cfg, *, input_samples, n_cha, rate) -> nn.Module
    scales_ms_leaf: "str | None"  # its temporal-scale leaf, if that leaf is in ms


#: Selectable architectures: ``arch`` name -> :class:`_Architecture`. The name is also the
#: key of that architecture's settings group, so :func:`build_backbone` needs no lookup
#: table of its own.
#:
#: ``scales_ms_leaf`` is ``None`` for v2 because its temporal kernels are sized in samples,
#: like its dilated ones, so the ``scales_ms`` shorthand below has nothing to convert into:
#: a v2 kernel size is stated in the settings and used as it stands.
ARCHITECTURES = {
    "eeg_inception_v1": _Architecture(
        add_eeg_inception_settings, build_eeg_inception, "scales_ms"),
    "eeg_inception_v2": _Architecture(
        add_eeg_inception_v2_settings, build_eeg_inception_v2, None),
}


def add_architecture_settings(
        classifier_group: "SettingsTree", *,
        arch: str = "eeg_inception_v1",
        scales_ms: "Sequence[float] | None" = None,
        defaults: "Mapping[str, Mapping] | None" = None) -> None:
    """Add the ``arch`` selector and one settings group per architecture.

    Every architecture in :data:`ARCHITECTURES` gets its own group, named after it, holding
    only the knobs that architecture really has; ``arch`` says which group
    :func:`build_backbone` reads. The groups the selector is not pointing at are inert.

    Parameters
    ----------
    classifier_group : SettingsTree
        The ``classifier`` group to add the selector and the architecture groups to.
    arch : str, optional
        The architecture selected by default; must be a key of :data:`ARCHITECTURES`.
    scales_ms : sequence of float, optional
        Temporal kernel scales, in ms, applied as the default to the temporal-scale leaf of
        every architecture that states that leaf in ms (today: v1). A duration means the
        same thing whatever the epochs were resampled to, so a pipeline that has an opinion
        about the timescales of its responses states it once here. ``None`` keeps each
        architecture's own default. It reaches no architecture whose kernels are sized in
        samples -- v2 -- because there is no rate to convert with at schema-building time;
        give those their sizes through ``defaults``.
    defaults : mapping of {arch name: {leaf: value}}, optional
        Per-architecture default overrides, passed straight to that architecture's
        ``add_settings`` as keyword arguments. This is how a pipeline pins the values it
        wants for one architecture in particular -- ``{"eeg_inception_v2":
        {"temp_scales_samples": (50, 25, 15), "dropout_rate": 0.2}}``. They become the
        group's **defaults**, so ``reset()`` returns to them and ``user_overrides()``
        reports only what was changed on top.

    Raises
    ------
    ValueError
        If ``arch``, or a key of ``defaults``, is not a known architecture.
    """
    if arch not in ARCHITECTURES:
        raise ValueError(
            f"arch must be one of {list(ARCHITECTURES)}, got {arch!r}.")
    defaults = dict(defaults or {})
    unknown = [name for name in defaults if name not in ARCHITECTURES]
    if unknown:
        raise ValueError(
            f"defaults has no architecture {unknown}; keys must be among "
            f"{list(ARCHITECTURES)}.")
    classifier_group.add_item(
        "arch", value=arch, value_options=list(ARCHITECTURES),
        info="Backbone architecture; its settings are the group of the same name below "
             "(the other architectures' groups are ignored)")
    for name, spec in ARCHITECTURES.items():
        group = classifier_group.add_group(
            name, info=f"{name} hyper-parameters (used when arch is {name!r})")
        overrides = {}
        if scales_ms is not None and spec.scales_ms_leaf is not None:
            overrides[spec.scales_ms_leaf] = [float(ms) for ms in scales_ms]
        overrides.update(defaults.get(name, {}))
        spec.add_settings(group, **overrides)


def build_backbone(classifier_cfg: dict, *, input_samples: int, n_cha: int,
                   rate: float):
    """Build the backbone ``classifier_cfg['arch']`` names, from its own settings group.

    ``input_samples`` and ``n_cha`` come from the actual feature array at fit time, so the
    backbone can never desync from the data; ``rate`` is the epoch rate (the resampling
    target, or the recording rate when resampling is off), which the architectures that
    state their temporal scales as durations use to turn them into samples. v2 sizes its
    kernels in samples and ignores it.
    """
    arch = classifier_cfg["arch"]
    if arch not in ARCHITECTURES:
        raise ValueError(
            f"classifier.arch must be one of {list(ARCHITECTURES)}, got {arch!r}.")
    return ARCHITECTURES[arch].build(classifier_cfg[arch], input_samples=input_samples,
                                     n_cha=n_cha, rate=rate)
