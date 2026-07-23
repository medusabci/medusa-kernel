"""Shared frequency-filter building blocks for the ``bci`` decoding pipelines.

Every ``bci`` paradigm band-passes its signal before decoding, so all of the filtering
plumbing lives here once instead of being copied into each package's ``_common``: the
filter-spec schema leaves, the runtime band-type options, the spec-to-filter builder, and
the two ready-made filtering *stages* a pipeline adds to its settings and then applies.

Three layers, low to high:

* :func:`make_filter` builds one :class:`IIRFilter` / :class:`FIRFilter` from a spec dict, and
  :func:`add_filter_leaves` adds the spec leaves (``filt_type``, ``cutoff``, ``order``, and an
  optional ``band_type``) to a settings group.
* :func:`add_band_filter_settings` / :func:`add_notch_and_filterbank_settings` build the two
  filtering *schemas* a pipeline picks between: a single band-pass group (motor decoding), or a
  line-noise notch followed by a parallel filter-bank group-list (VEP spellers).
* :func:`apply_notch_and_filterbank` runs that notch-then-bank stage on a signal.

The paradigm-specific *choice* of stage stays in each pipeline (which builder it calls, with
what defaults); the mechanics are shared here.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, get_args

from medusa.signal.frequency_filtering import IIRFilter, FIRFilter, BAND_TYPES

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from medusa.core.settings_tree import SettingsTree

__all__ = ["BAND_TYPE_OPTIONS", "add_filter_leaves", "make_filter",
           "add_band_filter_settings", "add_notch_and_filterbank_settings",
           "apply_notch_and_filterbank"]

#: Selectable band types, as a runtime list (``BAND_TYPES`` is a typing ``Literal``).
BAND_TYPE_OPTIONS = list(get_args(BAND_TYPES))


# --------------------------------------------------------------------------- #
# One filter: spec leaves + builder
# --------------------------------------------------------------------------- #
def add_filter_leaves(group, cutoff: list, order: int,
                      band_type: "str | None" = None) -> None:
    """Add the shared filter-spec leaves to ``group`` (a ``filt_type`` enum and a bounded order).

    Adds ``filt_type``, ``cutoff`` and ``order`` for every filter, plus ``band_type`` (an enum)
    when it is given. A bandstop notch omits ``band_type`` (it is always a bandstop); a plain
    band-pass passes ``band_type='bandpass'``.
    """
    group.add_item("filt_type", value="iir", value_options=["iir", "fir"],
                   info="Filter family")
    if band_type is not None:
        group.add_item("band_type", value=band_type, value_options=BAND_TYPE_OPTIONS,
                       info="Band type")
    group.add_item("cutoff", value=cutoff, info="Cutoff frequency/frequencies in Hz")
    group.add_item("order", value=order, value_range=[1, None], info="Filter order")


def make_filter(spec: dict, filt_method: "str | None" = None):
    """Build an :class:`IIRFilter` or :class:`FIRFilter` from a validated ``spec`` dict.

    ``spec`` is ``{filt_type: 'iir'|'fir', band_type, cutoff, order}`` (``filt_type`` is
    optional and defaults to ``'iir'``). ``filt_method`` defaults to the family's offline
    method (``sosfiltfilt`` for IIR, ``filtfilt`` for FIR). A ``SettingsTree`` cannot validate
    the filter-bank list elements, so a malformed spec raises a clear :class:`ValueError` here
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
    if band not in BAND_TYPE_OPTIONS:
        raise ValueError(f"filter band_type must be one of {BAND_TYPE_OPTIONS}, got {band!r}.")
    if filt_type not in ("iir", "fir"):
        raise ValueError(f"filter filt_type must be 'iir' or 'fir', got {filt_type!r}.")
    cutoff = spec["cutoff"]
    cutoff = tuple(cutoff) if isinstance(cutoff, (list, tuple)) else float(cutoff)
    order = int(spec["order"])
    if filt_type == "fir":
        return FIRFilter(order, cutoff, band, filt_method=filt_method or "filtfilt")
    return IIRFilter(order, cutoff, band, filt_method=filt_method or "sosfiltfilt")


# --------------------------------------------------------------------------- #
# Filtering stages: schema builders
# --------------------------------------------------------------------------- #
def add_band_filter_settings(settings: "SettingsTree", cutoff: list, order: int) -> None:
    """Add a single band-pass ``filter`` group (``filt_type``, ``band_type``, ``cutoff``, ``order``).

    The one-band schema: a plain band-pass (for example 8--30 Hz for motor decoding), not a
    filter bank, so it is a plain group rather than the speller's group-list. Apply it with
    ``make_filter(cfg["filter"])``.
    """
    f = settings.add_group("filter", info="Band-pass filter applied before spatial filtering")
    add_filter_leaves(f, cutoff=cutoff, order=order, band_type="bandpass")


def add_notch_and_filterbank_settings(settings: "SettingsTree") -> None:
    """Add the notch + parallel filter-bank schema (a ``notch_filtering`` and ``freq_filtering`` group).

    The notch is a single filter (default: a 50 Hz line-noise bandstop), applied first, with an
    ``enabled`` toggle. The ``freq_filtering`` group holds ``filterbank``, a **group-list** of
    filter groups that share one schema (each is ``{filt_type, band_type, cutoff, order}``). One
    filter is a plain band-pass; several make a filter bank whose sub-bands are processed in
    parallel and then combined (FBCCA scores for template matching, concatenated features for
    BWR). The default bank is one band-pass, 1--70 Hz. Apply it with
    :func:`apply_notch_and_filterbank`.
    """
    notch = settings.add_group(
        "notch_filtering", info="Line-noise notch (bandstop), applied before the filter bank")
    notch.add_item("enabled", value=True, info="Apply the notch")
    add_filter_leaves(notch, cutoff=[48.0, 52.0], order=4)       # band_type is always bandstop

    ff = settings.add_group("freq_filtering", info="Parallel filter-bank filtering")
    fb = ff.add_group_list(
        "filterbank",
        info="Parallel sub-band filters; one filter = a single band, several = a filter bank")
    add_filter_leaves(fb.element, cutoff=[1.0, 70.0], order=5, band_type="bandpass")
    fb.add_element()                                # default bank: one band-pass 1--70 Hz


# --------------------------------------------------------------------------- #
# Filtering stages: application
# --------------------------------------------------------------------------- #
def apply_notch_and_filterbank(signal: "NDArray", fs: float, notch: dict,
                               filterbank: list) -> "list[NDArray]":
    """Notch then each filter-bank filter -> one filtered signal per sub-band (parallel).

    ``notch`` is the ``notch_filtering`` config (a bandstop; skipped when ``enabled`` is
    False) applied first; ``filterbank`` is the list of filter specs (``freq_filtering``'s
    ``filterbank``). Returns a list of ``(n_samples, n_channels)`` arrays, one per filter-bank
    entry. The counterpart application to :func:`add_notch_and_filterbank_settings`.
    """
    x = signal
    if notch.get("enabled", True):
        x = make_filter({**notch, "band_type": "bandstop"}).fit_transform(signal, fs)
    if not filterbank:
        raise ValueError("freq_filtering.filterbank is empty; add at least one filter.")
    return [make_filter(spec).fit_transform(x, fs) for spec in filterbank]
