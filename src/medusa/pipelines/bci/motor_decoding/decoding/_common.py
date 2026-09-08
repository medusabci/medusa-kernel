"""Model-agnostic building blocks shared by the motor-decoding pipelines.

Small helpers used by every motor pipeline: cut labelled trials into epochs (band-passing
through the shared :mod:`medusa.pipelines.bci._filtering` helpers), turn a spatial-filter
projection into log-variance features, and add the shared TorchClassifier training settings.
The band-pass schema itself lives in :mod:`medusa.pipelines.bci._filtering`
(:func:`~medusa.pipelines.bci._filtering.add_band_filter_settings`). Keeping the rest here
lets each pipeline live in its own short module without repeating this plumbing. They are the
reusable pieces *below* the pipeline layer; the pipelines above wire them together with motor
defaults.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from medusa.core.data.signal import Signal
from medusa.signal.spatial_filtering import car
from medusa.signal.segmentation import segment_signal_around_events, resample_segments

from medusa.pipelines.base import harmonize_channels
from medusa.pipelines.bci._filtering import make_filter


def trial_segments(signal: Signal, onsets: NDArray, *, channels: list, apply_car: bool,
                   filter_spec: dict, window: tuple, baseline: "tuple | None",
                   target_fs: float) -> NDArray:
    """Cut labelled-trial segments: pick channels, CAR, band-pass, segment, resample.

    Returns a ``(n_trials, n_samples, n_channels)`` array. Every motor pipeline shares this:
    CSP learns spatial filters from the segments; a deep model consumes them directly. ``window``
    and ``baseline`` are ``(start, end)`` in milliseconds relative to each onset (``baseline``
    is ``None`` to disable DC baseline correction); ``target_fs`` of ``None`` keeps the native rate.
    """
    x = harmonize_channels(signal, channels)
    raw = car(x.signal) if apply_car else x.signal
    filtered = make_filter(filter_spec).fit_transform(raw, x.fs)
    seg = segment_signal_around_events(
        x.times, filtered, onsets, x.fs, window, baseline,
        norm="dc" if baseline is not None else None)
    if target_fs:
        seg = resample_segments(seg, window, target_fs)
    return seg


def log_var(projection: NDArray, normalize: bool = False) -> NDArray:
    """Log-variance features of a spatial-filter projection.

    ``projection`` is ``(n_trials, n_samples, n_components)``. Returns
    ``(n_trials, n_components)``: the natural log of each component's variance over time.
    With ``normalize`` the per-trial variances are scaled to sum to one first (the classic
    CSP normalisation).
    """
    v = np.var(projection, axis=1)
    if normalize:
        v = v / v.sum(axis=1, keepdims=True)
    return np.log(v)
