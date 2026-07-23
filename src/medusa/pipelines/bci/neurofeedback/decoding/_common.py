"""Model-agnostic building blocks shared by the neurofeedback pipelines.

Neurofeedback is continuous: there are no trials. A pipeline slides a window over the
signal, computes one feature per window (band power, a connectivity metric), and expresses
it relative to a calibration baseline. These helpers hold the shared pieces below the
pipeline layer: a band-pass filter builder, the sliding-window cutter, the settings schema
(windowing + reference), and the baseline-reference transform.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from medusa.core.settings_tree import SettingsTree
from medusa.core.data.signal import Signal
from medusa.signal.spatial_filtering import car

from medusa.pipelines.base import harmonize_channels
from medusa.pipelines.bci._filtering import add_filter_leaves, make_filter

#: How the streamed feature is expressed relative to the calibration baseline ``b``:
#: ``none`` = raw feature, ``subtract`` = ``x - b``, ``divide`` = ``x / b``,
#: ``percent`` = ``100 * (x - b) / b`` (ERD/ERS-style percent change).
REFERENCE_MODES = ["none", "subtract", "divide", "percent"]


def add_band_filter_settings(settings: SettingsTree, name: str, cutoff: list, order: int,
                             info: str) -> None:
    """Add a single band-pass filter group (``filt_type``, ``band_type``, ``cutoff``, ``order``)."""
    f = settings.add_group(name, info=info)
    add_filter_leaves(f, cutoff=cutoff, order=order, band_type="bandpass")


def add_windowing_settings(settings: SettingsTree) -> None:
    """Add the sliding-window timing group (feature window length + update rate)."""
    w = settings.add_group("windowing", info="Sliding-window timing")
    w.add_item("feature_window_t", value=2.0, value_range=[0, None],
               info="Window length used to compute one feature (s)")
    w.add_item("update_rate_t", value=0.25, value_range=[0, None],
               info="Feedback update period, i.e. the window step (s)")


def add_reference_settings(settings: SettingsTree) -> None:
    """Add the ``reference`` item (how the feature is expressed vs the baseline)."""
    settings.add_item("reference", value="percent", value_options=REFERENCE_MODES,
                      info="How to express the feature relative to the calibration baseline")


def preprocess(signal: Signal, channels: list, apply_car: bool,
               filter_spec: dict) -> "tuple[NDArray, float]":
    """Pick channels, optional CAR, band-pass; return ``(filtered_signal, fs)``."""
    x = harmonize_channels(signal, channels)
    raw = car(x.signal) if apply_car else x.signal
    return make_filter(filter_spec).fit_transform(raw, x.fs), x.fs


def sliding_windows(signal: NDArray, fs: float, window_t: float, rate_t: float) -> NDArray:
    """Cut ``signal`` into overlapping windows: ``(n_windows, n_samples, n_channels)``.

    Windows are ``window_t`` seconds long and start every ``rate_t`` seconds (the feedback
    update period), like the online loop. Returns an empty ``(0, n_samples, n_channels)``
    array when the signal is shorter than one window.
    """
    n, n_cha = signal.shape[0], signal.shape[1]
    w = int(round(window_t * fs))
    step = max(1, int(round(rate_t * fs)))
    if w <= 0 or n < w:
        return np.empty((0, max(w, 0), n_cha))
    starts = range(0, n - w + 1, step)
    return np.stack([signal[s:s + w] for s in starts])


def apply_reference(values: NDArray, baseline: float, mode: str) -> NDArray:
    """Express the per-window feature ``values`` relative to the calibration ``baseline``.

    See :data:`REFERENCE_MODES`. This is the explicit answer to the old code's commented-out
    baseline subtraction: the referencing is a configured, documented step.
    """
    v = np.asarray(values, dtype=float)
    if mode == "none":
        return v
    if mode == "subtract":
        return v - baseline
    if mode == "divide":
        return v / baseline
    if mode == "percent":
        return 100.0 * (v - baseline) / baseline
    raise ValueError(f"unknown reference mode {mode!r}; use one of {REFERENCE_MODES}.")


def check_signal(recording, cfg: dict, current_fs: "float | None") -> float:
    """Validate the signal, channels and sampling rate for a neurofeedback pipeline.

    Adopts ``fs`` from the first recording (``current_fs is None``), then requires every later
    recording to match. Returns the sampling rate to store on the pipeline. There are **no
    events** to check: neurofeedback slides its own windows over the continuous signal.
    """
    sig = recording.signals.get(cfg["signal_key"])
    if sig is None:
        raise ValueError(f"recording has no {cfg['signal_key']!r} signal.")
    if not cfg["channels"]:
        raise ValueError("no channels configured; set the 'channels' setting.")
    fs = sig.fs if current_fs is None else current_fs
    if sig.fs != fs:
        raise ValueError(f"fs mismatch: pipeline={fs}, recording={sig.fs}.")
    missing = [c for c in cfg["channels"] if c not in sig.channel_set.labels]
    if missing:
        raise ValueError(f"recording is missing channels {missing}.")
    return fs


def feature_trace(recording, cfg: dict, window_feature) -> NDArray:
    """One feature per sliding window over a recording: preprocess, window, apply ``window_feature``.

    ``window_feature(window, fs, cfg) -> float`` is the per-pipeline feature (band power, a
    connectivity metric). Returns the raw ``(n_windows,)`` feature trace (not yet baseline-referenced).
    """
    filtered, fs = preprocess(
        recording.signals[cfg["signal_key"]], cfg["channels"], cfg["car"], cfg["filter"])
    win = cfg["windowing"]
    windows = sliding_windows(filtered, fs, win["feature_window_t"], win["update_rate_t"])
    return np.array([window_feature(w, fs, cfg) for w in windows], dtype=float)


def calibrate_baseline(recordings, *, check_consistency, cfg: dict, window_feature) -> float:
    """The calibration baseline: the mean feature over every window of the calibration recordings.

    This is what ``fit`` does for neurofeedback -- an **unsupervised** baseline, no labels. It
    calls ``check_consistency`` on each recording (adopting/validating ``fs``) before use.
    """
    values = []
    for rec in recordings:
        check_consistency(rec)
        values.append(feature_trace(rec, cfg, window_feature))
    values = np.concatenate(values) if values else np.empty(0)
    if values.size == 0:
        raise ValueError(
            "no calibration windows; the calibration signal is shorter than one feature window.")
    return float(np.mean(values))
