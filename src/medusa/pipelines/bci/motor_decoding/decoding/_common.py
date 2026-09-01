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

from medusa.core.settings_tree import SettingsTree
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


def add_training_settings(clf_group: SettingsTree) -> None:
    """Add a ``training`` subgroup of :class:`~medusa.ml.torch_models.classification.TorchClassifier`
    hyper-parameters to a classifier settings group (shared by the deep motor pipelines).

    Torch-free: it only builds the settings schema. Keep ``device='auto'`` so a saved model
    reloads on any host (a fixed ``'cuda'`` fails to load on a CPU-only machine).
    """
    tr = clf_group.add_group("training", info="TorchClassifier training hyper-parameters")
    tr.add_item("max_epochs", value=100, value_range=[1, None], info="Maximum training epochs")
    tr.add_item("batch_size", value=64, value_range=[1, None], info="Mini-batch size")
    tr.add_item("learning_rate", value=1e-3, value_range=[0, None], info="Adam learning rate")
    tr.add_item("val_split", value=0.2, optional=True, value_range=[0, 1],
                info="Validation fraction for early stopping; switch it off to train "
                     "without a validation split")
    tr.add_item("patience", value=10, value_range=[1, None],
                info="Early-stopping patience (epochs)")
    tr.add_item("device", value="auto",
                info="Compute device ('auto', 'cpu', 'cuda', 'cuda:N', 'mps'); keep 'auto' "
                     "so a saved model reloads on any host")
    tr.add_item("verbose", value="epoch",
                value_options=["silent", "epoch", "full"],
                info="Training log: 'silent' / 'epoch' (one line per epoch) / "
                     "'full' (Lightning debug)")


def training_kwargs(cfg_training: dict) -> dict:
    """Map a ``classifier.training`` config dict to
    :class:`~medusa.ml.torch_models.classification.TorchClassifier` keyword arguments."""
    t = cfg_training
    return dict(lr=float(t["learning_rate"]), max_epochs=int(t["max_epochs"]),
                batch_size=int(t["batch_size"]), val_split=t["val_split"] or None,
                patience=int(t["patience"]), device=t["device"], verbose=t["verbose"])
