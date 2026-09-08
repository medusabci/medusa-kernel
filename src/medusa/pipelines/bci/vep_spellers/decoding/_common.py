"""Model-agnostic building blocks shared across the VEP-speller decoding pipelines.

Private helpers used by every Layer-1 pipeline and by the Layer-2 selector: the per-frame
onsets and the shared trial/cycle ordering. Reading a recording's events into the four
per-cycle arrays is the public seam, so it lives with the rest of the speller events
contract in :func:`~medusa.pipelines.bci.vep_spellers.data.cycle_arrays`. The
frequency-filtering schema and its application are shared with every ``bci`` paradigm and
live in :mod:`medusa.pipelines.bci._filtering`
(:func:`~medusa.pipelines.bci._filtering.add_notch_and_filterbank_settings` /
:func:`~medusa.pipelines.bci._filtering.apply_notch_and_filterbank`).

Events carry **one row per stimulation cycle** (``onset`` = cycle start; ``code_idx`` = the
code that cycle showed, ``0`` for single-code paradigms). The per-frame onsets come from
``onset + frame / fps_resolution``.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# Per-cycle arrays -> per-frame onsets / shared cycle ordering
# --------------------------------------------------------------------------- #
def _bit_onsets(cycle_onsets, n_frames, fps):
    """Expand cycle onsets to per-frame bit onsets ``cycle_onset + frame / fps``.

    Returns a flat array in cycle-major order (cycle 0 frames ``0..n_frames-1``, ...).
    """
    frame = np.arange(n_frames)
    return (np.asarray(cycle_onsets, dtype=float)[:, None]
            + frame[None, :] / fps).ravel()


def _trial_cycle_order(cycle_trial, cycle_idx):
    """Yield ``(trial, ordered_row_indices)``: each trial's cycle rows in ``cycle_idx`` order.

    The Layer-1 accumulators and the Layer-2 selector share this, so they go through a
    recording's cycles in the same order (each trial's rows sorted by ``cycle_idx``, stably).
    """
    cycle_trial = np.asarray(cycle_trial, dtype=int)
    cycle_idx = np.asarray(cycle_idx, dtype=int)
    for t in np.unique(cycle_trial):
        m = np.where(cycle_trial == t)[0]
        yield int(t), m[np.argsort(cycle_idx[m], kind="stable")]


# --------------------------------------------------------------------------- #
# Resampling: target_fs against the recording rate and the filter bank
# --------------------------------------------------------------------------- #
def _check_target_fs(cfg: dict, fs: float) -> None:
    """Check ``segmentation.target_fs`` against a recording's ``fs`` and its filter bank.

    Does nothing when ``target_fs`` is unset: the epochs then keep the recording's own
    rate. When it is set, :func:`~medusa.signal.segmentation.resample_segments` rewrites
    every epoch to that rate and drops everything from ``target_fs / 2`` (the new Nyquist
    frequency) up, so two settings have to agree with it:

    * The filter bank must already bound the signal below that limit -- every filter a
      band-pass or low-pass with an upper cutoff under it. A ``highpass`` or ``bandstop``
      leaves the signal unbounded above, and a cutoff at or above the limit means the top
      of the band the user asked for is thrown away by the resampling instead of reaching
      the classifier.
    * ``target_fs`` must not exceed the recording's rate, which would only interpolate.

    Raises ``ValueError`` naming the offending cutoff and the Nyquist limit.
    """
    target_fs = cfg["segmentation"]["target_fs"]
    if not target_fs:
        return
    if target_fs > fs:
        raise ValueError(
            f"segmentation.target_fs ({target_fs} Hz) is above the recording's fs "
            f"({fs} Hz); resampling up only interpolates, it recovers nothing. Lower "
            f"target_fs, or unset it to keep the recording's own rate.")
    nyquist = target_fs / 2
    for i, spec in enumerate(cfg["freq_filtering"]["filterbank"]):
        band_type = spec.get("band_type")
        if band_type not in ("bandpass", "lowpass"):
            raise ValueError(
                f"freq_filtering.filterbank[{i}] is a {band_type!r} filter, which "
                f"leaves the signal unbounded above, but segmentation.target_fs="
                f"{target_fs} Hz keeps only what is below {nyquist} Hz (Nyquist). Use a "
                f"bandpass or lowpass with an upper cutoff below {nyquist} Hz.")
        cutoff = spec.get("cutoff")
        upper = float(cutoff[-1] if isinstance(cutoff, (list, tuple)) else cutoff)
        if upper >= nyquist:
            raise ValueError(
                f"freq_filtering.filterbank[{i}] passes up to {upper} Hz, but "
                f"segmentation.target_fs={target_fs} Hz keeps only what is below "
                f"{nyquist} Hz (Nyquist), so resampling would discard the band from "
                f"{nyquist} Hz up. Lower the filter's upper cutoff below {nyquist} Hz, "
                f"or raise target_fs.")
