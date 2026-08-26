"""Model-agnostic building blocks shared across the VEP-speller decoding pipelines.

Private helpers used by every Layer-1 pipeline and the Layer-2 command decoder: the
per-cycle / per-frame event arrays and the shared trial/cycle ordering. The frequency-filtering
schema and its application are shared with every ``bci`` paradigm and live in
:mod:`medusa.pipelines.bci._filtering`
(:func:`~medusa.pipelines.bci._filtering.add_notch_and_filterbank_settings` /
:func:`~medusa.pipelines.bci._filtering.apply_notch_and_filterbank`).

Events carry **one row per stimulation cycle** (``onset`` = cycle start; ``code_idx`` = the
code that cycle showed, ``0`` for single-code paradigms). The per-frame onsets come from
``onset + frame / fps_resolution``.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# Events -> per-cycle / per-frame arrays
# --------------------------------------------------------------------------- #
def _cycle_arrays(events):
    """Per-cycle ``(onsets, trial_idx, cycle_idx, code_idx)`` from the stimulation events.

    One row per stimulation *cycle* (``onset`` = cycle start); rows with a null
    ``cycle_idx`` are ignored. ``code_idx`` (which code that cycle presented) is mandatory
    -- ``0`` for single-code paradigms (c-VEP / SSVEP).
    """
    df = events.df
    df = df[df["cycle_idx"].notna()]
    return (df["onset"].to_numpy(dtype=float),
            df["trial_idx"].to_numpy(dtype=int),
            df["cycle_idx"].to_numpy(dtype=int),
            df["code_idx"].to_numpy(dtype=int))


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
