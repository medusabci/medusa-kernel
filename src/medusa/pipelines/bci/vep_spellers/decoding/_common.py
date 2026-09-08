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
