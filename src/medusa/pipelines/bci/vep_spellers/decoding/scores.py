"""Pure scoring functions for VEP spellers: BWR training labels and the cumulative accumulators.

These are the model-agnostic, family-level pure functions the Layer-1 pipelines lean on:

* :func:`bwr_labels` -- per-frame target labels for BWR training, read straight from the codes.
* :func:`bwr_command_scores` -- the BWR cumulative ``(n_cycles, n_commands)`` matrix
  (classify each frame, then *concatenate-then-correlate* each command's code with the frame
  scores).
* :func:`tm_command_scores` -- the template-matching cumulative matrix (coherently average a
  trial's cycle segments, then score the average against each command's reference).

Each accumulator owns only the family-level cumulation, not the scoring *method* -- the method
(LDA frame classifier, CCA canonical correlation, a learned template, ...) lives in the
pipeline that calls it. Both emit the cumulative ``(n_cycles, n_commands)`` matrix the
paradigm-agnostic :class:`~medusa.pipelines.bci.vep_spellers.decoding.command_decoder.VEPCommandDecoder`
selects from.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from medusa.core.data.recording import Recording
from medusa.pipelines.bci.vep_spellers.data import SpellerData
from medusa.pipelines.bci.vep_spellers.decoding._common import (
    _cycle_arrays, _trial_cycle_order)

__all__ = [
    "bwr_labels",
    "bwr_command_scores",
    "tm_command_scores",
]


def bwr_labels(recording: Recording) -> NDArray:
    """Per-frame target labels (1 or 0) for BWR training, taken straight from the codes.

    Each cycle contributes the trial target's code *for that cycle*. Take a cycle of a
    trial whose target is command ``k`` and whose code index is ``c``. Its ``n_frames``
    labels are ``codes[k][c]``: 1 on the frames that light the target, 0 elsewhere. The
    labels are in cycle-major order, to match the per-frame scores.

    Raises
    ------
    ValueError
        If the recording has no ``spell_target`` (labels cannot be derived).
    """
    sd = SpellerData.from_recording(recording)
    if sd.spell_target is None:
        raise ValueError("recording has no spell_target; cannot derive BWR labels.")
    _, trial, _, code_idx = _cycle_arrays(recording.events)
    codes = sd.codes
    row = {uid: i for i, uid in enumerate(sd.command_uids)}
    target = list(sd.spell_target)
    return np.concatenate(
        [codes[row[str(target[t])], c] for t, c in zip(trial, code_idx)]).astype(int)


# --------------------------------------------------------------------------- #
# Cumulative score accumulators (pure functions, one per family)
# --------------------------------------------------------------------------- #
def _corr_rows(codes: NDArray, scores: NDArray) -> NDArray:
    """Absolute Pearson correlation ``|r|`` of every row of ``codes`` (n, m) with ``scores``.

    Magnitude, not signed. This matches the proven BWR command decoder, which ranks commands
    by ``|corr|`` (so a fully sign-inverted reconstruction still scores its code high). A row
    (or ``scores``) with zero variance gives ``-inf``, so it is never selected.
    """
    codes = codes.astype(float)
    scores = scores.astype(float)
    c = codes - codes.mean(axis=1, keepdims=True)
    s = scores - scores.mean()
    num = c @ s
    den = np.sqrt((c ** 2).sum(axis=1) * (s ** 2).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.abs(num / den)
    corr[~np.isfinite(corr)] = -np.inf
    return corr


def bwr_command_scores(frame_scores: NDArray, codes: NDArray, cycle_trial: NDArray,
                       cycle_idx: NDArray, cycle_code_idx: NDArray) -> NDArray:
    """Cumulative per-cycle BWR command correlations, shape ``(n_cycles, n_commands)``.

    Row ``i`` holds, for every command, the Pearson correlation between two things: the
    frame scores of all cycles of row ``i``'s trial up to and including cycle ``i`` (joined
    end to end), and that command's codes joined the same way. This is the
    *concatenate-then-correlate* rule of the c-VEP/ERP BWR decoder. Rows follow the input
    cycle order. A command whose code is constant over the used cycles gets ``-inf``.

    Parameters
    ----------
    frame_scores :
        ``(n_cycles * n_frames,)``. Per-frame target-class scores, in cycle-major order
        (as produced inside a BWR pipeline's ``predict``).
    codes :
        ``(n_commands, n_codes, n_frames)``. Per-command codes (``SpellerData.codes``).
    cycle_trial, cycle_idx, cycle_code_idx :
        ``(n_cycles,)`` each. Per-cycle trial index, repetition index, and code index
        (from the events).

    Returns
    -------
    numpy.ndarray
        ``(n_cycles, n_commands)``. Cumulative command correlations, ready for
        :func:`~medusa.pipelines.bci.vep_spellers.decoding.command_decoder.select_commands`.

    Raises
    ------
    ValueError
        If ``frame_scores.size`` is not ``n_cycles * n_frames``.

    Examples
    --------
    >>> import numpy as np
    >>> from medusa.pipelines.bci.vep_spellers.decoding import bwr_command_scores
    >>> codes = np.array([[[1, 1, 0, 0]], [[1, 0, 1, 0]]])   # 2 commands, 1 code, 4 frames
    >>> frame_scores = np.array([1.0, 1.0, 0.0, 0.0])        # 1 cycle; matches command 0
    >>> scores = bwr_command_scores(frame_scores, codes,
    ...                             cycle_trial=np.array([0]), cycle_idx=np.array([0]),
    ...                             cycle_code_idx=np.array([0]))
    >>> scores.shape
    (1, 2)
    >>> int(scores.argmax())            # command 0 wins
    0
    """
    frame_scores = np.asarray(frame_scores, dtype=float)
    codes = np.asarray(codes)
    cycle_code_idx = np.asarray(cycle_code_idx, dtype=int)
    n_commands, _, n_frames = codes.shape
    n_cycles = len(cycle_code_idx)
    if frame_scores.size != n_cycles * n_frames:
        raise ValueError(
            f"frame_scores has {frame_scores.size} values but expected "
            f"n_cycles*n_frames = {n_cycles}*{n_frames}.")
    scores_by_cycle = frame_scores.reshape(n_cycles, n_frames)
    out = np.full((n_cycles, n_commands), -np.inf)
    for _, order in _trial_cycle_order(cycle_trial, cycle_idx):
        for i in range(len(order)):
            used = order[:i + 1]
            s = scores_by_cycle[used].ravel()
            # every command's expected code over the used cycles (in `used` order),
            # concatenated to match the flattened frame scores.
            exp = codes[:, cycle_code_idx[used]].reshape(n_commands, -1)
            out[order[i]] = _corr_rows(exp, s)
    return out


def tm_command_scores(cycle_segments: NDArray, score_fn: Callable,
                      cycle_trial: NDArray, cycle_idx: NDArray) -> NDArray:
    """Cumulative template-matching command scores, shape ``(n_cycles, n_commands)``.

    The template-matching counterpart of :func:`bwr_command_scores`. Row ``i`` holds
    ``score_fn`` applied to the **coherent average** of the segments of all cycles of row
    ``i``'s trial up to and including cycle ``i`` (the c-VEP/SSVEP averaging rule). The
    per-command scoring *method* is passed in as ``score_fn``. It may be CCA canonical
    correlation against a synthetic or a learned reference, TRCA-filtered correlation, and
    so on. So this function owns only the family-level accumulation, not the method itself
    (CCA is a method inside
    :class:`~medusa.pipelines.bci.vep_spellers.decoding.template_matching.TMCCAPipeline`, not
    built in here).

    Parameters
    ----------
    cycle_segments :
        ``(n_cycles, n_samples, n_channels)``. One multichannel EEG segment per cycle (as
        produced inside a pipeline's ``predict``).
    score_fn :
        ``score_fn(avg_segment) -> ndarray (n_commands,)``. The per-command similarity for
        one coherently-averaged segment (for example ``TMCCAPipeline``'s CCA scorer).
    cycle_trial, cycle_idx :
        ``(n_cycles,)`` each. Per-cycle trial index and repetition index (from the events).

    Returns
    -------
    numpy.ndarray
        ``(n_cycles, n_commands)``. Cumulative command scores, ready for
        :func:`~medusa.pipelines.bci.vep_spellers.decoding.command_decoder.select_commands`.

    Examples
    --------
    >>> import numpy as np
    >>> from medusa.pipelines.bci.vep_spellers.decoding import tm_command_scores
    >>> cycle_segments = np.ones((2, 5, 3))     # 2 cycles, 5 samples, 3 channels
    >>> score_fn = lambda avg: avg.mean(axis=(0, 1)) * np.array([1.0, 2.0])
    >>> scores = tm_command_scores(cycle_segments, score_fn,
    ...                            cycle_trial=np.array([0, 0]),
    ...                            cycle_idx=np.array([0, 1]))
    >>> scores.shape
    (2, 2)
    >>> int(scores[-1].argmax())        # command 1 scores higher
    1
    """
    cycle_segments = np.asarray(cycle_segments, dtype=float)
    n_cycles = cycle_segments.shape[0]
    rows = {}
    for _, order in _trial_cycle_order(cycle_trial, cycle_idx):
        for i in range(len(order)):
            avg = cycle_segments[order[:i + 1]].mean(axis=0)     # (n_samples, n_channels)
            rows[int(order[i])] = np.asarray(score_fn(avg), dtype=float)
    n_commands = len(next(iter(rows.values()))) if rows else 0
    out = np.full((n_cycles, n_commands), -np.inf)
    for idx, r in rows.items():
        out[idx] = r
    return out
