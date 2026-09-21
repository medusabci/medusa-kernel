"""Pure scoring functions for VEP spellers: BWR training labels and the cumulative accumulators.

These are the model-agnostic, family-level pure functions the Layer-1 pipelines lean on:

* :func:`bwr_labels` -- per-frame target labels for BWR training, read straight from the codes.
* :func:`bwr_frame_scores` + :func:`bwr_command_scores_corr` -- the BWR cumulative
  ``(n_cycles, n_commands)`` matrix the BWR pipelines return (classify each frame, collapse
  the posterior to one score per frame, then *concatenate-then-correlate* each command's
  code with the frame scores).
* :func:`bwr_loglik_command_scores` -- the Bayes alternative (add up, per command, the log
  posterior of the levels its code claims). A pure function under study; no pipeline uses
  it, see its docstring for the class-prior bias.
* :func:`tm_command_scores` -- the template-matching cumulative matrix (coherently average a
  trial's cycle segments, then score the average against each command's reference).

Each accumulator owns only the family-level cumulation, not the scoring *method* -- the method
(LDA frame classifier, CCA canonical correlation, a learned template, ...) lives in the
pipeline that calls it. Both emit the cumulative ``(n_cycles, n_commands)`` matrix the
paradigm-agnostic
:func:`~medusa.pipelines.bci.vep_spellers.decoding.command_decoder.select_commands`
selects from.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from medusa.core.data.recording import Recording
from medusa.pipelines.bci.vep_spellers.data import SpellerData, cycle_arrays
from medusa.pipelines.bci.vep_spellers.decoding._common import _trial_cycle_order

__all__ = [
    "bwr_labels",
    "bwr_frame_scores",
    "bwr_command_scores_corr",
    "bwr_loglik_command_scores",
    "tm_command_scores",
]


def bwr_labels(recording: Recording) -> NDArray:
    """Per-frame target labels (1 or 0) for BWR training, taken straight from the codes.

    Each cycle contributes the trial target's code *for that cycle*. Take a cycle of a
    trial whose target is command ``k`` and whose code index is ``c``. Its ``n_frames``
    labels are ``codes[k][c]``: 1 on the frames that light the target, 0 elsewhere. The
    labels are in cycle-major order, to match the per-frame scores.

    Parameters
    ----------
    recording :
        A calibration recording: a :class:`~medusa.pipelines.bci.vep_spellers.data.SpellerData`
        experiment (for the codes and the ``spell_target``) plus one event row per
        stimulation cycle (for the trial and code index of each cycle).

    Returns
    -------
    numpy.ndarray
        ``(n_cycles * n_frames,)``. Integer labels, ``1`` on the frames that light the
        trial's target command and ``0`` elsewhere, in cycle-major order. They line up
        one-to-one with the per-frame features a BWR pipeline builds from the same
        recording, so they can be passed straight to a frame classifier's ``fit``.

    Raises
    ------
    ValueError
        If the recording has no ``spell_target`` (labels cannot be derived).

    Examples
    --------
    >>> from medusa.pipelines.bci.vep_spellers.decoding import bwr_labels
    >>> y = bwr_labels(recording)               # doctest: +SKIP
    >>> y[:8]                                   # doctest: +SKIP
    array([1, 1, 0, 1, 0, 0, 1, 0])
    """
    sd = SpellerData.from_recording(recording)
    if sd.spell_target is None:
        raise ValueError("recording has no spell_target; cannot derive BWR labels.")
    _, trial, _, code_idx = cycle_arrays(recording.events)
    codes = sd.codes
    row = {uid: i for i, uid in enumerate(sd.command_uids)}
    target = list(sd.spell_target)
    return np.concatenate(
        [codes[row[str(target[t])], c] for t, c in zip(trial, code_idx)]).astype(int)


# --------------------------------------------------------------------------- #
# Cumulative score accumulators (pure functions, one per family)
# --------------------------------------------------------------------------- #
def tm_command_scores(cycle_segments: NDArray, score_fn: Callable,
                      cycle_trial: NDArray, cycle_idx: NDArray) -> NDArray:
    """Cumulative template-matching command scores, shape ``(n_cycles, n_commands)``.

    The template-matching counterpart of :func:`bwr_command_scores_corr`. Row ``i`` holds
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


def bwr_frame_scores(proba: NDArray, classes: NDArray) -> NDArray:
    """Collapse a frame classifier's posterior to one score per frame, monotone in the level.

    ``proba`` is the ``(n_frames, n_levels)`` posterior of a classifier fitted on
    :func:`bwr_labels`, with its columns ordered as ``classes`` (sklearn's ``classes_``:
    the code levels seen at fit time, sorted). The score is the **posterior-expected code
    level** ``proba @ classes``. For a binary code (``classes == [0, 1]``) that is exactly
    ``proba[:, 1]``, the classic probability that the frame lit the target, so nothing
    changes there.

    A p-ary code needs the whole posterior. :func:`bwr_command_scores_corr` correlates the
    frame scores with each command's code *values*, so the score has to grow with the
    level the frame showed. Keeping one column alone (``proba[:, 1]``, the probability of
    level 1) is a bump at one level, not a monotone score, and it decodes deterministically
    wrong: a GF(p) m-sequence satisfies ``s[t + N/2] == (-s[t]) mod p``, so the command
    whose code is the level-negation of the target -- the one half a period away -- then
    scores higher than the target itself, from a perfect classifier included.

    Parameters
    ----------
    proba :
        ``(n_frames, n_levels)``. Class probabilities per frame, in cycle-major order.
    classes :
        ``(n_levels,)``. The code level each column stands for (``clf.classes_``).

    Returns
    -------
    numpy.ndarray
        ``(n_frames,)``. Per-frame scores, ready for :func:`bwr_command_scores_corr`.

    Raises
    ------
    ValueError
        If ``proba`` is not 2-D or its columns do not match ``classes``.

    Examples
    --------
    >>> import numpy as np
    >>> from medusa.pipelines.bci.vep_spellers.decoding import bwr_frame_scores
    >>> proba = np.array([[0.2, 0.8], [0.9, 0.1]])
    >>> bwr_frame_scores(proba, classes=[0, 1]).tolist()      # binary: P(target)
    [0.8, 0.1]
    >>> bwr_frame_scores(np.eye(3), classes=[0, 1, 2]).tolist()  # 3 levels: the level
    [0.0, 1.0, 2.0]
    """
    proba = np.asarray(proba, dtype=float)
    classes = np.asarray(classes, dtype=float).reshape(-1)
    if proba.ndim != 2 or proba.shape[1] != classes.size:
        raise ValueError(
            f"proba must be (n_frames, n_levels) with one column per class; got shape "
            f"{proba.shape} for {classes.size} classes {classes.tolist()}.")
    return proba @ classes


def bwr_command_scores_corr(frame_scores: NDArray, codes: NDArray, cycle_trial: NDArray,
                            cycle_idx: NDArray, cycle_code_idx: NDArray) -> NDArray:
    """Cumulative per-cycle BWR command correlations, shape ``(n_cycles, n_commands)``.

    Row ``i`` holds, for every command, the Pearson correlation between two things: the
    frame scores of all cycles of row ``i``'s trial up to and including cycle ``i`` (joined
    end to end), and that command's codes joined the same way. This is the
    *concatenate-then-correlate* rule of the c-VEP/ERP BWR decoder, and what the BWR
    pipelines return. Rows follow the input cycle order. A command whose code is constant
    over the used cycles gets ``-inf``. Feed it :func:`bwr_frame_scores` so that a p-ary
    code decodes right.

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
    >>> from medusa.pipelines.bci.vep_spellers.decoding import bwr_command_scores_corr
    >>> codes = np.array([[[1, 1, 0, 0]], [[1, 0, 1, 0]]])   # 2 commands, 1 code, 4 frames
    >>> frame_scores = np.array([1.0, 1.0, 0.0, 0.0])        # 1 cycle; matches command 0
    >>> scores = bwr_command_scores_corr(frame_scores, codes,
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


def bwr_loglik_command_scores(proba: NDArray, classes: NDArray, codes: NDArray,
                              cycle_trial: NDArray, cycle_idx: NDArray,
                              cycle_code_idx: NDArray,
                              eps: float = 1e-12) -> NDArray:
    """Cumulative per-cycle BWR log-likelihoods, shape ``(n_cycles, n_commands)``.

    The Bayes counterpart of :func:`bwr_command_scores_corr`. Row ``i`` holds, for every
    command, ``sum_t log P(level = code_cmd(t) | x_t)`` over the frames of all cycles of
    row ``i``'s trial up to and including cycle ``i``: how probable the levels that command
    claims are under the classifier's posterior. It reads the whole posterior instead of a
    scalar per frame, so a level-negated code (the half-period partner of a p-ary
    m-sequence) is simply a wrong code. The frames are treated as independent; with
    overlapping epochs the values are inflated, so they rank commands but are not
    calibrated probabilities.

    **Under study; the BWR pipelines do not use it.** The posterior carries the class
    prior the classifier learned, so for a binary code this sum equals the *uncentred*
    dot product of the code with the frame log-odds. When the commands' codes have
    unequal weights (random codebooks), the prior offset favours the codes with fewer
    lit frames, and this rule loses to :func:`bwr_command_scores_corr`, which centres
    both signals. Dividing the posterior by the class prior removes the bias and brings
    it level with correlation, not above it (synthetic random-code c-VEP, six seeds).

    Parameters
    ----------
    proba :
        ``(n_cycles * n_frames, n_levels)``. Per-frame posterior, columns as ``classes``,
        cycle-major order (``clf.predict_proba`` of the BWR features).
    classes :
        ``(n_levels,)``. The code level each column stands for (``clf.classes_``).
    codes, cycle_trial, cycle_idx, cycle_code_idx :
        As in :func:`bwr_command_scores_corr`.
    eps :
        Floor on the probabilities before the log, so one over-confident wrong frame
        cannot veto a command with ``-inf``.

    Returns
    -------
    numpy.ndarray
        ``(n_cycles, n_commands)``. Cumulative command log-likelihoods, ready for
        :func:`~medusa.pipelines.bci.vep_spellers.decoding.command_decoder.select_commands`.

    Raises
    ------
    ValueError
        If ``proba`` does not have one row per frame and one column per class, if the
        codes are not integer levels, or if they show a level the classifier was not
        fitted on.

    Examples
    --------
    >>> import numpy as np
    >>> from medusa.pipelines.bci.vep_spellers.decoding import bwr_loglik_command_scores
    >>> codes = np.array([[[1, 1, 0, 0]], [[1, 0, 1, 0]]])   # 2 commands, 1 code, 4 frames
    >>> proba = np.array([[0.1, 0.9], [0.2, 0.8], [0.9, 0.1], [0.8, 0.2]])  # P(0), P(1)
    >>> scores = bwr_loglik_command_scores(proba, classes=[0, 1], codes=codes,
    ...                                    cycle_trial=np.array([0]),
    ...                                    cycle_idx=np.array([0]),
    ...                                    cycle_code_idx=np.array([0]))
    >>> scores.shape
    (1, 2)
    >>> int(scores.argmax())            # command 0 wins
    0
    """
    proba = np.asarray(proba, dtype=float)
    classes = np.asarray(classes).reshape(-1)
    codes = np.asarray(codes)
    levels = np.rint(codes).astype(int)
    if not np.array_equal(levels, codes):
        raise ValueError("codes must hold integer levels.")
    codes = levels
    cycle_code_idx = np.asarray(cycle_code_idx, dtype=int)
    n_commands, _, n_frames = codes.shape
    n_cycles = len(cycle_code_idx)
    if proba.ndim != 2 or proba.shape != (n_cycles * n_frames, classes.size):
        raise ValueError(
            f"proba has shape {proba.shape} but expected (n_cycles*n_frames, n_levels) = "
            f"({n_cycles * n_frames}, {classes.size}).")
    # a level the classifier never saw has no posterior column: a mismatch
    known = np.isin(codes, classes)
    if not known.all():
        raise ValueError(
            f"codes show levels {np.unique(codes[~known]).tolist()} the classifier was "
            f"not fitted on (classes {classes.tolist()}).")
    # the `proba` column of every code level, so cols[k, c, t] indexes the posterior
    column = np.full(int(classes.max()) + 1, -1)
    column[classes.astype(int)] = np.arange(classes.size)
    cols = column[codes]                                  # (n_commands, n_codes, n_frames)
    logp = np.log(np.clip(proba, eps, 1.0)).reshape(n_cycles, n_frames, classes.size)
    # per cycle, per command: sum over frames of log P(level = the command's level there)
    per_cycle = np.empty((n_cycles, n_commands))
    frame = np.arange(n_frames)
    for i in range(n_cycles):
        per_cycle[i] = logp[i][frame[None, :], cols[:, cycle_code_idx[i], :]].sum(axis=1)
    out = np.full((n_cycles, n_commands), -np.inf)
    for _, order in _trial_cycle_order(cycle_trial, cycle_idx):  # same order as corr/TM
        out[order] = np.cumsum(per_cycle[order], axis=0)
    return out
