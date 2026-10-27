"""Score functions for VEP spellers: BWR training labels and cumulative command scores.

The Layer-1 pipelines build on these pure functions. None of them depends on a specific
model:

* :func:`bwr_labels` -- per-frame training labels for a BWR frame classifier.
* :func:`bwr_frame_scores` -- one score per frame from the classifier's probabilities.
* :func:`bwr_command_scores_corr` -- BWR command scores: correlation between each
  command's code and the frame scores.
* :func:`tm_command_scores` -- template-matching command scores: average the trial's
  cycles, then score the average against each command.

Both command-score functions return the same thing: a cumulative
``(n_cycles, n_commands)`` matrix whose row ``i`` scores every command using all cycles of
its trial up to and including cycle ``i``. This matrix is the input of
:func:`~medusa.pipelines.bci.vep_spellers.decoding.command_decoder.select_commands`. The
functions only accumulate over cycles; the scoring method itself (LDA, CCA, a learned
template, ...) comes from the pipeline.

Conventions: ``n_cycles`` counts the cycle events of the whole recording (all trials
stacked). The per-cycle trial, cycle and code indices come from
:func:`~medusa.pipelines.bci.vep_spellers.data.cycle_arrays`. Per-frame arrays are in
cycle-major order: all frames of the first cycle, then all frames of the second, and so on.
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
    "tm_command_scores",
]


def bwr_labels(recording: Recording) -> NDArray:
    """Per-frame training labels for a BWR frame classifier, read from the codes.

    The label of a frame is the class the trial's target command showed in that frame,
    as written in its code: for a cycle with target ``k`` and code index ``c``, the labels
    are ``codes[k, c]``. What each class stands for (a color, a texture, ...) depends on
    the paradigm; here it is only an integer.

    Parameters
    ----------
    recording :
        A calibration recording. Its experiment must be a
        :class:`~medusa.pipelines.bci.vep_spellers.data.SpellerData` with ``spell_target``
        set, and its events must have one row per stimulation cycle.

    Returns
    -------
    numpy.ndarray
        ``(n_cycles * n_frames,)``. Integer class labels in cycle-major order. They match the
        per-frame features a BWR pipeline builds from the same recording, so they can go
        straight into the frame classifier's ``fit``.

    Raises
    ------
    TypeError
        If ``recording.experiment`` is not a ``SpellerData``.
    ValueError
        If the recording has no ``spell_target``.

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
    _, trial_idx, _, code_idx = cycle_arrays(recording.events)
    codes = sd.codes
    row = {uid: i for i, uid in enumerate(sd.command_uids)}
    target = list(sd.spell_target)
    return np.concatenate(
        [codes[row[str(target[t])], c] for t, c in zip(trial_idx, code_idx)]).astype(int)


# --------------------------------------------------------------------------- #
# Cumulative score accumulators (pure functions, one per family)
# --------------------------------------------------------------------------- #
def tm_command_scores(cycle_segments: NDArray, score_fn: Callable,
                      trial_idx: NDArray, cycle_idx: NDArray) -> NDArray:
    """Cumulative template-matching command scores.

    Template matching averages a trial's cycles before scoring (coherent averaging, as in
    c-VEP and SSVEP). Row ``i`` of the output is ``score_fn`` applied to the average of
    the segments of all cycles of its trial up to and including cycle ``i``. The scoring
    method is ``score_fn`` (CCA against a synthetic or learned reference, TRCA-filtered
    correlation, ...); this function only does the averaging.

    Parameters
    ----------
    cycle_segments :
        ``(n_cycles, n_samples, n_channels)``. One EEG segment per cycle.
    score_fn :
        ``score_fn(avg_segment) -> (n_commands,)``. Scores every command against one
        averaged ``(n_samples, n_channels)`` segment; higher means a better match (for
        example, the CCA scorer of
        :class:`~medusa.pipelines.bci.vep_spellers.decoding.template_matching.TMCCAPipeline`).
    trial_idx, cycle_idx :
        ``(n_cycles,)`` each. Trial index and 0-based cycle index of each segment.

    Returns
    -------
    numpy.ndarray
        ``(n_cycles, n_commands)``. Cumulative command scores, rows in input order. Input
        of :func:`~medusa.pipelines.bci.vep_spellers.decoding.command_decoder.select_commands`.

    Examples
    --------
    Row 1 scores the average of cycles 0 and 1:

    >>> import numpy as np
    >>> from medusa.pipelines.bci.vep_spellers.decoding import tm_command_scores
    >>> cycle_segments = np.stack([np.full((5, 3), 1.0),     # cycle 0
    ...                            np.full((5, 3), 3.0)])    # cycle 1
    >>> score_fn = lambda avg: np.array([avg.mean(), 0.0])   # toy scorer, 2 commands
    >>> cycle_scores = tm_command_scores(cycle_segments, score_fn,
    ...                                  trial_idx=np.array([0, 0]),
    ...                                  cycle_idx=np.array([0, 1]))
    >>> cycle_scores
    array([[1., 0.],
           [2., 0.]])
    """
    cycle_segments = np.asarray(cycle_segments, dtype=float)
    n_cycles = cycle_segments.shape[0]
    rows = {}
    for _, order in _trial_cycle_order(trial_idx, cycle_idx):
        for i in range(len(order)):
            avg = cycle_segments[order[:i + 1]].mean(axis=0)     # (n_samples, n_channels)
            rows[int(order[i])] = np.asarray(score_fn(avg), dtype=float)
    n_commands = len(next(iter(rows.values()))) if rows else 0
    out = np.full((n_cycles, n_commands), -np.inf)
    for idx, r in rows.items():
        out[idx] = r
    return out


def _corr_rows(codes: NDArray, scores: NDArray) -> NDArray:
    """Absolute Pearson correlation ``|r|`` of each row of ``codes`` (n, m) with ``scores`` (m,).

    The sign is dropped, as in the original BWR decoder, so a sign-inverted reconstruction
    still matches its code. A constant row, or constant ``scores``, gives ``-inf`` so it is
    never selected.
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
    """One score per frame from a frame classifier's probabilities.

    The score is the expected class label under the classifier: ``proba @ classes``. For
    a binary code (``classes == [0, 1]``) this is ``proba[:, 1]``, the probability of
    class 1.

    Parameters
    ----------
    proba :
        ``(n_frames, n_classes)``. Class probabilities per frame, in cycle-major order
        (``predict_proba`` of a classifier fitted on :func:`bwr_labels`).
    classes :
        ``(n_classes,)``. The class label of each column of ``proba`` (``clf.classes_``).

    Returns
    -------
    numpy.ndarray
        ``(n_frames,)``. Per-frame scores. Input of :func:`bwr_command_scores_corr`.

    Raises
    ------
    ValueError
        If ``proba`` is not 2-D or does not have one column per class.

    Notes
    -----
    :func:`bwr_command_scores_corr` correlates these scores with the class labels in the
    codes, read as numbers. So the score must grow with the label, and a single column
    such as ``proba[:, 1]`` does not. With more than two classes, that decodes wrong even
    with a perfect classifier: in a GF(p) m-sequence, the code half a period away is the
    target's negation (``s[t + N/2] == -s[t] mod p``), and it scores higher than the
    target itself.

    Examples
    --------
    >>> import numpy as np
    >>> from medusa.pipelines.bci.vep_spellers.decoding import bwr_frame_scores
    >>> proba = np.array([[0.2, 0.8], [0.9, 0.1]])
    >>> bwr_frame_scores(proba, classes=[0, 1]).tolist()         # binary: P(class 1)
    [0.8, 0.1]
    >>> bwr_frame_scores(np.eye(3), classes=[0, 1, 2]).tolist()  # 3 classes: the label
    [0.0, 1.0, 2.0]
    """
    proba = np.asarray(proba, dtype=float)
    classes = np.asarray(classes, dtype=float).reshape(-1)
    if proba.ndim != 2 or proba.shape[1] != classes.size:
        raise ValueError(
            f"proba must be (n_frames, n_classes) with one column per class; got shape "
            f"{proba.shape} for {classes.size} classes {classes.tolist()}.")
    return proba @ classes


def bwr_command_scores_corr(frame_scores: NDArray, codes: NDArray, trial_idx: NDArray,
                            cycle_idx: NDArray, code_idx: NDArray) -> NDArray:
    """Cumulative BWR command scores by correlation.

    Row ``i`` holds, for every command, the absolute Pearson correlation between:

    * the frame scores of all cycles of its trial up to and including cycle ``i``, joined
      end to end in ``cycle_idx`` order, and
    * the command's code for those same cycles (each with its own ``code_idx``), joined
      the same way.

    This is the decoding rule of the BWR pipelines. Taking the absolute value means a
    sign-inverted reconstruction still matches its code. A command whose code is constant
    over the used cycles gets ``-inf``.

    Parameters
    ----------
    frame_scores :
        ``(n_cycles * n_frames,)``. One score per frame, in cycle-major order. It must grow
        with the class label the frame showed: use :func:`bwr_frame_scores` so that codes
        with more than two classes decode right.
    codes :
        ``(n_commands, n_codes, n_frames)``. The codes of every command
        (``SpellerData.codes``).
    trial_idx, cycle_idx, code_idx :
        ``(n_cycles,)`` each. Trial index, 0-based cycle index and code index of each cycle.

    Returns
    -------
    numpy.ndarray
        ``(n_cycles, n_commands)``. Cumulative absolute correlations in ``[0, 1]`` (or
        ``-inf``), rows in input order. Input of
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
    >>> frame_scores = np.array([1.0, 1.0, 0.0, 0.0])        # 1 cycle, matches command 0
    >>> cycle_scores = bwr_command_scores_corr(
    ...     frame_scores, codes, trial_idx=np.array([0]), cycle_idx=np.array([0]),
    ...     code_idx=np.array([0]))
    >>> cycle_scores
    array([[1., 0.]])
    """
    frame_scores = np.asarray(frame_scores, dtype=float)
    codes = np.asarray(codes)
    code_idx = np.asarray(code_idx, dtype=int)
    n_commands, _, n_frames = codes.shape
    n_cycles = len(code_idx)
    if frame_scores.size != n_cycles * n_frames:
        raise ValueError(
            f"frame_scores has {frame_scores.size} values but expected "
            f"n_cycles*n_frames = {n_cycles}*{n_frames}.")
    scores_by_cycle = frame_scores.reshape(n_cycles, n_frames)
    out = np.full((n_cycles, n_commands), -np.inf)
    for _, order in _trial_cycle_order(trial_idx, cycle_idx):
        for i in range(len(order)):
            used = order[:i + 1]
            s = scores_by_cycle[used].ravel()
            # every command's expected code over the used cycles (in `used` order),
            # concatenated to match the flattened frame scores.
            exp = codes[:, code_idx[used]].reshape(n_commands, -1)
            out[order[i]] = _corr_rows(exp, s)
    return out
