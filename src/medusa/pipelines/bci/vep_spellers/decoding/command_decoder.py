"""Layer-2 command selection for VEP spellers (paradigm-agnostic).

A Layer-1 pipeline's :meth:`predict` returns a *cumulative* ``(n_cycles, n_commands)`` score
matrix -- row ``i`` is the decision score for every command after cycle event ``i``,
accumulated the family's natural way (concatenate-then-correlate for BWR, coherent averaging
+ CCA for template matching). This module only **selects**: for each trial, after each cycle,
:func:`select_commands` takes the argmax of the cumulative score row over that trial's
*available* commands. It is paradigm-agnostic -- no codes, no per-family logic -- because
Layer 1 already did the family-specific accumulation.

Layer 2 is stateless and does no fitting, so it is a plain function, not an object: pair
:func:`select_commands` with
:func:`~medusa.pipelines.bci.vep_spellers.data.cycle_arrays`, which reads the per-cycle
trial and repetition indices out of a recording's events. :func:`command_decoding_accuracy`
and :func:`command_decoding_accuracy_per_cycle` score the result; the per-cycle curve is
also what a dynamic-stopping rule is designed and tuned against.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from medusa.pipelines.bci.vep_spellers.decoding._common import _trial_cycle_order

__all__ = [
    "select_commands",
    "command_decoding_accuracy",
    "command_decoding_accuracy_per_cycle",
]


def _available(trial_available_cmmds, command_uids, trial: int) -> "list[str]":
    """The command uids selectable in ``trial`` (all commands when unrestricted)."""
    if trial_available_cmmds is None:
        return list(command_uids)
    if trial >= len(trial_available_cmmds):
        raise ValueError(
            f"trial_available_cmmds has {len(trial_available_cmmds)} entries but "
            f"trial_idx {trial} was seen; index them by trial_idx.")
    return [str(u) for u in trial_available_cmmds[trial]]


def select_commands(cycle_scores: NDArray, command_uids: list[str],
                    trial_idx: NDArray, cycle_idx: NDArray,
                    trial_available_cmmds: list[list[str]] | None = None):
    """Select the decoded command per trial from a cumulative score matrix.

    The paradigm-agnostic Layer-2 rule. Row ``i`` of ``cycle_scores`` is the *cumulative*
    decision score after cycle event ``i`` (a Layer-1 pipeline already accumulated it in
    its family's natural way: :func:`~medusa.pipelines.bci.vep_spellers.decoding.bwr_command_scores`
    or :func:`~medusa.pipelines.bci.vep_spellers.decoding.tm_command_scores`). For each
    trial, and after each cycle, this takes the argmax over that trial's **available**
    commands. Multi-matrix paradigms are handled entirely by ``trial_available_cmmds``.

    Parameters
    ----------
    cycle_scores :
        ``(n_cycles, n_commands)``. Cumulative per-cycle command scores (a Layer-1
        pipeline's ``predict`` output).
    command_uids :
        Length ``n_commands``. Command ``uid``\\ s in score-column order
        (``SpellerData.command_uids``).
    trial_idx, cycle_idx :
        ``(n_cycles,)`` each. Per-cycle trial index and cycle index (one row per
        cycle, from the events).
    trial_available_cmmds :
        Per-trial list of available command ``uid``\\ s
        (``SpellerData.trial_available_cmmds``). ``None`` makes every command available.

    Returns
    -------
    sel_cmd : dict
        ``{trial_idx: command_uid}``. The final selection (all cycles).
    sel_cmd_per_cycle : dict
        ``{trial_idx: {n_cycles: command_uid}}``. The cumulative selection after each
        cycle (for dynamic stopping or accuracy-vs-cycles).
    scores : dict
        ``{trial_idx: {n_cycles: ndarray[n_available]}}``. Per-available-command score.

    Raises
    ------
    ValueError
        If ``trial_available_cmmds`` is given but has fewer entries than the largest
        ``trial_idx`` in ``cycle_trial`` (it is indexed by trial index, not by position).
    KeyError
        If a command ``uid`` listed in ``trial_available_cmmds`` is not in ``command_uids``.

    Examples
    --------
    >>> import numpy as np
    >>> from medusa.pipelines.bci.vep_spellers.decoding import select_commands
    >>> cycle_scores = np.array([[0.1, 0.9], [0.8, 0.2]])    # 2 cycles, 2 commands
    >>> selected, per_cycle, scores = select_commands(
    ...     cycle_scores, command_uids=["A", "B"],
    ...     trial_idx=np.array([0, 0]), cycle_idx=np.array([0, 1]))
    >>> selected[0]                     # the last cycle (idx 1) picks command "A"
    'A'

    On a real recording, the per-cycle arrays come from its events:

    >>> from medusa.pipelines.bci.vep_spellers import (
    ...     SpellerData, cycle_arrays, select_commands)
    >>> sd = SpellerData.from_recording(recording)
    >>> onsets, trial_idx, cycle_idx, code_idx = cycle_arrays(
    recording.events)
    >>> selected, per_cycle, scores = select_commands(
    ...     pipeline.predict(recording), sd.command_uids, trial, cycle,
    ...     sd.trial_available_cmmds)
    """
    cycle_scores = np.asarray(cycle_scores, dtype=float)
    command_uids = list(command_uids)
    cycle_idx = np.asarray(cycle_idx, dtype=int)
    row = {uid: i for i, uid in enumerate(command_uids)}

    selected, per_cycle, scores = {}, {}, {}
    for t, order in _trial_cycle_order(trial_idx, cycle_idx):
        avail = _available(trial_available_cmmds, command_uids, t)
        avail_cols = [row[u] for u in avail]
        per_cycle[t], scores[t] = {}, {}
        for i in order:
            corr = cycle_scores[i, avail_cols]
            nc = int(cycle_idx[i])
            per_cycle[t][nc] = avail[int(np.argmax(corr))]
            scores[t][nc] = corr
        selected[t] = per_cycle[t][max(per_cycle[t])]
    return selected, per_cycle, scores


def command_decoding_accuracy(sel_cmd: dict, target: dict) -> float:
    """Fraction of trials whose final selected command equals the target.

    The end-of-trial accuracy: every trial is counted once, using all of its cycles. Use
    :func:`command_decoding_accuracy_per_cycle` to see how that accuracy grows with the
    number of cycles.

    Parameters
    ----------
    sel_cmd :
        ``{trial_idx: command_uid}``. The final selection per trial: the first output
        of :func:`select_commands`.
    target :
        ``{trial_idx: command_uid}``. The command the user was asked to spell in each
        trial. Trials that are not in ``target`` are ignored, so a recording may mix
        labelled and unlabelled trials.

    Returns
    -------
    float
        Accuracy in ``[0, 1]``, or ``nan`` when no trial of ``selected`` appears in
        ``target`` (nothing could be scored).

    Examples
    --------
    >>> from medusa.pipelines.bci.vep_spellers.decoding import command_decoding_accuracy
    >>> selected = {0: "A", 1: "B", 2: "C"}
    >>> command_decoding_accuracy(selected, target={0: "A", 1: "B", 2: "D"})
    0.6666666666666666
    """
    keys = [t for t in sel_cmd if t in target]
    if not keys:
        return float("nan")
    correct = sum(str(sel_cmd[t]) == str(target[t]) for t in keys)
    return correct / len(keys)


def command_decoding_accuracy_per_cycle(sel_cmd_per_cycle: dict, target: dict) -> NDArray:
    """Accuracy as a function of the number of cycles used.

    The accuracy-vs-cycles curve of a speller: how well the decoder does when it is only
    allowed to see the first cycle, the first two, and so on. It is the usual way to read
    the speed/accuracy trade-off of a paradigm, and the curve early stopping is tuned on.

    Parameters
    ----------
    sel_cmd_per_cycle :
        ``{trial_idx: {cycle_idx: command_uid}}``. The cumulative selection after each
        cycle: the second output of :func:`select_commands`.
    target :
        ``{trial_idx: command_uid}``. The command the user was asked to spell in each
        trial. Trials that are not in ``target`` are ignored, so a recording may mix
        labelled and unlabelled trials.

    Returns
    -------
    numpy.ndarray
        ``(max_cycle_idx + 1,)``. Accuracy in ``[0, 1]`` indexed by the 0-based cycle
        index (the same ``cycle_idx`` as the events): entry ``n`` is the accuracy after
        cycle ``n``, that is after ``n + 1`` cycles. So entry ``0`` is the accuracy from a
        single cycle and the last entry is the final accuracy, the value
        :func:`command_decoding_accuracy` returns. Each entry is counted only over the
        trials that reached that many cycles, and is ``nan`` where no trial did. Trials
        may therefore have different lengths (for example when early stopping cut them
        short). An empty array is returned when no trial of ``per_cycle`` appears in
        ``target``.

    Examples
    --------
    >>> from medusa.pipelines.bci.vep_spellers.decoding import (
    ...     command_decoding_accuracy_per_cycle)
    >>> per_cycle = {0: {0: "B", 1: "A"},        # trial 0 is right from cycle 1 on
    ...              1: {0: "B", 1: "B"}}        # trial 1 is right from the start
    >>> command_decoding_accuracy_per_cycle(per_cycle, target={0: "A", 1: "B"}).tolist()
    [0.5, 1.0]
    """
    keys = [t for t in sel_cmd_per_cycle if t in target]
    if not keys:
        return np.array([])
    max_cyc = max(max(sel_cmd_per_cycle[t]) for t in keys)
    acc = np.full(max_cyc + 1, np.nan)
    for nc in range(max_cyc + 1):
        hits, total = 0, 0
        for t in keys:
            if nc in sel_cmd_per_cycle[t]:
                total += 1
                hits += str(sel_cmd_per_cycle[t][nc]) == str(target[t])
        if total:
            acc[nc] = hits / total
    return acc
