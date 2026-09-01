"""Layer-2 command selection for VEP spellers (paradigm-agnostic).

A Layer-1 pipeline's :meth:`predict` returns a *cumulative* ``(n_cycles, n_commands)`` score
matrix -- row ``i`` is the decision score for every command after cycle event ``i``,
accumulated the family's natural way (concatenate-then-correlate for BWR, coherent averaging
+ CCA for template matching). This module only **selects**: for each trial, after each cycle,
it takes the argmax of the cumulative score row over that trial's *available* commands (with
optional early stopping). It is paradigm-agnostic -- no codes, no per-family logic -- because
Layer 1 already did the family-specific accumulation.

:func:`select_commands` is the free-function core; :class:`VEPCommandDecoder` wraps it with a
configurable early-stopping threshold and adds an online :meth:`~VEPCommandDecoder.step`.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from medusa.core.settings_tree import Configurable, SettingsTree
from medusa.pipelines.bci.vep_spellers.data import SpellerData
from medusa.pipelines.bci.vep_spellers.decoding._common import (
    _cycle_arrays, _trial_cycle_order)

__all__ = [
    "VEPCommandDecoder",
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
                    cycle_trial: NDArray, cycle_idx: NDArray,
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
    cycle_trial, cycle_idx :
        ``(n_cycles,)`` each. Per-cycle trial index and repetition index (one row per
        cycle, from the events).
    trial_available_cmmds :
        Per-trial list of available command ``uid``\\ s
        (``SpellerData.trial_available_cmmds``). ``None`` makes every command available.

    Returns
    -------
    selected : dict
        ``{trial_idx: command_uid}``. The final selection (all cycles).
    per_cycle : dict
        ``{trial_idx: {n_cycles: command_uid}}``. The cumulative selection after each
        cycle (for dynamic stopping or accuracy-vs-cycles).
    scores : dict
        ``{trial_idx: {n_cycles: ndarray[n_available]}}``. Per-available-command score.

    Examples
    --------
    >>> import numpy as np
    >>> from medusa.pipelines.bci.vep_spellers.decoding import select_commands
    >>> cycle_scores = np.array([[0.1, 0.9], [0.8, 0.2]])    # 2 cycles, 2 commands
    >>> selected, per_cycle, scores = select_commands(
    ...     cycle_scores, command_uids=["A", "B"],
    ...     cycle_trial=np.array([0, 0]), cycle_idx=np.array([0, 1]))
    >>> selected[0]                     # the last cycle (idx 1) picks command "A"
    'A'
    """
    cycle_scores = np.asarray(cycle_scores, dtype=float)
    command_uids = list(command_uids)
    cycle_idx = np.asarray(cycle_idx, dtype=int)
    row = {uid: i for i, uid in enumerate(command_uids)}

    selected, per_cycle, scores = {}, {}, {}
    for t, order in _trial_cycle_order(cycle_trial, cycle_idx):
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


def command_decoding_accuracy(selected: dict, target: dict) -> float:
    """Fraction of trials whose final selected command equals the target.

    ``selected`` and ``target`` are both ``{trial_idx: command_uid}``. Trials that are not
    in ``target`` are ignored.
    """
    keys = [t for t in selected if t in target]
    if not keys:
        return float("nan")
    correct = sum(str(selected[t]) == str(target[t]) for t in keys)
    return correct / len(keys)


def command_decoding_accuracy_per_cycle(per_cycle: dict, target: dict) -> NDArray:
    """Accuracy as a function of the number of cycles used.

    Returns an array indexed by the 0-based cycle index (the same ``cycle_idx`` as the
    events). Entry ``n`` is the accuracy after cycle ``n``, i.e. after ``n + 1`` cycles,
    counted over the trials that reached that many cycles. So entry ``0`` is the accuracy
    after the first cycle and the last entry is the final accuracy.
    """
    keys = [t for t in per_cycle if t in target]
    if not keys:
        return np.array([])
    max_cyc = max(max(per_cycle[t]) for t in keys)
    acc = np.full(max_cyc + 1, np.nan)
    for nc in range(max_cyc + 1):
        hits, total = 0, 0
        for t in keys:
            if nc in per_cycle[t]:
                total += 1
                hits += str(per_cycle[t][nc]) == str(target[t])
        if total:
            acc[nc] = hits / total
    return acc


class VEPCommandDecoder(Configurable):
    """Layer-2 command decoder: cumulative per-cycle command scores -> selected commands.

    Paradigm-agnostic and rule-based (no fitting). A Layer-1 pipeline's :meth:`predict`
    returns a cumulative ``(n_cycles, n_commands)`` score matrix, accumulated the family's
    natural way (concatenate-then-correlate for BWR, coherent averaging + CCA for template
    matching). This decoder only **selects**. Offline, :meth:`decode` returns the per-trial
    selection, the per-cycle trajectory, and the scores. Online, :meth:`step` takes the
    current cumulative score row and returns the current best command (and whether the
    early-stopping threshold on the top score fired). Configure it through its
    :attr:`~medusa.core.settings_tree.Configurable.settings` (``stop_corr``).
    """

    @classmethod
    def default_settings(cls) -> SettingsTree:
        """The configuration schema: a single ``stop_corr`` early-stopping threshold."""
        s = SettingsTree()
        s.add_item("stop_corr", value=0.9, optional=True, enabled=False,
                   value_range=[-1.0, 1.0],
                   info="Early-stopping score threshold; switch it off to never stop "
                        "early")
        return s

    # ---- offline ----
    def decode(self, cycle_scores: NDArray, speller_data: SpellerData, events) -> dict:
        """Decode every trial in a recording's events; return the selections and scores.

        ``cycle_scores`` is a Layer-1 pipeline's cumulative ``(n_cycles, n_commands)`` output
        for this recording (its rows line up with the cycle events).
        """
        _, trial, cycle, _ = _cycle_arrays(events)
        selected, per_cycle, scores = select_commands(
            cycle_scores, speller_data.command_uids, trial, cycle,
            speller_data.trial_available_cmmds)
        return {"selected_commands": selected,
                "selected_commands_per_cycle": per_cycle,
                "scores": scores}

    # ---- online ----
    def step(self, cycle_scores_row: NDArray, speller_data: SpellerData,
             trial: int = 0) -> dict:
        """Select the current best command from one cumulative score row.

        ``cycle_scores_row`` has one cumulative score per command (length ``n_commands``).
        It is a single row of a Layer-1 pipeline's cumulative output (the Layer-1 pipeline
        does the cross-cycle accumulation). Returns
        ``{'selected': uid, 'scores': ndarray[n_available], 'stop': bool}``.
        """
        avail = _available(speller_data.trial_available_cmmds,
                           speller_data.command_uids, int(trial))
        row = {uid: i for i, uid in enumerate(speller_data.command_uids)}
        avail_cols = [row[u] for u in avail]
        corr = np.asarray(cycle_scores_row, dtype=float)[avail_cols]
        best = int(np.argmax(corr))
        threshold = self.cfg["stop_corr"]
        stop = (threshold is not None and np.isfinite(corr[best])
                and corr[best] >= threshold)
        return {"selected": avail[best], "scores": corr, "stop": bool(stop)}
