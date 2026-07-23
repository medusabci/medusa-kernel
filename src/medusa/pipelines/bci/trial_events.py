"""Shared event contract for trial-based decoding.

Some BCI paradigms decode independent, labelled trials: motor imagery, motor
execution, mental tasks, and so on. Each trial is one slice of signal around a cue.
The trials do not build on each other, so the only run information that is time-locked
lives in the recording's :class:`~medusa.core.data.events.Events` table, not in a data
class.

This module is that shared contract. It holds no pipeline and no workflow. It only says
how a trial-decoding recording stores its trials and how to read them back, so every
domain package (``motor_decoding``, a future ``mental_state_decoding``, and so on) reads
and writes the *same* columns. That keeps recordings interoperable across domains.

There is **one events row per trial**. Its ``onset`` marks the trial start. Two nullable
``Int64`` columns describe it: ``trial_idx`` (the trial number) and ``label`` (its class).
``label`` may be null in a test recording whose classes are unknown. This mirrors how
:mod:`~medusa.pipelines.bci.vep_spellers.data` defines ``SPELLER_EVENT_COLUMNS`` and
``validate_speller_events`` for spellers.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from medusa.core.data.events import Events
from medusa.core.data.recording import Recording

__all__ = [
    "TRIAL_EVENT_COLUMNS",
    "validate_trial_events",
    "trial_arrays",
    "trial_labels",
]

#: Required ``Events`` columns for trial decoding, as pandas nullable ``Int64``. There is
#: one row per trial; its ``onset`` marks the trial start. ``trial_idx`` is the trial
#: number and ``label`` is its class (may be null when the class is unknown, for example a
#: test run).
TRIAL_EVENT_COLUMNS = {
    "trial_idx": "Int64",
    "label": "Int64",
}


def validate_trial_events(events: Events) -> None:
    """Check that ``events`` is a usable trial-decoding timeline.

    A trial row is a row with a non-null ``trial_idx``. Any other row (in a mixed
    timeline) is ignored by the readers below.

    Raises
    ------
    ValueError
        If ``events`` is ``None``, a column in :data:`TRIAL_EVENT_COLUMNS` is missing, or
        there is no trial (every ``trial_idx`` is null).
    """
    if events is None:
        raise ValueError("recording has no events timeline.")
    cols = set(events.column_names)
    missing = [c for c in TRIAL_EVENT_COLUMNS if c not in cols]
    if missing:
        raise ValueError(
            f"events lack mandatory trial columns {missing}; required: "
            f"{list(TRIAL_EVENT_COLUMNS)}.")
    if not events.df["trial_idx"].notna().any():
        raise ValueError("events contain no trials (all trial_idx are n/a).")


def trial_arrays(events: Events) -> "tuple[NDArray, NDArray, NDArray]":
    """Return ``(onsets, trial_idx, labels)`` for the trial rows, in row order.

    A trial row is a row with a non-null ``trial_idx``; any other row is skipped.
    ``labels`` is a float array so an unknown class shows up as ``nan`` (a nullable
    ``Int64`` cannot store ``nan`` as an integer). :func:`trial_labels` is the integer,
    fully-labelled version used for training.
    """
    df = events.df
    df = df[df["trial_idx"].notna()]
    onsets = df["onset"].to_numpy(dtype=float)
    trial_idx = df["trial_idx"].to_numpy(dtype=int)
    labels = df["label"].to_numpy(dtype=float)      # nan where the class is unknown
    return onsets, trial_idx, labels


def trial_labels(recording: Recording) -> NDArray:
    """Return the integer class label of every trial in ``recording`` (the training ``y``).

    This is the trial-decoding counterpart of
    :func:`~medusa.pipelines.bci.vep_spellers.decoding.bwr_labels`.

    Raises
    ------
    ValueError
        If any trial has no label; a training recording must label every trial.
    """
    _, _, labels = trial_arrays(recording.events)
    if np.isnan(labels).any():
        raise ValueError(
            "some trials have no label; a training recording must label every trial.")
    return labels.astype(int)
