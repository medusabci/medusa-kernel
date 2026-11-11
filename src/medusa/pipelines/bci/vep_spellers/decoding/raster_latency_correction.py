"""Monitor raster-latency correction for VEP spellers.

A monitor draws each frame from the top of the screen to the bottom, so a command lower
on the screen is drawn later than its event onset says, by up to about 95% of a frame
(:func:`~medusa.stimulation_hardware.display.raster_latency`). Its EEG response starts
that much later too. Decoders that depend on timing (bit-wise reconstruction, template
matching) lose accuracy if they ignore it, because the commands of a speller sit on
different screen rows.

The latency of a command is known from its position on the screen
(:attr:`~medusa.pipelines.bci.vep_spellers.data.CommandInfo.screen_pos`), so it is
corrected, not estimated. The correction has two sides:

* **Fit** (target known). :func:`raster_corrected_events` shifts each cycle onset by the
  latency of its trial's target. The model then learns from segments aligned with the
  moment the target was drawn.
* **Predict** (target unknown). Each command is scored with the onsets shifted by its own
  latency, as if it were the target, and the best score wins. Commands with the same
  latency (the same screen row) share one scoring pass.

A pipeline switches both sides on with the ``raster_scan_fraction`` setting, added by
:func:`add_raster_latency_settings`.
"""

import copy

from numpy.typing import NDArray

from medusa.core.data.events import Events
from medusa.core.settings_tree import SettingsTree
from medusa.pipelines.bci.vep_spellers.data import SpellerData
from medusa.stimulation_hardware.display import (
    DEFAULT_SCAN_FRACTION,
    raster_latency,
)

__all__ = [
    "command_raster_latencies",
    "raster_corrected_events",
    "add_raster_latency_settings",
]


def command_raster_latencies(speller_data: SpellerData,
                             scan_fraction: float = DEFAULT_SCAN_FRACTION) -> NDArray:
    """Raster latency of every command of a speller, from its screen position.

    Parameters
    ----------
    speller_data :
        The run's speller data. Every command needs a ``screen_pos``, and
        ``fps_resolution`` gives the refresh rate.
    scan_fraction :
        Fraction of the frame period the scan takes from the top line to the bottom one
        (see :func:`~medusa.stimulation_hardware.display.raster_latency`).

    Returns
    -------
    latencies :
        ``(n_commands,)`` latency of each command, in seconds, in the order of
        :attr:`~medusa.pipelines.bci.vep_spellers.data.SpellerData.command_uids`. This is
        also the column order of every score matrix.

    Raises
    ------
    ValueError
        If a command has no ``screen_pos``.

    Examples
    --------
    >>> from medusa.pipelines.bci.vep_spellers import (
    ...     CommandInfo, SpellerData, command_raster_latencies)
    >>> cmds = {"A": CommandInfo("A", [1, 0], screen_pos=(0.25, 0.2)),
    ...         "B": CommandInfo("B", [0, 1], screen_pos=(0.75, 0.8))}
    >>> sd = SpellerData(mode="test", paradigm_conf={}, commands_info=cmds,
    ...                  fps_resolution=60)
    >>> (1e3 * command_raster_latencies(sd)).round(2)
    array([ 3.17, 12.67])
    """
    missing = [uid for uid, cmd in speller_data.commands_info.items()
               if cmd.screen_pos is None]
    if missing:
        raise ValueError(
            f"raster correction needs the screen position of every command, but these "
            f"commands have no screen_pos: {missing}.")
    positions = [cmd.screen_pos for cmd in speller_data.commands_info.values()]
    return raster_latency(positions, speller_data.fps_resolution, scan_fraction)


def raster_corrected_events(events: Events, speller_data: SpellerData,
                            scan_fraction: float = DEFAULT_SCAN_FRACTION) -> Events:
    """Copy of ``events`` with each cycle onset shifted by its target's raster latency.

    The target's stimulus is drawn ``latency`` seconds after the event onset, so its EEG
    response starts that much later. Shifting the onsets aligns the segments with the
    moment the target was drawn. Use it on calibration recordings, where the targets are
    known. At test time the target is unknown, so the decoder scores each command at its
    own latency instead (see :func:`command_raster_latencies`).

    The input is never modified. The copy has the same rows in the same order, so
    :func:`~medusa.pipelines.bci.vep_spellers.data.cycle_arrays` lines up with the
    original. Only the onsets of the cycle rows (non-null ``cycle_idx``) change.

    Parameters
    ----------
    events :
        The run's speller events (one row per stimulation cycle, see
        :data:`~medusa.pipelines.bci.vep_spellers.data.SPELLER_EVENT_COLUMNS`).
    speller_data :
        The run's speller data. It needs ``spell_target`` (one target per trial) and a
        ``screen_pos`` for every command.
    scan_fraction :
        Fraction of the frame period the scan takes from the top line to the bottom one
        (see :func:`~medusa.stimulation_hardware.display.raster_latency`).

    Returns
    -------
    corrected :
        A new :class:`~medusa.core.data.events.Events` with the shifted cycle onsets.

    Raises
    ------
    ValueError
        If ``speller_data`` has no ``spell_target``, if a trial has no target, or if a
        command has no ``screen_pos``.

    Examples
    --------
    >>> from medusa.core.data import Events
    >>> from medusa.pipelines.bci.vep_spellers import (
    ...     CommandInfo, SpellerData, raster_corrected_events)
    >>> cmds = {"A": CommandInfo("A", [1, 0], screen_pos=(0.25, 0.2)),
    ...         "B": CommandInfo("B", [0, 1], screen_pos=(0.75, 0.8))}
    >>> sd = SpellerData(mode="train", paradigm_conf={}, commands_info=cmds,
    ...                  fps_resolution=60, spell_target=["A", "B"])
    >>> events = Events(optional_columns={"trial_idx": "Int64", "cycle_idx": "Int64",
    ...                                   "code_idx": "Int64"})
    >>> events = events.append([
    ...     {"onset": 1.0, "duration": 0.0, "trial_idx": 0, "cycle_idx": 0, "code_idx": 0},
    ...     {"onset": 2.0, "duration": 0.0, "trial_idx": 1, "cycle_idx": 0, "code_idx": 0}])
    >>> corrected = raster_corrected_events(events, sd)
    >>> corrected.df["onset"].round(5).tolist()     # +3.17 ms (A) and +12.67 ms (B)
    [1.00317, 2.01267]
    >>> events.df["onset"].tolist()                  # the input is unchanged
    [1.0, 2.0]
    """
    if speller_data.spell_target is None:
        raise ValueError(
            "raster_corrected_events needs the trial targets (speller_data.spell_target), "
            "and this run has none. When the target is unknown, score each command at its "
            "own latency instead (command_raster_latencies).")
    latency_of = dict(zip(speller_data.command_uids,
                          command_raster_latencies(speller_data, scan_fraction)))

    corrected = copy.deepcopy(events)           # the input is never modified
    df = corrected.df
    is_cycle = df["cycle_idx"].notna()
    trials = df.loc[is_cycle, "trial_idx"].to_numpy(dtype=int)
    n_targets = len(speller_data.spell_target)
    if trials.size and trials.max() >= n_targets:
        raise ValueError(
            f"trial {trials.max()} has no target: spell_target has only {n_targets} "
            f"entries.")
    targets = [str(speller_data.spell_target[t]) for t in trials]
    df.loc[is_cycle, "onset"] += [latency_of[uid] for uid in targets]
    return corrected


def add_raster_latency_settings(settings: SettingsTree) -> None:
    """Add the ``raster_scan_fraction`` setting that switches the raster correction on.

    It is an optional item, off by default. When on, its value is the ``scan_fraction``
    the pipeline passes to :func:`raster_corrected_events` in ``fit`` and to
    :func:`command_raster_latencies` in ``predict``. When off, it reads as ``None`` and the
    pipeline makes no correction. Saved pipelines without it load with it off.

    Parameters
    ----------
    settings :
        The pipeline's settings tree (or one of its groups). The item is added in place.

    Examples
    --------
    >>> from medusa.core.settings_tree import SettingsTree
    >>> from medusa.pipelines.bci.vep_spellers.decoding.raster_latency_correction import (
    ...     add_raster_latency_settings)
    >>> s = SettingsTree()
    >>> add_raster_latency_settings(s)
    >>> s.to_dict()
    {'raster_scan_fraction': None}
    """
    settings.add_item(
        "raster_scan_fraction", value=DEFAULT_SCAN_FRACTION, optional=True, enabled=False,
        value_range=[0.0, 1.0],
        info="Correct the monitor raster latency of each command, from its screen_pos. "
             "The value is the fraction of the frame period the monitor takes to draw "
             "from the top line to the bottom one (0.95 for most monitors). Switch it off "
             "to skip the correction")
