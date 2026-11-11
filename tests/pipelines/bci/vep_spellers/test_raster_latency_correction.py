"""Raster-latency correction for spellers: command latencies and the fit-side shift.

The latency model itself is tested in ``tests/stimulation_hardware/display``. The predict
side (each command scored at its own latency) lives in the pipelines, so it is tested
with them, in ``test_bwr_eeg_inception_raster.py``.
"""
import numpy as np
import pandas as pd
import pytest

from medusa.core.data import Events
from medusa.core.settings_tree import SettingsTree
from medusa.pipelines.bci.vep_spellers import (
    CommandInfo,
    SpellerData,
    command_raster_latencies,
    raster_corrected_events,
)
from medusa.pipelines.bci.vep_spellers.decoding.raster_latency_correction import (
    add_raster_latency_settings,
)
from medusa.stimulation_hardware.display import DEFAULT_SCAN_FRACTION

FPS = 60.0
SCAN = DEFAULT_SCAN_FRACTION / FPS      # seconds from the top line to the bottom one

#: A 2 x 2 grid: A and B on the top row, C and D on the bottom one.
GRID = {"A": (0.25, 0.25), "B": (0.75, 0.25), "C": (0.25, 0.75), "D": (0.75, 0.75)}


def _speller(spell_target=None):
    """The four GRID commands, with these trial targets."""
    cmds = {uid: CommandInfo(uid, [1, 0, 1], screen_pos=pos) for uid, pos in GRID.items()}
    return SpellerData(mode="train", paradigm_conf={}, commands_info=cmds,
                       fps_resolution=FPS, spell_target=spell_target)


def _events():
    """Two trials of two cycles each, with one non-cycle row (a cue) between them."""
    events = Events(optional_columns={"trial_idx": "Int64", "cycle_idx": "Int64",
                                      "code_idx": "Int64"})
    cue = {"trial_idx": None, "cycle_idx": None, "code_idx": None}
    return events.append([
        {"onset": 1.0, "duration": 0.0, "trial_idx": 0, "cycle_idx": 0, "code_idx": 0},
        {"onset": 1.5, "duration": 0.0, "trial_idx": 0, "cycle_idx": 1, "code_idx": 0},
        {"onset": 2.5, "duration": 0.0, **cue},
        {"onset": 3.0, "duration": 0.0, "trial_idx": 1, "cycle_idx": 0, "code_idx": 0},
        {"onset": 3.5, "duration": 0.0, "trial_idx": 1, "cycle_idx": 1, "code_idx": 0},
    ])


# --------------------------------------------------------------------------- #
# command_raster_latencies
# --------------------------------------------------------------------------- #
def test_command_latencies_follow_the_codebook_order():
    expected = [0.25 * SCAN, 0.25 * SCAN, 0.75 * SCAN, 0.75 * SCAN]   # A, B, C, D
    np.testing.assert_allclose(command_raster_latencies(_speller()), expected)


def test_command_latencies_need_every_screen_pos():
    sd = _speller()
    sd.commands_info["C"].screen_pos = None
    with pytest.raises(ValueError, match="'C'"):
        command_raster_latencies(sd)


# --------------------------------------------------------------------------- #
# raster_corrected_events
# --------------------------------------------------------------------------- #
def test_each_cycle_moves_by_its_targets_latency():
    corrected = raster_corrected_events(_events(), _speller(spell_target=["A", "D"]))
    expected = [1.0 + 0.25 * SCAN, 1.5 + 0.25 * SCAN,     # trial 0, target A (top row)
                2.5,                                       # the cue does not move
                3.0 + 0.75 * SCAN, 3.5 + 0.75 * SCAN]      # trial 1, target D (bottom row)
    np.testing.assert_allclose(corrected.df["onset"], expected)


def test_the_input_events_are_not_modified():
    events = _events()
    before = events.df.copy()
    raster_corrected_events(events, _speller(spell_target=["A", "D"]))
    pd.testing.assert_frame_equal(events.df, before)


def test_rows_and_their_order_are_kept():
    events = _events()
    corrected = raster_corrected_events(events, _speller(spell_target=["A", "D"]))
    other_columns = [c for c in events.df.columns if c != "onset"]
    pd.testing.assert_frame_equal(corrected.df[other_columns], events.df[other_columns])


def test_events_correction_needs_the_targets():
    with pytest.raises(ValueError, match="spell_target"):
        raster_corrected_events(_events(), _speller(spell_target=None))


def test_events_correction_needs_a_target_for_every_trial():
    with pytest.raises(ValueError, match="trial 1 has no target"):
        raster_corrected_events(_events(), _speller(spell_target=["A"]))


# --------------------------------------------------------------------------- #
# add_raster_latency_settings
# --------------------------------------------------------------------------- #
def test_the_setting_is_off_by_default_and_reads_its_value_when_on():
    s = SettingsTree()
    add_raster_latency_settings(s)
    assert s.to_dict()["raster_scan_fraction"] is None
    s.update_from_dict({"raster_scan_fraction": 0.9})
    assert s.to_dict()["raster_scan_fraction"] == 0.9
