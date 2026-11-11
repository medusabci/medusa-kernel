"""Legacy c-VEP conversion: each command's ``screen_pos`` from the ``resize_info`` log.

The fixture file holds one layout of a 4 x 4 speller on a 1920 x 1080 screen: columns at
x = 326, 748, 1171, 1593 px and rows at y = 227, 458, 690, 921 px.
"""
import copy
import warnings
from pathlib import Path

import pytest

from medusa.core.legacy.convert import cvep_recording_to_v2
from medusa.core.legacy.recording import Recording as LegacyRecording

_FILE = Path(__file__).resolve().parent / "data" / "rec.cvep.bson"


@pytest.fixture(scope="module")
def _loaded():
    return LegacyRecording.load(str(_FILE))


@pytest.fixture
def legacy(_loaded):
    """A fresh copy of the fixture recording, so a test can edit its resize_info."""
    return copy.deepcopy(_loaded)


def _screen_pos(legacy):
    """``{content: screen_pos}`` of the converted recording."""
    sd = cvep_recording_to_v2(legacy).experiment
    return {cmd.content: cmd.screen_pos for cmd in sd.commands_info.values()}


def _shifted_layout(layout, dy):
    """A copy of ``layout`` with every command moved ``dy`` pixels down."""
    moved = copy.deepcopy(layout)
    for entries in moved["new_position"].values():
        for entry in entries:
            entry["new_pos"][1] += dy
    return moved


def test_screen_pos_is_the_pixel_centre_over_the_screen_size(legacy):
    pos = _screen_pos(legacy)
    assert len(pos) == 16
    assert pos["A"] == pytest.approx((326 / 1920, 227 / 1080))       # row 0, col 0
    assert pos["G"] == pytest.approx((1171 / 1920, 458 / 1080))      # row 1, col 2
    assert pos["P"] == pytest.approx((1593 / 1920, 921 / 1080))      # row 3, col 3


def test_the_layout_on_screen_at_the_first_onset_is_used(legacy):
    exp = legacy.cvepspellerdata
    first_onset = min(exp.onsets)
    layout = exp.resize_info[0]
    exp.resize_info = [_shifted_layout(layout, 8), layout]      # an early layout, then the real one
    exp.resize_onset = [first_onset - 10.0, first_onset - 5.0]

    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*layout changed")   # none mid-run
        pos = _screen_pos(legacy)

    assert pos["A"] == pytest.approx((326 / 1920, 227 / 1080))


def test_a_layout_change_during_the_run_warns(legacy):
    exp = legacy.cvepspellerdata
    first_onset = min(exp.onsets)
    layout = exp.resize_info[0]
    exp.resize_info = [layout, _shifted_layout(layout, 8)]      # the second one comes later
    exp.resize_onset = [first_onset - 5.0, first_onset + 5.0]

    with pytest.warns(UserWarning, match="layout changed during the run"):
        pos = _screen_pos(legacy)

    assert pos["A"] == pytest.approx((326 / 1920, 227 / 1080))  # the layout at the start


def test_files_without_resize_info_keep_screen_pos_unset(legacy):
    legacy.cvepspellerdata.resize_info = None
    assert set(_screen_pos(legacy).values()) == {None}


def test_positions_outside_the_screen_are_dropped_with_a_warning(legacy):
    legacy.cvepspellerdata.resize_info[0]["screen_size"] = [1000, 1080]   # x up to 1593 px

    with pytest.warns(UserWarning, match="outside the 1000x1080 screen"):
        pos = _screen_pos(legacy)

    assert set(pos.values()) == {None}
