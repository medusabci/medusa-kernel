"""Screens: listing screens and modes, and setting the refresh rate.

Most tests replace the operating-system backend with a fake one, so they run anywhere
and never touch a real screen. The last tests read the real Windows API; they only read,
or ask Windows to check a mode without applying it.
"""
import sys

import pytest

from medusa.stimulation_hardware.display import (
    DisplayMode,
    Screen,
    list_display_modes,
    list_screens,
    screens,
    set_fps,
)


class FakeBackend:
    """Two 1920x1080 screens, at 60 Hz and at 120 Hz. Records every mode it is set to."""

    def __init__(self):
        self.screens = [
            Screen(0, r"\\.\DISPLAY1", "Fake GPU", True, 0, 0, DisplayMode(1920, 1080, 60.0)),
            Screen(1, r"\\.\DISPLAY2", "Fake GPU", False, 1920, 0,
                   DisplayMode(1920, 1080, 120.0)),
        ]
        self.modes = [DisplayMode(1280, 720, 60.0), DisplayMode(1280, 720, 75.0),
                      DisplayMode(1920, 1080, 60.0), DisplayMode(1920, 1080, 120.0),
                      DisplayMode(1920, 1080, 144.0)]
        self.applied = []

    def list_screens(self):
        return self.screens

    def list_display_modes(self, device):
        return self.modes

    def set_mode(self, device, mode, persistent):
        self.applied.append((device, mode, persistent))


@pytest.fixture
def backend(monkeypatch):
    fake = FakeBackend()
    monkeypatch.setattr(screens, "_backend", lambda: fake)
    return fake


def test_list_screens_returns_the_screens_the_backend_reports(backend):
    found = list_screens()
    assert len(found) == 2
    assert [s.mode.fps for s in found] == [60.0, 120.0]


def test_list_display_modes_of_a_screen(backend):
    assert list_display_modes(1) == backend.modes


def test_set_fps_keeps_the_resolution_and_returns_the_previous_mode(backend):
    previous = set_fps(1, 144)

    assert previous == DisplayMode(1920, 1080, 120.0)
    assert backend.applied == [(r"\\.\DISPLAY2", DisplayMode(1920, 1080, 144.0), False)]


def test_set_fps_can_save_the_change(backend):
    set_fps(0, 120, persistent=True)
    assert backend.applied[0][2] is True


def test_set_fps_rejects_a_rate_not_available_at_the_current_resolution(backend):
    # 75 Hz exists, but only at 1280x720
    with pytest.raises(ValueError, match=r"available rates: \[60.0, 120.0, 144.0\]"):
        set_fps(0, 75)
    assert backend.applied == []


def test_a_missing_screen_is_rejected(backend):
    with pytest.raises(ValueError, match="there is no screen 5"):
        list_display_modes(5)


def test_other_systems_are_not_supported(monkeypatch):
    monkeypatch.setattr(screens.sys, "platform", "linux")
    with pytest.raises(NotImplementedError, match="only supported on Windows"):
        list_screens()


# --------------------------------------------------------------------------- #
# Real Windows API (read-only)
# --------------------------------------------------------------------------- #
windows_only = pytest.mark.skipif(sys.platform != "win32", reason="Windows API")


@pytest.fixture
def real_screens():
    found = list_screens()
    if not found:
        pytest.skip("no screen attached to the desktop")
    return found


@windows_only
def test_windows_reports_one_primary_screen_with_a_valid_mode(real_screens):
    assert sum(s.is_primary for s in real_screens) == 1
    for screen in real_screens:
        assert screen.mode.width > 0 and screen.mode.height > 0 and screen.mode.fps > 0


@windows_only
def test_windows_lists_the_current_mode_among_the_display_modes(real_screens):
    assert real_screens[0].mode in list_display_modes(0)


@windows_only
def test_windows_accepts_the_current_mode_without_changing_it(real_screens):
    from medusa.stimulation_hardware.display import _screens_windows

    screen = real_screens[0]
    _screens_windows.check_mode(screen.name, screen.mode)     # check only, no change
    assert list_screens()[0].mode == screen.mode
