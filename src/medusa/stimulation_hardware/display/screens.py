r"""Screens: list them, list their display modes, and set their refresh rate.

Windows only for now: on other systems every function raises ``NotImplementedError``.
Each operating system has its own private backend module, imported when a function is
called, so importing this module works everywhere.

The refresh rate is the nominal one the operating system reports. Windows reports whole
Hz, so a 59.94 Hz mode shows as 59 Hz. A remote session (for example Remote Desktop)
shows its virtual display, not the real monitors.
"""

import sys
from dataclasses import dataclass
from types import ModuleType

__all__ = [
    "DisplayMode",
    "Screen",
    "list_screens",
    "list_display_modes",
    "set_fps",
]


@dataclass(frozen=True, order=True)
class DisplayMode:
    """Resolution and refresh rate of a screen.

    Modes sort by width, then height, then refresh rate.

    Parameters
    ----------
    width, height :
        Resolution, in pixels.
    fps :
        Refresh rate (Hz).

    Examples
    --------
    >>> from medusa.stimulation_hardware.display import DisplayMode
    >>> DisplayMode(1920, 1080, 120.0)
    DisplayMode(width=1920, height=1080, fps=120.0)
    """

    width: int
    height: int
    fps: float


@dataclass(frozen=True)
class Screen:
    r"""One screen of the desktop, as the operating system reports it.

    Parameters
    ----------
    index :
        Position in :func:`list_screens`. The other functions take it as ``screen``.
    name :
        Device name, for example ``\\.\DISPLAY1`` on Windows.
    description :
        Display adapter, for example ``NVIDIA GeForce RTX 4090``. A remote or virtual
        display shows here, for example ``Microsoft Remote Display Adapter``.
    is_primary :
        Whether this is the primary screen.
    x, y :
        Top-left corner on the desktop, in pixels. The primary screen is at ``(0, 0)``.
    mode :
        The current display mode.

    Examples
    --------
    >>> from medusa.stimulation_hardware.display import DisplayMode, Screen
    >>> screen = Screen(index=0, name=r"\\.\DISPLAY1", description="GPU",
    ...                 is_primary=True, x=0, y=0, mode=DisplayMode(1920, 1080, 120.0))
    >>> screen.mode.fps
    120.0
    """

    index: int
    name: str
    description: str
    is_primary: bool
    x: int
    y: int
    mode: DisplayMode


def list_screens() -> list[Screen]:
    r"""List the screens of the desktop.

    Returns
    -------
    screens :
        One :class:`Screen` per screen, in the order the operating system reports them.
        ``len(screens)`` is the number of screens.

    Raises
    ------
    NotImplementedError
        If the operating system is not supported (only Windows for now).

    Examples
    --------
    >>> from medusa.stimulation_hardware.display import list_screens
    >>> for screen in list_screens():                                    # doctest: +SKIP
    ...     print(screen.index, screen.name, screen.mode.fps)
    0 \\.\DISPLAY1 120.0
    1 \\.\DISPLAY2 60.0
    """
    return _backend().list_screens()


def list_display_modes(screen: int) -> list[DisplayMode]:
    """List the display modes a screen supports.

    Parameters
    ----------
    screen :
        Index of the screen in :func:`list_screens`.

    Returns
    -------
    modes :
        Every supported :class:`DisplayMode`, sorted by width, height and refresh rate.

    Raises
    ------
    ValueError
        If there is no screen with that index.
    NotImplementedError
        If the operating system is not supported (only Windows for now).

    Examples
    --------
    >>> from medusa.stimulation_hardware.display import list_display_modes
    >>> modes = list_display_modes(0)                                    # doctest: +SKIP
    >>> [m.fps for m in modes if (m.width, m.height) == (1920, 1080)]    # doctest: +SKIP
    [60.0, 120.0, 144.0]
    """
    device = _get_screen(screen).name
    return _backend().list_display_modes(device)


def set_fps(screen: int, fps: float, persistent: bool = False) -> DisplayMode:
    """Set the refresh rate of a screen, keeping its resolution.

    The operating system checks the new mode before applying it, so nothing changes if
    it refuses.

    Parameters
    ----------
    screen :
        Index of the screen in :func:`list_screens`.
    fps :
        New refresh rate (Hz). It must be one of the rates :func:`list_display_modes`
        lists for the current resolution.
    persistent :
        Save the change in the user's display settings. If ``False`` (default), the
        change is not saved and Windows goes back to the saved mode at the next sign-in.

    Returns
    -------
    previous :
        The mode before the change. Pass ``previous.fps`` back to undo it.

    Raises
    ------
    ValueError
        If there is no screen with that index, or if it cannot run at ``fps`` at its
        current resolution.
    RuntimeError
        If the operating system refuses the change.
    NotImplementedError
        If the operating system is not supported (only Windows for now).

    Examples
    --------
    >>> from medusa.stimulation_hardware.display import set_fps
    >>> previous = set_fps(0, 120)                                       # doctest: +SKIP
    >>> set_fps(0, previous.fps)          # undo                         # doctest: +SKIP
    """
    target = _get_screen(screen)
    current = target.mode
    backend = _backend()

    # Rates available at the current resolution
    rates = []
    for mode in backend.list_display_modes(target.name):
        if (mode.width, mode.height) == (current.width, current.height):
            rates.append(mode.fps)
    if fps not in rates:
        raise ValueError(
            f"screen {screen} cannot run at {fps} Hz at {current.width}x{current.height}; "
            f"available rates: {sorted(set(rates))}.")

    new_mode = DisplayMode(current.width, current.height, float(fps))
    backend.set_mode(target.name, new_mode, persistent)
    return current


def _get_screen(screen: int) -> Screen:
    """The :class:`Screen` with this index; ``ValueError`` if there is none."""
    screens = _backend().list_screens()
    if not 0 <= screen < len(screens):
        raise ValueError(f"there is no screen {screen}; found {len(screens)} screen(s).")
    return screens[screen]


def _backend() -> ModuleType:
    """The backend module of this operating system (imported on first use)."""
    if sys.platform == "win32":
        from medusa.stimulation_hardware.display import _screens_windows
        return _screens_windows
    raise NotImplementedError(
        f"screen configuration is only supported on Windows for now, not on "
        f"{sys.platform!r}.")
