"""Windows backend of :mod:`medusa.stimulation_hardware.display.screens` (Win32 API).

Imported only on Windows, when a function of ``screens`` is called. It uses three
``user32`` functions through ``ctypes``: ``EnumDisplayDevicesW`` (the screens),
``EnumDisplaySettingsW`` (their modes) and ``ChangeDisplaySettingsExW`` (set a mode).
"""

import ctypes
from ctypes import wintypes

from medusa.stimulation_hardware.display.screens import DisplayMode, Screen

_user32 = ctypes.WinDLL("user32", use_last_error=True)

# Win32 constants
_ENUM_CURRENT_SETTINGS = 0xFFFFFFFF            # (DWORD)-1: the current mode
_DISPLAY_DEVICE_ATTACHED_TO_DESKTOP = 0x1
_DISPLAY_DEVICE_PRIMARY_DEVICE = 0x4
_DM_PELSWIDTH = 0x00080000
_DM_PELSHEIGHT = 0x00100000
_DM_DISPLAYFREQUENCY = 0x00400000
_CDS_UPDATEREGISTRY = 0x1                      # save the change in the user's settings
_CDS_TEST = 0x2                                # only check the mode, change nothing
_DISP_CHANGE_SUCCESSFUL = 0
_DISP_CHANGE_ERRORS = {
    1: "the computer must be restarted to use this mode",
    -1: "the display driver failed the mode",
    -2: "the mode is not supported",
    -3: "the settings could not be saved",
    -4: "invalid flags",
    -5: "invalid parameter",
    -6: "the system is DualView capable",
}


class _DisplayDevice(ctypes.Structure):
    """Win32 ``DISPLAY_DEVICEW``: one display output."""

    _fields_ = [
        ("cb", wintypes.DWORD),
        ("DeviceName", wintypes.WCHAR * 32),
        ("DeviceString", wintypes.WCHAR * 128),
        ("StateFlags", wintypes.DWORD),
        ("DeviceID", wintypes.WCHAR * 128),
        ("DeviceKey", wintypes.WCHAR * 128),
    ]


class _DevMode(ctypes.Structure):
    """Win32 ``DEVMODEW``, with the display variant of its position union."""

    _fields_ = [
        ("dmDeviceName", wintypes.WCHAR * 32),
        ("dmSpecVersion", wintypes.WORD),
        ("dmDriverVersion", wintypes.WORD),
        ("dmSize", wintypes.WORD),
        ("dmDriverExtra", wintypes.WORD),
        ("dmFields", wintypes.DWORD),
        ("dmPositionX", wintypes.LONG),
        ("dmPositionY", wintypes.LONG),
        ("dmDisplayOrientation", wintypes.DWORD),
        ("dmDisplayFixedOutput", wintypes.DWORD),
        ("dmColor", ctypes.c_short),
        ("dmDuplex", ctypes.c_short),
        ("dmYResolution", ctypes.c_short),
        ("dmTTOption", ctypes.c_short),
        ("dmCollate", ctypes.c_short),
        ("dmFormName", wintypes.WCHAR * 32),
        ("dmLogPixels", wintypes.WORD),
        ("dmBitsPerPel", wintypes.DWORD),
        ("dmPelsWidth", wintypes.DWORD),
        ("dmPelsHeight", wintypes.DWORD),
        ("dmDisplayFlags", wintypes.DWORD),
        ("dmDisplayFrequency", wintypes.DWORD),
        ("dmICMMethod", wintypes.DWORD),
        ("dmICMIntent", wintypes.DWORD),
        ("dmMediaType", wintypes.DWORD),
        ("dmDitherType", wintypes.DWORD),
        ("dmReserved1", wintypes.DWORD),
        ("dmReserved2", wintypes.DWORD),
        ("dmPanningWidth", wintypes.DWORD),
        ("dmPanningHeight", wintypes.DWORD),
    ]


_user32.EnumDisplayDevicesW.argtypes = [
    wintypes.LPCWSTR, wintypes.DWORD, ctypes.POINTER(_DisplayDevice), wintypes.DWORD]
_user32.EnumDisplayDevicesW.restype = wintypes.BOOL
_user32.EnumDisplaySettingsW.argtypes = [
    wintypes.LPCWSTR, wintypes.DWORD, ctypes.POINTER(_DevMode)]
_user32.EnumDisplaySettingsW.restype = wintypes.BOOL
_user32.ChangeDisplaySettingsExW.argtypes = [
    wintypes.LPCWSTR, ctypes.POINTER(_DevMode), wintypes.HWND, wintypes.DWORD,
    wintypes.LPVOID]
_user32.ChangeDisplaySettingsExW.restype = wintypes.LONG


def list_screens() -> list[Screen]:
    """The screens attached to the desktop, in Windows' order."""
    screens = []
    device_num = 0
    while True:
        device = _DisplayDevice(cb=ctypes.sizeof(_DisplayDevice))
        if not _user32.EnumDisplayDevicesW(None, device_num, ctypes.byref(device), 0):
            break                                   # no more display outputs
        device_num += 1
        if not device.StateFlags & _DISPLAY_DEVICE_ATTACHED_TO_DESKTOP:
            continue                                # an output with no screen in use
        current = _read_mode(device.DeviceName, _ENUM_CURRENT_SETTINGS)
        if current is None:
            continue
        screens.append(Screen(
            index=len(screens),
            name=device.DeviceName,
            description=device.DeviceString,
            is_primary=bool(device.StateFlags & _DISPLAY_DEVICE_PRIMARY_DEVICE),
            x=current.dmPositionX,
            y=current.dmPositionY,
            mode=_to_display_mode(current)))
    return screens


def list_display_modes(device: str) -> list[DisplayMode]:
    """Every mode the screen ``device`` supports, without repeats, sorted."""
    modes = set()
    mode_num = 0
    while True:
        devmode = _read_mode(device, mode_num)
        if devmode is None:
            break                                   # no more modes
        modes.add(_to_display_mode(devmode))
        mode_num += 1
    return sorted(modes)


def check_mode(device: str, mode: DisplayMode) -> None:
    """Ask Windows whether ``device`` can run ``mode``, without changing anything.

    Raises ``RuntimeError`` with Windows' reason if it cannot.
    """
    _change_settings(device, _mode_request(device, mode), _CDS_TEST)


def set_mode(device: str, mode: DisplayMode, persistent: bool) -> None:
    """Switch ``device`` to ``mode`` after checking it; save it if ``persistent``.

    Raises ``RuntimeError`` with Windows' reason if the mode is refused.
    """
    check_mode(device, mode)
    flags = _CDS_UPDATEREGISTRY if persistent else 0
    _change_settings(device, _mode_request(device, mode), flags)


def _read_mode(device: str, mode_num: int) -> "_DevMode | None":
    """Mode ``mode_num`` of ``device`` (or its current mode); ``None`` past the last one."""
    devmode = _DevMode(dmSize=ctypes.sizeof(_DevMode))
    if not _user32.EnumDisplaySettingsW(device, mode_num, ctypes.byref(devmode)):
        return None
    return devmode


def _to_display_mode(devmode: _DevMode) -> DisplayMode:
    return DisplayMode(width=devmode.dmPelsWidth, height=devmode.dmPelsHeight,
                       fps=float(devmode.dmDisplayFrequency))


def _mode_request(device: str, mode: DisplayMode) -> _DevMode:
    """The current mode of ``device`` with the resolution and rate of ``mode``."""
    devmode = _read_mode(device, _ENUM_CURRENT_SETTINGS)
    devmode.dmPelsWidth = mode.width
    devmode.dmPelsHeight = mode.height
    devmode.dmDisplayFrequency = round(mode.fps)
    devmode.dmFields = _DM_PELSWIDTH | _DM_PELSHEIGHT | _DM_DISPLAYFREQUENCY
    return devmode


def _change_settings(device: str, devmode: _DevMode, flags: int) -> None:
    """Call ``ChangeDisplaySettingsExW``; raise ``RuntimeError`` if it does not succeed."""
    result = _user32.ChangeDisplaySettingsExW(device, ctypes.byref(devmode), None, flags,
                                              None)
    if result != _DISP_CHANGE_SUCCESSFUL:
        reason = _DISP_CHANGE_ERRORS.get(result, f"error code {result}")
        raise RuntimeError(f"Windows refused the display mode for {device}: {reason}.")
