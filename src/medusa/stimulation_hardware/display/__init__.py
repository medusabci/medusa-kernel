"""Display hardware: how a monitor presents visual stimuli.

* :mod:`~medusa.stimulation_hardware.display.raster` -- :func:`raster_latency`, the
  moment a monitor draws a stimulus within a frame, from its position on the screen.
* :mod:`~medusa.stimulation_hardware.display.screens` -- the screens of the computer:
  :func:`list_screens`, :func:`list_display_modes` and :func:`set_fps` (Windows only for
  now).
"""

from medusa.stimulation_hardware.display.raster import (
    DEFAULT_SCAN_FRACTION,
    raster_latency,
)
from medusa.stimulation_hardware.display.screens import (
    DisplayMode,
    Screen,
    list_display_modes,
    list_screens,
    set_fps,
)

__all__ = [
    "DEFAULT_SCAN_FRACTION",
    "raster_latency",
    "DisplayMode",
    "Screen",
    "list_screens",
    "list_display_modes",
    "set_fps",
]
