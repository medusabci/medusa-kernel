"""Raster latency: when a monitor draws a stimulus within a frame.

A monitor draws each frame line by line, from the top of the screen to the bottom. So a
stimulus lower on the screen appears later than the frame starts: up to about 95% of the
frame period at the bottom line, which is 7.9 ms at 120 Hz (Nagel et al., 2018).

References
----------
Nagel, S., Dreher, W., Rosenstiel, W., & Spüler, M. (2018). The effect of monitor raster
latency on VEPs, ERPs and Brain-Computer Interface performance. Journal of Neuroscience
Methods, 295, 45-50. https://doi.org/10.1016/j.jneumeth.2017.11.018

Elze, T. (2010). Achieving precise display timing in visual neuroscience experiments.
Journal of Neuroscience Methods, 191(2), 171-179.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "DEFAULT_SCAN_FRACTION",
    "raster_latency",
]

#: Fraction of the frame period a monitor takes to draw from the top line to the bottom
#: one; the rest is vertical blanking. 0.95 is the value measured on CRT and LCD
#: monitors (Elze, 2010; Nagel et al., 2018).
DEFAULT_SCAN_FRACTION = 0.95


def raster_latency(pos: ArrayLike, fps: float,
                   scan_fraction: float = DEFAULT_SCAN_FRACTION) -> NDArray | float:
    """Delay (s) from the start of a frame until the monitor draws the stimulus at ``pos``.

    A stimulus at relative height ``y`` appears ``y * scan_fraction / fps`` seconds after
    the frame starts.

    Parameters
    ----------
    pos :
        ``(..., 2)`` ``(x, y)`` position of the stimulus centre relative to the screen,
        each in ``[0, 1]``: ``(0, 0)`` is the top-left corner and ``(1, 1)`` the
        bottom-right one. Use the whole monitor, not the application window. Only ``y``
        sets the latency: LCD and OLED panels refresh a whole line at once, and on a CRT
        the left-to-right sweep adds less than one line period.
    fps :
        Display refresh rate (Hz).
    scan_fraction :
        Fraction of the frame period the scan takes from the top line to the bottom one.

    Returns
    -------
    latency :
        ``(...)`` latency of each stimulus, in seconds; a ``float`` for a single
        ``(x, y)`` pair.

    Raises
    ------
    ValueError
        If ``pos`` does not end in an ``(x, y)`` axis or has a coordinate outside
        ``[0, 1]``, if ``fps`` is not positive, or if ``scan_fraction`` is outside
        ``(0, 1]``.

    Examples
    --------
    >>> from medusa.stimulation_hardware.display import raster_latency
    >>> # Centre of the screen at 60 Hz: half of 95% of the 16.7 ms frame
    >>> round(1e3 * raster_latency((0.5, 0.5), fps=60), 2)
    7.92
    >>> # One stimulus per row of a 4-row speller at 120 Hz
    >>> rows = [(0.5, 0.125), (0.5, 0.375), (0.5, 0.625), (0.5, 0.875)]
    >>> (1e3 * raster_latency(rows, fps=120)).round(2)
    array([0.99, 2.97, 4.95, 6.93])
    """
    pos = np.asarray(pos, dtype=float)
    if pos.shape[-1:] != (2,):
        raise ValueError(f"pos must have shape (..., 2) with (x, y) pairs, got "
                         f"{pos.shape}.")
    if not fps > 0:
        raise ValueError(f"fps must be > 0, got {fps!r}.")
    if not 0 < scan_fraction <= 1:
        raise ValueError(f"scan_fraction must be in (0, 1], got {scan_fraction!r}.")
    if np.any((pos < 0) | (pos > 1)):
        x, y = pos[..., 0], pos[..., 1]
        raise ValueError(f"pos must be relative to the screen, each coordinate in [0, 1] "
                         f"with (0, 0) at the top-left corner; got x in "
                         f"[{x.min():g}, {x.max():g}] and y in [{y.min():g}, {y.max():g}].")
    latency = pos[..., 1] * scan_fraction / fps
    return latency if latency.ndim else float(latency)
