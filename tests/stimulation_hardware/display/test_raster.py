"""Raster latency: when a monitor draws a stimulus within a frame."""
import numpy as np
import pytest

from medusa.stimulation_hardware.display import (
    DEFAULT_SCAN_FRACTION,
    raster_latency,
)

FPS = 60.0
SCAN = DEFAULT_SCAN_FRACTION / FPS      # seconds from the top line to the bottom one


def test_latency_grows_from_the_top_to_the_bottom_of_the_screen():
    assert raster_latency((0.0, 0.0), FPS) == 0.0
    assert raster_latency((0.5, 0.5), FPS) == pytest.approx(SCAN / 2)
    assert raster_latency((1.0, 1.0), FPS) == pytest.approx(SCAN)


def test_latency_does_not_depend_on_the_column():
    assert raster_latency((0.0, 0.4), FPS) == raster_latency((1.0, 0.4), FPS)


def test_latency_keeps_the_shape_of_the_positions():
    assert raster_latency(np.zeros((3, 4, 2)), FPS).shape == (3, 4)
    assert isinstance(raster_latency((0.5, 0.5), FPS), float)


@pytest.mark.parametrize("pos, fps, scan_fraction", [
    ((0.5,), FPS, 0.95),            # not an (x, y) pair
    ((0.5, 1.2), FPS, 0.95),        # below the screen
    ((-0.1, 0.5), FPS, 0.95),       # left of the screen
    ((0.5, 0.5), 0.0, 0.95),        # no refresh rate
    ((0.5, 0.5), FPS, 0.0),         # no scan time
    ((0.5, 0.5), FPS, 1.2),         # scan longer than a frame
])
def test_latency_rejects_bad_inputs(pos, fps, scan_fraction):
    with pytest.raises(ValueError):
        raster_latency(pos, fps, scan_fraction)
