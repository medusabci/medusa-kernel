import numpy as np
from numpy.typing import NDArray

from medusa.signal import transforms


def _phase_conn(signal: NDArray, n_segments: int, n_samples: int,
                n_channels: int) -> tuple[NDArray, NDArray]:
    """Build the tiled instantaneous-phase arrays shared by phase-based metrics.

    Private helper. ``signal`` is the canonical segmented signal with shape
    ``(n_segments, n_samples, n_channels)``. Returns the two broadcast phase
    arrays whose channel-pair difference feeds PLV, PLI and wPLI.
    """
    phase_data = np.angle(transforms.hilbert(signal))
    phase_data = np.ascontiguousarray(phase_data)
    angles_1 = np.reshape(np.tile(phase_data, (1, n_channels, 1)),
                          (n_segments, n_samples, n_channels * n_channels),
                          order='F')
    angles_2 = np.tile(phase_data, (1, 1, n_channels))

    return angles_1, angles_2
