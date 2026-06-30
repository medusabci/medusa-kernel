# External imports
import numpy as np
from numpy.typing import NDArray

# Medusa imports
from medusa.core.utils import check_data_dims

from ._phase_conn import _phase_conn

__all__ = ["pli"]


def pli(signal: NDArray) -> NDArray:
    """Compute the Phase Lag Index (PLI) connectivity matrix.

    The PLI measures the asymmetry of the distribution of instantaneous phase
    differences between channel pairs, discarding zero-lag (and pi-lag)
    interactions that are typically driven by volume conduction. Values range
    from 0 (no consistent lag) to 1 (a consistent, non-zero phase lag). The
    instantaneous phase is obtained from the analytic signal via the Hilbert
    transform.

    Parameters
    ----------
    signal
        (n_segments, n_samples, n_channels). M/EEG signal. A 2-D
        ``(n_samples, n_channels)`` array is accepted and promoted by
        :func:`medusa.core.utils.check_data_dims`.

    Returns
    -------
    NDArray
        (n_segments, n_channels, n_channels) PLI connectivity matrix per
        segment, with the diagonal set to 1. When a 2-D signal is passed, the
        singleton segment axis is squeezed out and the result is
        (n_channels, n_channels).

    Raises
    ------
    ValueError
        If ``signal`` does not have an ndim compatible with the
        ``'time_segments'`` representation.

    Examples
    --------
    >>> import numpy as np
    >>> from medusa.signal.metrics.connectivity.pli import pli
    >>> signal = np.random.default_rng(0).standard_normal((10, 1000, 64))
    >>> pli(signal).shape
    (10, 64, 64)

    A single, non-segmented recording returns a 2-D matrix:

    >>> signal2d = np.random.default_rng(0).standard_normal((500, 32))
    >>> pli(signal2d).shape
    (32, 32)
    """
    # Sanitise input (dtype-preserving)
    signal = np.asarray(signal)

    # Promote to the canonical segmented shape
    signal, inserted = check_data_dims(signal, rep_type='time_segments')

    # Variable initialization
    n_segments = signal.shape[0]
    n_samples = signal.shape[1]
    n_channels = signal.shape[2]

    # Helper function to get phase angles
    angles_1, angles_2 = _phase_conn(signal, n_segments, n_samples, n_channels)

    pli_vector = abs(np.mean(np.sign(np.sin(angles_1 - angles_2)), axis=1))
    pli_matrix = np.reshape(pli_vector, (n_segments, n_channels, n_channels),
                            order='F')

    for i in range(n_segments):
        np.fill_diagonal(pli_matrix[i], 1)

    # Squeeze back any inserted segment axis so a 2-D caller gets a 2-D matrix
    pli_matrix = np.squeeze(pli_matrix, axis=inserted) if inserted else pli_matrix

    return pli_matrix
