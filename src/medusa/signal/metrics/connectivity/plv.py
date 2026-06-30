# External imports
import numpy as np
from numpy.typing import NDArray

# Medusa imports
from medusa.core.utils import check_data_dims

from ._phase_conn import _phase_conn

__all__ = ["plv"]


def plv(signal: NDArray) -> NDArray:
    """Compute the Phase-Locking Value (PLV) connectivity matrix.

    The PLV quantifies the consistency of the instantaneous phase difference
    between every pair of channels across samples. Values range from 0 (no
    phase locking) to 1 (perfect phase locking). The instantaneous phase is
    obtained from the analytic signal via the Hilbert transform.

    Parameters
    ----------
    signal
        (n_segments, n_samples, n_channels). M/EEG signal. A 2-D
        ``(n_samples, n_channels)`` array is accepted and promoted by
        :func:`medusa.core.utils.check_data_dims`.

    Returns
    -------
    NDArray
        (n_segments, n_channels, n_channels) PLV connectivity matrix per
        segment. When a 2-D signal is passed, the singleton segment axis is
        squeezed out and the result is (n_channels, n_channels).

    Raises
    ------
    ValueError
        If ``signal`` does not have an ndim compatible with the
        ``'time_segments'`` representation.

    Examples
    --------
    >>> import numpy as np
    >>> from medusa.signal.metrics.connectivity.plv import plv
    >>> signal = np.random.default_rng(0).standard_normal((10, 1000, 32))
    >>> plv(signal).shape
    (10, 32, 32)

    A single, non-segmented recording returns a 2-D matrix:

    >>> signal2d = np.random.default_rng(0).standard_normal((1000, 32))
    >>> plv(signal2d).shape
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

    plv_vector = np.divide(
        abs(np.sum(np.exp(1j * (angles_1 - angles_2)), axis=1)),
        n_samples)
    plv_matrix = np.reshape(plv_vector, (n_segments, n_channels, n_channels),
                            order='F')

    # Squeeze back any inserted segment axis so a 2-D caller gets a 2-D matrix
    plv_matrix = np.squeeze(plv_matrix, axis=inserted) if inserted else plv_matrix

    return plv_matrix
