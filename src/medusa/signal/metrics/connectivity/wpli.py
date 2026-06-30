# External imports
import numpy as np
from numpy.typing import NDArray

# Medusa imports
from medusa.core.utils import check_data_dims

from ._phase_conn import _phase_conn

__all__ = ["wpli"]


def wpli(signal: NDArray) -> NDArray:
    """Compute the Weighted Phase Lag Index (wPLI) connectivity matrix.

    The wPLI extends the Phase Lag Index by weighting each phase difference by
    the magnitude of the imaginary part of the cross-spectrum, which makes the
    estimate more robust to noise and to small perturbations around zero lag.
    Values range from 0 to 1. The instantaneous phase is obtained from the
    analytic signal via the Hilbert transform.

    Parameters
    ----------
    signal
        (n_segments, n_samples, n_channels). M/EEG signal. A 2-D
        ``(n_samples, n_channels)`` array is accepted and promoted by
        :func:`medusa.core.utils.check_data_dims`.

    Returns
    -------
    NDArray
        (n_segments, n_channels, n_channels) wPLI connectivity matrix per
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
    >>> from medusa.signal.metrics.connectivity.wpli import wpli
    >>> signal = np.random.default_rng(0).standard_normal((5, 1500, 20))
    >>> wpli(signal).shape
    (5, 20, 20)

    A single, non-segmented recording returns a 2-D matrix:

    >>> signal2d = np.random.default_rng(0).standard_normal((2000, 16))
    >>> wpli(signal2d).shape
    (16, 16)
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

    imz = np.sin(angles_1 - angles_2)
    with np.errstate(divide='ignore', invalid='ignore'):
        wpli_vector = np.divide(
            abs(np.mean(np.multiply(abs(imz), np.sign(imz)), axis=1)),
            np.mean(abs(imz), axis=1)
        )
    wpli_matrix = np.nan_to_num(
        np.reshape(wpli_vector, (n_segments, n_channels, n_channels), order='F'))

    for i in range(n_segments):
        np.fill_diagonal(wpli_matrix[i], 1)

    # Squeeze back any inserted segment axis so a 2-D caller gets a 2-D matrix
    wpli_matrix = np.squeeze(wpli_matrix, axis=inserted) if inserted \
        else wpli_matrix

    return wpli_matrix
