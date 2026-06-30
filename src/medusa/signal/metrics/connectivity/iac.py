# External imports
import numpy as np
from joblib import Parallel, delayed
from numpy.typing import NDArray
from scipy.linalg import pinv
from scipy.stats import zscore

from medusa.core.utils import check_data_dims

# Medusa imports
from medusa.signal.transforms import hilbert

__all__ = ["iac"]


def __iac_ort(signal: NDArray, n_jobs: int = -1) -> NDArray:
    """Instantaneous Amplitude Correlation (IAC) with orthogonalization.

    Pairwise orthogonalisation (leakage correction) is applied before
    estimating the amplitude-envelope products.

    Parameters
    ----------
    signal
        (n_samples, n_channels). One canonical, channels-last segment.
    n_jobs
        Number of parallel workers (-1 = all CPUs).

    Returns
    -------
    numpy.ndarray
        (n_channels, n_channels, n_samples) dynamic IAC for the segment.
    """
    signal = np.asarray(signal, dtype=float)
    n_samples, n_channels = signal.shape

    # Z-score across time
    signal = zscore(signal, axis=0, ddof=1)

    # Precompute analytic signal envelopes of the original signals. The signal
    # is fed in its canonical 3-D form so the Hilbert transform runs along the
    # samples axis without triggering a promotion warning.
    envelopes = np.abs(hilbert(signal[np.newaxis]))[0]   # (n_samples, n_channels)

    def compute_row(r1: int) -> NDArray:
        """Compute dyn_IAC for one seed channel r1."""
        x = signal[:, r1]
        env_x = envelopes[:, r1]

        # Pseudoinverse of x (matches MATLAB pinv(x))
        inv_x = pinv(x.reshape(1, -1))

        row = np.zeros((n_channels, n_samples))

        for r2 in range(n_channels):
            if r1 == r2:
                continue

            y = signal[:, r2]

            # Projection coefficient
            coef = (inv_x.T @ y.reshape(-1, 1)).item()

            # Orthogonalised signal
            y_cor = y - x * coef

            # Envelope of corrected signal
            env_y = np.abs(hilbert(y_cor[np.newaxis, :, np.newaxis]))[0, :, 0]

            # Envelope product
            row[r2, :] = env_x * env_y

        return row

    # Parallel over seed channels
    rows = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(compute_row)(r1) for r1 in range(n_channels)
    )

    dyn_IAC = np.stack(rows, axis=0)

    return dyn_IAC


def __iac(signal: NDArray, n_jobs: int = -1) -> NDArray:
    """Instantaneous Amplitude Correlation (IAC) without leakage correction.

    Parameters
    ----------
    signal
        (n_samples, n_channels). One canonical, channels-last segment.
    n_jobs
        Number of parallel workers (-1 = all CPUs).

    Returns
    -------
    numpy.ndarray
        (n_channels, n_channels, n_samples) dynamic IAC for the segment.
    """
    signal = np.asarray(signal, dtype=float)
    n_samples, n_channels = signal.shape

    # Z-score across time
    signal = zscore(signal, axis=0, ddof=1)

    # Precompute envelopes once (big speedup). The signal is fed in its
    # canonical 3-D form so the Hilbert transform runs along the samples axis.
    envelopes = np.abs(hilbert(signal[np.newaxis]))[0]   # (n_samples, n_channels)

    def compute_row(r1: int) -> NDArray:
        """Compute dyn_IAC for one seed channel r1."""
        env_x = envelopes[:, r1]
        row = np.zeros((n_channels, n_samples))

        for r2 in range(n_channels):
            if r1 == r2:
                continue
            row[r2, :] = env_x * envelopes[:, r2]

        return row

    # Parallel over rows
    rows = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(compute_row)(r1) for r1 in range(n_channels)
    )

    dyn_IAC = np.stack(rows, axis=0)

    return dyn_IAC


def iac(signal: NDArray, ort: bool = True) -> NDArray:
    """Compute the Instantaneous Amplitude Correlation (IAC).

    The IAC tracks the instantaneous co-fluctuation of the amplitude envelopes
    of channel pairs, yielding a connectivity matrix that varies sample by
    sample. Optionally, the signals are pairwise orthogonalized before the
    envelopes are estimated, removing zero-lag leakage.

    References
    ----------
    Tewarie, P., Liuzzi, L., O'Neill, G. C., Quinn, A. J., Griffa, A.,
    Woolrich, M. W., ... & Brookes, M. J. (2019). Tracking dynamic brain
    networks using high temporal resolution MEG measures of functional
    connectivity. Neuroimage, 200, 38-50.

    O'Neill, G. C., Barratt, E. L., Hunt, B. A., Tewarie, P. K., & Brookes, M.
    J. (2015). Measuring electrophysiological connectivity by power envelope
    correlation: a technical review on MEG methods. Physics in Medicine &
    Biology, 60(21), R271.

    Parameters
    ----------
    signal
        (n_segments, n_samples, n_channels). M/EEG signal. A 2-D
        ``(n_samples, n_channels)`` array is accepted and promoted by
        :func:`medusa.core.utils.check_data_dims`.
    ort
        If True (default), signals are orthogonalized before computing the IAC.

    Returns
    -------
    NDArray
        (n_segments, n_channels, n_channels, n_samples) symmetric IAC matrix
        with a zero diagonal at every sample. When a 2-D signal is passed, the
        singleton segment axis is squeezed out and the result is
        (n_channels, n_channels, n_samples).

    Raises
    ------
    ValueError
        If ``signal`` contains non-numeric values, or if it does not have an
        ndim compatible with the ``'time_segments'`` representation.

    Examples
    --------
    >>> import numpy as np
    >>> from medusa.signal.metrics.connectivity.iac import iac
    >>> signal = np.random.default_rng(0).standard_normal((3, 2000, 8))
    >>> iac(signal, ort=True).shape
    (3, 8, 8, 2000)

    A single, non-segmented recording drops the segment axis:

    >>> signal2d = np.random.default_rng(0).standard_normal((1000, 8))
    >>> iac(signal2d, ort=False).shape
    (8, 8, 1000)
    """
    # Sanitise input (dtype-preserving) and validate dtype
    signal = np.asarray(signal)
    if not np.issubdtype(signal.dtype, np.number):
        raise ValueError('signal matrix contains non-numeric values')

    # Promote to the canonical segmented shape (n_segments, n_samples,
    # n_channels). The private functions consume one channels-last segment at
    # a time, honouring the channels-last contract of the rest of the package.
    signal, inserted = check_data_dims(signal, rep_type='time_segments')

    compute = __iac_ort if ort else __iac
    dyn_iac = np.stack(
        [compute(signal[s]) for s in range(signal.shape[0])], axis=0)
    # dyn_iac: (n_segments, n_channels, n_channels, n_samples)

    # Symmetrize and zero the diagonal at every time sample
    for i in range(dyn_iac.shape[0]):
        for t in range(dyn_iac.shape[3]):
            dyn_iac[i, :, :, t] = (dyn_iac[i, :, :, t] +
                                   dyn_iac[i, :, :, t].T) / 2
            np.fill_diagonal(dyn_iac[i, :, :, t], 0)

    # Squeeze back any inserted segment axis so a 2-D caller gets a 3-D result
    result = np.squeeze(dyn_iac, axis=inserted) if inserted else dyn_iac

    return result
