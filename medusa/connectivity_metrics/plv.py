# Built-in imports
import warnings, os

# External imports
import scipy.signal as sp_signal
import numpy as np

# Medusa imports
from medusa import transforms
from medusa.utils import check_dimensions


def plv(data):
    """
    This method implements the Phase-Locking Value (PLV) for M/EEG signals using the CPU.

    Parameters
    ----------
    data : numpy.ndarray
        M/EEG signal with shape:
        - [n_epochs, n_samples, n_channels] for multi-epoch data
        - [n_samples, n_channels] for single-epoch data (converted to 3D)

    Returns
    -------
    plv : numpy.ndarray
        Phase-locking value connectivity_metrics matrix for each epoch.
        Shape: [n_epochs, n_channels, n_channels].

    Examples
    --------
    >>> data = np.random.randn(1000, 64)  # 1000 samples, 64 channels
    >>> conn = plv(data)
    >>> print(conn.shape)
    (1, 64, 64)

    >>> multi_data = np.random.randn(10, 1500, 32)  # 10 epochs, 1500 samples, 32 channels
    >>> conn = plv(multi_data)
    >>> print(conn.shape)
    (10, 32, 32)
    """

    # Error check
    if type(data) != np.ndarray:
        raise ValueError("Parameter data must be of type numpy.ndarray")

    # Set to correct dimensions
    data = check_dimensions(data)

    # Variable initialization
    n_epochs = data.shape[0]
    n_samples = data.shape[1]
    n_chan = data.shape[2]

    # Connectivity computation
    phase_data = np.angle(transforms.hilbert(data))
    phase_data = np.ascontiguousarray(phase_data)
    angles_1 = np.reshape(np.tile(phase_data, (1, n_chan, 1)),
                          (n_epochs, n_samples, n_chan * n_chan),
                          order='F')
    angles_2 = np.tile(phase_data, (1, 1, n_chan))

    plv_vector = np.divide(
        abs(np.sum(np.exp(1j * (angles_1 - angles_2)), axis=1)),
        n_samples)
    plv = np.reshape(plv_vector, (n_epochs, n_chan, n_chan), order='F')

    return plv

