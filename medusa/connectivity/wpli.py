# Built-in imports
import warnings, os

# External imports
import scipy.signal as sp_signal
import numpy as np

# Medusa imports
from medusa import transforms
from medusa.utils import check_dimensions


def wpli(data):
    """
    This method implements the Weighted Phase Lag Index (wPLI) for M/EEG signals using CPU.

    Parameters
    ----------
    data : numpy.ndarray
        M/EEG signal array. Accepts either:
        - [n_samples, n_channels] for single-epoch data, or
        - [n_epochs, n_samples, n_channels] for multi-epoch data.

    Returns
    -------
    wpli : numpy.ndarray
        wPLI-based connectivity matrix.
        Shape: [n_epochs, n_channels, n_channels].

    Examples
    --------
    >>> data = np.random.randn(2000, 16)  # Single epoch, 16 channels
    >>> conn = wpli(data)
    >>> conn.shape
    (1, 16, 16)

    >>> data = np.random.randn(5, 1500, 20)  # 5 epochs, 1500 samples, 20 channels
    >>> conn = wpli(data)
    >>> conn.shape
    (5, 20, 20)

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

    imz = np.sin(angles_1 - angles_2)
    with np.errstate(divide='ignore', invalid='ignore'):
        wpli_vector = np.divide(
            abs(np.mean(np.multiply(abs(imz), np.sign(imz)), axis=1)),
            np.mean(abs(imz), axis=1)
        )
    wpli = np.nan_to_num(
        np.reshape(wpli_vector, (n_epochs, n_chan, n_chan), order='F'))

    return wpli

