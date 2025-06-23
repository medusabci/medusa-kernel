# Built-in imports
import warnings, os

# External imports
import scipy.signal as sp_signal
import numpy as np

# Medusa imports
from medusa import transforms
from medusa.utils import check_dimensions


def pli(data):
    """
    This method implements the Phase Lag Index (PLI) for M/EEG signals using CPU.

     Parameters
    ----------
    data : numpy.ndarray
        M/EEG signal array. Accepts either:
        - [n_samples, n_channels] for single-epoch data, or
        - [n_epochs, n_samples, n_channels] for multi-epoch data.

    Returns
    -------
    pli : numpy.ndarray
        PLI-based connectivity_metrics matrix.
        Shape: [n_epochs, n_channels, n_channels].

    Examples
    --------
    >>> from medusa.connectivity.pli import pli
    >>> data = np.random.randn(500, 32)  # 500 samples, 32 channels
    >>> result = pli(data)
    >>> result.shape
    (1, 32, 32)

    >>> from medusa.connectivity.pli import pli
    >>> data = np.random.randn(10, 1000, 64)  # 10 epochs, 1000 samples, 64 channels
    >>> result = pli(data)
    >>> result.shape
    (10, 64, 64)

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

    pli_vector = abs(np.mean(np.sign(np.sin(angles_1 - angles_2)), axis=1))
    pli = np.reshape(pli_vector, (n_epochs, n_chan, n_chan), order='F')

    for i in range(n_epochs):
        np.fill_diagonal(pli[i], 1)

    return pli



