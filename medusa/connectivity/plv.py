# Built-in imports
import warnings, os

# External imports
import scipy.signal as sp_signal
import numpy as np

# Medusa imports
from medusa import transforms
from medusa.utils import check_dimensions


def plv(data):
    """ This method implements three phase-based connectivity parameters using
    CPU: PLV, PLI, and wPLI.

    Parameters
    ----------
    data : numpy 2D matrix
        MEEG Signal. Allowed dimensions: [n_epochs, n_samples, n_channels],
        [n_samples, n_channels].
    measure: str or None
        Key of the phase connectivity measure to calculate: "plv", "pli" or
        "wpli". If None, the three phase connectivity measures are calculated.

    Returns
    -------
    plv : numpy 3D square matrix
        plv-based connectivity matrix. [n_epochs, n_channels x n_channels].
    pli : numpy 3D square matrix
        pli-based connectivity matrix. [n_epochs, n_channels x n_channels].
    wpli : numpy 3D square matrix
        wpli-based connectivity matrix. [n_epochs, n_channels x n_channels].

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

