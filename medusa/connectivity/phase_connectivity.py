# Built-in imports
import warnings, os

# External imports
import scipy.signal as sp_signal
import numpy as np

# Medusa imports
from medusa import transforms
from medusa.utils import check_dimensions


def __phase_connectivity_cpu(data, measure=None):
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

    if measure == 'pli':
        return __pli(angles_1, angles_2, n_epochs, n_chan, 'CPU')
    elif measure == 'wpli':
        return __wpli(angles_1, angles_2, n_epochs, n_chan, 'CPU')
    elif measure == 'plv':
        return __plv(angles_1, angles_2, n_epochs, n_samples, n_chan, 'CPU')
    else:
        plv = __plv(angles_1, angles_2, n_epochs, n_samples, n_chan, 'CPU')
        pli = __pli(angles_1, angles_2, n_epochs, n_chan, 'CPU')
        wpli = __wpli(angles_1, angles_2, n_epochs, n_chan, 'CPU')
        return plv, pli, wpli


def __plv(angles_1, angles_2, n_epochs, n_samples, n_chan):
    """ This auxiliary method implements both CPU and GPU PLV calculations"""
    plv_vector = np.divide(
        abs(np.sum(np.exp(1j * (angles_1 - angles_2)), axis=1)),
        n_samples)
    plv = np.reshape(plv_vector, (n_epochs, n_chan, n_chan), order='F')
    return plv


def __pli(angles_1, angles_2, n_epochs, n_chan):
    """ This auxiliary method implements both CPU and GPU PLI calculations"""
    pli_vector = abs(np.mean(np.sign(np.sin(angles_1 - angles_2)), axis=1))
    pli = np.reshape(pli_vector, (n_epochs, n_chan, n_chan), order='F')
    return pli


def __wpli(angles_1, angles_2, n_epochs, n_chan):
    """ This auxiliary method implements both CPU and GPU wPLI calculations"""
    imz = np.sin(angles_1 - angles_2)
    with np.errstate(divide='ignore', invalid='ignore'):
        wpli_vector = np.divide(
            abs(np.mean(np.multiply(abs(imz), np.sign(imz)), axis=1)),
            np.mean(abs(imz), axis=1)
        )
    wpli = np.nan_to_num(
        np.reshape(wpli_vector, (n_epochs, n_chan, n_chan), order='F'))
    return wpli


def phase_connectivity(data, measure=None):
    """ This method implements three phase-based connectivity parameters: PLV,
    PLI, and wPLI.

    REFERENCES:
    PLV: Mormann, F., Lehnertz, K., David, P., & Elger, C. E.
    (2000). Mean phase coherence as a measure for phase synchronization and its
    application to the EEG of epilepsy patients. Physica D: Nonlinear Phenomena,
    144(3-4), 358-369.

    PLI: Nolte, G., Bai, O., Wheaton, L., Mari, Z., Vorbach, S., & Hallett, M.
    (2004). Identifying true brain interaction from EEG data using the imaginary
    part of coherency. Clinical neurophysiology, 115(10), 2292-2307.

    wPLI: Vinck, M., Oostenveld, R., Van Wingerden, M., Battaglia, F., &
    Pennartz, C. M. (2011). An improved index of phase-synchronization for
    electrophysiological data in the presence of volume-conduction, noise and
    sample-size bias. Neuroimage, 55(4), 1548-1565.

    NOTE: PLV is sensitive to volume conduction effects

    Parameters
    ----------
    data : numpy matrix
        MEEG Signal. Allowed dimensions: [n_epochs, n_samples, n_channels],
        [n_samples, n_channels].
    measure: str or None
        Key of the phase connectivity measure to calculate: "plv", "pli" or
        "wpli". If None, the three phase connectivity measures are calculated.
    Returns
    -------
    plv : numpy 3D square matrix
        plv-based connectivity matrix. [n_epochs, n_channels, n_channels].
    pli : numpy 3D square matrix
        pli-based connectivity matrix. [n_epochs, n_channels, n_channels].
    wpli : numpy 3D square matrix
        wpli-based connectivity matrix. [n_epochs, n_channels, n_channels].

    """
    # Error check
    if not np.issubdtype(data.dtype, np.number):
        raise ValueError('data matrix contains non-numeric values')
    if measure is not None and (measure != 'plv' and measure != 'pli' and measure
                                != 'wpli'):
        raise ValueError('Unknown measure key. Available are: "plv", "pli" or'
                         '"wpli".')
    if measure is None:
        plv, pli, wpli, = __phase_connectivity_cpu(data, measure)
        return np.asarray(plv), np.asarray(pli), np.asarray(wpli)
    else:
        ph_cnn_measure = __phase_connectivity_cpu(data, measure)
        return np.asarray(ph_cnn_measure)
