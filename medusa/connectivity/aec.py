# Built-in imports
import warnings, os

# External imports
import numpy as np
from scipy import stats as sp_stats

# Medusa imports
import medusa.components
from medusa import signal_orthogonalization as orthogonalizate
from medusa.transforms import hilbert
from medusa.utils import check_dimensions


def __aec_cpu(data):
    """ This method implements the amplitude envelope correlation using CPU.

    NOTE: See the orthogonalized version.

    Parameters
    ----------
    data : numpy.ndarray
        MEEG Signal. [n_epochs, n_samples, n_channels].

    Returns
    -------
    aec : numpy.ndarray
        aec-based connectivity matrix. [n_epochs, n_channels, n_channels].

    """
    # Error check
    if type(data) != np.ndarray:
        raise ValueError("Parameter data must be of type numpy.ndarray")
    # Set to correct dimensions
    data = check_dimensions(data)

    #  Variable initialization
    n_epo = data.shape[0]
    n_cha = data.shape[2]
    aec = np.empty((n_epo, n_cha, n_cha))
    aec[:] = np.nan

    # AEC computation
    hilb = hilbert(data)
    envelope = abs(hilb)
    env = np.log(envelope ** 2)

    # Concurrent calculation for more than one epoch
    w_threads = []
    for epoch in env:
        t = medusa.components.ThreadWithReturnValue(target= np.corrcoef,
                                                    args=(epoch,None,False,))
        w_threads.append(t)
        t.start()

    for epoch_idx, thread in enumerate(w_threads):
        aec[epoch_idx, :, :] = thread.join()

    return aec


def __aec_ort_cpu(data):
    """ This method implements the orthogonalized version of the amplitude
    envelope correlation using CPU. This orthogonalized version minimizes the
    spurious connectivity caused by common sources (zero-lag correlations).

    Parameters
    ----------
    data : numpy.ndarray
        MEEG Signal. [n_epochs, n_samples, n_channels].

    Returns
    -------
    aec : numpy.ndarray
        aec-based connectivity matrix. [n_epochs, n_channels, n_channels].

    """
    # Error check
    if type(data) != np.ndarray:
        raise ValueError("Parameter data must be of type numpy.ndarray")

    # Set to correct dimensions
    data = check_dimensions(data)

    # Variable initialization
    n_epo = data.shape[0]
    n_samp = data.shape[1]
    n_cha = data.shape[2]
    aec_ort = np.empty((n_epo, n_cha, n_cha))
    aec_ort[:] = np.nan

    # AEC Ort Calculation
    signal_ort = orthogonalizate.signal_orthogonalization_cpu(data, data)
    signal_ort = np.transpose(np.reshape(np.transpose(signal_ort, (0, 3, 2, 1)),
                            (n_epo, n_cha * n_cha, n_samp)),(0,2,1))

    hilb_1 = hilbert(signal_ort)
    envelope_1 = np.abs(hilb_1)
    env = np.log(np.square(envelope_1))

    # Concurrent calculation for more than one epoch
    w_threads = []
    for epoch in env:
        t = medusa.components.ThreadWithReturnValue(target=__aec_ort_comp_aux,
                                                    args=(epoch,n_cha,))
        w_threads.append(t)
        t.start()

    for epoch_idx, thread in enumerate(w_threads):
        aec_ort[epoch_idx,:,:] = thread.join()

    return aec_ort


def __aec_ort_comp_aux(env, n_cha):
    """
    Auxiliary method that implements a function to compute the orthogonalized AEC.
    Parameters
    ----------
    env: numpy.ndarray
        Array with signal envelope. [n_epochs, n_samples, n_channels x n_channels].
    Returns
    -------
    aec_ort: numpy.ndarray
        AEC orthogonalized connectivity matrix. [n_channels, n_channels].
    """
    # Note: Orthogonalize A regarding B is not the same as orthogonalize B regarding
    # A, so we average lower and upper triangular matrices to construct the
    # symmetric matrix required for Orthogonalized AEC

    aec_tmp = np.corrcoef(env, rowvar=False)
    aec_tmp2 = np.transpose(
        np.reshape(
            np.transpose(aec_tmp),
            (int(aec_tmp.shape[0] * aec_tmp.shape[0] / n_cha), -1)
        )
    )
    idx = np.linspace(0, aec_tmp2.shape[1] - 1, n_cha).astype(np.int32)
    aec = aec_tmp2[:, idx]
    aec_upper = np.triu(np.squeeze(aec))
    aec_lower = np.transpose(np.tril(np.squeeze(aec)))
    aec_ort = (aec_upper + aec_lower) / 2
    aec_ort = abs(np.triu(aec_ort, 1) + np.transpose(aec_ort))
    return aec_ort


def aec(data, ort=True):
    """ This method implements the amplitude envelope correlation (using GPU if
    available). Based on the "ort" param, the signals could be orthogonalized
    before the computation of the amplitude envelope correlation.

    REFERENCES:
    Liu, Z., Fukunaga, M., de Zwart, J. A., & Duyn, J. H. (2010).
    Large-scale spontaneous fluctuations and correlations in brain electrical
    activity observed with magnetoencephalography. Neuroimage, 51(1), 102-111.

    Hipp, J. F., Hawellek, D. J., Corbetta, M., Siegel, M., & Engel,
    A. K. (2012). Large-scale cortical correlation structure of spontaneous
    oscillatory activity. Nature neuroscience, 15(6), 884-890.

    O’Neill, G. C., Barratt, E. L., Hunt, B. A., Tewarie, P. K., & Brookes, M.
    J. (2015). Measuring electrophysiological connectivity by power envelope
    correlation: a technical review on MEG methods. Physics in Medicine &
    Biology, 60(21), R271.

    Parameters
    ----------
    data : numpy.ndarray
        MEEG Signal. Allowed dimensions: [n_epochs, n_samples, n_channels] and
        [n_samples, n_channels].
    ort : bool
        If True, the signals on "data" will be orthogonalized before the
        computation of the amplitude envelope correlation.

    Returns
    -------
    aec : numpy.ndarray
        aec-based connectivity matrix. [n_epochs, n_channels, n_channels].

    """
    #  Error check
    if not np.issubdtype(data.dtype, np.number):
        raise ValueError('data matrix contains non-numeric values')

    if not ort:
        aec = __aec_cpu(data)
    else:
        aec = __aec_ort_cpu(data)
    return aec