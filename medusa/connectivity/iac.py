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

def __iac_cpu(data):
    """ This method implements the instantaneous amplitude correlation using
    CPU.

    NOTE: See the orthogonalized version. In the original paper, the
    orthogonalized version was used

    Parameters
    ----------
    data : numpy.ndarray
        MEEG Signal. [n_epochs, n_samples, n_channels].

    Returns
    -------
    iac : numpy.ndarray
        iac-based connectivity matrix.
        [n_epochs, n_channels, n_channels, n_samples].

    """
    # Error check
    if type(data) != np.ndarray:
        raise ValueError("Parameter data must be of type numpy.ndarray")

    # Set to correct dimensions
    data = check_dimensions(data)

    #  Variable initialization
    n_epo = data.shape[0]
    n_samp = data.shape[1]
    n_cha = data.shape[2]

    # IAC computation
    data = sp_stats.zscore(data, axis=1)

    hilb = hilbert(data)
    envelope = abs(hilb)
    iac = np.multiply(np.reshape(
        np.tile(envelope, (1, n_cha, 1)), (n_epo, n_samp, n_cha*n_cha), order='F'),
        np.tile(envelope, (1, 1, n_cha)))
    iac = np.reshape(np.transpose(iac,(0,2,1)), (n_epo, n_cha, n_cha, n_samp))

    # Set diagonal to 0
    diag_mask = np.ones((n_cha, n_cha))
    np.fill_diagonal(diag_mask, 0)
    iac = iac * np.repeat(np.repeat(diag_mask[None,:, :, None], n_samp, axis=-1),
                          n_epo,axis=0)

    return iac


def __iac_ort_cpu(data):
    """ This method implements the orthogonalized version of the instantaneous
    amplitude correlation using CPU. This orthogonalized version minimizes the
    spurious connectivity caused by common sources (zero-lag correlations).

    Parameters
    ----------
    data : numpy.ndarray
        MEEG Signal. [n_epochs, n_samples, n_channels].

    Returns
    -------
    iac_ort : numpy.ndarray
        iac-based connectivity matrix.
        [n_epochs, n_channels, n_channels, n_samples].
    """
    # Error check
    if type(data) != np.ndarray:
        raise ValueError("Parameter data must be of type numpy.ndarray")

    # Set to correct dimensions
    data = check_dimensions(data)

    #  Variable initialization
    n_epo = data.shape[0]
    n_samp = data.shape[1]
    n_cha = data.shape[2]

    # AEC Ort Calculation
    data = sp_stats.zscore(data, axis=1)

    signal_ort = orthogonalizate.signal_orthogonalization_cpu(data, data)
    signal_ort_2 = np.transpose(
        np.reshape(np.transpose(signal_ort, (0, 3, 2, 1)),
                   (n_epo, n_cha * n_cha, n_samp)), (0, 2, 1))

    hilb_1 = hilbert(signal_ort_2)
    envelope_1 = np.abs(hilb_1)

    iac = np.multiply(np.reshape(np.tile(
        envelope_1, (1, n_cha**2, 1)), (n_epo, n_samp, n_cha**2*n_cha**2),
        order='F'), np.tile(envelope_1, (1, 1, n_cha**2)))
    iac = np.reshape(np.transpose(iac,[0,2,1]), (n_epo,n_cha**2, n_cha**2, n_samp))
    iac_tmp2 = np.transpose(
        np.reshape(
            np.transpose(iac, (0,2, 1, 3)),
            (n_epo,int(iac.shape[1] * iac.shape[1] / n_cha), -1, n_samp)
        ), (0,2, 1, 3)
    )
    idx = np.linspace(0, iac_tmp2.shape[2]-1, n_cha).astype(np.int32)
    iac = iac_tmp2[:,:, idx, :]

    # Orthogonalize A regarding B is not the same as orthogonalize B regarding
    # A, so we average lower and upper triangular matrices to construct the
    # symmetric matrix required for Orthogonalized AEC

    iac_upper = np.triu(np.transpose(iac, (0,3, 1, 2)), k=1)
    iac_lower = np.transpose(np.tril(np.transpose(iac, (0,3, 1, 2)), k=-1), (0,1, 3,
                                                                           2))
    iac_ort = (iac_upper + iac_lower) / 2
    iac_ort = abs(np.triu(iac_ort, k=1) + np.transpose(iac_ort, (0,1, 3, 2)))

    return np.transpose(iac_ort, (0,2, 3, 1))


def iac(data, ort=True):
    """ This method implements the instantaneous amplitude correlation (using
    GPU if available). Based on the "ort" param, the signals could be
    orthogonalized before the computation of the amplitude envelope correlation.

    REFERENCES:
    Tewarie, P., Liuzzi, L., O'Neill, G. C., Quinn, A. J., Griffa,
    A., Woolrich, M. W., ... & Brookes, M. J. (2019). Tracking dynamic brain
    networks using high temporal resolution MEG measures of functional
    connectivity. Neuroimage, 200, 38-50.

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
        computation of the instantaneous amplitude correlation.

    Returns
    -------
    iac : numpy 2D square matrix
        iac-based connectivity matrix.
        [n_epochs, n_channels, n_channels, n_samples].

    """
    #  Error check
    if not np.issubdtype(data.dtype, np.number):
        raise ValueError('data matrix contains non-numeric values')

    if not ort:
        iac = __iac_cpu(data)

    else:
        iac = __iac_ort_cpu(data)

    return iac
