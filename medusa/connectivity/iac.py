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
    """
    Computes the Instantaneous Amplitude Correlation (IAC) using the standard method
    without signal orthogonalization. This version runs on the CPU.

    NOTE: The original research recommends using the orthogonalized version
    to mitigate spurious zero-lag correlations due to common sources.
    See `__iac_ort_cpu`.

    Parameters
    ----------
    data : numpy.ndarray
        M/EEG signal data of shape [n_epochs, n_samples, n_channels].

    Returns
    -------
    iac : numpy.ndarray
        IAC-based functional connectivity matrix.
        Shape: [n_epochs, n_channels, n_channels, n_samples].

    Examples
    --------
    >>> data = np.random.randn(10, 1000, 64)  # 10 epochs, 1000 time points, 64 channels
    >>> iac_matrix = __iac_cpu(data)
    >>> print(iac_matrix.shape)
    (10, 64, 64, 1000)
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
    """
    Computes the orthogonalized Instantaneous Amplitude Correlation (IAC) using CPU.
    Orthogonalization reduces spurious connectivity caused by signal leakage or
    volume conduction, preserving only the genuine amplitude correlations.

    Parameters
    ----------
    data : numpy.ndarray
        M/EEG signal data of shape [n_epochs, n_samples, n_channels].

    Returns
    -------
    iac_ort : numpy.ndarray
        Symmetrized IAC-based functional connectivity matrix after orthogonalization.
        Shape: [n_epochs, n_channels, n_channels, n_samples].

    Examples
    --------
    >>> data = np.random.randn(5, 1500, 32)  # 5 epochs, 1500 time points, 32 channels
    >>> iac_ort_matrix = __iac_ort_cpu(data)
    >>> print(iac_ort_matrix.shape)
    (5, 32, 32, 1500)
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

    # IAC Ort Calculation
    data = sp_stats.zscore(data, axis=1)

    # IAC Ort Calculation - Orthogonalized data has one additional dimension (the channel dimension is duplicated), as
    # each channel (1st channel dimension) is orthogonalized regarding every other channel (2nd channel dimension)
    signal_ort = orthogonalizate.signal_orthogonalization_cpu(data, data)
    # The two channel dimensions are merged to paralelize the computation of the AEC
    # epochs*chann*chann*samples -> epochs*chann^2*samples
    signal_ort = np.transpose(
        np.reshape(np.transpose(signal_ort, (0, 3, 2, 1)),
                   (n_epo, n_cha * n_cha, n_samp)), (0, 2, 1))

    hilb_1 = hilbert(signal_ort)
    envelope_1 = np.abs(hilb_1)

    # Comnputing IAC for each (duplicated) channel with every other one, obatining a chann^2*chann^2 matrix
    iac = np.multiply(np.reshape(np.tile(
        envelope_1, (1, n_cha**2, 1)), (n_epo, n_samp, n_cha**2*n_cha**2),
        order='F'), np.tile(envelope_1, (1, 1, n_cha**2)))
    # Reshape the data and take only the indices of interest, resulting in a chann*chann matrix
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
    # symmetric matrix required for Orthogonalized IAC
    iac_upper = np.triu(np.transpose(iac, (0,3, 1, 2)), k=1)
    iac_lower = np.transpose(np.tril(np.transpose(iac, (0,3, 1, 2)), k=-1), (0,1, 3,2))
    iac_ort = (iac_upper + iac_lower) / 2
    iac_ort = abs(np.triu(iac_ort, k=1) + np.transpose(iac_ort, (0,1, 3, 2)))

    return np.transpose(iac_ort, (0,2, 3, 1))


def iac(data, ort=True):
     """
    Computes the Instantaneous Amplitude Correlation (IAC) from M/EEG signals.
    Offers the option to orthogonalize the signals before estimating amplitude
    envelope correlations. CPU is used for computation.

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
        M/EEG signal array. Accepted shapes:
        - [n_epochs, n_samples, n_channels]
        - [n_samples, n_channels] (interpreted as one epoch)

    ort : bool, optional
        If True (default), signals are orthogonalized before computing the IAC.

    Returns
    -------
    iac : numpy.ndarray
        Functional connectivity matrix based on IAC.
        Shape: [n_epochs, n_channels, n_channels, n_samples].

    Examples
    --------
    >>> data = np.random.randn(1000, 64)  # One epoch, 1000 samples, 64 channels
    >>> conn = iac(data, ort=False)
    >>> print(conn.shape)
    (1, 64, 64, 1000)

    >>> data_multi = np.random.randn(3, 2000, 64)  # Three epochs
    >>> conn_ort = iac(data_multi, ort=True)
    >>> print(conn_ort.shape)
    (3, 64, 64, 2000)
    """

    #  Error check
     if not np.issubdtype(data.dtype, np.number):
         raise ValueError('data matrix contains non-numeric values')

     if not ort:
         iac = __iac_cpu(data)

     else:
         iac = __iac_ort_cpu(data)

     return iac
