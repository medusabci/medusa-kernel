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
    """
    Computes the Amplitude Envelope Correlation (AEC) on M/EEG data using the CPU.
    This version does not apply signal orthogonalization and may be affected by
    spurious correlations due to volume conduction or field spread.

    Parameters
    ----------
    data : numpy.ndarray
        M/EEG signal of shape [n_epochs, n_samples, n_channels].

    Returns
    -------
    aec : numpy.ndarray
        AEC-based functional connectivity_metrics matrix.
        Shape: [n_epochs, n_channels, n_channels].

    Examples
    --------
    >>> data = np.random.randn(5, 1000, 64)  # 5 epochs, 1000 samples, 64 channels
    >>> conn_matrix = __aec_cpu(data)
    >>> print(conn_matrix.shape)
    (5, 64, 64)
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
    """
    Computes the Orthogonalized Amplitude Envelope Correlation (AEC) on M/EEG data.
    This version reduces zero-lag artifacts by orthogonalizing the signals before
    computing correlations, thereby increasing the specificity of functional connectivity_metrics.

    Parameters
    ----------
    data : numpy.ndarray
        M/EEG signal of shape [n_epochs, n_samples, n_channels].

    Returns
    -------
    aec : numpy.ndarray
        Orthogonalized AEC-based functional connectivity_metrics matrix.
        Shape: [n_epochs, n_channels, n_channels].

    Examples
    --------
    >>> data = np.random.randn(3, 1500, 32)  # 3 epochs, 1500 samples, 32 channels
    >>> ort_conn = __aec_ort_cpu(data)
    >>> print(ort_conn.shape)
    (3, 32, 32)
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

    # AEC Ort Calculation - Orthogonalized data has one additional dimension (the channel dimension is duplicated), as
    # each channel (1st channel dimension) is orthogonalized regarding every other channel (2nd channel dimension)
    signal_ort = orthogonalizate.signal_orthogonalization_cpu(data, data)
    # The two channel dimensions are merged to paralelize the computation of the AEC
    # epochs*chann*chann*samples -> epochs*chann^2*samples
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
    Auxiliary function for computing orthogonalized AEC for a single epoch.

    Note: Orthogonalization is not symmetric. Therefore, the final connectivity_metrics
    matrix is symmetrized by averaging the upper and lower triangles.

    Parameters
    ----------
    env : numpy.ndarray
        Envelope matrix of shape [n_samples, n_channels * n_channels].

    n_cha : int
        Number of channels.

    Returns
    -------
    aec_ort : numpy.ndarray
        Symmetric AEC connectivity_metrics matrix for one epoch.
        Shape: [n_channels, n_channels].

    Example (internal use)
    ----------------------
    >>> env = np.random.randn(1000, 1024)  # e.g. for 32 channels → 32x32 = 1024
    >>> aec_epoch = __aec_ort_comp_aux(env, 32)
    >>> print(aec_epoch.shape)
    (32, 32)
    """

    # Correlate each (duplicated) channel with every other one, obatining a chann^2*chann^2 matrix
    aec_tmp = np.corrcoef(env, rowvar=False)
    # Reshape the data and take only the indices of interest, resulting in a chann*chann matrix
    aec_tmp2 = np.transpose(
        np.reshape(
            np.transpose(aec_tmp),
            (int(aec_tmp.shape[0] * aec_tmp.shape[0] / n_cha), -1)
        )
    )
    idx = np.linspace(0, aec_tmp2.shape[1] - 1, n_cha).astype(np.int32)
    aec = aec_tmp2[:, idx]

    # Orthogonalize A regarding B is not the same as orthogonalize B regarding
    # A, so we average lower and upper triangular matrices to construct the
    # symmetric matrix required for Orthogonalized AEC
    aec_upper = np.triu(np.squeeze(aec))
    aec_lower = np.transpose(np.tril(np.squeeze(aec)))
    aec_ort = (aec_upper + aec_lower) / 2
    aec_ort = abs(np.triu(aec_ort, 1) + np.transpose(aec_ort))
    return aec_ort


def aec(data, ort=True):
    """
    Computes the Amplitude Envelope Correlation (AEC) from M/EEG signals.
    Optionally performs orthogonalization to reduce spurious zero-lag correlations
    caused by common sources or volume conduction.

    This method supports both single-epoch and multi-epoch input formats.

    REFERENCES:
    Liu, Z., Fukunaga, M., de Zwart, J. A., & Duyn, J. H. (2010).
    Large-scale spontaneous fluctuations and correlations in brain electrical
    activity observed with magnetoencephalography. Neuroimage, 51(1), 102-111.

    Hipp, J. F., Hawellek, D. J., Corbetta, M., Siegel, M., & Engel,
    A. K. (2012). Large-scale cortical correlation structure of spontaneous
    oscillatory activity. Nature neuroscience, 15(6), 884-890.

    O’Neill, G. C., Barratt, E. L., Hunt, B. A., Tewarie, P. K., & Brookes, M.
    J. (2015). Measuring electrophysiological connectivity_metrics by power envelope
    correlation: a technical review on MEG methods. Physics in Medicine &
    Biology, 60(21), R271.

    Parameters
    ----------
    data : numpy.ndarray
        M/EEG signal. Accepted shapes:
        - [n_epochs, n_samples, n_channels]
        - [n_samples, n_channels] (interpreted as one epoch)

    ort : bool, optional
        If True (default), signals will be orthogonalized before computing AEC.

    Returns
    -------
    aec : numpy.ndarray
        AEC-based functional connectivity_metrics matrix.
        Shape: [n_epochs, n_channels, n_channels].

    Examples
    --------
    >>> from medusa.connectivity.aec import aec
    >>> data = np.random.randn(1000, 64)  # Single epoch, 1000 samples, 64 channels
    >>> conn = aec(data, ort=False)
    >>> print(conn.shape)
    (1, 64, 64)

    >>> from medusa.connectivity.aec import aec
    >>> data = np.random.randn(5, 2000, 32)  # Multi-epoch input
    >>> conn_ort = aec(data, ort=True)
    >>> print(conn_ort.shape)
    (5, 32, 32)
    """

    #  Error check
    if not np.issubdtype(data.dtype, np.number):
        raise ValueError('data matrix contains non-numeric values')

    if not ort:
        aec = __aec_cpu(data)
    else:
        aec = __aec_ort_cpu(data)
    return aec