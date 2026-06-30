# External imports
import numpy as np
from numpy.typing import NDArray

# Medusa imports
from medusa.core.utils import ThreadWithReturnValue, check_data_dims
from medusa.signal import orthogonalization as orthogonalizate
from medusa.signal.transforms import hilbert

__all__ = ["aec"]


def __aec(signal: NDArray) -> NDArray:
    """Amplitude Envelope Correlation (AEC) without orthogonalization.

    This version does not apply signal orthogonalization and may therefore be
    affected by spurious correlations due to volume conduction or field
    spread.

    Parameters
    ----------
    signal
        (n_segments, n_samples, n_channels). Canonical segmented signal.

    Returns
    -------
    numpy.ndarray
        (n_segments, n_channels, n_channels) AEC connectivity matrix.
    """
    # Variable initialization
    n_segments = signal.shape[0]
    n_channels = signal.shape[2]
    aec_matrix = np.empty((n_segments, n_channels, n_channels))
    aec_matrix[:] = np.nan

    # AEC computation
    hilb = hilbert(signal)
    envelope = abs(hilb)
    env = np.log(envelope ** 2)

    # Concurrent calculation for more than one segment
    w_threads = []
    for segment in env:
        t = ThreadWithReturnValue(target=np.corrcoef,
                                  args=(segment, None, False,))
        w_threads.append(t)
        t.start()

    for segment_idx, thread in enumerate(w_threads):
        aec_matrix[segment_idx, :, :] = thread.join()

    return aec_matrix


def __aec_ort(signal: NDArray) -> NDArray:
    """Orthogonalized Amplitude Envelope Correlation (AEC).

    This version reduces zero-lag artifacts by orthogonalizing the signals
    before computing correlations, thereby increasing the specificity of the
    functional connectivity estimate.

    Parameters
    ----------
    signal
        (n_segments, n_samples, n_channels). Canonical segmented signal.

    Returns
    -------
    numpy.ndarray
        (n_segments, n_channels, n_channels) orthogonalized AEC connectivity
        matrix.
    """
    # Variable initialization
    n_segments = signal.shape[0]
    n_samples = signal.shape[1]
    n_channels = signal.shape[2]
    aec_ort = np.empty((n_segments, n_channels, n_channels))
    aec_ort[:] = np.nan

    # AEC Ort Calculation - Orthogonalized data has one additional dimension
    # (the channel dimension is duplicated), as each channel (1st channel
    # dimension) is orthogonalized regarding every other channel (2nd channel
    # dimension)
    signal_ort = orthogonalizate.signal_orthogonalization(signal, signal)
    # The two channel dimensions are merged to parallelize the computation of
    # the AEC: segments*chann*chann*samples -> segments*chann^2*samples
    signal_ort = np.transpose(
        np.reshape(np.transpose(signal_ort, (0, 3, 2, 1)),
                   (n_segments, n_channels * n_channels, n_samples)),
        (0, 2, 1))

    hilb_1 = hilbert(signal_ort)
    envelope_1 = np.abs(hilb_1)
    env = np.log(np.square(envelope_1))

    # Concurrent calculation for more than one segment
    w_threads = []
    for segment in env:
        t = ThreadWithReturnValue(target=__aec_ort_comp_aux,
                                  args=(segment, n_channels,))
        w_threads.append(t)
        t.start()

    for segment_idx, thread in enumerate(w_threads):
        aec_ort[segment_idx, :, :] = thread.join()

    return aec_ort


def __aec_ort_comp_aux(env: NDArray, n_channels: int) -> NDArray:
    """Compute the orthogonalized AEC for a single segment.

    Orthogonalization is not symmetric, so the final connectivity matrix is
    symmetrized by averaging its upper and lower triangles.

    Parameters
    ----------
    env
        (n_samples, n_channels * n_channels). Log power envelope of the
        orthogonalized signal.
    n_channels
        Number of channels.

    Returns
    -------
    numpy.ndarray
        (n_channels, n_channels) symmetric AEC connectivity matrix for one
        segment.
    """
    # Correlate each (duplicated) channel with every other one, obtaining a
    # chann^2*chann^2 matrix
    aec_tmp = np.corrcoef(env, rowvar=False)
    # Reshape the data and take only the indices of interest, resulting in a
    # chann*chann matrix
    aec_tmp2 = np.transpose(
        np.reshape(
            np.transpose(aec_tmp),
            (int(aec_tmp.shape[0] * aec_tmp.shape[0] / n_channels), -1)
        )
    )
    idx = np.linspace(0, aec_tmp2.shape[1] - 1, n_channels).astype(np.int32)
    aec_sel = aec_tmp2[:, idx]

    # Orthogonalize A regarding B is not the same as orthogonalize B regarding
    # A, so we average lower and upper triangular matrices to construct the
    # symmetric matrix required for Orthogonalized AEC
    aec_upper = np.triu(np.squeeze(aec_sel))
    aec_lower = np.transpose(np.tril(np.squeeze(aec_sel)))
    aec_ort = (aec_upper + aec_lower) / 2
    aec_ort = abs(np.triu(aec_ort, 1) + np.transpose(aec_ort))
    return aec_ort


def aec(signal: NDArray, ort: bool = True) -> NDArray:
    """Compute the Amplitude Envelope Correlation (AEC) connectivity matrix.

    The AEC measures the correlation between the power envelopes of band-pass
    filtered channel pairs. Optionally, the signals are pairwise orthogonalized
    before computing the correlation to reduce spurious zero-lag correlations
    caused by common sources or volume conduction.

    References
    ----------
    Liu, Z., Fukunaga, M., de Zwart, J. A., & Duyn, J. H. (2010). Large-scale
    spontaneous fluctuations and correlations in brain electrical activity
    observed with magnetoencephalography. Neuroimage, 51(1), 102-111.

    Hipp, J. F., Hawellek, D. J., Corbetta, M., Siegel, M., & Engel, A. K.
    (2012). Large-scale cortical correlation structure of spontaneous
    oscillatory activity. Nature neuroscience, 15(6), 884-890.

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
        If True (default), signals are orthogonalized before computing the AEC.

    Returns
    -------
    NDArray
        (n_segments, n_channels, n_channels) AEC connectivity matrix per
        segment. When a 2-D signal is passed, the singleton segment axis is
        squeezed out and the result is (n_channels, n_channels).

    Raises
    ------
    ValueError
        If ``signal`` contains non-numeric values, or if it does not have an
        ndim compatible with the ``'time_segments'`` representation.

    Examples
    --------
    >>> import numpy as np
    >>> from medusa.signal.metrics.connectivity.aec import aec
    >>> signal = np.random.default_rng(0).standard_normal((5, 2000, 32))
    >>> aec(signal, ort=True).shape
    (5, 32, 32)

    A single, non-segmented recording returns a 2-D matrix:

    >>> signal2d = np.random.default_rng(0).standard_normal((1000, 64))
    >>> aec(signal2d, ort=False).shape
    (64, 64)
    """
    # Sanitise input (dtype-preserving) and validate dtype
    signal = np.asarray(signal)
    if not np.issubdtype(signal.dtype, np.number):
        raise ValueError('signal matrix contains non-numeric values')

    # Promote to the canonical segmented shape
    signal, inserted = check_data_dims(signal, rep_type='time_segments')

    if not ort:
        aec_matrix = __aec(signal)
    else:
        aec_matrix = __aec_ort(signal)

    # Squeeze back any inserted segment axis so a 2-D caller gets a 2-D matrix
    aec_matrix = np.squeeze(aec_matrix, axis=inserted) if inserted \
        else aec_matrix

    return aec_matrix
