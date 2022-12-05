# Built-in imports
import math
import ctypes

# External imports
import numpy as np
from scipy.spatial.distance import pdist
from scipy.signal import decimate

# Medusa imports
from medusa.components import ThreadWithReturnValue
from medusa.utils import check_dimensions


def central_tendency_measure(signal, r):
    """
    This method implements the central tendency measure (CTM). This parameter is
    useful to quantify the variability of a signal. It is based on calculating
    the second-order differences diagram of the time series and then counting
    the points within a radius "r". CTM assigns higher values to less variable
    signals

    References
    ----------
    Cohen, M. E., Hudson, D. L., & Deedwania, P. C. (1996). Applying
    continuous chaotic modeling to cardiac signal analysis. IEEE Engineering in
    Medicine and Biology Magazine, 15(5), 97-102.

    Parameters
    ----------
    signal : numpy.ndarray
        Signal with shape [n_epochs, n_samples, n_channels].
    r : double
        Radius used to compute the CTM.

    Returns
    -------
    ctm : numpy.ndarray
        CTM values for each channel in "signal". [n_epochs, n_channels].
    """

    #  Error check
    if not np.issubdtype(signal.dtype, np.number):
        raise ValueError('data matrix contains non-numeric values')

    # Check dimensions
    signal = check_dimensions(signal)

    # Signal dimensions
    n_epo = signal.shape[0]
    n_samp = signal.shape[1]
    n_cha = signal.shape[2]

    # Values within a range (mean +- 3 std)
    upper_bound = np.mean(signal, axis=1) + 3 * np.std(signal, axis=1)
    lower_bound = np.mean(signal, axis=1) - 3 * np.std(signal, axis=1)
    idx_within_range = np.logical_and((signal < upper_bound[:, None, :]),
                                      (signal > lower_bound[:, None, :]))
    idx_out_upper = (signal > upper_bound[:, None, :])
    idx_out_lower = (signal < lower_bound[:, None, :])

    # Maximum value in the above defined range
    max_value = np.empty((n_epo, n_cha))
    for ep in range(n_epo):
        for ch in range(n_cha):
            max_value[ep, ch] = np.max(
                abs(signal[ep, idx_within_range[ep, :, ch], ch]), axis=0)

    # Normalize the values within the range by its maximum.Values above that
    # range will be 1, and below the range will be - 1
    data_norm = np.zeros_like(signal)
    data_norm[idx_within_range] = np.divide(
        signal[idx_within_range],
        np.tile(max_value, (1, n_samp, 1)).flatten()[
            idx_within_range.flatten()])
    data_norm[idx_out_upper] = 1
    data_norm[idx_out_lower] = -1

    # Difference time series
    y = data_norm[:, 3:n_samp, :] - data_norm[:, 2:n_samp - 1, :]
    x = data_norm[:, 2:n_samp - 1, :] - data_norm[:, 1:n_samp - 2, :]

    # CTM - Values below the radius 'r'
    ctm = np.sum(np.sqrt(np.square(x) + np.square(y)) < r, axis=1) / (
                n_samp - 2)

    return ctm


def sample_entropy(signal, m, r, dist_type='chebyshev'):
    """ This method implements the sample entropy (SampEn). SampEn is an
    irregularity measure that assigns higher values to more irregular time
    sequences. It has two tuning parameters: the sequence length (m) and the
    tolerance (r)

    Notes: IF A = 0 or B = 0, SamEn would return an infinite value.
    However, the lowest non-zero conditional probability that SampEn should
    report is A/B = 2/[(N-m-1)*(N-m)].
    SampEn has the following limits:
                 - Lower bound: 0
                 - Upper bound : log(N-m) + log(N-m-1) - log(2)

    References
    ----------
    Richman, J. S., & Moorman, J. R. (2000). Physiological
    time-series analysis using approximate entropy and sample entropy. American
    Journal of Physiology-Heart and Circulatory Physiology.

    Parameters
    ----------
    signal : numpy.ndarray
        Signal with shape [n_epochs, n_samples, n_channels].
    m : int
        Sequence length
    r : float
        Tolerance
    dist_type : string
        Distance metric

    Returns
    -------
    sampen : numpy.ndarray
        SampEn value. [n_epochs, n_channels]
    """
    # Check dimensions
    signal = check_dimensions(signal)

    # Check Errors
    if m > signal.shape[1]:
        raise ValueError('Embedding dimension must be smaller than the signal '
                         'length (m<N).')
    if not isinstance(dist_type, str):
        raise ValueError('Distance type must be a string.')
    if dist_type not in ['braycurtis', 'canberra', 'chebyshev', 'cityblock',
                         'correlation', 'cosine', 'dice', 'euclidean',
                         'hamming', 'jaccard', 'jensenshannon', 'kulsinski',
                         'mahalanobis', 'matching', 'minkowski',
                         'rogerstanimoto', 'russellrao', 'seuclidean',
                         'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']:
        raise ValueError(
            'Distance type unknown. Please, check allowed distances'
            'in pdist function from scipy.spatial.distance module.')

    # Useful parameters
    n_epo = signal.shape[0]
    N = signal.shape[1]
    n_channels = signal.shape[2]
    sigma = np.std(signal, axis=1)
    templates_m = []
    templates_m_plus_one = []
    B, A, value = np.empty((n_epo, n_channels)), np.empty((n_epo, n_channels)),\
                  np.empty((n_epo, n_channels))

    # Calculate B values
    for i in range(N - m + 1):
        templates_m.append(signal[:, i:i + m, :])
    templates_m = np.array(templates_m)
    for e_idx in range(n_epo):
        w_threads = []
        for ch_idx in range(n_channels):
            t = ThreadWithReturnValue(
                target=pdist,
                args=(templates_m[:, e_idx, :, ch_idx], dist_type,))
            w_threads.append(t)
            t.start()
        for th_idx, thread in enumerate(w_threads):
            B[e_idx, th_idx] = np.sum(thread.join() <= sigma[0, th_idx] * r)

    # Check if there is any B = 0
    zeros_idx = np.where(B == 0)
    value[zeros_idx] = math.inf

    # Calculate A values
    m += 1
    for i in range(N - m + 1):
        templates_m_plus_one.append(signal[:, i:i + m, :])
    templates_m_plus_one = np.array(templates_m_plus_one)
    for e_idx in range(n_epo):
        w_threads = []
        for ch_idx in range(n_channels):
            t = ThreadWithReturnValue(
                target=pdist, args=(templates_m_plus_one[:, e_idx, :, ch_idx],
                                    dist_type,))
            w_threads.append(t)
            t.start()
        for th_idx, thread in enumerate(w_threads):
            A[e_idx, th_idx] = np.sum(thread.join() <= sigma[0, th_idx] * r)

    # Check if there is any A = 0
    zeros_idx = np.where(A == 0)
    value[zeros_idx] = math.inf

    non_inf = np.where(value != math.inf)
    value[non_inf] = -np.log((A[non_inf] / B[non_inf])*((N - m + 1) / (N - m - 1)))

    # If there is infinity values
    inf_indx = np.where(value == math.inf)
    value[inf_indx] = -np.log(2 / ((N - m - 1) * (N - m)))
    return value


def multiscale_entropy(signal, max_scale, m, r):
    """ This method implements the Multiscale Entropy (MSE). MSE is a
    measurement of complexity which measures the irregularity of a signal over
    multiple time-scales. This is accomplished through estimation of the Sample
    Entropy (SampEn) on coarse-grained versions of the original signal. As a
    result of the these calculations, MSE curves are obtained and can be used to
    compare the complexity of time-series. The MSE curve whose entropy values
    are higher for the most of time-scales is considered more complex.

    References
    ----------
    Costa, M., Goldberger, A. L., & Peng, C. K. (2005). Multiscale entropy
    analysis of biological signals, Physical review E, 71(2), 021906.

    Parameters
    ----------
    signal : numpy.ndarray
        MEEG Signal. [n_epochs, n_samples, n_channels].
    max_scale : int
        Maximum scale value
    m : int
        Sequence length
    r : float
        Tolerance

    Returns
    -------
    mse_result : numpy.ndarray
        SampEn values for each scale for each channel in "signal" .
        [n_epochs, max_scale, n_channels].

    """
    # Check dimensions
    signal = check_dimensions(signal)

    # Signal dimensions
    n_epo = signal.shape[0]
    n_channels = signal.shape[2]

    mse_result = np.empty((n_epo, max_scale, n_channels))
    w_threads = list()
    scales = list()
    for i in range(1, max_scale + 1):
        if i == 1:
            t = ThreadWithReturnValue(
                target=sample_entropy,
                args=(signal, m, r, 'chebyshev'))
        else:
            t = ThreadWithReturnValue(
                target=sample_entropy,
                args=(__coarse_grain(signal, i),
                      m, r, 'chebyshev'))
        w_threads.append(t)
        scales.append(i)
        t.start()
    for t_idx, thread in enumerate(w_threads):
        mse_result[:, t_idx, :] = thread.join()
    return mse_result


def __coarse_grain(signal, scale, decimate_mode=True):
    if decimate_mode:
        return decimate(signal, scale, axis=1)
    else:
        # Signal dimensions
        n_epo = signal.shape[0]
        N = signal.shape[1]
        n_cha = signal.shape[2]

        # Number of coarse grains in which the signal is split
        tau = int(round(N / scale))

        # Returned signal
        y = np.empty((n_epo,tau,n_cha))
        for i in range(tau):
            y[:, i, :] = np.mean(signal[:, i * scale:(i * scale + scale), :])
        return y


def __lempelziv_complexity(signal):
    """ Calculate the  signal binarisation and the Lempel-Ziv's complexity. This
    version allows the algorithm to be used for signals with more than one
    channel at the same time.

    Warnings: This function is deprecated and will be removed in future updates
    of MEDUSA Kernel.

    Parameters
    ---------
    signal: numpy 2D array
        Signal with shape [n_samples, n_channels]

    Returns
    -------
    lz_channel_values: numpy 1D matrix
        Lempel-Ziv values for each channel with shape [n_channels].
    """

    if signal.size == len(signal):
        signal = signal[:, np.newaxis]

    signal = __binarisation(signal, [signal.shape[0]], signal.shape[0])
    if signal.shape[1] == 1:
        return __lz_algorithm(signal)
    else:
        lz_channel_values = np.empty((signal.shape[1]))
        working_threads = list()
        for ch in range(signal.shape[1]):
            t = ThreadWithReturnValue(target=__lz_algorithm,
                                      args=(signal[:, ch],))
            working_threads.append(t)
            t.start()
        for index, thread in enumerate(working_threads):
            lz_channel_values[index] = thread.join()
        return lz_channel_values


def lempelziv_complexity(signal):
    """Calculate the  signal binarisation and the Lempel-Ziv's complexity.
    This function takes advantage of its implementation in C to achieve a high
    performance.

   Parameters
   ---------
   signal: numpy.ndarray
       MEEG Signal [n_epochs, n_samples, n_channels]

   Returns
   -------
   lz_result: numpy.ndarray
       Lempel-Ziv values for each epoch and each channel. [n_epochs, n_channels].
    """
    # Check dimensions
    signal = check_dimensions(signal)

    # Signal dimensions
    n_epo = signal.shape[0]
    n_sample = signal.shape[1]
    n_cha = signal.shape[2]

    # Initialize result matrix
    lz_result = np.empty((n_epo, n_cha))

    # Adapt the signal to the format required by the DLL
    signal = np.reshape(np.transpose(signal, (0, 2, 1)), (n_epo, -1))

    # Get function from dll
    dll_file = ".\computeLZC.dll"
    lib = ctypes.cdll.LoadLibrary(dll_file)
    lzc_func = lib.computeLempelZivCmplx  # Access function

    for e_idx, epochs in enumerate(signal):
        # Create empty output vector
        lz_channel_values = np.zeros(int(n_cha), dtype=np.double)

        # Define inputs and outputs
        lzc_func.restype = None
        array_1d_double = np.ctypeslib.ndpointer(dtype=np.double, ndim=1,
                                                 flags='CONTIGUOUS')
        lzc_func.argtypes = [array_1d_double, array_1d_double, ctypes.c_int32,
                             ctypes.c_int32]

        # Call the function in the dll
        lzc_func(signal[e_idx, :], lz_channel_values, int(n_sample * n_cha),
                 int(n_sample))
        lz_result[e_idx, :] = lz_channel_values
    return lz_result


def multiscale_lempelziv_complexity(signal, W):
    """Calculate the multiscale signal binarisation and the Multiscale
    Lempel-Ziv's complexity.

    References
    ----------
    Ibáñez-Molina, A. J., Iglesias-Parro, S., Soriano, M. F., & Aznarte, J. I,
    Multiscale Lempel-Ziv complexity for EEG measures. Clinical Neurophysiology,
    (2015), 126(3), 541–548.

    Parameters
    ---------
    signal: numpy.ndarray
        MEEG Signal [n_epochs, n_samples, n_channels]
    W: list or numpy 1D array
        Set of window length to consider at multiscale binarisation stage.
        Values must be odd.

    Returns
    -------
    result: numpy.ndarray
        Matrix of results with shape [n_epochs, n_window_length, n_channels]

    """
    # Check dimensions
    signal = check_dimensions(signal)

    # Signal dimensions
    n_epo = signal.shape[0]
    n_cha = signal.shape[2]

    # Useful parameter
    w_max = W[-1] + (W[1] - W[0])

    # Define a matrix to store results
    result = np.empty((n_epo, len(W), n_cha))

    # First get binarised signal
    for ep_idx, epoch in enumerate(signal):
        for w_idx, w in enumerate(W):
            binarised_signal = __binarisation(epoch, w, w_max, multiscale=True)

            # Parallelize the calculations if n_channel > 1
            if binarised_signal.shape[1] > 1:
                threads = []
                for ch in range(binarised_signal.shape[1]):
                    t = ThreadWithReturnValue(target=__lz_algorithm,
                                              args=(binarised_signal[:, ch],))
                    threads.append(t)
                    t.start()
                for ch_idx, t in enumerate(threads):
                    result[ep_idx, w_idx, ch_idx] = t.join()
            else:
                result[ep_idx, w_idx, :] = __lz_algorithm(binarised_signal)
    return result


def __lz_algorithm(signal):
    """
    Lempel-Ziv's complexity algorithm implemented in Python.

    References
    ----------
    F. Kaspar, H. G. Schuster, "Easily-calculable measure for the complexity of
    spatiotemporal patterns", Physical Review A, Volume 36, Number 2 (1987).

    Parameters
    ---------
    signal: numpy 1D array
        Signal with shape of [n_samples]

    Returns
    -------
    value: float
        Result of algorithm calculations.

    """
    signal = signal.flatten().tolist()
    i, k, l = 0, 1, 1
    c, k_max = 1, 1
    n = len(signal)
    while True:
        if signal[i + k - 1] == signal[l + k - 1]:
            k = k + 1
            if l + k > n:
                c = c + 1
                break
        else:
            if k > k_max:
                k_max = k
            i = i + 1
            if i == l:
                c = c + 1
                l = l + k_max
                if l + 1 > n:
                    break
                else:
                    i = 0
                    k = 1
                    k_max = 1
            else:
                k = 1

    value = c * (np.log2(n) / n)
    return value


def __binarisation(signal, w_length, w_max, multiscale=False):
    """
    This function returns a binarised version of original signal. It can be used
    in both multiscale and simple binarisations. If multiscale mode is chosen,
    signals are shortened so that they all have the same length, taking the
    maximum window as a reference. Binarisation is performed by means of a
    median-based comparison.

    Parameters
    ----------
    signal: numpy 2D array
        Signal with shape [n_samples x n_channels]
    w_length: int
        Window length to perform multiscale binarisation
    w_max: int
        Value of the maximum window length
    multiscale: bool
        If is True, performs the multiscale binarisation

    Returns
    -------
    signal_binarised: numpy.ndarray
        Signal binarised with shape of [n_samples_shortened x n_channels]
        The n_samples_shortened parameter is calculated from w_max to ensure that
        all signals have the same length, regardless of the window length used.
    """

    if multiscale:
        if w_length is None:
            raise ValueError('Width of window must be an integer value')
        if w_length % 2 == 0:
            raise ValueError('Width of window must be an odd value.')
        if w_max is None:
            raise ValueError('Maximum window width must be an integer value')

        #  Get smoothed version from original signal
        smoothed = __multiscale_median_threshold(signal, w_length)

        # Useful parameters
        half_wind = int((w_length - 1) / 2)
        max_length = signal.shape[0] + 1 - w_max
        length_diff = smoothed.shape[0] - max_length

        # Shorten original and smoothed version
        smoothed_shortened = \
            smoothed[int(length_diff / 2):-int(length_diff / 2), :]
        signal_shortened = \
            signal[half_wind: signal.shape[0] - half_wind, :]
        signal_shortened = \
            signal_shortened[int(length_diff / 2):-int(length_diff / 2), :]

        # Define template of binarised signal
        signal_binarised = \
            np.zeros((signal_shortened.shape[0], signal_shortened.shape[1]))

        # Binarise the signal
        idx_one = signal_shortened >= smoothed_shortened
        signal_binarised[idx_one] = 1

    else:
        signal_binarised = np.zeros((len(signal), signal.shape[1]))
        median = np.median(signal, axis=0)
        idx_one = signal >= median
        signal_binarised[np.squeeze(idx_one)] = 1

    return signal_binarised


def __multiscale_median_threshold(signal, w_length):
    """ Signal smoothing function. For each sample, we define a window in which
    the sample is in centre position. The median value of the window is
    calculated and assigned to this sample to obtain a smoothed version of the
    original signal.

    References
    ----------
    Ibáñez-Molina, A. J., Iglesias-Parro, S., Soriano, M. F., & Aznarte, J. I,
    Multiscale Lempel-Ziv complexity for EEG measures. Clinical Neurophysiology,
    (2015), 126(3), 541–548.

    Parameters
    ---------
    signal: numpy 2D array
        Signal with shape of [n_samples, n_channels]
    w_length: int
        Length of window to calculate median values in smoothing process

    Returns
    -------
    smoothed_signal: numpy 2D array
        Smoothed version with shape of [n_samples + 1 - w_length, n_channels]
        As a result of windowing, the final samples of the signal are lost.

    """
    # Template of smoothed signal
    smoothed_signal = np.zeros((
        signal.shape[0] + 1 - w_length, signal.shape[1]))

    half_wind = int((w_length - 1) / 2)

    # Index of sample to be smoothed from median window value
    index = 0

    # We define a window with samp in central position and
    # get median value to smooth original signal
    for samp in range(half_wind, signal.shape[0] - half_wind):
        smoothed_signal[index, :] = np.median(
            signal[samp - half_wind: samp + half_wind], axis=0)
        index += 1
    return smoothed_signal

