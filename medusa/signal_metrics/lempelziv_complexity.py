# Built-in imports
import math
import ctypes
import os

# External imports
import numpy as np
from scipy.spatial.distance import pdist
from scipy.signal import decimate

# Medusa imports
from medusa.components import ThreadWithReturnValue
from medusa.utils import check_dimensions

def lempelziv_complexity(signal):
    """ This function first binarizes the input signal and then calculates its Lempel-Ziv
    Complexity (LZC), a non-linear measure of signal regularity and compressibility.
    It supports multi-channel inputs and computes LZC independently for each channel
    using multithreading for efficiency.

    Parameters
    ---------
    signal: numpy 2D array
        Signal with shape [n_samples, n_channels]

    Returns
    -------
    lz_channel_values: numpy 1D matrix
        Lempel-Ziv values for each channel with shape [n_channels].

    Notes
    -----
    - The input signal is first binarized using the `__binarisation()` method.
    - The complexity is computed via the `__lz_algorithm()` function.
    - This implementation uses Python threading for parallel computation.

    Examples
    --------
    >>> import numpy as np
    >>> from medusa.signal_metrics.lempelziv_complexity import lempelziv_complexity

    >>> signal = np.random.randn(1000, 3)  # 1000 samples, 3 channels
    >>> lzc = lempelziv_complexity(signal)
    >>> print(lzc)
    [0.87 0.89 0.91]
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


def __lz_algorithm(signal):
    """
    Compute the Lempel-Ziv Complexity (LZC) of a 1D binary signal using a basic Python implementation.

    References
    ----------
    F. Kaspar, H. G. Schuster, "Easily-calculable measure for the complexity of
    spatiotemporal patterns", Physical Review A, Volume 36, Number 2 (1987).

    Parameters
    ---------
    signal: numpy 1D array
        Signal with shape of [n_samples]
        If not already flattened, it will be converted to 1D.

    Returns
    -------
    value: float
        Estimated Lempel-Ziv complexity of the input signal.

     Notes
    -----
    - This implementation assumes the signal is binary (0s and 1s).
    - For multichannel signals, apply this function independently to each channel.

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
    w_length : int
        Window length for the median filter. Must be an odd integer.
        Required only in multiscale mode.

    w_max : int
        Maximum window length for signal alignment in multiscale mode.
        Required only in multiscale mode.

    multiscale : bool, default=False
        If True, apply multiscale binarisation using a local median baseline.
        If False, use global median binarisation.

    Returns
    -------
    signal_binarised: numpy.ndarray
        Signal binarised with shape of [n_samples_shortened x n_channels]
        The n_samples_shortened parameter is calculated from w_max to ensure that
        all signals have the same length, regardless of the window length used.

    Notes
    -----
    - Multiscale mode reduces the length of the signal due to windowing effects.
    - This function is typically used prior to applying the Lempel-Ziv complexity.
    - For multiscale complexity estimation, ensure `w_length` and `w_max` are consistent
      across epochs or channels.
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