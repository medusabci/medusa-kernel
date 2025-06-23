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


def multiscale_lempelziv_complexity(signal, W):
    """Calculate the Multiscale Lempel-Ziv's complexity.

    This function applies multiscale binarisation to each epoch and channel of the input signal
    using a set of predefined window lengths. It then calculates the Lempel-Ziv complexity
    (a measure of sequence regularity and compressibility) over the binarised signal at each scale.
    The complexity is computed for each epoch, scale (window length), and channel independently.

    References
    ----------
    Ibáñez-Molina, A. J., Iglesias-Parro, S., Soriano, M. F., & Aznarte, J. I,
    Multiscale Lempel-Ziv complexity for EEG measures. Clinical Neurophysiology,
    (2015), 126(3), 541–548.

    Parameters
    ---------
    signal: numpy.ndarray
        Signal with shape [n_epochs, n_samples, n_channels]
    W: list[int] or numpy 1D array
        Sequence of odd integers representing the window lengths for multiscale binarisation.
        Each window defines a temporal scale for local median filtering.

    Returns
    -------
    result: numpy.ndarray
        Matrix containing Lempel-Ziv complexity values with shape [n_epochs, n_window_length, n_channels]

     Notes
    -----
    - All values in `W` must be odd to ensure proper median filtering.
    - The input signal is binarised using a local median threshold at each scale.
    - This function uses Python threading for parallel computation across channels.
    - The maximum window (`w_max`) is computed internally as the last value in `W` plus the
      spacing between the first two values, assuming uniform steps.

     Examples
    --------
    >>> import numpy as np
    >>> from medusa.signal_metrics.multiscale_lempelziv_complexity import multiscale_lempelziv_complexity

    >>> signal = np.random.randn(10, 1000, 2)
    >>> W = [5, 11, 21] # odd window lengths for multiscale binarisation
    >>> result = multiscale_lempelziv_complexity(signal, W)
    >>> print(result.shape)
    (10, 3, 2)  # 10 epochs, 3 scales, 2 channels

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


def __multiscale_median_threshold(signal, w_length):
    """ Apply median filtering to a signal for multiscale smoothing.

    This function performs temporal smoothing of each channel in the input signal by computing
    a moving median over a sliding window. For each sample, a window of length `w_length` is
    centered around it, and the median value of the window is assigned as the new value for that
    sample. This is used to reduce noise and extract relevant temporal structure for complexity
    analysis (e.g., Multiscale Lempel-Ziv Complexity).

    The output signal is shortened by `w_length - 1` samples due to edge effects from windowing.

    References
    ----------
    Ibáñez-Molina, A. J., Iglesias-Parro, S., Soriano, M. F., & Aznarte, J. I,
    Multiscale Lempel-Ziv complexity for EEG measures. Clinical Neurophysiology,
    (2015), 126(3), 541–548.

    Parameters
    ---------
    signal: numpy 2D array
        Input multichannel signal to be smoothed, with shape [n_samples, n_channels].
        Each column is treated independently.

    w_length: int
        Length of sliding window to calculate median values in smoothing process.
        Must be a positive odd integer to allow proper centering.

    Returns
    -------
    smoothed_signal: numpy 2D array
        Smoothed version with shape of [n_samples + 1 - w_length, n_channels]
        As a result of windowing, the final samples of the signal are lost.

    Notes
    -----
    - Only the central samples where the window fits fully within the signal are processed.
    - This function assumes that `w_length` is a valid odd integer; no internal check is performed.
    - Useful as preprocessing for complexity metrics like Lempel-Ziv after binarisation.

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

