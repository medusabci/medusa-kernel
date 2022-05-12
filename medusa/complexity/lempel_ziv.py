import numpy as np
import threading


def multiscale_lempelziv(signal, W):
    """
    Calculate the multiscale signal binarisation and the
    Multiscale Lempel-Ziv's complexity. This version allows the algorithm to be used
    for signals with more than one channel at the same time. Based on the implementation
    of Ibañez et al 2015 [1].

    Parameters
    ---------
    signal: numpy.ndarray
        Signal with shape [n_samples x n_channels]
    W: list or numpy.ndarray
        Set of window length to consider at multiscale binarisation stage. Values must be odd.

    Returns
    -------
    result: numpy.ndarray
        Matrix of results with shape [n_window_length, n_channels]

    References
    ----------
    [1] Ibáñez-Molina, A. J., Iglesias-Parro, S., Soriano, M. F., & Aznarte, J. I. (2015).
        Multiscale Lempel-Ziv complexity for EEG measures. Clinical Neurophysiology, 126(3), 541–548.
        https://doi.org/10.1016/j.clinph.2014.07.012
    """
    # Useful parameter
    w_max = W[-1] + (W[1]- W[0])

    # Adds a dimension on one-channel signals
    if signal.size == len(signal):
        signal = signal[:, np.newaxis]

    # Define a matrix to store results
    result = np.empty((len(W), signal.shape[1]))

    # First get binarised signal
    for w_idx, w in enumerate(W):
        binarised_signal = binarisation(signal, w, w_max,multiscale= True)

    # Parallelize the calculations if n_channel > 1
        if binarised_signal.shape[1] > 1:
            threads = []
            for ch in range(binarised_signal.shape[1]):
                t = threading.Thread(target=lz_algorithm, args=(binarised_signal[:, ch],
                                                                result, ch, w_idx))
                threads.append(t)
                t.start()
            for t in threads:
                t.join()
        else:
            result[w_idx,:] = lz_algorithm(binarised_signal,w_idx = w_idx)
    return result


def lempelziv(signal):
    """
    Calculate the  signal binarisation and the Lempel-Ziv's complexity. This version allows
    the algorithm to be used for signals with more than one channel at the same time.

    Parameters
    ---------
    signal: numpy.ndarray
        Signal with shape [n_samples x n_channels]

    Returns
    -------
    lz_channel_values: numpy.ndarray
        If n_channels = 1, function returns the lz_algorithm value. Else, results from algorithm
        of each channel are stored and returned at lz_channel_values. Shape of [n_channels]
    """

    if signal.size == len(signal):
        signal = signal[:, np.newaxis]

    signal = binarisation(signal)
    if signal.shape[1] == 1:
        return lz_algorithm(signal)
    else:
        lz_channel_values = np.empty(signal.shape[1])
        working_threads = list()
        for ch in range(signal.shape[1]):
            t = threading.Thread(target=lz_algorithm, args=(signal[:, ch], lz_channel_values, ch))
            working_threads.append(t)
            t.start()
        for thread in working_threads:
            thread.join()
        return lz_channel_values


def lz_algorithm(signal, result=None, ch_idx=None, w_idx=None):
    """
    Lempel-Ziv's complexity algorithm. This version can be called by lempelziv and
    multiscale_lempelziv functions. Based on Kaspar et al 1987 [1].

    Parameters
    ---------
    signal: numpy.ndarray
        Siganl with shape of [n_samples]
    result: numpy.ndarray or None
        Empty array to store results when threading
    ch_idx: int or None
        Index to identify the channel the signal corresponds to
    w_idx: int or None
        Index to identify the window length  the signal corresponds to

    Returns
    -------
    value: float
        Result of algorithm calculations. If threadinng at main function, the value
        is stored at result matrix

    References
    ----------
    [1]F. Kaspar, H. G. Schuster, "Easily-calculable measure for the
       complexity of spatiotemporal patterns", Physical Review A, Volume 36,
       Number 2 (1987).
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

    # If threading, value is stored at result matrix
    if result is not None:
        if ch_idx is not None and w_idx is None:
            result[ch_idx] = value
        elif ch_idx is None and w_idx is not None:
            result[w_idx,:] = value
        elif ch_idx is not None and w_idx is not None:
            result[w_idx, ch_idx] = value
    return value


def binarisation(signal, w_length=None, w_max=None, multiscale=False):
    """
    This function returns a binarised version of original signal. It can be used in both
    multiscale and simple binarisations. If multiscale mode is chosen, signals are shortened
    so that they all have the same length, taking the maximum window as a reference.
    Binarisation is performed by means of a median-based comparison.

    Parameters
    ----------
    signal: numpy.ndarray
        Signal with shape [n_samples x n_channels]
    w_length: int or None
        Window length to perform multiscale binarisation
    w_max: int or None
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
        if w_length % 2 == 0:
            raise Exception('Width of windows must be an odd value.')

        #  Get smoothed version from original signal
        smoothed = multiscale_median_threshold(signal, w_length)

        # Useful parameters
        half_wind = int((w_length - 1) / 2)
        max_length = signal.shape[0] + 1 - w_max
        length_diff = smoothed.shape[0] - max_length

        # Shorten original and smoothed version
        smoothed_shortened = smoothed[int(length_diff / 2):-int(length_diff / 2), :]
        signal_shortened = signal[half_wind: signal.shape[0] - half_wind, :]
        signal_shortened = signal_shortened[int(length_diff / 2):-int(length_diff / 2), :]

        # Define template of binarised signal
        signal_binarised = np.zeros((signal_shortened.shape[0], signal_shortened.shape[1]))

        # Binarise the signal
        idx_one = signal_shortened >= smoothed_shortened
        signal_binarised[idx_one] = 1

        return signal_binarised
    else:
        signal_binarised = np.zeros((len(signal), signal.shape[1]))
        median = np.median(signal, axis=0)
        idx_one = signal >= median
        signal_binarised[np.squeeze(idx_one)] = 1

        return signal_binarised


def multiscale_median_threshold(signal, w_length):
    """
    Signal smoothing function. For each sample, we define a window in which the sample is
    in centre position. The median value of the window is calculated and assigned to this
    sample to obtain a smoothed version of the original singal. Based on the implementation
    of Ibañez et al 2015 [1].

    Parameters
    ---------
    signal: numpy.ndarray
        Signal with shape of [n_samples x n_channels]
    w_length: int
        Length of window to calculate median values in smoothing process

    Returns
    -------
    smoothed_signal: numpy.ndarray
        Smoothed version with shape of [n_samples + 1 - w_length, n_channels]
        As a result of windowing, signal end samples are lost

    References
    ----------
    [1] Ibáñez-Molina, A. J., Iglesias-Parro, S., Soriano, M. F., & Aznarte, J. I. (2015).
        Multiscale Lempel-Ziv complexity for EEG measures. Clinical Neurophysiology, 126(3), 541–548.
        https://doi.org/10.1016/j.clinph.2014.07.012
    """
    # Template of smoothed signal
    smoothed_signal = np.zeros((signal.shape[0] + 1 - w_length, signal.shape[1]))

    half_wind = int((w_length - 1) / 2)

    # Index of sample to be smoothed from median window value
    index = 0

    # We define a window with samp in central position and
    # get median value to smooth original signal
    for samp in range(half_wind, signal.shape[0] - half_wind):
        smoothed_signal[index, :] = np.median(signal[samp - half_wind: samp + half_wind], axis=0)
        index += 1
    return smoothed_signal


if __name__ == '__main__':
    import time
    import scipy.signal as ss

    t = np.linspace(0, 100, 6000)
    signal = np.empty((6000,4))
    signal[:,0] = ss.chirp(t, f0=6, f1=16, t1=100, method='linear')
    signal[:, 1] = ss.chirp(t, f0=8, f1=28, t1=80, method='linear')
    signal[:, 2] = ss.chirp(t, f0=1, f1=50, t1=20, method='linear')
    signal[:, 3] = ss.chirp(t, f0=6, f1=60, t1=100, method='linear')
    start = time.time()
    mse = multiscale_lempelziv(signal=signal,W =np.arange(11,101,10))
    end = time.time()
    print(end - start)
