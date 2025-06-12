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
    dll_file = os.path.join(os.path.dirname(__file__),
                        'computeLZC.dll')
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

