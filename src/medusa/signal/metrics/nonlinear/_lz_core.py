"""Shared Lempel-Ziv complexity primitives.

Private helpers used by both :mod:`lempelziv_complexity` and
:mod:`multiscale_lempelziv_complexity`. Previously each public module carried
its own copy of these functions; the two ``_lz_algorithm`` copies had silently
diverged (one lacked the out-of-bounds / empty-sequence guards), so they could
disagree on degenerate inputs. They are unified here on the guarded variant,
which is numerically identical to the unguarded one for every sequence of
length >= 2 and additionally returns a finite value (instead of raising
``IndexError``) for length-0/1 sequences.
"""

# External imports
import numpy as np
from numpy.typing import NDArray


def _lz_algorithm(signal: NDArray) -> float:
    """Lempel-Ziv complexity (LZC) of a 1-D binary sequence.

    Implements the Kaspar-Schuster (1987) substring-parsing scheme and returns
    the normalised complexity ``c * log2(n) / n``.

    References
    ----------
    F. Kaspar, H. G. Schuster, "Easily-calculable measure for the complexity of
    spatiotemporal patterns", Physical Review A, 36(2), 1987.

    Parameters
    ----------
    signal
        1-D (or flattenable) binary sequence of 0s and 1s.

    Returns
    -------
    float
        Normalised Lempel-Ziv complexity. ``nan`` for an empty sequence.
    """
    signal = signal.flatten().tolist()

    if len(signal) == 0:
        return np.nan

    i, k, j = 0, 1, 1
    c, k_max = 1, 1
    n = len(signal)

    while True:
        if j + k > n or i + k > n:
            c = c + 1
            break
        if signal[i + k - 1] == signal[j + k - 1]:
            k += 1
        else:
            if k > k_max:
                k_max = k
            i += 1
            if i == j:
                c += 1
                j += k_max
                if j + 1 > n:
                    break
                else:
                    i, k, k_max = 0, 1, 1
            else:
                k = 1

    value = c * (np.log2(n) / n) if n > 0 else np.nan
    return value


def _multiscale_median_threshold(signal: NDArray, w_length: int) -> NDArray:
    """Centred sliding-window moving median used as the multiscale baseline.

    For each sample a window of ``w_length`` (odd) samples is centred on it and
    the per-channel median assigned to the output. The result is shortened by
    ``w_length - 1`` samples due to edge effects.

    Parameters
    ----------
    signal
        Signal with shape ``[n_samples, n_channels]``.
    w_length
        Odd window length.

    Returns
    -------
    numpy.ndarray
        Smoothed signal with shape ``[n_samples + 1 - w_length, n_channels]``.
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
            signal[samp - half_wind: samp + half_wind + 1], axis=0)
        index += 1
    return smoothed_signal


def _binarisation(signal: NDArray, w_length: int, w_max: int,
                  multiscale: bool = False) -> NDArray:
    """Binarise a multichannel signal by a median comparison.

    In the simple mode each channel is thresholded at its global median
    (``>=`` maps to 1). In multiscale mode a local (sliding-window) median
    baseline is used and signals are trimmed so every scale yields the same
    length, taking ``w_max`` as reference.

    Parameters
    ----------
    signal
        Signal with shape ``[n_samples, n_channels]``.
    w_length
        Window length for the median filter (odd). Used only in multiscale mode.
    w_max
        Maximum window length for alignment in multiscale mode.
    multiscale
        If True, use the local-median (multiscale) baseline; otherwise the
        global median.

    Returns
    -------
    numpy.ndarray
        Binarised signal with shape ``[n_samples_shortened, n_channels]``.

    Raises
    ------
    ValueError
        If ``multiscale`` is True and ``w_length`` is None or even, or
        ``w_max`` is None.
    """
    if multiscale:
        if w_length is None:
            raise ValueError('Width of window must be an integer value')
        if w_length % 2 == 0:
            raise ValueError('Width of window must be an odd value.')
        if w_max is None:
            raise ValueError('Maximum window width must be an integer value')

        #  Get smoothed version from original signal
        smoothed = _multiscale_median_threshold(signal, w_length)

        # Useful parameters
        half_wind = int((w_length - 1) / 2)
        max_length = signal.shape[0] + 1 - w_max
        length_diff = smoothed.shape[0] - max_length

        # Shorten original and smoothed version
        start = int(length_diff / 2)
        end = -int(length_diff / 2) if int(length_diff / 2) > 0 else None

        smoothed_shortened = smoothed[start:end, :]
        signal_shortened = signal[half_wind: signal.shape[0] - half_wind, :]
        signal_shortened = signal_shortened[start:end, :]

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
