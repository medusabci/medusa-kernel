# Built-in imports

# External imports
import numpy as np
from numpy.typing import NDArray

# Medusa imports
from medusa.signal.metrics.nonlinear._lz_core import (
    _binarisation,
    _lz_algorithm,
)

__all__ = ["lempelziv_complexity"]


def lempelziv_complexity(signal: NDArray) -> NDArray:
    """Lempel-Ziv complexity (LZC) of a signal.

    This function first binarizes the input signal and then calculates its
    Lempel-Ziv complexity (LZC), a nonlinear measure of signal regularity and
    compressibility. It supports multi-channel inputs and computes LZC
    independently for each channel.

    Parameters
    ----------
    signal :
        (n_segments, n_samples, n_channels) or (n_samples, n_channels). Input
        time-series signal.

    Returns
    -------
    NDArray
        Lempel-Ziv complexity. For a 3-D input the shape is
        ``(n_segments, n_channels)``; for a 2-D input it is ``(n_channels,)``.

    Raises
    ------
    ValueError
        If ``signal`` is neither 2-D ``(n_samples, n_channels)`` nor 3-D
        ``(n_segments, n_samples, n_channels)``.

    Notes
    -----
    - The input signal is first binarized using the median thresholding method
      ``_binarisation()``.
    - The complexity is computed via ``_lz_algorithm()`` for each
      segment/channel independently.

    Examples
    --------
    >>> import numpy as np
    >>> from medusa.signal.metrics.nonlinear.lempelziv_complexity import (
    ...     lempelziv_complexity,
    ... )
    >>> signal = np.random.randn(1000, 3)  # 1000 samples, 3 channels
    >>> lzc = lempelziv_complexity(signal)
    >>> lzc.shape
    (3,)
    >>> signal = np.random.randn(5, 1000, 3)  # 5 segments, samples, channels
    >>> lzc = lempelziv_complexity(signal)
    >>> lzc.shape
    (5, 3)
    """

    signal = np.asarray(signal)

    if signal.ndim == 2:
        # Single epoch: [samples, channels]
        signal = _binarisation(signal, [signal.shape[0]], signal.shape[0])
        if signal.shape[1] == 1:
            return _lz_algorithm(signal)
        else:
            return np.array([
                _lz_algorithm(signal[:, ch])
                for ch in range(signal.shape[1])
            ])

    elif signal.ndim == 3:
        # Multi-epoch: [epochs, samples, channels]
        n_epochs, _, n_channels = signal.shape
        lz_output = np.zeros((n_epochs, n_channels))
        for ep in range(n_epochs):
            bin_signal = _binarisation(signal[ep], [signal.shape[1]], signal.shape[1])
            for ch in range(n_channels):
                lz_output[ep, ch] = _lz_algorithm(bin_signal[:, ch])
        return lz_output

    else:
        raise ValueError(
            "Signal shape not recognized. Expected shape [samples, channels] or [epochs, samples, channels].")
