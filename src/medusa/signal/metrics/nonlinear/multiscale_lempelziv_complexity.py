# Built-in imports
from collections.abc import Sequence

# External imports
import numpy as np
from numpy.typing import NDArray

# Medusa imports
from medusa.core.utils import ThreadWithReturnValue, check_data_dims
from medusa.signal.metrics.nonlinear._lz_core import (
    _binarisation,
    _lz_algorithm,
)

__all__ = ["multiscale_lempelziv_complexity"]


def multiscale_lempelziv_complexity(
        signal: NDArray, windows: Sequence[int] | NDArray) -> NDArray:
    """Multiscale Lempel-Ziv complexity (MSLZC) of a signal.

    This function applies multiscale binarisation to each segment and channel
    of the input signal using a set of predefined window lengths. It then
    calculates the Lempel-Ziv complexity (a measure of sequence regularity and
    compressibility) over the binarised signal at each scale. The complexity
    is computed for each segment, scale (window length), and channel
    independently.

    Parameters
    ----------
    signal :
        (n_segments, n_samples, n_channels). Input time-series signal. A 2-D
        ``(n_samples, n_channels)`` array is accepted and promoted; the
        leading segment axis is squeezed back out of the result.
    windows :
        Sequence of odd integers representing the window lengths for
        multiscale binarisation. Each window defines a temporal scale for
        local median filtering.

    Returns
    -------
    NDArray
        (n_segments, n_windows, n_channels). Lempel-Ziv complexity for each
        segment, window length, and channel. For a 2-D input the segment axis
        is dropped, yielding ``(n_windows, n_channels)``.

    References
    ----------
    Ibáñez-Molina, A. J., Iglesias-Parro, S., Soriano, M. F., & Aznarte, J. I,
    Multiscale Lempel-Ziv complexity for EEG measures. Clinical Neurophysiology,
    (2015), 126(3), 541–548.

    Notes
    -----
    - All values in ``windows`` must be odd to ensure proper median filtering.
    - The input signal is binarised using a local median threshold at each
      scale.
    - This function uses Python threading for parallel computation across
      channels.
    - The maximum window (``w_max``) is computed internally as the last value
      in ``windows`` plus the spacing between the first two values, assuming
      uniform steps.

    Examples
    --------
    >>> import numpy as np
    >>> from medusa.signal.metrics.nonlinear.multiscale_lempelziv_complexity \
    ...     import multiscale_lempelziv_complexity
    >>> signal = np.random.randn(10, 1000, 2)  # 10 segments, samples, 2 ch
    >>> windows = [5, 11, 21]  # odd window lengths for multiscale binarisation
    >>> result = multiscale_lempelziv_complexity(signal, windows)
    >>> result.shape
    (10, 3, 2)
    """
    # Check dimensions
    signal, inserted = check_data_dims(signal, rep_type='time_segments')

    # Signal dimensions
    n_epo = signal.shape[0]
    n_cha = signal.shape[2]

    # Useful parameter
    w_max = windows[-1] + (windows[1] - windows[0])

    # Define a matrix to store results
    result = np.full((n_epo, len(windows), n_cha), np.nan)

    # First get binarised signal
    for ep_idx, epoch in enumerate(signal):
        for w_idx, w in enumerate(windows):
            binarised_signal = _binarisation(epoch, w, w_max, multiscale=True)

            # Parallelize the calculations if n_channel > 1
            if binarised_signal.shape[1] > 1:
                threads = []
                for ch in range(binarised_signal.shape[1]):
                    t = ThreadWithReturnValue(target=_lz_algorithm,
                                              args=(binarised_signal[:, ch],))
                    threads.append(t)
                    t.start()
                for ch_idx, t in enumerate(threads):
                    result[ep_idx, w_idx, ch_idx] = t.join()
            else:
                result[ep_idx, w_idx, :] = _lz_algorithm(binarised_signal)

    # Squeeze back the inserted segment axis so a 2-D caller gets a
    # de-segmented result
    if 0 in inserted:
        result = np.squeeze(result, axis=0)
    return result
