# Built-in imports

# External imports
import numpy as np
from numpy.typing import NDArray
from scipy.signal import decimate

# Medusa imports
from medusa.core.utils import ThreadWithReturnValue, check_data_dims
from medusa.signal.metrics.nonlinear.sample_entropy import sample_entropy

__all__ = ["multiscale_entropy"]


def multiscale_entropy(signal: NDArray, max_scale: int, m: int,
                       r: float) -> NDArray:
    """Multiscale entropy (MSE) of a signal.

    MSE quantifies the complexity of time-series data across multiple temporal
    scales. This is accomplished by estimating the sample entropy (SampEn) on
    coarse-grained versions of the original signal. The resulting MSE curves
    can be used to compare the complexity of time series: higher SampEn values
    at most scales generally indicate more complex and less predictable
    signals.

    Parameters
    ----------
    signal :
        (n_segments, n_samples, n_channels). Input time-series signal. A 2-D
        ``(n_samples, n_channels)`` array is accepted and promoted; the
        leading segment axis is squeezed back out of the result.
    max_scale :
        Maximum scale factor. Entropy is computed from scale 1 to
        ``max_scale``.
    m :
        Embedding dimension (sequence length) used in the sample entropy.
    r :
        Tolerance used in sample entropy, typically a fraction of the standard
        deviation.

    Returns
    -------
    NDArray
        (n_segments, max_scale, n_channels). Multiscale entropy results. Each
        element is the SampEn of a coarse-grained signal at a given scale. For
        a 2-D input the segment axis is dropped, yielding
        ``(max_scale, n_channels)``.

    References
    ----------
    Costa, M., Goldberger, A. L., & Peng, C. K. (2005). Multiscale entropy
    analysis of biological signals, Physical review E, 71(2), 021906.

    Examples
    --------
    >>> import numpy as np
    >>> from medusa.signal.metrics.nonlinear.multiscale_entropy import (
    ...     multiscale_entropy,
    ... )
    >>> signal = np.random.randn(3, 1000, 2)  # 3 segments, 1000 samples, 2 ch
    >>> mse = multiscale_entropy(signal, max_scale=5, m=2, r=0.2)
    >>> mse.shape
    (3, 5, 2)
    """

    # Check dimensions
    signal, inserted = check_data_dims(signal, rep_type='time_segments')

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

    # Squeeze back the inserted segment axis so a 2-D caller gets a
    # de-segmented result
    if 0 in inserted:
        mse_result = np.squeeze(mse_result, axis=0)
    return mse_result


def __coarse_grain(signal: NDArray, scale: int,
                   decimate_mode: bool = True) -> NDArray:
    """Coarse-grain a time series for Multiscale Entropy computation.

    Coarse-graining reduces the temporal resolution of a signal by averaging
    non-overlapping segments of length ``scale``. This simulates
    lower-resolution representations of the signal for multiscale analysis.

    Parameters
    ----------
    signal
        (n_segments, n_samples, n_channels). Original input signal.
    scale
        Coarse-graining scale factor. Each coarse-grained time point is the
        average of ``scale`` consecutive samples.
    decimate_mode
        If True, applies fast decimation using :func:`scipy.signal.decimate`.
        If False, performs standard averaging per segment.

    Returns
    -------
    numpy.ndarray
        (n_segments, tau, n_channels) coarse-grained signal, where
        ``tau = floor(n_samples / scale)``.

    Examples
    --------
    >>> import numpy as np
    >>> from medusa.signal.metrics.nonlinear.multiscale_entropy import (
    ...     __coarse_grain,
    ... )
    >>> signal = np.random.randn(1, 1000, 1)
    >>> y = __coarse_grain(signal, scale=5, decimate_mode=False)
    >>> y.shape
    (1, 200, 1)
    """
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
