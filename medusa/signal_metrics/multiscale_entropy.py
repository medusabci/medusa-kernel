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
from medusa.signal_metrics.sample_entropy import sample_entropy

def multiscale_entropy(signal, max_scale, m, r):
    """
     Computes the Multiscale Entropy (MSE) of a signal.

    MSE is a method to quantify the complexity of time-series data across
    multiple temporal scales. This is accomplished through estimation of the Sample
    Entropy (SampEn) on coarse-grained versions of the original signal. As a
    result of the these calculations, MSE curves are obtained and can be used to
    compare the complexity of time-series. Higher SampEn values at most scales
    generally indicate more complex and less predictable signals.

    References
    ----------
    Costa, M., Goldberger, A. L., & Peng, C. K. (2005). Multiscale entropy
    analysis of biological signals, Physical review E, 71(2), 021906.

    Parameters
    ----------
    signal : numpy.ndarray
       Input signal with shape [n_epochs, n_samples, n_channels].
    max_scale : int
        Maximum scale factor. Entropy is computed from scale 1 to `max_scale`.
    m : int
         Embedding dimension (sequence length) used in the Sample Entropy.
    r : float
        Tolerance used in Sample Entropy, typically a fraction of the
        standard deviation

    Returns
    -------
    mse_result : numpy.ndarray
        Multiscale Entropy results. Each element corresponds to the SampEn
        of a coarse-grained signal at a given scale.
        Shape: [n_epochs, max_scale, n_channels].

    Examples
    --------
    >>> import numpy as np
    >>> from medusa.signal_metrics.multiscale_entropy import multiscale_entropy

    >>> signal = np.random.randn(3, 1000, 2)  # 3 epochs, 1000 samples, 2 channels
    >>> mse = multiscale_entropy(signal, max_scale=5, m=2, r=0.2)

    >>> print(mse.shape)
    (3, 5, 2)

    >>> print(mse[0, :, 0])  # MSE curve for first epoch, first channel
    [1.21 1.10 0.97 0.88 0.82]
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
    """
        Performs coarse-graining of a time series for Multiscale Entropy computation.

        Coarse-graining reduces the temporal resolution of a signal by averaging
        non-overlapping segments of length `scale`. This simulates lower-resolution
        representations of the signal for multiscale analysis.

        Parameters
        ----------
        signal : numpy.ndarray
            Original input signal with shape [n_epochs, n_samples, n_channels].

        scale : int
            Coarse-graining scale factor. Each coarse-grained time point is the average
            of `scale` consecutive samples.

        decimate_mode : bool, optional
            If True, applies fast decimation using `scipy.signal.decimate`.
            If False, performs standard averaging per segment.

        Returns
        -------
        y : numpy.ndarray
            Coarse-grained signal with shape [n_epochs, tau, n_channels], where
            tau = floor(n_samples / scale).

        Examples
        --------
        >>> from medusa.signal_metrics.multiscale_entropy import __coarse_grain
        >>> signal = np.random.randn(1, 1000, 1)
        >>> y = __coarse_grain(signal, scale=5, decimate_mode=False)
        >>> print(y.shape)
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