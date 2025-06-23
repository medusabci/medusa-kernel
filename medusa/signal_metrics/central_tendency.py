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

def central_tendency_measure(signal, r):
    """
    This method implements the central tendency measure (CTM) of a time-series signal.

    CTM is a nonlinear metric used to quantify the variability of a signal. It is based on
    calculating a second-order difference plot  and counting the proportion of points that fall
    within a circular region of radius `r` centered at the origin.

    A higher CTM indicates lower signal variability (i.e., more regular behavior),
    while a lower CTM indicates greater variability.

    References
    ----------
    Cohen, M. E., Hudson, D. L., & Deedwania, P. C. (1996). Applying
    continuous chaotic modeling to cardiac signal analysis. IEEE Engineering in
    Medicine and Biology Magazine, 15(5), 97-102.

    Parameters
    ----------
    signal : numpy.ndarray
        Input signal with shape [n_epochs, n_samples, n_channels].
    r : double
        Radius used to compute the CTM. Should be a positive real number.

    Returns
    -------
    ctm : numpy.ndarray
        CTM values for each channel in "signal". [n_epochs, n_channels].

    Notes
    -----
    - The signal is first normalized to the range [-1, 1] based on a robust range
      (mean ± 3*std) per epoch and channel.
    - Second-order differences are computed as:
        x[n] = s[n] - s[n-1]
        y[n] = s[n+1] - s[n]
    - The Euclidean distance of (x[n], y[n]) is compared to `r`.

    Examples
    --------
    >>> import numpy as np
    >>> from medusa.central_tendency import central_tendency_measure

    >>> signal = np.random.randn(2, 1000, 1)  # 2 epochs, 1000 samples, 1 channel
    >>> r = 0.5
    >>> ctm_values = central_tendency_measure(signal, r)
    >>> print(ctm_values.shape)
    (2, 1)
    >>> print(ctm_values)
    [[0.43]
     [0.39]]
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