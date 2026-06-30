# Built-in imports

# External imports
import numpy as np
from numpy.typing import NDArray

# Medusa imports
from medusa.core.utils import check_data_dims

__all__ = ["central_tendency_measure"]


def central_tendency_measure(signal: NDArray, r: float) -> NDArray:
    """Central tendency measure (CTM) of a time-series signal.

    CTM is a nonlinear metric used to quantify the variability of a signal. It
    is based on a second-order difference plot and counts the proportion of
    points that fall within a circular region of radius ``r`` centred at the
    origin.

    A higher CTM indicates lower signal variability (i.e. more regular
    behaviour), while a lower CTM indicates greater variability.

    Parameters
    ----------
    signal :
        (n_segments, n_samples, n_channels). Input time-series signal. A 2-D
        ``(n_samples, n_channels)`` array is accepted and promoted; the
        leading segment axis is squeezed back out of the result.
    r :
        Radius used to compute the CTM. Should be a positive real number.

    Returns
    -------
    NDArray
        (n_segments, n_channels). CTM value for each segment and channel. For
        a 2-D input the segment axis is dropped, yielding ``(n_channels,)``.

    Raises
    ------
    ValueError
        If ``signal`` contains non-numeric values.

    References
    ----------
    Cohen, M. E., Hudson, D. L., & Deedwania, P. C. (1996). Applying
    continuous chaotic modeling to cardiac signal analysis. IEEE Engineering in
    Medicine and Biology Magazine, 15(5), 97-102.

    Notes
    -----
    - The signal is first normalized to the range [-1, 1] based on a robust
      range (mean ± 3*std) per segment and channel.
    - Second-order differences are computed as:
        x[n] = s[n] - s[n-1]
        y[n] = s[n+1] - s[n]
    - The Euclidean distance of (x[n], y[n]) is compared to ``r``.

    Examples
    --------
    >>> import numpy as np
    >>> from medusa.signal.metrics.nonlinear.central_tendency import (
    ...     central_tendency_measure,
    ... )
    >>> signal = np.random.randn(2, 1000, 1)  # 2 segments, 1000 samples, 1 ch
    >>> ctm = central_tendency_measure(signal, r=0.5)
    >>> ctm.shape
    (2, 1)
    """

    # Check dimensions (also coerces list-like input via numpy.asarray)
    signal, inserted = check_data_dims(signal, rep_type='time_segments')

    #  Error check
    if not np.issubdtype(signal.dtype, np.number):
        raise ValueError('data matrix contains non-numeric values')

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
            val = signal[ep, idx_within_range[ep, :, ch], ch]
            if len(val) > 0:
                max_value[ep, ch] = np.max(np.abs(val), axis=0)
            else:
                max_value[ep, ch] = np.nan

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

    # Squeeze back the inserted segment axis so a 2-D caller gets a
    # de-segmented result
    if 0 in inserted:
        ctm = np.squeeze(ctm, axis=0)

    return ctm
