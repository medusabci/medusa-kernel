import numpy as np
from numpy.typing import NDArray

from medusa.core.utils import check_data_dims

__all__ = ["shannon_spectral_entropy"]


def shannon_spectral_entropy(psd: NDArray, fs: float,
                             band: tuple[float, float] = (1, 70)) -> NDArray:
    """Normalized Shannon spectral entropy (SE) of a power spectral density.

    The PSD is first restricted to the selected frequency band and normalized
    to obtain a probability density function (PDF) across frequency bins. The
    Shannon entropy is then computed as::

        SE = -sum(p_i * log(p_i)) / log(N)

    where ``p_i`` are the normalized spectral coefficients and ``N`` is the
    number of frequency bins in the selected band. The entropy is normalized
    to the interval [0, 1].

    Zero-probability bins are handled according to the convention
    ``0 * log(0) = 0``. If the total power in the selected band is zero for a
    given segment and channel, the entropy is set to 0 (i.e. undefined cases
    are treated as zero entropy).

    Parameters
    ----------
    psd :
        (n_segments, n_frequencies, n_channels). Power spectral density. A 2-D
        ``(n_frequencies, n_channels)`` array is accepted and promoted; the
        leading segment axis is squeezed back out of the result.
    fs :
        Sampling frequency of the original signal (Hz).
    band :
        Frequency band ``[low_freq, high_freq)`` in Hz where the spectral
        entropy is computed. Default is ``(1, 70)``.

    Returns
    -------
    NDArray
        (n_segments, n_channels). Normalized Shannon spectral entropy. Values
        range between 0 (spectrally concentrated) and 1 (spectrally uniform).
        For a 2-D input the segment axis is dropped, yielding
        ``(n_channels,)``.

    Raises
    ------
    ValueError
        If ``psd`` does not have an acceptable number of dimensions for
        ``rep_type='freq_segments'``, or if ``band`` is not a length-2
        sequence ``[low_freq, high_freq]``.

    Examples
    --------
    >>> import numpy as np
    >>> from medusa.signal.metrics.nonlinear.shannon_spectral_entropy import (
    ...     shannon_spectral_entropy,
    ... )
    >>> fs = 256
    >>> psd = np.random.rand(2, 129, 3)  # 2 segments, 129 freqs, 3 channels
    >>> se = shannon_spectral_entropy(psd, fs, band=(4, 30))
    >>> se.shape
    (2, 3)
    """

    # Check dimensions
    psd, inserted = check_data_dims(psd, rep_type='freq_segments')
    band = np.asarray(band)

    # Check errors
    if band.ndim != 1 or band.shape[0] != 2:
        raise ValueError('Parameter band must be an array with the desired '
                         'band. E.g., Delta: [0, 4]')

    # Calculate freqs array
    freqs = np.linspace(0, fs / 2, psd.shape[1], endpoint=True)
    idx = (freqs >= band[0]) & (freqs < band[1])

    # Calculate total power
    band_psd = np.abs(psd[:, idx, :])  # [segments, bins, ch]
    total_power = np.sum(band_psd, axis=1, keepdims=True)  # [segments, 1, ch]

    # Avoid zero divisions in probability density function
    with np.errstate(divide="ignore", invalid="ignore"):
        pdf = np.divide(band_psd, total_power, out=np.zeros_like(band_psd),
                        where=total_power > 0)

    # Shannon: sum_{i: p_i>0} p_i log p_i
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.zeros_like(pdf)
        np.log(pdf, out=logp, where=pdf > 0)
    se = -np.sum(pdf * logp, axis=1)

    # Normalization
    n_bins = pdf.shape[1]
    if n_bins > 1:
        se = se / np.log(n_bins)
    else:
        se = np.zeros((psd.shape[0], psd.shape[2]), dtype=psd.dtype)

    # If total_power==0, entropy is not defined, so here it's set to 0
    se = np.where(np.squeeze(total_power, axis=1) > 0, se, 0.0)

    # Squeeze back the inserted segment axis so a 2-D caller gets a
    # de-segmented result
    if 0 in inserted:
        se = np.squeeze(se, axis=0)

    return se
