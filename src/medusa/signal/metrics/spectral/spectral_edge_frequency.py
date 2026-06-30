import numpy as np
from numpy.typing import NDArray

from medusa.core.utils import check_data_dims

__all__ = ["spectral_edge_frequency"]


def spectral_edge_frequency(
        psd: NDArray,
        fs: float,
        percentile: float = 95.0,
        band: tuple[float, float] | None = None,
        eps: float = 1e-20) -> NDArray:
    """Compute Spectral Edge Frequency (SEF) of a PSD within a frequency band.

    The Spectral Edge Frequency at a given percentile ``p`` is the frequency
    below which ``p``% of the total power in the specified band lies. For
    example, SEF 50 is equivalent to the Median Frequency.

    Parameters
    ----------
    psd :
        (n_segments, n_frequencies, n_channels). Power Spectral Density of
        the signal. 2-D ``(n_frequencies, n_channels)`` and 1-D
        ``(n_frequencies,)`` inputs are promoted via
        :func:`medusa.core.utils.check_data_dims`; the inserted segment axis
        is squeezed back out of the result.
    fs :
        Sampling frequency in Hz. Used to construct the frequency axis,
        assuming the PSD spans the range ``[0, fs/2]``.
    percentile :
        Edge percentile in the range ``[0, 100]``. Typically 90 or 95.
    band :
        Frequency band limits ``(low_freq, high_freq)`` in Hz used for the
        calculation. If None, defaults to the full range ``(0, fs/2)``.
    eps :
        Small epsilon value to detect near-zero power and avoid validity
        issues.

    Returns
    -------
    NDArray
        (n_segments, n_channels). SEF value for each segment and channel.
        Contains ``NaN`` where the total power in the band is approximately
        0. The segment axis is dropped when a 2-D PSD is supplied.

    Raises
    ------
    ValueError
        If ``psd`` has an ndim incompatible with the ``'freq_segments'``
        representation (raised by :func:`medusa.core.utils.check_data_dims`),
        if ``band`` does not have exactly two elements, or if ``percentile``
        is not between 0 and 100.

    Examples
    --------
    >>> import numpy as np
    >>> from medusa.signal.metrics.spectral.spectral_edge_frequency import (
    ...     spectral_edge_frequency)
    >>> rng = np.random.default_rng(0)
    >>> # 1 segment, 129 frequency bins (0-64 Hz), 2 channels
    >>> psd = rng.random((1, 129, 2))
    >>> sef = spectral_edge_frequency(psd, fs=128.0, percentile=95.0)
    >>> sef.shape
    (1, 2)
    """
    psd, inserted = check_data_dims(psd, rep_type='freq_segments')

    if band is None:
        band = (0, fs / 2)

    band = np.asarray(band, dtype=float)

    # Checks
    if band.shape != (2,):
        raise ValueError("band must be (low_freq, high_freq).")
    if not (0.0 <= percentile <= 100.0):
        raise ValueError("percentile must be between 0 and 100.")

    # Frequency axis for a one-sided PSD
    freqs = np.linspace(0, fs / 2, psd.shape[1])

    # Band selection
    idx = (freqs >= band[0]) & (freqs < band[1])
    freqs_in_band = freqs[idx]
    psd_band = np.maximum(psd[:, idx, :], 0.0)  # ensure non-negative power

    # Total and cumulative power in band
    total_power = np.sum(psd_band, axis=1)  # (n_segments, n_channels)
    cum_power = np.cumsum(psd_band, axis=1)  # (n_segments, n_bins, n_channels)

    # Target cumulative power
    frac = percentile / 100.0
    target = frac * total_power  # (n_segments, n_channels)

    # Handle near-zero power
    valid = total_power > eps

    # First index where cumulative >= target
    ge = cum_power >= target[:, np.newaxis, :]  # broadcast target
    sef_idx = np.argmax(ge, axis=1)  # (n_segments, n_channels) (0 if all False)

    # If all False because total_power ~ 0, mark invalid
    sef = freqs_in_band[sef_idx]
    sef = np.where(valid, sef, np.nan)

    # Squeeze back the inserted segment axis (2-D streaming convention)
    sef = np.squeeze(sef, axis=0) if 0 in inserted else sef

    return sef
