from numpy.typing import NDArray

from .spectral_edge_frequency import spectral_edge_frequency

__all__ = ["median_frequency"]


def median_frequency(
        psd: NDArray,
        fs: float,
        band: tuple[float, float] | None = None,
        eps: float = 1e-20) -> NDArray:
    """Compute the Median Frequency (MF) of a PSD within a frequency band.

    This function calculates the frequency below which 50% of the total power
    in the specified band lies. It is a convenience wrapper around
    :func:`spectral_edge_frequency` with ``percentile=50``.

    Parameters
    ----------
    psd :
        (n_segments, n_frequencies, n_channels). Power Spectral Density of
        the signal. 2-D ``(n_frequencies, n_channels)`` and 1-D
        ``(n_frequencies,)`` inputs are promoted and de-segmented by
        :func:`spectral_edge_frequency`.
    fs :
        Sampling frequency in Hz. Used to construct the frequency axis,
        assuming the PSD spans the range ``[0, fs/2]``.
    band :
        Frequency band limits ``(low_freq, high_freq)`` in Hz used for the
        calculation. If None, defaults to the full range ``(0, fs/2)``.
    eps :
        Small epsilon value passed to :func:`spectral_edge_frequency` to
        detect near-zero power.

    Returns
    -------
    NDArray
        (n_segments, n_channels). Median Frequency for each segment and
        channel. The segment axis is dropped when a 2-D PSD is supplied.

    See Also
    --------
    spectral_edge_frequency : Function used internally with percentile=50.

    Examples
    --------
    >>> import numpy as np
    >>> from medusa.signal.metrics.spectral.median_frequency import (
    ...     median_frequency)
    >>> rng = np.random.default_rng(0)
    >>> # 10 segments, 129 frequency bins, 16 channels
    >>> psd = rng.random((10, 129, 16))
    >>> mf = median_frequency(psd, fs=256.0, band=(1, 50))
    >>> mf.shape
    (10, 16)
    """
    return spectral_edge_frequency(
        psd,
        fs,
        percentile=50,
        band=band,
        eps=eps,
    )
