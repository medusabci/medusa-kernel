from typing import Literal

import numpy as np
from numpy.typing import NDArray

from medusa.core.utils import check_data_dims
from medusa.signal.transforms import normalize_psd

__all__ = ["band_power"]


def band_power(
        psd: NDArray,
        fs: float,
        band: tuple[float, float],
        power_type: Literal['absolute', 'relative'] = 'absolute',
        norm_range: tuple[float, float] | None = None) -> NDArray:
    """Compute the power of a signal within a frequency band from its PSD.

    The power is obtained by integrating the Power Spectral Density (PSD)
    over the requested frequency band. Pass the raw (un-normalised) PSD: the
    relative power mode performs its own normalisation internally.

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
    band :
        Frequency band limits ``(low_freq, high_freq)`` in Hz over which the
        power is computed. E.g. ``(8, 13)`` for the alpha band.
    power_type :
        ``'absolute'`` for the integrated power (normalised so it is
        comparable across sampling frequencies and PSD lengths), or
        ``'relative'`` for the fraction of power within ``band`` relative to
        the ``norm_range`` band.
    norm_range :
        Frequency range ``(low_freq, high_freq)`` in Hz used as the reference
        band for relative power. Required when ``power_type='relative'`` and
        ignored otherwise.

    Returns
    -------
    NDArray
        (n_segments, n_channels). Power within ``band`` for each segment and
        channel. The segment axis is dropped when a 2-D PSD is supplied.

    Raises
    ------
    ValueError
        If ``power_type`` is not ``'absolute'`` or ``'relative'``, if
        ``power_type='relative'`` and ``norm_range`` is not a two-number
        sequence, or if ``psd`` has an ndim incompatible with the
        ``'freq_segments'`` representation.

    Examples
    --------
    >>> import numpy as np
    >>> from medusa.signal.metrics.spectral.band_power import band_power
    >>> rng = np.random.default_rng(0)
    >>> # 1 segment, 129 frequency bins, 2 channels
    >>> psd = rng.random((1, 129, 2))
    >>> power = band_power(psd, fs=256.0, band=(8, 13))
    >>> power.shape
    (1, 2)
    """
    # Check errors
    if power_type not in ('absolute', 'relative'):
        raise ValueError("power_type must be 'absolute' or 'relative'.")

    psd, inserted = check_data_dims(psd, rep_type='freq_segments')

    # Frequency axis for a one-sided PSD
    freqs = np.linspace(0, fs / 2, psd.shape[1])

    if power_type == 'relative':
        if not (isinstance(norm_range, (list, tuple)) and
                len(norm_range) == 2 and
                all(isinstance(x, (int, float)) for x in norm_range)):
            raise ValueError(
                "norm_range must be a list or two-number tuple, "
                "such as [1, 70]."
            )
        psd = normalize_psd(
            psd, (norm_range[0], norm_range[1]), freqs, norm='rel')

    # Power within the target band
    in_band = np.logical_and(freqs >= band[0], freqs <= band[1])
    powers = np.sum(psd[:, in_band, :], axis=1)

    if power_type == 'absolute':
        # Normalise to make it comparable across different sampling
        # frequencies and PSD lengths.
        powers = powers * (fs / (2 * freqs.shape[0]))

    # Squeeze back the inserted segment axis (2-D streaming convention)
    powers = np.squeeze(powers, axis=0) if 0 in inserted else powers

    return powers
