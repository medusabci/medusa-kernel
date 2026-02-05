import numpy as np
from medusa.utils import check_dimensions

def spectral_edge_frequency(
        psd, fs,
        percentile=95.0,
        target_band=None,
        eps=1e-20):
    """
    Compute Spectral Edge Frequency (SEF) of a PSD within a frequency band.

    The Spectral Edge Frequency (SEF) at a given percentile `p` is the frequency
    below which `p`% of the total power in the specified band lies.
    For example, SEF 50 is equivalent to the Median Frequency.

    Parameters
    ----------
    psd : numpy.ndarray
        Power Spectral Density array with shape ``[n_epochs, n_freqs, n_channels]``.
        If dimensions are missing, use `np.newaxis` to expand them (e.g.,
        ``[n_freqs] -> [1, n_freqs, 1]``).
    fs : int or float
        Sampling frequency in Hz. Used to construct the frequency axis, assuming
        the PSD corresponds to the range ``[0, fs/2]``.
    percentile : float, optional
        Edge percentile in the range ``[0, 100]``. Typically 90 or 95.
        Defaults to 95.0.
    target_band : tuple or list, optional
        Frequency band limits ``(low_freq, high_freq)`` in Hz used for the
        calculation. If None, it defaults to the full range ``(0, fs/2)``.
    eps : float, optional
        Small epsilon value to detect near-zero power and avoid validity issues.
        Defaults to 1e-20.

    Returns
    -------
    sef : numpy.ndarray
        SEF values for each epoch and channel with shape ``[n_epochs, n_channels]``.
        Contains ``NaN`` for elements where total power in the band is approximately 0.

    Raises
    ------
    ValueError
        If `psd` does not have 3 dimensions.
        If `target_band` does not have exactly two elements.
        If `percentile` is not between 0 and 100.

    Examples
    --------
    >>> import numpy as np
    >>> # 1 epoch, 129 frequency bins (0-64Hz), 2 channels
    >>> psd_example = np.random.rand(1, 129, 2)
    >>> fs = 128
    >>> sef_val = spectral_edge_frequency(psd_example, fs, percentile=95.0)
    """
    psd = check_dimensions(psd)

    if target_band is None:
        target_band = (0, fs / 2)

    target_band = np.asarray(target_band, dtype=float)

    # Checks
    if psd.ndim != 3:
        raise ValueError(
            "psd must have shape [n_epochs, n_freqs, n_channels]. "
            "Use np.newaxis to expand missing dimensions."
        )
    if target_band.shape != (2,):
        raise ValueError("target_band must be (low_freq, high_freq).")
    if not (0.0 <= percentile <= 100.0):
        raise ValueError("percentile must be between 0 and 100.")

    # Frequency axis for a one-sided PSD
    freqs = np.linspace(0, fs / 2, psd.shape[1])

    # Band selection
    idx = (freqs >= target_band[0]) & (freqs < target_band[1])
    freqs_in_band = freqs[idx]
    psd_band = np.maximum(psd[:, idx, :], 0.0)  # ensure non-negative power

    # Total and cumulative power in band
    total_power = np.sum(psd_band, axis=1)  # [n_epochs, n_channels]
    cum_power = np.cumsum(psd_band, axis=1)  # [n_epochs, n_bins_in_band, n_channels]

    # Target cumulative power
    frac = percentile / 100.0
    target = frac * total_power  # [n_epochs, n_channels]

    # Handle near-zero power
    valid = total_power > eps

    # First index where cumulative >= target
    ge = cum_power >= target[:, np.newaxis, :]  # broadcast target
    sef_idx = np.argmax(ge, axis=1)  # [n_epochs, n_channels] (0 if all False)

    # If all False because total_power ~ 0, mark invalid
    sef = freqs_in_band[sef_idx]
    sef = np.where(valid, sef, np.nan)

    return sef



if __name__ == '__main__':
    # Example usage
    import numpy as np
    from medusa.signal_metrics.median_frequency import median_frequency

    fs = 256
    psd = np.random.rand(100, 1)  # 1 epochs, 100 freq bins, 1 channels
    sef_50 = spectral_edge_frequency(
        psd, fs,
        target_band=None,
        percentile=50)
    mf = median_frequency(
        psd, fs)

    print("SEF(50) values:\n", sef_50)
    print("MF values:\n", mf)