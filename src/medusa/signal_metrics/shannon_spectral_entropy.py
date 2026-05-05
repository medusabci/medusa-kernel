import numpy as np

def shannon_spectral_entropy(psd, fs, target_band=(1, 70)):
    """
    Compute the normalized Shannon spectral entropy (SE) of a power spectral
    density (PSD) within a specified frequency band.

    The PSD is first restricted to the selected frequency band and normalized
    to obtain a probability density function (PDF) across frequency bins.
    The Shannon entropy is then computed as:

        SE = -sum(p_i * log(p_i)) / log(N)

    where p_i are the normalized spectral coefficients and N is the number of
    frequency bins in the selected band. The entropy is normalized to the
    interval [0, 1].

    Zero-probability bins are handled according to the convention
    0 * log(0) = 0. If the total power in the selected band is zero for a
    given epoch and channel, the entropy is set to 0 (i.e., undefined cases
    are treated as zero entropy).

    Parameters
    ----------
    psd : numpy.ndarray
        Power spectral density of shape (n_epochs, n_freqs, n_channels).
        If the signal is not epoched or has a single channel, missing
        dimensions must be explicitly added using np.newaxis.
        For example, a single-channel non-epoched PSD of shape (n_freqs,)
        should be passed as psd[np.newaxis, :, np.newaxis].
    fs : float or int
        Sampling frequency of the original signal (Hz).
    target_band : array-like of length 2, optional
        Frequency band [low_freq, high_freq) in Hz where the spectral
        entropy is computed. Default is (1, 70).

    Returns
    -------
    se : numpy.ndarray
        Normalized Shannon spectral entropy with shape
        (n_epochs, n_channels). Values range between 0 (spectrally
        concentrated) and 1 (spectrally uniform).

    Examples
    --------
    >>> import numpy as np
    >>> from medusa.signal_metrics.shannon_spectral_entropy import shannon_spectral_entropy
    >>> fs = 256
    >>> psd = np.random.rand(2, 129, 3)  # 2 epochs, samples, 3 channels
    >>> entropy = shannon_spectral_entropy(psd, fs, [4, 30])
    >>> entropy.shape
    (2, 3)
    """

    # To numpy arrays
    psd = np.array(psd)
    target_band = np.array(target_band)

    # Check errors
    if psd.ndim != 3:
        raise ValueError('Parameter psd does not have correct dimensions. '
                         'Check the documentation for more information.')
    if target_band.ndim != 1 or target_band.shape[0] != 2:
        raise Exception('Parameter band must be an array with the desired '
                        'band. E.g., Delta: [0, 4]')

    # Calculate freqs array
    freqs = np.linspace(0, fs / 2, psd.shape[1], endpoint=True)
    idx = (freqs >= target_band[0]) & (freqs < target_band[1])

    # Calculate total power
    band_psd = np.abs(psd[:, idx, :])  # [epochs, bins, ch]
    total_power = np.sum(band_psd, axis=1, keepdims=True)  # [epochs, 1, ch]

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

    return se
