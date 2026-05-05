import numpy as np
from medusa.transforms import normalize_psd

def band_power(psd, fs, target_band, type='absolute', norm_range=None):
    """
    This method computes total power of the signal within a specific frequency band,
    using the Power Spectral Density (PSD). DO NOT INTRODUCE THE NOT INTRODUCE THE PSD NORMALIZED

    Parameters
    ----------
    psd : numpy array
        PSD of the signal with shape [n_epochs, n_samples, n_channels]. Some
        of these dimensions may not exist in advance. In these case, create new
        axis using np.newaxis. E.g., non-epoched single-channel psd with
        shape [n_samples] can be passed to this function with psd[numpy.newaxis,
        ..., numpy.newaxis]. Afterwards, you may use numpy.squeeze to eliminate
        those axes.
    fs : int
        Sampling frequency of the signal
    target_band : numpy 2D array
        Frequency band where to calculate the power in Hz. E.g., [8, 13]
    type : string
        Type of power to compute. Options are 'absolute' or 'relative'.

    Returns
    -------
    powers : numpy 2D array
        Power value for each epoch and channel within the target frequency band. [n_epochs, n_channels]

    Examples
    --------
    >>> import numpy as np
    >>> from medusa.signal_metrics.band_power import band_power
    >>> fs = 256  # Sampling frequency
    >>> psd = np.random.rand(1, 129, 2)  # Example PSD: 1 epoch, samples, 2 channels
    >>> band = [8, 13]  # Alpha band
    >>> power = band_power(psd, fs, band)
    >>> power.shape
    (1, 2)
    """

    # To numpy arrays
    psd = np.array(psd)

    # Check errors
    if type not in ['absolute', 'relative']:
        raise Exception("Parameter type must be 'absolute' or 'relative'")
    if len(psd.shape) != 3:
        raise Exception('Parameter psd must have shape [n_epochs x n_samples x '
                        'n_channels]')

    # Calculate freqs array
    freqs = np.linspace(0, fs/2, psd.shape[1])

    if type == 'relative':
        if not (isinstance(norm_range, (list, tuple)) and
                len(norm_range) == 2 and
                all(isinstance(x, (int, float)) for x in norm_range)):
            raise ValueError("norm_range must be a list or two-number tuple, such as [1, 70]")
        psd = normalize_psd(psd, [norm_range[0], norm_range[1]], freqs, norm='rel')

    # Compute power
    psd_target_samp = np.logical_and(freqs >= target_band[0], freqs <= target_band[1])
    band_power = np.sum(psd[:, psd_target_samp, :], axis=1)

    if type == 'absolute': # Normalize to make it comparable across different sampling frequencies and PSD lengths
        band_power = band_power * (fs / (2 * freqs.shape[0]))

    return band_power