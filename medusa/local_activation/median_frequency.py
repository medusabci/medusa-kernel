import numpy as np

def median_frequency(psd, fs, target_band=(1, 70)):
    """
    This method computes the median frequency (MF) of the signal in the given
    band.

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
    target_band : numpy array
        Frequency band where the MF will be computed. [low_freq, high_freq].
        Default [1, 70].

    Returns
    -------
    median_freqs : numpy 2D array
        MF value for each epoch and channel, with shape [n_epochs x n_channels].

          Examples
    --------
    >>> import numpy as np
    >>> from medusa.local_activation.median_frequency import median_frequency
    >>> fs = 256
    >>> psd = np.random.rand(10, 129, 16)  # 10 epochs, 129 samples, 16 channeles
    >>> mf = median_frequency(psd, fs, [1, 50])
    >>> mf.shape
    (10, 16)
    """
    # To numpy arrays
    psd = np.array(psd)
    target_band = np.array(target_band)

    # Check errors
    if len(psd.shape) != 3:
        raise ValueError('Parameter psd does not have correct dimensions. '
                         'Check the documentation for more information.')
    if len(target_band.shape) != 1 and target_band.shape[0] != 2:
        raise Exception('Parameter band must be an array with the desired '
                        'band. E.g., Delta: [0, 4]')

    # Calculate freqs array
    freqs = np.linspace(0, fs / 2, psd.shape[1])

    # Compute median frequency
    idx = np.logical_and(freqs >= target_band[0], freqs < target_band[1])
    freqs_in_band = freqs[idx]
    # Calculate total power
    total_power = np.sum(psd[:, idx, :], axis=1, keepdims=True)
    # Calculate cumulative power
    cum_power = np.cumsum(psd[:, idx, :], axis=1)
    # Get median frequency
    median_freq_idx = np.argmax(
        np.cumsum(cum_power <= (total_power / 2), axis=1), axis=1)
    median_freqs = freqs_in_band[median_freq_idx]

    return median_freqs