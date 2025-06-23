import numpy as np

def shannon_spectral_entropy(psd, fs, target_band=(1, 70)):
    """
    Computes the Shannon spectral entropy of the power spectral density (PSD)
    in the given band.

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
        Frequency band where the SE will be computed. [low_freq, high_freq].
        Default [1, 70]

    Returns
    -------
    sample_entropy : numpy 2D array
        SE value for each epoch and channel, with shape [n_epochs x n_channels].

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
    if len(psd.shape) != 3:
        raise ValueError('Parameter psd does not have correct dimensions. '
                         'Check the documentation for more information.')
    if len(target_band.shape) != 1 and target_band.shape[0] != 2:
        raise Exception('Parameter band must be an array with the desired '
                        'band. E.g., Delta: [0, 4]')

    # Calculate freqs array
    freqs = np.linspace(0, fs/2, psd.shape[1])

    # Compute shannon entropy
    idx = np.logical_and(freqs >= target_band[0], freqs < target_band[1])
    # Calculate total power
    total_power = np.sum(psd[:, idx, :], axis=1, keepdims=True)
    # Calculate the probability density function
    pdf = np.abs(psd[:, idx, :]) / total_power
    # Calculate shannon entropy
    se = -np.sum(pdf * np.log(pdf), axis=1) / np.log(pdf.shape[1])
    return se