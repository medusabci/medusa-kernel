import numpy as np

bands_rp = ([0, 4], [4, 8], [8, 13], [13, 19], [19, 30], [30, 70])


def band_power(psd, fs, bands):
    """
    This method computes the spectral power of the signal in the given bands.

    Parameters
    ----------
    psd : numpy array
        PSD of MEEG signal with shape [n_epochs, n_samples, n_channels]. Some
        of these dimensions may not exist in advance. In these case, create new
        axis using np.newaxis. E.g., non-epoched single-channel signal with
        shape [n_samples] can be passed to this function with psd[numpy.newaxis,
        ..., numpy.newaxis]. Afterwards, you may use numpy.squeeze to eliminate
        those axes.
    fs : int
        Sampling frequency of the signal
    bands : numpy 2D array
        Frequency bands where the RP will be computed. [[b1_start, b1_end], ...
        [bn_start, bn_end]]

    Returns
    -------
    powers : numpy 2D array
        RP value for each band, epoch and channel. [n_bands, n_epochs,
        n_channels]
    """
    # To numpy arrays
    psd = np.array(psd)
    bands = np.array(bands)

    # Check errors
    if len(psd.shape) != 3:
        raise Exception('Parameter psd must have shape [n_epochs x n_samples x '
                        'n_channels]')
    if len(bands.shape) != 2:
        raise Exception('Parameter bands must be a 2-D array of the desired '
                        'bands. Ej. Delta and theta bands: [[0, 4], [4, 8]]')

    # Calculate freqs array
    freqs = np.linspace(0, fs/2, psd.shape[1])

    # Compute powers
    powers = np.zeros((len(bands), psd.shape[0], psd.shape[2]))
    for b in range(len(bands)):
        band = bands[b]
        psd_samp = np.logical_and(freqs >= band[0], freqs < band[1])
        powers[b, :, :] = np.sum(psd[:, psd_samp, :], axis=1)

    powers = powers / np.sum(powers, axis=0, keepdims=True)
    return powers


def median_frequency(psd, fs, band=(1, 70)):
    """This method computes the median frequency (MF) of the signal in the given
    band.

    Parameters
    ----------
    psd : numpy array
        PSD of MEEG signal with shape [n_epochs, n_samples, n_channels]. Some
        of these dimensions may not exist in advance. In these case, create new
        axis using np.newaxis. E.g., non-epoched single-channel signal with
        shape [n_samples] can be passed to this function with psd[numpy.newaxis,
        ..., numpy.newaxis]. Afterwards, you may use numpy.squeeze to eliminate
        those axes.
    fs : int
        Sampling frequency of the signal
    band : numpy array
        Frequency band where the MF will be computed. [b1_start, b1_end].
        Default [1, 70].

    Returns
    -------
    median_freqs : numpy 2D array
        MF value for each epoch and channel with shape [n_epochs x n_channels].
    """
    # To numpy arrays
    psd = np.array(psd)
    band = np.array(band)

    # Check errors
    if len(psd.shape) != 3:
        raise ValueError('Parameter psd does not have correct dimensions. '
                         'Check the documentation for more information.')
    if len(band.shape) != 1 and band.shape[0] != 2:
        raise Exception('Parameter band must be an array with the desired '
                        'band. E.g., Delta: [0, 4]')

    # Calculate freqs array
    freqs = np.linspace(0, fs / 2, psd.shape[1])

    # Compute median frequency
    idx = np.logical_and(freqs >= band[0], freqs < band[1])
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


def individual_alpha_frequency(psd, fs, band=(4, 15)):
    """ This method computes the individual alpha frequency (IAF) of the signal
    in the given band. This is the another name for median frequency, usually
    used in MEEG studies.

    Parameters
    ----------
    psd : numpy array
        PSD of MEEG signal with shape [n_epochs, n_samples, n_channels]. Some
        of these dimensions may not exist in advance. In these case, create new
        axis using np.newaxis. E.g., non-epoched single-channel signal with
        shape [n_samples] can be passed to this function with psd[numpy.newaxis,
        ..., numpy.newaxis]. Afterwards, you may use numpy.squeeze to eliminate
        those axes.
    fs : int
        Sampling frequency of the signal
    band : numpy array
        Frequency band where the IAF will be computed. [b1_start, b1_end].
        Default [4, 15]

    Returns
    -------
    iaf : numpy 2D array
        IAF value for each epoch and channel with shape [n_epochs x n_channels].
    """
    return median_frequency(psd, fs, band=band)


def shannon_spectral_entropy(psd, fs, band=(1, 70)):
    """
    Computes the Shannon spectral entropy of the power spectral density (PSD)
    in the given band.

   Parameters
    ----------
    psd : numpy array
        PSD of MEEG signal with shape [n_epochs, n_samples, n_channels]. Some
        of these dimensions may not exist in advance. In these case, create new
        axis using np.newaxis. E.g., non-epoched single-channel signal with
        shape [n_samples] can be passed to this function with psd[numpy.newaxis,
        ..., numpy.newaxis]. Afterwards, you may use numpy.squeeze to eliminate
        those axes.
    fs : int
        Sampling frequency of the signal
    band : numpy array
        Frequency band where the SE will be computed. [b1_start, b1_end].
        Default [1, 70]

    Returns
    -------
    sample_entropy : numpy 2D array
        SE value for each epoch and channel with shape [n_epochs x n_channels].
    """
    # To numpy arrays
    psd = np.array(psd)
    band = np.array(band)

    # Check errors
    if len(psd.shape) != 3:
        raise ValueError('Parameter psd does not have correct dimensions. '
                         'Check the documentation for more information.')
    if len(band.shape) != 1 and band.shape[0] != 2:
        raise Exception('Parameter band must be an array with the desired '
                        'band. E.g., Delta: [0, 4]')

    # Calculate freqs array
    freqs = np.linspace(0, fs/2, psd.shape[1])

    # Compute shannon entropy
    idx = np.logical_and(freqs >= band[0], freqs < band[1])
    # Calculate total power
    total_power = np.sum(psd[:, idx, :], axis=1, keepdims=True)
    # Calculate the probability density function
    pdf = np.abs(psd[:, idx, :]) / total_power
    # Calculate shannon entropy
    se = -np.sum(pdf * np.log(pdf), axis=1) / np.log(pdf.shape[1])
    return se
