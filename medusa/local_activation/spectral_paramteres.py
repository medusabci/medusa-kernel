import numpy as np
import scipy.signal as ss
import medusa

def band_power(signal, fs, bands = None):
    """
    Computes the spectral power of the signal in the given bands. In case bands parameter
    were None, classical band definitions would be used.
    Parameters
    ----------
    signal: np.array [N_samples, N_channels]
    fs: int or float. Sampling rate.
    bands: frequency bands [[b1_start, b1_end], ... [bn_start, bn_end]]

    Returns
    -------

    """
    f, psd = ss.welch(signal,fs)
    # Check errors
    if len(psd.shape) > 3:
        raise Exception(
            'Parameter psd must have shape [samples], [samples x channels] or [epochs x samples x channels]')

    if len(np.array(bands).shape) != 2:
        raise Exception('Parameter bands must be a 2-D array of the desired bands. Ej. Delta and theta bands: [[0, 4], '
                        '[4, 8]]')

    # Reshape
    if len(psd.shape) == 1:
        psd = psd.reshape(psd.shape[0], 1)

    if len(psd.shape) == 2:
        psd = psd.reshape(1, psd.shape[0], psd.shape[1])

    # Calculate freqs array
    freqs = np.linspace(0, fs / 2, psd.shape[1])

    # Compute powers
    powers = np.zeros((len(bands), psd.shape[0], psd.shape[2]))
    for b in range(len(bands)):
        band = bands[b]
        psd_samp = np.logical_and(freqs >= band[0], freqs < band[1])
        powers[b, :, :] = np.sum(psd[:, psd_samp, :], axis=1) * (fs / (2 * freqs.shape[0]))

    return powers

def median_frequency(psd, fs, bands):
    """
    Computes the Shannon entropy of the PSD in the given bands.

    :param psd: Power Spectral Density (PSD) of the signal with shape [samples], [samples x channels] or
                [epochs x samples x channels]. It assumes PSD is the one-sided spectrum.
    :type psd: numpy array or list

    :param fs: sample frequency
    :type fs: int or float

    :param bands: frequency bands [[b1_start, b1_end], ... [bn_start, bn_end]]
    :type bands: list

    """
    # Check errors
    if len(psd.shape) > 3:
        raise Exception('Parameter psd must have shape [samples], [samples x channels] or [epochs x samples x channels]')

    if len(np.array(bands).shape) != 2:
        raise Exception('Parameter bands must be a 2-D array of the desired bands. Ej. Delta and theta bands: [[0, 4], '
                        '[4, 8]]')

    # Reshape
    if len(psd.shape) == 1:
        psd = psd.reshape(psd.shape[0], 1)

    if len(psd.shape) == 2:
        psd = psd.reshape(1, psd.shape[0], psd.shape[1])

    # Calculate freqs array
    freqs = np.linspace(0, fs / 2, psd.shape[1])

    # Compute median frequency
    median_freqs = np.zeros((len(bands), psd.shape[0], psd.shape[2]))
    for b in range(len(bands)):
        band = bands[b]
        idx = np.logical_and(freqs >= band[0], freqs < band[1])
        freqs_in_band = freqs[idx]
        # Calculate total power
        total_power = np.sum(psd[:, idx, :], axis=1, keepdims=True)
        # Calculate cumulative power
        cum_power = np.cumsum(psd[:, idx, :], axis=1)
        # Get median frequency
        median_freq_idx = np.argmax(np.cumsum(cum_power <= (total_power / 2), axis=1), axis=1)
        median_freqs[b, :, :] = freqs_in_band[median_freq_idx]
    return median_freqs


def shannon_entropy(psd, fs, bands):
    """
    Computes the Shannon entropy of the PSD in the given bands.

    :param psd: Power Spectral Density (PSD) of the signal with shape [samples], [samples x channels] or
                [epochs x samples x channels]. It assumes PSD is the one-sided spectrum.
    :type psd: numpy array or list

    :param fs: sample frequency
    :type fs: int or float

    :param bands: frequency bands [[b1_start, b1_end], ... [bn_start, bn_end]]
    :type bands: list

    """
    # Check errors
    if len(psd.shape) > 3:
        raise Exception('Parameter psd must have shape [samples], [samples x channels] or [epochs x samples x channels]')

    if len(np.array(bands).shape) != 2:
        raise Exception('Parameter bands must be a 2-D array of the desired bands. Ej. Delta and theta bands: [[0, 4], '
                        '[4, 8]]')

    # Reshape
    if len(psd.shape) == 1:
        psd = psd.reshape(psd.shape[0], 1)

    if len(psd.shape) == 2:
        psd = psd.reshape(1, psd.shape[0], psd.shape[1])

    # Calculate freqs array
    freqs = np.linspace(0, fs / 2, psd.shape[1])

    # Compute shannon entropy
    se = np.zeros((len(bands), psd.shape[0], psd.shape[2]))
    for b in range(len(bands)):
        band = bands[b]
        idx = np.logical_and(freqs >= band[0], freqs < band[1])
        # Calculate total power
        total_power = np.sum(psd[:, idx, :], axis=1, keepdims=True)
        # Calculate the probability density function
        pdf = np.abs(psd[:, idx, :]) / total_power
        # Calculate shannon entropy
        se[b, :, :] = np.squeeze(-np.sum(pdf * np.log(pdf), axis=1) / np.log(pdf.shape[1]))

    return np.array(se)

