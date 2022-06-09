import numpy as np
import scipy.signal as spsig
import medusa

bands_rp = ([0, 4], [4, 8], [8, 13], [13, 19], [19, 30], [30, 70])


def band_power(psd, fs, bands):
    """
    This method computes the spectral power of the signal in the given bands.

    Parameters
    ----------
    psd : numpy array
        PSD of MEEG Signal. [n_samples x n_channels]
    fs : int
        Sampling frequency of the signal
    bands : numpy 2D array
        Frequency bands where the RP will be computed. [[b1_start, b1_end], ...
        [bn_start, bn_end]]

    Returns
    -------
    powers : numpy 2D array
        RP value for each band and channel in "signal". [n_bands x n_epochs x
         n_channels].
    """

    # Check errors
    if len(psd.shape) != 3:
        raise Exception('Parameter psd must have shape [n_epochs x n_samples x '
                        'n_channels]')

    if len(np.array(bands).shape) != 2:
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
    """
    This method computes the median frequency of the signal in the given band.

    Parameters
    ----------
    psd : numpy array
        PSD of MEEG Signal. [n_samples x n_channels]
    fs : int
        Sampling frequency of the signal
    band : numpy array
        Frequency band where the MF will be computed. [b1_start, b1_end].
        Default [1, 70]

    Returns
    -------
    median_freqs : numpy 2D array
        MF value for each band and channel in "signal". [n_epoch x n_channels].
    """
    # Check errors
    if len(psd.shape) != 3:
        raise Exception('Parameter psd must have shape [n_epochs x n_samples x '
                        'n_channels]')

    if np.array(band).shape[0] != 2:
        raise Exception('Parameter band must be a 2-D array of the desired '
                        'band. Ej. Delta: [0, 4]')

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


def indiv_alpha_frequency(psd, fs, band=(4, 15)):
    """
    This method computes the individual alpha frequency of the signal in the
    given band.

    Parameters
    ----------
    psd : numpy array
        PSD of MEEG Signal. [n_samples x n_channels]
    fs : int
        Sampling frequency of the signal
    band : numpy array
        Frequency band where the IAF will be computed. [b1_start, b1_end].
        Default [9, 15]

    Returns
    -------
    iaf : numpy 2D array
        IAF value for each band and channel in "signal". [1 x n_channels].
    """
    # Check errors
    if len(psd.shape) != 3:
        raise Exception('Parameter psd must have shape [n_epochs x n_samples x '
                        'n_channels]')

    if np.array(band).shape[0] != 2:
        raise Exception('Parameter band must be a 2-D array of the desired '
                        'band. Ej. Delta: [0, 4]')

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
    iaf_idx = np.argmax(
        np.cumsum(cum_power <= (total_power / 2), axis=1), axis=1)
    iaf = freqs_in_band[iaf_idx]

    return iaf


def shannon_spectral_entropy(psd, fs, band=(1, 70)):
    """
    Computes the Shannon spectral entropy of the PSD in the given band.

   Parameters
    ----------
    psd : numpy array
        PSD of MEEG Signal. [n_samples x n_channels]
    fs : int
        Sampling frequency of the signal
    band : numpy array
        Frequency band where the SE will be computed. [b1_start, b1_end].
        Default [1, 70]

    Returns
    -------
    se_freqs : numpy 2D array
        SE value for each band and channel in "signal". [1 x n_channels].
    """

    # Check errors
    if len(psd.shape) != 3:
        raise Exception('Parameter psd must have shape [n_epochs x n_samples x '
                        'n_channels]')

    if np.array(band).shape[0] != 2:
        raise Exception('Parameter band must be a 2-D array of the desired '
                        'band. Ej. Delta: [0, 4]')

    # Calculate freqs array
    freqs = np.linspace(0, fs/2, psd.shape[1])

    # Compute shannon entropy
    idx = np.logical_and(freqs >= band[0], freqs < band[1])
    # Calculate total power
    total_power = np.sum(psd[:, idx, :], axis=1, keepdims=True)
    # Calculate the probability density function
    pdf = np.abs(psd[:, idx, :]) / total_power
    # Calculate shannon entropy
    se = np.squeeze(-np.sum(pdf * np.log(pdf), axis=1) / np.log(pdf.shape[1]))

    return np.array(se)

def power_spectral_density(signal,fs,epoch_len=None):
    """
    This method allows to compute the power spectral density by means of
    Welch's periodogram method.

    Parameters
    ----------
    signal : numpy 2D matrix
        MEEG Signal. [n_samples x n_channels].
    fs : int
        Sampling frequency of the signal
    epoch_len : int or None
        Length of the epochs in which divide the signal. If None,
        the power spectral density will be calculated from the
        entire signal.

    Returns
    -------
    f : numpy 1D array
        Array of sample frequencies.

    psd: numpy 2D array
        PSD of MEEG Signal. [n_epochs, n_samples, n_channels]
        """

    if len(signal.shape) < 2:
        signal = signal[ ..., np.newaxis]

    if epoch_len is not None:
        if not isinstance(epoch_len,int):
            raise ValueError("Epoch length must be a integer"
                             "value.")
        if epoch_len > signal.shape[0]:
            raise ValueError("Epoch length must be shorter than"
                             "signal duration")
    else:
        epoch_len = signal.shape[0]

    signal_epoched = medusa.get_epochs(signal, epoch_len)

    # TODO BORRAR ESTO?
    # if len(signal_epoched.shape) < 2:
    #     signal_epoched = signal_epoched[np.newaxis, ..., np.newaxis]
    if len(signal_epoched.shape) < 3:
        signal_epoched = signal_epoched[np.newaxis, ...]
        #TODO BORRAR ESTO?

        # if signal.shape[1] == 1:
        #     signal_epoched = signal_epoched[np.newaxis, ...]
        #     signal_epoched = signal_epoched[..., np.newaxis]
        # else:
        #     signal_epoched = signal_epoched[np.newaxis, ...]

    # Estimating the PSD
    # Get the number of samples for the PSD length
    n_samp = signal_epoched.shape[1]
    # Compute the PSD
    f, psd = spsig.welch(signal_epoched, fs=fs, window='boxcar',
                         nperseg=n_samp, noverlap=0, axis=-2)

    return f,psd


def compute_spectral_metric(signal, fs, param, epoch_len=None, bands=bands_rp):
    """ This method allows to compute the different spectral parameters
    implemented in MEDUSA in an easy way. It is just necessary to provide the
    signal, its sampling frequency, the epoch length, and the desired parameter.

    Parameters
    ----------
    signal : numpy 2D matrix
        MEEG Signal. [n_samples x n_channels].
    fs : int
        Sampling frequency of "signal"
    epoch_len : int
        Epoch length in samples.
    param : string
        Parameter to be calculated. Possible values: 'RP', 'MF', 'IAF', 'SE'
    bands : numpy 2D array
        Frequency bands where the RP will be computed. [[b1_start, b1_end], ...
        [bn_start, bn_end]]. By default, the canonical bands will be used

    Returns
    -------
    param_values : numpy 2D array
        Per-channel values of the spectral parameter selected in "param"
        [n_epochs x n_channels]. If "param" is "RP", "spect_param" will be
        [n_bands x n_epochs x n_channels]

    """
    if not np.issubdtype(signal.dtype, np.number):
        raise ValueError('data matrix contains non-numeric values')

    # if epoch_len is None:
    #     epoch_len = signal.shape[0]


    f, psd = power_spectral_density(signal,fs,epoch_len)

    # # Epoching
    # signal_epoched = medusa.get_epochs(signal, epoch_len)
    # if len(signal_epoched.shape) < 2:
    #     signal_epoched = signal_epoched[np.newaxis, ..., np.newaxis]
    # elif len(signal_epoched.shape) < 3:
    #     if signal.shape[1] == 1:
    #         signal_epoched = signal_epoched[..., np.newaxis]
    #     else:
    #         signal_epoched = signal_epoched[np.newaxis, ...]
    #
    # # Estimating the PSD
    # # Get the number of samples for the PSD length
    # n_samp = signal_epoched.shape[1]
    # # Compute the PSD
    # f, psd = spsig.welch(signal_epoched, fs=fs, window='boxcar',
    #                      nperseg=n_samp, noverlap=0, axis=-2)

    # # Initialize output variable
    # if param == 'RP':
    #     param_values = np.zeros((len(bands),
    #                              signal_epoched.shape[0],
    #                              signal_epoched.shape[2]))
    # else:
    #     param_values = np.zeros((signal_epoched.shape[0],
    #                              signal_epoched.shape[2]))

    # Calculate the parameters
    if param == 'RP':
        param_values = band_power(psd, fs, bands)
    elif param == 'MF':
        param_values = median_frequency(psd, fs)
    elif param == 'IAF':
        param_values = indiv_alpha_frequency(psd, fs)
    elif param == 'SE':
        param_values = shannon_spectral_entropy(psd, fs)
    else:
        raise ValueError("Unknown spectral parameter")

    return param_values


if __name__ == "__main__":
    import scipy.io
    import time
    import matplotlib.pyplot as plt
    # mat = scipy.io.loadmat('Path/File.mat')
    # vector = np.array(mat["signal"])[:, :]
    # signal = vector.T
    signal = np.random.random(500000)
    param = 'RP'

    output = compute_spectral_metric(signal, fs=1000, param=param, epoch_len=None)
    aa = output[:, 5, :]
    bb = np.sum(aa, axis=0)
    cc = 0

