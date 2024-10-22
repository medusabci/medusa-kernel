import numpy as np
from scipy.signal import welch as welch_sp
from scipy.signal import hilbert as hilbert_sp
from medusa.utils import check_dimensions
import pywt
from scipy import ndimage
from scipy.signal import windows, ShortTimeFFT, detrend

def hilbert(signal):
    """This method implements the Hilbert transform.

    Parameters
    ----------
    signal :  numpy.ndarray
        Input signal with shape [n_epochs x n_samples x n_channels].

    Returns
    -------
    hilb : numpy 3D matrix
        Analytic signal of x [n_epochs, n_samples, n_channels].
    """
    # Check dimensions
    signal = np.asarray(signal)
    signal = check_dimensions(signal)
    # Check errors
    if np.iscomplexobj(signal):
        raise ValueError("Signal must be real.")

    return hilbert_sp(signal, axis=1)


def power_spectral_density(signal, fs, segment_pct=80, overlap_pct=50,
                           window='boxcar'):
    """This method allows to compute the one-sided power spectral density (PSD)
    by means of Welch's periodogram method. This method wraps around
    scipy.signal.welch method to compute the PSD, allowing to pass epoched
    signals and defining the segment length and overlap in percentage,
    simplifying the use for specific purposes. For more advanced configurations,
    use the original scipy (or equivalent) function.

    Parameters
    ----------
    signal : numpy nd array
        Signal with shape [n_epochs x n_samples x n_channels].
    fs : int
        Sampling frequency of the signal
    segment_pct: float
        Percentage of the signal (n_samples) used to calculate the FFT. Default:
        80% of the signal.
    overlap_pct: float
        Percentage of overlap (n_samples) for the Welch method. Default: 50% of
        the signal.
    window:
        Desired window to use. See scipy.signal.welch docs for more details

    Returns
    -------
    f : numpy 1D array
        Array of sampled frequencies.
    psd: numpy 2D array
        PSD of the signal with shape [n_epochs, n_samples, n_channels]
    """

    # Check signal dimensions
    signal = check_dimensions(signal)
    # Get the number of samples for the PSD length
    n_samp = signal.shape[1]
    # Get nperseg and noverlap
    nperseg = n_samp * segment_pct / 100
    noverlap = n_samp * overlap_pct / 100
    # Compute the PSD
    f, psd = welch_sp(signal, fs=fs, window=window, nperseg=nperseg,
                      noverlap=noverlap, axis=1)
    return f, psd


def normalize_psd(psd, norm='rel'):
    """Normalizes the PSD using different methods.

    Parameters
    ----------
    psd: numpy array or list
        Power Spectral Density (PSD) of the signal with shape [samples],
        [samples x channels] or [epochs x samples x channels]. It assumes PSD
        is the one-sided spectrum.
    norm: string
        Normalization to be performed. Choose z for z-score or rel for
        relative power.

    """
    # To numpy arrays
    psd = np.array(psd)

    # Check errors
    if len(psd.shape) != 3:
        raise Exception('Parameter psd must have shape [n_epochs x n_samples x '
                        'n_channels]')

    if norm == 'rel':
        p = np.sum(psd, axis=1, keepdims=True)
        psd_norm = psd / p
    elif norm == 'z':
        m = np.mean(psd, keepdims=True, axis=1)
        s = np.std(psd, keepdims=True, axis=1)
        psd_norm = (psd - m) / s
    else:
        raise Exception('Unknown normalization. Choose z or rel')

    return psd_norm

def fourier_spectrogram(signal, fs, time_window=1, overlap_pct=80,
                        smooth=True, smooth_sigma=2, apply_detrend=True,
                        apply_normalization=True, scale_to=None):

    """This method calculates the spectrogram of a signal from the Short Time
        Fourier Transform (STFT) with a gaussian window.

        Implementation based on https://github.com/drammock/spectrogram-tutorial

    Parameters
    ----------
    signal : numpy nd array
        Signal with shape [n_samples].
    fs : int
        Sampling frequency of the signal
    time_window: float
        Length in seconds of the gaussian window used in STFT.
    overlap_pct: int
        Percentage of the signal that is overlapped during the STFT calculation.
        Default: 80% of the window.
    smooth: bool
        Define if a gaussian filter is used to smooth the final result.
        Default: True
    smooth_sigma: float
        Sigma value used for the gaussian filter if smooth option is True.
    apply_detrend: bool
        Define if linear de-trending  is applied to the signal before the
        STFT. Default: True
    apply_normalization: bool
        Define if normalization  is applied to the signal before the
        STFT. Default: True
    scale_to: ‘magnitude’, ‘psd’ | None
        Choose how the output is scaled, so each STFT column represents either
        'magnitude' or a PSD spectrum.


    Returns
    -------
    Sx : numpy 2D array
        Spectrogram of the signal with shape [n_frequencies, n_samples].
    times: numpy 1D array
        Numpy array  with the time stamps.
    frequencies: numpy 1D array
        Numpy array  with the frequency values. The maximum value is the
        fs/2."""

    # Check errors
    if not 0 <= overlap_pct <= 100:
        raise ValueError(f"Error: overlap_pct parameter expected "
                         f"between 0 and 100.")

    signal = signal.squeeze()
    if len(signal.shape) > 1:
        raise ValueError("Only one-channel signals are supported.")

    # Apply detrend and normalization
    if apply_detrend:
        signal = detrend(signal, type='linear')
    if apply_normalization:
        stddev = signal.std()
        signal = signal / stddev

    # Convert window from seconds to numbers of samples
    window_nsamp = int(time_window * fs)

    # Define gaussian window
    window_sigma = (window_nsamp + 1) / 6
    window = windows.gaussian(window_nsamp, window_sigma, sym=True)

    # Compute the number of samples to overlap
    noverlap = overlap_pct * window_nsamp / 100

    # Compute the hop of gaussian window in sammples
    step_nsamp = int(window_nsamp - noverlap)

    # Spectrogram using SFFT
    SFT = ShortTimeFFT(win=window,hop=step_nsamp,fs=fs,
                              scale_to=scale_to)
    Sx = SFT.spectrogram(signal)

    # Smooth
    if smooth:
        Sx = ndimage.gaussian_filter(Sx, sigma=smooth_sigma)

    # Compute times and frequencies vectors
    t_lo, t_hi, f_lo, f_hi = SFT.extent(len(signal))
    times = np.arange(t_lo, t_hi, SFT.delta_t)
    frequencies = np.arange(f_lo, f_hi, SFT.delta_f)

    return Sx, times, frequencies

def __cone_of_influency(center_frequency, N, fs):
    f_0 = center_frequency * 2 * np.pi
    cmor_flambda = 4 * np.pi / (f_0 + np.sqrt(2 + f_0 ** 2))
    coi = (1.0 / np.sqrt(2)) * cmor_flambda * (
            N / 2 - np.abs(np.arange(0, N) - (N - 1) / 2)) * (1.0 / fs)
    return 1.0/coi

def __compute_scales(N, filters_per_octave):
    nOctaves = int(np.log2(2 * np.floor(N / 2.0)))
    scales = 2 ** np.arange(1, nOctaves, 1.0 / filters_per_octave)
    return scales

def cwt_spectrogram(signal, fs, filters_per_octave=5, center_frequency=1,
                    bandwidth_frequency=1.5, apply_detrend=True,
                    apply_normalization=True, smooth=True,smooth_sigma=2):

    """This method calculates the spectrogram of a signal from the Continuous
        Wavelet Transform (CWT) with complex Morlet wavelets.

        Implementation based on https://gist.github.com/MiguelonGonzalez

    Parameters
    ----------
    signal : numpy nd array
        Signal with shape [n_samples].
    fs : int
        Sampling frequency of the signal
    filters_per_octave: int
        Number of filters used to compute CWT.
    center_frequency: float
        Center frequency used to compute CWT using Morlet wavelets.
    bandwidth_frequency: float
        Band width used to compute CWT using Morlet wavelets.
    smooth: bool
        Define if a gaussian filter is used to smooth the final result.
        Default: True
    smooth_sigma: float
        Sigma value used for the gaussian filter if smooth option is True.
    apply_detrend: bool
        Define if linear de-trending  is applied to the signal before the
        CWT. Default: True
    apply_normalization: bool
        Define if normalization  is applied to the signal before the
        CWT. Default: True


    Returns
    -------
    power : numpy 2D array
        Spectrogram of the signal with shape [n_frequencies, n_samples].
    times: numpy 1D array
        Numpy array  with the time stamps.
    frequencies: numpy 1D array
        Numpy array  with the frequency values. The maximum value is the
        fs/2.
    coif: numpy 1D array
        Numpy array  with shape [n_samples] with cone of influence values."""

    # Check errors
    signal = signal.squeeze()
    if len(signal.shape) > 1:
        raise ValueError("Only one-channel signals are supported.")

    # Apply detrend and normalization
    if apply_detrend:
        signal = detrend(signal, type='linear')
    if apply_normalization:
        stddev = signal.std()
        signal = signal / stddev

    N = len(signal)
    dt = 1.0 / fs
    times = np.arange(N) * dt

    scales = __compute_scales(N,filters_per_octave)

    c, freqs = pywt.cwt(signal, scales,
                        f'cmor{bandwidth_frequency}-{center_frequency}')
    frequencies = pywt.scale2frequency(
        f'cmor{bandwidth_frequency}-{center_frequency}', scales) / dt

    power = np.abs(c * np.conj(c))

    # smooth a bit
    if smooth:
        power = ndimage.gaussian_filter(power, sigma=smooth_sigma)

    coif = __cone_of_influency(center_frequency,N,fs)

    return power[frequencies<=0.5*fs], times, frequencies[frequencies<=0.5*fs], \
        coif

def cross_cwt(signal1, signal2, fs, mode='spectrogram', filters_per_octave=5,
              center_frequency=1, bandwidth_frequency=1.5, apply_detrend=True,
                        apply_normalization=True, smooth=True,smooth_sigma=2):

    """This method calculates the Cross Wavelet Transform of two signals from
        the Continuous Wavelet Transform (CWT) with complex Morlet wavelets.
        Both spectrogram and phase can be calculated.

    Implementation based on https://gist.github.com/MiguelonGonzalezParameters

    Parameters
    ----------
    signal1 : numpy nd array
        Signal with shape [n_samples].
    signal2 : numpy nd array
        Signal with shape [n_samples].
    fs : int
        Sampling frequency of the signal
    mode: 'spectrogram', 'phase' | None
        Choose the desired result. 'spectrogram' returns the cross spectrogram
        between both signals while 'phase' returns the cross phase.
    filters_per_octave: int
        Number of filters used to compute CWT.
    center_frequency: float
        Center frequency used to compute CWT using Morlet wavelets.
    bandwidth_frequency: float
        Band width used to compute CWT using Morlet wavelets.
    smooth: bool
        Define if a gaussian filter is used to smooth the final result.
        Default: True
    smooth_sigma: float
        Sigma value used for the gaussian filter if smooth option is True.
    apply_detrend: bool
        Define if linear de-trending  is applied to the signal before the
        CWT. Default: True
    apply_normalization: bool
        Define if normalization  is applied to the signal before the
        CWT. Default: True


    Returns
    -------
    power : numpy 2D array
        Spectrogram of the signal with shape [n_frequencies, n_samples].
    times: numpy 1D array
        Numpy array  with the time stamps.
    frequencies: numpy 1D array
        Numpy array  with the frequency values. The maximum value is the
        fs/2.
    coif: numpy 1D array
        Numpy array  with shape [n_samples] with cone of influence values."""

    # Check errors
    signal1 = signal1.squeeze()
    if len(signal1.shape) > 1:
        raise ValueError("Signal 1 Error: Only one-channel signals are supported.")

    signal2 = signal2.squeeze()
    if len(signal2.shape) > 1:
        raise ValueError("Signal 2 Error: Only one-channel signals are supported.")

    if not (len(signal1) == len(signal2)):
        raise ValueError("Signals must have the same length")

    if mode is None:
        mode = 'spectrogram'

    # Apply detrend and normalization
    if apply_detrend:
        signal1 = detrend(signal1, type='linear')
        signal2 = detrend(signal2, type='linear')
    if apply_normalization:
        stddev1 = signal1.std()
        signal1 = signal1 / stddev1
        stddev2 = signal2.std()
        signal2 = signal2 / stddev2

    N = len(signal1)
    dt = 1.0 / fs
    times = np.arange(N) * dt


    scales = __compute_scales(N,filters_per_octave)

    c1, _ = pywt.cwt(signal1, scales,
                        f'cmor{bandwidth_frequency}-{center_frequency}')
    c2, _ = pywt.cwt(signal2, scales,
                        f'cmor{bandwidth_frequency}-{center_frequency}')

    result = c1 * np.conj(c2)
    frequencies = pywt.scale2frequency(
        f'cmor{bandwidth_frequency}-{center_frequency}', scales) / dt


    if mode == 'spectrogram':
        result = np.abs(result)
    elif mode == 'phase':
        result = np.angle(result)

    if smooth:
        result = ndimage.gaussian_filter(result, sigma=smooth_sigma)

    coif = __cone_of_influency(center_frequency, N, fs)

    return result[frequencies<=0.5*fs], times, frequencies[frequencies<=0.5*fs],\
        coif




