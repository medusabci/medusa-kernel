import warnings
from typing import Literal

import numpy as np
import pywt
from numpy.typing import NDArray
from scipy import ndimage
from scipy.signal import ShortTimeFFT, windows
from scipy.signal import hilbert as hilbert_sp
from scipy.signal import welch as welch_sp

from medusa.core.utils import check_data_dims

__all__ = [
    "hilbert",
    "power_spectral_density",
    "normalize_psd",
    "fourier_spectrogram",
    "cwt_spectrogram",
    "cross_cwt",
]


def hilbert(signal: NDArray) -> NDArray:
    """Compute the analytic signal via the Hilbert transform.

    Parameters
    ----------
    signal :
        Shape (n_segments, n_samples, n_channels). Input real-valued signal.
        2-D ``(n_samples, n_channels)`` and 1-D ``(n_samples,)`` arrays are
        promoted by :func:`medusa.core.utils.check_data_dims`.

    Returns
    -------
    NDArray
        Shape (n_segments, n_samples, n_channels). Complex-valued analytic
        signal; magnitude is the instantaneous envelope and phase is the
        instantaneous phase. Output dtype is complex with the same floating
        precision as the input.

    Raises
    ------
    ValueError
        If ``signal`` is complex-valued.

    Examples
    --------
    >>> import numpy as np
    >>> from medusa.signal.transforms import hilbert
    >>> sig = np.random.default_rng(0).standard_normal((1, 512, 4))
    >>> analytic = hilbert(sig)
    >>> analytic.shape
    (1, 512, 4)
    """
    signal = check_data_dims(np.asarray(signal), rep_type='time_segments')[0]
    if np.iscomplexobj(signal):
        raise ValueError("Signal must be real.")
    return hilbert_sp(signal, axis=1)


def power_spectral_density(
    signal: NDArray,
    fs: float,
    segment: float = 0.8,
    overlap: float = 0.5,
    window: str = 'boxcar',
) -> tuple[NDArray, NDArray]:
    """Compute the one-sided power spectral density (PSD) via Welch's method.

    Wraps :func:`scipy.signal.welch` to accept segmented signals in the
    canonical 3-D shape and to express segment length and overlap as
    fractions of ``n_samples``. For finer control of Welch parameters
    (detrending, scaling, custom ``nperseg`` independent of the input
    length), call :func:`scipy.signal.welch` directly.

    Parameters
    ----------
    signal :
        Shape (n_segments, n_samples, n_channels). Input signal.
    fs :
        Sampling frequency in Hz.
    segment :
        Fraction of ``n_samples`` used as the FFT window length, in
        ``(0, 1]``. Replaces the legacy ``segment_pct`` argument
        (``segment = segment_pct / 100``). Default 0.8.
    overlap :
        Fraction of the segment length used as the inter-segment overlap,
        in ``[0, 1)``. Replaces the legacy ``overlap_pct``. Default 0.5.
    window :
        Window function name passed through to :func:`scipy.signal.welch`.
        Default ``'boxcar'``.

    Returns
    -------
    f : NDArray
        Shape (n_frequencies,). Sampled frequencies in Hz.
    psd : NDArray
        Shape (n_segments, n_frequencies, n_channels). One-sided PSD.

    Examples
    --------
    >>> import numpy as np
    >>> from medusa.signal.transforms import power_spectral_density
    >>> sig = np.random.default_rng(0).standard_normal((1, 1024, 4))
    >>> f, psd = power_spectral_density(sig, fs=256.0,
    ...                                 segment=0.8, overlap=0.5)
    >>> psd.shape[0], psd.shape[2]
    (1, 4)
    """
    signal = check_data_dims(np.asarray(signal), rep_type='time_segments')[0]
    n_samp = signal.shape[1]
    nperseg = int(round(n_samp * segment))
    noverlap = int(round(n_samp * overlap))
    f, psd = welch_sp(signal, fs=fs, window=window, nperseg=nperseg,
                      noverlap=noverlap, axis=1)
    return f, psd


def normalize_psd(
    psd: NDArray,
    band: tuple[float, float] | None = None,
    fxx: NDArray | None = None,
    norm: Literal['rel', 'z'] = 'rel',
) -> NDArray:
    """Normalise a PSD by relative power or z-score along the frequency axis.

    Parameters
    ----------
    psd :
        Shape (n_segments, n_frequencies, n_channels). One-sided PSD.
        2-D ``(n_frequencies, n_channels)`` and 1-D ``(n_frequencies,)`` are
        promoted by :func:`medusa.core.utils.check_data_dims`.
    band :
        Optional frequency band ``(f_low, f_high)`` in Hz. When given, all
        frequencies outside the band are set to zero before the
        normalisation step. Requires ``fxx``.
    fxx :
        Shape (n_frequencies,). Frequency vector (Hz) corresponding to the
        ``n_frequencies`` axis of ``psd``. Required only when ``band`` is
        given.
    norm :
        Normalisation method. ``'rel'`` divides by the per-segment,
        per-channel total power; ``'z'`` standardises along the frequency
        axis (subtract mean, divide by std). Default ``'rel'``.

    Returns
    -------
    NDArray
        Same shape as the canonical input. Output dtype is the input dtype
        promoted to floating point.

    Raises
    ------
    ValueError
        If ``band`` is given without ``fxx``, or if ``norm`` is not one of
        ``'rel'``, ``'z'``.

    Examples
    --------
    >>> import numpy as np
    >>> from medusa.signal.transforms import normalize_psd
    >>> psd = np.random.default_rng(0).random((1, 129, 4))
    >>> rel = normalize_psd(psd, norm='rel')
    >>> np.allclose(rel.sum(axis=1), 1.0)
    True
    """
    if band is not None and fxx is None:
        raise ValueError(
            "If 'band' is given, the frequency vector 'fxx' must also be "
            "provided."
        )

    psd = check_data_dims(np.asarray(psd), rep_type='freq_segments')[0]

    if band is not None:
        psd = psd.copy()
        psd[:, fxx < band[0], :] = 0
        psd[:, fxx > band[1], :] = 0

    if norm == 'rel':
        p = np.sum(psd, axis=1, keepdims=True)
        return psd / p
    if norm == 'z':
        m = np.mean(psd, axis=1, keepdims=True)
        s = np.std(psd, axis=1, keepdims=True)
        return (psd - m) / s
    raise ValueError(
        f"Unknown normalisation {norm!r}. Choose 'rel' or 'z'."
    )

def fourier_spectrogram(
    signal: NDArray,
    fs: float,
    time_window: float = 1.0,
    overlap: float = 0.8,
    smooth: bool = True,
    smooth_sigma: float = 2.0,
    scale_to: Literal['magnitude', 'psd'] | None = None,
) -> tuple[NDArray, NDArray, NDArray]:
    """Compute the STFT spectrogram with a Gaussian window.

    Implementation based on
    https://github.com/drammock/spectrogram-tutorial.

    Parameters
    ----------
    signal :
        Shape (n_segments, n_samples, n_channels). Input signal. 2-D
        ``(n_samples, n_channels)`` and 1-D ``(n_samples,)`` arrays are
        promoted by :func:`medusa.core.utils.check_data_dims`.
    fs :
        Sampling frequency in Hz.
    time_window :
        Length in seconds of the Gaussian window used in the STFT.
    overlap :
        Fraction of the window length used as the inter-window overlap,
        in ``[0, 1)``. Replaces the legacy ``overlap_pct`` (``overlap =
        overlap_pct / 100``). Default 0.8.
    smooth :
        If True, apply a Gaussian filter to smooth the resulting
        spectrogram. Default True.
    smooth_sigma :
        Sigma of the Gaussian smoothing filter. Used only when
        ``smooth=True``.
    scale_to :
        Pass-through to :class:`scipy.signal.ShortTimeFFT` controlling the
        column scaling: ``'magnitude'``, ``'psd'`` or ``None``.

    Returns
    -------
    Sx : NDArray
        Shape (n_segments, n_frequencies, n_times, n_channels).
        Spectrogram of the input signal.
    times : NDArray
        Shape (n_times,). Time stamps in seconds.
    frequencies : NDArray
        Shape (n_frequencies,). Frequency values in Hz; the maximum is
        ``fs / 2``.

    Raises
    ------
    ValueError
        If ``overlap`` is not in ``[0, 1)``.

    Examples
    --------
    >>> import numpy as np
    >>> from medusa.signal.transforms import fourier_spectrogram
    >>> sig = np.random.default_rng(0).standard_normal((1, 2048, 4))
    >>> Sx, t, f = fourier_spectrogram(sig, fs=256.0, time_window=0.5,
    ...                                overlap=0.5, smooth=False)
    >>> Sx.shape[0], Sx.shape[3]
    (1, 4)
    """
    if not 0.0 <= overlap < 1.0:
        raise ValueError(
            f"overlap must be in [0, 1); got {overlap!r}."
        )

    signal = check_data_dims(np.asarray(signal), rep_type='time_segments')[0]
    n_segments, n_samples, n_channels = signal.shape

    window_nsamp = int(time_window * fs)
    window_sigma = (window_nsamp + 1) / 6
    win = windows.gaussian(window_nsamp, window_sigma, sym=True)

    noverlap = overlap * window_nsamp
    step_nsamp = int(window_nsamp - noverlap)

    sft = ShortTimeFFT(win=win, hop=step_nsamp, fs=fs, scale_to=scale_to)

    # Probe shape with the first slice to allocate the output array.
    probe = sft.spectrogram(signal[0, :, 0])
    n_frequencies, n_times = probe.shape
    out = np.empty(
        (n_segments, n_frequencies, n_times, n_channels), dtype=probe.dtype
    )
    out[0, :, :, 0] = probe
    for s in range(n_segments):
        for c in range(n_channels):
            if s == 0 and c == 0:
                continue
            out[s, :, :, c] = sft.spectrogram(signal[s, :, c])

    if smooth:
        # Smooth each (segment, channel) plane independently along (f, t).
        for s in range(n_segments):
            for c in range(n_channels):
                out[s, :, :, c] = ndimage.gaussian_filter(
                    out[s, :, :, c], sigma=smooth_sigma
                )

    t_lo, t_hi, f_lo, f_hi = sft.extent(n_samples)
    times = np.linspace(t_lo, t_hi, n_times)
    frequencies = np.linspace(f_lo, f_hi, n_frequencies)

    return out, times, frequencies

def __cone_of_influency(center_frequency: float, N: int, fs: float) -> NDArray:
    f_0 = center_frequency * 2 * np.pi
    cmor_flambda = 4 * np.pi / (f_0 + np.sqrt(2 + f_0 ** 2))
    coi = (1.0 / np.sqrt(2)) * cmor_flambda * (
            N / 2 - np.abs(np.arange(0, N) - (N - 1) / 2)) * (1.0 / fs)
    return 1.0/coi

def __compute_scales(N: int, filters_per_octave: int) -> NDArray:
    nOctaves = int(np.log2(2 * np.floor(N / 2.0)))
    scales = 2 ** np.arange(1, nOctaves, 1.0 / filters_per_octave)
    return scales

def cwt_spectrogram(
    signal: NDArray,
    fs: float,
    filters_per_octave: int = 5,
    center_frequency: float = 1.0,
    bandwidth_frequency: float = 1.5,
    smooth: bool = True,
    smooth_sigma: float = 2.0,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Compute the spectrogram via the CWT with complex Morlet wavelets.

    Parameters
    ----------
    signal :
        Shape (n_segments, n_samples, n_channels). Input signal. 2-D
        ``(n_samples, n_channels)`` and 1-D ``(n_samples,)`` arrays are
        promoted by :func:`medusa.core.utils.check_data_dims`.
    fs :
        Sampling frequency in Hz.
    filters_per_octave :
        Number of wavelet filters per octave. Default 5.
    center_frequency :
        Center frequency of the Morlet wavelet (``cmor`` parameter).
    bandwidth_frequency :
        Bandwidth of the Morlet wavelet (``cmor`` parameter).
    smooth :
        If True, apply a Gaussian filter to smooth the resulting
        spectrogram. Default True.
    smooth_sigma :
        Sigma of the Gaussian smoothing filter. Used only when
        ``smooth=True``.

    Returns
    -------
    power : NDArray
        Shape (n_segments, n_frequencies, n_times, n_channels). CWT power
        spectrogram, restricted to frequencies ``<= fs / 2``.
    times : NDArray
        Shape (n_times,). Time stamps in seconds (``n_times == n_samples``
        because the CWT preserves the time axis).
    frequencies : NDArray
        Shape (n_frequencies,). Frequency values in Hz, restricted to
        ``<= fs / 2``.
    coif : NDArray
        Shape (n_times,). Cone-of-influence values; identical across
        segments and channels because they only depend on
        ``(n_samples, fs, center_frequency)``.

    Examples
    --------
    >>> import numpy as np
    >>> from medusa.signal.transforms import cwt_spectrogram
    >>> sig = np.random.default_rng(0).standard_normal((1, 512, 4))
    >>> power, t, f, coif = cwt_spectrogram(sig, fs=256.0, smooth=False)
    >>> power.shape[0], power.shape[3], t.shape[0], coif.shape[0]
    (1, 4, 512, 512)
    """
    signal = check_data_dims(np.asarray(signal), rep_type='time_segments')[0]
    n_segments, n_samples, n_channels = signal.shape

    dt = 1.0 / fs
    times = np.arange(n_samples) * dt
    scales = __compute_scales(n_samples, filters_per_octave)
    wavelet = f'cmor{bandwidth_frequency}-{center_frequency}'

    frequencies = pywt.scale2frequency(wavelet, scales) / dt
    n_frequencies = len(scales)

    # Probe to discover output dtype; pywt.cwt returns complex-valued c.
    probe_c, _ = pywt.cwt(signal[0, :, 0], scales, wavelet)
    power = np.empty(
        (n_segments, n_frequencies, n_samples, n_channels),
        dtype=np.abs(probe_c).dtype,
    )
    probe_power = np.abs(probe_c * np.conj(probe_c))
    power[0, :, :, 0] = probe_power
    for s in range(n_segments):
        for c in range(n_channels):
            if s == 0 and c == 0:
                continue
            coeffs, _ = pywt.cwt(signal[s, :, c], scales, wavelet)
            power[s, :, :, c] = np.abs(coeffs * np.conj(coeffs))

    if smooth:
        for s in range(n_segments):
            for c in range(n_channels):
                power[s, :, :, c] = ndimage.gaussian_filter(
                    power[s, :, :, c], sigma=smooth_sigma
                )

    coif = __cone_of_influency(center_frequency, n_samples, fs)

    freq_mask = frequencies <= 0.5 * fs
    return (
        power[:, freq_mask, :, :],
        times,
        frequencies[freq_mask],
        coif,
    )

def cross_cwt(
    signal1: NDArray,
    signal2: NDArray,
    fs: float,
    mode: Literal['spectrogram', 'phase'] | None = 'spectrogram',
    filters_per_octave: int = 5,
    center_frequency: float = 1,
    bandwidth_frequency: float = 1.5,
    smooth: bool = True,
    smooth_sigma: float = 2,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Cross Wavelet Transform of two signals via complex Morlet wavelets.

    .. deprecated:: 2.0
        The current 1-D × 1-D contract and 2-D output do not fit the K9.B
        signal-shape conventions; the meaning of "channels" for a cross-
        spectrogram between two multi-channel signals (pairwise vs full
        outer-product) is also an unresolved design question. The function
        will be revised in a future release with a possibly incompatible
        signature. New code should treat its API as unstable.

    Implementation based on https://gist.github.com/MiguelonGonzalezParameters

    Parameters
    ----------
    signal1
        (n_samples,). First one-channel signal.
    signal2
        (n_samples,). Second one-channel signal.
    fs
        Sampling frequency of the signal in Hz.
    mode
        Choose the desired result. ``'spectrogram'`` returns the cross
        spectrogram between both signals while ``'phase'`` returns the cross
        phase. ``None`` is treated as ``'spectrogram'``.
    filters_per_octave
        Number of filters used to compute the CWT.
    center_frequency
        Center frequency used to compute the CWT using Morlet wavelets.
    bandwidth_frequency
        Band width used to compute the CWT using Morlet wavelets.
    smooth
        Define if a gaussian filter is used to smooth the final result.
    smooth_sigma
        Sigma value used for the gaussian filter if ``smooth`` is True.

    Returns
    -------
    power : numpy.ndarray
        (n_frequencies, n_samples). Spectrogram of the signal.
    times : numpy.ndarray
        (n_samples,). Time stamps.
    frequencies : numpy.ndarray
        (n_frequencies,). Frequency values; the maximum value is ``fs / 2``.
    coif : numpy.ndarray
        (n_samples,). Cone-of-influence values.

    Raises
    ------
    ValueError
        If either signal has more than one channel, or the two signals differ
        in length.
    """
    warnings.warn(
        "cross_cwt is deprecated and pending API revision in a future "
        "release: the 1-D × 1-D contract and 2-D output do not fit the "
        "K9.B signal-shape conventions, and the channel-pairing semantics "
        "for the multi-channel case are unresolved. The function still "
        "works for now but its signature may change.",
        DeprecationWarning,
        stacklevel=2,
    )

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




