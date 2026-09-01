"""Offline analysis for motor decoding: spectrograms, ERD/ERS and simple statistics.

Plain functions (no class, no fitting, no persistence) that turn motor-imagery / motor-execution
trials into arrays for inspection and plotting. They are the reusable NumPy layer *below* the
pipelines: give them epoched trials and they return arrays; the rendering is done separately by
:mod:`medusa.plots` (for example :func:`~medusa.plots.plot_timeheatmap` for an ERD/ERS map).

Three analyses, following the classic motor-imagery workflow:

* **Spectrograms** -- per-trial time-frequency power, either a Gaussian-window STFT
  (:func:`spectrogram`) or, for one band, the Hilbert envelope power
  (:func:`instantaneous_band_power`).
* **ERD/ERS** -- event-related de-/synchronization (:func:`erd_ers`): band power as a percent
  change from a pre-onset baseline, ``100 * (A - R) / R`` (Pfurtscheller). Negative values are
  ERD (a power drop, the hallmark of motor imagery over sensorimotor cortex), positive are ERS.
* **Statistics** -- a bootstrap significance mask for an ERD/ERS map
  (:func:`erd_ers_significance`) and a two-class discriminability map
  (:func:`class_discriminability`, signed r-squared).

The compute functions take plain arrays; :func:`motor_trials` is a small convenience that reads a
recording's trial timeline and cuts the epochs, so a whole analysis reads as
``epochs, labels, fs = motor_trials(rec, channels=..., window=(-2, 5))`` then
``power, times, freqs = spectrogram(epochs, fs, epoch_start=-2)`` then ``erd_ers`` ->
``erd_ers_significance``. Pass the epoch window's start as ``epoch_start`` so the spectrogram's
times come out onset-relative (it defaults to ``0`` and cannot know the window otherwise).

**Time convention:** every time axis and window here is in **seconds, relative to the trial
onset** (a baseline of ``(-2.0, 0.0)`` is the 2 s before onset). :func:`motor_trials` takes its
epoch window in seconds too and converts to the pipelines' millisecond convention internally.
"""
from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray

from medusa.core.data.recording import Recording
from medusa.signal.transforms import (
    fourier_spectrogram, cwt_spectrogram, hilbert, power_spectral_density)
from medusa.signal.frequency_filtering import IIRFilter
from medusa.signal.metrics.discriminability import signed_r2

from medusa.pipelines.bci.trial_events import trial_arrays, validate_trial_events
from medusa.pipelines.bci.motor_decoding.decoding._common import trial_segments

__all__ = [
    "motor_trials",
    "spectrogram",
    "instantaneous_band_power",
    "erd_ers",
    "erd_ers_significance",
    "class_discriminability",
    "trial_band_power",
    "trial_psd",
    "discriminability_spectrum",
    "optimal_band",
]


def _as_epochs(epochs: NDArray) -> NDArray:
    """Coerce ``epochs`` to a float array and require the ``(n_trials, n_samples, n_channels)`` shape.

    A 2-D array is ambiguous (is it one trial, or one channel?), so it is rejected with a clear
    error instead of being silently mis-shaped by the per-trial loop / the Hilbert transform.
    """
    epochs = np.asarray(epochs, dtype=float)
    if epochs.ndim != 3:
        raise ValueError(
            f"epochs must be 3-D (n_trials, n_samples, n_channels), got shape {epochs.shape}.")
    return epochs


# --------------------------------------------------------------------------- #
# Trials: read the trial timeline and cut broadband epochs
# --------------------------------------------------------------------------- #
def motor_trials(recording: Recording, *, channels: list[str], signal_key: str = "eeg",
                 band: tuple[float, float] = (1.0, 45.0), filt_order: int = 5,
                 car: bool = False, window: tuple[float, float] = (-2.0, 5.0),
                 target_fs: float = 0.0) -> tuple[NDArray, NDArray, float]:
    """Cut labelled motor trials from a recording: ``(epochs, labels, fs)``.

    A thin convenience over the shared trial contract
    (:func:`~medusa.pipelines.bci.trial_events.trial_arrays`) and the motor segmentation primitive
    (:func:`~medusa.pipelines.bci.motor_decoding.decoding._common.trial_segments`). It picks and
    reorders the channels, optionally applies a common-average reference, band-passes to a
    **broadband** range (so a spectrogram can resolve several frequencies), and segments each
    trial around its onset. No baseline normalization is applied at cut time: the ERD/ERS
    baseline is handled later, inside :func:`erd_ers`.

    Parameters
    ----------
    recording :
        A recording whose events carry the trial timeline (one row per trial; see
        :data:`~medusa.pipelines.bci.trial_events.TRIAL_EVENT_COLUMNS`).
    channels :
        Channel labels to keep, in the order the returned epochs will use.
    signal_key :
        Which signal stream to read (default ``"eeg"``).
    band :
        ``(low, high)`` band-pass cut-offs in Hz. Broadband by default so a spectrogram sees
        the full mu/beta range; narrow it for a single-band analysis.
    filt_order :
        IIR band-pass order.
    car :
        Apply a common-average reference before filtering. ``False`` by default (CAR can
        rank-deficient a small montage).
    window :
        ``(start, end)`` epoch window in **seconds** relative to each trial onset. The default
        ``(-2, 5)`` keeps a 2 s pre-onset baseline and 5 s of the trial.
    target_fs :
        Resample the epochs to this rate (Hz); ``0`` keeps the native rate.

    Returns
    -------
    epochs : numpy.ndarray
        ``(n_trials, n_samples, n_channels)`` band-passed trial epochs.
    labels : numpy.ndarray
        ``(n_trials,)`` float class labels (``nan`` where a trial is unlabelled, e.g. a test run).
    fs : float
        Sampling rate of the returned epochs (``target_fs`` if it was set, else the native rate).
    """
    validate_trial_events(recording.events)
    onsets, _, labels = trial_arrays(recording.events)
    signal = recording.signals[signal_key]
    filter_spec = {"filt_type": "iir", "band_type": "bandpass",
                   "cutoff": list(band), "order": int(filt_order)}
    window_ms = (window[0] * 1000.0, window[1] * 1000.0)
    epochs = trial_segments(signal, onsets, channels=channels, apply_car=car,
                          filter_spec=filter_spec, window=window_ms, baseline=None,
                          target_fs=target_fs)
    fs = float(target_fs) if target_fs else float(signal.fs)
    return epochs, labels, fs


# --------------------------------------------------------------------------- #
# Spectrograms: per-trial time-frequency power
# --------------------------------------------------------------------------- #
def spectrogram(epochs: NDArray, fs: float, *, method: str = "fourier",
                epoch_start: float = 0.0, time_window: float = 1.0, overlap: float = 0.9,
                freq_range: tuple[float, float] | None = None, smooth: bool = True,
                smooth_sigma: float = 2.0) -> tuple[NDArray, NDArray, NDArray]:
    """Per-trial time-frequency power of epoched trials.

    A thin wrapper over the kernel time-frequency transforms
    (:func:`~medusa.signal.transforms.fourier_spectrogram` /
    :func:`~medusa.signal.transforms.cwt_spectrogram`) that keeps the trial axis and puts the
    time axis onto trial-onset-relative seconds. The returned ``power`` is the ``A`` fed to
    :func:`erd_ers`.

    Parameters
    ----------
    epochs :
        ``(n_trials, n_samples, n_channels)`` trial epochs.
    fs :
        Sampling rate (Hz).
    method :
        ``"fourier"`` -- a Gaussian-window STFT (coarser time, even frequency spacing).
        ``"wavelet"`` -- a complex-Morlet CWT (full time resolution, log-spaced frequency).
    epoch_start :
        Time of the epoch's first sample relative to the trial onset, in **seconds** (e.g.
        ``-2.0`` for an epoch that starts 2 s before onset). Added to the returned ``times`` so
        they are onset-relative.
    time_window, overlap :
        STFT window length (s) and window overlap fraction (``method="fourier"`` only).
    freq_range :
        Optional ``(low, high)`` Hz band to crop the output frequencies to.
    smooth, smooth_sigma :
        Optional Gaussian smoothing of the time-frequency plane.

    Returns
    -------
    power : numpy.ndarray
        ``(n_trials, n_freqs, n_times, n_channels)`` time-frequency power.
    times : numpy.ndarray
        ``(n_times,)`` onset-relative time in seconds.
    freqs : numpy.ndarray
        ``(n_freqs,)`` frequency in Hz.
    """
    epochs = _as_epochs(epochs)
    if method == "fourier":
        power, times, freqs = fourier_spectrogram(
            epochs, fs, time_window=time_window, overlap=overlap,
            smooth=smooth, smooth_sigma=smooth_sigma)
    elif method == "wavelet":
        power, times, freqs, _ = cwt_spectrogram(
            epochs, fs, smooth=smooth, smooth_sigma=smooth_sigma)
    else:
        raise ValueError(f"method must be 'fourier' or 'wavelet', got {method!r}.")
    times = np.asarray(times, dtype=float) + epoch_start
    freqs = np.asarray(freqs, dtype=float)
    if freq_range is not None:
        lo, hi = freq_range
        keep = (freqs >= lo) & (freqs <= hi)
        if not keep.any():
            raise ValueError(
                f"freq_range {freq_range} selects no frequencies; available "
                f"[{freqs.min():.4g}, {freqs.max():.4g}] Hz.")
        power, freqs = power[:, keep], freqs[keep]
    return power, times, freqs


def instantaneous_band_power(epochs: NDArray, fs: float, band: tuple[float, float], *,
                             epoch_start: float = 0.0, order: int = 5
                             ) -> tuple[NDArray, NDArray]:
    """Instantaneous power in one frequency band, at full time resolution.

    Band-passes each trial, then squares the Hilbert envelope
    (:func:`~medusa.signal.transforms.hilbert`) to get the instantaneous band power at every
    sample. Unlike :func:`spectrogram` this keeps every time sample (no windowing), so it is the
    natural input to an ERD/ERS *time course* in a chosen band (e.g. mu 8--13 Hz).

    Parameters
    ----------
    epochs :
        ``(n_trials, n_samples, n_channels)`` trial epochs.
    fs :
        Sampling rate (Hz).
    band :
        ``(low, high)`` band-pass cut-offs in Hz.
    epoch_start :
        Time of the first sample relative to the trial onset, in seconds (see :func:`spectrogram`).
    order :
        IIR band-pass order.

    Returns
    -------
    power : numpy.ndarray
        ``(n_trials, n_samples, n_channels)`` instantaneous band power.
    times : numpy.ndarray
        ``(n_samples,)`` onset-relative time in seconds.
    """
    epochs = _as_epochs(epochs)
    filt = IIRFilter(order, tuple(band), "bandpass")
    filt.fit(fs)
    # IIRFilter works on one (n_samples, n_channels) trial at a time; loop over trials.
    filtered = np.stack([filt.transform(epochs[i]) for i in range(len(epochs))])
    power = np.abs(hilbert(filtered)) ** 2
    times = epoch_start + np.arange(epochs.shape[1]) / fs
    return power, times


# --------------------------------------------------------------------------- #
# ERD/ERS: band power as a percent change from a baseline
# --------------------------------------------------------------------------- #
def erd_ers(power: NDArray, times: NDArray, baseline: tuple[float, float], *,
            ref_mode: str = "classic", average: bool = True) -> NDArray:
    """Event-related de-/synchronization: band power as a percent change from a baseline.

    The classic Pfurtscheller transform ``ERD/ERS = 100 * (A - R) / R``, where ``A`` is the band
    power at each point and ``R`` is the reference (baseline) power. Negative values are ERD (a
    power decrease, e.g. the mu/beta drop during motor imagery), positive values are ERS.

    It is agnostic to how ``power`` was produced -- a :func:`spectrogram` map
    ``(n_trials, n_freqs, n_times, n_channels)`` or an :func:`instantaneous_band_power` trace
    ``(n_trials, n_samples, n_channels)``. In both the **time axis is the second-to-last** and
    the trial axis is the first; the baseline is taken over the time axis.

    Parameters
    ----------
    power :
        ``(n_trials, ..., n_times, n_channels)`` band power. Time is axis ``-2``.
    times :
        ``(n_times,)`` onset-relative time in seconds (from :func:`spectrogram` /
        :func:`instantaneous_band_power`).
    baseline :
        ``(start, end)`` reference window in seconds. Must lie inside ``times``.
    ref_mode :
        ``"classic"`` -- one reference per point, averaged over the baseline window *and* over
        trials (the standard grand-average baseline). ``"trial"`` -- a separate reference per
        trial (each trial referenced to its own baseline).
    average :
        Return the trial-averaged map (``True``) or keep the per-trial ERD/ERS (``False``, needed
        by :func:`erd_ers_significance`).

    Returns
    -------
    numpy.ndarray
        ERD/ERS in percent. With ``average`` the trial axis is dropped
        (``(..., n_times, n_channels)``); otherwise it is kept (``(n_trials, ..., n_times,
        n_channels)``).
    """
    power = np.asarray(power, dtype=float)
    times = np.asarray(times, dtype=float)
    if power.shape[-2] != times.shape[0]:
        raise ValueError(
            f"times has {times.shape[0]} points but power's time axis (-2) has "
            f"{power.shape[-2]}.")
    lo, hi = float(baseline[0]), float(baseline[1])
    if hi < lo:
        raise ValueError(f"baseline start must be <= end, got {baseline}.")
    if lo < times[0] or hi > times[-1]:
        raise ValueError(
            f"baseline {baseline} lies outside the epoch [{times[0]:.4g}, {times[-1]:.4g}] s.")
    mask = (times >= lo) & (times <= hi)
    if not mask.any():
        raise ValueError(f"baseline {baseline} selects no time samples.")

    take = [slice(None)] * power.ndim
    take[-2] = mask
    ref = power[tuple(take)].mean(axis=-2, keepdims=True)     # reference power per point
    if ref_mode == "classic":
        ref = ref.mean(axis=0, keepdims=True)                # grand-average baseline over trials
    elif ref_mode != "trial":
        raise ValueError(f"ref_mode must be 'classic' or 'trial', got {ref_mode!r}.")
    if not np.all(ref):
        raise ValueError(
            "baseline reference power is zero at one or more points (a flat or dead channel, or "
            "a zero-filled baseline?); ERD/ERS is undefined there. Drop those channels first.")
    erds_trials = 100.0 * (power - ref) / ref
    return erds_trials.mean(axis=0) if average else erds_trials


# --------------------------------------------------------------------------- #
# Statistics: bootstrap significance + two-class discriminability
# --------------------------------------------------------------------------- #
def erd_ers_significance(erds_trials: NDArray, *, alpha: float = 0.05, n_boot: int = 1000,
                         statistic: str = "mean", null: float = 0.0,
                         rng: "int | np.random.Generator | None" = None
                         ) -> "tuple[NDArray, NDArray, NDArray]":
    """Bootstrap significance of an ERD/ERS map against the baseline.

    A non-parametric test of "is this ERD/ERS real?". It is a plain **percentile** bootstrap: for
    every point it resamples the trials with replacement, builds the bootstrap distribution of the
    across-trial ``statistic`` (mean or median), and forms a two-sided percentile confidence
    interval. A point is **significant** when that interval excludes ``null`` (``0`` = no change
    from baseline) -- i.e. the de-/synchronization is reliably away from zero.

    The legacy ``bootts`` used a *studentized* bootstrap-t; this shares its intent (the
    CI-excludes-zero rule) but not its exact interval, so significance masks are not numerically
    identical to that MATLAB pipeline. The percentile interval is first-order and can be slightly
    anti-conservative for small, skewed samples.

    Parameters
    ----------
    erds_trials :
        ``(n_trials, ...)`` per-trial ERD/ERS in percent (``erd_ers(..., average=False)``). The
        bootstrap resamples the leading trial axis; the remaining axes are the map.
    alpha :
        Significance level; the interval is the ``alpha/2`` .. ``1 - alpha/2`` percentiles.
    n_boot :
        Number of bootstrap resamples. Memory scales as ``n_boot`` times the map size, so reduce
        it (or the number of channels) for a very large time-frequency map.
    statistic :
        ``"mean"`` or ``"median"`` across trials.
    null :
        The value the interval must exclude to be significant (``0`` for an ERD/ERS percent map).
    rng :
        Seed or :class:`numpy.random.Generator` for reproducibility.

    Returns
    -------
    mask : numpy.ndarray
        Boolean map (the shape of one trial), ``True`` where the ERD/ERS is significant.
    low, high : numpy.ndarray
        The confidence-interval bounds, same shape as ``mask``.
    """
    erds_trials = np.asarray(erds_trials, dtype=float)
    n_trials = erds_trials.shape[0]
    if n_trials < 2:
        raise ValueError(f"need at least 2 trials to bootstrap, got {n_trials}.")
    if statistic == "mean":
        reduce = np.mean
    elif statistic == "median":
        reduce = np.median
    else:
        raise ValueError(f"statistic must be 'mean' or 'median', got {statistic!r}.")
    min_boot = int(np.ceil(2.0 / alpha))
    if n_boot < min_boot:
        warnings.warn(
            f"n_boot={n_boot} is too few to resolve alpha={alpha}: the {alpha / 2:.4g} tail "
            f"quantile needs at least {min_boot} resamples (ceil(2/alpha)); the confidence "
            f"interval will be unstable and anti-conservative.", stacklevel=2)
    rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)

    map_shape = erds_trials.shape[1:]
    boot = np.empty((n_boot, *map_shape))
    for b in range(n_boot):
        idx = rng.integers(0, n_trials, n_trials)
        boot[b] = reduce(erds_trials[idx], axis=0)
    low = np.quantile(boot, alpha / 2, axis=0)
    high = np.quantile(boot, 1.0 - alpha / 2, axis=0)
    mask = (low > null) | (high < null)
    return mask, low, high


def class_discriminability(feature: NDArray, labels: NDArray, *,
                           classes: tuple[float, float] | None = None,
                           signed: bool = True) -> NDArray:
    """Two-class discriminability map: the signed r-squared between two label groups.

    For every point it computes the squared point-biserial correlation
    (:func:`~medusa.signal.metrics.discriminability.signed_r2`) between the two classes' trials --
    how well that point separates them. Applied to a spectrogram or an ERD/ERS map it shows
    *where* (time, frequency, channel) the two motor classes differ; applied to band-power
    features it is a per-channel discriminability.

    Parameters
    ----------
    feature :
        ``(n_trials, ...)`` per-trial feature (e.g. an ERD/ERS map, a spectrogram, or band power).
        The trial axis is the leading one.
    labels :
        ``(n_trials,)`` class labels. ``nan`` entries (unlabelled trials) are ignored.
    classes :
        The two labels to contrast, as ``(a, b)``. Defaults to the two labels present (an error
        if there are not exactly two).
    signed :
        Keep the sign of the class-mean difference (which class is larger), or return the
        unsigned r-squared.

    Returns
    -------
    numpy.ndarray
        The (signed) r-squared for each point, with the trial axis removed.
    """
    feature = np.asarray(feature, dtype=float)
    labels = np.asarray(labels, dtype=float)
    present = np.unique(labels[~np.isnan(labels)])
    if classes is None:
        if present.size != 2:
            raise ValueError(
                f"need exactly two classes, found {present.tolist()}; pass classes=(a, b).")
        a, b = present
    else:
        a, b = classes
    c1, c2 = feature[labels == a], feature[labels == b]
    if len(c1) == 0 or len(c2) == 0:
        raise ValueError(f"one of the classes {(a, b)} has no trials.")
    return signed_r2(c1, c2, signed=signed, axis=0)


# --------------------------------------------------------------------------- #
# Band features + subject-specific optimal band
# --------------------------------------------------------------------------- #
def trial_band_power(epochs: NDArray, fs: float, band: tuple[float, float], *,
                     epoch_start: float = 0.0, window: tuple[float, float] | None = None,
                     order: int = 5) -> NDArray:
    """Per-trial mean power in one band: ``(n_trials, n_channels)``.

    Band-passes each trial and takes the Hilbert-envelope power
    (:func:`instantaneous_band_power`), then averages over time -- over the whole epoch, or over
    a ``window`` ``(start, end)`` in seconds (e.g. the imagery period). One scalar per trial and
    channel, ready for a scalp topography or a per-band class comparison.

    Parameters
    ----------
    epochs :
        ``(n_trials, n_samples, n_channels)`` trial epochs.
    fs :
        Sampling rate (Hz).
    band :
        ``(low, high)`` band in Hz.
    epoch_start :
        Time of the first sample relative to the trial onset, in seconds.
    window :
        Optional ``(start, end)`` time window in seconds to average over (onset-relative).
    order :
        IIR band-pass order.

    Returns
    -------
    numpy.ndarray
        ``(n_trials, n_channels)`` mean band power.
    """
    power, times = instantaneous_band_power(epochs, fs, band, epoch_start=epoch_start, order=order)
    if window is not None:
        sel = (times >= window[0]) & (times <= window[1])
        if not sel.any():
            raise ValueError(
                f"window {window} selects no samples of [{times[0]:.4g}, {times[-1]:.4g}] s.")
        power = power[:, sel]
    return power.mean(axis=1)


def trial_psd(epochs: NDArray, fs: float, *, epoch_start: float = 0.0,
              window: tuple[float, float] | None = None,
              freq_range: tuple[float, float] = (1.0, 40.0),
              segment: float = 0.5, overlap: float = 0.25) -> tuple[NDArray, NDArray]:
    """Per-trial power spectral density: ``(psd (n_trials, n_freqs, n_channels), freqs)``.

    Welch power spectrum (:func:`~medusa.signal.transforms.power_spectral_density`) of each
    trial -- over the whole epoch, or over a ``window`` ``(start, end)`` in seconds -- cropped to
    ``freq_range``. The spectral counterpart of :func:`trial_band_power`, and the building block
    of :func:`discriminability_spectrum`; average it within a class for a per-class mean spectrum.

    Parameters
    ----------
    epochs :
        ``(n_trials, n_samples, n_channels)`` trial epochs.
    fs :
        Sampling rate (Hz).
    epoch_start :
        Time of the first sample relative to the trial onset, in seconds.
    window :
        Optional ``(start, end)`` time window in seconds to analyse (the whole epoch if ``None``).
    freq_range :
        ``(low, high)`` Hz range to keep.
    segment, overlap :
        Welch segment length and overlap, as fractions of the analysed samples
        (``overlap`` must be smaller than ``segment``).

    Returns
    -------
    psd : numpy.ndarray
        ``(n_trials, n_freqs, n_channels)`` power spectral density.
    freqs : numpy.ndarray
        ``(n_freqs,)`` frequency in Hz.
    """
    epochs = _as_epochs(epochs)
    if window is not None:
        times = epoch_start + np.arange(epochs.shape[1]) / fs
        sel = (times >= window[0]) & (times <= window[1])
        if not sel.any():
            raise ValueError(
                f"window {window} selects no samples of [{times[0]:.4g}, {times[-1]:.4g}] s.")
        epochs = epochs[:, sel]
    freqs, psd = power_spectral_density(epochs, fs, segment=segment, overlap=overlap)
    freqs = np.asarray(freqs, dtype=float)
    keep = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    if not keep.any():
        raise ValueError(
            f"freq_range {freq_range} selects no frequencies; available "
            f"[{freqs.min():.4g}, {freqs.max():.4g}] Hz.")
    return psd[:, keep], freqs[keep]


def discriminability_spectrum(epochs: NDArray, labels: NDArray, fs: float, *,
                              epoch_start: float = 0.0,
                              window: tuple[float, float] | None = None,
                              freq_range: tuple[float, float] = (1.0, 40.0),
                              segment: float = 0.5, overlap: float = 0.25,
                              signed: bool = True) -> tuple[NDArray, NDArray]:
    """Two-class discriminability (signed r-squared) at every frequency and channel.

    Computes each trial's power spectrum (Welch,
    :func:`~medusa.signal.transforms.power_spectral_density`) over the analysis ``window``, then
    the class discriminability (:func:`class_discriminability`) per frequency bin. The ``|r2|``
    profile over frequency shows which band best separates the two classes for *this* data --
    feed it to :func:`optimal_band` to pick a subject-specific band.

    Parameters
    ----------
    epochs :
        ``(n_trials, n_samples, n_channels)`` trial epochs.
    labels :
        ``(n_trials,)`` class labels (``nan`` trials are ignored; exactly two classes).
    fs :
        Sampling rate (Hz).
    epoch_start :
        Time of the first sample relative to the trial onset, in seconds.
    window :
        Optional ``(start, end)`` time window in seconds to analyse (the whole epoch if ``None``).
    freq_range :
        ``(low, high)`` Hz range to keep.
    segment, overlap :
        Welch segment length and overlap, as fractions of the analysed samples
        (``overlap`` must be smaller than ``segment``).
    signed :
        Keep the sign of the class-mean difference (which class has more power).

    Returns
    -------
    r2 : numpy.ndarray
        ``(n_freqs, n_channels)`` signed r-squared.
    freqs : numpy.ndarray
        ``(n_freqs,)`` frequency in Hz.
    """
    psd, freqs = trial_psd(epochs, fs, epoch_start=epoch_start, window=window,
                           freq_range=freq_range, segment=segment, overlap=overlap)
    return class_discriminability(psd, labels, signed=signed), freqs


def optimal_band(r2_spectrum: NDArray, freqs: NDArray, *,
                 channels: "list[int] | None" = None, min_width: float = 2.0,
                 rel_height: float = 0.5, smooth_hz: float = 2.0) -> tuple[float, float]:
    """The frequency band that best separates the two classes -- for subject-specific analysis.

    Motor-imagery discriminative frequencies vary by subject, so a fixed band (mu, beta, ...) is
    rarely optimal. Given the :func:`discriminability_spectrum` output, this aggregates ``|r2|``
    over the selected channels into a per-frequency profile, **smooths** it (single-bin r-squared
    peaks are noisy, so the choice is band-based, not bin-based), finds the peak, and returns the
    contiguous band around it down to ``rel_height`` of the peak height (half-maximum by default),
    widened to at least ``min_width`` Hz and clamped to the analysed range.

    Parameters
    ----------
    r2_spectrum :
        ``(n_freqs, n_channels)`` signed r-squared (from :func:`discriminability_spectrum`).
    freqs :
        ``(n_freqs,)`` frequency in Hz (assumed ascending, roughly evenly spaced).
    channels :
        Indices of the channels to base the choice on (e.g. the sensorimotor channels); all
        channels when ``None``.
    min_width :
        Minimum band width in Hz.
    rel_height :
        Fraction of the peak height that bounds the band (``0.5`` = full width at half maximum).
    smooth_hz :
        Width (Hz) of the moving-average smoothing applied to the profile before peak-finding
        (``0`` disables it).

    Returns
    -------
    tuple of float
        ``(low, high)`` of the selected band in Hz.
    """
    r2 = np.abs(np.asarray(r2_spectrum, dtype=float))
    freqs = np.asarray(freqs, dtype=float)
    if channels is not None:
        r2 = r2[:, channels]
    profile = r2.mean(axis=1) if r2.ndim > 1 else r2

    df = float(np.median(np.diff(freqs))) if freqs.size > 1 else 1.0
    width = int(round(smooth_hz / df)) if df > 0 else 0
    if width > 1:                                            # moving-average smooth the profile
        profile = np.convolve(profile, np.ones(width) / width, mode="same")

    peak = int(profile.argmax())
    thresh = rel_height * profile[peak]
    lo_i = peak
    while lo_i > 0 and profile[lo_i - 1] >= thresh:
        lo_i -= 1
    hi_i = peak
    while hi_i < len(profile) - 1 and profile[hi_i + 1] >= thresh:
        hi_i += 1
    lo, hi = float(freqs[lo_i]), float(freqs[hi_i])
    if hi - lo < min_width:                                  # enforce a minimum width, then clamp
        center = float(freqs[peak])
        lo, hi = center - min_width / 2, center + min_width / 2
    return max(lo, float(freqs[0])), min(hi, float(freqs[-1]))
