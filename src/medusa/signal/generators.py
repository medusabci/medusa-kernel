"""Synthetic biosignal generators.

This module provides a small family of *signal generators*: configurable
objects that synthesise continuous signals on demand. They are primarily
meant for testing, prototyping and teaching — e.g. feeding deterministic or
statistically-controlled signals into filters, metrics or BCI pipelines
without acquisition hardware.

Two families are provided:

- **Elementary waveforms** — :class:`SinusoidalSignalGenerator`,
  :class:`SquareSignalGenerator`, :class:`SawtoothSignalGenerator`,
  :class:`RampSignalGenerator`, :class:`ImpulseSignalGenerator` and
  :class:`ChirpSignalGenerator`.
- **Biologically plausible biosignal models** —
  :class:`EEGSignalGenerator` (1/f background plus oscillatory rhythms),
  :class:`ECGSignalGenerator` (PQRST Gaussian model) and
  :class:`EOGSignalGenerator` (blinks, saccades and slow drift).

All generators follow the medusa-kernel signal-shape contract: every
:meth:`SignalGenerator.get_chunk` call returns an array with the canonical
``'time'`` shape ``(n_samples, n_channels)``.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy import signal as sp_signal

__all__ = [
    'SignalGenerator',
    'SinusoidalSignalGenerator',
    'SquareSignalGenerator',
    'SawtoothSignalGenerator',
    'RampSignalGenerator',
    'ImpulseSignalGenerator',
    'ChirpSignalGenerator',
    'EEGSignalGenerator',
    'ECGSignalGenerator',
    'EOGSignalGenerator',
]

_NoiseType = Literal['white', 'pink']


def _pink_noise(
    n_samples: int,
    n_channels: int,
    rng: np.random.Generator,
    exponent: float = 1.0,
) -> NDArray:
    """Generate unit-variance power-law (``1/f**exponent``) noise.

    The noise is synthesised by shaping the spectrum of Gaussian white
    noise in the frequency domain. ``exponent=0`` yields white noise,
    ``exponent=1`` pink (``1/f``) noise and ``exponent=2`` brown noise.

    Parameters
    ----------
    n_samples
        Number of samples per channel.
    n_channels
        Number of channels.
    rng
        Random generator used to draw the underlying white noise.
    exponent
        Power-law exponent of the power spectral density.

    Returns
    -------
    NDArray
        Shape ``(n_samples, n_channels)``. Noise normalised to unit
        standard deviation per channel.
    """
    white = rng.standard_normal((n_samples, n_channels))
    freqs = np.fft.rfftfreq(n_samples)
    scaling = np.ones_like(freqs)
    scaling[1:] = 1.0 / (freqs[1:] ** (exponent / 2.0))
    spectrum = np.fft.rfft(white, axis=0) * scaling[:, None]
    noise = np.fft.irfft(spectrum, n=n_samples, axis=0)
    noise = noise / (np.std(noise, axis=0, keepdims=True) + 1e-12)
    return noise


class SignalGenerator(ABC):
    """Abstract base class for synthetic signal generators.

    A generator is configured once (sampling rate, optional measurement
    noise, random seed) and can then produce arbitrarily many independent
    chunks via :meth:`get_chunk`. Subclasses implement
    :meth:`_synthesise`, which returns the noiseless waveform; the base
    class adds the optional measurement noise and enforces the output
    shape contract.

    Parameters
    ----------
    fs
        Sampling frequency in Hz. Must be greater than 0.
    noise_type
        Measurement noise added on top of the synthesised waveform.
        ``'white'`` for Gaussian white noise (``mean``, ``sigma`` taken
        from ``noise_params``) or ``'pink'`` for ``1/f`` noise
        (``sigma``, ``exponent`` from ``noise_params``). ``None`` adds no
        noise.
    noise_params
        Parameters of the noise model. For ``'white'``: ``mean``
        (default 0.0) and ``sigma`` (default 1.0). For ``'pink'``:
        ``sigma`` (default 1.0) and ``exponent`` (default 1.0).
    seed
        Seed for the internal :class:`numpy.random.Generator`. Pass an
        integer for reproducible output.

    Notes
    -----
    Randomness uses a per-instance :class:`numpy.random.Generator`
    (``self.rng``), never the legacy global ``np.random`` state, so two
    generators seeded identically produce identical signals.
    """

    def __init__(
        self,
        fs: float,
        noise_type: _NoiseType | None = None,
        noise_params: dict | None = None,
        seed: int | None = None,
    ) -> None:
        if fs <= 0:
            raise ValueError('Parameter fs must be greater than 0')
        if noise_type not in (None, 'white', 'pink'):
            raise ValueError(
                "Parameter noise_type must be None, 'white' or 'pink'")
        self.fs = float(fs)
        self.noise_type = noise_type
        self.noise_params = dict(noise_params) if noise_params else {}
        self.rng = np.random.default_rng(seed)

    # -- Helpers available to subclasses -----------------------------------

    def _n_samples(self, duration: float) -> int:
        """Number of samples in a chunk of ``duration`` seconds."""
        if duration <= 0:
            raise ValueError('Parameter duration must be greater than 0')
        return int(round(duration * self.fs))

    def _time_vector(self, duration: float) -> NDArray:
        """Time vector (in seconds) of shape ``(n_samples,)``."""
        return np.arange(self._n_samples(duration)) / self.fs

    def _add_noise(self, chunk: NDArray) -> NDArray:
        """Add the configured measurement noise to ``chunk``."""
        if self.noise_type is None:
            return chunk
        if self.noise_type == 'white':
            mean = self.noise_params.get('mean', 0.0)
            sigma = self.noise_params.get('sigma', 1.0)
            return chunk + self.rng.normal(mean, sigma, chunk.shape)
        # 'pink'
        sigma = self.noise_params.get('sigma', 1.0)
        exponent = self.noise_params.get('exponent', 1.0)
        noise = _pink_noise(
            chunk.shape[0], chunk.shape[1], self.rng, exponent)
        return chunk + sigma * noise

    # -- Public / abstract API ---------------------------------------------

    @abstractmethod
    def _synthesise(self, duration: float, n_channels: int) -> NDArray:
        """Return the noiseless waveform of shape ``(n_samples, n_channels)``."""

    def get_chunk(self, duration: float, n_channels: int = 1) -> NDArray:
        """Generate a chunk of signal.

        Parameters
        ----------
        duration
            Duration of the chunk in seconds. Must be greater than 0.
        n_channels
            Number of channels to generate.

        Returns
        -------
        NDArray
            Shape ``(n_samples, n_channels)`` where
            ``n_samples == round(duration * fs)``.
        """
        if n_channels < 1:
            raise ValueError('Parameter n_channels must be greater than 0')
        chunk = np.asarray(self._synthesise(duration, n_channels), dtype=float)
        return self._add_noise(chunk)

    def __repr__(self) -> str:
        return (f'{type(self).__name__}(fs={self.fs}, '
                f'noise_type={self.noise_type!r})')


class SinusoidalSignalGenerator(SignalGenerator):
    """Sum-of-sinusoids generator.

    Generates a signal made of one or more superimposed sinusoids, shared
    across all channels, optionally corrupted with measurement noise.

    Parameters
    ----------
    fs
        Sampling frequency in Hz.
    freqs
        Frequencies of the sinusoids in Hz.
    amplitudes
        Amplitude of each sinusoid. If ``None``, every sinusoid has unit
        amplitude. Must match the length of ``freqs`` when provided.
    phases
        Initial phase of each sinusoid in radians. If ``None``, all phases
        are 0. Must match the length of ``freqs`` when provided.
    noise_type
        Measurement noise model; see :class:`SignalGenerator`. Defaults to
        ``'white'`` for backward compatibility.
    noise_params
        Noise parameters; see :class:`SignalGenerator`.
    seed
        Seed for reproducible noise.

    Examples
    --------
    >>> gen = SinusoidalSignalGenerator(fs=200, freqs=[1, 5], noise_type=None)
    >>> chunk = gen.get_chunk(10, 5)
    >>> chunk.shape
    (2000, 5)
    >>> float(round(chunk[250, 0], 6))  # sin(2*pi*1*1.25) + sin(2*pi*5*1.25)
    2.0
    """

    def __init__(
        self,
        fs: float,
        freqs: Sequence[float],
        amplitudes: Sequence[float] | None = None,
        phases: Sequence[float] | None = None,
        noise_type: _NoiseType | None = 'white',
        noise_params: dict | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(fs, noise_type, noise_params, seed)
        self.freqs = np.asarray(freqs, dtype=float)
        if amplitudes is None:
            self.amplitudes = np.ones_like(self.freqs)
        else:
            self.amplitudes = np.asarray(amplitudes, dtype=float)
        if phases is None:
            self.phases = np.zeros_like(self.freqs)
        else:
            self.phases = np.asarray(phases, dtype=float)
        if not (self.freqs.shape == self.amplitudes.shape == self.phases.shape):
            raise ValueError(
                'Parameters freqs, amplitudes and phases must have the same '
                'length')

    def _synthesise(self, duration: float, n_channels: int) -> NDArray:
        t = self._time_vector(duration)[:, None]
        wave = np.zeros((t.shape[0], 1))
        for freq, amp, phase in zip(self.freqs, self.amplitudes, self.phases):
            wave += amp * np.sin(2 * np.pi * freq * t + phase)
        return np.tile(wave, (1, n_channels))


class SquareSignalGenerator(SignalGenerator):
    """Square-wave generator.

    Parameters
    ----------
    fs
        Sampling frequency in Hz.
    freq
        Fundamental frequency of the square wave in Hz.
    amplitude
        Peak amplitude of the wave.
    duty
        Duty cycle in ``[0, 1]`` (fraction of the period spent high).
    offset
        Constant value added to the wave.
    noise_type, noise_params, seed
        See :class:`SignalGenerator`.

    Examples
    --------
    >>> gen = SquareSignalGenerator(fs=250, freq=5, duty=0.25)
    >>> gen.get_chunk(1.0, 2).shape
    (250, 2)
    """

    def __init__(
        self,
        fs: float,
        freq: float,
        amplitude: float = 1.0,
        duty: float = 0.5,
        offset: float = 0.0,
        noise_type: _NoiseType | None = None,
        noise_params: dict | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(fs, noise_type, noise_params, seed)
        if not 0.0 <= duty <= 1.0:
            raise ValueError('Parameter duty must lie in [0, 1]')
        self.freq = float(freq)
        self.amplitude = float(amplitude)
        self.duty = float(duty)
        self.offset = float(offset)

    def _synthesise(self, duration: float, n_channels: int) -> NDArray:
        t = self._time_vector(duration)
        wave = self.amplitude * sp_signal.square(
            2 * np.pi * self.freq * t, duty=self.duty) + self.offset
        return np.tile(wave[:, None], (1, n_channels))


class SawtoothSignalGenerator(SignalGenerator):
    """Sawtooth / triangle-wave generator.

    Parameters
    ----------
    fs
        Sampling frequency in Hz.
    freq
        Fundamental frequency in Hz.
    amplitude
        Peak amplitude of the wave.
    width
        Rising-ramp fraction of the period in ``[0, 1]``. ``1.0`` gives a
        rising sawtooth, ``0.0`` a falling sawtooth and ``0.5`` a
        symmetric triangle wave.
    offset
        Constant value added to the wave.
    noise_type, noise_params, seed
        See :class:`SignalGenerator`.

    Examples
    --------
    >>> gen = SawtoothSignalGenerator(fs=250, freq=2, width=0.5)
    >>> gen.get_chunk(1.0).shape
    (250, 1)
    """

    def __init__(
        self,
        fs: float,
        freq: float,
        amplitude: float = 1.0,
        width: float = 1.0,
        offset: float = 0.0,
        noise_type: _NoiseType | None = None,
        noise_params: dict | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(fs, noise_type, noise_params, seed)
        if not 0.0 <= width <= 1.0:
            raise ValueError('Parameter width must lie in [0, 1]')
        self.freq = float(freq)
        self.amplitude = float(amplitude)
        self.width = float(width)
        self.offset = float(offset)

    def _synthesise(self, duration: float, n_channels: int) -> NDArray:
        t = self._time_vector(duration)
        wave = self.amplitude * sp_signal.sawtooth(
            2 * np.pi * self.freq * t, width=self.width) + self.offset
        return np.tile(wave[:, None], (1, n_channels))


class RampSignalGenerator(SignalGenerator):
    """Linear ramp generator (``y = slope * t + intercept``).

    Useful for testing detrending and baseline-removal code.

    Parameters
    ----------
    fs
        Sampling frequency in Hz.
    slope
        Slope of the ramp in units per second.
    intercept
        Value of the ramp at ``t = 0``.
    noise_type, noise_params, seed
        See :class:`SignalGenerator`.

    Examples
    --------
    >>> gen = RampSignalGenerator(fs=100, slope=2.0, intercept=1.0)
    >>> chunk = gen.get_chunk(1.0)
    >>> chunk.shape
    (100, 1)
    >>> float(round(chunk[0, 0], 6)), float(round(chunk[50, 0], 6))
    (1.0, 2.0)
    """

    def __init__(
        self,
        fs: float,
        slope: float = 1.0,
        intercept: float = 0.0,
        noise_type: _NoiseType | None = None,
        noise_params: dict | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(fs, noise_type, noise_params, seed)
        self.slope = float(slope)
        self.intercept = float(intercept)

    def _synthesise(self, duration: float, n_channels: int) -> NDArray:
        t = self._time_vector(duration)
        wave = self.slope * t + self.intercept
        return np.tile(wave[:, None], (1, n_channels))


class ImpulseSignalGenerator(SignalGenerator):
    """Periodic impulse-train (Dirac comb) generator.

    Places impulses of a given amplitude at a regular rate, with all other
    samples set to ``0``. Optional timing jitter makes the train aperiodic.

    Parameters
    ----------
    fs
        Sampling frequency in Hz.
    rate
        Number of impulses per second (Hz). Must be greater than 0.
    amplitude
        Amplitude of each impulse.
    jitter
        Standard deviation, in seconds, of zero-mean Gaussian jitter
        applied to each impulse time. ``0`` produces a strictly periodic
        train.
    noise_type, noise_params, seed
        See :class:`SignalGenerator`.

    Examples
    --------
    >>> gen = ImpulseSignalGenerator(fs=100, rate=10, amplitude=1.0)
    >>> chunk = gen.get_chunk(1.0)
    >>> int((chunk != 0).sum())
    10
    """

    def __init__(
        self,
        fs: float,
        rate: float,
        amplitude: float = 1.0,
        jitter: float = 0.0,
        noise_type: _NoiseType | None = None,
        noise_params: dict | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(fs, noise_type, noise_params, seed)
        if rate <= 0:
            raise ValueError('Parameter rate must be greater than 0')
        if jitter < 0:
            raise ValueError('Parameter jitter must be non-negative')
        self.rate = float(rate)
        self.amplitude = float(amplitude)
        self.jitter = float(jitter)

    def _synthesise(self, duration: float, n_channels: int) -> NDArray:
        n_samples = self._n_samples(duration)
        wave = np.zeros((n_samples, 1))
        times = np.arange(0.0, duration, 1.0 / self.rate)
        if self.jitter > 0:
            times = times + self.rng.normal(0.0, self.jitter, times.shape)
        idx = np.round(times * self.fs).astype(int)
        idx = idx[(idx >= 0) & (idx < n_samples)]
        wave[idx, 0] = self.amplitude
        return np.tile(wave, (1, n_channels))


class ChirpSignalGenerator(SignalGenerator):
    """Frequency-swept cosine (chirp) generator.

    Sweeps the instantaneous frequency from ``f0`` to ``f1`` across the
    duration of each generated chunk.

    Parameters
    ----------
    fs
        Sampling frequency in Hz.
    f0
        Instantaneous frequency at the start of the chunk in Hz.
    f1
        Instantaneous frequency at the end of the chunk in Hz.
    amplitude
        Peak amplitude of the chirp.
    method
        Frequency-sweep profile, forwarded to :func:`scipy.signal.chirp`:
        ``'linear'``, ``'quadratic'``, ``'logarithmic'`` or
        ``'hyperbolic'``.
    noise_type, noise_params, seed
        See :class:`SignalGenerator`.

    Examples
    --------
    >>> gen = ChirpSignalGenerator(fs=500, f0=1, f1=50)
    >>> gen.get_chunk(2.0, 3).shape
    (1000, 3)
    """

    def __init__(
        self,
        fs: float,
        f0: float,
        f1: float,
        amplitude: float = 1.0,
        method: Literal[
            'linear', 'quadratic', 'logarithmic', 'hyperbolic'] = 'linear',
        noise_type: _NoiseType | None = None,
        noise_params: dict | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(fs, noise_type, noise_params, seed)
        self.f0 = float(f0)
        self.f1 = float(f1)
        self.amplitude = float(amplitude)
        self.method = method

    def _synthesise(self, duration: float, n_channels: int) -> NDArray:
        t = self._time_vector(duration)
        t1 = t[-1] if t.size else duration
        wave = self.amplitude * sp_signal.chirp(
            t, f0=self.f0, f1=self.f1, t1=t1, method=self.method)
        return np.tile(wave[:, None], (1, n_channels))


class EEGSignalGenerator(SignalGenerator):
    """Biologically plausible EEG generator.

    Synthesises EEG-like activity as a ``1/f`` (pink) background plus a set
    of narrow-band oscillatory rhythms with slowly waxing-and-waning
    amplitude (mimicking, e.g., the spindling of the occipital alpha
    rhythm). Each channel is generated independently. Amplitudes are
    expressed in microvolts.

    Parameters
    ----------
    fs
        Sampling frequency in Hz.
    oscillations
        Iterable of ``(freq, amplitude)`` pairs, with ``freq`` in Hz and
        ``amplitude`` in microvolts, describing the rhythmic components.
        Defaults to a single ~10 Hz alpha rhythm of 15 µV.
    background_amplitude
        Standard deviation, in microvolts, of the ``1/f`` background.
    pink_exponent
        Power-law exponent of the background spectrum (1.0 = pink).
    modulation_freq
        Frequency, in Hz, of the slow amplitude envelope applied to each
        oscillation (waxing and waning). ``0`` disables the modulation.
    noise_type, noise_params, seed
        Optional extra measurement noise; see :class:`SignalGenerator`.

    Examples
    --------
    >>> gen = EEGSignalGenerator(fs=250, seed=0)
    >>> gen.get_chunk(2.0, 8).shape
    (500, 8)
    """

    def __init__(
        self,
        fs: float,
        oscillations: Sequence[tuple[float, float]] | None = None,
        background_amplitude: float = 20.0,
        pink_exponent: float = 1.0,
        modulation_freq: float = 0.2,
        noise_type: _NoiseType | None = None,
        noise_params: dict | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(fs, noise_type, noise_params, seed)
        self.oscillations = (
            list(oscillations) if oscillations is not None else [(10.0, 15.0)])
        self.background_amplitude = float(background_amplitude)
        self.pink_exponent = float(pink_exponent)
        self.modulation_freq = float(modulation_freq)

    def _synthesise(self, duration: float, n_channels: int) -> NDArray:
        n_samples = self._n_samples(duration)
        t = (np.arange(n_samples) / self.fs)[:, None]
        # 1/f background
        chunk = self.background_amplitude * _pink_noise(
            n_samples, n_channels, self.rng, self.pink_exponent)
        # Oscillatory rhythms with independent per-channel phase
        for freq, amp in self.oscillations:
            phase = self.rng.uniform(0, 2 * np.pi, (1, n_channels))
            osc = amp * np.sin(2 * np.pi * freq * t + phase)
            if self.modulation_freq > 0:
                env_phase = self.rng.uniform(0, 2 * np.pi, (1, n_channels))
                envelope = 0.5 * (1 + np.sin(
                    2 * np.pi * self.modulation_freq * t + env_phase))
                osc = osc * envelope
            chunk += osc
        return chunk


class ECGSignalGenerator(SignalGenerator):
    """Biologically plausible ECG generator (PQRST Gaussian model).

    Each cardiac cycle is synthesised as a sum of five Gaussian bumps
    approximating the P wave, the QRS complex (Q, R, S) and the T wave.
    Beats are placed according to the configured heart rate, with optional
    heart-rate variability. The waveform is shared across channels (extra
    per-channel measurement noise can be added via ``noise_type``).
    Amplitudes are expressed in millivolts.

    Parameters
    ----------
    fs
        Sampling frequency in Hz.
    heart_rate
        Mean heart rate in beats per minute.
    amplitude
        Global scaling factor applied to the PQRST template (the template
        has a unit-amplitude R peak, so ``amplitude`` is roughly the R-peak
        height in mV).
    hrv_std
        Standard deviation, in seconds, of zero-mean Gaussian jitter added
        to each RR interval (heart-rate variability). ``0`` produces a
        perfectly periodic rhythm.
    noise_type, noise_params, seed
        Optional measurement noise; see :class:`SignalGenerator`.

    Notes
    -----
    The template (time offset relative to the R peak, amplitude, Gaussian
    width) loosely follows the dynamical ECG model of McSharry et al.
    (2003) but is parameterised directly in time rather than on the unit
    circle.

    Examples
    --------
    >>> gen = ECGSignalGenerator(fs=250, heart_rate=60, seed=0)
    >>> gen.get_chunk(4.0).shape
    (1000, 1)
    """

    # (time offset [s] relative to R, amplitude [mV], Gaussian width [s])
    _PQRST_TEMPLATE = (
        (-0.20, 0.08, 0.025),   # P
        (-0.02, -0.10, 0.010),  # Q
        (0.00, 1.00, 0.012),    # R
        (0.02, -0.15, 0.012),   # S
        (0.32, 0.25, 0.040),    # T
    )

    def __init__(
        self,
        fs: float,
        heart_rate: float = 60.0,
        amplitude: float = 1.0,
        hrv_std: float = 0.0,
        noise_type: _NoiseType | None = None,
        noise_params: dict | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(fs, noise_type, noise_params, seed)
        if heart_rate <= 0:
            raise ValueError('Parameter heart_rate must be greater than 0')
        if hrv_std < 0:
            raise ValueError('Parameter hrv_std must be non-negative')
        self.heart_rate = float(heart_rate)
        self.amplitude = float(amplitude)
        self.hrv_std = float(hrv_std)

    def _beat_times(self, duration: float) -> NDArray:
        """R-peak times (s) covering ``duration`` with optional HRV."""
        rr = 60.0 / self.heart_rate
        times = []
        # Start half an RR before 0 so the first beat's P/Q are included
        current = -rr / 2.0
        while current < duration:
            times.append(current)
            interval = rr
            if self.hrv_std > 0:
                interval += self.rng.normal(0.0, self.hrv_std)
            current += max(interval, 1.0 / self.fs)
        return np.asarray(times)

    def _synthesise(self, duration: float, n_channels: int) -> NDArray:
        t = self._time_vector(duration)
        wave = np.zeros(t.shape[0])
        for r_time in self._beat_times(duration):
            for dt, amp, width in self._PQRST_TEMPLATE:
                center = r_time + dt
                wave += amp * np.exp(
                    -((t - center) ** 2) / (2.0 * width ** 2))
        wave *= self.amplitude
        return np.tile(wave[:, None], (1, n_channels))


class EOGSignalGenerator(SignalGenerator):
    """Biologically plausible EOG generator (blinks, saccades, drift).

    Synthesises electrooculogram-like activity combining three phenomena:

    - **Eye blinks** — brief positive deflections modelled as Gaussian
      bumps occurring as a Poisson process.
    - **Saccades** — fast gaze shifts modelled as smooth (sigmoid) steps
      that move the baseline and hold it until the next saccade,
      producing the characteristic EOG staircase.
    - **Slow drift** — low-frequency ``1/f`` baseline wander.

    Amplitudes are expressed in microvolts. Each channel is generated
    independently.

    Parameters
    ----------
    fs
        Sampling frequency in Hz.
    blink_rate
        Mean number of blinks per minute.
    blink_amplitude
        Peak amplitude of a blink in microvolts.
    blink_width
        Standard deviation of the blink Gaussian in seconds.
    saccade_rate
        Mean number of saccades per minute.
    saccade_amplitude
        Standard deviation, in microvolts, of the (zero-mean) gaze step of
        each saccade.
    drift_amplitude
        Standard deviation, in microvolts, of the slow baseline drift.
    noise_type, noise_params, seed
        Optional measurement noise; see :class:`SignalGenerator`.

    Examples
    --------
    >>> gen = EOGSignalGenerator(fs=250, seed=0)
    >>> gen.get_chunk(10.0, 2).shape
    (2500, 2)
    """

    def __init__(
        self,
        fs: float,
        blink_rate: float = 15.0,
        blink_amplitude: float = 200.0,
        blink_width: float = 0.05,
        saccade_rate: float = 30.0,
        saccade_amplitude: float = 50.0,
        drift_amplitude: float = 10.0,
        noise_type: _NoiseType | None = None,
        noise_params: dict | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(fs, noise_type, noise_params, seed)
        for name, value in (
                ('blink_rate', blink_rate), ('saccade_rate', saccade_rate)):
            if value < 0:
                raise ValueError(f'Parameter {name} must be non-negative')
        self.blink_rate = float(blink_rate)
        self.blink_amplitude = float(blink_amplitude)
        self.blink_width = float(blink_width)
        self.saccade_rate = float(saccade_rate)
        self.saccade_amplitude = float(saccade_amplitude)
        self.drift_amplitude = float(drift_amplitude)

    def _poisson_times(self, rate_per_min: float, duration: float) -> NDArray:
        """Random event times (s) from a Poisson process over ``duration``."""
        expected = rate_per_min / 60.0 * duration
        n_events = self.rng.poisson(expected)
        return np.sort(self.rng.uniform(0.0, duration, n_events))

    def _synthesise(self, duration: float, n_channels: int) -> NDArray:
        n_samples = self._n_samples(duration)
        t = np.arange(n_samples) / self.fs
        chunk = np.zeros((n_samples, n_channels))
        tau = 0.02  # saccade transition time constant (s)
        for ch in range(n_channels):
            channel = np.zeros(n_samples)
            # Slow drift
            if self.drift_amplitude > 0:
                channel += self.drift_amplitude * _pink_noise(
                    n_samples, 1, self.rng, exponent=2.0)[:, 0]
            # Blinks
            for bt in self._poisson_times(self.blink_rate, duration):
                channel += self.blink_amplitude * np.exp(
                    -((t - bt) ** 2) / (2.0 * self.blink_width ** 2))
            # Saccades (smooth steps that accumulate into a staircase)
            for st in self._poisson_times(self.saccade_rate, duration):
                step = self.rng.normal(0.0, self.saccade_amplitude)
                channel += step * 0.5 * (1.0 + np.tanh((t - st) / tau))
            chunk[:, ch] = channel
        return chunk
