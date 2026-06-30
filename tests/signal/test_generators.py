"""Tests for :mod:`medusa.signal.generators`.

Covers the improved :class:`SignalGenerator` base class (shape contract,
noise models, reproducibility, validation) and every concrete generator,
including the biologically plausible EEG/ECG/EOG models.
"""

from __future__ import annotations

import numpy as np
import pytest

from medusa.signal import generators as gens


# ---------------------------------------------------------------------------
# Base-class contract: shape, validation, reproducibility, noise.
# ---------------------------------------------------------------------------


def test_get_chunk_shape_and_default_n_channels():
    gen = gens.SinusoidalSignalGenerator(fs=250, freqs=[10], noise_type=None)
    assert gen.get_chunk(2.0).shape == (500, 1)
    assert gen.get_chunk(2.0, 8).shape == (500, 8)


def test_n_samples_rounds_duration():
    gen = gens.RampSignalGenerator(fs=256)
    # 0.1 s * 256 Hz = 25.6 -> rounds to 26
    assert gen.get_chunk(0.1).shape[0] == 26


@pytest.mark.parametrize("bad_fs", [0, -10])
def test_invalid_fs_raises(bad_fs):
    with pytest.raises(ValueError, match="fs"):
        gens.RampSignalGenerator(fs=bad_fs)


def test_invalid_noise_type_raises():
    with pytest.raises(ValueError, match="noise_type"):
        gens.RampSignalGenerator(
            fs=100, noise_type="brown")  # type: ignore[arg-type]


def test_invalid_duration_raises():
    gen = gens.RampSignalGenerator(fs=100)
    with pytest.raises(ValueError, match="duration"):
        gen.get_chunk(0)


def test_invalid_n_channels_raises():
    gen = gens.RampSignalGenerator(fs=100)
    with pytest.raises(ValueError, match="n_channels"):
        gen.get_chunk(1.0, 0)


def test_seed_makes_noise_reproducible():
    kwargs = dict(fs=100, freqs=[5], noise_type="white", seed=42)
    a = gens.SinusoidalSignalGenerator(**kwargs).get_chunk(1.0, 3)
    b = gens.SinusoidalSignalGenerator(**kwargs).get_chunk(1.0, 3)
    np.testing.assert_array_equal(a, b)


def test_white_noise_changes_signal():
    base = gens.SinusoidalSignalGenerator(
        fs=200, freqs=[5], noise_type=None).get_chunk(2.0)
    noisy = gens.SinusoidalSignalGenerator(
        fs=200, freqs=[5], noise_type="white",
        noise_params={"mean": 0.0, "sigma": 1.0}, seed=0).get_chunk(2.0)
    assert not np.allclose(base, noisy)


def test_pink_noise_runs_and_is_finite():
    gen = gens.RampSignalGenerator(
        fs=200, slope=0.0, noise_type="pink",
        noise_params={"sigma": 2.0, "exponent": 1.0}, seed=0)
    out = gen.get_chunk(2.0, 4)
    assert out.shape == (400, 4)
    assert np.isfinite(out).all()


# ---------------------------------------------------------------------------
# Sinusoidal.
# ---------------------------------------------------------------------------


def test_sinusoidal_exact_value_no_noise():
    gen = gens.SinusoidalSignalGenerator(
        fs=200, freqs=[1, 5], noise_type=None)
    chunk = gen.get_chunk(10, 5)
    assert chunk.shape == (2000, 5)
    assert chunk[250, 0] == pytest.approx(2.0)


def test_sinusoidal_amplitudes_and_phases():
    gen = gens.SinusoidalSignalGenerator(
        fs=100, freqs=[10], amplitudes=[3.0], phases=[np.pi / 2],
        noise_type=None)
    chunk = gen.get_chunk(1.0)
    # sin(2*pi*10*0 + pi/2) = 1 -> 3 * 1
    assert chunk[0, 0] == pytest.approx(3.0)


def test_sinusoidal_mismatched_lengths_raise():
    with pytest.raises(ValueError, match="same"):
        gens.SinusoidalSignalGenerator(
            fs=100, freqs=[1, 2], amplitudes=[1.0])


# ---------------------------------------------------------------------------
# Square / Sawtooth / Ramp / Impulse / Chirp.
# ---------------------------------------------------------------------------


def test_square_is_bipolar_levels():
    gen = gens.SquareSignalGenerator(fs=1000, freq=10, amplitude=2.0)
    chunk = gen.get_chunk(1.0)
    assert set(np.unique(np.round(chunk, 6))) <= {-2.0, 2.0}


def test_square_invalid_duty_raises():
    with pytest.raises(ValueError, match="duty"):
        gens.SquareSignalGenerator(fs=100, freq=5, duty=1.5)


def test_sawtooth_within_amplitude_range():
    gen = gens.SawtoothSignalGenerator(fs=1000, freq=5, amplitude=1.0)
    chunk = gen.get_chunk(1.0)
    assert chunk.min() >= -1.0001
    assert chunk.max() <= 1.0001


def test_sawtooth_invalid_width_raises():
    with pytest.raises(ValueError, match="width"):
        gens.SawtoothSignalGenerator(fs=100, freq=5, width=2.0)


def test_ramp_is_linear():
    gen = gens.RampSignalGenerator(fs=100, slope=2.0, intercept=1.0)
    chunk = gen.get_chunk(1.0)
    assert chunk[0, 0] == pytest.approx(1.0)
    assert chunk[50, 0] == pytest.approx(2.0)  # 1 + 2 * 0.5
    # constant first difference
    diffs = np.diff(chunk[:, 0])
    np.testing.assert_allclose(diffs, diffs[0])


def test_impulse_count_and_positions():
    gen = gens.ImpulseSignalGenerator(fs=100, rate=10, amplitude=1.0)
    chunk = gen.get_chunk(1.0)
    nz = np.flatnonzero(chunk[:, 0])
    assert nz.size == 10
    # impulses every fs/rate = 10 samples starting at 0
    np.testing.assert_array_equal(nz, np.arange(0, 100, 10))


def test_impulse_invalid_rate_raises():
    with pytest.raises(ValueError, match="rate"):
        gens.ImpulseSignalGenerator(fs=100, rate=0)


def test_chirp_shape_and_finite():
    gen = gens.ChirpSignalGenerator(fs=500, f0=1, f1=50)
    out = gen.get_chunk(2.0, 3)
    assert out.shape == (1000, 3)
    assert np.isfinite(out).all()


# ---------------------------------------------------------------------------
# Biosignal models: EEG / ECG / EOG.
# ---------------------------------------------------------------------------


def test_eeg_shape_and_finite():
    gen = gens.EEGSignalGenerator(fs=250, seed=0)
    out = gen.get_chunk(2.0, 8)
    assert out.shape == (500, 8)
    assert np.isfinite(out).all()


def test_eeg_channels_are_independent():
    gen = gens.EEGSignalGenerator(fs=250, seed=0)
    out = gen.get_chunk(2.0, 4)
    assert not np.allclose(out[:, 0], out[:, 1])


def test_eeg_has_alpha_peak_in_spectrum():
    fs = 250
    gen = gens.EEGSignalGenerator(
        fs=fs, oscillations=[(10.0, 40.0)], background_amplitude=5.0,
        modulation_freq=0.0, seed=0)
    sig = gen.get_chunk(8.0)[:, 0]
    freqs = np.fft.rfftfreq(sig.size, d=1 / fs)
    psd = np.abs(np.fft.rfft(sig)) ** 2
    peak_freq = freqs[np.argmax(psd)]
    assert abs(peak_freq - 10.0) < 1.0


def test_ecg_shape_and_r_peak_count():
    fs = 250
    gen = gens.ECGSignalGenerator(fs=fs, heart_rate=60, seed=0)
    sig = gen.get_chunk(5.0)[:, 0]
    assert sig.shape == (1250,)
    # ~60 bpm over 5 s -> ~5 R peaks; count strong positive prominences
    r_peaks = np.flatnonzero(
        (sig[1:-1] > sig[:-2]) & (sig[1:-1] > sig[2:]) & (sig[1:-1] > 0.5))
    assert 4 <= r_peaks.size <= 6


def test_ecg_higher_heart_rate_more_beats():
    fs = 250

    def count_r(sig):
        s = sig[:, 0]
        return np.flatnonzero(
            (s[1:-1] > s[:-2]) & (s[1:-1] > s[2:]) & (s[1:-1] > 0.5)).size

    slow = gens.ECGSignalGenerator(
        fs=fs, heart_rate=50, seed=0).get_chunk(10.0)
    fast = gens.ECGSignalGenerator(
        fs=fs, heart_rate=120, seed=0).get_chunk(10.0)
    assert count_r(fast) > count_r(slow)


def test_eog_shape_and_finite():
    gen = gens.EOGSignalGenerator(fs=250, seed=0)
    out = gen.get_chunk(10.0, 2)
    assert out.shape == (2500, 2)
    assert np.isfinite(out).all()


def test_eog_blinks_produce_large_positive_deflections():
    gen = gens.EOGSignalGenerator(
        fs=250, blink_rate=60, blink_amplitude=200.0, saccade_rate=0.0,
        drift_amplitude=0.0, seed=1)
    sig = gen.get_chunk(10.0)[:, 0]
    # With frequent 200 µV blinks the maximum must be clearly large.
    assert sig.max() > 100.0


def test_eog_negative_rate_raises():
    with pytest.raises(ValueError, match="blink_rate"):
        gens.EOGSignalGenerator(fs=250, blink_rate=-1)

