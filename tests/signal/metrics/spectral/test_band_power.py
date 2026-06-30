"""Tests for :mod:`medusa.signal.metrics.spectral.band_power`.

Uses synthetic sinusoids and the package's own Welch wrapper to keep
the test self-contained.
"""

from __future__ import annotations

import numpy as np
import pytest

from medusa.signal.metrics.spectral.band_power import band_power
from medusa.signal.transforms import power_spectral_density


def _make_psd(freq_hz: float, fs: int = 256, duration: int = 4,
              n_channels: int = 2, seed: int = 0):
    rng = np.random.default_rng(seed)
    # Deterministic sinusoid + tiny noise (the generator's noise is not
    # seedable; we add our own to avoid global RNG side effects).
    t = np.linspace(0, duration - 1 / fs, fs * duration)
    sig = np.sin(2 * np.pi * freq_hz * t)[:, None] * np.ones((1, n_channels))
    sig = sig + 0.01 * rng.standard_normal(sig.shape)
    # Welch wrapper expects [n_epochs, n_samples, n_channels].
    sig = sig[np.newaxis, ...]
    f, psd = power_spectral_density(sig, fs=fs)
    return f, psd, fs


def test_band_power_concentrates_power_in_target_band():
    """A 10 Hz sinusoid must have far more power in [8, 12] than [20, 30]."""
    _, psd, fs = _make_psd(freq_hz=10.0)
    inside = band_power(psd, fs, [8, 12], power_type="absolute")
    outside = band_power(psd, fs, [20, 30], power_type="absolute")
    assert np.all(inside > 100 * outside)
    assert inside.shape == (1, 2)


def test_band_power_relative_is_in_unit_interval():
    _, psd, fs = _make_psd(freq_hz=10.0)
    rel = band_power(
        psd, fs, [8, 12], power_type="relative", norm_range=[1, 70]
    )
    assert rel.shape == (1, 2)
    assert np.all(rel >= 0.0)
    assert np.all(rel <= 1.0)


def test_band_power_invalid_power_type_raises():
    _, psd, fs = _make_psd(freq_hz=10.0)
    with pytest.raises(ValueError):
        band_power(psd, fs, [8, 12], power_type="not-a-type")


def test_band_power_accepts_2d_psd_streaming():
    """A 2-D (n_freqs, n_channels) PSD is promoted and de-segmented."""
    psd_2d = np.random.default_rng(0).random((129, 2))
    powers = band_power(psd_2d, 256, [8, 12])
    assert powers.shape == (2,)

