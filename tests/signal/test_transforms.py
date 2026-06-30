"""Tests for :mod:`medusa.signal.transforms`."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from medusa.signal import generators as signal_generators
from medusa.signal import transforms


def test_power_spectral_density_peaks_at_input_frequency():
    generator = signal_generators.SinusoidalSignalGenerator(
        fs=200, freqs=[1, 5], noise_type=None
    )
    chunk = generator.get_chunk(10, 5)
    f, psd = transforms.power_spectral_density(
        signal=chunk, fs=200, segment=0.8, overlap=0.5, window="boxcar"
    )
    assert psd.shape == (1, 801, 5)
    # Expected analytic value is 4 for a unit-amplitude sinusoid at 5 Hz
    np.testing.assert_allclose(psd[0, 8, 1], 4, rtol=1e-6)


def test_power_spectral_density_segment_overlap_match_legacy_pct():
    """Regression: segment=0.8, overlap=0.5 must equal segment_pct=80, overlap_pct=50.

    Pins the rename so a future implementation change cannot silently shift
    the numerical result of the canonical defaults.
    """
    rng = np.random.default_rng(0)
    sig = rng.standard_normal((1, 2000, 4))
    f_new, psd_new = transforms.power_spectral_density(
        sig, fs=250.0, segment=0.8, overlap=0.5, window="boxcar"
    )
    # Equivalent legacy call: nperseg = 2000*80/100 = 1600, noverlap = 1000
    from scipy.signal import welch as welch_sp
    f_ref, psd_ref = welch_sp(
        sig, fs=250.0, window="boxcar", nperseg=1600, noverlap=1000, axis=1
    )
    np.testing.assert_array_equal(f_new, f_ref)
    np.testing.assert_allclose(psd_new, psd_ref)


# ---------------------------------------------------------------------------
# fourier_spectrogram (K9.B 4-D TFR contract)
# ---------------------------------------------------------------------------


class TestFourierSpectrogram:
    def test_returns_4d_for_3d_input(self):
        rng = np.random.default_rng(0)
        sig = rng.standard_normal((2, 1024, 4))
        Sx, t, f = transforms.fourier_spectrogram(
            sig, fs=256.0, time_window=0.5, overlap=0.5, smooth=False
        )
        assert Sx.ndim == 4
        assert Sx.shape[0] == 2  # n_segments
        assert Sx.shape[3] == 4  # n_channels
        assert Sx.shape[1] == f.shape[0]  # n_frequencies
        assert Sx.shape[2] == t.shape[0]  # n_times

    def test_2d_input_promoted_with_warning(self):
        rng = np.random.default_rng(1)
        sig = rng.standard_normal((1024, 3))
        with pytest.warns(UserWarning, match="promoted"):
            Sx, _, _ = transforms.fourier_spectrogram(
                sig, fs=256.0, time_window=0.5, overlap=0.5, smooth=False
            )
        assert Sx.shape[0] == 1  # promoted segment axis
        assert Sx.shape[3] == 3  # n_channels preserved

    def test_per_channel_independence(self):
        """Channel c of the multi-channel call must equal the 1-channel call."""
        rng = np.random.default_rng(2)
        sig = rng.standard_normal((1, 1024, 3))
        Sx_multi, _, _ = transforms.fourier_spectrogram(
            sig, fs=256.0, time_window=0.5, overlap=0.5, smooth=False
        )
        for c in range(3):
            single = sig[:, :, c:c + 1]
            Sx_single, _, _ = transforms.fourier_spectrogram(
                single, fs=256.0, time_window=0.5, overlap=0.5, smooth=False
            )
            np.testing.assert_allclose(
                Sx_multi[:, :, :, c], Sx_single[:, :, :, 0]
            )

    def test_overlap_out_of_range_raises(self):
        rng = np.random.default_rng(3)
        sig = rng.standard_normal((1, 512, 1))
        with pytest.raises(ValueError, match=r"overlap"):
            transforms.fourier_spectrogram(sig, fs=256.0, overlap=1.0)
        with pytest.raises(ValueError, match=r"overlap"):
            transforms.fourier_spectrogram(sig, fs=256.0, overlap=-0.1)


# ---------------------------------------------------------------------------
# cwt_spectrogram (K9.B 4-D TFR contract)
# ---------------------------------------------------------------------------


class TestCwtSpectrogram:
    def test_returns_4d_for_3d_input(self):
        rng = np.random.default_rng(4)
        sig = rng.standard_normal((2, 256, 3))
        power, t, f, coif = transforms.cwt_spectrogram(
            sig, fs=128.0, smooth=False
        )
        assert power.ndim == 4
        assert power.shape[0] == 2  # n_segments
        assert power.shape[3] == 3  # n_channels
        assert power.shape[1] == f.shape[0]  # n_frequencies
        # n_times == n_samples for CWT (preserves time axis)
        assert power.shape[2] == 256
        assert t.shape == (256,)
        assert coif.shape == (256,)

    def test_frequencies_below_nyquist(self):
        rng = np.random.default_rng(5)
        sig = rng.standard_normal((1, 256, 1))
        _, _, f, _ = transforms.cwt_spectrogram(sig, fs=128.0, smooth=False)
        assert (f <= 0.5 * 128.0).all()

    def test_per_channel_independence(self):
        rng = np.random.default_rng(6)
        sig = rng.standard_normal((1, 256, 3))
        power_multi, _, _, _ = transforms.cwt_spectrogram(
            sig, fs=128.0, smooth=False
        )
        for c in range(3):
            single = sig[:, :, c:c + 1]
            power_single, _, _, _ = transforms.cwt_spectrogram(
                single, fs=128.0, smooth=False
            )
            np.testing.assert_allclose(
                power_multi[:, :, :, c], power_single[:, :, :, 0]
            )


# ---------------------------------------------------------------------------
# cross_cwt: deprecation warning is fired (K9 deferred function)
# ---------------------------------------------------------------------------


def test_cross_cwt_emits_deprecation_warning():
    rng = np.random.default_rng(7)
    s1 = rng.standard_normal(256)
    s2 = rng.standard_normal(256)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", DeprecationWarning)
        transforms.cross_cwt(s1, s2, fs=128.0, smooth=False)
    assert any(
        issubclass(w.category, DeprecationWarning) for w in captured
    )

