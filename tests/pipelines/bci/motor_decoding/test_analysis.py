"""Tests for motor_decoding.analysis: spectrograms, ERD/ERS and statistics."""
import numpy as np
import pytest

from medusa.pipelines.bci.motor_decoding import analysis as A


# --------------------------------------------------------------------------- #
# motor_trials
# --------------------------------------------------------------------------- #
class TestMotorTrials:

    def test_shapes_and_labels(self, mi_recording, mi_channels):
        epochs, labels, fs = A.motor_trials(mi_recording, channels=mi_channels,
                                            window=(-1.5, 4.0))
        assert epochs.shape[0] == 16                          # 16 trials
        assert epochs.shape[2] == len(mi_channels)
        assert fs == 250.0
        np.testing.assert_array_equal(labels, [0, 1] * 8)

    def test_window_length(self, mi_recording, mi_channels):
        epochs, _, fs = A.motor_trials(mi_recording, channels=mi_channels, window=(-1.0, 3.0))
        assert epochs.shape[1] == int(round((3.0 - (-1.0)) * fs))

    def test_target_fs_resamples(self, mi_recording, mi_channels):
        epochs, _, fs = A.motor_trials(mi_recording, channels=mi_channels,
                                       window=(-1.0, 3.0), target_fs=100.0)
        assert fs == 100.0
        assert epochs.shape[1] == int(round(4.0 * 100.0))

    def test_channels_reordered(self, mi_recording):
        epochs, _, _ = A.motor_trials(mi_recording, channels=["C4", "C3"], window=(-1.0, 3.0))
        assert epochs.shape[2] == 2


# --------------------------------------------------------------------------- #
# spectrogram
# --------------------------------------------------------------------------- #
class TestSpectrogram:

    @staticmethod
    def _tone(freq, fs=250.0, n_trials=6, n_samples=1000, n_ch=2, seed=0):
        rng = np.random.default_rng(seed)
        t = np.arange(n_samples) / fs
        x = 0.3 * rng.standard_normal((n_trials, n_samples, n_ch))
        return x + np.sin(2 * np.pi * freq * t)[None, :, None]

    def test_shape_and_axes(self):
        power, times, freqs = A.spectrogram(self._tone(10.0), 250.0)
        assert power.ndim == 4 and power.shape[0] == 6 and power.shape[-1] == 2
        assert power.shape[1] == freqs.shape[0] and power.shape[2] == times.shape[0]

    def test_peak_at_tone_frequency(self):
        power, _, freqs = A.spectrogram(self._tone(12.0), 250.0, freq_range=(5, 30))
        peak = freqs[power.mean(axis=(0, 2, 3)).argmax()]
        assert abs(peak - 12.0) < 2.0

    def test_freq_range_crop(self):
        _, _, freqs = A.spectrogram(self._tone(10.0), 250.0, freq_range=(8, 20))
        assert freqs.min() >= 8 and freqs.max() <= 20

    def test_epoch_start_shifts_times(self):
        _, t0, _ = A.spectrogram(self._tone(10.0), 250.0, epoch_start=0.0)
        _, t1, _ = A.spectrogram(self._tone(10.0), 250.0, epoch_start=-2.0)
        np.testing.assert_allclose(t1, t0 - 2.0)

    def test_wavelet_method(self):
        power, times, freqs = A.spectrogram(self._tone(10.0), 250.0, method="wavelet",
                                            freq_range=(5, 30))
        assert power.ndim == 4 and power.shape[0] == 6

    def test_bad_method_raises(self):
        with pytest.raises(ValueError, match="method"):
            A.spectrogram(self._tone(10.0), 250.0, method="nope")

    def test_requires_3d_epochs(self):
        with pytest.raises(ValueError, match="3-D"):
            A.spectrogram(np.random.default_rng(0).standard_normal((5, 500)), 250.0)

    def test_empty_freq_range_raises(self):
        with pytest.raises(ValueError, match="selects no frequencies"):
            A.spectrogram(self._tone(10.0), 250.0, freq_range=(200, 300))
        with pytest.raises(ValueError, match="selects no frequencies"):
            A.spectrogram(self._tone(10.0), 250.0, freq_range=(30, 8))    # reversed


# --------------------------------------------------------------------------- #
# instantaneous_band_power
# --------------------------------------------------------------------------- #
class TestInstantaneousBandPower:

    def test_shape_and_times(self):
        epochs = np.random.default_rng(0).standard_normal((4, 500, 3))
        power, times = A.instantaneous_band_power(epochs, 250.0, band=(8, 12), epoch_start=-1.0)
        assert power.shape == epochs.shape
        assert times.shape == (500,)
        np.testing.assert_allclose(times[0], -1.0)

    def test_in_band_stronger_than_out_of_band(self):
        fs, n = 250.0, 1000
        t = np.arange(n) / fs
        tone = np.sin(2 * np.pi * 10.0 * t)[None, :, None] * np.ones((3, n, 1))
        in_band, _ = A.instantaneous_band_power(tone, fs, band=(8, 12))
        out_band, _ = A.instantaneous_band_power(tone, fs, band=(20, 30))
        assert in_band.mean() > 10 * out_band.mean()

    def test_requires_3d_epochs(self):
        """A 2-D (n_trials, n_samples) array is ambiguous and must be rejected, not mis-shaped."""
        with pytest.raises(ValueError, match="3-D"):
            A.instantaneous_band_power(np.random.default_rng(0).standard_normal((5, 500)),
                                       250.0, band=(8, 12))


# --------------------------------------------------------------------------- #
# erd_ers
# --------------------------------------------------------------------------- #
class TestErdErs:

    def test_exact_formula_trial_mode(self):
        # 1 trial, 1 freq, 2 times, 1 channel; baseline = first time bin (power 2)
        power = np.array([[[[2.0], [1.0]]]])
        times = np.array([0.0, 1.0])
        erds = A.erd_ers(power, times, baseline=(0.0, 0.0), ref_mode="trial")
        np.testing.assert_allclose(erds.ravel(), [0.0, -50.0])   # 100*(A-2)/2

    def test_classic_averages_baseline_over_trials(self):
        # two trials with different baselines; classic uses the grand-mean baseline
        power = np.array([[[[2.0], [1.0]]], [[[4.0], [2.0]]]])   # (2 trials,1,2,1)
        times = np.array([0.0, 1.0])
        erds = A.erd_ers(power, times, baseline=(0.0, 0.0), ref_mode="classic")
        # R = mean(2, 4) = 3; trial0: [100*(2-3)/3, 100*(1-3)/3], trial1: [100*(4-3)/3, 100*(2-3)/3]
        # averaged over trials: [(-33.3+33.3)/2, (-66.6-33.3)/2] = [0, -50]
        np.testing.assert_allclose(erds.ravel(), [0.0, -50.0], atol=1e-9)

    def test_average_false_keeps_trials(self):
        power = np.ones((5, 3, 10, 2))
        times = np.linspace(-1, 1, 10)
        erds = A.erd_ers(power, times, baseline=(-1, 0), average=False)
        assert erds.shape == (5, 3, 10, 2)

    def test_works_on_3d_bandpower(self):
        power = np.ones((5, 10, 2))                             # (trials, times, channels)
        times = np.linspace(-1, 1, 10)
        erds = A.erd_ers(power, times, baseline=(-1, 0))
        assert erds.shape == (10, 2)

    def test_baseline_outside_epoch_raises(self):
        power = np.ones((2, 10, 1))
        times = np.linspace(0, 1, 10)
        with pytest.raises(ValueError, match="outside the epoch"):
            A.erd_ers(power, times, baseline=(-1.0, 0.0))

    def test_bad_ref_mode_raises(self):
        power = np.ones((2, 10, 1))
        times = np.linspace(-1, 1, 10)
        with pytest.raises(ValueError, match="ref_mode"):
            A.erd_ers(power, times, baseline=(-1, 0), ref_mode="nope")

    def test_reversed_baseline_raises(self):
        power = np.ones((2, 10, 1))
        times = np.linspace(-2, 5, 10)
        with pytest.raises(ValueError, match="start must be <= end"):
            A.erd_ers(power, times, baseline=(0.0, -2.0))

    def test_zero_reference_power_raises(self):
        """A dead/flat channel gives a zero baseline; ERD/ERS is undefined there -> clear error."""
        power = np.ones((3, 10, 2))
        power[:, :, 1] = 0.0                                    # channel 1 is dead
        times = np.linspace(-1, 1, 10)
        with pytest.raises(ValueError, match="reference power is zero"):
            A.erd_ers(power, times, baseline=(-1, 0))

    def test_detects_erd_end_to_end(self, mi_recording, mi_channels):
        epochs, _, fs = A.motor_trials(mi_recording, channels=mi_channels, window=(-1.5, 4.0))
        power, times = A.instantaneous_band_power(epochs, fs, band=(8, 12), epoch_start=-1.5)
        erds = A.erd_ers(power, times, baseline=(-1.5, -0.5))
        # after onset the ERD channels drop below baseline (negative %)
        assert erds[times > 0.5].mean() < -10.0


# --------------------------------------------------------------------------- #
# erd_ers_significance
# --------------------------------------------------------------------------- #
class TestErdErsSignificance:

    def test_consistent_effect_is_significant(self):
        rng = np.random.default_rng(0)
        erds_trials = -50.0 + 5.0 * rng.standard_normal((20, 4))   # strong, consistent ERD
        mask, lo, hi = A.erd_ers_significance(erds_trials, n_boot=500, rng=0)
        assert mask.shape == (4,)
        assert mask.all()                                          # all bins significant
        assert np.all(hi < 0)                                      # CI below zero

    def test_pure_noise_mostly_not_significant(self):
        rng = np.random.default_rng(1)
        erds_trials = rng.standard_normal((30, 50))               # mean ~0
        mask, _, _ = A.erd_ers_significance(erds_trials, n_boot=500, rng=2)
        assert mask.mean() < 0.15                                 # ~alpha false positives

    def test_deterministic_with_seed(self):
        erds = np.random.default_rng(3).standard_normal((15, 6)) - 20
        m1, l1, h1 = A.erd_ers_significance(erds, n_boot=200, rng=7)
        m2, l2, h2 = A.erd_ers_significance(erds, n_boot=200, rng=7)
        np.testing.assert_array_equal(m1, m2)
        np.testing.assert_allclose(l1, l2)

    def test_preserves_map_shape(self):
        erds = np.random.default_rng(0).standard_normal((10, 8, 20, 3)) - 30
        mask, lo, hi = A.erd_ers_significance(erds, n_boot=100, rng=0)
        assert mask.shape == (8, 20, 3) == lo.shape == hi.shape

    def test_too_few_trials_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            A.erd_ers_significance(np.ones((1, 4)))

    def test_bad_statistic_raises(self):
        with pytest.raises(ValueError, match="statistic"):
            A.erd_ers_significance(np.ones((5, 4)), statistic="mode")

    def test_warns_when_n_boot_too_few_for_alpha(self):
        erds = np.random.default_rng(0).standard_normal((10, 3)) - 20
        with pytest.warns(UserWarning, match="too few to resolve alpha"):
            A.erd_ers_significance(erds, alpha=0.01, n_boot=50, rng=0)


# --------------------------------------------------------------------------- #
# class_discriminability
# --------------------------------------------------------------------------- #
class TestClassDiscriminability:

    def test_separable_classes_high_r2(self):
        rng = np.random.default_rng(0)
        feat = rng.standard_normal((20, 3))
        labels = np.array([0, 1] * 10, dtype=float)
        feat[labels == 1] += 5.0                                  # strongly separated
        r2 = A.class_discriminability(feat, labels)
        assert r2.shape == (3,)
        assert np.all(np.abs(r2) > 0.7)

    def test_identical_classes_near_zero(self):
        feat = np.random.default_rng(0).standard_normal((20, 3))
        labels = np.array([0, 1] * 10, dtype=float)              # no class effect
        r2 = A.class_discriminability(feat, labels)
        assert np.all(np.abs(r2) < 0.3)

    def test_sign_follows_class_difference(self):
        feat = np.zeros((10, 1))
        labels = np.array([0, 1] * 5, dtype=float)
        feat[labels == 0] = 1.0                                   # class 0 larger
        r2 = A.class_discriminability(feat, labels, signed=True)
        assert r2[0] > 0                                          # sign of mean(c0)-mean(c1)

    def test_nan_labels_ignored(self):
        feat = np.random.default_rng(0).standard_normal((12, 2))
        labels = np.array([0, 1, np.nan, 0, 1, np.nan, 0, 1, 0, 1, 0, 1])
        r2 = A.class_discriminability(feat, labels)               # only 0/1 used
        assert r2.shape == (2,)

    def test_more_than_two_classes_needs_explicit_pair(self):
        feat = np.random.default_rng(0).standard_normal((9, 2))
        labels = np.array([0, 1, 2] * 3, dtype=float)
        with pytest.raises(ValueError, match="exactly two"):
            A.class_discriminability(feat, labels)
        r2 = A.class_discriminability(feat, labels, classes=(0, 2))   # explicit pair works
        assert r2.shape == (2,)

    def test_end_to_end_lateralization(self, mi_recording, mi_channels):
        """Class discriminability of mu power is lateralized (C3 vs C4 separate the classes)."""
        epochs, labels, fs = A.motor_trials(mi_recording, channels=mi_channels, window=(-1.5, 4.0))
        power, times = A.instantaneous_band_power(epochs, fs, band=(8, 12), epoch_start=-1.5)
        feat = power[:, times > 0.5].mean(axis=1)                 # mean post-onset mu power/channel
        r2 = A.class_discriminability(feat, labels)
        # C3 (idx 0) and C4 (idx 2) discriminate; CZ (idx 1) barely
        assert abs(r2[0]) > abs(r2[1]) and abs(r2[2]) > abs(r2[1])


# --------------------------------------------------------------------------- #
# trial_band_power / discriminability_spectrum / optimal_band
# --------------------------------------------------------------------------- #
def _two_class_tone(freq=10.0, fs=250.0, n=40, ns=1000, nc=4, seed=0):
    """Epochs where a `freq` tone is stronger on ch0 for class 0 and ch2 for class 1."""
    rng = np.random.default_rng(seed)
    t = np.arange(ns) / fs
    epochs = 0.3 * rng.standard_normal((n, ns, nc))
    labels = np.array([0, 1] * (n // 2), dtype=float)
    for i in range(n):
        epochs[i, :, 0 if labels[i] == 0 else 2] += 1.2 * np.sin(2 * np.pi * freq * t)
    return epochs, labels, fs


class TestTrialBandPower:

    def test_shape(self):
        epochs = np.random.default_rng(0).standard_normal((5, 800, 3))
        bp = A.trial_band_power(epochs, 250.0, (8, 12))
        assert bp.shape == (5, 3)

    def test_window_restriction(self):
        epochs, _, fs = _two_class_tone()
        full = A.trial_band_power(epochs, fs, (8, 12))
        win = A.trial_band_power(epochs, fs, (8, 12), epoch_start=-2.0, window=(0.0, 1.0))
        assert full.shape == win.shape == (40, 4)

    def test_empty_window_raises(self):
        epochs = np.random.default_rng(0).standard_normal((5, 800, 3))
        with pytest.raises(ValueError, match="selects no samples"):
            A.trial_band_power(epochs, 250.0, (8, 12), window=(10.0, 20.0))


class TestTrialPsd:

    def test_shape_and_peak(self):
        epochs, _, fs = _two_class_tone(freq=12.0)
        psd, freqs = A.trial_psd(epochs, fs, freq_range=(5, 30))
        assert psd.shape == (40, freqs.shape[0], 4)
        # channel 0 carries the 12 Hz tone -> spectral peak near 12 Hz
        peak = freqs[psd[:, :, 0].mean(axis=0).argmax()]
        assert abs(peak - 12.0) < 1.5

    def test_window_and_freq_range(self):
        epochs, _, fs = _two_class_tone()
        psd, freqs = A.trial_psd(epochs, fs, epoch_start=-2.0, window=(0.0, 1.5),
                                 freq_range=(8, 20))
        assert freqs.min() >= 8 and freqs.max() <= 20
        assert psd.shape[0] == 40

    def test_requires_3d(self):
        with pytest.raises(ValueError, match="3-D"):
            A.trial_psd(np.zeros((5, 800)), 250.0)

    def test_backs_discriminability_spectrum(self):
        """discriminability_spectrum == class_discriminability of trial_psd (DRY check)."""
        epochs, labels, fs = _two_class_tone()
        psd, freqs = A.trial_psd(epochs, fs, freq_range=(5, 30))
        r2_direct = A.class_discriminability(psd, labels)
        r2_spec, freqs2 = A.discriminability_spectrum(epochs, labels, fs, freq_range=(5, 30))
        np.testing.assert_allclose(r2_spec, r2_direct)
        np.testing.assert_allclose(freqs2, freqs)


class TestDiscriminabilitySpectrum:

    def test_peaks_at_discriminative_frequency(self):
        epochs, labels, fs = _two_class_tone(freq=10.0)
        r2, freqs = A.discriminability_spectrum(epochs, labels, fs, freq_range=(5, 30))
        assert r2.shape == (freqs.shape[0], 4)
        peak = freqs[np.abs(r2).mean(axis=1).argmax()]
        assert abs(peak - 10.0) < 1.5

    def test_freq_range_crop(self):
        epochs, labels, fs = _two_class_tone()
        _, freqs = A.discriminability_spectrum(epochs, labels, fs, freq_range=(8, 20))
        assert freqs.min() >= 8 and freqs.max() <= 20

    def test_requires_3d(self):
        with pytest.raises(ValueError, match="3-D"):
            A.discriminability_spectrum(np.zeros((5, 800)), np.array([0, 1, 0, 1, 0]), 250.0)


class TestOptimalBand:

    def test_finds_band_around_peak(self):
        epochs, labels, fs = _two_class_tone(freq=10.0)
        r2, freqs = A.discriminability_spectrum(epochs, labels, fs, freq_range=(5, 30))
        lo, hi = A.optimal_band(r2, freqs, channels=[0, 2])
        assert lo <= 10.0 <= hi and (hi - lo) < 12.0

    def test_respects_min_width(self):
        freqs = np.arange(5, 30, 1.0)
        r2 = np.zeros((len(freqs), 2))
        r2[freqs == 10.0] = 1.0                                   # a single sharp spike at 10 Hz
        lo, hi = A.optimal_band(r2, freqs, min_width=4.0)
        assert (hi - lo) >= 4.0 and lo <= 10.0 <= hi

    def test_channels_subset(self):
        freqs = np.arange(5, 30, 1.0)
        r2 = np.zeros((len(freqs), 3))
        r2[freqs == 22.0, 2] = 1.0                               # discriminative only on channel 2
        lo, hi = A.optimal_band(r2, freqs, channels=[2])
        assert lo <= 22.0 <= hi
