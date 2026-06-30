"""Tests for :mod:`medusa.signal.segmentation`.

The module produces segments following the canonical ``time_segments``
representation ``(n_segments, n_samples, n_channels)`` from a continuous
``time`` signal ``(n_samples, n_channels)``. Sliding-window extraction
(:func:`segment_signal`) and event-anchored extraction
(:func:`segment_signal_around_events`) are covered, along with the
normalization / resampling helpers, the feasibility report and the
time-to-sample-index mapping (including the out-of-order timestamps path).
"""

from __future__ import annotations

import numpy as np
import pytest

from medusa.signal.segmentation import (
    EventSegmentFeasibility,
    check_event_segments_feasibility,
    normalize_segments,
    resample_segments,
    segment_signal,
    segment_signal_around_events,
    times_to_sample_indices,
)


# ---------------------------------------------------------------------------
# segment_signal — sliding window.
# ---------------------------------------------------------------------------


def test_segment_signal_overlapping_shape():
    signal = np.random.default_rng(0).standard_normal((1000, 8))
    segments = segment_signal(signal, segment_length=250, stride=125)
    # windows: 1000 - 250 + 1 = 751, taken at 0, 125, ..., 750 -> 7
    assert segments.shape == (7, 250, 8)
    assert np.isfinite(segments).all()


def test_segment_signal_default_stride_is_non_overlapping():
    signal = np.arange(1000 * 4).reshape(1000, 4).astype(float)
    segments = segment_signal(signal, segment_length=250)
    assert segments.shape == (4, 250, 4)
    # Non-overlapping: concatenating the segments reconstructs the signal.
    np.testing.assert_array_equal(
        segments.reshape(1000, 4), signal)


def test_segment_signal_content_matches_source():
    signal = np.arange(20 * 2).reshape(20, 2).astype(float)
    segments = segment_signal(signal, segment_length=5, stride=5)
    np.testing.assert_array_equal(segments[0], signal[0:5])
    np.testing.assert_array_equal(segments[1], signal[5:10])


def test_segment_signal_1d_input_promoted_to_single_channel():
    signal = np.random.default_rng(1).standard_normal(1000)
    segments = segment_signal(signal, segment_length=200, stride=200)
    assert segments.shape == (5, 200, 1)


def test_segment_signal_with_dc_norm_zero_mean():
    signal = np.random.default_rng(2).standard_normal((600, 3)) + 5.0
    segments = segment_signal(signal, segment_length=200, norm='dc')
    np.testing.assert_allclose(
        segments.mean(axis=1), 0.0, atol=1e-12)


def test_segment_signal_with_z_norm_unit_std():
    signal = np.random.default_rng(3).standard_normal((600, 3))
    segments = segment_signal(signal, segment_length=200, norm='z')
    np.testing.assert_allclose(segments.mean(axis=1), 0.0, atol=1e-12)
    np.testing.assert_allclose(segments.std(axis=1), 1.0, atol=1e-12)


@pytest.mark.parametrize("bad_length", [0, -5])
def test_segment_signal_rejects_non_positive_length(bad_length):
    signal = np.zeros((100, 2))
    with pytest.raises(ValueError, match="segment_length"):
        segment_signal(signal, segment_length=bad_length)


def test_segment_signal_rejects_non_positive_stride():
    signal = np.zeros((100, 2))
    with pytest.raises(ValueError, match="stride"):
        segment_signal(signal, segment_length=10, stride=0)


# ---------------------------------------------------------------------------
# segment_signal_around_events — event-anchored extraction.
# ---------------------------------------------------------------------------


@pytest.fixture()
def event_signal():
    fs = 250.0
    timestamps = np.arange(0, 10, 1 / fs)
    signal = np.random.default_rng(0).standard_normal((timestamps.size, 4))
    onsets = np.array([2.0, 4.0, 6.0])
    return timestamps, signal, onsets, fs


def test_segment_around_events_shape(event_signal):
    timestamps, signal, onsets, fs = event_signal
    segments = segment_signal_around_events(
        timestamps, signal, onsets, fs, segment_window=(0, 500))
    # 500 ms at 250 Hz -> 125 samples
    assert segments.shape == (3, 125, 4)
    assert np.isfinite(segments).all()


def test_segment_around_events_negative_window(event_signal):
    timestamps, signal, onsets, fs = event_signal
    segments = segment_signal_around_events(
        timestamps, signal, onsets, fs, segment_window=(-200, 200))
    # 400 ms at 250 Hz -> 100 samples
    assert segments.shape == (3, 100, 4)


def test_segment_around_events_content_matches_source(event_signal):
    timestamps, signal, onsets, fs = event_signal
    segments = segment_signal_around_events(
        timestamps, signal, onsets, fs, segment_window=(0, 400))
    onset_idx = times_to_sample_indices(timestamps, onsets)
    for k, idx in enumerate(onset_idx):
        np.testing.assert_array_equal(
            segments[k], signal[idx:idx + 100])


def test_segment_around_events_1d_signal(event_signal):
    timestamps, _, onsets, fs = event_signal
    signal = np.random.default_rng(5).standard_normal(timestamps.size)
    segments = segment_signal_around_events(
        timestamps, signal, onsets, fs, segment_window=(0, 400))
    assert segments.shape == (3, 100, 1)


def test_segment_around_events_baseline_dc_norm(event_signal):
    timestamps, signal, onsets, fs = event_signal
    segments = segment_signal_around_events(
        timestamps, signal, onsets, fs,
        segment_window=(0, 400), baseline_window=(-200, 0), norm='dc')
    assert segments.shape == (3, 100, 4)
    assert np.isfinite(segments).all()


def test_segment_around_events_norm_without_baseline_raises(event_signal):
    timestamps, signal, onsets, fs = event_signal
    with pytest.raises(ValueError, match="baseline"):
        segment_signal_around_events(
            timestamps, signal, onsets, fs,
            segment_window=(0, 400), norm='z')


def test_segment_around_events_baseline_without_norm_raises(event_signal):
    timestamps, signal, onsets, fs = event_signal
    with pytest.raises(ValueError, match="norm"):
        segment_signal_around_events(
            timestamps, signal, onsets, fs,
            segment_window=(0, 400), baseline_window=(-200, 0))


def test_segment_around_events_infeasible_raises(event_signal):
    timestamps, signal, _, fs = event_signal
    # First onset's window starts before the recording.
    onsets = np.array([0.1, 4.0])
    with pytest.raises(ValueError, match="cannot be extracted"):
        segment_signal_around_events(
            timestamps, signal, onsets, fs, segment_window=(-500, 500))


def test_segment_around_events_out_of_range_onset_raises(event_signal):
    timestamps, signal, _, fs = event_signal
    onsets = np.array([4.0, 100.0])  # 100 s is past the recording
    with pytest.raises(ValueError, match="cannot be extracted"):
        segment_signal_around_events(
            timestamps, signal, onsets, fs, segment_window=(0, 400))


# ---------------------------------------------------------------------------
# normalize_segments.
# ---------------------------------------------------------------------------


def test_normalize_segments_z():
    segments = np.random.default_rng(0).standard_normal((5, 250, 8)) * 3 + 2
    out = normalize_segments(segments, norm='z')
    assert out.shape == segments.shape
    np.testing.assert_allclose(out.mean(axis=1), 0.0, atol=1e-12)
    np.testing.assert_allclose(out.std(axis=1), 1.0, atol=1e-12)


def test_normalize_segments_dc():
    segments = np.random.default_rng(1).standard_normal((5, 250, 8)) + 10
    out = normalize_segments(segments, norm='dc')
    np.testing.assert_allclose(out.mean(axis=1), 0.0, atol=1e-12)


def test_normalize_segments_with_external_stats():
    segments = np.ones((2, 10, 3))
    norm_segments = np.full((2, 10, 3), 4.0)
    out = normalize_segments(segments, norm_segments=norm_segments, norm='dc')
    # mean of norm_segments is 4 -> 1 - 4 = -3
    np.testing.assert_allclose(out, -3.0)


def test_normalize_segments_invalid_norm_raises():
    segments = np.zeros((2, 10, 3))
    with pytest.raises(ValueError, match="norm"):
        normalize_segments(segments, norm='bogus')  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# resample_segments.
# ---------------------------------------------------------------------------


def test_resample_segments_downsample_shape():
    segments = np.random.default_rng(0).standard_normal((5, 500, 8))
    out = resample_segments(segments, window=(0, 1000), target_fs=128)
    assert out.shape == (5, 128, 8)


def test_resample_segments_upsample_shape():
    segments = np.random.default_rng(1).standard_normal((3, 100, 2))
    out = resample_segments(segments, window=(0, 500), target_fs=500)
    assert out.shape == (3, 250, 2)


# ---------------------------------------------------------------------------
# check_event_segments_feasibility / EventSegmentFeasibility.
# ---------------------------------------------------------------------------


def test_feasibility_all_valid():
    fs = 250.0
    timestamps = np.arange(0, 4, 1 / fs)
    onsets = np.array([1.0, 2.0, 3.0])
    report = check_event_segments_feasibility(
        timestamps, onsets, fs, window=(-100, 100))
    assert isinstance(report, EventSegmentFeasibility)
    assert report.all_valid
    assert bool(report) is True
    assert report.invalid_idx.size == 0
    np.testing.assert_array_equal(report.valid_idx, [0, 1, 2])


def test_feasibility_before_start_and_after_end():
    fs = 250.0
    timestamps = np.arange(0, 4, 1 / fs)
    onsets = np.array([0.1, 2.0, 3.99])
    report = check_event_segments_feasibility(
        timestamps, onsets, fs, window=(-500, 500))
    assert not report.all_valid
    np.testing.assert_array_equal(report.invalid_idx, [0, 2])
    assert report.before_start[0]
    assert report.after_end[2]
    assert report.valid[1]


def test_feasibility_out_of_range():
    fs = 250.0
    timestamps = np.arange(0, 4, 1 / fs)
    onsets = np.array([2.0, 10.0])
    report = check_event_segments_feasibility(
        timestamps, onsets, fs, window=(0, 100))
    assert report.out_of_range[1]
    assert not report.valid[1]
    # Masks are mutually exclusive: out_of_range is not also a boundary flag.
    assert not report.before_start[1]
    assert not report.after_end[1]


def test_feasibility_empty_onsets():
    fs = 250.0
    timestamps = np.arange(0, 4, 1 / fs)
    report = check_event_segments_feasibility(
        timestamps, np.array([]), fs, window=(0, 100))
    assert report.all_valid  # vacuously true
    assert report.valid.size == 0


# ---------------------------------------------------------------------------
# times_to_sample_indices.
# ---------------------------------------------------------------------------


def test_times_to_sample_indices_sorted():
    timestamps = np.arange(0, 1, 0.1)
    out = times_to_sample_indices(timestamps, np.array([0.22, 0.51]))
    np.testing.assert_array_equal(out, [2, 5])


def test_times_to_sample_indices_rounds_to_nearest():
    timestamps = np.array([0.0, 1.0, 2.0, 3.0])
    out = times_to_sample_indices(timestamps, np.array([0.4, 0.6, 2.4]))
    # Each query maps to its closest sample.
    np.testing.assert_array_equal(out, [0, 1, 2])


def test_times_to_sample_indices_exact_tie_picks_higher_index():
    # 0.5 is exactly between samples 0 and 1; ties resolve to the higher index.
    timestamps = np.array([0.0, 1.0])
    out = times_to_sample_indices(timestamps, np.array([0.5]))
    np.testing.assert_array_equal(out, [1])


def test_times_to_sample_indices_unsorted_returns_original_indices():
    timestamps = np.array([0.0, 0.3, 0.1, 0.2])
    out = times_to_sample_indices(timestamps, np.array([0.11]))
    # 0.11 is closest to value 0.1, which sits at original index 2
    np.testing.assert_array_equal(out, [2])


def test_times_to_sample_indices_empty_timestamps():
    out = times_to_sample_indices(np.array([]), np.array([1.0, 2.0]))
    assert out.shape == (2,)
    assert out.size == 2


def test_times_to_sample_indices_matches_bruteforce_when_unsorted():
    rng = np.random.default_rng(0)
    timestamps = rng.permutation(np.linspace(0, 10, 200))
    queries = rng.uniform(0, 10, size=20)
    out = times_to_sample_indices(timestamps, queries)
    expected = np.array(
        [int(np.argmin(np.abs(timestamps - q))) for q in queries])
    np.testing.assert_array_equal(out, expected)


