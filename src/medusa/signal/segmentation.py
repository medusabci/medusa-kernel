"""Segmentation of continuous biosignals into segments.

This module provides free functions to split a continuous signal into
segments — either with a sliding window (:func:`segment_signal`) or anchored
to event onsets (:func:`segment_signal_around_events`) — plus helpers to
normalize and resample the resulting segments.

All functions follow the medusa-kernel signal-shape contract. Continuous
signals are accepted as ``(n_samples, n_channels)`` (the ``'time'``
representation); the produced segments follow the canonical
``(n_segments, n_samples, n_channels)`` shape (the ``'time_segments'``
representation).
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.signal import resample

from medusa.core.utils import check_data_dims

__all__ = [
    "segment_signal",
    "segment_signal_around_events",
    "normalize_segments",
    "resample_segments",
    "EventSegmentFeasibility",
    "check_event_segments_feasibility",
    "times_to_sample_indices",
]


def segment_signal(
    signal: NDArray,
    segment_length: int,
    stride: int | None = None,
    norm: Literal['z', 'dc'] | None = None,
) -> NDArray:
    """Split a continuous signal into segments with a sliding window.

    Parameters
    ----------
    signal
        Shape ``(n_samples, n_channels)``. Continuous signal to segment. A
        1-D array of shape ``(n_samples,)`` is treated as single-channel.
    segment_length
        Segment length in samples. Must be greater than 0.
    stride
        Separation between the start of consecutive segments in samples. If
        ``None``, ``stride`` is set to ``segment_length`` (non-overlapping
        segments).
    norm
        Normalization applied to each segment. ``'z'`` for Z-score, ``'dc'``
        for DC (mean) subtraction. Statistics are computed over the samples
        axis of each segment. ``None`` disables normalization.

    Returns
    -------
    NDArray
        Shape ``(n_segments, segment_length, n_channels)``. Extracted
        segments.

    Examples
    --------
    >>> import numpy as np
    >>> from medusa.signal.segmentation import segment_signal
    >>> signal = np.random.randn(1000, 8)
    >>> segments = segment_signal(signal, segment_length=250, stride=125)
    >>> segments.shape
    (7, 250, 8)
    """
    # Validate parameters
    if segment_length <= 0:
        raise ValueError('Parameter segment_length must be greater than 0')
    if stride is None:
        stride = segment_length
    if stride <= 0:
        raise ValueError('Parameter stride must be None or greater than 0')
    # Promote to canonical (n_samples, n_channels)
    signal = np.asarray(signal)
    signal, _ = check_data_dims(signal, rep_type='time')
    # Assure ints
    n_channels = signal.shape[1]
    segment_length = int(segment_length)
    stride = int(stride)
    # Extract segments: (n_starts, 1, segment_length, n_channels) -> stride and
    # drop the singleton window axis explicitly to preserve segment/channel axes
    windows = np.lib.stride_tricks.sliding_window_view(
        signal, (segment_length, n_channels))
    segments = windows[::stride, 0]
    # Normalize
    if norm is not None:
        segments = normalize_segments(segments, norm=norm)
    return segments


def segment_signal_around_events(
    timestamps: NDArray,
    signal: NDArray,
    onsets: NDArray,
    fs: float,
    segment_window: tuple[float, float],
    baseline_window: tuple[float, float] | None = None,
    norm: Literal['z', 'dc'] | None = None,
) -> NDArray:
    """Extract signal segments anchored to event onsets.

    For each onset a segment is extracted using the temporal window
    ``segment_window`` (in ms, relative to the onset). Optionally, each
    segment can be normalized using the statistics of a per-event baseline
    window.

    Parameters
    ----------
    timestamps
        Shape ``(n_samples,)``. Timestamp of each biosignal sample.
    signal
        Shape ``(n_samples, n_channels)``. Continuous signal. A 1-D array of
        shape ``(n_samples,)`` is treated as single-channel.
    onsets
        Shape ``(n_events,)``. Timestamps of the events.
    fs
        Sampling frequency in Hz.
    segment_window
        Temporal window in ms of the segment, relative to each onset (0 ms is
        the onset). For example, ``(0, 1000)`` takes the segment from 0 ms to
        1000 ms after each onset.
    baseline_window
        Temporal window in ms of the baseline, relative to each onset. For
        example, ``(-500, 100)`` takes the baseline from 500 ms before to 100
        ms after each onset. This chunk is used to compute the normalization
        statistics. Required when ``norm`` is not ``None``.
    norm
        Normalization applied to each segment using the baseline statistics.
        ``'z'`` for Z-score, ``'dc'`` for DC (mean) subtraction. ``None``
        disables normalization.

    Returns
    -------
    NDArray
        Shape ``(n_events, n_samples, n_channels)``. Extracted segments.

    Raises
    ------
    ValueError
        If the requested windows fall outside the available samples, if at
        least one onset lies outside the timestamp range, or if ``norm`` and
        ``baseline_window`` are not provided consistently.

    Examples
    --------
    >>> import numpy as np
    >>> from medusa.signal.segmentation import segment_signal_around_events
    >>> fs = 250.0
    >>> timestamps = np.arange(0, 10, 1 / fs)
    >>> signal = np.random.randn(timestamps.size, 4)
    >>> onsets = np.array([2.0, 4.0, 6.0])
    >>> segments = segment_signal_around_events(
    ...     timestamps, signal, onsets, fs, segment_window=(0, 500))
    >>> segments.shape
    (3, 125, 4)
    """
    # Error prevention: validate every onset and fail with an informative
    # message that lists the offending onsets. To extract only the feasible
    # events instead of failing, pre-filter onsets with
    # check_event_segments_feasibility(...).valid (see its Examples).
    report = check_event_segments_feasibility(
        timestamps, onsets, fs, segment_window)
    if not report.all_valid:
        raise ValueError(_describe_infeasibility(report, 'segment'))
    if baseline_window is not None:
        baseline_report = check_event_segments_feasibility(
            timestamps, onsets, fs, baseline_window)
        if not baseline_report.all_valid:
            raise ValueError(
                _describe_infeasibility(baseline_report, 'baseline'))
        if norm is None:
            raise ValueError(
                'If parameter baseline_window is not None, please specify the '
                'normalization type with parameter norm')
    if norm is not None and baseline_window is None:
        raise ValueError(
            'If parameter norm is not None, please specify the baseline '
            'window with parameter baseline_window')

    # Promote to canonical (n_samples, n_channels)
    signal = np.asarray(signal)
    signal, _ = check_data_dims(signal, rep_type='time')
    n_channels = signal.shape[1]
    # Segment window in samples
    segment_window_s = np.round(
        np.asarray(segment_window) * fs / 1000).astype(int)
    l_segment = segment_window_s[1] - segment_window_s[0]
    # Extract one segment per onset
    onset_idx = times_to_sample_indices(timestamps, onsets)
    windows = np.lib.stride_tricks.sliding_window_view(
        signal, (l_segment, n_channels))
    segments = windows[onset_idx + segment_window_s[0], 0]
    # Baseline normalization
    if baseline_window is not None and norm is not None:
        baseline_window_s = np.round(
            np.asarray(baseline_window) * fs / 1000).astype(int)
        l_baseline = baseline_window_s[1] - baseline_window_s[0]
        baselines = np.lib.stride_tricks.sliding_window_view(
            signal, (l_baseline, n_channels))[
            onset_idx + baseline_window_s[0], 0]
        segments = normalize_segments(
            segments, norm_segments=baselines, norm=norm)
    return segments


def normalize_segments(
    segments: NDArray,
    norm_segments: NDArray | None = None,
    norm: Literal['z', 'dc'] = 'z',
) -> NDArray:
    """Normalize segments along the samples axis.

    Parameters
    ----------
    segments
        Shape ``(n_segments, n_samples, n_channels)``. Segments to normalize.
    norm_segments
        Shape ``(n_segments, n_samples, n_channels)``. Segments used to
        compute the normalization statistics. If ``None``, ``norm_segments``
        defaults to ``segments``.
    norm
        ``'z'`` for Z-score normalization or ``'dc'`` for DC (mean)
        subtraction. Statistics are computed over the samples axis.

    Returns
    -------
    NDArray
        Same shape as ``segments``. Normalized segments.

    Examples
    --------
    >>> import numpy as np
    >>> from medusa.signal.segmentation import normalize_segments
    >>> segments = np.random.randn(5, 250, 8)
    >>> normalized = normalize_segments(segments, norm='z')
    >>> normalized.shape
    (5, 250, 8)
    """
    if norm not in ('z', 'dc'):
        raise ValueError("Parameter norm must be 'z' or 'dc'")
    # Promote to canonical (n_segments, n_samples, n_channels)
    segments = np.asarray(segments)
    segments, inserted = check_data_dims(segments, rep_type='time_segments')
    if norm_segments is None:
        norm_segments = segments
    else:
        norm_segments = np.asarray(norm_segments)
        norm_segments, _ = check_data_dims(
            norm_segments, rep_type='time_segments')
    # Normalization (statistics over the samples axis)
    mean = np.mean(norm_segments, axis=1, keepdims=True)
    if norm == 'z':
        std = np.std(norm_segments, axis=1, keepdims=True)
        out = (segments - mean) / std
    else:  # 'dc'
        out = segments - mean
    return np.squeeze(out, axis=inserted) if inserted else out


def resample_segments(
    segments: NDArray,
    window: tuple[float, float],
    target_fs: float,
) -> NDArray:
    """Resample segments to a target sampling frequency.

    .. important::
        No anti-aliasing filter is applied.

    Parameters
    ----------
    segments
        Shape ``(n_segments, n_samples, n_channels)``. Segments to resample.
    window
        Temporal window in ms covered by each segment. For example,
        ``(0, 1000)`` for segments spanning 0 ms to 1000 ms. Used together
        with ``target_fs`` to compute the target number of samples.
    target_fs
        Target sampling frequency in Hz.

    Returns
    -------
    NDArray
        Shape ``(n_segments, target_n_samples, n_channels)``. Resampled
        segments.

    Examples
    --------
    >>> import numpy as np
    >>> from medusa.signal.segmentation import resample_segments
    >>> segments = np.random.randn(5, 500, 8)
    >>> resampled = resample_segments(segments, window=(0, 1000), target_fs=128)
    >>> resampled.shape
    (5, 128, 8)
    """
    # Promote to canonical (n_segments, n_samples, n_channels)
    segments = np.asarray(segments)
    segments, inserted = check_data_dims(segments, rep_type='time_segments')
    # Target number of samples from the window length (ms) and target_fs
    window_length = window[1] - window[0]  # ms
    target_n_samples = int(np.floor((target_fs * window_length) / 1000))
    out = resample(segments, target_n_samples, axis=1)
    return np.squeeze(out, axis=inserted) if inserted else out


@dataclass(frozen=True)
class EventSegmentFeasibility:
    """Per-onset feasibility report for event-anchored segmentation.

    Each attribute is a boolean mask of shape ``(n_onsets,)`` aligned with the
    ``onsets`` passed to :func:`check_event_segments_feasibility`. The masks
    are mutually exclusive: an onset flagged as ``out_of_range`` is never also
    flagged as ``before_start`` or ``after_end``.

    Attributes
    ----------
    valid
        ``True`` where the segment can be fully extracted. Use
        ``onsets[report.valid]`` to keep only the extractable events.
    before_start
        ``True`` where the window starts before the first sample.
    after_end
        ``True`` where the window extends past the last sample.
    out_of_range
        ``True`` where the onset itself lies outside the timestamp range.
    """

    valid: NDArray
    before_start: NDArray
    after_end: NDArray
    out_of_range: NDArray

    def __bool__(self) -> bool:
        return self.all_valid

    @property
    def all_valid(self) -> bool:
        """``True`` if every onset is extractable."""
        return bool(np.all(self.valid))

    @property
    def valid_idx(self) -> NDArray:
        """Indices of the extractable onsets."""
        return np.flatnonzero(self.valid)

    @property
    def invalid_idx(self) -> NDArray:
        """Indices of the onsets that cannot be extracted."""
        return np.flatnonzero(~self.valid)


def check_event_segments_feasibility(
    timestamps: NDArray,
    onsets: NDArray,
    fs: float,
    window: tuple[float, float],
) -> EventSegmentFeasibility:
    """Report, per onset, whether an event-anchored segment can be extracted.

    An onset cannot be extracted when its window falls partially outside the
    recording, or when the onset itself lies outside the timestamp range. This
    function validates *every* onset (so it is robust to unsorted onsets) using
    the same sample-index mapping as the extraction path
    (:func:`times_to_sample_indices`); therefore a ``valid`` onset is
    guaranteed to yield an in-bounds extraction in
    :func:`segment_signal_around_events`.

    Parameters
    ----------
    timestamps
        Shape ``(n_samples,)``. Sorted timestamp of each biosignal sample.
    onsets
        Shape ``(n_events,)``. Timestamps of the events.
    fs
        Sampling frequency in Hz.
    window
        Temporal window in ms relative to each onset. For example,
        ``(0, 1000)`` takes the window from 0 ms to 1000 ms after each onset.

    Returns
    -------
    EventSegmentFeasibility
        Per-onset boolean masks (``valid``, ``before_start``, ``after_end``,
        ``out_of_range``) plus convenience helpers (``all_valid``,
        ``valid_idx``, ``invalid_idx``).

    Examples
    --------
    Discard the onsets whose window does not fit and segment only the rest:

    >>> import numpy as np
    >>> from medusa.signal.segmentation import (
    ...     check_event_segments_feasibility, segment_signal_around_events)
    >>> fs = 250.0
    >>> timestamps = np.arange(0, 4, 1 / fs)
    >>> signal = np.random.randn(timestamps.size, 4)
    >>> onsets = np.array([0.1, 2.0, 3.99])
    >>> report = check_event_segments_feasibility(
    ...     timestamps, onsets, fs, window=(-500, 500))
    >>> report.all_valid
    False
    >>> report.invalid_idx.tolist()
    [0, 2]
    >>> segments = segment_signal_around_events(
    ...     timestamps, signal, onsets[report.valid], fs,
    ...     segment_window=(-500, 500))
    >>> segments.shape
    (1, 250, 4)
    """
    timestamps = np.asarray(timestamps)
    onsets = np.asarray(onsets)
    n_samples = timestamps.shape[0]
    n_onsets = onsets.shape[0]
    before_start = np.zeros(n_onsets, dtype=bool)
    after_end = np.zeros(n_onsets, dtype=bool)
    out_of_range = np.zeros(n_onsets, dtype=bool)
    if n_onsets:
        # Onsets outside the recorded time range
        out_of_range = (onsets < timestamps.min()) | (onsets > timestamps.max())
        in_range = ~out_of_range
        # Per-onset sample bounds, computed exactly as the extraction does so
        # that a valid result guarantees in-bounds fancy indexing (a negative
        # start would otherwise wrap around silently and extract wrong samples)
        onset_idx = times_to_sample_indices(timestamps, onsets)
        window_s = np.round(np.asarray(window) * fs / 1000).astype(int)
        start = onset_idx + window_s[0]
        stop = onset_idx + window_s[1]  # exclusive end
        before_start = in_range & (start < 0)
        after_end = in_range & (stop > n_samples)
    valid = ~(out_of_range | before_start | after_end)
    return EventSegmentFeasibility(
        valid=valid,
        before_start=before_start,
        after_end=after_end,
        out_of_range=out_of_range,
    )


def _describe_infeasibility(
    report: EventSegmentFeasibility,
    kind: str,
) -> str:
    """Build an informative error message from a feasibility report."""
    invalid = report.invalid_idx
    reasons = []
    if report.out_of_range.any():
        reasons.append(
            f"{int(report.out_of_range.sum())} onset(s) outside the "
            f"timestamp range")
    if report.before_start.any():
        reasons.append(
            f"{int(report.before_start.sum())} {kind}(s) start before the "
            f"first sample")
    if report.after_end.any():
        reasons.append(
            f"{int(report.after_end.sum())} {kind}(s) extend past the last "
            f"sample")
    return (
        f"{invalid.size} of {report.valid.size} {kind}s cannot be extracted "
        f"(onset indices {invalid.tolist()}): {'; '.join(reasons)}. "
        f"Pre-filter onsets with "
        f"check_event_segments_feasibility(...).valid to discard them.")


def _nearest_idx_in_sorted(
    sorted_timestamps: NDArray,
    query_times: NDArray,
) -> NDArray:
    """Nearest-index search assuming ``sorted_timestamps`` is non-decreasing."""
    array = sorted_timestamps
    # Get insert positions
    idxs = np.searchsorted(array, query_times, side="left")
    # Find indexes where the previous index is closer
    prev_idx_is_less = ((idxs == len(array)) | (
        np.fabs(query_times - array[np.maximum(idxs - 1, 0)]) < np.fabs(
            query_times - array[np.minimum(idxs, len(array) - 1)])))
    idxs[prev_idx_is_less] -= 1
    return idxs


def times_to_sample_indices(
    timestamps: NDArray,
    query_times: NDArray,
) -> NDArray:
    """Map query times to the index of their nearest signal sample.

    For each value in ``query_times`` return the index of the sample whose
    timestamp is closest in time. The fast path uses binary search
    (:func:`numpy.searchsorted`) and assumes ``timestamps`` is sorted. To stay
    correct when samples arrive out of order (e.g. reordered UDP packets), the
    input is checked and, if unsorted, the search is performed on a sorted view
    and the result is mapped back to the original positions. Complexity is
    ``O(n_query · log n_samples)`` when sorted and
    ``O(n_samples · log n_samples)`` otherwise — never the quadratic cost of a
    brute-force search.

    Parameters
    ----------
    timestamps
        Shape ``(n_samples,)``. Timestamps of the signal. Need not be
        sorted, although the sorted case is faster.
    query_times
        Shape ``(n_query,)``. Query times (e.g. event onsets) to locate.

    Returns
    -------
    NDArray
        Shape ``(n_query,)``. Index (into the original, possibly unsorted
        ``timestamps``) of the nearest timestamp to each query time.

    Examples
    --------
    >>> import numpy as np
    >>> from medusa.signal.segmentation import times_to_sample_indices
    >>> timestamps = np.arange(0, 1, 0.1)
    >>> times_to_sample_indices(timestamps, np.array([0.22, 0.51]))
    array([2, 5])

    Out-of-order timestamps return original-array indices:

    >>> timestamps = np.array([0.0, 0.3, 0.1, 0.2])
    >>> times_to_sample_indices(timestamps, np.array([0.11]))
    array([2])
    """
    timestamps = np.asarray(timestamps)
    query_times = np.asarray(query_times)
    if timestamps.size == 0:
        return np.empty(np.shape(query_times), dtype=np.intp)
    # Fast path for already-ordered timestamps (the common case)
    is_sorted = timestamps.size < 2 or bool(
        np.all(timestamps[:-1] <= timestamps[1:]))
    if is_sorted:
        return _nearest_idx_in_sorted(timestamps, query_times)
    # Unsorted timestamps (e.g. out-of-order UDP packets): search on a sorted
    # view and map the indices back to the original positions
    order = np.argsort(timestamps, kind='stable')
    nearest = _nearest_idx_in_sorted(timestamps[order], query_times)
    return order[nearest]
