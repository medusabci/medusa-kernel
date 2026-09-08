"""Segmentation of continuous biosignals into segments.

- :func:`segment_signal`: split a signal with a sliding window.
- :func:`segment_signal_around_events`: cut one segment per event onset,
  with optional per-event baseline normalization.
- :func:`check_event_segments_feasibility`: report which onsets can be cut,
  before attempting it.
- :func:`normalize_segments`, :func:`resample_segments`: post-process an
  existing segment array.
- :func:`times_to_sample_indices`: locate event times in a signal.

The extraction functions follow the medusa-kernel signal-shape contract:
continuous signals are accepted as ``(n_samples, n_channels)`` (the ``'time'``
representation) and segments are returned as
``(n_segments, n_samples, n_channels)`` (the ``'time_segments'``
representation). Under-dimensioned input is promoted by
:func:`medusa.core.utils.check_data_dims`, which emits a ``UserWarning``;
:func:`normalize_segments` and :func:`resample_segments` squeeze the promoted
axes back out, so they preserve the caller's number of axes.

Elsewhere in medusa the same array is called *epochs*, or *trials* when it is
anchored to events (see :mod:`medusa.pipelines.bci` and the tutorials).
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
    signal :
        Shape ``(n_samples, n_channels)``. Continuous signal to segment. A
        1-D array of shape ``(n_samples,)`` is treated as single-channel.
    segment_length :
        Segment length in samples. Must be greater than 0 and no longer than
        the signal.
    stride :
        Separation between the start of consecutive segments in samples. If
        ``None``, ``stride`` is set to ``segment_length`` (non-overlapping
        segments).
    norm :
        Normalization applied to each segment. ``'z'`` for Z-score, ``'dc'``
        for DC (mean) subtraction. Statistics are computed over the samples
        axis of each segment, so ``'z'`` yields ``NaN`` for a channel that is
        constant within a segment. ``None`` disables normalization.

    Returns
    -------
    NDArray
        Shape ``(n_segments, segment_length, n_channels)``, with
        ``n_segments = (n_samples - segment_length) // stride + 1``. Trailing
        samples that do not fill a whole window are dropped, never padded.
        With ``norm=None`` this is a read-only view into ``signal`` (no data
        is copied); any other ``norm`` returns a new writable array.

    Raises
    ------
    ValueError
        If ``segment_length`` or ``stride`` is not greater than 0, or if
        ``segment_length`` is longer than the signal.

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

    Cuts one segment per onset with the temporal window ``segment_window``
    (in ms, relative to the onset), optionally normalized with the statistics
    of a per-event baseline window. Every onset must fit in the recording: a
    single infeasible onset aborts the whole call, so pre-filter the onsets
    with :func:`check_event_segments_feasibility` when that is not guaranteed.

    Parameters
    ----------
    timestamps :
        Shape ``(n_samples,)``. Timestamp of each biosignal sample, in
        increasing order.
    signal :
        Shape ``(n_samples, n_channels)``. Continuous signal. A 1-D array of
        shape ``(n_samples,)`` is treated as single-channel.
    onsets :
        Shape ``(n_events,)``. Timestamps of the events. Need not be sorted.
    fs :
        Sampling frequency in Hz. Only converts the ms windows into samples;
        each event is anchored to its nearest sample in ``timestamps``, so an
        ``fs`` that disagrees with them silently changes the segment length.
    segment_window :
        Temporal window in ms of the segment, relative to each onset (0 ms is
        the onset), end exclusive. For example, ``(0, 1000)`` takes the
        segment from 0 ms to 1000 ms after each onset.
    baseline_window :
        Temporal window in ms of the baseline, relative to each onset. For
        example, ``(-500, 100)`` takes the baseline from 500 ms before to 100
        ms after each onset. Used to compute the normalization statistics.
        Required when ``norm`` is not ``None``, and checked for feasibility
        separately from ``segment_window``.
    norm :
        Normalization applied to each segment using the baseline statistics.
        ``'z'`` for Z-score, ``'dc'`` for DC (mean) subtraction. ``None``
        disables normalization. A baseline with no variance makes ``'z'``
        non-finite; see :func:`normalize_segments`.

    Returns
    -------
    NDArray
        Shape ``(n_events, n_samples, n_channels)``. One segment per onset, in
        the order given. Each window edge is rounded to samples on its own, so
        ``n_samples`` can differ by one from rounding the window duration.

    Raises
    ------
    ValueError
        If the requested windows fall outside the available samples, if at
        least one onset lies outside the timestamp range, or if ``norm`` and
        ``baseline_window`` are not provided consistently. The message names
        the offending onsets.

    See Also
    --------
    check_event_segments_feasibility : Report which onsets can be extracted.

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
    # Validate every onset up front, so the error can name the offending ones
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
    segments :
        Shape ``(n_segments, n_samples, n_channels)``. Segments to normalize.
        A 2-D array is treated as one segment, a 1-D array as one
        single-channel segment.
    norm_segments :
        Shape ``(n_segments, n_baseline_samples, n_channels)``. Segments whose
        statistics are used, typically a baseline window. Its length along the
        samples axis may differ from ``segments``; only the segments and
        channels axes must match. If ``None``, the statistics come from
        ``segments`` itself.
    norm :
        ``'z'`` for Z-score normalization or ``'dc'`` for DC (mean)
        subtraction. Statistics are computed over the samples axis (population
        standard deviation). The ``'z'`` division is unguarded: a channel with
        no variance in the statistics window gives ``NaN`` or ``±inf`` and a
        numpy ``RuntimeWarning``, not an error.

    Returns
    -------
    NDArray
        Same shape as ``segments``. Normalized segments.

    Raises
    ------
    ValueError
        If ``norm`` is not ``'z'`` or ``'dc'``.

    Examples
    --------
    Correct each segment with the statistics of a shorter baseline window:

    >>> import numpy as np
    >>> from medusa.signal.segmentation import normalize_segments
    >>> segments = np.random.randn(5, 250, 8)
    >>> baselines = np.random.randn(5, 50, 8)
    >>> normalize_segments(segments, norm_segments=baselines, norm='dc').shape
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

    Parameters
    ----------
    segments :
        Shape ``(n_segments, n_samples, n_channels)``. Segments to resample.
        A 2-D array is treated as one segment, a 1-D array as one
        single-channel segment.
    window :
        Temporal window in ms covered by each segment. For example,
        ``(0, 1000)`` for segments spanning 0 ms to 1000 ms. The output length
        follows from this and ``target_fs`` alone; the length of ``segments``
        is never inspected, so a ``window`` that does not describe them
        time-scales the data instead of resampling it.
    target_fs :
        Target sampling frequency in Hz.

    Returns
    -------
    NDArray
        Shape ``(n_segments, target_n_samples, n_channels)``, where
        ``target_n_samples = floor(target_fs * window_length / 1000)``, and
        with the same number of axes as ``segments``. Resampled segments.

    Notes
    -----
    Resampling happens in the frequency domain
    (:func:`scipy.signal.resample`), which treats each segment as one period
    of a periodic signal. Downsampling truncates the spectrum, so content
    above the new Nyquist frequency is removed rather than aliased. The cost
    is ringing at the segment edges when the first and last samples do not
    match, as with an uncorrected drift or step; baseline-correct or detrend
    beforehand when that matters.

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

    Returned by :func:`check_event_segments_feasibility`. Every attribute is a
    boolean mask of shape ``(n_onsets,)`` aligned with the ``onsets`` that were
    checked, so ``onsets[report.valid]`` keeps the extractable events; apply
    the same mask to any per-event labels to keep them in step.

    ``valid`` is exactly the negation of the other three masks, which say *why*
    an onset was rejected::

        valid == ~(out_of_range | before_start | after_end)

    ``out_of_range`` never coincides with the other two, because an onset that
    is not in the recording has no window to place. ``before_start`` and
    ``after_end`` are not exclusive of each other: both are ``True`` when the
    window overhangs the recording at both ends.

    ``bool(report)`` is :attr:`all_valid`, so ``if not report:`` means "some
    onset cannot be extracted", not "the report is empty" — a report over zero
    onsets is ``True``.

    Attributes
    ----------
    valid
        The segment can be fully extracted.
    before_start
        The window reaches back before the first sample.
    after_end
        The window reaches past the last sample.
    out_of_range
        The onset itself lies outside the range of the timestamps.
    """

    valid: NDArray
    before_start: NDArray
    after_end: NDArray
    out_of_range: NDArray

    def __bool__(self) -> bool:
        """Alias of :attr:`all_valid`."""
        return self.all_valid

    @property
    def all_valid(self) -> bool:
        """``True`` if every onset is extractable (vacuously so if empty)."""
        return bool(np.all(self.valid))

    @property
    def valid_idx(self) -> NDArray:
        """Positions in ``onsets`` of the extractable events."""
        return np.flatnonzero(self.valid)

    @property
    def invalid_idx(self) -> NDArray:
        """Positions in ``onsets`` of the events that cannot be extracted."""
        return np.flatnonzero(~self.valid)


def check_event_segments_feasibility(
    timestamps: NDArray,
    onsets: NDArray,
    fs: float,
    window: tuple[float, float],
) -> EventSegmentFeasibility:
    """Report, per onset, whether an event-anchored segment can be extracted.

    An onset is infeasible when its window falls partly outside the recording,
    or when the onset itself lies outside the range of the timestamps. Each
    onset is checked on its own, so ``onsets`` need not be sorted.

    Use this before :func:`segment_signal_around_events`, which raises as soon
    as *one* onset does not fit. That is a real risk for events near the
    beginning or the end of a recording, for a wide window, or for a marker
    list that may contain strays. Segment ``onsets[report.valid]`` to keep the
    events that do fit, and read the other three masks to report why the rest
    were dropped. See :class:`EventSegmentFeasibility` for how the masks
    relate.

    Parameters
    ----------
    timestamps :
        Shape ``(n_samples,)``. Timestamp of each biosignal sample, in
        increasing order.
    onsets :
        Shape ``(n_events,)``. Timestamps of the events.
    fs :
        Sampling frequency in Hz. Only converts ``window`` into a number of
        samples.
    window :
        Temporal window in ms relative to each onset, end exclusive. For
        example, ``(0, 1000)`` takes the window from 0 ms to 1000 ms after each
        onset. ``window[1] > window[0]`` is assumed but not checked.

    Returns
    -------
    EventSegmentFeasibility
        Per-onset masks ``valid``, ``before_start``, ``after_end`` and
        ``out_of_range``.

    Notes
    -----
    Only ``timestamps`` is inspected, never the signal, so a ``valid`` onset is
    in bounds for :func:`segment_signal_around_events` as long as the signal
    has as many samples as ``timestamps``. A baseline window is a separate
    window: check it with a second call.

    Examples
    --------
    Drop the events whose window does not fit, keeping their labels in step:

    >>> import numpy as np
    >>> from medusa.signal.segmentation import (
    ...     check_event_segments_feasibility, segment_signal_around_events)
    >>> fs = 250.0
    >>> timestamps = np.arange(0, 4, 1 / fs)
    >>> signal = np.random.randn(timestamps.size, 4)
    >>> onsets = np.array([0.1, 2.0, 3.99])
    >>> labels = np.array(['a', 'b', 'c'])
    >>> report = check_event_segments_feasibility(
    ...     timestamps, onsets, fs, window=(-500, 500))
    >>> report.all_valid
    False
    >>> report.invalid_idx.tolist()   # positions in `onsets`, not samples
    [0, 2]
    >>> bool(report.before_start[0]), bool(report.after_end[2])
    (True, True)
    >>> segments = segment_signal_around_events(
    ...     timestamps, signal, onsets[report.valid], fs,
    ...     segment_window=(-500, 500))
    >>> segments.shape, labels[report.valid].tolist()
    ((1, 250, 4), ['b'])
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
    """Nearest index, assuming a non-decreasing and non-empty input.

    Exact ties resolve to the higher index.
    """
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
    timestamp is closest in time. A query exactly halfway between two samples
    maps to the later one, and a query outside the range of ``timestamps`` maps
    to the first or last sample instead of raising — use
    :func:`check_event_segments_feasibility` to detect that case.

    Parameters
    ----------
    timestamps :
        Shape ``(n_samples,)``. Timestamps of the signal. Need not be sorted:
        out-of-order input (e.g. reordered UDP packets) is detected and handled
        at the cost of a sort, so the returned indices always refer to
        ``timestamps`` as given.
    query_times :
        Shape ``(n_query,)``. Query times (e.g. event onsets) to locate.

    Returns
    -------
    NDArray
        Shape ``(n_query,)``. Index (into the original, possibly unsorted
        ``timestamps``) of the nearest timestamp to each query time. If
        ``timestamps`` is empty the values are uninitialized memory, not usable
        indices.

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
