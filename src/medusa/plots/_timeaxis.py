"""Internal helpers shared by the time-axis plots (timeline + timeheatmap).

The common input contract: plain arrays validated against the medusa-kernel
signal contract (:func:`medusa.core.utils.check_data_dims`), an **optional**
sampling rate, and an optional event overlay. Three helpers, each doing real
shared work — there is deliberately no trivial ``_ensure_ax``/``_xspan`` glue
(axes are caller-supplied; axis spans are inlined).
"""

import medusa_style
import numpy as np
from numpy.typing import ArrayLike, NDArray

from medusa.core.utils import check_data_dims

# Event-overlay chrome (single source for the LOOK of an event marker; the COLOR
# comes from medusa-style: the categorical cycle when colored by an event hue,
# else the palette's annotation `highlight` role). Onset markers are
# intentionally thinner than data lines so they read as annotations, not signals.
_EVENT_LINE_STYLE = "--"
_EVENT_LINE_WIDTH = 1.0
_EVENT_LINE_ALPHA = 0.9
_EVENT_SPAN_ALPHA = 0.15


def _prepare(data: ArrayLike, fs: float | None,
             times: ArrayLike | None
             ) -> tuple[NDArray, NDArray, str, list[float]]:
    """Normalize a time-domain array into ``(data2d, times, unit, boundaries)``.

    Accepts a 1-D/2-D ``(n_samples, n_channels)`` recording or a 3-D
    ``(n_segments, n_samples, n_channels)`` segmented signal (concatenated along
    time, the per-join times returned as ``boundaries``). The sampling rate is
    optional: with neither ``fs`` nor ``times`` the axis is the sample index
    (``unit == 'samples'``); otherwise it is seconds (``unit == 's'``).
    """
    data = np.asarray(data, dtype=float)
    if data.ndim == 3:
        data, _ = check_data_dims(data, rep_type="time_segments")
        n_seg, n_samp, n_ch = data.shape
        joins = [k * n_samp for k in range(1, n_seg)]
        data = data.reshape(n_seg * n_samp, n_ch)
    else:
        data, _ = check_data_dims(data, rep_type="time")
        joins = []
    n = data.shape[0]
    if times is not None:
        times = np.asarray(times, dtype=float).ravel()
        if times.shape[0] != n:
            raise ValueError(f"times has length {times.shape[0]}; expected {n}.")
        unit = "s"
    elif fs is not None:
        times = np.arange(n) / float(fs)
        unit = "s"
    else:
        times = np.arange(n, dtype=float)
        unit = "samples"
    return data, times, unit, [float(times[j]) for j in joins]


def _channel_labels(cha_labels: ArrayLike | None, n_ch: int) -> list[str]:
    """Validate ``cha_labels`` against ``n_ch`` (auto ``Ch1..`` when None)."""
    if cha_labels is None:
        return [f"Ch{i + 1}" for i in range(n_ch)]
    cha_labels = list(cha_labels)
    if len(cha_labels) != n_ch:
        raise ValueError(
            f"cha_labels has length {len(cha_labels)}; expected {n_ch} "
            f"channels.")
    return [str(lab) for lab in cha_labels]


def _events_df(events):
    """Extract the events DataFrame (``.df`` or the object itself); ``None`` ok."""
    return getattr(events, "df", events)


def _check_events(events, hue: str | None = None) -> None:
    """Validate an events table for the overlay.

    Requires ``onset`` and ``duration`` columns, plus the ``hue`` column when a
    ``hue`` is given. A ``None``/empty table is valid (nothing is drawn).

    Raises
    ------
    ValueError
        If a required column is missing.
    """
    df = _events_df(events)
    if df is None or len(df) == 0:
        return
    cols = list(df.columns) if hasattr(df, "columns") else list(df)
    required = ["onset", "duration"]
    if hue is not None:
        required.append(hue)
    missing = [c for c in required if c not in cols]
    if missing:
        raise ValueError(
            f"events table is missing required column(s) {missing}; present "
            f"columns: {cols}. Provide 'onset' and 'duration'"
            f"{f' and event_hue {hue!r}' if hue is not None else ''}.")


def _overlay_events(ax, events, *, hue: str | None = None,
                    animated: bool = False, line_zorder: float = 1.5,
                    span_zorder: float = 0.4
                    ) -> tuple[list, list, list, dict]:
    """Overlay an event timeline: onset lines (duration 0), spans (duration > 0).

    ``events`` is anything exposing a ``.df`` with ``onset``/``duration`` columns
    (e.g. :class:`~medusa.core.data.events.Events`) or such a DataFrame; it is
    validated by :func:`_check_events`. ``hue`` names a column to color events by
    (categorical brand cycle). ``line_zorder``/``span_zorder`` place the overlay
    relative to the host plot — events sit *below* the stacked traces of a line
    plot but *above* a heatmap image.

    Does **not** draw a legend itself: the host plot owns a single combined legend
    and is handed one proxy handle/label per ``hue`` category.

    Returns
    -------
    artists : list
        Every drawn line/span artist.
    handles, labels : list
        One legend proxy handle + label per ``hue`` category (empty without ``hue``).
    color_map : dict
        Maps each ``hue`` value to the color it was drawn with (empty without
        ``hue``), so hosts can color a mirrored overview to match the legend.

    Raises
    ------
    ValueError
        If the events table lacks ``onset``/``duration`` (or the ``hue`` column
        when ``hue`` is given).
    """
    _check_events(events, hue)
    df = _events_df(events)
    if df is None or len(df) == 0:
        return [], [], [], {}
    onsets = np.asarray(df["onset"], dtype=float)
    durations = np.asarray(df["duration"], dtype=float)
    if hue is not None:
        hue_values = list(df[hue])
        cats = list(dict.fromkeys(hue_values))
        color_of = {c: medusa_style.categorical_color(i) for i, c in enumerate(cats)}
        colors = [color_of[v] for v in hue_values]
    else:
        hue_values = [None] * len(df)
        colors = [medusa_style.current_theme().highlight] * len(df)

    artists: list = []
    handles: list = []
    labels: list = []
    seen: set = set()
    for onset, dur, color, value in zip(onsets, durations, colors, hue_values):
        label = None
        if value is not None and value not in seen:
            label = str(value)
            seen.add(value)
        if not np.isfinite(dur) or dur <= 0:
            art = ax.axvline(onset, color=color, ls=_EVENT_LINE_STYLE,
                             lw=_EVENT_LINE_WIDTH, alpha=_EVENT_LINE_ALPHA,
                             zorder=line_zorder, label=label)
        else:
            art = ax.axvspan(onset, onset + dur, color=color,
                             alpha=_EVENT_SPAN_ALPHA,
                             zorder=span_zorder, label=label)
        art.set_animated(animated)
        artists.append(art)
        if label is not None:
            handles.append(art)
            labels.append(label)
    color_map = color_of if hue is not None else {}
    return artists, handles, labels, color_map
