"""Stacked multichannel signal traces over a time axis.

Channels are stacked a fixed ``offset`` apart in signal units (auto-scaled to the
data), so the y-axis stays interpretable. ``fs`` is optional, segmented input is
concatenated with a boundary line per join, and events are overlaid automatically.
Several signals can be overlaid on the same channel layout for comparison
(:meth:`TimeLinePlot.add_overlay`), and a visual amplitude :meth:`~TimeLinePlot.set_gain`
zooms the traces without changing the channel spacing. One-shot
:func:`plot_timeline` or stateful :class:`TimeLinePlot` (in-place ``set_data``,
blit-ready).
"""

import matplotlib as mpl
import medusa_style
import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.offsetbox import (
    AnchoredOffsetbox,
    AuxTransformBox,
    HPacker,
    TextArea,
)
from matplotlib.patches import Rectangle
from matplotlib.ticker import MultipleLocator
from numpy.typing import ArrayLike, NDArray

from medusa.plots._timeaxis import (
    _channel_labels,
    _check_events,
    _overlay_events,
    _prepare,
)
from medusa.plots.utils import _resolve_colors

__all__ = ["plot_timeline", "TimeLinePlot"]

#: Auto ``offset`` = this many median per-channel standard deviations. ~4 std
#: between baselines keeps typical activity inside its lane without crowding.
_AUTO_OFFSET_STD = 4.0

#: Segment-join boundary line chrome (dotted ink hairlines between segments).
_BOUNDARY_STYLE, _BOUNDARY_LW, _BOUNDARY_ALPHA = ":", 1.0, 0.5
#: Amplitude scale-bar stroke width.
_SCALE_BAR_LW = 1.6


def _nice_number(x: float) -> float:
    """Round a positive magnitude to a 1/2/5 x 10**k 'nice' number (scale bar)."""
    if not np.isfinite(x) or x <= 0:
        return 1.0
    exp = np.floor(np.log10(x))
    frac = x / 10 ** exp
    nice = (1.0 if frac < 1.5 else 2.0 if frac < 3.5
            else 5.0 if frac < 7.5 else 10.0)
    return float(nice * 10 ** exp)


class TimeLinePlot:
    """Stacked multichannel traces over a time axis, static-first and blit-ready.

    Each channel is drawn at its true amplitude (mean-centered on its baseline by
    default), and baselines are stacked a fixed ``offset`` apart (in signal
    units), so the y-axis reads in data units. The axes furniture (baselines,
    labels, scale bar) is built on the first :meth:`set_data`; later calls update
    only the line data in place and return the updated artists. Extra signals are
    overlaid on the same baselines with :meth:`add_overlay` (e.g. raw vs filtered),
    and :meth:`set_gain` scales the visual trace amplitude without reflowing the
    channel spacing. The drawn artists live in :attr:`artists`.

    Parameters
    ----------
    ax
        Axes to draw into (**required**).
    fs
        Sampling rate in Hz; sample-index x-axis when ``None``.
    cha_labels
        Channel labels (length ``n_channels``); auto ``Ch1..`` when ``None``.
    offset
        Distance between channel baselines in signal units; auto-scaled to the
        data (``_AUTO_OFFSET_STD`` x median std) when ``None``.
    center
        Mean-center each channel on its baseline; set ``False`` to keep its DC.
    color
        A single trace color (all channels) or one per channel; theme color when
        ``None``.
    line_width
        Trace line width.
    gain
        Visual amplitude multiplier applied to every trace (channel spacing is
        unchanged); the scale bar reports the true amplitude (``offset / gain``).
    scale_bar
        Draw the amplitude scale bar (reports the offset height).
    amplitude_unit
        Unit label for the scale bar.
    time_grid
        x-tick / vertical-grid spacing in x-axis units; no grid when ``None``.
    events, event_hue
        Event overlay (see :func:`~medusa.plots._timeaxis._overlay_events`);
        ``event_hue`` colors by an events column.
    legend, legend_loc, legend_fontsize
        Single combined legend (overlaid signals by ``label`` + event ``hue``
        categories); ``legend=False`` draws none. Drawn only when there is at
        least one labelled entry. ``legend_fontsize=None`` (default) follows the
        active style's ``legend.fontsize``.
    animated
        Mark data artists ``animated=True`` for host-driven blitting.

    Examples
    --------
    >>> tl = TimeLinePlot(ax, fs=256.0)       # doctest: +SKIP
    >>> _ = tl.set_data(window)               # doctest: +SKIP
    >>> _ = tl.add_overlay(filtered, color="crimson", label="filtered")  # doctest: +SKIP
    """

    def __init__(
            self, ax, *,
            fs: float | None = None,
            cha_labels: ArrayLike | None = None,
            offset: float | None = None,
            center: bool = True,
            color: str | list | None = None,
            line_width: float = 0.8,
            gain: float = 1.0,
            scale_bar: bool = True,
            amplitude_unit: str = "a.u.",
            time_grid: float | None = None,
            events=None,
            event_hue: str | None = None,
            legend: bool = True,
            legend_loc: str = "upper right",
            legend_fontsize: float | None = None,
            animated: bool = False):
        _check_events(events, event_hue)  # fail early on a malformed table
        self.ax = ax
        self.mappable = None  # line plots carry no color scale
        self.artists: dict = {}
        self._fs = fs
        self._cha_labels = None if cha_labels is None else list(cha_labels)
        self._offset = offset  # None -> auto from the data at build
        self._center = center
        self._color = color  # None -> the active theme's color, resolved at build
        self.line_width = line_width
        self._gain = float(gain)
        self.scale_bar = scale_bar
        self.amplitude_unit = amplitude_unit
        self._time_grid = time_grid
        self._events = events
        self._event_hue = event_hue
        self._legend = legend
        self._legend_loc = legend_loc
        self._legend_fontsize = legend_fontsize
        self._animated = animated
        self._n_channels = None
        self._baselines: list[float] = []
        self._offset_value = 1.0
        self._boundaries: list[float] | None = None
        # Each layer: {"lines", "data2d", "t", "color", "label"}; layer 0 = primary.
        self._layers: list[dict] = []
        self._event_handles: list = []
        self._event_labels: list = []
        self._event_color_map: dict = {}

    # -- public -----------------------------------------------------------
    def set_data(self, data, *, fs: float | None = None,
                 times: ArrayLike | None = None,
                 label: str | None = None) -> list[Artist]:
        """Render/update the primary signal's traces; return the updated artists.

        ``label`` names the primary signal in the combined legend (useful when
        overlays are added). Rebuilds the axes furniture when the channel count
        changes; otherwise updates the existing lines in place.
        """
        data2d, t, unit, boundaries = _prepare(
            data, self._fs if fs is None else fs, times)
        n_ch = data2d.shape[1]
        if self._n_channels != n_ch or not self._layers:
            self._build(data2d, unit)
        if label is not None:
            self._layers[0]["label"] = label
        updated = self._set_layer_data(self._layers[0], data2d, t)
        updated += self._update_boundaries(boundaries)
        self._fit_xlim()
        self._refresh_legend()
        return updated

    def add_overlay(self, data, *, color: str | None = None,
                    label: str | None = None, times: ArrayLike | None = None,
                    fs: float | None = None) -> list[Artist]:
        """Overlay another signal on the SAME channel baselines (for comparison).

        The overlay must share the primary's channel count. ``color`` defaults to
        the next brand categorical color; ``label`` names it in the legend; its own
        ``times``/``fs`` let it sit at a different time range (e.g. end-to-end).
        Returns the overlay's line artists.

        Raises
        ------
        RuntimeError
            If called before :meth:`set_data`.
        ValueError
            If the overlay's channel count differs from the primary's.
        """
        if self._n_channels is None:
            raise RuntimeError("call set_data() before add_overlay().")
        data2d, t, _unit, _boundaries = _prepare(
            data, self._fs if fs is None else fs, times)
        if data2d.shape[1] != self._n_channels:
            raise ValueError(
                f"overlay has {data2d.shape[1]} channels; expected "
                f"{self._n_channels} (the primary signal's channel count).")
        if color is None:
            color = medusa_style.categorical_color(len(self._layers))
        colors = _resolve_colors(color, self._n_channels)
        lines = self._make_lines(colors)
        layer = {"lines": lines, "data2d": None, "t": None, "color": colors,
                 "label": label}
        self._layers.append(layer)
        self.artists.setdefault("overlays", []).append(lines)
        updated = self._set_layer_data(layer, data2d, t)
        self._fit_xlim()
        self._refresh_legend()
        return updated

    def set_gain(self, gain: float) -> list[Artist]:
        """Set the visual amplitude gain on every layer; return updated artists.

        Channel spacing is unchanged; the scale bar is relabeled to the true
        amplitude (``offset / gain``).
        """
        self._gain = float(gain)
        updated: list[Artist] = []
        for layer in self._layers:
            if layer["data2d"] is not None:
                updated += self._set_layer_data(layer, layer["data2d"],
                                                layer["t"])
        if self.scale_bar and self._n_channels is not None:
            self._draw_scale_bar()
        return updated

    def set_events(self, events, *, hue: str | None = None) -> None:
        """Replace the overlaid events (and optionally the ``hue`` column)."""
        _check_events(events, hue if hue is not None else self._event_hue)
        self._events = events
        if hue is not None:
            self._event_hue = hue
        self._overlay_current_events()
        self._refresh_legend()

    def show_legend(self, on: bool) -> None:
        """Toggle the combined legend."""
        self._legend = bool(on)
        self._refresh_legend()

    def event_colors(self) -> dict:
        """Map each event ``hue`` value to the color it was drawn with."""
        return dict(self._event_color_map)

    @property
    def n_channels(self) -> int | None:
        """Number of stacked channels (``None`` before the first ``set_data``)."""
        return self._n_channels

    @property
    def offset_value(self) -> float:
        """Resolved distance between channel baselines, in signal units."""
        return self._offset_value

    # -- internals --------------------------------------------------------
    def _build(self, data2d: NDArray, unit: str) -> None:
        self._clear_all()
        n_ch = data2d.shape[1]
        self._n_channels = n_ch
        labels = _channel_labels(self._cha_labels, n_ch)
        self._offset_value = self._compute_offset(data2d)
        self._baselines = [(n_ch - 1 - i) * self._offset_value
                           for i in range(n_ch)]

        colors = _resolve_colors(self._color, n_ch)
        if colors is None:
            # Signals are DATA: use a data color (the first categorical), never
            # the foreground/text color. All channels share it (the fixed stacking
            # offset separates them); override via the `color` argument.
            colors = [medusa_style.categorical_color(0)] * n_ch
        lines = self._make_lines(colors)
        self.artists["lines"] = lines
        self._layers = [{"lines": lines, "data2d": None, "t": None,
                         "color": colors, "label": None}]
        self._boundaries = None

        half = 0.6 * self._offset_value
        self.ax.set_yticks(self._baselines)
        self.ax.set_yticklabels(labels)
        self.ax.set_ylim(-half, (n_ch - 1) * self._offset_value + half)
        self.ax.set_xlabel("Time (s)" if unit == "s" else "Samples")
        self.ax.margins(x=0)
        for side in ("top", "right"):
            self.ax.spines[side].set_visible(False)
        self.ax.grid(False)  # stacked baselines stand in for a y-grid
        if self._time_grid is not None:
            self.ax.xaxis.set_major_locator(MultipleLocator(self._time_grid))
            self.ax.set_axisbelow(True)
            self.ax.grid(True, axis="x")
        if self.scale_bar:
            self._draw_scale_bar()
        self._overlay_current_events()

    def _make_lines(self, colors: list) -> list[Artist]:
        """Create one (empty) Line2D per channel with the given per-channel colors."""
        lines = []
        for i in range(self._n_channels):
            (ln,) = self.ax.plot([], [], color=colors[i], lw=self.line_width,
                                 zorder=2)
            ln.set_animated(self._animated)
            lines.append(ln)
        return lines

    def _clear_all(self) -> None:
        """Remove every tracked artist (flattening nested overlay line lists)."""
        for value in self.artists.values():
            items = value if isinstance(value, list) else [value]
            for item in items:
                for art in (item if isinstance(item, list) else [item]):
                    try:
                        art.remove()
                    except (NotImplementedError, ValueError):
                        pass
        self.artists = {}
        self._layers = []
        self._event_handles, self._event_labels = [], []

    def _compute_offset(self, data: NDArray) -> float:
        """Resolve the channel offset (explicit, or auto from the median std)."""
        if self._offset is not None:
            return float(self._offset)
        stds = [np.nanstd(data[:, i]) for i in range(data.shape[1])]
        scale = float(np.nanmedian(stds)) if stds else float(np.nanstd(data))
        if not np.isfinite(scale) or scale == 0:
            scale = 1.0
        return _AUTO_OFFSET_STD * scale

    def _set_layer_data(self, layer: dict, data2d: NDArray,
                        t: NDArray) -> list[Artist]:
        """Set a layer's per-channel y = baseline + gain * (centered) signal."""
        layer["data2d"] = data2d
        layer["t"] = t
        updated: list[Artist] = []
        for i, ln in enumerate(layer["lines"]):
            col = data2d[:, i]
            if self._center:
                col = col - np.nanmean(col)
            ln.set_data(t, self._baselines[i] + self._gain * col)
            updated.append(ln)
        return updated

    def _fit_xlim(self) -> None:
        """Span the x-axis across every layer's time range."""
        spans = [layer["t"] for layer in self._layers if layer["t"] is not None
                 and len(layer["t"])]
        if spans:
            lo = min(float(t[0]) for t in spans)
            hi = max(float(t[-1]) for t in spans)
            self.ax.set_xlim(lo, hi)

    def _update_boundaries(self, boundaries: list[float]) -> list[Artist]:
        if boundaries == self._boundaries:
            return []
        for ln in self.artists.pop("boundaries", []):
            ln.remove()
        ink = medusa_style.current_theme().plot_fg
        lines = [self.ax.axvline(x, color=ink, ls=_BOUNDARY_STYLE,
                                 lw=_BOUNDARY_LW, alpha=_BOUNDARY_ALPHA,
                                 zorder=1.2) for x in boundaries]
        for ln in lines:
            ln.set_animated(self._animated)
        self.artists["boundaries"] = lines
        self._boundaries = list(boundaries)
        return lines

    def _overlay_current_events(self) -> None:
        """(Re)draw the event overlay and cache its legend handles/labels."""
        for art in self.artists.pop("events", []):
            try:
                art.remove()
            except (NotImplementedError, ValueError):
                pass
        self._event_handles, self._event_labels = [], []
        self._event_color_map = {}
        if self._events is None:
            return
        arts, handles, labels, color_map = _overlay_events(
            self.ax, self._events, hue=self._event_hue, animated=self._animated)
        self.artists["events"] = arts
        self._event_handles, self._event_labels = handles, labels
        self._event_color_map = color_map

    def _refresh_legend(self) -> None:
        """Draw the single combined legend (signal layers + event categories)."""
        existing = self.ax.get_legend()
        if existing is not None:
            existing.remove()
        if not self._legend:
            return
        handles: list = []
        labels: list = []
        for layer in self._layers:
            if layer["label"]:
                handles.append(layer["lines"][0])
                labels.append(layer["label"])
        handles += self._event_handles
        labels += self._event_labels
        if handles:
            fontsize = (mpl.rcParams["legend.fontsize"]
                        if self._legend_fontsize is None
                        else self._legend_fontsize)
            self.ax.legend(handles, labels, loc=self._legend_loc,
                           fontsize=fontsize, framealpha=0.9)

    def _draw_scale_bar(self) -> None:
        # An anchored bar (a zero-width Rectangle = a vertical tick of the right
        # data height) packed beside its label, pinned just outside the axes.
        # The y-axis is in data units, so the bar height equals the value; under a
        # visual gain the label reports the true amplitude (value / gain).
        old = self.artists.pop("scale_bar", None)
        if old is not None:
            old.remove()
        value = _nice_number(self._offset_value)
        label_value = value / self._gain if self._gain else value
        text_size = mpl.rcParams["legend.fontsize"]
        ink = medusa_style.current_theme().plot_fg
        bars = AuxTransformBox(self.ax.transData)
        bars.add_artist(
            Rectangle((0, 0), 0, value, ec=ink, lw=_SCALE_BAR_LW, fc="none"))
        packer = HPacker(
            children=[bars, TextArea(f"{label_value:g} {self.amplitude_unit}",
                                     textprops=dict(color=ink, size=text_size,
                                                    va="center"))],
            align="center", pad=0, sep=4)
        box = AnchoredOffsetbox(loc="upper left", pad=0, borderpad=0,
                                child=packer, frameon=False,
                                bbox_to_anchor=(1.01, 1.0),
                                bbox_transform=self.ax.transAxes)
        box.set_clip_on(False)
        self.ax.add_artist(box)
        self.artists["scale_bar"] = box


def plot_timeline(data, ax, *, fs: float | None = None,
                  times: ArrayLike | None = None,
                  cha_labels: ArrayLike | None = None,
                  offset: float | None = None,
                  time_grid: float | None = None, events=None,
                  event_hue: str | None = None,
                  **kwargs) -> tuple[Axes, dict]:
    """Plot stacked multichannel traces (one-shot, publication-ready).

    Parameters
    ----------
    data
        ``(n_samples, n_channels)`` or segmented ``(n_segments, n_samples,
        n_channels)`` (concatenated with a boundary line per join).
    ax
        Axes to draw into (**required**).
    fs
        Sampling rate in Hz; sample-index x-axis when ``None``.
    times
        Per-sample timestamps; overrides ``fs``.
    cha_labels, offset, time_grid, events, event_hue
        See :class:`TimeLinePlot`.
    **kwargs
        Forwarded to :class:`TimeLinePlot` (``center``, ``color``, ``gain``,
        ``scale_bar``, ``amplitude_unit``, ``line_width``, ``legend``,
        ``legend_loc``, ``legend_fontsize``, ``animated``).

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes drawn into.
    artists : dict
        ``"lines"`` (one per channel), plus ``"boundaries"`` / ``"events"`` /
        ``"scale_bar"`` / ``"overlays"`` when present.

    Examples
    --------
    >>> ax, artists = plot_timeline(data, ax, fs=256.0, events=events)  # doctest: +SKIP
    >>> artists["lines"][0].set_color("crimson")                        # doctest: +SKIP
    """
    tl = TimeLinePlot(ax, fs=fs, cha_labels=cha_labels, offset=offset,
                      time_grid=time_grid, events=events, event_hue=event_hue,
                      **kwargs)
    tl.set_data(data, times=times)
    return tl.ax, tl.artists
