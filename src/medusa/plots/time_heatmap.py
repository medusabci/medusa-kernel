"""Heatmap of time-distributed multichannel data.

Two modes on one axes, selected by shape: **channel rows** (2-D / segmented — one
image row per channel) and **per-channel images** (3-D + ``y_values`` — a stacked
2-D band per channel, e.g. a spectrogram/scalogram). ``method="imshow"`` (fast,
equidistant samples) or ``"pcolormesh"`` (places cells at the true
``times``/``y_values``, for non-uniform axes). One-shot :func:`plot_timeheatmap`
or stateful :class:`TimeHeatmapPlot` (in-place ``set_data``, blit-ready).
"""

import matplotlib as mpl
import medusa_style
import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from numpy.typing import ArrayLike, NDArray

from medusa.core.utils import check_data_dims
from medusa.plots._timeaxis import (
    _channel_labels,
    _check_events,
    _overlay_events,
    _prepare,
)
from medusa.plots.utils import _add_colorbar, _resolve_cmap

#: Color of the dotted segment-join separators drawn ON TOP of the heatmap
#: image. A kernel-local default (the medusa-style colormaps are
#: theme-independent, so this contrasting white is too): there is no
#: medusa-style palette role for "a line over a colormap".
_SEPARATOR_COLOR = "white"

__all__ = ["plot_timeheatmap", "TimeHeatmapPlot"]

_METHODS = ("imshow", "pcolormesh")
#: Per-band y_values tick size, as a fraction of the active ``font.size`` (the
#: stacked frequency ticks must stay small enough to fit many bands).
_BAND_TICK_FONT_FRAC = 0.65


class TimeHeatmapPlot:
    """Heatmap of time-distributed multichannel data, static-first and blit-ready.

    Exposes :attr:`artists` (``"image"`` rows mode / ``"images"`` stack mode, plus
    ``"boundaries"``/``"y_axis"`` when present), :attr:`mappable` and
    :attr:`cbar`.

    Parameters
    ----------
    ax
        Axes to draw into (**required**); a single axes for both modes.
    fs
        Sampling rate in Hz; sample-index x-axis when ``None``.
    cha_labels
        Channel labels (length ``n_channels``); auto ``Ch1..`` when ``None``.
    cmap
        Colormap; the medusa-style sequential colormap by default
        (``medusa_style.mpl.sequential_cmap()``).
    clim
        Color limits; data range when ``None``.
    y_values
        Per-channel y-coordinates (length ``n_y``); selects per-channel-image mode
        for a 3-D ``(n_y, n_samples, n_channels)`` input (e.g. spectrogram freqs).
    method
        ``"imshow"`` (fast, equidistant) or ``"pcolormesh"`` (true coordinates).
    gap
        Vertical gap between stacked bands, as a fraction of the slot (image mode).
    y_ticks_per_channel
        Number of ``y_values`` ticks per band (image mode).
    events, event_hue
        Event overlay (see :func:`~medusa.plots._timeaxis._overlay_events`); drawn
        above the image. ``event_hue`` colors by an events column.
    legend, legend_loc, legend_fontsize
        Single combined legend of event ``hue`` categories; ``legend=False`` draws
        none (drawn only when there is at least one labelled category).
    colorbar, cbar_label, cbar_fraction, cbar_pad
        Color-scale colorbar (handle on :attr:`cbar`); ``cbar_fraction`` (0.05) is
        the axes-width share, ``cbar_pad`` (0.02) the gap.
    animated
        Mark image artists ``animated=True`` for host-driven blitting.

    Examples
    --------
    >>> hm = TimeHeatmapPlot(ax, fs=256.0); _ = hm.set_data(signal)  # doctest: +SKIP
    >>> hm = TimeHeatmapPlot(ax, y_values=freqs, colorbar=True)      # doctest: +SKIP
    >>> _ = hm.set_data(power, times=t)  # (n_freqs, n_times, n_channels)  # doctest: +SKIP
    """

    def __init__(
            self, ax, *,
            fs: float | None = None,
            cha_labels: ArrayLike | None = None,
            cmap=None,
            clim: tuple[float, float] | None = None,
            y_values: ArrayLike | None = None,
            method: str = "imshow",
            gap: float = 0.1,
            y_ticks_per_channel: int = 3,
            events=None,
            event_hue: str | None = None,
            legend: bool = True,
            legend_loc: str = "upper right",
            legend_fontsize: float | None = None,
            colorbar: bool = False,
            cbar_label: str | None = None,
            cbar_fraction: float = 0.05,
            cbar_pad: float = 0.02,
            animated: bool = False):
        if method not in _METHODS:
            raise ValueError(
                f"method must be one of {_METHODS}, got {method!r}.")
        _check_events(events, event_hue)  # fail early on a malformed table
        self.ax = ax
        self.mappable = None
        self.cbar = None
        self.artists: dict = {}
        self._cmap = _resolve_cmap(cmap, medusa_style.mpl.sequential_cmap())
        self.clim = clim
        self._fs = fs
        self._cha_labels = None if cha_labels is None else list(cha_labels)
        self._y_values = (None if y_values is None
                          else np.asarray(y_values, dtype=float))
        self._method = method
        self._gap = float(gap)
        self._y_ticks = int(y_ticks_per_channel)
        self._events = events
        self._event_hue = event_hue
        self._legend = legend
        self._legend_loc = legend_loc
        self._legend_fontsize = legend_fontsize
        self._event_handles: list = []
        self._event_labels: list = []
        self._event_color_map: dict = {}
        self._colorbar = colorbar
        self._cbar_label = cbar_label
        self._cbar_fraction = float(cbar_fraction)
        self._cbar_pad = float(cbar_pad)
        self._animated = animated
        self._n_channels = None
        self._boundaries: list[float] | None = None

    # -- public -----------------------------------------------------------
    def set_data(self, data, *, fs: float | None = None,
                 times: ArrayLike | None = None,
                 y_values: ArrayLike | None = None) -> list[Artist]:
        """Render/update the heatmap; return the updated artists.

        A 3-D/4-D input combined with ``y_values`` is drawn as per-channel image
        bands (4-D is segmented and concatenated along time with a boundary line
        per join); otherwise the input is a 2-D / segmented signal drawn as channel
        rows. Pass irregular timestamps via ``times`` (honored when
        ``method="pcolormesh"``).
        """
        yv = (self._y_values if y_values is None
              else np.asarray(y_values, dtype=float))
        data = np.asarray(data, dtype=float)
        if yv is not None and data.ndim >= 3:
            if "image" in self.artists:  # switching channel-rows -> image stack
                self._reset()
            updated = self._set_image_stack(data, yv, fs, times)
        else:
            if "images" in self.artists:  # switching image stack -> channel-rows
                self._reset()
            updated = self._set_raster(data, fs, times)
        # Events are static chrome over the image: drawn once (a _reset() clears
        # them, so they are re-overlaid on the next build), not per-frame updated.
        if self._events is not None and "events" not in self.artists:
            self._overlay_current_events()
        self._refresh_legend()
        self._draw_colorbar()
        return updated

    def set_clim(self, lo: float, hi: float) -> None:
        """Set the color limits live on the image(s), mappable and colorbar.

        Persists ``self.clim`` so a later :meth:`set_data` keeps these limits
        instead of reverting to the auto data range.
        """
        self.clim = (float(lo), float(hi))
        if "image" in self.artists:
            self.artists["image"].set_clim(lo, hi)
        for im in self.artists.get("images", []):
            im.set_clim(lo, hi)
        if self.mappable is not None:
            self.mappable.set_clim(lo, hi)
        if self.cbar is not None:
            self.cbar.update_normal(self.mappable)

    def set_events(self, events, *, hue: str | None = None) -> None:
        """Replace the overlaid events (and optionally the ``hue`` column)."""
        _check_events(events, hue if hue is not None else self._event_hue)
        self._events = events
        if hue is not None:
            self._event_hue = hue
        self._overlay_current_events()
        self._refresh_legend()

    def set_cmap(self, cmap) -> None:
        """Set the colormap live on the image(s), mappable and colorbar."""
        self._cmap = _resolve_cmap(cmap, medusa_style.mpl.sequential_cmap())
        if "image" in self.artists:
            self.artists["image"].set_cmap(self._cmap)
        for im in self.artists.get("images", []):
            im.set_cmap(self._cmap)
        if self.mappable is not None:
            self.mappable.set_cmap(self._cmap)
        if self.cbar is not None:
            self.cbar.update_normal(self.mappable)

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

    # -- internals --------------------------------------------------------
    def _draw_colorbar(self) -> None:
        """Draw the colorbar once, then re-point it at later mappables.

        Created lazily on the first ``set_data`` (the mappable only exists then).
        Across a :meth:`_reset` the colorbar axes persists, so :func:`_add_colorbar`
        updates it to the new mappable instead of re-adding one (which would shrink
        the axes again).
        """
        if not self._colorbar or self.mappable is None:
            return
        self.cbar = _add_colorbar(
            self.mappable, self.ax, label=self._cbar_label,
            fraction=self._cbar_fraction, pad=self._cbar_pad, cbar=self.cbar)

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
            self.ax, self._events, hue=self._event_hue,
            animated=self._animated, line_zorder=3.0, span_zorder=2.5)
        self.artists["events"] = arts
        self._event_handles, self._event_labels = handles, labels
        self._event_color_map = color_map

    def _refresh_legend(self) -> None:
        """Draw the single combined legend (event ``hue`` categories)."""
        existing = self.ax.get_legend()
        if existing is not None:
            existing.remove()
        if not self._legend or not self._event_handles:
            return
        fontsize = (mpl.rcParams["legend.fontsize"]
                    if self._legend_fontsize is None else self._legend_fontsize)
        self.ax.legend(self._event_handles, self._event_labels,
                       loc=self._legend_loc, fontsize=fontsize,
                       framealpha=0.9)

    def _reset(self) -> None:
        """Tear down all artists + axis furniture (mode or channel count changed).

        Mirrors :meth:`TimeLinePlot._build`'s teardown so a later ``set_data``
        with a different channel count (or mode) redraws cleanly instead of
        reusing stale rows/bands or indexing past the old image list.
        """
        yax = self.artists.pop("y_axis", None)
        if yax is not None:
            yax.remove()  # the per-band y_values twin lives on the figure, not ax
        self.ax.clear()
        self.artists = {}
        self._boundaries = None
        self._n_channels = None

    def _clim_for(self, values: NDArray) -> tuple[float, float]:
        if self.clim is not None:
            return self.clim
        lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
        return (lo - 0.5, hi + 0.5) if lo == hi else (lo, hi)

    def _time_axis(self, n_time, fs, times) -> tuple[NDArray, str]:
        fs = self._fs if fs is None else fs
        if times is not None:
            return np.asarray(times, dtype=float).ravel(), "s"
        if fs is not None:
            return np.arange(n_time) / float(fs), "s"
        return np.arange(n_time, dtype=float), "samples"

    def _setup_axes(self, n_ch: int, unit: str) -> None:
        labels = _channel_labels(self._cha_labels, n_ch)
        self.ax.set_yticks([i + 0.5 for i in range(n_ch)])
        self.ax.set_yticklabels(labels)
        self.ax.set_ylim(n_ch, 0)
        self.ax.set_xlabel("Time (s)" if unit == "s" else "Samples")
        self.ax.grid(False)  # a grid over a heatmap is just noise

    def _set_raster(self, data, fs, times) -> list[Artist]:
        data2d, t, unit, boundaries = _prepare(
            data, self._fs if fs is None else fs, times)
        img = data2d.T  # (n_channels, n_samples)
        n_ch = img.shape[0]
        if self._n_channels is not None and n_ch != self._n_channels:
            self._reset()  # channel count changed -> rebuild from scratch
        clim = self._clim_for(img)
        first = "image" not in self.artists
        if self._method == "imshow":
            x0, x1 = float(t[0]), float(t[-1])
            if first:
                im = self.ax.imshow(img, extent=(x0, x1, n_ch, 0),
                                    origin="upper", aspect="auto",
                                    cmap=self._cmap, interpolation="nearest",
                                    zorder=0)
                im.set_animated(self._animated)
            else:
                im = self.artists["image"]
                im.set_data(img)
                im.set_extent((x0, x1, n_ch, 0))
            im.set_clim(*clim)
        else:  # pcolormesh: cells at true timestamps (non-equidistant safe)
            old = self.artists.get("image")
            if old is not None:
                old.remove()
            im = self.ax.pcolormesh(t, np.arange(n_ch) + 0.5, img,
                                    cmap=self._cmap, shading="nearest",
                                    vmin=clim[0], vmax=clim[1], zorder=0)
            im.set_animated(self._animated)
            self.ax.set_xlim(float(t[0]), float(t[-1]))
        self.artists["image"] = im
        if first:
            self._n_channels = n_ch
            self._setup_axes(n_ch, unit)
        self.mappable = im
        return [im, *self._update_boundaries(boundaries)]

    def _set_image_stack(self, data, y_values, fs, times) -> list[Artist]:
        # Per-channel 2-D images stacked as bands in one axes (channel 0 on top),
        # a small gap between them, and the y_values drawn in place per band — so
        # every channel keeps a readable vertical scale (e.g. frequency). A 4-D
        # (segmented) input is concatenated along time, with a boundary per join.
        data = np.asarray(data, dtype=float)
        if data.ndim == 4:
            data, _ = check_data_dims(data, rep_type="time_freq_segments")
            n_seg, n_y, n_samp, n_ch = data.shape
            joins = [k * n_samp for k in range(1, n_seg)]
            data = np.concatenate([data[k] for k in range(n_seg)], axis=1)
        else:
            data, _ = check_data_dims(data, rep_type="time_freq")
            joins = []
        n_y, n_time, n_ch = data.shape
        if self._n_channels is not None and n_ch != self._n_channels:
            self._reset()  # channel count changed -> rebuild bands from scratch
        if len(y_values) != n_y:
            raise ValueError(
                f"y_values has length {len(y_values)}; expected {n_y} "
                "(the per-channel image height).")
        t, unit = self._time_axis(n_time, fs, times)
        x0, x1 = float(t[0]), float(t[-1])
        clim = self._clim_for(data)
        band_h = 1.0 - self._gap
        y0, y1 = float(y_values[0]), float(y_values[-1])
        span = (y1 - y0) or 1.0
        first = "images" not in self.artists
        old_images = self.artists.get("images", [])
        images: list = []
        for i in range(n_ch):
            bottom = (n_ch - 1 - i)  # channel 0 on top, slot height 1.0
            if self._method == "imshow":
                extent = (x0, x1, bottom, bottom + band_h)
                if first:
                    im = self.ax.imshow(data[:, :, i], extent=extent,
                                        origin="lower", aspect="auto",
                                        cmap=self._cmap, vmin=clim[0],
                                        vmax=clim[1], interpolation="nearest",
                                        zorder=1)
                    im.set_animated(self._animated)
                else:
                    im = old_images[i]
                    im.set_data(data[:, :, i])
                    im.set_extent(extent)
                    im.set_clim(*clim)
            else:  # pcolormesh: cells at true (times, y_values) coordinates
                if not first:
                    old_images[i].remove()
                y_band = bottom + (y_values - y0) / span * band_h
                im = self.ax.pcolormesh(t, y_band, data[:, :, i],
                                        cmap=self._cmap, shading="nearest",
                                        vmin=clim[0], vmax=clim[1], zorder=1)
                im.set_animated(self._animated)
            images.append(im)
        self.artists["images"] = images
        if first:
            self._setup_band_axes(n_ch, y_values, band_h)
            self.ax.set_xlabel("Time (s)" if unit == "s" else "Samples")
        self.ax.set_xlim(x0, x1)
        self._n_channels = n_ch
        sm = ScalarMappable(norm=Normalize(*clim), cmap=self._cmap)
        sm.set_array([])
        self.mappable = sm
        boundary_x = [float(t[j]) for j in joins]
        return list(images) + self._update_boundaries(boundary_x)

    def _setup_band_axes(self, n_ch, y_values, band_h) -> None:
        # Channel names at band centers (main y-axis, pushed out), and the
        # y_values ticks in place per band on a twin y-axis sitting at the edge.
        labels = _channel_labels(self._cha_labels, n_ch)
        gap = self._gap
        centers = [(n_ch - 1 - i) + band_h / 2 for i in range(n_ch)]
        self.ax.set_ylim(-gap / 2, n_ch - gap / 2)
        self.ax.set_yticks(centers)
        self.ax.set_yticklabels(labels, fontweight="semibold")
        self.ax.tick_params(axis="y", length=0, pad=26)
        self.ax.grid(False)  # a grid over a heatmap is just noise

        n_y = len(y_values)
        idx = np.unique(np.linspace(0, n_y - 1, self._y_ticks).round()
                        .astype(int))
        y0, y1 = float(y_values[0]), float(y_values[-1])
        span = (y1 - y0) or 1.0
        positions, ticklabels = [], []
        for i in range(n_ch):
            bottom = (n_ch - 1 - i)
            for j in idx:
                # imshow places rows uniformly (linear in index); pcolormesh
                # places them at the true value (linear in y_values).
                frac = (j / (n_y - 1) if n_y > 1 else 0.0) if (
                    self._method == "imshow") else (y_values[j] - y0) / span
                positions.append(bottom + frac * band_h)
                ticklabels.append(f"{y_values[j]:g}")
        yax = self.ax.twinx()
        yax.set_frame_on(False)
        yax.yaxis.set_ticks_position("left")
        yax.yaxis.set_label_position("left")
        yax.set_ylim(self.ax.get_ylim())
        yax.set_yticks(positions)
        yax.set_yticklabels(
            ticklabels, fontsize=_BAND_TICK_FONT_FRAC * mpl.rcParams["font.size"])
        yax.tick_params(axis="y", length=2, pad=1)
        yax.grid(False)  # the twin (y_values) axis must not draw gridlines
        self.artists["y_axis"] = yax

    def _update_boundaries(self, boundaries: list[float]) -> list[Artist]:
        if boundaries == self._boundaries:
            return []
        for ln in self.artists.pop("boundaries", []):
            ln.remove()
        lines = [self.ax.axvline(x, color=_SEPARATOR_COLOR, ls=":", lw=1.0,
                                 alpha=0.7, zorder=1.5) for x in boundaries]
        for ln in lines:
            ln.set_animated(self._animated)
        self.artists["boundaries"] = lines
        self._boundaries = list(boundaries)
        return lines


def plot_timeheatmap(data, ax, *, fs: float | None = None,
                     times: ArrayLike | None = None,
                     cha_labels: ArrayLike | None = None, cmap=None,
                     clim: tuple[float, float] | None = None,
                     y_values: ArrayLike | None = None,
                     method: str = "imshow", events=None,
                     event_hue: str | None = None, colorbar: bool = False,
                     cbar_label: str | None = None, cbar_fraction: float = 0.05,
                     cbar_pad: float = 0.02,
                     **kwargs) -> tuple[Axes, dict]:
    """Plot a heatmap of time-distributed multichannel data (one-shot).

    Parameters
    ----------
    data
        ``(n_samples, n_channels)`` / segmented ``(n_segments, n_samples,
        n_channels)`` (channel rows), or ``(n_y, n_samples, n_channels)`` with
        ``y_values`` (per-channel image bands).
    ax
        Axes to draw into (**required**).
    fs
        Sampling rate in Hz; sample-index x-axis when ``None``.
    times
        Per-sample timestamps; overrides ``fs`` (use with ``"pcolormesh"`` for
        irregular sampling).
    cha_labels, cmap, clim, y_values, method, events, event_hue
        See :class:`TimeHeatmapPlot`.
    colorbar, cbar_label, cbar_fraction, cbar_pad
        Colorbar and its compact sizing (``cbar_fraction`` 0.05, ``cbar_pad``
        0.02).
    **kwargs
        Forwarded to :class:`TimeHeatmapPlot` (``gap``, ``y_ticks_per_channel``,
        ``animated``).

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes drawn into.
    artists : dict
        ``"image"`` or ``"images"`` (+ ``"y_axis"``), and ``"boundaries"`` /
        ``"events"`` when present.

    Examples
    --------
    >>> ax, artists = plot_timeheatmap(signal, ax, fs=256.0, colorbar=True)  # doctest: +SKIP
    >>> ax, artists = plot_timeheatmap(power, ax, times=t, y_values=freqs,   # doctest: +SKIP
    ...                                method="pcolormesh", colorbar=True)
    """
    hm = TimeHeatmapPlot(ax, fs=fs, cha_labels=cha_labels, cmap=cmap, clim=clim,
                         y_values=y_values, method=method, events=events,
                         event_hue=event_hue, colorbar=colorbar,
                         cbar_label=cbar_label, cbar_fraction=cbar_fraction,
                         cbar_pad=cbar_pad, **kwargs)
    hm.set_data(data, times=times)
    return hm.ax, hm.artists
