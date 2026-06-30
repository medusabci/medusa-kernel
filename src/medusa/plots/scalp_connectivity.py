"""Scalp connectivity: thresholded edges between channel nodes over the head.

One-shot :func:`plot_connectivity` or stateful :class:`ConnectivityPlot`, drawn
over the shared head substrate (:func:`~medusa.plots.plot_scalp`).
"""

import medusa_style
import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from numpy.typing import ArrayLike, NDArray

from medusa.plots.scalp import _draw_head_artists, _scalp_positions
from medusa.plots.utils import _add_colorbar, _resolve_cmap

__all__ = ["plot_connectivity", "ConnectivityPlot"]


def _align_matrix(adjacency: ArrayLike, channel_set, labels) -> NDArray:
    """Align a square matrix to the plotted ``labels`` (full-set or subset)."""
    adj = np.asarray(adjacency, dtype=float)
    m = len(labels)
    if adj.shape == (m, m):
        return adj
    n = channel_set.n_channels
    if adj.shape == (n, n):
        idx = channel_set.index(labels)
        return adj[np.ix_(idx, idx)]
    raise ValueError(
        f"adjacency must be ({m}, {m}) [located channels] or ({n}, {n}) "
        f"[all channels], got {adj.shape}.")


class ConnectivityPlot:
    """Thresholded connectivity edges over a head, static-first and blit-ready.

    Plots every channel in ``channel_set`` (pick the channels you want first).
    Draws the head once; :meth:`set_data` redraws the strongest edges of an
    adjacency matrix (color = weight, width = magnitude). Exposes :attr:`artists`,
    :attr:`mappable`, :attr:`cbar`.

    Parameters
    ----------
    channel_set
        Channel set providing electrode positions.
    ax
        Axes to draw into (**required**).
    threshold
        Percentile of ``|weight|`` (``[0, 100]``) above which edges are drawn.
    cmap, clim
        Colormap (diverging default) and color limits (shown edges when ``None``).
    max_line_width
        Width of the strongest edge (others scale linearly).
    colorbar, cbar_label, cbar_fraction, cbar_pad
        Color-scale colorbar (handle on :attr:`cbar`) and its compact sizing.
    radius, line_width, head_color, skin_color, show_sensors, show_labels, sensor_color, sensor_size, label_color
        Head-substrate styling; see :func:`~medusa.plots.plot_scalp`.
    animated
        Mark the edge collection ``animated=True`` for host-driven blitting.

    Examples
    --------
    >>> cp = ConnectivityPlot(cs, ax, threshold=90, colorbar=True)  # doctest: +SKIP
    >>> _ = cp.set_data(adjacency)                                  # doctest: +SKIP
    """

    def __init__(self, channel_set, ax, *,
                 threshold: float = 85.0,
                 cmap=None, clim: tuple[float, float] | None = None,
                 radius: float = 1.0, line_width: float = 2.0,
                 head_color: str | None = None, skin_color: str | None = None,
                 show_sensors: bool = True, show_labels: bool = False,
                 sensor_color: str | None = None, sensor_size: float = 14.0,
                 sensor_edge_color: str | None = None,
                 label_color: str | None = None,
                 label_fontsize: float | None = None,
                 max_line_width: float = 3.0,
                 colorbar: bool = False, cbar_label: str | None = None,
                 cbar_fraction: float = 0.05, cbar_pad: float = 0.02,
                 animated: bool = False):
        self.ax = ax
        self.channel_set = channel_set
        self.threshold = threshold
        self.clim = clim
        self.max_line_width = max_line_width
        self._colorbar = colorbar
        self._cbar_label = cbar_label
        self._cbar_fraction = float(cbar_fraction)
        self._cbar_pad = float(cbar_pad)
        self._animated = animated
        self._cmap = _resolve_cmap(cmap, medusa_style.mpl.diverging_cmap())
        self._pos, self._labels = _scalp_positions(channel_set)
        self.artists = _draw_head_artists(
            self.ax, self._pos, self._labels, radius=radius,
            line_width=line_width, head_color=head_color, skin_color=skin_color,
            show_sensors=show_sensors, show_labels=show_labels,
            sensor_color=sensor_color, sensor_size=sensor_size,
            sensor_edge_color=sensor_edge_color, label_color=label_color,
            label_fontsize=label_fontsize)
        self.mappable = None
        self.cbar = None

    def set_data(self, adjacency: ArrayLike) -> list[Artist]:
        """Render/update the connectivity edges; return the updated artists."""
        adj = _align_matrix(adjacency, self.channel_set, self._labels)
        m = len(self._labels)
        iu = np.triu_indices(m, 1)
        weights = adj[iu]
        finite = weights[np.isfinite(weights)]
        thr = np.percentile(np.abs(finite), self.threshold) if finite.size else 0.0
        keep = np.isfinite(weights) & (np.abs(weights) >= thr)
        rows, cols = iu[0][keep], iu[1][keep]
        shown = weights[keep]
        segments = [[self._pos[r], self._pos[c]] for r, c in zip(rows, cols)]

        if shown.size:
            clim = self.clim if self.clim is not None else (
                float(shown.min()), float(shown.max()))
        else:
            clim = self.clim if self.clim is not None else (0.0, 1.0)
        if clim[0] == clim[1]:
            clim = (clim[0] - 0.5, clim[1] + 0.5)
        sm = ScalarMappable(norm=Normalize(*clim), cmap=self._cmap)
        colors = sm.to_rgba(shown)
        span = clim[1] - clim[0]
        widths = (self.max_line_width * np.abs(shown - clim[0]) / span
                  if span else np.full(shown.shape, self.max_line_width))

        if "edges" not in self.artists:
            lc = LineCollection(segments, colors=colors, linewidths=widths,
                                zorder=2, capstyle="round")
            lc.set_animated(self._animated)
            self.ax.add_collection(lc)
            self.artists["edges"] = lc
        else:
            lc = self.artists["edges"]
            lc.set_segments(segments)
            lc.set_color(colors)
            lc.set_linewidths(widths)
        sm.set_array([])
        self.mappable = sm
        if self._colorbar:
            self.cbar = _add_colorbar(
                self.mappable, self.ax, label=self._cbar_label,
                fraction=self._cbar_fraction, pad=self._cbar_pad, cbar=self.cbar)
        return [lc]


def plot_connectivity(adjacency: ArrayLike, channel_set, ax, *,
                      threshold: float = 85.0,
                      cmap=None, clim: tuple[float, float] | None = None,
                      show_sensors: bool = True, show_labels: bool = False,
                      colorbar: bool = False, cbar_label: str | None = None,
                      cbar_fraction: float = 0.05, cbar_pad: float = 0.02,
                      **kwargs) -> tuple[Axes, dict]:
    """Plot scalp connectivity edges (one-shot).

    Plots every channel in ``channel_set`` (pick the channels you want first).

    Parameters
    ----------
    adjacency
        ``(n_located, n_located)`` or ``(n_channels, n_channels)`` weight matrix
        (the latter is auto-subset to the located channels).
    channel_set
        Channel set providing electrode positions.
    ax
        Axes to draw into (**required**).
    threshold, cmap, clim, show_sensors, show_labels
        See :class:`ConnectivityPlot`.
    colorbar, cbar_label, cbar_fraction, cbar_pad
        Colorbar and its compact sizing.
    **kwargs
        Forwarded to :class:`ConnectivityPlot` (head styling, ``max_line_width``).

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes drawn into.
    artists : dict
        Named artists (head substrate + ``"edges"``), for post-hoc styling.

    Examples
    --------
    >>> ax, artists = plot_connectivity(adj, cs, ax, threshold=90,
    ...                                 colorbar=True, cbar_label="PLV")  # doctest: +SKIP
    """
    cp = ConnectivityPlot(
        channel_set, ax, threshold=threshold, cmap=cmap, clim=clim,
        show_sensors=show_sensors, show_labels=show_labels, colorbar=colorbar,
        cbar_label=cbar_label, cbar_fraction=cbar_fraction, cbar_pad=cbar_pad,
        **kwargs)
    cp.set_data(adjacency)
    return cp.ax, cp.artists
