"""Scalp topography: a scalar field over the head (interpolated or discrete).

One-shot :func:`plot_topography` or stateful :class:`TopographicPlot`, drawn over
the shared head substrate (:func:`~medusa.plots.plot_scalp`).
"""

import medusa_style
import numpy as np
from matplotlib import patches
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import griddata
from scipy.spatial.distance import pdist

from medusa.plots.scalp import _draw_head_artists, _scalp_positions
from medusa.plots.utils import _add_colorbar, _resolve_cmap

__all__ = ["plot_topography", "TopographicPlot"]

#: Contour overlay styling (interpolated mode): faint ink isolines.
_CONTOUR_ALPHA = 0.4
_CONTOUR_LW = 0.6


def _align_values(values: ArrayLike, channel_set, labels) -> NDArray:
    """Align ``values`` to the plotted ``labels`` (accepts full-set or subset)."""
    values = np.asarray(values, dtype=float).ravel()
    if values.shape[0] == len(labels):
        return values
    if values.shape[0] == channel_set.n_channels:
        return values[channel_set.index(labels)]
    raise ValueError(
        f"values has length {values.shape[0]}; expected {len(labels)} (located "
        f"channels) or {channel_set.n_channels} (all channels).")


def _min_spacing(pos: NDArray) -> float:
    """Smallest distance between any two points (fallback for < 2 points)."""
    if len(pos) < 2:
        return 0.2
    return float(pdist(pos).min())


def _knn_values(targets: NDArray, pos: NDArray, values: NDArray,
                k: int) -> NDArray:
    """Mean value of the ``k`` nearest real electrodes for each target point."""
    out = np.empty(len(targets))
    for i, p in enumerate(targets):
        d = np.linalg.norm(pos - p, axis=1)
        out[i] = values[np.argsort(d)[:k]].mean()
    return out


class TopographicPlot:
    """Scalar-field topography over a head, static-first and blit-ready.

    Plots every channel in ``channel_set`` (pick the channels you want first).
    Draws the head once; :meth:`set_data` redraws the value layer. Exposes
    :attr:`artists`, :attr:`mappable`, :attr:`cbar`.

    Parameters
    ----------
    channel_set
        Channel set providing electrode positions.
    ax
        Axes to draw into (**required**).
    interpolate
        Interpolated scalp surface (default) or discrete per-channel discs.
    contour
        Draw contour lines (interpolated mode; skipped when ``animated``).
    cmap, clim
        Colormap (sequential default) and color limits (data range when ``None``).
    interp_resolution, interp_neighbors, extra_radius
        Interpolation grid resolution, ring-electrode neighbours, and mask-radius
        margin (interpolated mode).
    colorbar, cbar_label, cbar_fraction, cbar_pad
        Color-scale colorbar (handle on :attr:`cbar`) and its compact sizing.
    radius, line_width, head_color, skin_color, show_sensors, show_labels, sensor_color, sensor_size, label_color
        Head-substrate styling; see :func:`~medusa.plots.plot_scalp`.
    animated
        Mark value artists ``animated=True`` for host-driven blitting.

    Examples
    --------
    >>> tp = TopographicPlot(cs, ax, colorbar=True)  # doctest: +SKIP
    >>> _ = tp.set_data(values)                      # doctest: +SKIP
    """

    def __init__(
            self, channel_set, ax, *,
            interpolate: bool = True,
            contour: bool = True,
            cmap=None,
            clim: tuple[float, float] | None = None,
            radius: float = 1.0,
            line_width: float = 2.0,
            head_color: str | None = None,
            skin_color: str | None = None,
            show_sensors: bool = True,
            show_labels: bool = False,
            sensor_color: str | None = None,
            sensor_size: float = 14.0,
            sensor_edge_color: str | None = None,
            label_color: str | None = None,
            label_fontsize: float | None = None,
            interp_resolution: int = 200,
            interp_neighbors: int = 3,
            extra_radius: float = 0.0,
            colorbar: bool = False,
            cbar_label: str | None = None,
            cbar_fraction: float = 0.05,
            cbar_pad: float = 0.02,
            animated: bool = False):
        self.ax = ax
        self.channel_set = channel_set
        self.interpolate = interpolate
        self.contour = contour
        self.clim = clim
        self.radius = radius
        self.extra_radius = extra_radius
        self.interp_resolution = interp_resolution
        self.interp_neighbors = interp_neighbors
        self._colorbar = colorbar
        self._cbar_label = cbar_label
        self._cbar_fraction = float(cbar_fraction)
        self._cbar_pad = float(cbar_pad)
        self._animated = animated
        self._cmap = _resolve_cmap(cmap, medusa_style.mpl.sequential_cmap())
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

    # -- public -----------------------------------------------------------
    def set_data(self, values: ArrayLike) -> list[Artist]:
        """Render/update the scalar field; return the updated artists."""
        v = _align_values(values, self.channel_set, self._labels)
        clim = (self.clim if self.clim is not None
                else (float(np.nanmin(v)), float(np.nanmax(v))))
        if clim[0] == clim[1]:
            clim = (clim[0] - 0.5, clim[1] + 0.5)
        updated = (self._set_interp(v, clim) if self.interpolate
                   else self._set_discrete(v, clim))
        if self._colorbar and self.mappable is not None:
            self.cbar = _add_colorbar(
                self.mappable, self.ax, label=self._cbar_label,
                fraction=self._cbar_fraction, pad=self._cbar_pad, cbar=self.cbar)
        return updated

    # -- internals --------------------------------------------------------
    def _set_interp(self, v, clim):
        pos = self._pos
        # Virtual electrodes on a ring stabilize the edge interpolation.
        ring = 1.5
        ang = np.arange(0, 2 * np.pi, 2 * np.pi / 16)
        add_xy = np.column_stack([ring * np.cos(ang), ring * np.sin(ang)])
        add_v = _knn_values(add_xy, pos, v, self.interp_neighbors)
        pts = np.vstack([pos, add_xy])
        vals = np.concatenate([v, add_v])
        lin = np.linspace(-ring, ring, self.interp_resolution)
        gx, gy = np.meshgrid(lin, lin)
        zi = griddata(pts, vals, (gx, gy), method="cubic")
        mask_r = max(self.radius,
                     float(np.linalg.norm(pos, axis=1).max())) + self.extra_radius
        zi[np.sqrt(gx ** 2 + gy ** 2) > mask_r] = np.nan

        if "image" not in self.artists:
            im = self.ax.imshow(
                zi, extent=(-ring, ring, -ring, ring), origin="lower",
                cmap=self._cmap, interpolation="bilinear", zorder=0)
            im.set_clip_path(patches.Circle((0, 0), self.radius,
                                            transform=self.ax.transData))
            im.set_animated(self._animated)
            self.artists["image"] = im
        else:
            im = self.artists["image"]
            im.set_data(zi)
        im.set_clim(*clim)
        self.mappable = im
        updated: list = [im]

        if self.contour and not self._animated:
            if "contour" in self.artists:
                self.artists.pop("contour").remove()
            cs = self.ax.contour(
                gx, gy, zi, colors=medusa_style.current_theme().plot_fg,
                alpha=_CONTOUR_ALPHA, linewidths=_CONTOUR_LW, zorder=1)
            cs.set_clip_path(patches.Circle((0, 0), self.radius,
                                            transform=self.ax.transData))
            self.artists["contour"] = cs
            updated.append(cs)
        return updated

    def _set_discrete(self, v, clim):
        sm = ScalarMappable(norm=Normalize(*clim), cmap=self._cmap)
        colors = sm.to_rgba(v)
        if "discs" not in self.artists:
            r = 0.5 * _min_spacing(self._pos)
            discs = []
            for (x, y), c in zip(self._pos, colors):
                patch = patches.Circle((x, y), r, facecolor=c,
                                       edgecolor="none", zorder=4)
                patch.set_animated(self._animated)
                self.ax.add_patch(patch)
                discs.append(patch)
            self.artists["discs"] = discs
        else:
            for patch, c in zip(self.artists["discs"], colors):
                patch.set_facecolor(c)
        sm.set_array([])
        self.mappable = sm
        return list(self.artists["discs"])


def plot_topography(values: ArrayLike, channel_set, ax, *,
                    interpolate: bool = True, contour: bool = True,
                    cmap=None, clim: tuple[float, float] | None = None,
                    show_sensors: bool = True, show_labels: bool = False,
                    colorbar: bool = False, cbar_label: str | None = None,
                    cbar_fraction: float = 0.05, cbar_pad: float = 0.02,
                    **kwargs) -> tuple[Axes, dict]:
    """Plot a scalp topography (one-shot).

    Plots every channel in ``channel_set`` (pick the channels you want first).

    Parameters
    ----------
    values
        ``(n_located,)`` or ``(n_channels,)`` scalar per channel.
    channel_set
        Channel set providing electrode positions.
    ax
        Axes to draw into (**required**).
    interpolate, contour, cmap, clim, show_sensors, show_labels
        See :class:`TopographicPlot` (pass a diverging ``cmap`` for signed values).
    colorbar, cbar_label, cbar_fraction, cbar_pad
        Colorbar and its compact sizing.
    **kwargs
        Forwarded to :class:`TopographicPlot` (head styling, interpolation params).

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes drawn into.
    artists : dict
        Head substrate + ``"image"``/``"contour"`` or ``"discs"``.

    Examples
    --------
    >>> ax, artists = plot_topography(values, cs, ax, colorbar=True)  # doctest: +SKIP
    """
    tp = TopographicPlot(
        channel_set, ax, interpolate=interpolate, contour=contour, cmap=cmap,
        clim=clim, show_sensors=show_sensors, show_labels=show_labels,
        colorbar=colorbar, cbar_label=cbar_label, cbar_fraction=cbar_fraction,
        cbar_pad=cbar_pad, **kwargs)
    tp.set_data(values)
    return tp.ax, tp.artists
