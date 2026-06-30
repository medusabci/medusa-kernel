"""Head substrate for scalp plots: the 2-D head diagram + projected sensors.

:func:`plot_scalp` is the shared substrate for :mod:`~medusa.plots.scalp_topography`
and :mod:`~medusa.plots.scalp_connectivity`. Positions come from projecting the
channel set's 3-D coordinates (:func:`~medusa.core.data.eeg.project_to_2d`).
"""

import warnings

import matplotlib as mpl
import medusa_style
import numpy as np
from matplotlib.axes import Axes

from medusa.core.data.eeg import project_to_2d as project_azimuthal

__all__ = ["plot_scalp"]

# Head-diagram geometry (substrate shape, not brand identity) — named so the
# drawing code reads as intent rather than magic numbers.
_OUTLINE_RES = 400              # samples along the head / ear / nose curves
_NOSE_W, _NOSE_H = 0.15, 0.22   # nose half-angle and protrusion fractions
_EAR_A, _EAR_B = 0.08, 0.16     # ear ellipse semi-axes
_EAR_ANGLE = 0.9 * np.pi / 8    # ear arc opening angle
_EAR_OFFSET = 0.058             # lateral ear offset (* radius)
_LABEL_DROP = 0.045             # channel-label y-offset below the sensor
_VIEW_MARGIN = 0.2              # axes padding around the head
_NOSE_HEADROOM = 0.1            # extra top headroom for the nose
_SENSOR_EDGE_WIDTH = 0.5        # sensor marker halo width
_LABEL_FONT_FRAC = 0.7         # channel-label size as a fraction of font.size
# Stacking order: fills < outline < sensors < labels.
_Z_FILL, _Z_OUTLINE, _Z_SENSORS, _Z_LABELS = 0.5, 3, 5, 6


def _bare_axes(ax) -> None:
    """Strip ``ax`` to a blank canvas for diagram drawing — a head, not a chart.

    A scalp head is geometry, not a plot, so it deliberately overrides the gridded
    chart look of the active MEDUSA style for this one axes: equal aspect, no
    ticks / spines / grid, and a transparent face so it composites over whatever
    background is active (light or dark).
    """
    ax.set_aspect("equal", "box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor("none")


def _scalp_positions(channel_set):
    """2-D projected positions + labels of every located channel in the set."""
    pos3d = channel_set.get_positions(types=None)
    labels = list(channel_set.uids)
    pos2d = project_azimuthal(pos3d)
    valid = ~np.isnan(pos2d).any(axis=1)
    if not valid.all():
        dropped = [labels[i] for i in range(len(labels)) if not valid[i]]
        warnings.warn(
            f"Dropping {len(dropped)} channel(s) without a resolved location "
            f"from the scalp plot: {dropped}", stacklevel=2)
    pos2d = pos2d[valid]
    labels = [lab for lab, ok in zip(labels, valid) if ok]
    if not labels:
        raise ValueError("No located channels to plot.")
    return pos2d, labels


def __ear_rho(theta, a, b):
    """Ellipse radius for the ear outline at the given angles."""
    d1 = np.cos(theta) ** 2 / a ** 2
    d2 = np.sin(theta) ** 2 / b ** 2
    return 1.0 / np.sqrt(d1 + d2)


def _draw_head_artists(ax, pos, labels, *, radius, line_width, head_color,
                       skin_color, show_sensors, show_labels, sensor_color,
                       sensor_size, label_color, sensor_edge_color=None,
                       label_fontsize=None):
    """Draw the head diagram + sensors (+ optional labels); return handles.

    Any color left as ``None`` falls back to the active medusa-style theme: the
    foreground (``plot_fg``) for outlines / sensors / labels, the background
    (``plot_bg``) for the sensor-marker halo, so the diagram matches the figure.
    """
    palette = medusa_style.current_theme().palette
    if head_color is None:
        head_color = palette.plot_fg
    if sensor_color is None:
        sensor_color = palette.plot_fg
    if label_color is None:
        label_color = palette.plot_fg
    if sensor_edge_color is None:
        sensor_edge_color = palette.plot_bg
    handles: dict = {}
    n = _OUTLINE_RES

    # Nose
    nose_rho = np.array([radius, radius + radius * _NOSE_H, radius])
    nose_theta = np.array([np.pi / 2 + _NOSE_W * np.pi / 2, np.pi / 2,
                           np.pi / 2 - _NOSE_W * np.pi / 2])
    nose_x, nose_y = nose_rho * np.cos(nose_theta), nose_rho * np.sin(nose_theta)

    # Ears
    offset = _EAR_OFFSET * radius
    tr = np.linspace(-np.pi / 2 - _EAR_ANGLE, np.pi / 2 + _EAR_ANGLE, n)
    tl = np.linspace(np.pi / 2 - _EAR_ANGLE, 3 * np.pi / 2 + _EAR_ANGLE, n)
    ear_xr = __ear_rho(tr, _EAR_A, _EAR_B) * np.cos(tr) + radius + offset
    ear_yr = __ear_rho(tr, _EAR_A, _EAR_B) * np.sin(tr)
    ear_xl = __ear_rho(tl, _EAR_A, _EAR_B) * np.cos(tl) - radius - offset
    ear_yl = __ear_rho(tl, _EAR_A, _EAR_B) * np.sin(tl)

    # Head outline
    ht = np.linspace(0, 2 * np.pi, n)
    head_x, head_y = radius * np.cos(ht), radius * np.sin(ht)

    if skin_color is not None:
        handles["head-fill"] = ax.fill(
            head_x, head_y, facecolor=skin_color, edgecolor="none",
            zorder=_Z_FILL)[0]
        handles["nose-fill"] = ax.fill(
            nose_x, nose_y, facecolor=skin_color, edgecolor="none",
            zorder=_Z_FILL)[0]
        handles["right-ear-fill"] = ax.fill(
            ear_xr, ear_yr, facecolor=skin_color, edgecolor="none",
            zorder=_Z_FILL)[0]
        handles["left-ear-fill"] = ax.fill(
            ear_xl, ear_yl, facecolor=skin_color, edgecolor="none",
            zorder=_Z_FILL)[0]

    lk = dict(color=head_color, linewidth=line_width, solid_capstyle="round",
              zorder=_Z_OUTLINE)
    handles["head-line"] = ax.plot(head_x, head_y, **lk)[0]
    handles["nose-line"] = ax.plot(nose_x, nose_y, **lk)[0]
    handles["right-ear-line"] = ax.plot(ear_xr, ear_yr, **lk)[0]
    handles["left-ear-line"] = ax.plot(ear_xl, ear_yl, **lk)[0]

    if show_sensors:
        handles["sensors"] = ax.scatter(
            pos[:, 0], pos[:, 1], s=sensor_size, facecolors=sensor_color,
            edgecolors=sensor_edge_color, linewidths=_SENSOR_EDGE_WIDTH,
            zorder=_Z_SENSORS)
    if show_labels:
        fontsize = (_LABEL_FONT_FRAC * mpl.rcParams["font.size"]
                    if label_fontsize is None else label_fontsize)
        handles["labels"] = [
            ax.text(x, y - _LABEL_DROP, lab, ha="center", va="top",
                    fontsize=fontsize, color=label_color, zorder=_Z_LABELS)
            for (x, y), lab in zip(pos, labels)]

    # View box around the head, then blank the axes (a head, not a chart).
    extent = radius + _VIEW_MARGIN
    if len(pos):
        extent = max(
            extent, float(np.linalg.norm(pos, axis=1).max()) + _VIEW_MARGIN)
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent + _NOSE_HEADROOM)
    _bare_axes(ax)
    return handles


def plot_scalp(channel_set, ax, *,
               radius: float = 1.0, line_width: float = 2.0,
               head_color: str | None = None, skin_color: str | None = None,
               show_sensors: bool = True, show_labels: bool = False,
               sensor_color: str | None = None, sensor_size: float = 14.0,
               sensor_edge_color: str | None = None,
               label_color: str | None = None,
               label_fontsize: float | None = None) -> tuple[Axes, dict]:
    """Draw a 2-D head diagram with the channel set's sensor positions.

    Every channel in ``channel_set`` is plotted; select the channels you want
    (e.g. ``channel_set.pick(types="EEG")``) *before* calling.

    Parameters
    ----------
    channel_set
        ``ChannelSet`` (or compatible) providing ``get_positions`` / ``uids``.
    ax
        Axes to draw into (**required**).
    radius
        Head radius; the montage equator maps to it.
    line_width
        Outline width for head, nose and ears.
    head_color, skin_color
        Outline color and fill color. ``head_color=None`` (default) uses the
        active medusa-style theme foreground (``plot_fg``); ``skin_color=None``
        means no fill.
    show_sensors, show_labels
        Draw sensor markers and channel labels.
    sensor_color, sensor_size, sensor_edge_color
        Sensor marker fill color/size and the halo (edge) color. ``None`` (default)
        uses the active theme foreground for the marker and the theme background
        (``plot_bg``) for the halo, so markers read cleanly on any background.
    label_color, label_fontsize
        Channel-label color and size. ``label_fontsize=None`` (default) scales the
        label with the active style's ``font.size``.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes drawn into.
    artists : dict
        Named head artists (``"head-line"``, ``"sensors"``, ``"labels"``, ...).

    Examples
    --------
    >>> ax, artists = plot_scalp(channel_set, ax)  # doctest: +SKIP
    """
    pos, labels = _scalp_positions(channel_set)
    artists = _draw_head_artists(
        ax, pos, labels, radius=radius, line_width=line_width,
        head_color=head_color, skin_color=skin_color, show_sensors=show_sensors,
        show_labels=show_labels, sensor_color=sensor_color,
        sensor_size=sensor_size, sensor_edge_color=sensor_edge_color,
        label_color=label_color, label_fontsize=label_fontsize)
    return ax, artists
