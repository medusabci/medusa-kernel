"""Mean trace with a shaded error/distribution band, over a generic x-axis.

x-agnostic: the same primitive serves an ERP over time or a mean PSD over
frequency. Input is ``(n_observations, n_points[, n_series])``; the band is
summarized across the observation axis.
"""

import medusa_style
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import ArrayLike, NDArray

from medusa.plots.utils import _resolve_colors

__all__ = ["plot_shaded_line"]

_ERROR_HELP = ("error must be 'std', 'sem', or 'ci<level>' (e.g. 'ci95', "
               "'ci75', 'ci50')")


def _parse_ci_level(error: str) -> float:
    """Confidence level (percent) from a ``'ci<level>'`` spec."""
    try:
        level = float(error[2:])
    except (ValueError, IndexError):
        raise ValueError(f"{_ERROR_HELP}; got {error!r}.") from None
    if not 0.0 < level < 100.0:
        raise ValueError(
            f"CI level must be in (0, 100); got {level} from {error!r}.")
    return level


def _summary(data: NDArray, error: str) -> tuple[NDArray, NDArray]:
    """Mean and ± band half-width across the observation axis (axis 0).

    ``data`` has shape ``(n_observations, n_points, n_series)``. ``error`` is
    ``"std"``, ``"sem"`` or ``"ci<level>"`` (two-sided Student-t CI). Fewer than
    two observations give a zero-width band.
    """
    e = error.lower()
    is_ci = e.startswith("ci")
    if not is_ci and e not in ("std", "sem"):
        raise ValueError(f"{_ERROR_HELP}; got {error!r}.")
    level = _parse_ci_level(e) if is_ci else None  # validates the level early

    n = data.shape[0]
    mean = np.nanmean(data, axis=0)
    if n < 2:
        return mean, np.zeros_like(mean)
    std = np.nanstd(data, axis=0, ddof=1)  # unbiased: sem/CI are mean inference
    if e == "std":
        return mean, std
    sem = std / np.sqrt(n)
    if e == "sem":
        return mean, sem
    from scipy.stats import t  # lazy: only when a CI is requested
    return mean, float(t.ppf(0.5 + level / 200.0, df=n - 1)) * sem


def plot_shaded_line(
        data, ax, *,
        x: ArrayLike | None = None,
        error: str = "ci95",
        series_labels: ArrayLike | None = None,
        color: str | list | None = None, label: str | None = None,
        legend: bool = True, line_width: float = 1.5,
        band_alpha: float = 0.25,
        **kwargs) -> tuple[Axes, dict]:
    """Plot a per-point mean with a shaded error/distribution band.

    Parameters
    ----------
    data
        ``(n_observations, n_points)`` or ``(n_observations, n_points, n_series)``;
        the band is summarized across the leading observation axis.
    ax
        Axes to draw into (**required**).
    x
        x-coordinates (length ``n_points``), e.g. times or frequencies; the point
        index when ``None``. Set ``ax.set_xlabel`` for the meaning.
    error
        Band half-width: ``"std"``, ``"sem"``, or ``"ci<level>"`` — a two-sided
        Student-t confidence interval at any level (``"ci95"``, ``"ci67"``, ...).
    series_labels
        Per-series legend labels (length ``n_series``); unlabelled when ``None``.
    color
        A single color (all series) or one per series; the brand palette when
        ``None``. ``label`` names a single-series line for the legend.
    legend
        Draw a legend when any series is labelled.
    line_width, band_alpha
        Mean-trace width and band opacity.
    **kwargs
        Forwarded to the mean-trace ``ax.plot``.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes drawn into.
    artists : dict
        ``"mean"`` (one ``Line2D`` per series) and ``"band"`` (one
        ``PolyCollection`` per series).

    Examples
    --------
    >>> ax, artists = plot_shaded_line(epochs, ax, x=t, error="ci95")  # doctest: +SKIP

    Overlay two conditions:

    >>> plot_shaded_line(cond_a, ax, x=t, color="C0", label="A")  # doctest: +SKIP
    >>> plot_shaded_line(cond_b, ax, x=t, color="C1", label="B")  # doctest: +SKIP
    """
    data = np.asarray(data, dtype=float)
    if data.ndim == 2:
        data = data[:, :, None]  # (n_obs, n_pts) -> single series
    elif data.ndim != 3:
        raise ValueError(
            "data must be (n_observations, n_points) or "
            f"(n_observations, n_points, n_series); got shape {data.shape}.")
    n_pts, n_series = data.shape[1], data.shape[2]

    if x is None:
        x = np.arange(n_pts, dtype=float)
    else:
        x = np.asarray(x, dtype=float).ravel()
        if x.shape[0] != n_pts:
            raise ValueError(f"x has length {x.shape[0]}; expected {n_pts}.")

    mean, half = _summary(data, error)

    if series_labels is not None:
        labels = [str(s) for s in series_labels]
        if len(labels) != n_series:
            raise ValueError(
                f"series_labels has length {len(labels)}; expected {n_series}.")
    elif n_series == 1:
        labels = [label]
    else:
        labels = [None] * n_series

    colors = _resolve_colors(color, n_series)
    if colors is None:
        colors = [medusa_style.categorical_color(i) for i in range(n_series)]

    means, bands = [], []
    for i in range(n_series):
        (line,) = ax.plot(x, mean[:, i], color=colors[i], lw=line_width,
                          label=labels[i], zorder=2, **kwargs)
        band = ax.fill_between(x, mean[:, i] - half[:, i], mean[:, i] + half[:, i],
                               color=colors[i], alpha=band_alpha, linewidth=0,
                               zorder=1)
        means.append(line)
        bands.append(band)

    ax.margins(x=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if legend and any(lab is not None for lab in labels):
        # fontsize unset -> inherit the active style's legend.fontsize.
        ax.legend(loc="upper right", framealpha=0.9)
    return ax, {"mean": means, "band": bands}
