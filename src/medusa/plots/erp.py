"""Event-related potential (ERP) plots: segment-averaged waveform ± error band.

An ERP is the **mean across the segment (trial) axis** of an epoched signal
``(n_segments, n_samples[, n_series])``, with an optional error / confidence
band. These two functions are the reusable ERP primitives that back the
``medusa.widgets.erp_viewer.ERPViewer`` GUI and stand alone for notebooks / papers:

* :func:`plot_erp` — one ERP on a single axes (mean ± band), with per-trial
  baseline correction, a stimulus-onset marker and optional temporal smoothing.
* :func:`plot_erp_grid` — one ERP panel per channel, arranged as a near-square
  grid via :func:`~medusa.plots.optimal_grid` with shared axes.

Both build on :func:`~medusa.plots.plot_shaded_line` (the generic mean±band
engine) rather than reimplementing the summary statistics, and consume the
``medusa_style`` visual identity like the rest of :mod:`medusa.plots`.
"""

import medusa_style
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike, NDArray

from medusa.plots.shaded_line import plot_shaded_line
from medusa.plots.utils import optimal_grid

__all__ = ["plot_erp", "plot_erp_grid", "plot_erp_channels"]

#: Stimulus-onset marker chrome (a faint dashed annotation, not a data line).
_ONSET_LS = "--"
_ONSET_LW = 1.0
_ONSET_ALPHA = 0.6


def _resolve_times(n_samples: int, fs: float | None,
                   times: ArrayLike | None) -> NDArray:
    """Per-sample time vector from an explicit ``times``, epoch limits, or ``fs``.

    ``times`` may be the full ``(n_samples,)`` vector or a 2-tuple ``(t0, t1)``
    of epoch limits (linearly spaced). With neither ``times`` nor ``fs`` the axis
    is the sample index.
    """
    if times is not None:
        t = np.asarray(times, dtype=float).ravel()
        if t.shape[0] == 2 and n_samples != 2:  # epoch limits (t_start, t_end)
            return np.linspace(t[0], t[1], n_samples)
        if t.shape[0] != n_samples:
            raise ValueError(
                f"times has length {t.shape[0]}; expected {n_samples} (a full "
                f"time vector) or 2 (epoch limits t_start, t_end).")
        return t
    if fs is not None:
        return np.arange(n_samples) / float(fs)
    return np.arange(n_samples, dtype=float)


def _baseline_correct(data: NDArray, times: NDArray,
                      baseline: tuple[float, float]) -> NDArray:
    """Subtract each segment's mean over the baseline window (per channel/series).

    ``data`` is ``(n_segments, n_samples[, n_series])``; the window is a
    ``(t_lo, t_hi)`` interval in the units of ``times``.
    """
    lo, hi = float(baseline[0]), float(baseline[1])
    mask = (times >= lo) & (times <= hi)
    if not mask.any():
        raise ValueError(
            f"baseline window {baseline} selects no samples in the epoch "
            f"[{times[0]:.4g}, {times[-1]:.4g}].")
    return data - data[:, mask].mean(axis=1, keepdims=True)


def _smooth_time(data: NDArray, window: int) -> NDArray:
    """Moving-average smooth along the sample axis (axis 1); ``<2`` is a no-op."""
    w = int(window)
    if w < 2:
        return data
    from scipy.ndimage import uniform_filter1d  # lazy: only when smoothing
    return uniform_filter1d(data, size=w, axis=1, mode="nearest")


def plot_erp(
        data, ax, *,
        times: ArrayLike | None = None, fs: float | None = None,
        error: str | None = "ci95",
        baseline: tuple[float, float] | None = None,
        smooth: int | None = None,
        color: str | list | None = None, label: str | None = None,
        series_labels: ArrayLike | None = None,
        show_onset: bool = True, onset: float = 0.0,
        line_width: float = 1.5, band_alpha: float = 0.25,
        amplitude_limits: tuple[float, float] | None = None,
        legend: bool = True, label_axes: bool = True,
        amplitude_unit: str = "µV", time_unit: str = "s") -> tuple[Axes, dict]:
    """Plot one ERP: the mean over segments with an optional error band.

    Parameters
    ----------
    data
        ``(n_segments, n_samples)`` (one ERP) or ``(n_segments, n_samples,
        n_series)`` (overlaid conditions); the mean/band are taken over the
        leading **segment** axis.
    ax
        Axes to draw into (**required**).
    times, fs
        Time base: an explicit ``times`` vector (or ``(t_start, t_end)`` epoch
        limits), or a sampling rate ``fs`` (Hz). The sample index when neither is
        given.
    error
        Band statistic across segments: ``"none"``/``None`` (no band), ``"std"``,
        ``"sem"``, or ``"ci<level>"`` (Student-t CI, e.g. ``"ci95"``).
    baseline
        ``(t_lo, t_hi)`` window (in ``times`` units) subtracted per segment before
        averaging; ``None`` skips baseline correction.
    smooth
        Moving-average window in samples applied along time; ``None`` / ``<2``
        skips smoothing.
    color, label, series_labels
        Trace color(s) and legend label(s); see
        :func:`~medusa.plots.plot_shaded_line`.
    show_onset, onset
        Draw a dashed vertical marker at the stimulus onset time ``onset``
        (default ``0``).
    line_width, band_alpha
        Mean-trace width and band opacity.
    amplitude_limits
        Fixed ``(y_lo, y_hi)`` amplitude range; auto-scaled when ``None``.
    legend
        Draw a legend when a label is present.
    label_axes
        Set the ``time``/``amplitude`` axis labels (disable for grid panels that
        share one figure-level label).
    amplitude_unit, time_unit
        Axis-label units.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes drawn into.
    artists : dict
        ``plot_shaded_line`` artists (``"mean"``, ``"band"``) plus ``"onset"``
        (the onset line, when ``show_onset``).

    Examples
    --------
    >>> ax, artists = plot_erp(epochs, ax, fs=256, error="sem")  # doctest: +SKIP
    """
    data = np.asarray(data, dtype=float)
    if data.ndim not in (2, 3):
        raise ValueError(
            "data must be (n_segments, n_samples) or (n_segments, n_samples, "
            f"n_series); got shape {data.shape}.")
    t = _resolve_times(data.shape[1], fs, times)
    if baseline is not None:
        data = _baseline_correct(data, t, baseline)
    if smooth is not None:
        data = _smooth_time(data, smooth)

    ax, artists = plot_shaded_line(
        data, ax, x=t, error=error, color=color, label=label,
        series_labels=series_labels, legend=legend, line_width=line_width,
        band_alpha=band_alpha)

    if show_onset:
        artists["onset"] = ax.axvline(
            onset, color=medusa_style.current_theme().plot_fg,
            ls=_ONSET_LS, lw=_ONSET_LW, alpha=_ONSET_ALPHA, zorder=1.5)
    if amplitude_limits is not None:
        ax.set_ylim(*amplitude_limits)
    if label_axes:
        ax.set_xlabel(f"time ({time_unit})")
        ax.set_ylabel(f"amplitude ({amplitude_unit})")
    return ax, artists


def plot_erp_grid(
        data, fig: Figure | None = None, *,
        cha_labels: ArrayLike | None = None,
        times: ArrayLike | None = None, fs: float | None = None,
        error: str | None = "ci95",
        baseline: tuple[float, float] | None = None,
        smooth: int | None = None,
        share_x: bool = True, share_y: bool = True,
        show_onset: bool = True, onset: float = 0.0,
        color: str | list | None = None,
        line_width: float = 1.5, band_alpha: float = 0.25,
        ncols: int | None = None, titles: bool = True,
        amplitude_limits: tuple[float, float] | None = None,
        amplitude_unit: str = "µV", time_unit: str = "s",
        ) -> tuple[Figure, NDArray, list]:
    """Plot one ERP panel per channel on a near-square shared-axis grid.

    Parameters
    ----------
    data
        ``(n_segments, n_samples, n_channels)``; one panel per channel (the last
        axis), each the mean over segments.
    fig
        Figure to draw into (**cleared** first); a new one is created when
        ``None``. Pass the host canvas's figure to redraw a GUI panel in place.
    cha_labels
        Per-channel titles (``Ch1..`` when ``None``).
    times, fs, error, baseline, smooth, show_onset, onset, color, line_width, band_alpha, amplitude_limits
        Per-panel ERP options; see :func:`plot_erp`.
    share_x, share_y
        Share the time / amplitude axis across panels. Shared axes give a fair
        cross-channel comparison and auto-scale to a common range.
    ncols
        Force the column count (rows follow); :func:`~medusa.plots.optimal_grid`
        chooses a near-square layout when ``None``.
    titles
        Show the channel label as each panel's title.
    amplitude_unit, time_unit
        Units for the shared figure-level axis labels.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure drawn into.
    axes : numpy.ndarray
        The ``(rows, cols)`` axes grid (unused cells are hidden).
    artists : list of dict
        Per-channel :func:`plot_erp` artist dicts (channel order).

    Examples
    --------
    >>> fig, axes, artists = plot_erp_grid(epochs, cha_labels=cs.labels, fs=256)  # doctest: +SKIP
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 3:
        raise ValueError(
            "data must be (n_segments, n_samples, n_channels); got shape "
            f"{data.shape}.")
    n_ch = data.shape[2]
    labels = ([f"Ch{i + 1}" for i in range(n_ch)] if cha_labels is None
              else [str(c) for c in cha_labels])
    if len(labels) != n_ch:
        raise ValueError(
            f"cha_labels has length {len(labels)}; expected {n_ch} channels.")

    if ncols is not None:
        cols = max(1, int(ncols))
        rows = int(np.ceil(n_ch / cols))
    else:
        rows, cols = optimal_grid(n_ch)

    if fig is None:
        fig = Figure(figsize=(min(3.2 * cols, 18.0), min(2.4 * rows, 12.0)))
    else:
        fig.clear()
    # Constrained layout reflows the panels so per-panel titles never collide
    # with the row above and the shared figure labels always fit.
    fig.set_layout_engine("constrained")
    axes = fig.subplots(rows, cols, sharex=share_x, sharey=share_y,
                        squeeze=False)
    flat = axes.ravel()

    artists = []
    for k in range(n_ch):
        _, art = plot_erp(
            data[:, :, k], flat[k], times=times, fs=fs, error=error,
            baseline=baseline, smooth=smooth, color=color, show_onset=show_onset,
            onset=onset, line_width=line_width, band_alpha=band_alpha,
            amplitude_limits=amplitude_limits, legend=False, label_axes=False)
        if titles:
            flat[k].set_title(labels[k])
        artists.append(art)
    for j in range(n_ch, len(flat)):  # blank the unused cells
        flat[j].set_visible(False)

    # With sharex, matplotlib hides x tick labels on every non-last-row axes; when
    # a column's real last-row cell is a blanked one (n_ch not a multiple of cols),
    # its bottom-most VISIBLE panel would lose its time labels. Re-enable them on
    # the last filled panel of each column.
    if share_x:
        for c in range(cols):
            filled = [r for r in range(rows) if r * cols + c < n_ch]
            if filled and max(filled) < rows - 1:
                axes[max(filled), c].tick_params(axis="x", labelbottom=True)

    # One shared figure-level label instead of repeating it on every panel.
    fig.supxlabel(f"time ({time_unit})")
    fig.supylabel(f"amplitude ({amplitude_unit})")
    return fig, axes, artists


def plot_erp_channels(
        data, ax, *,
        times: ArrayLike | None = None, fs: float | None = None,
        baseline: tuple[float, float] | None = None, smooth: int | None = None,
        channels: ArrayLike | None = None,
        color: str | None = None, channel_color: str | None = None,
        mean_width: float = 2.0, channel_width: float = 0.6,
        channel_alpha: float = 0.4,
        show_onset: bool = True, onset: float = 0.0,
        amplitude_limits: tuple[float, float] | None = None,
        label: str | None = None, legend: bool = True, label_axes: bool = True,
        amplitude_unit: str = "µV", time_unit: str = "s") -> tuple[Axes, dict]:
    """Plot a grand-average ERP over one axes with the per-channel ERPs behind it.

    Multichannel counterpart to :func:`plot_erp_grid` that collapses to a **single**
    axes instead of a grid: the mean ERP across segments and channels is drawn
    thick, with each channel's own segment-averaged ERP drawn thin and translucent
    underneath — a compact "channel overlay" summary of where and how consistently
    a component appears across the montage.

    Parameters
    ----------
    data
        ``(n_segments, n_samples, n_channels)``; averaged over the segment axis per
        channel, then over channels for the thick mean.
    ax
        Axes to draw into (**required**).
    times, fs, baseline, smooth, show_onset, onset, amplitude_limits
        Time base and per-ERP options; see :func:`plot_erp`.
    channels
        Channel indices to draw as thin background lines (all channels when
        ``None``). The thick mean is always over **all** channels in ``data``.
    color
        Mean-trace color (the brand's first categorical color when ``None``).
    channel_color
        Thin per-channel line color (defaults to ``color``).
    mean_width, channel_width, channel_alpha
        Widths of the thick mean and thin channel lines, and the channel opacity.
    label
        Legend label for the mean trace.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes drawn into.
    artists : dict
        ``"mean"`` (the thick ``Line2D``), ``"channels"`` (the thin ``Line2D``\\ s)
        and ``"onset"`` (when ``show_onset``).

    Examples
    --------
    >>> ax, artists = plot_erp_channels(epochs, ax, fs=256)  # doctest: +SKIP
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 3:
        raise ValueError(
            "data must be (n_segments, n_samples, n_channels); got shape "
            f"{data.shape}.")
    t = _resolve_times(data.shape[1], fs, times)
    if baseline is not None:
        data = _baseline_correct(data, t, baseline)
    if smooth is not None:
        data = _smooth_time(data, smooth)

    erp = np.nanmean(data, axis=0)              # (n_samples, n_channels)
    mean = np.nanmean(erp, axis=1)              # grand mean over channels
    n_ch = erp.shape[1]
    sel = range(n_ch) if channels is None else [int(c) for c in channels]

    color = color if color is not None else medusa_style.categorical_color(0)
    ch_color = channel_color if channel_color is not None else color

    channel_lines = [
        ax.plot(t, erp[:, i], color=ch_color, lw=channel_width,
                alpha=channel_alpha, zorder=1)[0]
        for i in sel]
    (mean_line,) = ax.plot(t, mean, color=color, lw=mean_width, label=label,
                           zorder=3)
    artists = {"mean": mean_line, "channels": channel_lines}

    ax.margins(x=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if show_onset:
        artists["onset"] = ax.axvline(
            onset, color=medusa_style.current_theme().plot_fg, ls=_ONSET_LS,
            lw=_ONSET_LW, alpha=_ONSET_ALPHA, zorder=1.5)
    if amplitude_limits is not None:
        ax.set_ylim(*amplitude_limits)
    if label_axes:
        ax.set_xlabel(f"time ({time_unit})")
        ax.set_ylabel(f"amplitude ({amplitude_unit})")
    if legend and label is not None:
        ax.legend(loc="upper right", framealpha=0.9)
    return ax, artists
