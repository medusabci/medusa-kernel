"""Qt-free matplotlib builders for the ERP viewer (reused by live + export + GIF).

These functions/classes turn an :class:`~medusa.widgets.erp_viewer.analysis.ERPAnalyzer`
into figures, composing the reusable :mod:`medusa.plots` engines
(:func:`~medusa.plots.plot_erp`, :func:`~medusa.plots.plot_erp_grid`,
:class:`~medusa.plots.TopographicPlot`) — no plotting logic is reimplemented here.
Because there is **no Qt**, the exact same code path renders the on-screen canvas,
the exported publication figures and the animated-GIF frames:

* :func:`render_split` / :func:`render_mean` — the two temporal views;
* :class:`SpatialView` — a stateful topography + time-cursor panel whose
  :meth:`~SpatialView.set_time` is a cheap per-frame update (drives the slider);
* :func:`render_topography_gif` — the evolving topography as a looping GIF.
"""

from pathlib import Path

import medusa_style
import numpy as np
from matplotlib.figure import Figure

from medusa.plots.erp import plot_erp, plot_erp_channels, plot_erp_grid
from medusa.plots.scalp_topography import TopographicPlot

__all__ = ["render_split", "render_mean", "SpatialView",
           "render_topography_still", "render_topography_gif"]


def _erp_options(analyzer, error, line_width, band_alpha, onset, show_onset,
                 amplitude_limits):
    """Common per-panel :func:`plot_erp` keyword arguments (segments pre-processed).

    Baseline/smoothing are applied by the analyzer (single source, cached), so the
    plot functions receive already-processed segments and skip them here.
    """
    return dict(
        times=analyzer.times, error=error, baseline=None, smooth=None,
        show_onset=show_onset, onset=onset, line_width=line_width,
        band_alpha=band_alpha, amplitude_limits=amplitude_limits)


def render_split(fig: Figure, analyzer, idxs, *, error="ci95", share_y=True,
                 show_onset=True, onset=0.0, line_width=1.5, band_alpha=0.25,
                 color=None, amplitude_limits=None,
                 amplitude_unit="µV", time_unit="s") -> Figure:
    """Split view: one ERP panel per selected channel, on a shared-axis grid."""
    idxs = list(idxs)
    if not idxs:
        raise ValueError("render_split needs at least one channel index.")
    seg = analyzer.processed_segments(idxs)          # (n_seg, n_samp, n_sel)
    labels = [analyzer.cha_labels[i] for i in idxs]
    plot_erp_grid(
        seg, fig=fig, cha_labels=labels, times=analyzer.times, error=error,
        share_y=share_y, show_onset=show_onset, onset=onset, color=color,
        line_width=line_width, band_alpha=band_alpha,
        amplitude_limits=amplitude_limits, amplitude_unit=amplitude_unit,
        time_unit=time_unit)
    return fig


def render_mean(fig: Figure, analyzer, idxs, *, error="ci95", style="band",
                traces_alpha=0.45, traces_width=0.6, show_onset=True, onset=0.0,
                line_width=1.5, band_alpha=0.25, color=None, amplitude_limits=None,
                amplitude_unit="µV", time_unit="s") -> Figure:
    """Mean view: a single ERP averaged over segments then the selected channels.

    ``style="band"`` shades an error band around the mean (across trials);
    ``style="traces"`` overlays each selected channel's ERP thin + translucent
    under the thick across-channel mean (no band).
    """
    idxs = list(idxs)
    if not idxs:
        raise ValueError("render_mean needs at least one channel index.")
    fig.clear()
    ax = fig.subplots()
    n = len(idxs)
    label = (f"mean of {n} channels" if n > 1 else analyzer.cha_labels[idxs[0]])
    if style == "traces":
        # Channel-overlay: per-channel ERPs thin + translucent, grand mean thick.
        plot_erp_channels(
            analyzer.processed_segments(idxs), ax, times=analyzer.times,
            channel_alpha=traces_alpha, channel_width=traces_width,
            mean_width=max(line_width, 2.0), show_onset=show_onset, onset=onset,
            color=color, label=label, amplitude_limits=amplitude_limits,
            amplitude_unit=amplitude_unit, time_unit=time_unit)
    else:
        plot_erp(analyzer.channel_mean_segments(idxs),
                 ax, **_erp_options(analyzer, error, line_width, band_alpha,
                                    onset, show_onset, amplitude_limits),
                 color=color, label=label, amplitude_unit=amplitude_unit,
                 time_unit=time_unit)
    ax.set_title(f"Mean ERP ({n} channel{'s' if n != 1 else ''})")
    return fig


class SpatialView:
    """Topography + a synchronized ERP panel, updated cheaply per frame.

    Draws the head **once** (via :class:`~medusa.plots.TopographicPlot`); below it,
    an ERP of the selected channels — either a *channel overlay*
    (:func:`~medusa.plots.plot_erp_channels`: grand mean thick + each channel thin)
    or a *shaded summary* (:func:`~medusa.plots.plot_erp`: mean ± error band),
    chosen by ``erp_style`` — with a movable cursor marking the topography's time.
    :meth:`set_time` re-points only the value layer and the cursor, so dragging the
    slider is a light in-place update (no head redraw). Structural changes
    (interpolation, colormap, limits, selection, ERP style) go through
    :meth:`rebuild`; the ERP style/onset attributes are read there, so set them on
    the instance before rebuilding.

    ``show_erp=False`` drops the ERP panel entirely (topography only) — used by the
    export path when the user does not want the ERP included.
    """

    def __init__(self, fig: Figure, analyzer, idxs, *, interpolate=True,
                 contour=True, cmap=None, clim=None, interp_resolution=200,
                 show_onset=True, onset=0.0, amplitude_unit="µV", time_unit="s",
                 show_erp=True, erp_style="traces", error="ci95",
                 band_alpha=0.25, line_width=1.5, traces_alpha=0.45,
                 traces_width=0.6) -> None:
        self.fig = fig
        self.analyzer = analyzer
        self.amplitude_unit = amplitude_unit
        self.time_unit = time_unit
        self._show_onset = show_onset
        self._onset = onset
        self._show_erp = show_erp
        self._erp_style = erp_style
        self._error = error
        self._band_alpha = band_alpha
        self._line_width = line_width
        self._traces_alpha = traces_alpha
        self._traces_width = traces_width
        self.time_ax = None
        self.cursor = None
        self.rebuild(idxs, interpolate=interpolate, contour=contour, cmap=cmap,
                     clim=clim, interp_resolution=interp_resolution)

    def rebuild(self, idxs, *, interpolate=True, contour=True, cmap=None,
                clim=None, interp_resolution=200) -> None:
        """(Re)build the head (+ ERP panel) for the selected channels ``idxs``."""
        analyzer = self.analyzer
        self._idxs = list(idxs)
        self.fig.clear()
        if self._show_erp:
            gs = self.fig.add_gridspec(2, 1, height_ratios=[3.0, 1.0], hspace=0.32)
            self.topo_ax = self.fig.add_subplot(gs[0])
            self.time_ax = self.fig.add_subplot(gs[1])
        else:  # topography only (export without the ERP)
            self.topo_ax = self.fig.add_subplot(111)
            self.time_ax = None
            self.cursor = None

        cmap = cmap if cmap is not None else medusa_style.mpl.diverging_cmap()
        self.clim = tuple(clim) if clim is not None else analyzer.symmetric_clim()
        # The topography reflects the CURRENT channel selection: only the selected
        # sensors are shown and interpolated among. colorbar=False so the head axis
        # spans the full cell (equal-aspect centers it) and we add our own colorbar
        # in an inset just OUTSIDE the head — a built-in one would push it off-centre.
        self.topo = TopographicPlot(
            analyzer.topo_channel_set(self._idxs), self.topo_ax,
            interpolate=interpolate, contour=contour, cmap=cmap, clim=self.clim,
            interp_resolution=interp_resolution, colorbar=False)

        if self._show_erp:
            self._draw_erp(self._idxs)
        self._t_idx = 0
        self.set_time(0)   # creates the mappable
        cax = self.topo_ax.inset_axes([1.03, 0.15, 0.045, 0.7])
        self.fig.colorbar(self.topo.mappable, cax=cax,
                          label=f"amplitude ({self.amplitude_unit})")

    def _draw_erp(self, idxs) -> None:
        """The selected channels' ERP (overlay or shaded) + a moving time cursor."""
        ax = self.time_ax
        accent = medusa_style.current_theme().accent_primary
        if self._erp_style == "band":
            # Shaded summary: mean over the selected channels ± error band.
            plot_erp(
                self.analyzer.channel_mean_segments(idxs), ax,
                times=self.analyzer.times, error=self._error, baseline=None,
                smooth=None, show_onset=self._show_onset, onset=self._onset,
                line_width=self._line_width, band_alpha=self._band_alpha,
                legend=False, label_axes=True,
                amplitude_unit=self.amplitude_unit, time_unit=self.time_unit)
        else:
            # Channel overlay: each channel thin + translucent, grand mean thick.
            plot_erp_channels(
                self.analyzer.processed_segments(idxs), ax,
                times=self.analyzer.times, channel_alpha=self._traces_alpha,
                channel_width=self._traces_width,
                mean_width=max(self._line_width, 2.0), show_onset=self._show_onset,
                onset=self._onset, legend=False, label_axes=True,
                amplitude_unit=self.amplitude_unit, time_unit=self.time_unit)
        # The cursor is the one artist set_time moves; accent color to stand out.
        self.cursor = ax.axvline(self.analyzer.times[0], color=accent, lw=1.6,
                                 zorder=4)

    def set_time(self, t_idx: int):
        """Move the topography (+ cursor) to sample ``t_idx`` (cheap in-place update)."""
        self._t_idx = int(np.clip(t_idx, 0, self.analyzer.n_samples - 1))
        t = self.analyzer.times[self._t_idx]
        updated = self.topo.set_data(
            self.analyzer.topography_values(self._t_idx, self._idxs))
        if self.cursor is not None:
            self.cursor.set_xdata([t, t])
        self.topo_ax.set_title(f"t = {t:.4g} {self.time_unit}")
        return updated

    @property
    def t_idx(self) -> int:
        return self._t_idx


def _spatial_view(fig, analyzer, idxs, *, include_erp, interpolate, contour,
                  cmap, clim, interp_resolution, erp_style, error, band_alpha,
                  line_width, traces_alpha, traces_width, show_onset, onset,
                  amplitude_unit, time_unit):
    """Build a :class:`SpatialView` on ``fig`` (all channels when ``idxs`` None)."""
    idxs = list(range(analyzer.n_channels)) if idxs is None else list(idxs)
    return SpatialView(
        fig, analyzer, idxs, interpolate=interpolate, contour=contour, cmap=cmap,
        clim=clim, interp_resolution=interp_resolution, show_onset=show_onset,
        onset=onset, amplitude_unit=amplitude_unit, time_unit=time_unit,
        show_erp=include_erp, erp_style=erp_style, error=error,
        band_alpha=band_alpha, line_width=line_width, traces_alpha=traces_alpha,
        traces_width=traces_width)


def render_topography_still(
        fig: Figure, analyzer, idxs, t_idx, *, include_erp=True, interpolate=True,
        contour=True, cmap=None, clim=None, interp_resolution=200,
        erp_style="traces", error="ci95", band_alpha=0.25, line_width=1.5,
        traces_alpha=0.45, traces_width=0.6, show_onset=True, onset=0.0,
        amplitude_unit="µV", time_unit="s") -> Figure:
    """One topography frame at sample ``t_idx``, optionally with the ERP panel.

    Backs both the still-topography export and the GIF-dialog preview: builds a
    :class:`SpatialView` on ``fig`` and points it at ``t_idx``. ``include_erp``
    toggles the ERP panel beneath the head (``erp_style`` picks overlay vs shaded).
    """
    view = _spatial_view(
        fig, analyzer, idxs, include_erp=include_erp, interpolate=interpolate,
        contour=contour, cmap=cmap, clim=clim,
        interp_resolution=interp_resolution, erp_style=erp_style, error=error,
        band_alpha=band_alpha, line_width=line_width, traces_alpha=traces_alpha,
        traces_width=traces_width, show_onset=show_onset, onset=onset,
        amplitude_unit=amplitude_unit, time_unit=time_unit)
    view.set_time(t_idx)
    return fig


def render_topography_gif(
        path, analyzer, *, idxs=None, t_indices=None, t_start=None, t_stop=None,
        step=1, interpolate=True, contour=True, cmap=None, clim=None,
        interp_resolution=120, figsize=(4.6, 5.4), fps=10, dpi=100, loop=0,
        include_erp=True, erp_style="traces", error="ci95", band_alpha=0.25,
        line_width=1.5, traces_alpha=0.45, traces_width=0.6, show_onset=True,
        onset=0.0, amplitude_unit="µV", time_unit="s"):
    """Write the evolving topography over time as an animated GIF.

    Renders through :class:`SpatialView`, so a frame looks exactly like the live
    spatial view: the head plus (when ``include_erp``) the ERP panel with a moving
    time cursor. The ERP is drawn once and only the head values + cursor move per
    frame.

    Parameters
    ----------
    path
        Output ``.gif`` path (parent directories are created).
    analyzer
        The :class:`~medusa.widgets.erp_viewer.analysis.ERPAnalyzer`.
    idxs
        Channel indices to include (all channels when ``None``).
    t_indices
        Explicit sample indices to render as frames. When ``None`` they are built
        from ``t_start``/``t_stop`` (times, defaulting to the full epoch) taken
        every ``step`` samples.
    step
        Sample stride between frames when ``t_indices`` is ``None``.
    interpolate, contour, cmap, clim, interp_resolution
        Topography styling; ``clim`` defaults to the analyzer's symmetric range so
        colors are comparable across frames. A lower ``interp_resolution`` renders
        faster.
    figsize, dpi
        Frame size (inches) and resolution.
    fps
        Playback frames per second.
    include_erp, erp_style, error, band_alpha, line_width, traces_alpha, traces_width, show_onset, onset
        Whether to draw the ERP panel and how (see :class:`SpatialView`).
    loop
        GIF looping: ``0`` loops forever, ``None`` plays exactly once, ``n``
        repeats ``n`` times after the first play (so a count of ``1`` actually
        plays twice — use ``None`` for a single play).

    Returns
    -------
    pathlib.Path
        The written path.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    if t_indices is None:
        lo = 0 if t_start is None else analyzer.time_index(t_start)
        hi = analyzer.n_samples - 1 if t_stop is None else analyzer.time_index(t_stop)
        lo, hi = sorted((lo, hi))
        t_indices = list(range(lo, hi + 1, max(1, int(step))))
    t_indices = list(t_indices)
    if not t_indices:
        raise ValueError("no frames to render (empty time range).")

    fig = Figure(figsize=figsize)
    FigureCanvasAgg(fig)  # offscreen canvas so the writer can rasterize frames
    view = _spatial_view(
        fig, analyzer, idxs, include_erp=include_erp, interpolate=interpolate,
        contour=contour, cmap=cmap, clim=clim,
        interp_resolution=interp_resolution, erp_style=erp_style, error=error,
        band_alpha=band_alpha, line_width=line_width, traces_alpha=traces_alpha,
        traces_width=traces_width, show_onset=show_onset, onset=onset,
        amplitude_unit=amplitude_unit, time_unit=time_unit)

    writer = _loop_pillow_writer(fps=fps, loop=loop)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with writer.saving(fig, str(path), dpi):
        for ti in t_indices:
            view.set_time(ti)
            writer.grab_frame(facecolor=fig.get_facecolor())
    return path


def _loop_pillow_writer(*, fps, loop=0):
    """A :class:`~matplotlib.animation.PillowWriter` that honors a ``loop`` count.

    Matplotlib's ``PillowWriter.finish`` hardcodes ``loop=0`` (loop forever); this
    subclass threads the requested count through instead. Built lazily so the
    animation backend is only imported when a GIF is actually exported.
    """
    from matplotlib.animation import PillowWriter

    class _LoopPillowWriter(PillowWriter):
        def finish(self):
            # loop=None -> omit the loop extension entirely (plays exactly once);
            # 0 -> loop forever; n -> repeat n times after the first play.
            extra = {} if loop is None else {"loop": loop}
            self._frames[0].save(
                self.outfile, save_all=True, append_images=self._frames[1:],
                duration=int(1000 / self.fps), **extra)

    return _LoopPillowWriter(fps=fps)
