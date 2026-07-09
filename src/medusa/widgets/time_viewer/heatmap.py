"""Interactive Qt viewer for per-channel time-frequency heatmaps.

:class:`TimeHeatmapViewer` browses a per-channel **heatmap** (e.g. one spectrogram
or scalogram per channel) stacked as bands in one axes. It adds a *Colour* control
group — colormap, contrast, and **how many ``y_values`` (frequency) ticks each
band shows** — to the shared
:class:`~medusa.widgets.time_viewer.base._BaseTimeViewerWindow` chrome (visible
channels, time window, events, overview scrubber, export) and delegates drawing to
the Qt-free :class:`~medusa.plots.TimeHeatmapPlot` engine.

The per-band frequency axis defaults to 3 ticks (the band's lower edge, midpoint
and upper edge); the *Freq ticks/ch* control re-lays them live via
:meth:`~medusa.plots.TimeHeatmapPlot.set_y_ticks_per_channel`.

Example
-------
>>> from medusa.widgets.time_viewer import TimeHeatmapViewer     # doctest: +SKIP
>>> v = TimeHeatmapViewer(cha_labels=["Fz", "Cz", "Pz", "Oz"])  # doctest: +SKIP
>>> v.add_timeheatmap(power, y_values=freqs, times=t)  # (n_freqs, n_times, n_ch)  # doctest: +SKIP
>>> v.show()   # blocks until the window is closed              # doctest: +SKIP
"""

import sys

import medusa_style
import numpy as np
from PySide6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
)

from medusa.plots import TimeHeatmapPlot
from medusa.widgets.time_viewer.base import _BaseTimeViewerWindow, _styled

__all__ = ["TimeHeatmapViewer", "TimeHeatmapViewerWindow"]

#: Heatmap colormap menu — the medusa-style sequential default first, then
#: perceptually-sound alternatives.
_COLORMAPS = (medusa_style.mpl.SEQUENTIAL_CMAP_NAME,
              "inferno", "magma", "plasma", "cividis", "Greys")

#: Default / maximum per-channel ``y_values`` ticks offered by the control.
_DEFAULT_Y_TICKS = 3
_MAX_Y_TICKS = 12


class TimeHeatmapViewerWindow(_BaseTimeViewerWindow):
    """The heatmap viewer window (stacked per-channel bands + colour/contrast)."""

    _MODE = "heatmap"
    _TITLE = "Time Viewer — Heatmap"
    _EXPORT_NAME = "heatmap"

    def __init__(self, *, cha_labels=None, channels_visible=None,
                 reverse_channels=True, window=10.0, step=2.0,
                 zoom_factor=1.2, y_ticks_per_channel=_DEFAULT_Y_TICKS):
        # View-specific state, set before the base builds the panel.
        self._contrast = 1.0
        self._clim_data = None       # (min, max) for contrast anchoring
        self._hm_overview = None     # (times, median_image, cmap_name)
        self._y_ticks = int(y_ticks_per_channel)
        # Raw input of the shown heatmap, kept so the *Reverse order* toggle can
        # rebuild the bands with the opposite channel ordering.
        self._hm_input = None
        super().__init__(
            cha_labels=cha_labels, channels_visible=channels_visible,
            reverse_channels=reverse_channels, window=window, step=step,
            zoom_factor=zoom_factor)

    # -- view-specific control group --------------------------------------
    def _build_view_controls(self):
        self._color_group = QGroupBox("Colour")
        cgl = QVBoxLayout(self._color_group)
        crow = QHBoxLayout()
        crow.addWidget(QLabel("Map:"))
        self._cmap_combo = QComboBox()
        self._cmap_combo.addItems(_COLORMAPS)
        self._cmap_combo.currentTextChanged.connect(self._on_cmap)
        crow.addWidget(self._cmap_combo)
        cgl.addLayout(crow)
        contrast = QHBoxLayout()
        contrast.addWidget(QLabel("Contrast:"))
        contrast.addWidget(self._icon_button("zoom_out", "Less contrast",
                                             lambda: self._on_gain_step(-1)))
        contrast.addWidget(self._icon_button("zoom_in", "More contrast",
                                             lambda: self._on_gain_step(1)))
        cgl.addLayout(contrast)
        # How many per-band frequency ticks (low/mid/high by default).
        yrow = QHBoxLayout()
        yrow.addWidget(QLabel("Freq ticks/ch:"))
        self._yticks_spin = QSpinBox()
        self._yticks_spin.setRange(2, _MAX_Y_TICKS)
        self._yticks_spin.setValue(self._y_ticks)
        self._yticks_spin.setToolTip(
            "Number of frequency (y_values) ticks per channel band; the default "
            "3 marks the band's lower edge, midpoint and upper edge.")
        self._yticks_spin.valueChanged.connect(self._on_yticks_changed)
        yrow.addWidget(self._yticks_spin)
        cgl.addLayout(yrow)
        return [self._color_group]

    # -- public API -------------------------------------------------------
    def add_timeheatmap(self, data, *, y_values, fs=None, times=None,
                        cmap=None, clim=None, label=None, events=None,
                        event_hue=None):
        """Show a per-channel heatmap (e.g. spectrogram); one signal per viewer.

        ``data`` is ``(n_y, n_times, n_channels)`` or segmented
        ``(n_segments, n_y, n_times, n_channels)`` and ``y_values`` are the
        per-channel y-coordinates (e.g. frequencies). ``events`` is BIDS-aligned
        (an :class:`~medusa.core.data.events.Events` or an ``onset``/``duration``
        DataFrame with an optional ``event_hue`` column).

        Raises
        ------
        RuntimeError
            If a heatmap has already been shown (this view takes one signal).
        """
        if self._engine is not None:
            raise RuntimeError(
                "a signal is already shown; heatmap mode takes one signal.")
        data = np.asarray(data, dtype=float)
        y_values = np.asarray(y_values, dtype=float)
        n_ch = data.shape[-1]
        n_time = data.shape[1] if data.ndim == 3 else \
            data.shape[0] * data.shape[2]
        times = self._resolve_times(n_time, fs, times, 0.0)
        ordered = data if self._reverse else data[..., ::-1]
        self._clim_data = (float(np.nanmin(data)), float(np.nanmax(data))) \
            if clim is None else (float(clim[0]), float(clim[1]))

        with _styled():
            self._n_cha = n_ch
            self._init_channels(n_ch)
            cmap = cmap or _COLORMAPS[0]
            self._hm_input = {"data": data, "y_values": y_values, "times": times,
                              "cmap": cmap, "clim": clim, "label": label}
            self._engine = TimeHeatmapPlot(
                self._ax, cha_labels=self._engine_labels(), y_values=y_values,
                cmap=cmap, clim=self._clim_data, colorbar=True, legend=True,
                y_ticks_per_channel=self._y_ticks)
            self._engine.set_data(ordered, times=times)
            self._n_signals += 1
            self._signal_labels = [label]
            self._sync_combo(self._cmap_combo, cmap)
            # Cap the tick control at the number of available frequency rows.
            self._sync_spin(self._yticks_spin, self._y_ticks,
                            maximum=min(_MAX_Y_TICKS, max(2, y_values.size)))
            self._register_events(events, event_hue, 0.0)
            self._record_heatmap_overview(ordered, times, cmap)
            self._extend_extent(float(times[0]), float(times[-1]))
            self._after_add()
        return self

    # -- mode-specific hooks ----------------------------------------------
    def _apply_channel_ylim(self):
        if self._vis_ch is None or self._engine is None:
            return
        # Heatmap bands: channel i occupies slot [n-1-i, n-i); show s..e-1. The
        # per-band frequency ticks live on a secondary axis that tracks this.
        n = self._n_cha
        s, e = self._vis_ch
        self._ax.set_ylim(n - e, n - s)

    def _on_gain_step(self, direction):
        # For a heatmap the "gain" wheel/step adjusts colour contrast: squeeze or
        # widen the clim symmetrically around its midpoint.
        if self._engine is None or self._clim_data is None:
            return
        with _styled():
            self._contrast *= self._zoom_factor ** direction
            lo, hi = self._clim_data
            mid, half = (hi + lo) / 2, (hi - lo) / 2
            half = half / self._contrast if self._contrast else half
            self._engine.set_clim(mid - half, mid + half)
            self._canvas.draw_idle()

    def _on_cmap(self, name):
        if self._engine is None:
            return
        with _styled():
            self._engine.set_cmap(name)
            self._canvas.draw_idle()
            if self._hm_input is not None:   # keep the reverse rebuild's cmap fresh
                self._hm_input["cmap"] = name
            if self._hm_overview is not None:
                t, img, _ = self._hm_overview
                self._hm_overview = (t, img, name)
                self._refresh_scrubber()

    def _on_yticks_changed(self, value):
        self._y_ticks = int(value)
        if self._engine is None:
            return
        with _styled():
            self._engine.set_y_ticks_per_channel(self._y_ticks)
        # Re-lay does not touch the limits; reassert the viewport so the paged
        # window (and the tracking secondary axis) stay put, then redraw.
        self._apply_viewport()

    def _reset_scale(self):
        self._contrast = 1.0
        if self._clim_data is not None:
            self._engine.set_clim(*self._clim_data)

    def _layout_margins(self):
        # Fill the canvas: the left holds the per-band frequency ticks + channel
        # names; the right holds the colorbar (coupled to the axes via gridspec,
        # so this right edge moves the bar + its labels, not just the plot).
        return dict(left=0.115, right=0.94, top=0.98, bottom=0.10)

    def _refresh_scrubber(self):
        events, colors = self._scrubber_events()
        if self._hm_overview is not None:
            self._scrubber.set_overview(
                mode="heatmap", t_extent=self._t_extent,
                image=self._hm_overview, events=events,
                event_hue=self._event_hue, event_colors=colors, maxy=1.0)

    def _rebuild_series_visibility(self, layout):
        layout.addWidget(self._section_label("Series"))
        label = self._signal_labels[0] if self._signal_labels else None
        imgs = self._engine.artists.get("images", [])
        vis = all(im.get_visible() for im in imgs) if imgs else True
        layout.addWidget(self._vis_check(
            label or "Heatmap", vis, self._toggle_heatmap))

    def _clear_mode_state(self):
        self._contrast = 1.0
        self._clim_data = None
        self._hm_overview = None

    def _clear_stored_inputs(self):
        self._hm_input = None

    def _snapshot_scale(self):
        return self._contrast

    def _restore_scale(self, snapshot):
        # Re-apply the colour contrast on the rebuilt image (reversing does not
        # change the data range, so the anchored clim baseline is the same).
        if snapshot is None or self._engine is None:
            return
        self._contrast = float(snapshot)
        if self._clim_data is not None and self._contrast != 1.0:
            lo, hi = self._clim_data
            mid, half = (hi + lo) / 2, (hi - lo) / 2
            half = half / self._contrast if self._contrast else half
            self._engine.set_clim(mid - half, mid + half)

    def _render_stored_signals(self):
        # Redraw the single stored heatmap under the current _reverse flag,
        # re-attaching the accumulated events and the live colormap.
        inp, events, hue = self._hm_input, self._events, self._event_hue
        self._hm_input = None
        self._events = None
        self._event_hue = None
        if inp is None:
            return
        self.add_timeheatmap(
            inp["data"], y_values=inp["y_values"], times=inp["times"],
            cmap=inp["cmap"], clim=inp["clim"], label=inp["label"],
            events=events, event_hue=hue)

    # -- heatmap internals ------------------------------------------------
    def _toggle_heatmap(self, on):
        if self._engine is None:
            return
        for im in self._engine.artists.get("images", []):
            im.set_visible(on)
        if "image" in self._engine.artists:
            self._engine.artists["image"].set_visible(on)
        self._canvas.draw_idle()

    def _record_heatmap_overview(self, ordered, times, cmap):
        img = np.median(ordered.reshape(ordered.shape[0], len(times), -1)
                        if ordered.ndim == 3 else
                        np.concatenate([ordered[k] for k in
                                        range(ordered.shape[0])], axis=1),
                        axis=2)
        self._hm_overview = (times, img, cmap)


class TimeHeatmapViewer:
    """Headless-friendly handle that owns the ``QApplication`` and the window.

    Reuses an existing ``QApplication`` when one is running (a process holds only
    one), so it composes with a larger Qt app.
    """

    def __init__(self, *, cha_labels=None, channels_visible=None,
                 reverse_channels=True, window=10.0, step=2.0,
                 zoom_factor=1.2, y_ticks_per_channel=_DEFAULT_Y_TICKS):
        self.app = medusa_style.qt.application(sys.argv)
        self.window = TimeHeatmapViewerWindow(
            cha_labels=cha_labels, channels_visible=channels_visible,
            reverse_channels=reverse_channels, window=window, step=step,
            zoom_factor=zoom_factor, y_ticks_per_channel=y_ticks_per_channel)

    def add_timeheatmap(self, data, **kwargs):
        """Show a per-channel heatmap (see :meth:`TimeHeatmapViewerWindow.add_timeheatmap`)."""
        self.window.add_timeheatmap(data, **kwargs)
        return self

    def add_events(self, events, **kwargs):
        """Overlay extra events (see :meth:`TimeHeatmapViewerWindow.add_events`)."""
        self.window.add_events(events, **kwargs)
        return self

    def clear(self):
        """Reset the viewer."""
        self.window.clear()

    def show(self):
        """Show the window and run the event loop (blocks until closed)."""
        self.window.show()
        self.app.exec()
