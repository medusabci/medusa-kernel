"""Interactive Qt viewer for stacked multichannel time-series traces.

:class:`TimeLineViewer` browses a long multichannel recording as stacked
**traces**, optionally overlaying transformations of the *same* data (e.g. raw vs
filtered) on the same channel baselines. It adds an *Amplitude* (gain) control to
the shared :class:`~medusa.widgets.time_viewer.base._BaseTimeViewerWindow` chrome
(visible channels, time window, events, overview scrubber, export) and delegates
drawing to the Qt-free :class:`~medusa.plots.TimeLinePlot` engine.

Repeated :meth:`~TimeLineViewerWindow.add_timeline` calls overlay different
versions of one signal (**not** unrelated signals), so every call must pass the
same number of channels.

Example
-------
>>> import numpy as np
>>> from medusa.widgets.time_viewer import TimeLineViewer        # doctest: +SKIP
>>> v = TimeLineViewer(cha_labels=["Fz", "Cz", "Pz", "Oz"], channels_visible=4)  # doctest: +SKIP
>>> v.add_timeline(raw, fs=250.0, label="raw")                  # doctest: +SKIP
>>> v.add_timeline(filtered, fs=250.0, label="filtered")  # same 4 channels  # doctest: +SKIP
>>> v.show()   # blocks until the window is closed              # doctest: +SKIP
"""

import sys

import medusa_style
import numpy as np
from PySide6.QtWidgets import QDoubleSpinBox, QGroupBox, QHBoxLayout, QLabel

from medusa.plots import TimeLinePlot
from medusa.widgets.time_viewer.base import _BaseTimeViewerWindow, _styled

__all__ = ["TimeLineViewer", "TimeLineViewerWindow"]


class TimeLineViewerWindow(_BaseTimeViewerWindow):
    """The timeline viewer window (stacked traces + amplitude gain)."""

    _MODE = "timeline"
    _TITLE = "Time Viewer — Timeline"
    _EXPORT_NAME = "timeline"

    def __init__(self, *, cha_labels=None, channels_visible=None,
                 reverse_channels=True, window=10.0, step=2.0,
                 zoom_factor=1.2, amplitude_unit="a.u."):
        # View-specific state, set before the base builds the panel (its
        # _build_view_controls hook reads them).
        self._amplitude_unit = amplitude_unit
        self._gain = 1.0
        self._overview_layers: list[dict] = []
        # Raw inputs of every added signal, kept so the *Reverse order* toggle can
        # rebuild the stack with the opposite channel ordering.
        self._signals: list[dict] = []
        super().__init__(
            cha_labels=cha_labels, channels_visible=channels_visible,
            reverse_channels=reverse_channels, window=window, step=step,
            zoom_factor=zoom_factor)

    # -- view-specific control group --------------------------------------
    def _build_view_controls(self):
        self._amp_group = QGroupBox("Amplitude")
        al = QHBoxLayout(self._amp_group)
        al.addWidget(QLabel("Gain:"))
        self._gain_spin = QDoubleSpinBox()
        self._gain_spin.setRange(0.01, 1000.0)
        self._gain_spin.setDecimals(2)
        self._gain_spin.setValue(1.0)
        self._gain_spin.valueChanged.connect(self._on_gain_value)
        al.addWidget(self._gain_spin)
        al.addWidget(self._icon_button("zoom_out", "Decrease gain",
                                       lambda: self._on_gain_step(-1)))
        al.addWidget(self._icon_button("zoom_in", "Increase gain",
                                       lambda: self._on_gain_step(1)))
        return [self._amp_group]

    # -- public API -------------------------------------------------------
    def add_timeline(self, data, *, fs=None, times=None, color=None,
                     label=None, t_offset=0.0, events=None, event_hue=None):
        """Show a multichannel time-series, or overlay a transformation of it.

        This view is built around a single signal seen through different
        transformations (e.g. raw vs filtered), **not** several unrelated
        signals. The first call fixes the channel layout (count + labels); every
        later call **overlays another version of that same signal** on the same
        channel baselines, in a distinct colour named by ``label`` in the legend.
        Consequently every call must pass the **same number of channels** (a
        mismatch raises :class:`ValueError`), and channel ``i`` must refer to the
        same channel across calls.

        Parameters
        ----------
        data : numpy.ndarray
            ``(n_samples, n_channels)`` or segmented ``(n_segments, n_samples,
            n_channels)``. ``n_channels`` must match the first signal added.
        fs : float, optional
            Sampling rate (Hz). Ignored when ``times`` is given.
        times : array-like, optional
            Explicit per-sample timestamps (seconds); overrides ``fs``.
        color : str, optional
            Trace colour for this version; defaults to the next brand colour.
        label : str, optional
            Name shown in the legend (e.g. ``"raw"``, ``"filtered"``).
        t_offset : float, optional
            Shift this version in time (seconds). Usually ``0``, since overlays
            of the same recording share its timebase.
        events, event_hue : optional
            BIDS events overlay — an :class:`~medusa.core.data.events.Events` or
            a DataFrame with ``onset``/``duration`` (and an optional
            ``event_hue`` column) — drawn as onset lines (``duration == 0``) or
            shaded spans (``duration > 0``), coloured/legended by ``event_hue``.

        Raises
        ------
        ValueError
            If an overlay's channel count differs from the first signal's.
        """
        data = np.asarray(data, dtype=float)
        data2d = data.reshape(-1, data.shape[-1]) if data.ndim >= 2 else \
            data.reshape(-1, 1)
        n_ch = data2d.shape[1]
        times = self._resolve_times(data2d.shape[0], fs, times, t_offset)
        raw_lo, raw_hi = float(times[0]), float(times[-1])
        ordered = data if self._reverse else data[..., ::-1]

        with _styled():
            if self._engine is None:
                self._n_cha = n_ch
                self._init_channels(n_ch)
                self._engine = TimeLinePlot(
                    self._ax, cha_labels=self._engine_labels(),
                    color=color, gain=self._gain,
                    amplitude_unit=self._amplitude_unit, legend=True)
                self._engine.set_data(ordered, times=times, label=label)
            else:
                if n_ch != self._n_cha:
                    raise ValueError(
                        f"add_timeline got {n_ch} channels but the viewer was "
                        f"set up with {self._n_cha}. Overlays must be the same "
                        f"signal under a different transformation (e.g. raw vs "
                        f"filtered), so every call needs the same number of "
                        f"channels.")
                col = color or medusa_style.categorical_color(self._n_signals)
                self._engine.add_overlay(ordered, color=col, label=label,
                                         times=times)
            self._n_signals += 1
            self._signal_labels.append(label)
            self._register_events(events, event_hue, t_offset)
            self._record_overview_layer(data2d, times, color)
            self._extend_extent(raw_lo, raw_hi)
            # Store only after a successful draw (a channel-count mismatch raises
            # above), so a reverse rebuild never replays a rejected overlay.
            self._signals.append({"data": data, "times": times,
                                  "color": color, "label": label})
            self._after_add()
        return self

    # -- mode-specific hooks ----------------------------------------------
    def _apply_channel_ylim(self):
        if self._vis_ch is None or self._engine is None:
            return
        n = self._n_cha
        s, e = self._vis_ch
        off = self._engine.offset_value
        half = 0.6 * off
        self._ax.set_ylim((n - e) * off - half, (n - 1 - s) * off + half)

    def _on_gain_step(self, direction):
        if self._engine is None:
            return
        with _styled():
            self._gain *= self._zoom_factor ** direction
            self._engine.set_gain(self._gain)
            self._sync_spin(self._gain_spin, self._gain)
            self._canvas.draw_idle()

    def _on_gain_value(self, value):
        if self._engine is None:
            return
        self._gain = float(value)
        with _styled():
            self._engine.set_gain(self._gain)
            self._canvas.draw_idle()

    def _reset_scale(self):
        self._gain = 1.0
        self._engine.set_gain(1.0)
        self._sync_spin(self._gain_spin, 1.0)

    def _layout_margins(self):
        # Fill the canvas; keep a slim right margin for the amplitude scale bar
        # (anchored just outside the axes) and room below for the time axis.
        return dict(left=0.06, right=0.93, top=0.98, bottom=0.10)

    def _refresh_scrubber(self):
        events, colors = self._scrubber_events()
        maxy = max((float(np.nanmax(np.abs(layer["values"])))
                    for layer in self._overview_layers
                    if layer["values"].size), default=1.0) or 1.0
        self._scrubber.set_overview(
            mode="timeline", t_extent=self._t_extent,
            layers=self._overview_layers, events=events,
            event_hue=self._event_hue, event_colors=colors, maxy=maxy)

    def _rebuild_series_visibility(self, layout):
        if not self._signal_labels:
            return
        layout.addWidget(self._section_label("Series"))
        for idx, label in enumerate(self._signal_labels):
            lines = self._series_lines(idx)
            vis = all(ln.get_visible() for ln in lines) if lines else True
            layout.addWidget(self._vis_check(
                label or f"Signal {idx + 1}", vis,
                lambda on, i=idx: self._toggle_series(i, on)))

    def _clear_mode_state(self):
        self._gain = 1.0
        self._overview_layers = []

    def _clear_stored_inputs(self):
        self._signals = []

    def _snapshot_scale(self):
        return self._gain

    def _restore_scale(self, snapshot):
        if snapshot is None or self._engine is None:
            return
        self._gain = float(snapshot)
        self._engine.set_gain(self._gain)
        self._sync_spin(self._gain_spin, self._gain)

    def _render_stored_signals(self):
        # Replay every stored signal (primary first) under the current _reverse
        # flag; re-attach the accumulated events to the primary re-add.
        signals, events, hue = self._signals, self._events, self._event_hue
        self._signals = []
        self._events = None
        self._event_hue = None
        for i, sig in enumerate(signals):
            self.add_timeline(
                sig["data"], times=sig["times"], color=sig["color"],
                label=sig["label"],
                events=(events if i == 0 else None), event_hue=hue)

    # -- timeline internals -----------------------------------------------
    def _record_overview_layer(self, data2d, times, color):
        # Mirror the engine's per-layer default so the scrubber matches the main
        # plot: every layer (primary + overlays) uses its categorical data color.
        col = color or medusa_style.categorical_color(self._n_signals - 1)
        self._overview_layers.append({
            "times": times, "values": np.median(data2d, axis=1), "color": col})

    def _series_lines(self, idx):
        arts = self._engine.artists
        if idx == 0:
            return arts.get("lines", [])
        overlays = arts.get("overlays", [])
        return overlays[idx - 1] if idx - 1 < len(overlays) else []

    def _toggle_series(self, idx, on):
        if self._engine is None:
            return
        for ln in self._series_lines(idx):
            ln.set_visible(on)
        self._canvas.draw_idle()


class TimeLineViewer:
    """Headless-friendly handle that owns the ``QApplication`` and the window.

    Reuses an existing ``QApplication`` when one is running (a process holds only
    one), so it composes with a larger Qt app.
    """

    def __init__(self, *, cha_labels=None, channels_visible=None,
                 reverse_channels=True, window=10.0, step=2.0,
                 zoom_factor=1.2, amplitude_unit="a.u."):
        # Theme the whole Qt application from the MEDUSA single source of truth
        # (Fusion + QSS + palette + bundled fonts + app icon); reuses an existing
        # QApplication if one is already running.
        self.app = medusa_style.qt.application(sys.argv)
        self.window = TimeLineViewerWindow(
            cha_labels=cha_labels, channels_visible=channels_visible,
            reverse_channels=reverse_channels, window=window, step=step,
            zoom_factor=zoom_factor, amplitude_unit=amplitude_unit)

    def add_timeline(self, data, **kwargs):
        """Show a multichannel time-series, or overlay a transformation of it.

        Call once to show a signal; call again with the **same number of
        channels** to overlay another version of it (e.g. raw vs filtered). See
        :meth:`TimeLineViewerWindow.add_timeline` for the full parameter list.
        """
        self.window.add_timeline(data, **kwargs)
        return self

    def add_events(self, events, **kwargs):
        """Overlay extra events (see :meth:`TimeLineViewerWindow.add_events`)."""
        self.window.add_events(events, **kwargs)
        return self

    def clear(self):
        """Reset the viewer."""
        self.window.clear()

    def show(self):
        """Show the window and run the event loop (blocks until closed)."""
        self.window.show()
        self.app.exec()
