"""Interactive Qt viewer for multichannel time-series and time-frequency data.

:class:`TimeViewer` opens a window that browses a long multichannel recording:
stacked **traces** (optionally overlaying transformations of the *same* data,
e.g. raw vs filtered) or a per-channel **heatmap** (e.g. spectrograms), with a
control panel (visible channels, time window, amplitude gain / colour contrast,
events, colormap), an overview **scrubber** with a draggable window, keyboard
navigation, and one-click export.

The trace view is built around *one* signal: repeated :meth:`~TimeViewerWindow.add_timeline`
calls overlay different transformations of it (raw vs filtered, before vs after
artifact removal, ...), **not** unrelated signals — so every call must pass the
same number of channels.

All rendering is delegated to the Qt-free engines in :mod:`medusa.plots`
(:class:`~medusa.plots.TimeLinePlot` / :class:`~medusa.plots.TimeHeatmapPlot`); the
widget only drives them and moves the viewport. Drawing happens inside a scoped
matplotlib style context built from ``medusa_style`` (the MEDUSA single source of
truth), so the global matplotlib state is never mutated.

Example
-------
>>> import numpy as np
>>> from medusa.widgets.time_viewer import TimeViewer        # doctest: +SKIP
>>> v = TimeViewer(cha_labels=["Fz", "Cz", "Pz", "Oz"], channels_visible=4)  # doctest: +SKIP
>>> raw = np.random.randn(5000, 4)                                           # doctest: +SKIP
>>> v.add_timeline(raw, fs=250.0, label="raw")                              # doctest: +SKIP
>>> v.add_timeline(filtered, fs=250.0, label="filtered")  # same 4 channels  # doctest: +SKIP
>>> v.show()   # blocks until the window is closed                           # doctest: +SKIP
"""

import sys
from pathlib import Path

import matplotlib.style as mstyle
import medusa_style
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QAction, QCursor, QIcon, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from medusa.core.data.events import Events
from medusa.plots import TimeHeatmapPlot, TimeLinePlot, save_figure
from medusa.widgets.plot_visualizer import _ExportDialog

__all__ = ["TimeViewer", "TimeViewerWindow"]

#: Heatmap colormap menu — the medusa-style sequential default first, then
#: perceptually-sound alternatives.
_COLORMAPS = (medusa_style.mpl.SEQUENTIAL_CMAP_NAME,
              "inferno", "magma", "plasma", "cividis", "Greys")

#: Toolbar / control icon size (px).
_ICON_PX = 18

#: Cached medusa-style rcParams for the scoped drawing context (see _styled()).
_STYLE_RC = None


def _styled():
    """Scoped MEDUSA matplotlib style for the rendering engines.

    Delegates to medusa-style for the rcParams but applies them through a
    *scoped* matplotlib style context, so the host application's global rcParams
    are never mutated. Resolved once and cached — the viewer does not switch
    theme at runtime.
    """
    global _STYLE_RC
    if _STYLE_RC is None:
        _STYLE_RC = medusa_style.mpl.rcparams()
    return mstyle.context(_STYLE_RC)


# Window / layout geometry (widget-local).
_WINDOW_SIZE = (1180, 760)
_SPLITTER_SIZES = [260, 920]
_PANEL_MAX_W = 300
_MAIN_FIGSIZE = (8.0, 4.5)
_SCRUBBER_FIGSIZE = (8.0, 0.7)

#: Bundled control icons (the viewer-specific glyphs medusa-style does not ship).
_ICON_DIR = Path(__file__).parent / "icons"

#: Local control-icon stems mapped onto a medusa-style *themed* icon (recolors to
#: the active theme, dims when disabled, repaints live on a theme switch). Every
#: control now has a themed SVG; the local-PNG fallback in :func:`_icon` only
#: covers a name absent from both the map and medusa-style.
_MEDUSA_ICONS = {
    "download": "download",
    "ff": "fast_forward",
    "rr": "fast_rewind",
    "zoom_in": "zoom_in",
    "zoom_out": "zoom_out",
    "decrease_channels": "keyboard_double_arrow_up",
    "increase_channels": "keyboard_double_arrow_down",
    "gear": "settings",
}


def _icon(name: str) -> QIcon:
    """A toolbar / control icon.

    Prefers the medusa-style themed icon (theme-recoloring, disabled-dimming);
    falls back to the bundled PNG for the viewer-specific glyphs medusa-style
    does not provide. ``name`` may carry a ``.png`` suffix (legacy call sites).
    """
    stem = name.rsplit(".", 1)[0]
    ms_name = _MEDUSA_ICONS.get(stem)
    if ms_name is not None:
        try:
            return medusa_style.qt.icon(ms_name)
        except FileNotFoundError:  # pragma: no cover - bundled asset present
            pass
    path = _ICON_DIR / name
    return QIcon(str(path)) if path.exists() else QIcon()


# --------------------------------------------------------------------------- #
# Event helper (control layer): coerce a BIDS-style input to an onset/duration
# DataFrame. Events are the unified model (see medusa.core.data.events and
# plots/time_line.py): an `Events`, or a DataFrame with `onset`, `duration`
# (optional, defaults 0) and an optional categorical `event_hue` column.
# --------------------------------------------------------------------------- #
def _to_event_df(events, t_offset: float):
    """Coerce an ``Events`` or DataFrame into a shifted onset/duration DataFrame.

    A ``duration`` column is optional (filled with 0 = instantaneous marker);
    ``onset`` is shifted by ``t_offset``. Any extra columns (e.g. the hue column)
    are preserved.
    """
    df = events.df if isinstance(events, Events) else events
    df = pd.DataFrame(df).copy()
    if "onset" not in df.columns:
        raise ValueError("events need an 'onset' column (BIDS).")
    if "duration" not in df.columns:
        df["duration"] = 0.0
    df["onset"] = df["onset"].astype(float) + float(t_offset)
    df["duration"] = df["duration"].astype(float)
    return df


# --------------------------------------------------------------------------- #
# Canvases
# --------------------------------------------------------------------------- #
class _MainCanvas(FigureCanvas):
    """Main plotting canvas; the wheel emits a gain/contrast step."""

    gainStep = Signal(int)

    def __init__(self):
        # Build the figure under the MEDUSA style so figure-level rcParams
        # (facecolor, dpi) are inherited, not just the axes drawing later.
        with _styled():
            self.fig = Figure(figsize=_MAIN_FIGSIZE)
            self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def wheelEvent(self, event):  # noqa: N802 (Qt override)
        delta = event.angleDelta().y()
        if delta:
            self.gainStep.emit(1 if delta > 0 else -1)


class _ScrubberCanvas(FigureCanvas):
    """Overview strip with a draggable/resizable time window.

    Shows a compact full-record overview (median trace, or median image for a
    heatmap) plus faint mirrored events, and a translucent window rectangle with
    two edge handles. Dragging the centre pans, dragging an edge resizes; the
    cursor changes accordingly. Emits :attr:`windowChanged` ``(lo, hi)`` live.
    """

    windowChanged = Signal(float, float)

    def __init__(self):
        with _styled():
            self.fig = Figure(figsize=_SCRUBBER_FIGSIZE)
            self.ax = self.fig.add_axes((0.06, 0.06, 0.88, 0.88))
        super().__init__(self.fig)
        self.setMinimumHeight(60)
        self.setMaximumHeight(96)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._t_extent = (0.0, 1.0)
        self._window = (0.0, 1.0)
        self._maxy = 1.0
        self._eps = 0.01
        self._focus = None
        self._grab_x = 0.0
        self._win_patch = self._edge_lo = self._edge_hi = None
        self.mpl_connect("button_press_event", self._on_press)
        self.mpl_connect("button_release_event", self._on_release)
        self.mpl_connect("motion_notify_event", self._on_motion)

    # -- drawing ----------------------------------------------------------
    def set_overview(self, *, mode, t_extent, layers=None, image=None,
                     events=None, event_hue=None, event_colors=None, maxy=1.0):
        self.ax.clear()
        self._t_extent = (float(t_extent[0]), float(t_extent[1]))
        self._maxy = float(maxy) or 1.0
        if mode == "timeline" and layers:
            for layer in layers:
                self.ax.plot(layer["times"], layer["values"],
                             color=layer["color"], lw=0.6, alpha=0.75, zorder=2)
        elif mode == "heatmap" and image is not None:
            times, img, cmap = image
            self.ax.imshow(img, extent=(times[0], times[-1],
                                        -self._maxy, self._maxy),
                           aspect="auto", origin="lower", cmap=cmap,
                           alpha=0.85, zorder=1)
        self._draw_events(events, event_hue, event_colors or {})
        self.ax.set_xlim(*self._t_extent)
        self.ax.set_ylim(-self._maxy * 1.05, self._maxy * 1.05)
        self.ax.set_axis_off()
        self.fig.patch.set_alpha(0)
        self.ax.patch.set_alpha(0)
        span = self._t_extent[1] - self._t_extent[0]
        self._eps = max(1e-9, 0.01 * span)
        self._win_patch = self._edge_lo = self._edge_hi = None
        self._draw_window()
        self.draw_idle()

    def _draw_events(self, events, hue, colors):
        # Mirror the events with the SAME per-category colors as the legend
        # (passed in `colors`, keyed by the hue value); fall back to the
        # medusa-style annotation color when there is no hue.
        if events is None:
            return
        event_default = medusa_style.current_theme().palette.highlight
        df = getattr(events, "df", events)
        has_hue = bool(hue) and hue in df.columns
        for _, r in df.iterrows():
            onset, dur = float(r["onset"]), float(r["duration"])
            color = (colors.get(r[hue], event_default) if has_hue
                     else event_default)
            if dur > 0:
                self.ax.axvspan(onset, onset + dur, color=color, alpha=0.25,
                                zorder=1.5)
            else:
                self.ax.axvline(onset, color=color, ls="--", lw=0.9, alpha=0.85,
                                zorder=1.5)

    def set_window(self, lo, hi):
        self._window = (float(lo), float(hi))
        self._draw_window()
        self.draw_idle()

    def _draw_window(self):
        for art in (self._win_patch, self._edge_lo, self._edge_hi):
            if art is not None:
                try:
                    art.remove()
                except (NotImplementedError, ValueError):
                    pass
        lo, hi = self._window
        # selection window rectangle/handles — the medusa-style primary accent
        accent = medusa_style.current_theme().palette.accent_primary
        self._win_patch = self.ax.axvspan(lo, hi, color=accent, alpha=0.18,
                                          zorder=3)
        self._edge_lo = self.ax.axvline(lo, color=accent, lw=2.2, zorder=4)
        self._edge_hi = self.ax.axvline(hi, color=accent, lw=2.2, zorder=4)

    # -- interaction ------------------------------------------------------
    def _focus_at(self, event):
        if event.xdata is None:
            return None
        lo, hi = self._window
        if abs(event.xdata - lo) < self._eps:
            return "lo"
        if abs(event.xdata - hi) < self._eps:
            return "hi"
        if lo < event.xdata < hi:
            return "center"
        return None

    def _on_press(self, event):
        if event.inaxes is not self.ax or event.button != 1:
            return
        self._focus = self._focus_at(event)
        self._grab_x = event.xdata if event.xdata is not None else 0.0

    def _on_release(self, event):
        if self._focus is not None:
            self._focus = None
            QApplication.restoreOverrideCursor()
            self.windowChanged.emit(*self._window)

    def _on_motion(self, event):
        if event.inaxes is not self.ax:
            QApplication.restoreOverrideCursor()
            return
        if self._focus is None:  # hover: hint the available action
            QApplication.restoreOverrideCursor()
            hit = self._focus_at(event)
            if hit in ("lo", "hi"):
                QApplication.setOverrideCursor(QCursor(Qt.SizeHorCursor))
            elif hit == "center":
                QApplication.setOverrideCursor(QCursor(Qt.OpenHandCursor))
            return
        if event.xdata is None:
            return
        lo, hi = self._window
        t0, t1 = self._t_extent
        x = event.xdata
        if self._focus == "lo":
            lo = min(max(x, t0), hi - self._eps)
        elif self._focus == "hi":
            hi = max(min(x, t1), lo + self._eps)
        else:  # center: pan, keeping the window length and clamping to extent
            length = hi - lo
            lo += x - self._grab_x
            hi = lo + length
            if lo < t0:
                lo, hi = t0, t0 + length
            elif hi > t1:
                lo, hi = t1 - length, t1
            self._grab_x = x
            QApplication.setOverrideCursor(QCursor(Qt.ClosedHandCursor))
        self._window = (lo, hi)
        self._draw_window()
        self.draw_idle()
        self.windowChanged.emit(lo, hi)


# --------------------------------------------------------------------------- #
# Window
# --------------------------------------------------------------------------- #
class TimeViewerWindow(QMainWindow):
    """The viewer window: toolbar + main canvas + control panel + scrubber."""

    def __init__(self, *, cha_labels=None, channels_visible=None,
                 reverse_channels=True, window=10.0, step=2.0,
                 zoom_factor=1.2, amplitude_unit="a.u."):
        super().__init__()
        self.setWindowTitle("Time Viewer")
        medusa_style.qt.set_window_icon(self)  # the MEDUSA brand icon
        self.resize(*_WINDOW_SIZE)

        # Configuration / state.
        self._cha_labels = None if cha_labels is None else list(cha_labels)
        self._channels_visible = channels_visible
        self._reverse = reverse_channels
        self._window = float(window)
        self._step = float(step)
        self._zoom_factor = float(zoom_factor)
        self._amplitude_unit = amplitude_unit

        self._mode = None            # "timeline" | "heatmap"
        self._engine = None
        self._n_cha = None
        self._n_signals = 0
        self._vis_window = None      # [lo, hi]
        self._vis_ch = None          # [start, end] display rows
        self._t_extent = None        # (t0, t1)
        self._gain = 1.0
        self._contrast = 1.0
        self._clim_data = None       # (min, max) for heatmap contrast anchoring
        self._events = None          # accumulated BIDS events DataFrame
        self._event_hue = None       # categorical column driving colour/legend
        self._signal_labels: list = []   # one per added signal (None allowed)
        self._overview_layers: list[dict] = []
        self._hm_overview = None     # (times, median_image, cmap_name)

        self._build_ui()

    # -- UI ---------------------------------------------------------------
    def _build_ui(self):
        bar = QToolBar(self)
        bar.setMovable(False)
        bar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        bar.setIconSize(QSize(_ICON_PX, _ICON_PX))
        self.addToolBar(bar)
        self._add_action(bar, "Export…", self.export, "Ctrl+S",
                         icon=_icon("download.png"))
        self._add_action(bar, "Reset view", self.reset_view, "Ctrl+0",
                         icon=medusa_style.qt.icon("refresh"))
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        bar.addWidget(spacer)
        self._status_label = QLabel("", self)
        self._status_label.setStyleSheet(
            f"color: {medusa_style.current_theme().palette.text_secondary}; "
            f"padding-right: 10px;")
        bar.addWidget(self._status_label)

        self._canvas = _MainCanvas()
        self._canvas.gainStep.connect(self._on_gain_step)
        self._ax = self._canvas.ax
        self._nav = NavigationToolbar2QT(self._canvas, self)

        plot_box = QWidget()
        plot_lay = QVBoxLayout(plot_box)
        plot_lay.setContentsMargins(0, 0, 0, 0)
        plot_lay.addWidget(self._nav)
        plot_lay.addWidget(self._canvas)

        self._scrubber = _ScrubberCanvas()
        self._scrubber.windowChanged.connect(self._on_scrubber_window)

        # Right column: the plot with the scrubber directly beneath it, so the
        # overview strip spans only the plot width (never under the left panel).
        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0)
        right_lay.setSpacing(2)
        right_lay.addWidget(plot_box, 1)
        right_lay.addWidget(self._scrubber)

        self._panel = self._build_panel()

        # Left control panel (full height) | right plot+scrubber column.
        main = QSplitter(Qt.Horizontal, self)
        main.addWidget(self._panel)
        main.addWidget(right)
        main.setStretchFactor(1, 1)
        main.setCollapsible(1, False)
        main.setSizes(_SPLITTER_SIZES)
        self.setCentralWidget(main)

        self.statusBar().showMessage("Add a signal to begin.")
        self._install_shortcuts()

    def _build_panel(self):
        panel = QWidget()
        panel.setMaximumWidth(_PANEL_MAX_W)
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(6, 6, 6, 6)

        # Channels.
        chans = QGroupBox("Channels")
        cl = QVBoxLayout(chans)
        row = QHBoxLayout()
        row.addWidget(QLabel("Visible:"))
        self._chan_spin = QSpinBox()
        self._chan_spin.setMinimum(1)
        self._chan_spin.setMaximum(1)
        self._chan_spin.valueChanged.connect(self._on_channels_visible)
        row.addWidget(self._chan_spin)
        cl.addLayout(row)
        pager = QHBoxLayout()
        pager.addWidget(self._icon_button(
            "decrease_channels.png", "Previous channels",
            lambda: self._page_channels(-1)))
        pager.addWidget(self._icon_button(
            "increase_channels.png", "Next channels",
            lambda: self._page_channels(1)))
        cl.addLayout(pager)
        self._reverse_check = QCheckBox("Reverse order")
        self._reverse_check.setChecked(self._reverse)
        self._reverse_check.toggled.connect(self._on_reverse_toggle)
        cl.addWidget(self._reverse_check)
        lay.addWidget(chans)

        # Amplitude (timeline).
        self._amp_group = QGroupBox("Amplitude")
        al = QHBoxLayout(self._amp_group)
        al.addWidget(QLabel("Gain:"))
        self._gain_spin = QDoubleSpinBox()
        self._gain_spin.setRange(0.01, 1000.0)
        self._gain_spin.setDecimals(2)
        self._gain_spin.setValue(1.0)
        self._gain_spin.valueChanged.connect(self._on_gain_value)
        al.addWidget(self._gain_spin)
        al.addWidget(self._icon_button("zoom_out.png", "Decrease gain",
                                       lambda: self._on_gain_step(-1)))
        al.addWidget(self._icon_button("zoom_in.png", "Increase gain",
                                       lambda: self._on_gain_step(1)))
        lay.addWidget(self._amp_group)

        # Colour (heatmap).
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
        contrast.addWidget(self._icon_button("zoom_out.png", "Less contrast",
                                             lambda: self._on_gain_step(-1)))
        contrast.addWidget(self._icon_button("zoom_in.png", "More contrast",
                                             lambda: self._on_gain_step(1)))
        cgl.addLayout(contrast)
        lay.addWidget(self._color_group)
        self._color_group.setVisible(False)

        # Time.
        tgrp = QGroupBox("Time")
        tl = QVBoxLayout(tgrp)
        wrow = QHBoxLayout()
        wrow.addWidget(QLabel("Window (s):"))
        self._window_spin = QDoubleSpinBox()
        self._window_spin.setRange(0.1, 1e6)
        self._window_spin.setDecimals(2)
        self._window_spin.setValue(self._window)
        self._window_spin.valueChanged.connect(self._on_window_length)
        wrow.addWidget(self._window_spin)
        tl.addLayout(wrow)
        srow = QHBoxLayout()
        srow.addWidget(QLabel("Step (s):"))
        self._step_spin = QDoubleSpinBox()
        self._step_spin.setRange(0.01, 1e6)
        self._step_spin.setDecimals(2)
        self._step_spin.setValue(self._step)
        self._step_spin.valueChanged.connect(
            lambda v: setattr(self, "_step", float(v)))
        srow.addWidget(self._step_spin)
        tl.addLayout(srow)
        nav = QHBoxLayout()
        nav.addWidget(self._icon_button("rr.png", "Rewind",
                                        lambda: self._shift_window(-1)))
        nav.addWidget(self._icon_button("ff.png", "Forward",
                                        lambda: self._shift_window(1)))
        tl.addLayout(nav)
        lay.addWidget(tgrp)

        # Legend & visibility: a master legend toggle plus per-item checkboxes
        # (signal series + each event category) rebuilt as content is added.
        vgrp = QGroupBox("Legend && visibility")
        vl = QVBoxLayout(vgrp)
        self._legend_check = QCheckBox("Show legend")
        self._legend_check.setChecked(True)
        self._legend_check.toggled.connect(self._on_legend_visible)
        vl.addWidget(self._legend_check)
        self._vis_container = QWidget()
        self._vis_layout = QVBoxLayout(self._vis_container)
        self._vis_layout.setContentsMargins(0, 4, 0, 0)
        self._vis_layout.setSpacing(2)
        vl.addWidget(self._vis_container)
        lay.addWidget(vgrp)

        lay.addStretch(1)
        return panel

    def _add_action(self, bar, text, slot, shortcut=None, icon=None):
        act = QAction(text, self)
        if icon is not None and not icon.isNull():
            act.setIcon(icon)
        act.triggered.connect(slot)
        if shortcut:
            act.setShortcut(QKeySequence(shortcut))
        bar.addAction(act)
        return act

    def _icon_button(self, icon_name, tooltip, slot):
        """A compact icon button (falls back to a text label if the icon is gone)."""
        btn = QPushButton()
        ic = _icon(icon_name)
        if ic.isNull():
            btn.setText(tooltip)
        else:
            btn.setIcon(ic)
            btn.setIconSize(QSize(_ICON_PX, _ICON_PX))
        btn.setToolTip(tooltip)
        btn.clicked.connect(slot)
        return btn

    def _install_shortcuts(self):
        QShortcut(QKeySequence(Qt.Key_Left), self, lambda: self._shift_window(-1))
        QShortcut(QKeySequence(Qt.Key_Right), self, lambda: self._shift_window(1))
        QShortcut(QKeySequence(Qt.Key_Up), self, lambda: self._page_channels(-1))
        QShortcut(QKeySequence(Qt.Key_Down), self, lambda: self._page_channels(1))
        QShortcut(QKeySequence(Qt.Key_Plus), self, lambda: self._on_gain_step(1))
        QShortcut(QKeySequence(Qt.Key_Equal), self, lambda: self._on_gain_step(1))
        QShortcut(QKeySequence(Qt.Key_Minus), self, lambda: self._on_gain_step(-1))

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
        if self._mode == "heatmap":
            raise RuntimeError("this viewer already shows a heatmap.")
        data = np.asarray(data, dtype=float)
        data2d = data.reshape(-1, data.shape[-1]) if data.ndim >= 2 else \
            data.reshape(-1, 1)
        n_ch = data2d.shape[1]
        times = self._resolve_times(data2d.shape[0], fs, times, t_offset)
        raw_lo, raw_hi = float(times[0]), float(times[-1])
        ordered = data if self._reverse else data[..., ::-1]

        with _styled():
            if self._engine is None:
                self._mode = "timeline"
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
            self._after_add()
        return self

    def add_timeheatmap(self, data, *, y_values, fs=None, times=None,
                        cmap=None, clim=None, label=None, events=None,
                        event_hue=None):
        """Show a per-channel heatmap (e.g. spectrogram); one signal per viewer.

        ``data`` is ``(n_y, n_times, n_channels)`` or segmented
        ``(n_segments, n_y, n_times, n_channels)`` and ``y_values`` are the
        per-channel y-coordinates (e.g. frequencies). ``events`` is BIDS-aligned
        (an :class:`~medusa.core.data.events.Events` or an ``onset``/``duration``
        DataFrame with an optional ``event_hue`` column).
        """
        if self._engine is not None:
            raise RuntimeError(
                "a signal is already shown; heatmap mode takes one signal.")
        data = np.asarray(data, dtype=float)
        n_ch = data.shape[-1]
        n_time = data.shape[1] if data.ndim == 3 else \
            data.shape[0] * data.shape[2]
        times = self._resolve_times(n_time, fs, times, 0.0)
        ordered = data if self._reverse else data[..., ::-1]
        self._clim_data = (float(np.nanmin(data)), float(np.nanmax(data))) \
            if clim is None else (float(clim[0]), float(clim[1]))

        with _styled():
            self._mode = "heatmap"
            self._n_cha = n_ch
            self._init_channels(n_ch)
            cmap = cmap or _COLORMAPS[0]
            self._engine = TimeHeatmapPlot(
                self._ax, cha_labels=self._engine_labels(), y_values=y_values,
                cmap=cmap, clim=self._clim_data, colorbar=True, legend=True)
            self._engine.set_data(ordered, times=times)
            self._n_signals += 1
            self._signal_labels = [label]
            self._sync_combo(self._cmap_combo, cmap)
            self._register_events(events, event_hue, 0.0)
            self._record_heatmap_overview(ordered, times, cmap)
            self._extend_extent(float(times[0]), float(times[-1]))
            self._after_add()
        return self

    def add_events(self, events, *, event_hue=None, t_offset=0.0):
        """Overlay extra events (an ``Events`` or an onset/duration DataFrame)."""
        if self._t_extent is None:
            raise RuntimeError("add a signal before events.")
        with _styled():
            self._register_events(events, event_hue, t_offset)
            self._refresh_scrubber()
            self._rebuild_visibility()
            self._canvas.draw_idle()
        return self

    def clear(self):
        """Remove everything and reset the viewer to its empty state."""
        self._ax.clear()
        for cb in list(self._canvas.fig.axes):
            if cb is not self._ax:
                cb.remove()
        self._engine = None
        self._mode = None
        self._n_cha = self._vis_window = self._vis_ch = None
        self._t_extent = self._clim_data = None
        self._n_signals = 0
        self._gain = self._contrast = 1.0
        self._events = None
        self._event_hue = None
        self._signal_labels = []
        self._overview_layers, self._hm_overview = [], None
        self._rebuild_visibility()
        self._canvas.draw_idle()

    # -- navigation / view ------------------------------------------------
    def _shift_window(self, direction):
        if self._vis_window is None:
            return
        lo, hi = self._vis_window
        length = hi - lo
        lo += direction * self._step
        hi += direction * self._step
        self._vis_window = list(self._make_window_feasible(lo, hi, length))
        self._apply_viewport()
        self._scrubber.set_window(*self._vis_window)
        self._update_status()

    def _make_window_feasible(self, lo, hi, length):
        t0, t1 = self._t_extent
        if hi > t1:
            hi, lo = t1, t1 - length
        if lo < t0:
            lo, hi = t0, min(t0 + length, t1)
        return lo, hi

    def _page_channels(self, direction):
        if self._vis_ch is None:
            return
        n, k = self._n_cha, min(self._channels_visible, self._n_cha)
        start = max(0, min(self._vis_ch[0] + direction * k, n - k))
        self._vis_ch = [start, start + k]
        self._apply_viewport()
        self._update_status()

    def _apply_viewport(self):
        # Engines force-reset xlim AND ylim on every set_data, so always reassert
        # both the time window and the visible-channel block after any draw.
        if self._vis_window is not None:
            self._ax.set_xlim(self._vis_window[0], self._vis_window[1])
        self._apply_channel_ylim()
        self._canvas.draw_idle()

    def _apply_channel_ylim(self):
        if self._vis_ch is None or self._engine is None:
            return
        n = self._n_cha
        s, e = self._vis_ch
        if self._mode == "timeline":
            off = self._engine.offset_value
            half = 0.6 * off
            self._ax.set_ylim((n - e) * off - half, (n - 1 - s) * off + half)
        else:  # heatmap bands: channel i occupies slot [n-1-i, n-i)
            self._ax.set_ylim(n - e, n - s)

    def reset_view(self):
        if self._engine is None:
            return
        self._gain = self._contrast = 1.0
        with _styled():
            if self._mode == "timeline":
                self._engine.set_gain(1.0)
                self._sync_spin(self._gain_spin, 1.0)
            elif self._clim_data is not None:
                self._engine.set_clim(*self._clim_data)
        t0, t1 = self._t_extent
        self._vis_window = [t0, min(t0 + self._window, t1)]
        self._vis_ch = [0, min(self._channels_visible, self._n_cha)]
        self._apply_viewport()
        self._scrubber.set_window(*self._vis_window)
        self._update_status()

    # -- control callbacks ------------------------------------------------
    def _on_gain_step(self, direction):
        if self._engine is None:
            return
        with _styled():
            if self._mode == "timeline":
                self._gain *= self._zoom_factor ** direction
                self._engine.set_gain(self._gain)
                self._sync_spin(self._gain_spin, self._gain)
                self._canvas.draw_idle()
            elif self._clim_data is not None:
                self._contrast *= self._zoom_factor ** direction
                lo, hi = self._clim_data
                mid, half = (hi + lo) / 2, (hi - lo) / 2
                half = half / self._contrast if self._contrast else half
                self._engine.set_clim(mid - half, mid + half)
                self._canvas.draw_idle()

    def _on_gain_value(self, value):
        if self._mode != "timeline" or self._engine is None:
            return
        self._gain = float(value)
        with _styled():
            self._engine.set_gain(self._gain)
            self._canvas.draw_idle()

    def _on_cmap(self, name):
        if self._mode == "heatmap" and self._engine is not None:
            with _styled():
                self._engine.set_cmap(name)
                self._canvas.draw_idle()
                if self._hm_overview is not None:
                    t, img, _ = self._hm_overview
                    self._hm_overview = (t, img, name)
                    self._refresh_scrubber()

    def _on_window_length(self, value):
        self._window = float(value)
        if self._vis_window is None:
            return
        lo = self._vis_window[0]
        self._vis_window = list(self._make_window_feasible(
            lo, lo + self._window, self._window))
        self._apply_viewport()
        self._scrubber.set_window(*self._vis_window)
        self._update_status()

    def _on_channels_visible(self, value):
        if self._n_cha is None:
            return
        self._channels_visible = int(value)
        start = min(self._vis_ch[0], self._n_cha - 1) if self._vis_ch else 0
        k = min(self._channels_visible, self._n_cha)
        start = max(0, min(start, self._n_cha - k))
        self._vis_ch = [start, start + k]
        self._apply_viewport()
        self._update_status()

    def _on_reverse_toggle(self, checked):
        # Reversing channel order means rebuilding from the stored data; for
        # simplicity we only flip the flag for signals added afterwards and warn.
        self._reverse = checked
        if self._engine is not None:
            self.statusBar().showMessage(
                "Reverse applies to signals added next; clear() to re-add.",
                4000)

    def _on_legend_visible(self, checked):
        if self._engine is not None:
            with _styled():
                self._engine.show_legend(checked)
                self._canvas.draw_idle()

    def _on_scrubber_window(self, lo, hi):
        self._vis_window = [lo, hi]
        self._ax.set_xlim(lo, hi)
        self._canvas.draw_idle()
        self._update_status()

    # -- export -----------------------------------------------------------
    def export(self):
        if self._engine is None:
            QMessageBox.warning(self, "Nothing to export", "Add a signal first.")
            return
        fig = self._canvas.fig
        # Build the dialog under the viewer's style so its default export
        # background / dpi reflect the MEDUSA-themed figure (the viewer never
        # mutates global rcParams, so the dialog's rcParams reads must be scoped).
        with _styled():
            dlg = _ExportDialog(self, batch=False,
                                default_size=tuple(fig.get_size_inches()),
                                default_name="time_viewer")
        if dlg.exec() != QDialog.Accepted:
            return
        opts = dlg.options()
        live = tuple(fig.get_size_inches())
        try:
            if opts["size"] is not None:
                fig.set_size_inches(*opts["size"])
            save_figure(fig, opts["path"], transparent=opts["transparent"],
                        facecolor=opts["facecolor"], dpi=opts["dpi"],
                        bbox_inches=opts["bbox_inches"])
        except Exception as exc:  # surface backend/IO errors
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        finally:
            fig.set_size_inches(*live)
            self._canvas.draw_idle()
        self.statusBar().showMessage(f"Saved {opts['path']}", 5000)

    # -- helpers ----------------------------------------------------------
    def _resolve_times(self, n, fs, times, t_offset):
        if times is not None:
            return np.asarray(times, dtype=float).ravel() + t_offset
        if fs is not None:
            return np.arange(n) / float(fs) + t_offset
        return np.arange(n, dtype=float) + t_offset

    def _init_channels(self, n_ch):
        if self._cha_labels is None:
            self._cha_labels = [str(i) for i in range(n_ch)]
        if self._channels_visible is None:
            self._channels_visible = n_ch
        self._channels_visible = min(self._channels_visible, n_ch)
        self._vis_ch = [0, self._channels_visible]
        self._sync_spin(self._chan_spin, self._channels_visible,
                        maximum=n_ch)
        self._amp_group.setVisible(self._mode == "timeline")
        self._color_group.setVisible(self._mode == "heatmap")

    def _engine_labels(self):
        return self._cha_labels if self._reverse else self._cha_labels[::-1]

    def _register_events(self, events, event_hue, t_offset):
        """Merge a BIDS events input into the table and overlay it on the engine."""
        if events is None:
            return
        df = _to_event_df(events, t_offset)
        if event_hue is not None:
            if event_hue not in df.columns:
                raise ValueError(
                    f"event_hue {event_hue!r} is not a column of the events "
                    f"(have {list(df.columns)}).")
            self._event_hue = event_hue
        self._events = df if self._events is None else pd.concat(
            [self._events, df], ignore_index=True)
        self._events = self._events.sort_values("onset").reset_index(drop=True)
        self._engine.set_events(self._events, hue=self._event_hue)

    def _record_overview_layer(self, data2d, times, color):
        # Mirror the engine's per-layer default so the scrubber matches the main
        # plot: every layer (primary + overlays) uses its categorical data color.
        col = color or medusa_style.categorical_color(self._n_signals - 1)
        self._overview_layers.append({
            "times": times, "values": np.median(data2d, axis=1), "color": col})

    def _record_heatmap_overview(self, ordered, times, cmap):
        img = np.median(ordered.reshape(ordered.shape[0], len(times), -1)
                        if ordered.ndim == 3 else
                        np.concatenate([ordered[k] for k in
                                        range(ordered.shape[0])], axis=1),
                        axis=2)
        self._hm_overview = (times, img, cmap)

    def _extend_extent(self, lo, hi):
        if self._t_extent is None:
            self._t_extent = (lo, hi)
            self._vis_window = [lo, min(lo + self._window, hi)]
        else:
            self._t_extent = (min(self._t_extent[0], lo),
                              max(self._t_extent[1], hi))

    def _after_add(self):
        self._tighten_layout()
        self._apply_viewport()
        self._refresh_scrubber()
        self._scrubber.set_window(*self._vis_window)
        self._rebuild_visibility()
        self._update_status()
        self.statusBar().clearMessage()

    def _tighten_layout(self):
        """Trim wasted figure margins (keep room for labels + scale bar/colorbar)."""
        fig = self._canvas.fig
        if self._mode == "heatmap":
            # Leave the right margin to the colorbar; widen the left a touch for
            # the per-band frequency axis + channel labels.
            fig.subplots_adjust(left=0.11, top=0.97, bottom=0.11)
        else:
            # Keep a small right margin for the amplitude scale bar.
            fig.subplots_adjust(left=0.07, right=0.90, top=0.97, bottom=0.11)

    def _refresh_scrubber(self):
        events = self._events
        colors = self._engine.event_colors() if self._engine else {}
        if self._mode == "timeline":
            maxy = max((float(np.nanmax(np.abs(layer["values"])))
                        for layer in self._overview_layers
                        if layer["values"].size), default=1.0) or 1.0
            self._scrubber.set_overview(
                mode="timeline", t_extent=self._t_extent,
                layers=self._overview_layers, events=events,
                event_hue=self._event_hue, event_colors=colors, maxy=maxy)
        elif self._hm_overview is not None:
            self._scrubber.set_overview(
                mode="heatmap", t_extent=self._t_extent,
                image=self._hm_overview, events=events,
                event_hue=self._event_hue, event_colors=colors, maxy=1.0)

    # -- visibility section ----------------------------------------------
    def _rebuild_visibility(self):
        """Repopulate the per-item show/hide checkboxes (series + event types)."""
        self._clear_layout(self._vis_layout)
        if self._engine is None:
            return
        if self._mode == "timeline" and self._signal_labels:
            self._vis_layout.addWidget(self._section_label("Series"))
            for idx, label in enumerate(self._signal_labels):
                lines = self._series_lines(idx)
                vis = all(ln.get_visible() for ln in lines) if lines else True
                self._vis_layout.addWidget(self._vis_check(
                    label or f"Signal {idx + 1}", vis,
                    lambda on, i=idx: self._toggle_series(i, on)))
        elif self._mode == "heatmap":
            self._vis_layout.addWidget(self._section_label("Series"))
            label = self._signal_labels[0] if self._signal_labels else None
            imgs = self._engine.artists.get("images", [])
            vis = all(im.get_visible() for im in imgs) if imgs else True
            self._vis_layout.addWidget(self._vis_check(
                label or "Heatmap", vis, self._toggle_heatmap))
        if self._events is not None and len(self._events) > 0:
            self._vis_layout.addWidget(self._section_label("Events"))
            arts = self._engine.artists.get("events", [])
            cats = self._event_categories()
            if cats:
                col = list(self._events[self._event_hue])
                for cat in cats:
                    vis = next((arts[i].get_visible()
                                for i, v in enumerate(col)
                                if v == cat and i < len(arts)), True)
                    self._vis_layout.addWidget(self._vis_check(
                        str(cat), vis,
                        lambda on, c=cat: self._toggle_event_category(c, on)))
            else:  # events without a hue -> one toggle for all of them
                vis = all(a.get_visible() for a in arts) if arts else True
                self._vis_layout.addWidget(self._vis_check(
                    "Events", vis,
                    lambda on: self._toggle_event_category(None, on)))

    def _series_lines(self, idx):
        arts = self._engine.artists
        if idx == 0:
            return arts.get("lines", [])
        overlays = arts.get("overlays", [])
        return overlays[idx - 1] if idx - 1 < len(overlays) else []

    def _toggle_series(self, idx, on):
        if self._engine is None or self._mode != "timeline":
            return
        for ln in self._series_lines(idx):
            ln.set_visible(on)
        self._canvas.draw_idle()

    def _toggle_heatmap(self, on):
        if self._engine is None:
            return
        for im in self._engine.artists.get("images", []):
            im.set_visible(on)
        if "image" in self._engine.artists:
            self._engine.artists["image"].set_visible(on)
        self._canvas.draw_idle()

    def _toggle_event_category(self, cat, on):
        if self._engine is None or self._events is None:
            return
        arts = self._engine.artists.get("events", [])
        if cat is None or not self._event_hue:
            for a in arts:
                a.set_visible(on)
        else:
            for i, v in enumerate(self._events[self._event_hue]):
                if v == cat and i < len(arts):
                    arts[i].set_visible(on)
        self._canvas.draw_idle()

    def _event_categories(self):
        """Distinct ``event_hue`` values in table order (empty if no hue)."""
        if (self._events is None or len(self._events) == 0
                or not self._event_hue):
            return []
        seen: list = []
        for v in self._events[self._event_hue]:
            if v not in seen:
                seen.append(v)
        return seen

    @staticmethod
    def _vis_check(text, checked, slot):
        cb = QCheckBox(text)
        cb.setChecked(checked)  # set before connecting -> no spurious toggle
        cb.toggled.connect(slot)
        return cb

    @staticmethod
    def _section_label(text):
        lab = QLabel(text)
        lab.setStyleSheet(
            f"color: {medusa_style.current_theme().palette.text_secondary}; "
            f"font-size: 11px; margin-top: 4px;")
        return lab

    @staticmethod
    def _clear_layout(layout):
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
            else:
                sub = item.layout()
                if sub is not None:
                    TimeViewerWindow._clear_layout(sub)

    def _update_status(self):
        if self._vis_window is None:
            return
        lo, hi = self._vis_window
        s, e = self._vis_ch
        self._status_label.setText(
            f"{lo:.1f}–{hi:.1f} s  ·  ch {s}–{e} / {self._n_cha}")

    @staticmethod
    def _sync_spin(spin, value, maximum=None):
        spin.blockSignals(True)
        if maximum is not None:
            spin.setMaximum(maximum)
        spin.setValue(value)
        spin.blockSignals(False)

    @staticmethod
    def _sync_combo(combo, text):
        combo.blockSignals(True)
        idx = combo.findText(text)
        if idx < 0:
            combo.addItem(text)
            idx = combo.findText(text)
        combo.setCurrentIndex(idx)
        combo.blockSignals(False)


class TimeViewer:
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
        self.window = TimeViewerWindow(
            cha_labels=cha_labels, channels_visible=channels_visible,
            reverse_channels=reverse_channels, window=window, step=step,
            zoom_factor=zoom_factor, amplitude_unit=amplitude_unit)

    def add_timeline(self, data, **kwargs):
        """Show a multichannel time-series, or overlay a transformation of it.

        Call once to show a signal; call again with the **same number of
        channels** to overlay another version of it (e.g. raw vs filtered) — the
        viewer compares transformations of one signal, not unrelated signals. See
        :meth:`TimeViewerWindow.add_timeline` for the full parameter list.
        """
        self.window.add_timeline(data, **kwargs)
        return self

    def add_timeheatmap(self, data, **kwargs):
        """Show a per-channel heatmap (see the window method)."""
        self.window.add_timeheatmap(data, **kwargs)
        return self

    def add_events(self, events, **kwargs):
        """Overlay extra events."""
        self.window.add_events(events, **kwargs)
        return self

    def clear(self):
        """Reset the viewer."""
        self.window.clear()

    def show(self):
        """Show the window and run the event loop (blocks until closed)."""
        self.window.show()
        self.app.exec()
