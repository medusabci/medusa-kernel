"""Shared foundation for the time-series viewers (timeline + time-heatmap).

The viewer is split into a small class hierarchy: this module holds everything the
two views share — the window chrome (toolbar, control panel skeleton, overview
scrubber), viewport navigation, the BIDS event model, export and keyboard
shortcuts — as :class:`_BaseTimeViewerWindow`, and the two concrete windows
(:mod:`~medusa.widgets.time_viewer.timeline` and
:mod:`~medusa.widgets.time_viewer.heatmap`) subclass it and fill in the handful of
mode-specific hooks (the view-specific control group, the y-axis scaling, the
scrubber overview, the reset/gain behaviour and the visibility rows).

All rendering is delegated to the Qt-free engines in :mod:`medusa.plots`; drawing
happens inside a scoped ``medusa_style`` context (:func:`_styled`) so the host
application's global matplotlib state is never mutated. PySide6 is a core
dependency of medusa-kernel.
"""

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
    QCheckBox,
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
    QVBoxLayout,
    QWidget,
)

from medusa.core.data.events import Events
from medusa.plots import save_figure
from medusa.widgets._toolbar import (
    add_main_toolbar,
    add_toolbar_spacer,
    add_toolbar_status_label,
    pin_toolbar_width,
)
from medusa.widgets.plot_visualizer import _ExportDialog

__all__ = ["_BaseTimeViewerWindow"]

#: Toolbar / control icon size (px).
_ICON_PX = 18

#: Window / layout geometry (widget-local).
_WINDOW_SIZE = (1180, 760)
_SPLITTER_SIZES = [260, 920]
_PANEL_MAX_W = 300
_MAIN_FIGSIZE = (8.0, 4.5)
_SCRUBBER_FIGSIZE = (8.0, 0.7)

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


def _icon(name: str) -> QIcon:
    """A themed medusa-style icon (recolors to the theme), or an empty icon.

    ``name`` is a medusa-style icon name; medusa-style is the single source of
    truth for the control glyphs, so there is no local icon set to fall back on.
    """
    try:
        return medusa_style.qt.icon(name)
    except Exception:  # pragma: no cover - bundled asset lookup
        return QIcon()


def _to_event_df(events, t_offset: float):
    """Coerce an ``Events`` or DataFrame into a shifted onset/duration DataFrame.

    Events are the unified model (see :mod:`medusa.core.data.events` and
    :mod:`medusa.plots`): an :class:`~medusa.core.data.events.Events`, or a
    DataFrame with ``onset``, ``duration`` (optional, defaults 0) and an optional
    categorical ``event_hue`` column. ``onset`` is shifted by ``t_offset``; any
    extra columns (e.g. the hue column) are preserved.
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
            # The strip is axis-off, so let the overview fill almost the whole
            # figure (a hair of vertical padding keeps the window rectangle's
            # edge handles from touching the frame).
            self.ax = self.fig.add_axes((0.01, 0.04, 0.98, 0.92))
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
        event_default = medusa_style.current_theme().highlight
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

    def reset(self):
        """Blank the overview + window so a stale strip is never draggable.

        Called from the viewer's ``clear()``: without it the previous window
        rectangle stays on screen and still wired, and dragging it would fire
        ``windowChanged`` against a viewer that no longer has a signal.
        """
        self.ax.clear()
        self.ax.set_axis_off()
        self._win_patch = self._edge_lo = self._edge_hi = None
        self._focus = None
        self.unsetCursor()
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
        accent = medusa_style.current_theme().accent_primary
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
        if self._focus == "center":
            self.setCursor(QCursor(Qt.ClosedHandCursor))

    def _on_release(self, event):
        if self._focus is not None:
            self._focus = None
            self.unsetCursor()
            self.windowChanged.emit(*self._window)

    def _on_motion(self, event):
        # The cursor is set on this canvas widget (not pushed on Qt's global
        # override-cursor stack): a drag emits dozens of motion events, so a
        # per-event push/pop stack would leak and leave the app cursor stuck.
        if event.inaxes is not self.ax:
            if self._focus is None:
                self.unsetCursor()
            return
        if self._focus is None:  # hover: hint the available action
            hit = self._focus_at(event)
            if hit in ("lo", "hi"):
                self.setCursor(QCursor(Qt.SizeHorCursor))
            elif hit == "center":
                self.setCursor(QCursor(Qt.OpenHandCursor))
            else:
                self.unsetCursor()
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
        self._window = (lo, hi)
        self._draw_window()
        self.draw_idle()
        self.windowChanged.emit(lo, hi)


# --------------------------------------------------------------------------- #
# Base window
# --------------------------------------------------------------------------- #
class _BaseTimeViewerWindow(QMainWindow):
    """Shared window: toolbar + main canvas + control panel + scrubber.

    Subclasses set the class attributes below and override the mode-specific
    hooks (``_build_view_controls``, ``_apply_channel_ylim``, ``_on_gain_step``,
    ``_reset_scale``, ``_layout_margins``, ``_refresh_scrubber``,
    ``_rebuild_series_visibility`` and, optionally, ``_clear_mode_state``). To
    support the live *Reverse order* toggle they also store their raw inputs and
    implement ``_render_stored_signals`` (replay them under the current channel
    ordering) plus ``_clear_stored_inputs`` / ``_snapshot_scale`` /
    ``_restore_scale``.
    """

    #: ``"timeline"`` / ``"heatmap"`` — drives the scrubber overview and status.
    _MODE = "timeline"
    #: Window title and the default export file stem.
    _TITLE = "Time Viewer"
    _EXPORT_NAME = "time_viewer"

    def __init__(self, *, cha_labels=None, channels_visible=None,
                 reverse_channels=True, window=10.0, step=2.0,
                 zoom_factor=1.2):
        super().__init__()
        self.setWindowTitle(self._TITLE)
        medusa_style.qt.set_window_icon(self)  # the MEDUSA brand icon
        self.resize(*_WINDOW_SIZE)

        # Configuration / state shared by both views.
        self._cha_labels = None if cha_labels is None else list(cha_labels)
        self._channels_visible = channels_visible
        self._reverse = reverse_channels
        self._window = float(window)
        self._step = float(step)
        self._zoom_factor = float(zoom_factor)

        self._engine = None
        self._n_cha = None
        self._n_signals = 0
        self._vis_window = None      # [lo, hi]
        self._vis_ch = None          # [start, end] display rows
        self._t_extent = None        # (t0, t1)
        self._events = None          # accumulated BIDS events DataFrame
        self._event_hue = None       # categorical column driving colour/legend
        self._signal_labels: list = []   # one per added signal (None allowed)

        self._build_ui()

    # -- UI ---------------------------------------------------------------
    def _build_ui(self):
        bar = add_main_toolbar(self)
        self._add_action(bar, "Export…", self.export, "Ctrl+S",
                         icon=_icon("download"))
        self._add_action(bar, "Reset view", self.reset_view, "Ctrl+0",
                         icon=medusa_style.qt.icon("refresh"))
        add_toolbar_spacer(bar)
        self._status_label = QLabel("", self)
        self._status_label.setStyleSheet(
            f"color: {medusa_style.current_theme().text_secondary}; "
            f"padding-right: 10px;")
        add_toolbar_status_label(bar, self._status_label)
        pin_toolbar_width(bar)

        self._canvas = _MainCanvas()
        self._canvas.gainStep.connect(self._on_gain_step)
        self._ax = self._canvas.ax
        # The colorbar (heatmap) subdivides the axes' subplotspec via
        # make_axes_gridspec; stash the pristine spec so a rebuild can restore it
        # and never accumulate nested gridspecs (which would drift the colorbar).
        self._orig_subplotspec = self._ax.get_subplotspec()
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
        lay.addWidget(self._build_channels_group())
        for group in self._build_view_controls():   # mode-specific controls
            lay.addWidget(group)
        lay.addWidget(self._build_time_group())
        lay.addWidget(self._build_legend_group())
        lay.addStretch(1)
        return panel

    def _build_channels_group(self):
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
            "keyboard_double_arrow_up", "Previous channels",
            lambda: self._page_channels(-1)))
        pager.addWidget(self._icon_button(
            "keyboard_double_arrow_down", "Next channels",
            lambda: self._page_channels(1)))
        cl.addLayout(pager)
        self._reverse_check = QCheckBox("Reverse order")
        self._reverse_check.setChecked(self._reverse)
        self._reverse_check.toggled.connect(self._on_reverse_toggle)
        cl.addWidget(self._reverse_check)
        return chans

    def _build_time_group(self):
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
        nav.addWidget(self._icon_button("fast_rewind", "Rewind",
                                        lambda: self._shift_window(-1)))
        nav.addWidget(self._icon_button("fast_forward", "Forward",
                                        lambda: self._shift_window(1)))
        tl.addLayout(nav)
        return tgrp

    def _build_legend_group(self):
        # A master legend toggle plus per-item checkboxes (signal series + each
        # event category), rebuilt as content is added.
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
        return vgrp

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

    # -- mode-specific hooks (overridden by the subclasses) ---------------
    def _build_view_controls(self) -> list:
        """The view-specific control group(s), inserted after *Channels*."""
        return []

    def _apply_channel_ylim(self):
        """Set the y-limits to the visible-channel block (view-specific)."""
        raise NotImplementedError

    def _on_gain_step(self, direction):
        """Wheel / +/- step: amplitude gain (timeline) or contrast (heatmap)."""
        raise NotImplementedError

    def _reset_scale(self):
        """Reset the view-specific scale (gain -> 1, or clim -> data range)."""
        raise NotImplementedError

    def _layout_margins(self) -> dict:
        """``subplots_adjust`` margins that leave room for the view's furniture."""
        return dict(left=0.08, right=0.94, top=0.98, bottom=0.10)

    def _refresh_scrubber(self):
        """Rebuild the overview strip from the current content (view-specific)."""
        raise NotImplementedError

    def _rebuild_series_visibility(self, layout):
        """Add the *Series* show/hide rows for this view (event rows are shared)."""

    def _clear_mode_state(self):
        """Reset any view-specific accumulated state (overview layers, ...)."""

    def _clear_stored_inputs(self):
        """Drop the stored raw signal inputs kept for the reverse rebuild."""

    def _render_stored_signals(self):
        """Re-add every stored signal under the current ``_reverse`` flag.

        Called by :meth:`_rebuild_reversed` after the engine has been torn down;
        must re-register the events too (they are cleared with the engine).
        """
        raise NotImplementedError

    def _snapshot_scale(self):
        """Capture the view scale (gain / contrast) to restore across a rebuild."""
        return None

    def _restore_scale(self, snapshot):
        """Re-apply a scale captured by :meth:`_snapshot_scale` after a rebuild."""

    # -- public API (shared) ----------------------------------------------
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

    def _teardown_engine(self):
        """Tear down the engine + axes furniture, keeping the window config.

        Shared by :meth:`clear` (a full reset) and :meth:`_rebuild_reversed` (a
        rebuild from the stored inputs). Restores the axes' pristine subplotspec so
        a re-created colorbar never nests a fresh gridspec on top of the previous
        one (which would shift the bar leftward on every rebuild).
        """
        self._ax.clear()
        for cb in list(self._canvas.fig.axes):
            if cb is not self._ax:
                cb.remove()
        if self._orig_subplotspec is not None:
            self._ax.set_subplotspec(self._orig_subplotspec)
        self._engine = None
        self._t_extent = None
        self._n_signals = 0
        self._signal_labels = []
        self._clear_mode_state()

    def clear(self):
        """Remove everything and reset the viewer to its empty state."""
        self._teardown_engine()
        self._n_cha = self._vis_window = self._vis_ch = None
        self._events = None
        self._event_hue = None
        self._clear_stored_inputs()
        self._scrubber.reset()   # drop the stale (still-wired) overview window
        self._rebuild_visibility()
        self._canvas.draw_idle()

    def _rebuild_reversed(self):
        """Rebuild every stored signal under the current ``_reverse`` flag.

        The channel order is baked into the data at draw time, so flipping the
        *Reverse order* toggle means re-drawing from the stored inputs. The
        viewport, the view-specific scale (gain / contrast) and the events survive
        the rebuild — only the top-to-bottom channel order changes.
        """
        vis_window = list(self._vis_window) if self._vis_window else None
        vis_ch = list(self._vis_ch) if self._vis_ch else None
        scale = self._snapshot_scale()
        self._teardown_engine()
        self._render_stored_signals()   # re-adds signals + events, reversed
        self._restore_scale(scale)
        if vis_window is not None:
            self._vis_window = vis_window
        if vis_ch is not None:
            self._vis_ch = vis_ch
        self._apply_viewport()
        self._refresh_scrubber()
        if self._vis_window is not None:
            self._scrubber.set_window(*self._vis_window)
        self._rebuild_visibility()
        self._update_status()
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

    def reset_view(self):
        if self._engine is None:
            return
        with _styled():
            self._reset_scale()
        t0, t1 = self._t_extent
        self._vis_window = [t0, min(t0 + self._window, t1)]
        self._vis_ch = [0, min(self._channels_visible, self._n_cha)]
        self._apply_viewport()
        self._scrubber.set_window(*self._vis_window)
        self._update_status()

    # -- control callbacks (shared) ---------------------------------------
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
        # Flip the on-screen channel order live: reversing is equivalent to
        # re-drawing every stored signal with the opposite channel ordering, so
        # rebuild in place (keeping the viewport, scale and events) rather than
        # only affecting signals added afterwards.
        checked = bool(checked)
        if checked == self._reverse:
            return
        self._reverse = checked
        if self._engine is None:
            return
        with _styled():
            self._rebuild_reversed()

    def _on_legend_visible(self, checked):
        if self._engine is not None:
            with _styled():
                self._engine.show_legend(checked)
                self._canvas.draw_idle()

    def _on_scrubber_window(self, lo, hi):
        # Ignore stray drags of a stale overview strip after clear() (no signal):
        # _vis_ch is None then, and _update_status would unpack it.
        if self._n_cha is None:
            return
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
                                default_name=self._EXPORT_NAME)
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

    # -- helpers (shared) -------------------------------------------------
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
        self._sync_spin(self._chan_spin, self._channels_visible, maximum=n_ch)

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
        """Trim wasted figure margins (view-specific, see ``_layout_margins``)."""
        self._canvas.fig.subplots_adjust(**self._layout_margins())

    def _scrubber_events(self):
        """The current events table and the engine's per-category event colors."""
        colors = self._engine.event_colors() if self._engine else {}
        return self._events, colors

    # -- visibility section (shared frame; series rows are view-specific) --
    def _rebuild_visibility(self):
        """Repopulate the per-item show/hide checkboxes (series + event types)."""
        self._clear_layout(self._vis_layout)
        if self._engine is None:
            return
        self._rebuild_series_visibility(self._vis_layout)
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
            f"color: {medusa_style.current_theme().text_secondary}; "
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
                    _BaseTimeViewerWindow._clear_layout(sub)

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
