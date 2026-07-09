"""Interactive Qt viewer for Event-Related Potentials (ERPs).

:class:`ERPViewer` opens a window that explores an epoched signal
``(n_segments, n_samples, n_channels)`` two ways:

* a **Temporal View** — either *Split* (one segment-averaged ERP per selected
  channel, arranged on a shared-axis grid) or *Mean* (a single ERP averaged over
  the selected channels), both with an optional error/CI band;
* a **Spatial View** — the scalp topography of the ERP at a time point, driven by
  a time slider (or an animation), with a synchronized ERP panel (channel overlay
  or shaded summary) + time cursor below.

A left control panel selects channels and drives the shared analysis (error band,
baseline, smoothing, amplitude limits), the plot options (mode, colors, topography
interpolation/colormap) and one-click exports (each temporal/spatial figure, the
ERP grid, the mean ERP, and the evolving topography as a GIF).

The layering mirrors the time viewers (:mod:`medusa.widgets.time_viewer`): all number
crunching lives in the Qt-free :class:`~medusa.widgets.erp_viewer.analysis.ERPAnalyzer`
(cached) and all drawing in the Qt-free
:mod:`~medusa.widgets.erp_viewer.rendering` builders (which compose the reusable
:mod:`medusa.plots` engines). This class only wires Qt controls to them, inside a
scoped ``medusa_style`` context so global matplotlib state is never mutated.

Example
-------
>>> from medusa.core.data import ChannelSet                       # doctest: +SKIP
>>> from medusa.widgets.erp_viewer import ERPViewer               # doctest: +SKIP
>>> cs = ChannelSet(); cs.add_unipolar_eeg_channels(["Fz", "Cz", "Pz", "Oz"])  # doctest: +SKIP
>>> ERPViewer(epochs, fs=256, channel_set=cs).show()             # blocks  # doctest: +SKIP
"""

import sys

import matplotlib.style as mstyle
import medusa_style
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from medusa.plots import save_figure
from medusa.widgets._toolbar import (
    add_main_toolbar,
    add_toolbar_spacer,
    add_toolbar_status_label,
    pin_toolbar_width,
)
from medusa.widgets.erp_viewer.analysis import ERPAnalyzer
from medusa.widgets.erp_viewer.rendering import (
    SpatialView,
    render_mean,
    render_split,
    render_topography_gif,
    render_topography_still,
)
from medusa.widgets.plot_visualizer import _ExportDialog, _FigurePreview

__all__ = ["ERPViewer", "ERPViewerWindow"]

# Window / layout geometry.
_WINDOW_SIZE = (1250, 800)
_SPLITTER_SIZES = [332, 918]
_PANEL_MIN_W = 210
_TEMPORAL_FIGSIZE = (8.2, 5.4)
_SPATIAL_FIGSIZE = (5.4, 6.2)
#: Default GIF frame size (inches): taller when the ERP panel is included.
_GIF_FIGSIZE = (4.6, 5.4)
_TOPO_ONLY_FIGSIZE = (5.6, 5.2)
_CM_PER_IN = 2.54                 # the size spins are metric; matplotlib is inches

# Error-band menu: label -> (medusa.plots error token, needs a CI level).
_ERROR_CHOICES = (("None", "none", False), ("SD", "std", False),
                  ("SEM", "sem", False), ("CI", "ci", True))
#: Spatial ERP-panel style menu: label -> rendering.SpatialView erp_style token.
_SPATIAL_ERP_CHOICES = (("Channel overlay", "traces"), ("Shaded summary", "band"))
#: Diverging-first colormap menu for the signed ERP topography.
_TOPO_CMAPS = ("(brand diverging)", "RdBu_r", "coolwarm", "seismic", "PuOr_r")

_STYLE_RC = None


def _styled():
    """Scoped MEDUSA matplotlib style for the rendering builders (never global)."""
    global _STYLE_RC
    if _STYLE_RC is None:
        _STYLE_RC = medusa_style.mpl.rcparams()
    return mstyle.context(_STYLE_RC)


def _icon(name):
    """A themed medusa-style icon, or an empty icon if the glyph is absent."""
    try:
        return medusa_style.qt.icon(name)
    except Exception:  # pragma: no cover - bundled asset lookup
        from PySide6.QtGui import QIcon
        return QIcon()


class _ERPCanvas(FigureCanvas):
    """A matplotlib canvas whose figure inherits the MEDUSA figure-level style."""

    def __init__(self, figsize):
        with _styled():
            self.fig = Figure(figsize=figsize)
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)


class _GifExportDialog(QDialog):
    """Collects the animated-topography GIF options, with a live still preview.

    Beyond the animation options (time range, stride, fps, looping) the user picks
    the **frame size** (cm) and **DPI** — together they set each frame's pixel
    dimensions — and, when a ``preview_fn`` is supplied, a representative frame
    (the middle of the chosen range) is rendered on the right and re-rendered
    (debounced) as options change, so the GIF is WYSIWYG.
    """

    def __init__(self, parent, *, t_min, t_max, time_unit, n_samples,
                 default_fps, default_step, default_resolution, default_figsize,
                 preview_fn=None):
        super().__init__(parent)
        self.setWindowTitle("Export topography GIF")
        self._preview_fn = preview_fn
        form = QFormLayout()

        self.t_start = QDoubleSpinBox(self)
        self.t_start.setDecimals(4)
        self.t_start.setRange(t_min, t_max)
        self.t_start.setValue(t_min)
        self.t_start.setSuffix(f" {time_unit}")
        self.t_stop = QDoubleSpinBox(self)
        self.t_stop.setDecimals(4)
        self.t_stop.setRange(t_min, t_max)
        self.t_stop.setValue(t_max)
        self.t_stop.setSuffix(f" {time_unit}")
        rng = QHBoxLayout()
        rng.addWidget(self.t_start)
        rng.addWidget(QLabel("→"))
        rng.addWidget(self.t_stop)
        form.addRow("Time range:", rng)

        self.step = QSpinBox(self)
        self.step.setRange(1, max(1, n_samples - 1))
        self.step.setValue(max(1, int(default_step)))
        self.step.setSuffix(" samples")
        form.addRow("Frame stride:", self.step)

        self.fps = QSpinBox(self)
        self.fps.setRange(1, 60)
        self.fps.setValue(int(default_fps))
        form.addRow("FPS:", self.fps)

        # Frame size (cm) — the UI is metric, matplotlib works in inches.
        self.width = QDoubleSpinBox(self)
        self.width.setRange(2.0, 60.0)
        self.width.setDecimals(1)
        self.width.setValue(default_figsize[0] * _CM_PER_IN)
        self.width.setSuffix(" cm")
        self.height = QDoubleSpinBox(self)
        self.height.setRange(2.0, 60.0)
        self.height.setDecimals(1)
        self.height.setValue(default_figsize[1] * _CM_PER_IN)
        self.height.setSuffix(" cm")
        size = QHBoxLayout()
        size.addWidget(self.width)
        size.addWidget(QLabel("×"))
        size.addWidget(self.height)
        form.addRow("Frame size:", size)

        self.dpi = QSpinBox(self)
        self.dpi.setRange(40, 400)
        self.dpi.setValue(100)
        form.addRow("DPI:", self.dpi)

        self.resolution = QSpinBox(self)
        self.resolution.setRange(40, 400)
        self.resolution.setValue(int(default_resolution))
        self.resolution.setToolTip("Interpolation grid resolution (higher = "
                                   "smoother but slower).")
        form.addRow("Interp. resolution:", self.resolution)

        self.loop = QComboBox(self)
        self.loop.addItems(["Loop forever", "Play once"])
        form.addRow("Looping:", self.loop)

        self.contour = QCheckBox("Contour lines", self)
        self.contour.setChecked(True)   # contours on by default (as in the views)
        form.addRow("", self.contour)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Save | QDialogButtonBox.Cancel, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        # Assemble: options on the left; an optional live preview on the right.
        form_host = QWidget(self)
        form_host.setLayout(form)
        root = QVBoxLayout(self)
        if preview_fn is not None:
            self._preview = _FigurePreview(self)
            preview_box = QGroupBox("Preview (mid-frame)", self)
            pv = QVBoxLayout(preview_box)
            pv.setContentsMargins(6, 6, 6, 6)
            pv.addWidget(self._preview)
            columns = QHBoxLayout()
            columns.addWidget(form_host, 0)
            columns.addWidget(preview_box, 1)
            root.addLayout(columns)
            self.setMinimumSize(720, 460)
            # Debounced re-render so dragging a spin box is not a render storm.
            self._preview_timer = QTimer(self)
            self._preview_timer.setSingleShot(True)
            self._preview_timer.setInterval(200)
            self._preview_timer.timeout.connect(self._render_preview)
            for w in (self.t_start, self.t_stop, self.width, self.height,
                      self.dpi, self.resolution):
                w.valueChanged.connect(self._schedule_preview)
            self.contour.toggled.connect(self._schedule_preview)
            self._render_preview()
        else:
            self._preview = None
            root.addWidget(form_host)
            self.setMinimumWidth(360)
        root.addWidget(buttons)

    def _schedule_preview(self, *_):
        if self._preview is not None:
            self._preview_timer.start()

    def _render_preview(self):
        if self._preview_fn is None:
            return
        try:  # a transient bad value mid-edit must never crash the modal dialog
            pix = self._preview_fn(self.options())
        except Exception:
            return
        if pix is not None:
            self._preview.set_pixmap(pix)

    def options(self) -> dict:
        return dict(
            t_start=self.t_start.value(), t_stop=self.t_stop.value(),
            step=self.step.value(), fps=self.fps.value(),
            # UI is metric (cm); matplotlib's figsize wants inches.
            figsize=(self.width.value() / _CM_PER_IN,
                     self.height.value() / _CM_PER_IN),
            interp_resolution=self.resolution.value(), dpi=self.dpi.value(),
            # loop: 0 = forever; None = play exactly once (omit the GIF loop
            # extension — a count of 1 would actually play twice in most viewers).
            loop=0 if self.loop.currentIndex() == 0 else None,
            contour=self.contour.isChecked())


class ERPViewerWindow(QMainWindow):
    """The ERP viewer window: control panel + temporal/spatial tabs."""

    def __init__(self, data, *, fs=None, times=None, channel_set=None,
                 cha_labels=None, error="ci95", baseline=None, smooth=None,
                 amplitude_limits=None, mode="split", share_y=True,
                 line_width=1.5, band_alpha=0.25, color=None, onset=0.0,
                 show_onset=True, interpolate=True, contour=True,
                 interp_resolution=200, topo_cmap=None, topo_clim=None,
                 symmetric_clim=None, amplitude_unit="µV", time_unit="s",
                 anim_fps=10, anim_step=1, mean_style="band",
                 spatial_erp_style="traces", export_include_erp=True):
        super().__init__()
        self.setWindowTitle("ERP Viewer")
        medusa_style.qt.set_window_icon(self)
        self.resize(*_WINDOW_SIZE)

        self.analyzer = ERPAnalyzer(
            data, fs=fs, times=times, channel_set=channel_set,
            cha_labels=cha_labels)
        self.analyzer.set_baseline(baseline)
        self.analyzer.set_smoothing(smooth)

        # View / style configuration (mirrors the constructor arguments).
        self._mode = str(mode).lower()
        self._error, self._ci_level = self._split_error(error)
        self._line_width = float(line_width)
        self._band_alpha = float(band_alpha)
        self._color = color
        self._onset = float(onset)
        self._show_onset = bool(show_onset)
        self._share_y = bool(share_y)
        self._amp_limits = (None if amplitude_limits is None
                            else (float(amplitude_limits[0]),
                                  float(amplitude_limits[1])))
        self._interpolate = bool(interpolate)
        self._contour = bool(contour)
        self._interp_resolution = int(interp_resolution)
        self._topo_cmap = topo_cmap
        self._custom_cmaps = {}   # combo label -> Colormap, for a non-preset cmap
        # symmetric_clim=None (default) is "auto": symmetric limits unless an
        # explicit topo_clim was supplied, in which case honor that fixed range.
        self._symmetric_clim = ((topo_clim is None) if symmetric_clim is None
                                else bool(symmetric_clim))
        self._topo_clim = (tuple(topo_clim) if topo_clim is not None
                           else self.analyzer.symmetric_clim())
        self._amplitude_unit = amplitude_unit
        self._time_unit = time_unit
        self._anim_fps = int(anim_fps)
        self._anim_step = int(anim_step)
        # Mean-mode spread: "band" (shaded error) or "traces" (per-channel overlay).
        self._mean_style = str(mean_style).lower()
        self._traces_alpha = 0.45
        self._traces_width = 0.6
        # Spatial view ERP panel: "traces" (channel overlay) or "band" (shaded
        # summary), and whether that panel is included in topography exports.
        self._spatial_erp_style = str(spatial_erp_style).lower()
        self._export_include_erp = bool(export_include_erp)

        self._spatial = None            # rendering.SpatialView (lazy)
        self._spatial_dirty = True      # rebuild pending on first Spatial show

        # Debounce for the recompute-heavy controls (avoids redraw storms).
        self._pending_temporal = False
        self._pending_spatial = False
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(120)
        self._debounce.timeout.connect(self._apply_pending)

        # Playback timer for the spatial animation.
        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._advance_frame)

        self._build_ui()
        self._sync_controls_from_config()
        self._refresh_temporal()

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #
    def _build_ui(self):
        bar = add_main_toolbar(self)
        self._add_action(bar, "Export view…", self.export_current_view,
                         "Ctrl+S", icon=_icon("save_as"))
        add_toolbar_spacer(bar)
        self._status_label = QLabel("", self)
        self._status_label.setStyleSheet(
            f"color: {medusa_style.current_theme().text_secondary}; "
            f"padding-right: 10px;")
        add_toolbar_status_label(bar, self._status_label)
        pin_toolbar_width(bar)

        panel = self._build_panel()
        tabs = self._build_tabs()

        main = QSplitter(Qt.Horizontal, self)
        main.addWidget(panel)
        main.addWidget(tabs)
        main.setStretchFactor(1, 1)
        # Free splitter: either side can be dragged (and collapsed) so the control
        # panel is never clipped by the plot's size hint.
        main.setCollapsible(0, True)
        main.setCollapsible(1, True)
        main.setSizes(_SPLITTER_SIZES)
        self.setCentralWidget(main)
        self._update_status()

    def _build_panel(self):
        panel = QWidget()
        panel.setMinimumWidth(_PANEL_MIN_W)
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(0, 0, 0, 0)
        # Scrollable, with a horizontal scrollbar as a fallback so controls are
        # never clipped when the panel is dragged narrow.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.addWidget(self._build_channels_group())
        lay.addWidget(self._build_analysis_group())
        # View-specific controls: a stacked page per tab so only the active view's
        # options (and exports) are shown.
        self._options_stack = QStackedWidget()
        self._options_stack.addWidget(self._build_temporal_options())   # index 0
        self._options_stack.addWidget(self._build_spatial_options())    # index 1
        lay.addWidget(self._options_stack)
        lay.addStretch(1)
        scroll.setWidget(inner)
        outer.addWidget(scroll)
        return panel

    def _build_channels_group(self):
        grp = QGroupBox("Channels")
        lay = QVBoxLayout(grp)
        self._chan_list = QListWidget()
        self._chan_list.setSelectionMode(QListWidget.NoSelection)
        self._chan_list.blockSignals(True)
        for label in self.analyzer.cha_labels:
            item = QListWidgetItem(str(label))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            self._chan_list.addItem(item)
        self._chan_list.blockSignals(False)
        self._chan_list.itemChanged.connect(self._on_channels_changed)
        self._chan_list.setMinimumHeight(150)
        lay.addWidget(self._chan_list)
        row = QHBoxLayout()
        for text, on in (("All", True), ("None", False)):
            btn = QPushButton(text)
            btn.setMaximumWidth(64)   # compact, left-aligned (not stretched wide)
            btn.clicked.connect(lambda _=False, v=on: self._set_all_channels(v))
            row.addWidget(btn)
        row.addStretch(1)
        lay.addLayout(row)
        return grp

    def _build_analysis_group(self):
        # Signal processing that feeds BOTH views (baseline/smoothing change the
        # ERP mean that also drives the topography; the onset marker shows in both).
        grp = QGroupBox("Signal (shared)")
        form = self._form(grp)

        t_lo, t_hi = float(self.analyzer.times[0]), float(self.analyzer.times[-1])
        self._baseline_check = QCheckBox("Baseline")
        self._baseline_check.toggled.connect(self._on_baseline_changed)
        self._base_lo = self._make_spin(t_lo, t_hi, t_lo, self._time_unit)
        self._base_hi = self._make_spin(t_lo, t_hi, min(0.0, t_hi)
                                        if t_lo <= 0.0 <= t_hi else t_hi,
                                        self._time_unit)
        self._base_lo.valueChanged.connect(self._on_baseline_changed)
        self._base_hi.valueChanged.connect(self._on_baseline_changed)
        base_row = QHBoxLayout()
        base_row.setContentsMargins(0, 0, 0, 0)
        base_row.addWidget(self._base_lo, 1)
        base_row.addWidget(QLabel("→"), 0)
        base_row.addWidget(self._base_hi, 1)
        form.addRow(self._baseline_check, base_row)

        self._smooth_check = QCheckBox("Smoothing")
        self._smooth_check.toggled.connect(self._on_smooth_changed)
        self._smooth_spin = QSpinBox()
        self._smooth_spin.setRange(2, max(2, self.analyzer.n_samples))
        self._smooth_spin.setValue(min(5, self.analyzer.n_samples))
        self._smooth_spin.setSuffix(" smp")
        self._smooth_spin.setMinimumWidth(44)
        self._smooth_spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._smooth_spin.valueChanged.connect(self._on_smooth_changed)
        form.addRow(self._smooth_check, self._smooth_spin)

        self._onset_check = QCheckBox("Stimulus onset")
        self._onset_check.toggled.connect(self._on_onset_toggled)
        form.addRow("", self._onset_check)
        return grp

    @staticmethod
    def _export_group(specs):
        grp = QGroupBox("Export")
        lay = QVBoxLayout(grp)
        for text, slot in specs:
            btn = QPushButton(text)
            btn.clicked.connect(slot)
            lay.addWidget(btn)
        return grp

    def _build_temporal_options(self):
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(0, 0, 0, 0)

        grp = QGroupBox("Temporal plot")
        form = self._form(grp)
        self._temporal_form = form
        self._mode_combo = self._combo(
            ["Split (per channel)", "Mean (across channels)"])
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        form.addRow("Mode:", self._mode_combo)

        self._sharey_check = QCheckBox("Shared y-axis")
        self._sharey_check.toggled.connect(self._on_sharey_toggled)
        form.addRow("", self._sharey_check)

        # Mean-mode spread: error band OR per-channel overlay (one or the other).
        self._mean_style_combo = self._combo(["Error band", "Channel overlay"])
        self._mean_style_combo.currentIndexChanged.connect(
            self._on_mean_style_changed)
        form.addRow("Mean spread:", self._mean_style_combo)

        # Error band statistic (split panels + mean band mode).
        self._error_combo = self._combo([c[0] for c in _ERROR_CHOICES])
        self._error_combo.currentIndexChanged.connect(self._on_error_changed)
        self._ci_spin = QSpinBox()
        self._ci_spin.setRange(50, 99)
        self._ci_spin.setValue(self._ci_level)
        self._ci_spin.setSuffix(" %")
        # Fixed, content-sized (so "95 %" is never clipped); the combo takes the rest.
        self._ci_spin.setFixedWidth(86)
        self._ci_spin.valueChanged.connect(self._on_error_changed)
        self._err_widget = QWidget()   # container so the whole row can be hidden
        err_row = QHBoxLayout(self._err_widget)
        err_row.setContentsMargins(0, 0, 0, 0)
        err_row.addWidget(self._error_combo, 1)
        err_row.addWidget(self._ci_spin, 0)
        form.addRow("Error band:", self._err_widget)

        self._lw_spin = self._make_spin(0.5, 6.0, self._line_width, "",
                                        decimals=1, step=0.5)
        self._lw_spin.valueChanged.connect(self._on_style_changed)
        form.addRow("Mean line width:", self._lw_spin)

        self._alpha_spin = self._make_spin(0.0, 1.0, self._band_alpha, "",
                                           decimals=2, step=0.05)
        self._alpha_spin.valueChanged.connect(self._on_style_changed)
        form.addRow("Band opacity:", self._alpha_spin)

        self._trace_alpha_spin = self._make_spin(0.05, 1.0, self._traces_alpha,
                                                 "", decimals=2, step=0.05)
        self._trace_alpha_spin.valueChanged.connect(self._on_style_changed)
        form.addRow("Overlay opacity:", self._trace_alpha_spin)

        # Fixed amplitude (y-axis) limits — temporal-only.
        self._amp_check = QCheckBox("Amplitude")
        self._amp_check.toggled.connect(self._on_amp_changed)
        lo0, hi0 = self.analyzer.amplitude_range()
        span = max(abs(lo0), abs(hi0), 1.0) * 4.0
        self._amp_lo = self._make_spin(-span, span, lo0, self._amplitude_unit)
        self._amp_hi = self._make_spin(-span, span, hi0, self._amplitude_unit)
        self._amp_lo.valueChanged.connect(self._on_amp_changed)
        self._amp_hi.valueChanged.connect(self._on_amp_changed)
        amp_row = QHBoxLayout()
        amp_row.setContentsMargins(0, 0, 0, 0)
        amp_row.addWidget(self._amp_lo, 1)
        amp_row.addWidget(QLabel("→"), 0)
        amp_row.addWidget(self._amp_hi, 1)
        form.addRow(self._amp_check, amp_row)
        v.addWidget(grp)

        # Grid + mean already cover both temporal views (and the toolbar's
        # "Export view…" saves whichever is on screen), so no "current plot" button.
        v.addWidget(self._export_group((
            ("ERP grid…", self.export_grid),
            ("Mean ERP…", self.export_mean),
        )))
        return page

    def _build_spatial_options(self):
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(0, 0, 0, 0)

        grp = QGroupBox("Topography")
        form = self._form(grp)
        self._interp_check = QCheckBox("Interpolate")
        self._interp_check.toggled.connect(self._on_topo_changed)
        form.addRow("", self._interp_check)
        self._contour_check = QCheckBox("Contour lines")
        self._contour_check.toggled.connect(self._on_topo_changed)
        form.addRow("", self._contour_check)

        self._cmap_combo = self._combo(_TOPO_CMAPS)
        self._cmap_combo.currentIndexChanged.connect(self._on_topo_changed)
        form.addRow("Colormap:", self._cmap_combo)

        self._res_spin = QSpinBox()
        self._res_spin.setRange(40, 400)
        self._res_spin.setValue(self._interp_resolution)
        self._res_spin.setMinimumWidth(44)
        self._res_spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._res_spin.valueChanged.connect(self._on_topo_changed)
        form.addRow("Resolution:", self._res_spin)

        self._symclim_check = QCheckBox("Symmetric limits")
        self._symclim_check.toggled.connect(self._on_topo_changed)
        form.addRow("", self._symclim_check)
        self._clim_lo = self._make_spin(-abs(self._topo_clim[0]) * 4 - 1,
                                        abs(self._topo_clim[1]) * 4 + 1,
                                        self._topo_clim[0], self._amplitude_unit)
        self._clim_hi = self._make_spin(-abs(self._topo_clim[0]) * 4 - 1,
                                        abs(self._topo_clim[1]) * 4 + 1,
                                        self._topo_clim[1], self._amplitude_unit)
        self._clim_lo.valueChanged.connect(self._on_topo_changed)
        self._clim_hi.valueChanged.connect(self._on_topo_changed)
        clim_row = QHBoxLayout()
        clim_row.setContentsMargins(0, 0, 0, 0)
        clim_row.addWidget(self._clim_lo, 1)
        clim_row.addWidget(QLabel("→"), 0)
        clim_row.addWidget(self._clim_hi, 1)
        form.addRow("Color limits:", clim_row)

        # ERP panel below the head: channel overlay vs shaded summary, and whether
        # to keep it in the exported topography / GIF.
        self._spatial_erp_combo = self._combo(
            [c[0] for c in _SPATIAL_ERP_CHOICES])
        self._spatial_erp_combo.currentIndexChanged.connect(
            self._on_spatial_erp_changed)
        form.addRow("ERP plot:", self._spatial_erp_combo)
        self._export_erp_check = QCheckBox("Include ERP in export")
        self._export_erp_check.toggled.connect(self._on_export_include_changed)
        form.addRow("", self._export_erp_check)
        v.addWidget(grp)

        v.addWidget(self._export_group((
            ("Current topography…", self.export_spatial),
            ("Topography GIF…", self.export_gif),
        )))
        return page

    def _build_tabs(self):
        self._tabs = QTabWidget()

        # Temporal tab. A small minimum width lets the splitter shrink the plot so
        # the control panel always gets its width.
        self._temporal_canvas = _ERPCanvas(_TEMPORAL_FIGSIZE)
        self._temporal_canvas.setMinimumWidth(200)
        tpage = QWidget()
        tlay = QVBoxLayout(tpage)
        tlay.setContentsMargins(0, 0, 0, 0)
        tlay.addWidget(NavigationToolbar2QT(self._temporal_canvas, self))
        tlay.addWidget(self._temporal_canvas)
        self._tabs.addTab(tpage, "Temporal View")

        # Spatial tab.
        self._spatial_canvas = _ERPCanvas(_SPATIAL_FIGSIZE)
        self._spatial_canvas.setMinimumWidth(200)
        spage = QWidget()
        slay = QVBoxLayout(spage)
        slay.setContentsMargins(0, 0, 0, 0)
        slay.addWidget(NavigationToolbar2QT(self._spatial_canvas, self))
        slay.addWidget(self._spatial_canvas, 1)
        # Footer on ONE centered line: play button, the time slider (its container
        # margins matched to the butterfly's x-axis in _align_slider so it tracks
        # the ERP), and the time readout.
        ctrl = QHBoxLayout()
        ctrl.setSpacing(0)
        self._play_btn = QToolButton()
        self._play_btn.setIcon(_icon("play"))   # crisp themed vector (medusa-style)
        self._play_btn.setIconSize(QSize(20, 20))
        self._play_btn.setFixedSize(QSize(40, 30))
        self._play_btn.setToolTip("Play / pause the topography over time")
        self._play_btn.clicked.connect(self._toggle_play)
        ctrl.addWidget(self._play_btn, 0, Qt.AlignVCenter)
        self._slider_row = QWidget()
        self._slider_layout = QHBoxLayout(self._slider_row)
        self._slider_layout.setContentsMargins(0, 0, 0, 0)
        self._time_slider = QSlider(Qt.Horizontal)
        self._time_slider.setRange(0, self.analyzer.n_samples - 1)
        self._time_slider.valueChanged.connect(self._on_time_changed)
        self._slider_layout.addWidget(self._time_slider)
        ctrl.addWidget(self._slider_row, 1)
        self._time_readout = QLabel("")
        self._time_readout.setFixedWidth(72)
        self._time_readout.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        ctrl.addWidget(self._time_readout, 0, Qt.AlignVCenter)
        slay.addLayout(ctrl)
        self._spatial_canvas.mpl_connect("resize_event",
                                         lambda _e: self._align_slider())
        self._tabs.addTab(spage, "Spatial View")
        self._tabs.currentChanged.connect(self._on_tab_changed)

        if not self.analyzer.has_positions:
            self._tabs.setTabEnabled(1, False)
            self._tabs.setTabToolTip(
                1, "No located sensors in the channel set — provide a ChannelSet "
                   "with electrode positions to enable the spatial view.")
        return self._tabs

    # ------------------------------------------------------------------ #
    # Small widget helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _form(parent):
        """A form whose fields fill (and shrink to) the panel width."""
        form = QFormLayout(parent)
        # Fields expand to the available width and, with the small minimum widths
        # below, shrink with a narrow panel — so controls always adapt, never clip.
        form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        form.setLabelAlignment(Qt.AlignLeft)
        form.setContentsMargins(6, 6, 6, 6)
        form.setHorizontalSpacing(6)
        return form

    @staticmethod
    def _make_spin(lo, hi, value, suffix, *, decimals=3, step=None):
        spin = QDoubleSpinBox()
        spin.setDecimals(decimals)
        spin.setRange(lo, hi)
        spin.setValue(value)
        if suffix:
            spin.setSuffix(f" {suffix}")
        if step is not None:
            spin.setSingleStep(step)
        # Expanding, with a readable-value floor, so it adapts to the panel width
        # without forcing it wide or shrinking the value out of view.
        spin.setMinimumWidth(58)
        spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        return spin

    @staticmethod
    def _combo(items):
        """A combo box that fills the field width and shrinks with a narrow panel."""
        combo = QComboBox()
        combo.addItems(list(items))
        combo.setMinimumWidth(56)
        combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        return combo

    def _add_action(self, bar, text, slot, shortcut=None, icon=None):
        act = QAction(text, self)
        if icon is not None and not icon.isNull():
            act.setIcon(icon)
        act.triggered.connect(slot)
        if shortcut:
            act.setShortcut(QKeySequence(shortcut))
        bar.addAction(act)
        return act

    @staticmethod
    def _split_error(error):
        """Split a medusa.plots error token into (menu token, CI level)."""
        if error is None:
            return "none", 95
        e = str(error).lower()
        if e.startswith("ci"):
            try:
                return "ci", int(float(e[2:]))
            except ValueError:
                return "ci", 95
        if e in ("none", "std", "sem"):
            return e, 95
        return "none", 95

    def _sync_controls_from_config(self):
        """Push the constructor configuration into the freshly-built widgets."""
        idx = next((i for i, c in enumerate(_ERROR_CHOICES)
                    if c[1] == self._error), 3)
        self._block_all(True)
        self._error_combo.setCurrentIndex(idx)
        self._ci_spin.setValue(self._ci_level)
        self._ci_spin.setVisible(_ERROR_CHOICES[idx][2])
        self._mode_combo.setCurrentIndex(0 if self._mode == "split" else 1)
        self._sharey_check.setChecked(self._share_y)
        self._mean_style_combo.setCurrentIndex(
            0 if self._mean_style == "band" else 1)
        self._onset_check.setChecked(self._show_onset)
        base = self.analyzer.baseline
        self._baseline_check.setChecked(base is not None)
        if base is not None:
            self._base_lo.setValue(base[0])
            self._base_hi.setValue(base[1])
        self._set_row_enabled(self._base_lo, base is not None)
        self._set_row_enabled(self._base_hi, base is not None)
        smooth = self.analyzer.smoothing
        self._smooth_check.setChecked(smooth is not None)
        if smooth is not None:
            self._smooth_spin.setValue(smooth)
        self._smooth_spin.setEnabled(smooth is not None)
        self._amp_check.setChecked(self._amp_limits is not None)
        if self._amp_limits is not None:
            self._amp_lo.setValue(self._amp_limits[0])
            self._amp_hi.setValue(self._amp_limits[1])
        self._set_row_enabled(self._amp_lo, self._amp_limits is not None)
        self._set_row_enabled(self._amp_hi, self._amp_limits is not None)
        self._interp_check.setChecked(self._interpolate)
        self._contour_check.setChecked(self._contour)
        self._res_spin.setValue(self._interp_resolution)
        cmap_idx = self._register_topo_cmap(self._topo_cmap)
        self._cmap_combo.setCurrentIndex(cmap_idx)
        self._symclim_check.setChecked(self._symmetric_clim)
        self._set_row_enabled(self._clim_lo, not self._symmetric_clim)
        self._set_row_enabled(self._clim_hi, not self._symmetric_clim)
        self._spatial_erp_combo.setCurrentIndex(
            next((i for i, c in enumerate(_SPATIAL_ERP_CHOICES)
                  if c[1] == self._spatial_erp_style), 0))
        self._export_erp_check.setChecked(self._export_include_erp)
        self._lw_spin.setValue(self._line_width)
        self._alpha_spin.setValue(self._band_alpha)
        self._trace_alpha_spin.setValue(self._traces_alpha)
        self._update_temporal_visibility()
        self._block_all(False)

    def _block_all(self, blocked):
        for w in (self._error_combo, self._ci_spin, self._mode_combo,
                  self._sharey_check, self._mean_style_combo, self._onset_check,
                  self._baseline_check, self._base_lo, self._base_hi,
                  self._smooth_check, self._smooth_spin, self._amp_check,
                  self._amp_lo, self._amp_hi, self._interp_check,
                  self._contour_check, self._res_spin, self._cmap_combo,
                  self._symclim_check, self._clim_lo, self._clim_hi,
                  self._lw_spin, self._alpha_spin, self._trace_alpha_spin,
                  self._spatial_erp_combo, self._export_erp_check):
            w.blockSignals(blocked)

    @staticmethod
    def _set_row_enabled(widget, enabled):
        widget.setEnabled(enabled)

    # ------------------------------------------------------------------ #
    # Selection
    # ------------------------------------------------------------------ #
    def _selected_idxs(self):
        return [i for i in range(self._chan_list.count())
                if self._chan_list.item(i).checkState() == Qt.Checked]

    def _set_all_channels(self, checked):
        self._chan_list.blockSignals(True)
        state = Qt.Checked if checked else Qt.Unchecked
        for i in range(self._chan_list.count()):
            self._chan_list.item(i).setCheckState(state)
        self._chan_list.blockSignals(False)
        self._on_channels_changed()

    def _on_channels_changed(self, *_):
        self._request(temporal=True, spatial=True)

    # ------------------------------------------------------------------ #
    # Control callbacks
    # ------------------------------------------------------------------ #
    def _current_error(self):
        token = _ERROR_CHOICES[self._error_combo.currentIndex()][1]
        return f"ci{self._ci_spin.value()}" if token == "ci" else token

    def _on_error_changed(self, *_):
        needs_ci = _ERROR_CHOICES[self._error_combo.currentIndex()][2]
        self._ci_spin.setVisible(needs_ci)
        self._error = _ERROR_CHOICES[self._error_combo.currentIndex()][1]
        # The spatial ERP panel (shaded style) uses the same error band.
        self._request(temporal=True, spatial=True)

    def _on_baseline_changed(self, *_):
        on = self._baseline_check.isChecked()
        self._set_row_enabled(self._base_lo, on)
        self._set_row_enabled(self._base_hi, on)
        # Sort so a lo > hi drag can never form an empty (invalid) window.
        interval = (tuple(sorted((self._base_lo.value(), self._base_hi.value())))
                    if on else None)
        self.analyzer.set_baseline(interval)
        self._request(temporal=True, spatial=True)

    def _on_smooth_changed(self, *_):
        on = self._smooth_check.isChecked()
        self._smooth_spin.setEnabled(on)
        self.analyzer.set_smoothing(self._smooth_spin.value() if on else None)
        self._request(temporal=True, spatial=True)

    def _on_amp_changed(self, *_):
        on = self._amp_check.isChecked()
        self._set_row_enabled(self._amp_lo, on)
        self._set_row_enabled(self._amp_hi, on)
        self._amp_limits = ((self._amp_lo.value(), self._amp_hi.value())
                            if on else None)
        self._request(temporal=True, spatial=False)

    def _on_onset_toggled(self, *_):
        self._show_onset = self._onset_check.isChecked()
        self._request(temporal=True, spatial=True)

    def _on_mode_changed(self, *_):
        self._mode = "split" if self._mode_combo.currentIndex() == 0 else "mean"
        self._update_temporal_visibility()
        self._request(temporal=True, spatial=False)

    def _on_sharey_toggled(self, *_):
        self._share_y = self._sharey_check.isChecked()
        self._request(temporal=True, spatial=False)

    def _on_mean_style_changed(self, *_):
        self._mean_style = ("band" if self._mean_style_combo.currentIndex() == 0
                            else "traces")
        self._update_temporal_visibility()
        self._request(temporal=True, spatial=False)

    def _update_temporal_visibility(self):
        """Show only the controls that apply to the current mode / mean spread.

        Shared y-axis is split-only; mean spread is mean-only; the error band +
        band-opacity rows apply whenever a band is drawn (split, or mean with the
        "Error band" spread); the overlay-opacity row only to the channel overlay.
        """
        form = self._temporal_form
        split = self._mode == "split"
        band = split or self._mean_style == "band"
        needs_ci = _ERROR_CHOICES[self._error_combo.currentIndex()][2]
        form.setRowVisible(self._sharey_check, split)
        form.setRowVisible(self._mean_style_combo, not split)
        form.setRowVisible(self._err_widget, band)
        form.setRowVisible(self._alpha_spin, band)
        form.setRowVisible(self._trace_alpha_spin, not split and not band)
        self._ci_spin.setVisible(band and needs_ci)

    def _on_style_changed(self, *_):
        self._line_width = self._lw_spin.value()
        self._band_alpha = self._alpha_spin.value()
        self._traces_alpha = self._trace_alpha_spin.value()
        # The spatial ERP panel shares these line/band styles.
        self._request(temporal=True, spatial=True)

    def _on_topo_changed(self, *_):
        self._interpolate = self._interp_check.isChecked()
        self._contour = self._contour_check.isChecked()
        self._interp_resolution = self._res_spin.value()
        self._symmetric_clim = self._symclim_check.isChecked()
        self._set_row_enabled(self._clim_lo, not self._symmetric_clim)
        self._set_row_enabled(self._clim_hi, not self._symmetric_clim)
        self._request(temporal=False, spatial=True)

    def _on_spatial_erp_changed(self, *_):
        self._spatial_erp_style = _SPATIAL_ERP_CHOICES[
            self._spatial_erp_combo.currentIndex()][1]
        self._request(temporal=False, spatial=True)

    def _on_export_include_changed(self, *_):
        # Export-only preference (no live redraw needed).
        self._export_include_erp = self._export_erp_check.isChecked()

    # ------------------------------------------------------------------ #
    # Debounced refresh scheduling
    # ------------------------------------------------------------------ #
    def _request(self, *, temporal, spatial):
        self._pending_temporal |= temporal
        self._pending_spatial |= spatial
        self._debounce.start()

    def _apply_pending(self):
        # A bad control value (e.g. an out-of-range manual limit) must never let
        # an analysis/render error escape the timer slot and abort the app; report
        # it on the status bar and always clear the pending flags.
        try:
            if self._pending_temporal:
                self._refresh_temporal()
            if self._pending_spatial:
                self._spatial_dirty = True
                if self._tabs.currentIndex() == 1:  # only rebuild when visible
                    self._rebuild_spatial()
        except Exception as exc:
            self.statusBar().showMessage(f"Could not update view: {exc}", 6000)
        finally:
            self._pending_temporal = self._pending_spatial = False
        self._update_status()

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #
    def _register_topo_cmap(self, cmap):
        """Combo index for ``cmap``, appending a non-preset name/object.

        A preset selects its existing entry; any other colormap name or
        ``Colormap`` object is appended (objects tracked in ``_custom_cmaps``) and
        selected, so an explicit ``topo_cmap`` is never silently dropped.
        """
        if cmap is None:
            return 0
        if isinstance(cmap, str) and cmap in _TOPO_CMAPS:
            return _TOPO_CMAPS.index(cmap)
        label = cmap if isinstance(cmap, str) else getattr(cmap, "name", "custom")
        if not isinstance(cmap, str):
            self._custom_cmaps[label] = cmap
        existing = self._cmap_combo.findText(label)
        if existing >= 0:
            return existing
        self._cmap_combo.addItem(label)
        return self._cmap_combo.count() - 1

    def _resolve_cmap(self):
        idx = self._cmap_combo.currentIndex()
        if idx == 0:
            return None  # "(brand diverging)"
        text = self._cmap_combo.currentText()
        # A non-preset cmap object is stored by label; a name resolves as-is.
        return self._custom_cmaps.get(text, text)

    def _resolve_clim(self):
        if self._symmetric_clim:
            return self.analyzer.symmetric_clim()
        return (self._clim_lo.value(), self._clim_hi.value())

    def _refresh_temporal(self):
        idxs = self._selected_idxs()
        fig = self._temporal_canvas.fig
        if not idxs:
            fig.clear()
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "Select at least one channel.", ha="center",
                    va="center", transform=ax.transAxes)
            ax.set_axis_off()
            self._temporal_canvas.draw_idle()
            return
        error = self._current_error()
        with _styled():
            if self._mode == "split":
                render_split(
                    fig, self.analyzer, idxs, error=error,
                    share_y=self._share_y, show_onset=self._show_onset,
                    onset=self._onset, line_width=self._line_width,
                    band_alpha=self._band_alpha, color=self._color,
                    amplitude_limits=self._amp_limits,
                    amplitude_unit=self._amplitude_unit,
                    time_unit=self._time_unit)
            else:
                render_mean(
                    fig, self.analyzer, idxs, error=error,
                    style=self._mean_style, traces_alpha=self._traces_alpha,
                    traces_width=self._traces_width,
                    show_onset=self._show_onset, onset=self._onset,
                    line_width=self._line_width, band_alpha=self._band_alpha,
                    color=self._color, amplitude_limits=self._amp_limits,
                    amplitude_unit=self._amplitude_unit,
                    time_unit=self._time_unit)
        self._temporal_canvas.draw_idle()

    def _show_spatial_message(self, text):
        """Blank the spatial canvas and show a centered message (no valid topo)."""
        fig = self._spatial_canvas.fig
        fig.clear()
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        self._spatial_canvas.draw_idle()

    def _spatial_erp_kwargs(self):
        """ERP-panel kwargs shared by the live SpatialView, still export and GIF."""
        return dict(
            erp_style=self._spatial_erp_style, error=self._current_error(),
            band_alpha=self._band_alpha, line_width=self._line_width,
            traces_alpha=self._traces_alpha, traces_width=self._traces_width)

    def _sync_spatial_erp(self, view):
        """Push the current ERP-panel config onto an existing view before rebuild."""
        view._show_onset = self._show_onset
        view._onset = self._onset
        view._erp_style = self._spatial_erp_style
        view._error = self._current_error()
        view._band_alpha = self._band_alpha
        view._line_width = self._line_width
        view._traces_alpha = self._traces_alpha
        view._traces_width = self._traces_width

    def _rebuild_spatial(self):
        if not self.analyzer.has_positions:
            return
        idxs = self._selected_idxs()
        # The topography follows the channel selection; it needs at least one
        # selected channel that has a sensor position.
        if not self.analyzer.any_located(idxs):
            self._show_spatial_message(
                "Select at least one channel with a sensor position.")
            self._spatial = None
            self._spatial_dirty = True
            return
        keep = self._spatial.t_idx if self._spatial is not None else 0
        with _styled():
            if self._spatial is None:
                self._spatial = SpatialView(
                    self._spatial_canvas.fig, self.analyzer, idxs,
                    interpolate=self._interpolate, contour=self._contour,
                    cmap=self._resolve_cmap(), clim=self._resolve_clim(),
                    interp_resolution=self._interp_resolution,
                    show_onset=self._show_onset, onset=self._onset,
                    amplitude_unit=self._amplitude_unit,
                    time_unit=self._time_unit, **self._spatial_erp_kwargs())
            else:
                self._sync_spatial_erp(self._spatial)
                self._spatial.rebuild(
                    idxs, interpolate=self._interpolate, contour=self._contour,
                    cmap=self._resolve_cmap(), clim=self._resolve_clim(),
                    interp_resolution=self._interp_resolution)
            self._spatial.set_time(min(keep, self.analyzer.n_samples - 1))
        self._sync_slider(self._spatial.t_idx)
        self._spatial_canvas.draw_idle()
        self._spatial_dirty = False
        self._update_time_readout()
        self._align_slider()

    def _align_slider(self):
        """Match the slider track to the butterfly's x-axis so it tracks the ERP.

        The footer is one row (play button | slider | readout); insetting the
        slider container by the butterfly's left/right margins — minus the fixed
        play-button and readout widths that flank it — makes the handle sit under
        the time it selects instead of drifting from the plotted trace.
        """
        if self._spatial is None:
            return
        pos = self._spatial.time_ax.get_position()
        w = self._spatial_canvas.width()
        left = int(round(pos.x0 * w)) - self._play_btn.width()
        right = int(round((1.0 - pos.x1) * w)) - self._time_readout.width()
        self._slider_layout.setContentsMargins(max(0, left), 0, max(0, right), 0)

    def _on_tab_changed(self, index):
        # Show only the active view's controls.
        self._options_stack.setCurrentIndex(min(index, 1))
        if index == 1:
            if self._spatial_dirty:
                self._rebuild_spatial()
            self._align_slider()
        elif self._play_timer.isActive():
            # Leaving the spatial tab: stop playing (the pause button is on that
            # now-hidden page, so it would otherwise run against a hidden canvas).
            self._play_timer.stop()
            self._play_btn.setIcon(_icon("play"))

    # ------------------------------------------------------------------ #
    # Spatial time control
    # ------------------------------------------------------------------ #
    def _on_time_changed(self, value):
        if self._spatial is None or self._spatial_dirty:
            self._update_time_readout(value)
            return
        with _styled():
            self._spatial.set_time(value)
        self._spatial_canvas.draw_idle()
        self._update_time_readout(value)

    def _sync_slider(self, idx):
        self._time_slider.blockSignals(True)
        self._time_slider.setValue(int(idx))
        self._time_slider.blockSignals(False)

    def _toggle_play(self):
        if self._play_timer.isActive():
            self._play_timer.stop()
            self._play_btn.setIcon(_icon("play"))
        else:
            if not self.analyzer.has_positions:
                return
            if self._spatial_dirty:
                self._rebuild_spatial()
            self._play_timer.start(max(20, int(1000 / max(1, self._anim_fps))))
            self._play_btn.setIcon(_icon("pause"))

    def _advance_frame(self):
        nxt = (self._time_slider.value() + self._anim_step) % self.analyzer.n_samples
        self._time_slider.setValue(nxt)

    def _update_time_readout(self, idx=None):
        if idx is None:
            idx = self._time_slider.value()
        t = self.analyzer.times[int(idx)]
        self._time_readout.setText(f"{t:.4g} {self._time_unit}")

    # ------------------------------------------------------------------ #
    # Status
    # ------------------------------------------------------------------ #
    def _update_status(self):
        n_sel = len(self._selected_idxs())
        self._status_label.setText(
            f"{n_sel}/{self.analyzer.n_channels} channels  ·  "
            f"{self.analyzer.n_segments} segments  ·  {self._mode}")

    # ------------------------------------------------------------------ #
    # Export
    # ------------------------------------------------------------------ #
    def export_current_view(self):
        """Export whichever tab is showing (temporal or spatial)."""
        if self._tabs.currentIndex() == 0:
            self.export_temporal()
        else:
            self.export_spatial()

    def _quiesce_for_export(self):
        """Freeze the live figure before a modal export dialog opens.

        ``QDialog.exec`` runs a nested event loop that still dispatches this
        window's timers, so a running topography animation or a pending
        debounced redraw would keep mutating the very figure being previewed and
        saved. Stop playback and flush any pending refresh first, so the export
        (and its preview) is exactly the frame the user is looking at.
        """
        if self._play_timer.isActive():
            self._play_timer.stop()
            self._play_btn.setIcon(_icon("play"))
        if self._debounce.isActive():
            self._debounce.stop()
            self._apply_pending()

    def _export_figure(self, fig, default_name):
        self._quiesce_for_export()
        with _styled():
            dlg = _ExportDialog(self, batch=False,
                                default_size=tuple(fig.get_size_inches()),
                                default_name=default_name, fig=fig)
        if dlg.exec() != QDialog.Accepted:
            return
        opts = dlg.options()
        live = tuple(fig.get_size_inches())
        try:
            if opts["size"] is not None:
                fig.set_size_inches(*opts["size"])
            save_figure(fig, opts["path"], transparent=opts["transparent"],
                        facecolor=opts["facecolor"], dpi=opts["dpi"],
                        bbox_inches=opts["bbox_inches"],
                        pad_inches=opts["pad_inches"])
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        finally:
            fig.set_size_inches(*live)
        self.statusBar().showMessage(f"Saved {opts['path']}", 5000)

    def export_temporal(self):
        if not self._selected_idxs():
            QMessageBox.warning(self, "Nothing to export", "Select a channel.")
            return
        self._export_figure(self._temporal_canvas.fig, f"erp_{self._mode}")

    def export_spatial(self):
        if not self.analyzer.has_positions:
            QMessageBox.warning(self, "No spatial view",
                                "The channel set has no located sensors.")
            return
        if not self.analyzer.any_located(self._selected_idxs()):
            QMessageBox.warning(
                self, "No spatial view",
                "Select at least one channel with a sensor position.")
            return
        if self._export_include_erp:
            # WYSIWYG: export the live figure (head + ERP panel) as shown.
            if self._spatial_dirty or self._spatial is None:
                self._rebuild_spatial()
            self._export_figure(self._spatial_canvas.fig, "erp_topography")
            return
        # Topography only: build an offscreen still (no ERP) at the current time.
        t_idx = self._spatial.t_idx if self._spatial is not None else 0

        def build(fig, idxs):
            render_topography_still(
                fig, self.analyzer, idxs, t_idx, include_erp=False,
                interpolate=self._interpolate, contour=self._contour,
                cmap=self._resolve_cmap(), clim=self._resolve_clim(),
                interp_resolution=self._interp_resolution,
                show_onset=self._show_onset, onset=self._onset,
                amplitude_unit=self._amplitude_unit, time_unit=self._time_unit)
        self._export_offscreen(build, _TOPO_ONLY_FIGSIZE, "erp_topography")

    def _export_offscreen(self, builder, figsize, default_name):
        """Render a fresh figure with ``builder(fig)`` and export it (never shown)."""
        idxs = self._selected_idxs()
        if not idxs:
            QMessageBox.warning(self, "Nothing to export", "Select a channel.")
            return
        try:
            with _styled():
                fig = Figure(figsize=figsize)
                builder(fig, idxs)
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        self._export_figure(fig, default_name)

    def export_grid(self):
        def build(fig, idxs):
            render_split(
                fig, self.analyzer, idxs, error=self._current_error(),
                share_y=self._share_y, show_onset=self._show_onset,
                onset=self._onset, line_width=self._line_width,
                band_alpha=self._band_alpha, color=self._color,
                amplitude_limits=self._amp_limits,
                amplitude_unit=self._amplitude_unit, time_unit=self._time_unit)
        self._export_offscreen(build, _TEMPORAL_FIGSIZE, "erp_grid")

    def export_mean(self):
        def build(fig, idxs):
            render_mean(
                fig, self.analyzer, idxs, error=self._current_error(),
                style=self._mean_style, traces_alpha=self._traces_alpha,
                traces_width=self._traces_width, show_onset=self._show_onset,
                onset=self._onset, line_width=self._line_width,
                band_alpha=self._band_alpha, color=self._color,
                amplitude_limits=self._amp_limits,
                amplitude_unit=self._amplitude_unit, time_unit=self._time_unit)
        self._export_offscreen(build, (7.0, 4.4), "erp_mean")

    def _gif_preview_pixmap(self, opts):
        """Render the mid-frame still for the GIF dialog's live preview."""
        import io

        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from PySide6.QtGui import QPixmap

        idxs = self._selected_idxs()
        lo = self.analyzer.time_index(opts["t_start"])
        hi = self.analyzer.time_index(opts["t_stop"])
        t_idx = (min(lo, hi) + max(lo, hi)) // 2
        with _styled():
            fig = Figure(figsize=opts["figsize"])
            FigureCanvasAgg(fig)
            render_topography_still(
                fig, self.analyzer, idxs, t_idx,
                include_erp=self._export_include_erp,
                interpolate=self._interpolate, contour=opts["contour"],
                cmap=self._resolve_cmap(), clim=self._resolve_clim(),
                interp_resolution=opts["interp_resolution"],
                show_onset=self._show_onset, onset=self._onset,
                amplitude_unit=self._amplitude_unit, time_unit=self._time_unit,
                **self._spatial_erp_kwargs())
        w, h = fig.get_size_inches()
        # Cap the preview raster so re-rendering stays responsive while dragging.
        dpi = max(16.0, min(float(opts["dpi"]), 700.0 / max(w, h)))
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, facecolor=fig.get_facecolor())
        pix = QPixmap()
        pix.loadFromData(buf.getvalue(), "PNG")
        return pix

    def export_gif(self):
        if not self.analyzer.has_positions:
            QMessageBox.warning(self, "No spatial view",
                                "The channel set has no located sensors.")
            return
        # The GIF follows the channel selection (like the live view); refuse an
        # empty/unlocated selection instead of silently falling back to all.
        if not self.analyzer.any_located(self._selected_idxs()):
            QMessageBox.warning(
                self, "No spatial view",
                "Select at least one channel with a sensor position.")
            return
        default_figsize = (_GIF_FIGSIZE if self._export_include_erp
                           else (_TOPO_ONLY_FIGSIZE[1], _TOPO_ONLY_FIGSIZE[1]))
        dlg = _GifExportDialog(
            self, t_min=float(self.analyzer.times[0]),
            t_max=float(self.analyzer.times[-1]), time_unit=self._time_unit,
            n_samples=self.analyzer.n_samples, default_fps=self._anim_fps,
            default_step=self._anim_step,
            default_resolution=max(80, self._interp_resolution // 2),
            default_figsize=default_figsize,
            preview_fn=self._gif_preview_pixmap)
        if dlg.exec() != QDialog.Accepted:
            return
        opts = dlg.options()
        path, _ = QFileDialog.getSaveFileName(
            self, "Save topography GIF", "erp_topography.gif", "GIF (*.gif)")
        if not path:
            return
        try:
            with _styled():
                render_topography_gif(
                    path, self.analyzer, idxs=self._selected_idxs(),
                    t_start=opts["t_start"],
                    t_stop=opts["t_stop"], step=opts["step"],
                    interpolate=self._interpolate, contour=opts["contour"],
                    cmap=self._resolve_cmap(), clim=self._resolve_clim(),
                    interp_resolution=opts["interp_resolution"],
                    figsize=opts["figsize"], fps=opts["fps"], dpi=opts["dpi"],
                    loop=opts["loop"], include_erp=self._export_include_erp,
                    show_onset=self._show_onset, onset=self._onset,
                    amplitude_unit=self._amplitude_unit,
                    time_unit=self._time_unit, **self._spatial_erp_kwargs())
        except Exception as exc:
            QMessageBox.critical(self, "GIF export failed", str(exc))
            return
        self.statusBar().showMessage(f"Saved {path}", 5000)


class ERPViewer:
    """Headless-friendly handle that owns the ``QApplication`` and the window.

    Reuses an existing ``QApplication`` when one is running, so it composes with a
    larger Qt app. See :class:`ERPViewerWindow` for the full parameter list.

    Parameters
    ----------
    data
        Epoched signal ``(n_segments, n_samples, n_channels)``.
    fs, times, channel_set, cha_labels
        Time base, spatial layout and channel names (see :class:`ERPViewerWindow`).
    **kwargs
        ERP / styling / topography / animation options (see :class:`ERPViewerWindow`).
    """

    def __init__(self, data, *, fs=None, times=None, channel_set=None,
                 cha_labels=None, **kwargs):
        self.app = medusa_style.qt.application(sys.argv)
        self.window = ERPViewerWindow(
            data, fs=fs, times=times, channel_set=channel_set,
            cha_labels=cha_labels, **kwargs)

    def show(self):
        """Show the window and run the event loop (blocks until closed)."""
        self.window.show()
        self.app.exec()
