"""A modern Qt browser for reviewing many matplotlib figures.

:class:`PlotVisualizer` opens a window that pages through a set of figures with a
**thumbnail filmstrip** (carousel), per-figure pan/zoom toolbar, keyboard
navigation, and one-click export (single figure or the whole set), reusing
:func:`medusa.plots.save_figure`. It is meant for research, where you often need
to skim and compare a lot of figures quickly.

Example
-------
>>> from medusa.widgets.plot_visualizer import PlotVisualizer   # doctest: +SKIP
>>> viz = PlotVisualizer()                                      # doctest: +SKIP
>>> for fig in figures:                                         # doctest: +SKIP
...     viz.add_figure(fig, title=fig.axes[0].get_title())      # doctest: +SKIP
>>> viz.show_figs()   # blocks until the window is closed       # doctest: +SKIP
"""

import html
import io
import re
import sys
from pathlib import Path

import matplotlib as mpl
import medusa_style
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from matplotlib.colors import to_hex
from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtGui import (
    QAction,
    QColor,
    QFontMetrics,
    QKeySequence,
    QPainter,
    QPixmap,
)
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from medusa.widgets._toolbar import (
    add_main_toolbar,
    add_toolbar_spacer,
    add_toolbar_status_label,
    pin_toolbar_width,
)

__all__ = ["PlotVisualizer", "PlotVisualizerWindow"]

_FORMATS = ("png", "pdf", "svg", "jpg")
_CARD_MIN_W = 180      # minimum filmstrip card width (px); cards fill the pane
_THUMB_MAX_H = 320     # cap on thumbnail height (px) so square figures stay sane
_CM_PER_IN = 2.54      # matplotlib works in inches; the UI is metric (cm)
_WINDOW_SIZE = (1100, 720)  # default window size (px)


def _filmstrip_qss() -> str:
    """Custom QSS for the thumbnail filmstrip, colored from the medusa-style
    palette. This is widget-specific chrome the global medusa-style theme does
    not cover: cards toggle their own highlight via the ``selected`` dynamic
    property (no stylesheet swap), so nothing shifts between states.
    """
    pal = medusa_style.current_theme()
    return f"""
QListWidget {{ border: none; outline: 0; background: {pal.surface}; }}
QListWidget::item {{ border: none; }}
QListWidget::item:selected {{ background: transparent; }}
QWidget#card {{ border-radius: 8px; background: transparent; }}
QWidget#card:hover {{ background: {pal.surface_variant}; }}
QWidget#card[selected="true"] {{ background: {pal.selection}; }}
QWidget#card[selected="true"]:hover {{ background: {pal.selection}; }}
QLabel#cardtitle {{ font-weight: 600; }}
QLabel#cardthumb {{ border: 1px solid {pal.border}; }}
"""


def _rc_facecolor() -> str:
    """The active style's export background as a hex string (MEDUSA paper white).

    Resolves matplotlib's ``'auto'`` sentinel (savefig falls back to the figure
    face, then to white) so this is safe whether or not the MEDUSA style is active.
    """
    fc = mpl.rcParams["savefig.facecolor"]
    if fc == "auto":
        fc = mpl.rcParams["figure.facecolor"]
    if fc == "auto":
        fc = "white"
    return to_hex(fc)


def _rc_savefig_dpi() -> int:
    """The active style's export dpi, resolving matplotlib's ``'figure'`` sentinel."""
    dpi = mpl.rcParams["savefig.dpi"]
    if dpi == "figure":
        dpi = mpl.rcParams["figure.dpi"]
    return int(round(float(dpi)))


def _figure_title(fig, index: int) -> str:
    """Best-effort human title: suptitle, else first axes title, else index."""
    suptitle = getattr(fig, "_suptitle", None)
    if suptitle is not None and suptitle.get_text().strip():
        return suptitle.get_text().strip()
    for ax in fig.axes:
        if ax.get_title().strip():
            return ax.get_title().strip()
    return f"Figure {index + 1}"


def _slug(text: str) -> str:
    """Filesystem-safe stem from a title."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_") or "figure"


#: Extensions treated as a replaceable image-format suffix (a superset of the
#: selectable ``_FORMATS``) so a real typed extension is swapped — but a dot that
#: is part of the name (e.g. ``sub-01_run-1.2``) is preserved and appended to.
_KNOWN_EXTS = frozenset(_FORMATS) | {"jpeg", "tif", "tiff", "eps", "ps"}


def _with_format_suffix(raw_path: str, fmt: str) -> str:
    """Coerce ``raw_path``'s extension to ``fmt`` without mangling dotted names.

    Replaces the suffix only when it is already a recognized image extension;
    otherwise the trailing dot is part of the filename and ``.<fmt>`` is appended.
    A path with no filename component is returned unchanged (the caller validates
    the destination separately), so this never raises on ``"."`` / ``"C:\\"``.
    """
    p = Path(raw_path)
    if not p.name:
        return raw_path
    if p.suffix.lower().lstrip(".") in _KNOWN_EXTS:
        return str(p.with_suffix(f".{fmt}"))
    return f"{p}.{fmt}"


def _thumbnail(fig, long_px: int = 360) -> QPixmap:
    """Render ``fig`` to a ``QPixmap`` source for the filmstrip.

    Rendered a bit larger than a card so it stays sharp when cards grow with the
    pane; it is scaled down per card width in ``_relayout_cards``. Uses the
    figure's *own* background so a dark or transparent figure thumbnails faithfully
    instead of being forced onto white.
    """
    w, h = fig.get_size_inches()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=max(20.0, long_px / max(w, h)),
                bbox_inches="tight", facecolor=fig.get_facecolor())
    pix = QPixmap()
    pix.loadFromData(buf.getvalue(), "PNG")
    return pix


class _FigurePreview(QWidget):
    """Live WYSIWYG preview of the figure to be exported, over a checkerboard.

    The checkerboard backdrop reveals a transparent export (a solid/paper
    background hides it, exactly as the saved file will look). The pixmap is
    re-rendered by :class:`_ExportDialog` whenever an export option changes and
    is scaled here to fit, so the preview always tracks the current settings.
    """

    _TILE = 9  # checkerboard tile size (px)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pix: QPixmap | None = None
        pal = medusa_style.current_theme()
        self._light = QColor(pal.surface)
        self._dark = QColor(pal.surface_variant)
        self.setMinimumSize(300, 240)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_pixmap(self, pix: QPixmap):
        self._pix = pix
        self.update()

    def paintEvent(self, _event):
        painter = QPainter(self)
        rect = self.rect()
        t = self._TILE
        painter.fillRect(rect, self._light)
        for y in range(rect.top(), rect.bottom() + 1, t):
            for x in range(rect.left(), rect.right() + 1, t):
                if ((x // t) + (y // t)) % 2:
                    painter.fillRect(x, y, t, t, self._dark)
        if self._pix is not None and not self._pix.isNull():
            scaled = self._pix.scaled(rect.size(), Qt.KeepAspectRatio,
                                      Qt.SmoothTransformation)
            x = rect.x() + (rect.width() - scaled.width()) // 2
            y = rect.y() + (rect.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)
        painter.end()


class _ExportDialog(QDialog):
    """Single dialog gathering all export options (one figure or the whole set).

    When a ``fig`` is supplied (single-figure exports), the dialog shows a live
    preview that re-renders the figure exactly as it will be saved — the chosen
    size, background/transparency and margin trimming — so the export is WYSIWYG.
    In ``batch`` mode (the whole set) no figure is previewed and per-figure size
    is fixed to each figure's own dimensions.
    """

    #: Cap on the preview raster's long side (px) so re-rendering stays snappy.
    _PREVIEW_MAX_PX = 900

    def __init__(self, parent, *, batch: bool,
                 default_size: "tuple[float, float] | None" = None,
                 default_name: str = "figure", fig=None):
        super().__init__(parent)
        self._batch = batch
        self._default_name = default_name
        # A preview only makes sense for a single figure (not the batch set).
        self._fig = None if batch else fig
        if default_size is None:  # fall back to the active style's figure size
            default_size = tuple(mpl.rcParams["figure.figsize"])
        self._bg = QColor(_rc_facecolor())  # default to the style's paper color
        self._aspect = (default_size[0] / default_size[1]
                        if default_size[1] else 1.0)
        self._syncing_size = False           # re-entrancy guard for lock-aspect
        self.setWindowTitle("Export all figures" if batch else "Export figure")

        # Debounced preview re-render so dragging a spin box is not a savefig storm.
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(150)
        self._preview_timer.timeout.connect(self._render_preview)

        form = QFormLayout()

        self.path_edit = QLineEdit(self)
        browse = QPushButton("Browse…", self)
        browse.clicked.connect(self._browse)
        dest = QHBoxLayout()
        dest.addWidget(self.path_edit)
        dest.addWidget(browse)
        form.addRow("Folder:" if batch else "File:", dest)

        self.fmt = QComboBox(self)
        self.fmt.addItems(_FORMATS)
        self.fmt.currentTextChanged.connect(self._on_format_changed)
        form.addRow("Format:", self.fmt)

        self.dpi = QSpinBox(self)
        self.dpi.setRange(30, 1200)
        self.dpi.setValue(_rc_savefig_dpi())
        self.dpi.valueChanged.connect(self._on_dims_changed)
        form.addRow("DPI:", self.dpi)

        if not batch:
            self.width = QDoubleSpinBox(self)
            self.width.setRange(1.0, 300.0)
            self.width.setDecimals(1)
            self.width.setValue(default_size[0] * _CM_PER_IN)
            self.width.setSuffix(" cm")
            self.width.valueChanged.connect(self._on_width_changed)
            self.height = QDoubleSpinBox(self)
            self.height.setRange(1.0, 300.0)
            self.height.setDecimals(1)
            self.height.setValue(default_size[1] * _CM_PER_IN)
            self.height.setSuffix(" cm")
            self.height.valueChanged.connect(self._on_height_changed)
            size = QHBoxLayout()
            size.addWidget(self.width)
            size.addWidget(QLabel("×"))
            size.addWidget(self.height)
            form.addRow("Size:", size)

            self.lock_aspect = QCheckBox("Lock aspect ratio", self)
            self.lock_aspect.setChecked(True)
            self.lock_aspect.toggled.connect(self._on_lock_toggled)
            form.addRow("", self.lock_aspect)

            self.dims_label = QLabel(self)
            self.dims_label.setStyleSheet(
                f"color: {medusa_style.current_theme().text_secondary};")
            self.dims_label.setToolTip(
                "Full-canvas pixel size at this DPI; 'trim margins' crops it to "
                "the drawn content.")
            form.addRow("", self.dims_label)

        self.pad = QDoubleSpinBox(self)
        self.pad.setRange(0.0, 5.0)
        self.pad.setDecimals(2)
        self.pad.setSingleStep(0.05)
        self.pad.setValue(0.13)          # ~0.05 in, the save_figure default, in cm
        self.pad.setSuffix(" cm")
        self.pad.setToolTip("Whitespace kept around the trimmed content.")
        self.pad.valueChanged.connect(self._schedule_preview)
        form.addRow("Margin padding:", self.pad)

        self.tight = QCheckBox("Trim margins (tight bounding box)", self)
        self.tight.setChecked(True)
        self.tight.toggled.connect(self._on_tight_toggled)
        form.addRow("", self.tight)

        self.transparent = QCheckBox("Transparent background", self)
        self.transparent.toggled.connect(self._toggle_bg)
        form.addRow("", self.transparent)

        self.bg_btn = QPushButton(self)
        self.bg_btn.clicked.connect(self._pick_bg)
        self._refresh_bg_btn()
        form.addRow("Background:", self.bg_btn)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Save | QDialogButtonBox.Cancel, self)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)

        # Assemble: the option form on the left, an optional live preview on the
        # right (single-figure exports only). Buttons span the whole dialog.
        form_host = QWidget(self)
        form_host.setLayout(form)
        root = QVBoxLayout(self)
        if self._fig is not None:
            self._preview = _FigurePreview(self)
            preview_box = QGroupBox("Preview", self)
            pv = QVBoxLayout(preview_box)
            pv.setContentsMargins(6, 6, 6, 6)
            pv.addWidget(self._preview)
            columns = QHBoxLayout()
            columns.addWidget(form_host, 0)
            columns.addWidget(preview_box, 1)
            root.addLayout(columns)
            self.setMinimumSize(780, 500)
            self._update_dims()
            self._render_preview()
        else:
            self._preview = None
            root.addWidget(form_host)
            self.setMinimumWidth(380)
        root.addWidget(buttons)

    # -- callbacks --------------------------------------------------------
    def _browse(self):
        if self._batch:
            folder = QFileDialog.getExistingDirectory(self, "Select folder")
            if folder:
                self.path_edit.setText(folder)
        else:
            fmt = self.fmt.currentText()
            path, _ = QFileDialog.getSaveFileName(
                self, "Save figure", f"{self._default_name}.{fmt}",
                f"{fmt.upper()} (*.{fmt})")
            if path:
                self.path_edit.setText(path)

    def _on_format_changed(self, fmt: str):
        # The Format combo is authoritative: keep the destination's extension in
        # step so choosing e.g. "pdf" actually writes a PDF (savefig infers the
        # format from the suffix). Only rewrites an already-typed file path.
        if self._batch:
            return
        current = self.path_edit.text().strip()
        if current:
            self.path_edit.setText(_with_format_suffix(current, fmt))

    def _on_tight_toggled(self, tight: bool):
        self.pad.setEnabled(tight)   # pad_inches only applies to a tight bbox
        self._schedule_preview()

    def _on_lock_toggled(self, locked: bool):
        if locked and self.height.value():
            # Capture the current ratio so the user can dial in a custom shape,
            # then lock it for subsequent edits.
            self._aspect = self.width.value() / self.height.value()
        self._schedule_preview()

    def _on_width_changed(self, *_):
        if self.lock_aspect.isChecked() and not self._syncing_size and self._aspect:
            self._syncing_size = True
            self.height.setValue(self.width.value() / self._aspect)
            self._syncing_size = False
        self._on_dims_changed()

    def _on_height_changed(self, *_):
        if self.lock_aspect.isChecked() and not self._syncing_size:
            self._syncing_size = True
            self.width.setValue(self.height.value() * self._aspect)
            self._syncing_size = False
        self._on_dims_changed()

    def _on_dims_changed(self, *_):
        self._update_dims()
        self._schedule_preview()

    def _update_dims(self):
        if self._batch or self._fig is None:
            return
        w_px = round(self.width.value() / _CM_PER_IN * self.dpi.value())
        h_px = round(self.height.value() / _CM_PER_IN * self.dpi.value())
        self.dims_label.setText(f"≈ {w_px} × {h_px} px  ·  {self.dpi.value()} dpi")

    def _toggle_bg(self, transparent: bool):
        self.bg_btn.setEnabled(not transparent)
        self._schedule_preview()

    def _pick_bg(self):
        color = QColorDialog.getColor(self._bg, self, "Background color")
        if color.isValid():
            self._bg = color
            self._refresh_bg_btn()
            self._schedule_preview()

    def _refresh_bg_btn(self):
        self.bg_btn.setText(self._bg.name())
        self.bg_btn.setStyleSheet(f"background-color: {self._bg.name()};")

    def _accept(self):
        text = self.path_edit.text().strip()
        # Non-batch needs a real filename (a bare "." / "C:\" has no name part).
        if not text or (not self._batch and not Path(text).name):
            QMessageBox.warning(self, "Missing destination",
                                "Choose a destination first.")
            return
        self.accept()

    # -- preview ----------------------------------------------------------
    def _schedule_preview(self, *_):
        if self._fig is not None:
            self._preview_timer.start()

    def _render_preview(self):
        """Rasterize the figure with the current options and show it in the panel.

        Renders through ``savefig`` (as the export does), so the preview matches
        the saved file. The figure is briefly resized and always restored, and the
        raster's long side is capped for a responsive re-render.
        """
        if self._fig is None:
            return
        opts = self.options()
        fig = self._fig
        live = tuple(fig.get_size_inches())
        buf = io.BytesIO()
        try:
            if opts["size"] is not None:
                fig.set_size_inches(*opts["size"])
            w, h = fig.get_size_inches()
            dpi = max(16.0, min(float(opts["dpi"]),
                                self._PREVIEW_MAX_PX / max(w, h)))
            save_kw = dict(format="png", dpi=dpi, bbox_inches=opts["bbox_inches"],
                           pad_inches=opts["pad_inches"])
            if opts["transparent"]:
                fig.savefig(buf, transparent=True, **save_kw)
            else:
                fc = opts["facecolor"]
                fig.savefig(buf, transparent=False, facecolor=fc, edgecolor=fc,
                            **save_kw)
        except Exception:
            # A transient bad value mid-edit must never crash the modal dialog.
            return
        finally:
            fig.set_size_inches(*live)
        pix = QPixmap()
        pix.loadFromData(buf.getvalue(), "PNG")
        self._preview.set_pixmap(pix)

    # -- result -----------------------------------------------------------
    def options(self) -> dict:
        transparent = self.transparent.isChecked()
        raw_path = self.path_edit.text().strip()
        fmt = self.fmt.currentText()
        # Non-batch: the file suffix follows the selected format so the saved
        # format is always the chosen one (batch paths are folders — left alone).
        path = (_with_format_suffix(raw_path, fmt)
                if raw_path and not self._batch else raw_path)
        opts = dict(
            path=path,
            fmt=fmt,
            dpi=self.dpi.value(),
            bbox_inches="tight" if self.tight.isChecked() else None,
            # UI is metric (cm); matplotlib's pad_inches wants inches.
            pad_inches=self.pad.value() / _CM_PER_IN,
            transparent=transparent,
            facecolor=None if transparent else self._bg.name(),
            # matplotlib needs inches; the UI is cm.
            size=None if self._batch else (self.width.value() / _CM_PER_IN,
                                           self.height.value() / _CM_PER_IN),
        )
        return opts


class _Filmstrip(QListWidget):
    """List whose resize re-lays the cards so they fill the pane width."""

    def __init__(self, on_resize, parent=None):
        super().__init__(parent)
        self._on_resize = on_resize

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._on_resize()


class PlotVisualizerWindow(QMainWindow):
    """The browser window: filmstrip + canvas + navigation/export toolbar."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Plot Visualizer")
        self.resize(*_WINDOW_SIZE)

        self._figs: list = []
        self._canvases: list[FigureCanvas] = []
        self._titles: list[str] = []
        self._sizes: list[tuple[float, float]] = []  # authored figsizes
        self._cards: list[QWidget] = []              # filmstrip card widgets
        self._items: list[QListWidgetItem] = []
        self._relaying = False                         # re-entrancy guard
        self._index = -1

        # Navigation / export toolbar.
        bar = add_main_toolbar(self)
        self._act_prev = self._action(bar, "Previous", self.previous,
                                      [QKeySequence(Qt.Key_Left), "PgUp"],
                                      icon="arrow_back")
        self._act_next = self._action(bar, "Next", self.next,
                                      [QKeySequence(Qt.Key_Right), "PgDown"],
                                      icon="arrow_forward")
        bar.addSeparator()
        self._action(bar, "Export…", self.export_current, ["Ctrl+S"],
                     icon="save_as")
        self._action(bar, "Export all…", self.export_all, ["Ctrl+Shift+S"],
                     icon="download")
        add_toolbar_spacer(bar)
        self._counter = QLabel("0 / 0", self)
        self._counter.setStyleSheet("padding-right: 10px;")
        add_toolbar_status_label(bar, self._counter)
        pin_toolbar_width(bar)

        # Filmstrip (left) + figure stack (right), in a resizable splitter.
        # Cards carry their own border highlight, so the list's selection
        # background is suppressed.
        self.filmstrip = _Filmstrip(self._relayout_cards, self)
        self.filmstrip.setMinimumWidth(_CARD_MIN_W + 16)
        self.filmstrip.setSpacing(2)
        self.filmstrip.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.filmstrip.setStyleSheet(_filmstrip_qss())
        self.filmstrip.currentRowChanged.connect(self._on_filmstrip)

        self.stack = QStackedWidget(self)

        splitter = QSplitter(Qt.Horizontal, self)
        splitter.addWidget(self.filmstrip)
        splitter.addWidget(self.stack)
        splitter.setStretchFactor(1, 1)
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)
        splitter.setHandleWidth(1)
        splitter.setSizes([_CARD_MIN_W + 30, 860])
        self.setCentralWidget(splitter)
        self.statusBar().showMessage("Add figures, then browse with ← / →.")

    # -- building blocks --------------------------------------------------
    def _action(self, bar, text, slot, shortcuts, icon=None):
        act = QAction(text, self)
        if icon is not None:  # theme-recoloring medusa-style SVG
            act.setIcon(medusa_style.qt.icon(icon))
        act.triggered.connect(slot)
        act.setShortcuts([QKeySequence(s) for s in shortcuts])
        bar.addAction(act)
        return act

    def add_figure(self, fig, title: "str | None" = None):
        """Add a matplotlib ``Figure`` (optionally with a display ``title``)."""
        index = len(self._figs)
        title = title or _figure_title(fig, index)

        canvas = FigureCanvas(fig)
        page = QWidget(self)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(NavigationToolbar2QT(canvas, page))
        layout.addWidget(canvas)
        self.stack.addWidget(page)

        card = self._make_card(index, title, _thumbnail(fig))
        item = QListWidgetItem(self.filmstrip)
        item.setToolTip(title)
        self.filmstrip.setItemWidget(item, card)

        self._figs.append(fig)
        self._canvases.append(canvas)
        self._titles.append(title)
        self._sizes.append(tuple(fig.get_size_inches()))
        self._cards.append(card)
        self._items.append(item)
        self._relayout_cards()
        canvas.draw_idle()
        if self._index < 0:
            self._set_current(0)
        else:
            self._update_counter()

    def _make_card(self, index: int, title: str, src: QPixmap) -> QWidget:
        """Filmstrip card: the wrapped title above a framed thumbnail.

        Sizing (card width, thumbnail scale, item height) is set by
        :meth:`_relayout_cards`, which runs whenever the pane is resized so cards
        always fill the filmstrip width. ``src`` is kept for re-scaling.
        """
        card = QWidget()
        card.setObjectName("card")
        card.setAttribute(Qt.WA_StyledBackground, True)  # paint rounded bg
        card.setProperty("selected", False)
        lay = QVBoxLayout(card)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)
        label = QLabel(f"{index + 1}.  {title}")
        label.setObjectName("cardtitle")
        label.setWordWrap(True)
        image = QLabel()
        image.setObjectName("cardthumb")
        image.setAlignment(Qt.AlignCenter)
        lay.addWidget(label)
        lay.addWidget(image)
        card._title = label
        card._image = image
        card._src = src
        return card

    def _relayout_cards(self):
        """Resize every card (and rescale its thumbnail) to the pane width."""
        if self._relaying or not self._cards:
            return
        self._relaying = True
        try:
            w = self.filmstrip.viewport().width()
            if w <= 0:
                return
            inner = max(60, w - 22)  # card margins + a small buffer
            for item, card in zip(self._items, self._cards):
                card._title.setFixedWidth(inner)
                thumb = card._src.scaled(inner, _THUMB_MAX_H, Qt.KeepAspectRatio,
                                         Qt.SmoothTransformation)
                card._image.setPixmap(thumb)
                title_h = QFontMetrics(card._title.font()).boundingRect(
                    0, 0, inner, 4000, Qt.TextWordWrap, card._title.text()
                ).height()
                # margins (16) + spacing (6) + thumb border (2) + pad (8)
                item.setSizeHint(QSize(w - 2, title_h + thumb.height() + 32))
        finally:
            self._relaying = False

    @staticmethod
    def _select_card(card, selected: bool):
        """Toggle a card's ``selected`` state and re-apply the stylesheet."""
        card.setProperty("selected", selected)
        card.style().unpolish(card)
        card.style().polish(card)

    # -- navigation -------------------------------------------------------
    def _set_current(self, index: int):
        if not self._figs:
            return
        prev = self._index
        self._index = index % len(self._figs)
        self.stack.setCurrentIndex(self._index)
        self.filmstrip.blockSignals(True)
        self.filmstrip.setCurrentRow(self._index)
        self.filmstrip.blockSignals(False)
        if 0 <= prev < len(self._cards) and prev != self._index:
            self._select_card(self._cards[prev], False)
        self._select_card(self._cards[self._index], True)
        self.filmstrip.scrollToItem(self.filmstrip.item(self._index))
        self._update_counter()

    def _update_counter(self):
        n = len(self._figs)
        if not n:
            self._counter.setText("0 / 0")
            return
        title = self._titles[self._index]
        if len(title) > 60:
            title = title[:57] + "…"
        muted = medusa_style.current_theme().text_secondary
        self._counter.setText(
            f"<b>{self._index + 1} / {n}</b>&nbsp;&nbsp;·&nbsp;&nbsp;"
            f"<span style='color: {muted};'>{html.escape(title)}</span>")

    def _on_filmstrip(self, row: int):
        if row >= 0 and row != self._index:
            self._set_current(row)

    def next(self):
        self._set_current(self._index + 1)

    def previous(self):
        self._set_current(self._index - 1)

    # -- export -----------------------------------------------------------
    def export_current(self):
        if not self._figs:
            QMessageBox.warning(self, "No figure", "There is nothing to export.")
            return
        i = self._index
        dlg = _ExportDialog(self, batch=False, default_size=self._sizes[i],
                            default_name=_slug(self._titles[i]),
                            fig=self._figs[i])
        if dlg.exec() != QDialog.Accepted:
            return
        opts = dlg.options()
        try:
            self._save(self._figs[i], self._canvases[i], opts["path"], opts,
                       opts["size"])
        except Exception as exc:  # surface any backend/IO error to the user
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        self.statusBar().showMessage(f"Saved {opts['path']}", 5000)

    def export_all(self):
        if not self._figs:
            QMessageBox.warning(self, "No figures", "There is nothing to export.")
            return
        dlg = _ExportDialog(self, batch=True)
        if dlg.exec() != QDialog.Accepted:
            return
        opts = dlg.options()
        folder = Path(opts["path"])
        ext = opts["fmt"]
        saved = 0
        try:
            for i, (fig, canvas, title, size) in enumerate(zip(
                    self._figs, self._canvases, self._titles, self._sizes)):
                path = folder / f"{i + 1:02d}_{_slug(title)}.{ext}"
                self._save(fig, canvas, str(path), opts, size)
                saved += 1
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        self.statusBar().showMessage(f"Saved {saved} figures to {folder}", 5000)

    def _save(self, fig, canvas, path, opts, size):
        """Export ``fig`` via :func:`medusa.plots.save_figure`, keeping the view.

        The figure is briefly resized to the requested export ``size`` and then
        restored to its live (displayed) size so the on-screen canvas is unchanged.
        """
        from medusa.plots import save_figure
        live = tuple(fig.get_size_inches())
        try:
            if size is not None:
                fig.set_size_inches(*size)
            save_figure(fig, path, transparent=opts["transparent"],
                        facecolor=opts["facecolor"], dpi=opts["dpi"],
                        bbox_inches=opts["bbox_inches"],
                        pad_inches=opts["pad_inches"])
        finally:
            fig.set_size_inches(*live)
            canvas.draw_idle()

    # -- maintenance ------------------------------------------------------
    def clear(self):
        """Remove every figure from the browser."""
        while self.stack.count():
            page = self.stack.widget(0)
            self.stack.removeWidget(page)
            page.deleteLater()
        self.filmstrip.clear()
        self._figs.clear()
        self._canvases.clear()
        self._titles.clear()
        self._sizes.clear()
        self._cards.clear()
        self._items.clear()
        self._index = -1
        self._update_counter()


class PlotVisualizer:
    """Headless-friendly handle that owns the ``QApplication`` and the window.

    Reuses an existing ``QApplication`` when one is already running (a process
    holds only one), so it composes with a larger Qt app or a Qt matplotlib
    backend.
    """

    def __init__(self):
        # Theme the whole Qt application from the MEDUSA single source of truth
        # (Fusion + QSS + palette + bundled fonts + app icon); reuses an existing
        # QApplication if one is already running.
        self.app = medusa_style.qt.application(sys.argv)
        self.window = PlotVisualizerWindow()

    def add_figure(self, fig, title: "str | None" = None):
        """Add one figure (optionally titled)."""
        self.window.add_figure(fig, title=title)

    def add_figures(self, figs, titles: "list[str] | None" = None):
        """Add several figures at once; ``titles`` aligns with ``figs`` if given."""
        titles = titles if titles is not None else [None] * len(figs)
        for fig, title in zip(figs, titles):
            self.window.add_figure(fig, title=title)

    def show_figs(self):
        """Show the window and run the event loop (blocks until closed)."""
        self.window.show()
        self.app.exec()

    def clear(self):
        """Remove every figure from the browser."""
        self.window.clear()
