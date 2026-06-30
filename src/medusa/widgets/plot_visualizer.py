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
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QAction, QColor, QFontMetrics, QKeySequence, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
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
    QToolBar,
    QVBoxLayout,
    QWidget,
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
    pal = medusa_style.current_theme().palette
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


class _ExportDialog(QDialog):
    """Single dialog gathering all export options (one figure or the whole set)."""

    def __init__(self, parent, *, batch: bool,
                 default_size: "tuple[float, float] | None" = None,
                 default_name: str = "figure"):
        super().__init__(parent)
        self._batch = batch
        self._default_name = default_name
        if default_size is None:  # fall back to the active style's figure size
            default_size = tuple(mpl.rcParams["figure.figsize"])
        self._bg = QColor(_rc_facecolor())  # default to the style's paper color
        self.setWindowTitle("Export all figures" if batch else "Export figure")
        self.setMinimumWidth(380)
        form = QFormLayout(self)

        self.path_edit = QLineEdit(self)
        browse = QPushButton("Browse…", self)
        browse.clicked.connect(self._browse)
        dest = QHBoxLayout()
        dest.addWidget(self.path_edit)
        dest.addWidget(browse)
        form.addRow("Folder:" if batch else "File:", dest)

        self.fmt = QComboBox(self)
        self.fmt.addItems(_FORMATS)
        form.addRow("Format:", self.fmt)

        self.dpi = QSpinBox(self)
        self.dpi.setRange(30, 1200)
        self.dpi.setValue(_rc_savefig_dpi())
        form.addRow("DPI:", self.dpi)

        if not batch:
            self.width = QDoubleSpinBox(self)
            self.width.setRange(1.0, 300.0)
            self.width.setDecimals(1)
            self.width.setValue(default_size[0] * _CM_PER_IN)
            self.width.setSuffix(" cm")
            self.height = QDoubleSpinBox(self)
            self.height.setRange(1.0, 300.0)
            self.height.setDecimals(1)
            self.height.setValue(default_size[1] * _CM_PER_IN)
            self.height.setSuffix(" cm")
            size = QHBoxLayout()
            size.addWidget(self.width)
            size.addWidget(QLabel("×"))
            size.addWidget(self.height)
            form.addRow("Size:", size)

        self.tight = QCheckBox("Trim margins (tight bounding box)", self)
        self.tight.setChecked(True)
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
        form.addRow(buttons)

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

    def _toggle_bg(self, transparent: bool):
        self.bg_btn.setEnabled(not transparent)

    def _pick_bg(self):
        color = QColorDialog.getColor(self._bg, self, "Background color")
        if color.isValid():
            self._bg = color
            self._refresh_bg_btn()

    def _refresh_bg_btn(self):
        self.bg_btn.setText(self._bg.name())
        self.bg_btn.setStyleSheet(f"background-color: {self._bg.name()};")

    def _accept(self):
        if not self.path_edit.text().strip():
            QMessageBox.warning(self, "Missing destination",
                                "Choose a destination first.")
            return
        self.accept()

    # -- result -----------------------------------------------------------
    def options(self) -> dict:
        transparent = self.transparent.isChecked()
        opts = dict(
            path=self.path_edit.text().strip(),
            fmt=self.fmt.currentText(),
            dpi=self.dpi.value(),
            bbox_inches="tight" if self.tight.isChecked() else None,
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
        bar = QToolBar(self)
        bar.setMovable(False)
        bar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.addToolBar(bar)
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
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        bar.addWidget(spacer)
        self._counter = QLabel("0 / 0", self)
        self._counter.setStyleSheet("padding-right: 10px;")
        bar.addWidget(self._counter)

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
        muted = medusa_style.current_theme().palette.text_secondary
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
                            default_name=_slug(self._titles[i]))
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
                        bbox_inches=opts["bbox_inches"])
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
