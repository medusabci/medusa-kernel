"""Tests for the shared main-window toolbar helper (Qt, offscreen).

Verifies the single source of truth that keeps every ``medusa.widgets``
main-window toolbar consistent: standard construction (non-movable,
text-beside-icon, one icon size) and a min-width pin that stops buttons
collapsing into Qt's ``»`` overflow menu without ever reducing a widget's
content-driven minimum width.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QLabel,
    QMainWindow,
    QToolBar,
    QToolButton,
    QWidget,
)

from medusa.widgets._toolbar import (  # noqa: E402
    TOOLBAR_ICON_PX,
    add_main_toolbar,
    add_toolbar_spacer,
    add_toolbar_status_label,
    pin_toolbar_width,
)


def _visible_extension(bar):
    return [e for e in bar.children()
            if e.metaObject().className() == "QToolBarExtension" and e.isVisible()]


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _window(content_min=0, order="toolbar_first", spacer=False, n=5):
    """Build a QMainWindow with a pinned toolbar and a min-width central widget."""
    win = QMainWindow()

    def build_toolbar():
        bar = add_main_toolbar(win)
        for i in range(n):
            bar.addAction(f"Act {i}")
        if spacer:
            add_toolbar_spacer(bar)
        pin_toolbar_width(bar)
        return bar

    def build_central():
        central = QWidget()
        central.setMinimumWidth(content_min)
        win.setCentralWidget(central)

    if order == "toolbar_first":
        bar = build_toolbar()
        build_central()
    else:
        build_central()
        bar = build_toolbar()
    win.show()
    return win, bar


class TestConstruction:
    def test_standard_configuration(self, qapp):
        win = QMainWindow()
        bar = add_main_toolbar(win)
        assert not bar.isMovable()
        assert bar.toolButtonStyle() == Qt.ToolButtonTextBesideIcon
        assert bar.iconSize().width() == TOOLBAR_ICON_PX
        assert bar.iconSize().height() == TOOLBAR_ICON_PX
        assert bar in win.findChildren(QToolBar)     # docked on the window


class TestPin:
    def test_prevents_overflow(self, qapp):
        # Pinned window cannot be narrowed below the toolbar's needed width, so
        # the buttons can never collapse into the "»" extension menu.
        win, bar = _window()
        assert bar.minimumWidth() >= bar.sizeHint().width()
        assert win.minimumSizeHint().width() >= bar.sizeHint().width()

    def test_preserves_larger_content_min(self, qapp):
        # A content-driven minimum wider than the toolbar must survive intact.
        big = 1600
        win, bar = _window(content_min=big)
        assert bar.sizeHint().width() < big           # toolbar is the smaller one
        assert win.minimumSizeHint().width() >= big   # content floor not reduced

    @pytest.mark.parametrize("order", ["toolbar_first", "central_first"])
    @pytest.mark.parametrize("spacer", [False, True])
    def test_order_and_spacer_independent(self, qapp, order, spacer):
        # Works whether the toolbar is built before or after the central widget,
        # and whether or not it has an expanding spacer.
        win, bar = _window(order=order, spacer=spacer)
        assert win.minimumSizeHint().width() >= bar.sizeHint().width()


class TestStatusLabel:
    def test_long_label_clips_instead_of_overflowing(self, qapp):
        # A long, runtime-set status label must NOT push the toolbar into the "»"
        # overflow menu: it clips to zero width while the buttons stay visible.
        win = QMainWindow()
        bar = add_main_toolbar(win)
        for i in range(3):
            bar.addAction(f"Act {i}")
        add_toolbar_spacer(bar)
        add_toolbar_status_label(bar, QLabel("status " * 40, win))  # very long
        pin_toolbar_width(bar)
        win.setCentralWidget(QWidget())
        win.show()
        qapp.processEvents()
        win.resize(120, 400)                          # narrower than the label wants
        qapp.processEvents()
        assert not _visible_extension(bar)            # no "»" appeared
        shown = [b.text() for b in bar.findChildren(QToolButton)
                 if b.isVisible() and b.text()]
        assert len(shown) == 3                        # every action still visible

    def test_label_excluded_from_pin(self, qapp):
        # The pin protects the buttons, not the volatile label: a huge label must
        # not inflate the toolbar's pinned minimum width.
        win = QMainWindow()
        bar = add_main_toolbar(win)
        bar.addAction("Only action")
        add_toolbar_spacer(bar)
        add_toolbar_status_label(bar, QLabel("x" * 300, win))
        pin_toolbar_width(bar)
        assert bar.minimumWidth() < 300               # label width did not leak in
