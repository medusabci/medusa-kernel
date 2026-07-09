"""Shared construction for the main-window toolbars in ``medusa.widgets``.

Every viewer/editor window (:class:`~medusa.widgets.settings_tree.TreeViewer`,
:class:`~medusa.widgets.settings_tree.SettingsTreeBuilder`, the ERP viewer, the
plot visualizer, the recording inspector, the time viewers) docks one toolbar at
the top of a :class:`QMainWindow`. This module is the single place that decides
how those toolbars look and behave, so they stay consistent:

* non-movable, text-beside-icon buttons at one shared icon size, and
* a minimum width that stops the buttons collapsing into Qt's ``»`` overflow
  menu when the window is narrowed.

Icons themselves come from ``medusa_style`` at the call site (this module owns no
palette or icon set); it only standardises geometry and behaviour.
"""

from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import QSizePolicy, QToolBar, QWidget

__all__ = [
    "TOOLBAR_ICON_PX",
    "add_main_toolbar",
    "add_toolbar_spacer",
    "add_toolbar_status_label",
    "pin_toolbar_width",
]

#: Icon edge length (px) shared by every main-window toolbar.
TOOLBAR_ICON_PX = 18


def add_main_toolbar(window, name="Actions"):
    """Create, configure, and dock the standard top toolbar on ``window``.

    Returns the :class:`QToolBar` so the caller can add its own actions and
    widgets. Once the toolbar is fully populated, call :func:`pin_toolbar_width`
    on it so its buttons never spill into the ``»`` overflow menu.

    Parameters
    ----------
    window : QMainWindow
        The window to dock the toolbar on.
    name : str
        Toolbar object name (used by Qt for its context-menu entry).
    """
    bar = QToolBar(name, window)
    bar.setMovable(False)
    bar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
    bar.setIconSize(QSize(TOOLBAR_ICON_PX, TOOLBAR_ICON_PX))
    window.addToolBar(bar)
    return bar


def add_toolbar_spacer(bar):
    """Append an expanding spacer so widgets added after it sit flush right.

    Returns the spacer widget (rarely needed by the caller).
    """
    spacer = QWidget(bar)
    spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
    bar.addWidget(spacer)
    return spacer


def add_toolbar_status_label(bar, label):
    """Dock a trailing status/counter ``label`` that clips instead of overflowing.

    A plain :class:`QLabel` on a toolbar forces the whole toolbar into the ``»``
    overflow menu once its (often runtime-updated, arbitrarily long) text no
    longer fits. This makes the label shrink to zero width instead -- so a long
    status string degrades gracefully by clipping while the action buttons stay
    put -- and keeps it out of :func:`pin_toolbar_width`'s measurement (the pin
    then protects only the buttons, never the volatile label). Add it *after*
    :func:`add_toolbar_spacer` so it sits flush right.

    The caller styles the label; this only fixes its size behaviour and docks it.
    """
    label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
    label.setMinimumWidth(0)
    bar.addWidget(label)
    return label


def pin_toolbar_width(bar, margin=8):
    """Stop ``bar`` collapsing its buttons into the ``»`` overflow menu.

    Fixes the toolbar's own minimum width to the width it needs to show
    everything it currently holds. A :class:`QMainWindow`'s minimum width is the
    maximum of its toolbar's and its central widget's minimums, so this only ever
    *raises* the window's floor -- it never reduces a content-driven minimum, and
    it does not matter whether the toolbar is built before or after the central
    widget. Call once, after every action and widget has been added.

    Parameters
    ----------
    bar : QToolBar
        The fully populated toolbar to pin.
    margin : int
        Slack (px) added on top of the toolbar's size hint.
    """
    bar.setMinimumWidth(bar.sizeHint().width() + margin)
