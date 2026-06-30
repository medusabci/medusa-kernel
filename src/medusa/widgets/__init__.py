"""Reusable PySide6-based widgets for Kernel consumers.

PySide6 is a core dependency of medusa-kernel, so importing these widgets needs
no extra. To keep headless use light, the non-visual core (``medusa.core`` /
``medusa.signal`` / ``medusa.ml``) never imports Qt; the full Qt GUI
(``QtWidgets``/``QtGui``) lives here, while ``medusa.plots`` loads only
``PySide6.QtCore`` (for medusa-style theming).
"""
# Kernel policy: the visual layer always defaults to the LIGHT theme (see
# medusa.plots). A host (e.g. the dark medusa-platform GUI) can switch afterwards
# with ``medusa_style.use_theme("dark")``.
import medusa_style as _medusa_style

_medusa_style.use_theme("light")

from medusa.widgets.plot_visualizer import PlotVisualizer
from medusa.widgets.settings_tree import SettingsTreeBuilder, TreeViewer
from medusa.widgets.time_viewer import TimeViewer

__all__ = [
    "PlotVisualizer",
    "SettingsTreeBuilder",
    "TreeViewer",
    "TimeViewer",
]
