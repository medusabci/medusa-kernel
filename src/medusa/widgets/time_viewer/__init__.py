"""Interactive Qt viewer for multichannel time-series and time-frequency data.

* :class:`TimeViewer` -- public handle (owns the ``QApplication``, blocking ``show``).
* :class:`TimeViewerWindow` -- the ``QMainWindow`` (toolbar, control panel, scrubber).

Renders via the Qt-free engines in :mod:`medusa.plots`; events follow the BIDS
:class:`~medusa.core.data.events.Events` model. PySide6 is a core dependency of medusa-kernel.
"""
from medusa.widgets.time_viewer.time_viewer import TimeViewer, TimeViewerWindow

__all__ = ["TimeViewer", "TimeViewerWindow"]
