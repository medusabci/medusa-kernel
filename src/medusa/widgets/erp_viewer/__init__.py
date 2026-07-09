"""Interactive Qt viewer for Event-Related Potentials (ERPs).

* :class:`ERPViewer` -- public handle (owns the ``QApplication``, blocking ``show``).
* :class:`ERPViewerWindow` -- the ``QMainWindow`` (control panel + temporal/spatial tabs).

The number crunching lives in the Qt-free :class:`~medusa.widgets.erp_viewer.analysis.ERPAnalyzer`
and the drawing in the Qt-free :mod:`~medusa.widgets.erp_viewer.rendering` builders,
which compose the reusable :mod:`medusa.plots` ERP / topography engines. PySide6 is
a core dependency of medusa-kernel.
"""
from medusa.widgets.erp_viewer.erp_viewer import ERPViewer, ERPViewerWindow

__all__ = ["ERPViewer", "ERPViewerWindow"]
