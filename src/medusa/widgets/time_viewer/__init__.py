"""Interactive Qt viewers for multichannel time-series and time-frequency data.

Two sibling viewers share a common base (toolbar, control panel, overview
scrubber, keyboard navigation, export) and differ only in the view they present:

* :class:`TimeLineViewer` — stacked **traces**, optionally overlaying
  transformations of the *same* signal (e.g. raw vs filtered), with an amplitude
  (gain) control.
* :class:`TimeHeatmapViewer` — a per-channel **heatmap** (e.g. a spectrogram per
  channel), with colormap / contrast controls and a configurable number of
  per-band frequency ticks.

Each ``*Viewer`` is the public handle (owns the ``QApplication``, blocking
``show``); the ``*ViewerWindow`` is the underlying ``QMainWindow``. Rendering is
delegated to the Qt-free engines in :mod:`medusa.plots`; events follow the BIDS
:class:`~medusa.core.data.events.Events` model. PySide6 is a core dependency of
medusa-kernel.
"""
from medusa.widgets.time_viewer.heatmap import (
    TimeHeatmapViewer,
    TimeHeatmapViewerWindow,
)
from medusa.widgets.time_viewer.timeline import (
    TimeLineViewer,
    TimeLineViewerWindow,
)

__all__ = [
    "TimeLineViewer",
    "TimeLineViewerWindow",
    "TimeHeatmapViewer",
    "TimeHeatmapViewerWindow",
]
