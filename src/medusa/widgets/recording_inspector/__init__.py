"""Master-detail editor for a :class:`~medusa.core.data.recording.Recording`'s
metadata.

Three entry points, mirroring the other viewers:

* :class:`RecordingInspectorWidget` -- the embeddable ``QWidget`` (nav tree +
  per-node pages + summary strip + validate-then-Apply action bar).
* :class:`RecordingInspectorWindow` -- a standalone ``QMainWindow`` with a slim
  toolbar (Revert / Validate / Save…).
* :class:`RecordingInspector` -- a headless-friendly handle that owns the
  ``QApplication`` and a blocking :meth:`~RecordingInspector.show`.

The Qt-free rules (summaries, validation, commit helpers) live in
:mod:`medusa.widgets.recording_inspector.inspect`.
"""

from medusa.widgets.recording_inspector.recording_inspector import (
    RecordingInspector,
    RecordingInspectorWidget,
    RecordingInspectorWindow,
)

__all__ = ["RecordingInspector", "RecordingInspectorWidget",
           "RecordingInspectorWindow"]
