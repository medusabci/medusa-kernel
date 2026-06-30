"""Settings-tree GUIs — ``medusa.widgets.settings_tree``.

Run:  python examples/settings_tree_usage.py

Two tools over a :class:`medusa.core.settings_tree.SettingsTree`:

  * :class:`TreeViewer`        — the *end-user* value editor (search, reset,
    list-element add/remove, context menu);
  * :class:`SettingsTreeBuilder` — the *developer* schema-authoring tool
    (add/remove/reorder items and groups, edit every field).

Both share the active ``medusa_style`` theme (themed SVG icons, light by
default). Set ``MEDUSA_EXAMPLE_HEADLESS=1`` (with ``QT_QPA_PLATFORM=offscreen``)
to build the windows without entering the blocking Qt event loop.
"""
import os

import medusa_style

from medusa.core.settings_tree import SettingsTree
from medusa.widgets.settings_tree import SettingsTreeBuilder, TreeViewer

HEADLESS = bool(os.environ.get("MEDUSA_EXAMPLE_HEADLESS"))

# A small schema: a scalar plus a "frequency filter" group with a list cutoff.
settings = SettingsTree()
settings.add_item("update_rate", value=0.2,
                  info="Update rate (s) of the plot", value_range=[0, None])
freq = settings.add_group("frequency_filter", info="Real-time IIR filter")
freq.add_item("apply", value=True, info="Apply the filter")
freq.add_item("type", value="highpass",
              value_options=["highpass", "lowpass", "bandpass", "stopband"],
              info="Filter type")
freq.add_item("cutoff_freq", value=[1.0, 30.0],
              element_schema={"value_range": [0, None]},
              info="One cutoff for high/low-pass, two for band/stop")
freq.add_item("order", value=5, value_range=[1, None], info="Filter order")

# A QApplication must exist before the windows are built; medusa_style themes it.
app = medusa_style.qt.application()
viewer = TreeViewer(settings)              # end-user value editor
builder = SettingsTreeBuilder(settings)    # developer schema authoring

if not HEADLESS:
    viewer.show()
    builder.show()
    app.exec()
else:
    print("headless: built TreeViewer + SettingsTreeBuilder (event loop skipped)")
