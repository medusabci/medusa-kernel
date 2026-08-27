"""Settings-tree GUIs — ``medusa.widgets.settings_tree``.

Run:  python examples/settings_tree_usage.py

Two tools over a :class:`medusa.core.settings_tree.SettingsTree`:

  * :class:`TreeViewer`          — the *end-user* value editor (search, reset,
    and add/remove for both list values **and group-lists**, context menu);
  * :class:`SettingsTreeBuilder` — the *developer* schema-authoring tool.

The schema below includes a **group-list** (``filterbank``): a variable-length list of
same-schema *filter groups*. In the viewer each filter renders as its own editable
sub-form; use the group-list's ``+ Add element`` (or ``Ins``) to append a filter, the
per-row ``-`` (or ``Del``) to remove one, and ``Reset`` to restore the defaults. It
projects to a plain list of dicts, so consuming code reads it like any list value.

Both tools share the active ``medusa_style`` theme (themed SVG icons, light by default).
"""
import medusa_style

from medusa.core.settings_tree import SettingsTree
from medusa.widgets.settings_tree import SettingsTreeBuilder, TreeViewer

_BAND_TYPES = ["bandpass", "bandstop", "lowpass", "highpass"]

# --------------------------------------------------------------------------- #
# A schema: a scalar, a single "notch" group, and a "filterbank" GROUP-LIST.
# --------------------------------------------------------------------------- #
settings = SettingsTree()
settings.add_item("update_rate", value=0.2, info="Update rate (s)", value_range=[0, None])

notch = settings.add_group("notch", info="Line-noise notch (bandstop)")
notch.add_item("enabled", value=True, info="Apply the notch")
notch.add_item("cutoff", value=[48.0, 52.0], info="Bandstop edges (Hz)")
notch.add_item("order", value=4, value_range=[1, None], info="Filter order")

# The group-list: populate the element *template* once, then seed default elements.
filterbank = settings.add_group_list(
    "filterbank", info="Parallel sub-band filters — add/remove filters in the viewer")
element = filterbank.element
element.add_item("filt_type", value="iir", value_options=["iir", "fir"],
                 info="Filter family")
element.add_item("band_type", value="bandpass", value_options=_BAND_TYPES,
                 info="Band type")
element.add_item("cutoff", value=[1.0, 70.0], info="Cutoff frequencies (Hz)")
element.add_item("order", value=5, value_range=[1, None], info="Filter order")
filterbank.add_element()                                       # a band-pass 1–70
filterbank.add_element({"band_type": "bandpass", "cutoff": [8.0, 15.0], "order": 4})
settings.snapshot_defaults()          # baseline the group-list default (as Configurable does)

# A terminal view of the whole schema: 'values' is the plain configuration,
# 'full' adds constraints, editor hints, help text and the edited defaults.
settings.print_tree(title="Schema (values)")
print()
settings.print_tree(detail="full", title="Schema (full)")
print()

# The group-list projects to a plain list of dicts -- read like any list value.
print("filterbank config (a list of dicts):")
for spec in settings.to_dict()["filterbank"]:
    print("   ", spec)
print("\nEdit the filters in the viewer, then it reads back the same shape via "
      "viewer.get_settings().to_dict().")

# --------------------------------------------------------------------------- #
# A QApplication must exist before the windows are built; medusa_style themes it.
# --------------------------------------------------------------------------- #
app = medusa_style.qt.application()
viewer = TreeViewer(settings)              # end-user value editor (edit/add/remove filters)
builder = SettingsTreeBuilder(settings)    # developer schema authoring

viewer.show()
builder.show()
app.exec()
