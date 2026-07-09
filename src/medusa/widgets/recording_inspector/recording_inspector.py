"""Qt editor to explore and update the metadata of a :class:`Recording`.

A master-detail navigator: a left tree mirrors the :class:`Recording` structure
(Identity / Shared sidecar / Streams -> Sidecar - Channels - Sensors / Events /
Experiment) and drives a right pane of purpose-built pages, under a persistent
read-only summary strip and over a validate-then-Apply action bar.

It edits **metadata only** -- never the ``[n_samples x n_channels]`` sample
arrays, and never ``n_channels`` (the channel table is fixed-row). The widget
edits a working **clone** of the caller's recording in place through the real
engine mutators; **Apply** re-validates the whole clone and swaps it into the
live object atomically, so a host holding the reference sees one consistent
update. The Qt-free rules live in :mod:`.inspect`.

Three public entry points, mirroring the other viewers:

* :class:`RecordingInspectorWidget` -- the embeddable ``QWidget``.
* :class:`RecordingInspectorWindow` -- a standalone ``QMainWindow`` (toolbar).
* :class:`RecordingInspector` -- a headless-friendly handle that owns the
  ``QApplication`` and a blocking :meth:`~RecordingInspector.show`.
"""

import json
import sys

import medusa_style
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from medusa.core.data._bids import BIDS_CHANNEL_TYPES
from medusa.core.data.channels import REFERENCE_METHODS, Sensor
from medusa.core.data.signal import Signal as _Signal
from medusa.core.settings_tree import SettingsTree
from medusa.widgets._toolbar import add_main_toolbar, pin_toolbar_width
from medusa.widgets.recording_inspector import inspect as I
from medusa.widgets.settings_tree import SettingsTreeWidget

__all__ = ["RecordingInspectorWidget", "RecordingInspectorWindow",
           "RecordingInspector"]

_WINDOW_SIZE = (1180, 760)     # px, matching the time viewers
_NAV_WIDTH = 260               # px, left navigator default width
_REF_METHOD_NONE = "—"    # em dash = "no reference scheme" (None)
_SENSOR_NONE = "(none)"
_EVENTS_ROW_CAP = 500          # rows rendered in the read-only events table
_ROLE_BASE = int(Qt.UserRole) + 1   # nav item's stable label (survives badge redraws)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _scalar_text(value) -> str:
    """Render a sidecar/metadata scalar for a line edit (``None`` -> empty)."""
    return "" if value is None else str(value)


def _smart_scalar(text: str):
    """Parse edited text back to a scalar (blank -> ``None``; bool/int/float/str)."""
    t = text.strip()
    if t == "":
        return None
    low = t.lower()
    if low in ("true", "false"):
        return low == "true"
    for cast in (int, float):
        try:
            return cast(t)
        except ValueError:
            pass
    return t


def _muted(text: str) -> QLabel:
    """A secondary-colored, word-wrapping info label."""
    label = QLabel(text)
    label.setWordWrap(True)
    label.setStyleSheet(f"color: {medusa_style.current_theme().text_secondary};")
    return label


def _tint(widget, ok: bool, message: str = "") -> None:
    """Red-tint an editor when its content is invalid (theme error color)."""
    color = "" if ok else medusa_style.current_theme().error
    widget.setStyleSheet("" if ok else f"color: {color}; border: 1px solid {color};")
    widget.setToolTip(message if not ok else "")


def _section(title: str) -> "tuple[QGroupBox, QVBoxLayout]":
    """A titled group box and its vertical layout."""
    box = QGroupBox(title)
    layout = QVBoxLayout(box)
    return box, layout


def _readonly_item(text: str) -> QTableWidgetItem:
    item = QTableWidgetItem(text)
    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
    return item


def _fit_table(table: QTableWidget) -> None:
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
    table.horizontalHeader().setStretchLastSection(True)
    table.setEditTriggers(QAbstractItemView.NoEditTriggers)   # cell widgets edit
    table.setSelectionBehavior(QAbstractItemView.SelectRows)
    table.setAlternatingRowColors(True)
    table.verticalHeader().setVisible(False)


# --------------------------------------------------------------------------- #
# Reusable validated editors
# --------------------------------------------------------------------------- #
class _LabelEdit(QLineEdit):
    """A line edit for a BIDS *label* entity; blank means *unset* (``None``).

    Emits :attr:`committed` with the validated value (``str`` or ``None``) only
    when it is valid; tints red otherwise.
    """
    committed = Signal(object)

    def __init__(self, name, value, *, required=False, parent=None):
        super().__init__("" if value is None else str(value), parent)
        self._name = name
        self._required = required
        self.editingFinished.connect(self._on_finished)
        self.textChanged.connect(lambda _t: self._validate(commit=False))

    def _current(self):
        text = self.text().strip()
        return text if text else None

    def _validate(self, commit):
        value = self._current()
        msg = I.check_label(value, self._name, required=self._required)
        _tint(self, msg is None, msg or "")
        if commit and msg is None:
            self.committed.emit(value)
        return msg is None

    def _on_finished(self):
        self._validate(commit=True)


class _IndexEdit(QLineEdit):
    """A line edit for a BIDS *index* entity (``run``); preserves zero-padding."""
    committed = Signal(object)

    def __init__(self, name, value, *, parent=None):
        super().__init__("" if value is None else str(value), parent)
        self._name = name
        self.editingFinished.connect(self._on_finished)
        self.textChanged.connect(lambda _t: self._validate(commit=False))

    def _current(self):
        text = self.text().strip()
        return text if text else None

    def _validate(self, commit):
        value = self._current()
        msg = I.check_index(value, self._name)
        _tint(self, msg is None, msg or "")
        if commit and msg is None:
            self.committed.emit(value)
        return msg is None

    def _on_finished(self):
        self._validate(commit=True)


class _DictEditor(QWidget):
    """An editable view of a free-form dict (participant / scan / experiment / ...).

    Wraps a :class:`SettingsTreeWidget` (the value editor) plus an *add field* row,
    since that widget edits existing leaves but cannot add keys. Values are read
    back with :meth:`values`; call it in a page's ``flush()``. Falls back to a
    read-only JSON view if the dict cannot be projected to a settings tree.
    """

    def __init__(self, data, host, *, parent=None):
        super().__init__(parent)
        self._host = host
        self._data = dict(data) if data else {}
        self._tree = None
        self._fallback = None
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._body = QVBoxLayout()
        self._layout.addLayout(self._body)
        self._build_body()
        self._layout.addWidget(self._add_field_row())

    def _build_body(self):
        while self._body.count():
            item = self._body.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._tree = self._fallback = None
        try:
            self._tree = SettingsTreeWidget(
                SettingsTree.from_dict(self._data), search=False)
            self._body.addWidget(self._tree)
        except Exception:   # non-projectable dict -> honest read-only JSON view
            self._fallback = QPlainTextEdit(
                json.dumps(self._data, indent=2, default=str))
            self._fallback.setReadOnly(True)
            self._body.addWidget(_muted(
                "This metadata can't be edited as a form; shown read-only."))
            self._body.addWidget(self._fallback)

    def _add_field_row(self):
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        name = QLineEdit()
        name.setPlaceholderText("new field")
        value = QLineEdit()
        value.setPlaceholderText("value")
        add = QToolButton()
        add.setIcon(medusa_style.qt.icon("add"))
        add.setToolTip("Add field")
        add.clicked.connect(lambda: self._add(name, value))
        layout.addWidget(name, 1)
        layout.addWidget(value, 1)
        layout.addWidget(add, 0)
        return row

    def _add(self, name_edit, value_edit):
        name = name_edit.text().strip()
        if not name:
            return
        self._data = self.values()
        self._data[name] = _smart_scalar(value_edit.text())
        name_edit.clear()
        value_edit.clear()
        self._build_body()
        self._host.notify_change()

    def values(self) -> dict:
        if self._tree is not None:
            return self._tree.get_values()
        return dict(self._data)


# --------------------------------------------------------------------------- #
# Page base
# --------------------------------------------------------------------------- #
class _Page(QScrollArea):
    """A scrollable page bound to the host + working recording.

    The page expands to fill the right panel. A page whose main content is a
    table / tree editor calls :meth:`fill` on it so it consumes the spare height
    (no wasted blank space, one scrollbar); a form-like page just stacks widgets
    and gets a trailing stretch so they sit at the top.
    """

    def __init__(self, host, working, key=None):
        super().__init__()
        self.host = host
        self.working = working
        self.key = key
        self.setWidgetResizable(True)
        # Grow with the panel so a short page never leaves the lower half blank.
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._content = QWidget()
        self.setWidget(self._content)
        self.body = QVBoxLayout(self._content)
        self._fills = False
        self.build()
        if not self._fills:            # form-like page -> top-align its widgets
            self.body.addStretch(1)

    def fill(self, widget, stretch=1):
        """Add ``widget`` as the page's expanding main content (consumes height)."""
        self.body.addWidget(widget, stretch)
        self._fills = True

    def build(self):   # overridden
        ...

    def flush(self):   # overridden by dict-backed pages
        ...


# --------------------------------------------------------------------------- #
# Identity page
# --------------------------------------------------------------------------- #
class _IdentityPage(_Page):

    def build(self):
        bids = self.working.bids
        box, layout = _section("BIDS identity (filename entities)")
        form = QFormLayout()
        layout.addLayout(form)
        for name in ("subject", "session", "task", "acquisition"):
            editor = _LabelEdit(name, getattr(bids, name),
                                required=(name == "subject"))
            editor.committed.connect(
                lambda value, n=name: self._set_entity(n, value))
            label = "subject *" if name == "subject" else name
            form.addRow(label, editor)
        run_edit = _IndexEdit("run", bids.run)
        run_edit.committed.connect(lambda value: self._set_entity("run", value))
        form.addRow("run", run_edit)

        self._basename = _muted("")
        self._refresh_basename()
        layout.addWidget(self._basename)
        self.body.addWidget(box)

        part_box, part_layout = _section("Participant (participants.tsv)")
        self._participant = _DictEditor(bids.participant, self.host)
        part_layout.addWidget(_muted(
            "Subject-level phenotype: age, sex, handedness, group, ..."))
        part_layout.addWidget(self._participant)
        self.body.addWidget(part_box)

        scan_box, scan_layout = _section("Scan (scans.tsv)")
        self._scan = _DictEditor(bids.scan, self.host)
        scan_layout.addWidget(_muted("Per-scan metadata: acq_time, ..."))
        scan_layout.addWidget(self._scan)
        self.body.addWidget(scan_box)

    def _set_entity(self, name, value):
        setattr(self.working.bids, name, value)
        self._refresh_basename()
        self.host.notify_change()

    def _refresh_basename(self):
        self._basename.setText(
            "Filename preview:  " + (self.working.bids.basename() or "sub-?"))

    def flush(self):
        self.working.bids.participant = self._participant.values()
        self.working.bids.scan = self._scan.values()


# --------------------------------------------------------------------------- #
# Shared sidecar page (recording-wide fields, broadcast)
# --------------------------------------------------------------------------- #
class _SharedSidecarPage(_Page):

    def build(self):
        self.body.addWidget(_muted(
            "Recording-wide sidecar fields (e.g. TaskName, Instructions). Editing "
            "a value here broadcasts it to every stream's _<datatype>.json draft. "
            "Only populated / divergent fields are listed."))
        rows = [r for r in I.shared_sidecar_rows(self.working)
                if r["any_set"] or r["divergent"]]
        table = QTableWidget(len(rows), 3)
        table.setHorizontalHeaderLabels(["Field", "Streams", "Value (broadcast)"])
        _fit_table(table)
        for r, row in enumerate(rows):
            field = row["field"]
            table.setItem(r, 0, _readonly_item(field))
            present = _readonly_item(f"{row['present_in']}/{row['n_streams']}")
            if row["divergent"]:
                present.setText("differs")
                present.setForeground(Qt.red)
            table.setItem(r, 1, present)
            editor = QLineEdit(_scalar_text(row["value"]))
            editor.editingFinished.connect(
                lambda f=field, e=editor: self._broadcast(f, e))
            table.setCellWidget(r, 2, editor)
        self.fill(table)

        add_box, add_layout = _section("Add recording-wide field")
        add_row = QWidget()
        add_hl = QHBoxLayout(add_row)
        add_hl.setContentsMargins(0, 0, 0, 0)
        self._new_name = QComboBox()
        self._new_name.setEditable(True)
        self._new_name.addItems(self._candidate_fields())
        self._new_value = QLineEdit()
        self._new_value.setPlaceholderText("value")
        add_btn = QToolButton()
        add_btn.setIcon(medusa_style.qt.icon("add"))
        add_btn.setToolTip("Broadcast to all streams")
        add_btn.clicked.connect(self._add_field)
        add_hl.addWidget(self._new_name, 1)
        add_hl.addWidget(self._new_value, 1)
        add_hl.addWidget(add_btn, 0)
        add_layout.addWidget(add_row)
        self.body.addWidget(add_box)

    def _candidate_fields(self):
        return [r["field"] for r in I.shared_sidecar_rows(self.working)
                if not r["any_set"]]

    def _broadcast(self, field, editor):
        self.working.set_sidecar(None, **{field: _smart_scalar(editor.text())})
        self.host.notify_change()

    def _add_field(self):
        name = self._new_name.currentText().strip()
        if not name:
            return
        self.working.set_sidecar(None, **{name: _smart_scalar(self._new_value.text())})
        self.host.rebuild_current()
        self.host.notify_change()


# --------------------------------------------------------------------------- #
# Streams overview page
# --------------------------------------------------------------------------- #
class _StreamsOverviewPage(_Page):

    def build(self):
        self.body.addWidget(_muted(
            "Data streams in this run. A stream key doubles as the acq-<key> "
            "entity when two streams share a datatype. Metadata only: streams "
            "cannot be added or removed here (that needs sample data)."))
        keys = list(self.working.data)
        table = QTableWidget(len(keys), 7)
        table.setHorizontalHeaderLabels(
            ["key", "datatype", "channels", "fs (Hz)", "duration (s)",
             "samples", "dtype"])
        _fit_table(table)
        for r, key in enumerate(keys):
            data = self.working.data[key]
            table.setItem(r, 0, _readonly_item(key))
            if isinstance(data, _Signal):
                info = I.stream_info(data)
                cells = [info["datatype"] or "?", str(info["n_channels"]),
                         _scalar_text(info["fs"]),
                         f"{info['duration']:.3f}" if info["duration"] else "n/a",
                         str(info["n_samples"]), info["dtype"]]
            else:
                cells = [data.bids_datatype() or "?", "n/a", "n/a", "n/a", "n/a",
                         type(data).__name__]
            for c, text in enumerate(cells, start=1):
                table.setItem(r, c, _readonly_item(text))
        self._table = table
        self.fill(table)

        rename = QPushButton(medusa_style.qt.icon("edit"),
                             "Rename selected stream…")
        rename.clicked.connect(self._rename_selected)
        self.body.addWidget(rename)

    def _rename_selected(self):
        row = self._table.currentRow()
        if row < 0:
            QMessageBox.information(self, "Rename stream",
                                    "Select a stream row first.")
            return
        old = self._table.item(row, 0).text()
        new, ok = QInputDialog.getText(self, "Rename stream",
                                       f"New key for {old!r}:", text=old)
        if not ok or not new.strip() or new.strip() == old:
            return
        try:
            self.working.rename_data(old, new.strip())
        except (KeyError, ValueError) as exc:
            QMessageBox.warning(self, "Rename failed", str(exc))
            return
        self.host.rebuild_all()   # structure changed (nav keys)
        self.host.notify_change()


# --------------------------------------------------------------------------- #
# Single stream page (read-only info)
# --------------------------------------------------------------------------- #
class _StreamPage(_Page):

    def build(self):
        data = self.working.data[self.key]
        box, layout = _section(f"Stream '{self.key}'")
        if isinstance(data, _Signal):
            info = I.stream_info(data)
            counts = ", ".join(f"{t}: {n}"
                               for t, n in info["type_counts"].items())
            pairs = [
                ("BIDS datatype", info["datatype"] or "n/a"),
                ("BIDS basename", self.working.bids_basename(info["datatype"],
                                                             self.key)),
                ("samples × channels",
                 f"{info['n_samples']} × {info['n_channels']}  "
                 "(n_channels locked)"),
                ("sampling rate", f"{info['fs']} Hz"),
                ("duration", f"{info['duration']:.3f} s"
                 if info["duration"] else "n/a"),
                ("dtype", info["dtype"]),
                ("channel types", counts or "n/a"),
            ]
        else:
            pairs = [("class", type(data).__name__),
                     ("BIDS datatype", data.bids_datatype() or "n/a"),
                     ("editable", "no (non-signal modality; read-only)")]
        layout.addWidget(_kv_grid(pairs))
        layout.addWidget(_muted(
            "Sample arrays are never editable here. Edit this stream's metadata "
            "under its Sidecar / Channels / Sensors nodes."))
        self.body.addWidget(box)


def _kv_grid(pairs) -> QWidget:
    """A two-column read-only key/value grid."""
    widget = QWidget()
    grid = QGridLayout(widget)
    grid.setContentsMargins(0, 0, 0, 0)
    for r, (k, v) in enumerate(pairs):
        key_label = QLabel(f"{k}:")
        key_label.setStyleSheet(
            f"color: {medusa_style.current_theme().text_secondary};")
        grid.addWidget(key_label, r, 0, Qt.AlignRight | Qt.AlignTop)
        value_label = QLabel(str(v))
        value_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        value_label.setWordWrap(True)
        grid.addWidget(value_label, r, 1)
    grid.setColumnStretch(1, 1)
    return widget


# --------------------------------------------------------------------------- #
# Sidecar page (per stream)
# --------------------------------------------------------------------------- #
class _SidecarPage(_Page):

    def build(self):
        signal = self.working.data[self.key]
        sidecar = self.working.sidecars.get(self.key, {})
        datatype = signal.bids_datatype() if isinstance(signal, _Signal) else None
        rows = I.sidecar_rows(datatype, sidecar, signal
                              if isinstance(signal, _Signal) else None)
        n_set = sum(1 for r in rows if r["value"] is not None)
        self.body.addWidget(_muted(
            f"_{datatype or '?'}.json sidecar draft · {n_set}/{len(rows)} "
            "fields set. Grey rows are derived from the signal (read-only; a BIDS "
            "export recomputes them). Requirement level is per the BIDS schema."))
        order = {"required": 0, "recommended": 1, "optional": 2, "extra": 3}
        rows = sorted(rows, key=lambda r: (order.get(r["level"], 9), r["field"]))
        table = QTableWidget(len(rows), 3)
        table.setHorizontalHeaderLabels(["Field", "Level", "Value"])
        _fit_table(table)
        for r, row in enumerate(rows):
            field = row["field"]
            table.setItem(r, 0, _readonly_item(field))
            table.setItem(r, 1, _readonly_item(row["level"]))
            if row["is_derived"]:
                item = _readonly_item(_scalar_text(row["derived_value"])
                                      + "   (derived)")
                item.setForeground(Qt.gray)
                table.setItem(r, 2, item)
            else:
                editor = QLineEdit(_scalar_text(row["value"]))
                if row["level"] == "required" and row["value"] is None:
                    _tint(editor, False, "required field is unset")
                editor.editingFinished.connect(
                    lambda f=field, e=editor: self._set_field(f, e))
                table.setCellWidget(r, 2, editor)
        self.fill(table)

    def _set_field(self, field, editor):
        value = _smart_scalar(editor.text())
        self.working.sidecars[self.key][field] = value
        _tint(editor, not (value is None
                           and self._is_required(field)),
              "required field is unset")
        self.host.notify_change()

    def _is_required(self, field):
        from medusa.core.data._bids import sidecar_fields
        signal = self.working.data[self.key]
        dt = signal.bids_datatype() if isinstance(signal, _Signal) else None
        return sidecar_fields(dt).get(field) == "required"


# --------------------------------------------------------------------------- #
# Channels page (fixed-row montage table)
# --------------------------------------------------------------------------- #
class _ChannelsPage(_Page):

    def build(self):
        signal = self.working.data[self.key]
        cs = signal.channel_set
        summary = I.channelset_summary(cs)
        counts = ", ".join(f"{t}: {n}"
                            for t, n in summary["type_counts"].items())
        self.body.addWidget(_kv_grid([
            ("channels", f"{summary['n_channels']} (locked to signal width)"),
            ("types", counts or "n/a"),
            ("reference scheme", summary["reference_method"] or "n/a")]))
        self.body.addWidget(_muted(
            "One row per data column (channels[i] <-> signal[:, i]). Rows are "
            "fixed: renaming/retyping is allowed, adding/removing is not (that "
            "would change n_channels)."))

        sensor_labels = list(cs.sensors)
        headers = ["label", "ch_type", "unit", "sensor", "ref. scheme",
                   "reference"]
        has_extras = any(vars_extra(c) for c in cs.channels)
        if has_extras:
            headers.append("extras")
        table = QTableWidget(len(cs.channels), len(headers))
        table.setHorizontalHeaderLabels(headers)
        _fit_table(table)
        for r, c in enumerate(cs.channels):
            table.setCellWidget(r, 0, self._label_edit(cs, c))
            table.setCellWidget(r, 1, self._ch_type_combo(c))
            table.setCellWidget(r, 2, self._unit_edit(c))
            table.setCellWidget(r, 3, self._sensor_combo(c, sensor_labels))
            table.setCellWidget(r, 4, self._ref_method_combo(c))
            table.setCellWidget(r, 5, self._reference_edit(cs, c))
            if has_extras:
                table.setItem(r, 6, _readonly_item(
                    ", ".join(f"{k}={v}" for k, v in vars_extra(c).items())))
        self.fill(table)

    def _label_edit(self, cs, channel):
        edit = QLineEdit(channel.label)

        def commit():
            old, new = channel.label, edit.text().strip()
            if not new or new == old:
                edit.setText(old)
                return
            try:
                cs.rename_channel(old, new)
            except ValueError as exc:
                _tint(edit, False, str(exc))
                return
            _tint(edit, True)
            self.host.notify_change()
        edit.editingFinished.connect(commit)
        return edit

    def _ch_type_combo(self, channel):
        combo = QComboBox()
        combo.addItems(BIDS_CHANNEL_TYPES)
        combo.setCurrentText(channel.ch_type)
        combo.currentTextChanged.connect(
            lambda t: (setattr(channel, "ch_type", t), self.host.notify_change()))
        return combo

    def _unit_edit(self, channel):
        edit = QLineEdit("" if channel.unit is None else str(channel.unit))
        edit.editingFinished.connect(
            lambda: (setattr(channel, "unit", edit.text().strip() or None),
                     self.host.notify_change()))
        return edit

    def _sensor_combo(self, channel, sensor_labels):
        combo = QComboBox()
        combo.addItems([_SENSOR_NONE] + sensor_labels)
        combo.setCurrentText(channel.sensor if channel.sensor in sensor_labels
                             else _SENSOR_NONE)
        combo.currentTextChanged.connect(
            lambda t: (setattr(channel, "sensor",
                               None if t == _SENSOR_NONE else t),
                       self.host.notify_change()))
        return combo

    def _ref_method_combo(self, channel):
        combo = QComboBox()
        combo.addItems([_REF_METHOD_NONE, *REFERENCE_METHODS])
        combo.setCurrentText(channel.reference_method or _REF_METHOD_NONE)
        combo.currentTextChanged.connect(
            lambda t: (setattr(channel, "reference_method",
                               None if t == _REF_METHOD_NONE else t),
                       self.host.notify_change()))
        return combo

    def _reference_edit(self, cs, channel):
        edit = QLineEdit(_reference_text(channel.reference))
        edit.setPlaceholderText("sensor label(s), comma-separated")

        def commit():
            value, err = _parse_reference(edit.text(), channel.reference_method,
                                          cs.sensors)
            if err is not None:
                _tint(edit, False, err)
                return
            _tint(edit, True)
            channel.reference = value
            self.host.notify_change()
        edit.editingFinished.connect(commit)
        return edit


def vars_extra(channel):
    from medusa.core.data.channels import _CHANNEL_CORE
    return {k: v for k, v in vars(channel).items() if k not in _CHANNEL_CORE}


def _reference_text(reference):
    if reference is None:
        return ""
    if isinstance(reference, (list, tuple)):
        return ", ".join(str(r) for r in reference)
    return str(reference)


def _parse_reference(text, method, sensors):
    """Parse the reference operand cell -> (value, error). ``None`` clears it."""
    tokens = [t.strip() for t in text.split(",") if t.strip()]
    if method is None or not tokens:
        return None, None
    missing = [t for t in tokens if t not in sensors]
    if missing:
        return None, f"unregistered sensor(s): {missing}"
    if method == "average":
        return (tokens if len(tokens) > 1 else tokens[0]), None
    # common / bipolar: a single operand.
    if len(tokens) != 1:
        return None, f"{method} reference expects a single sensor label"
    return tokens[0], None


# --------------------------------------------------------------------------- #
# Sensors page
# --------------------------------------------------------------------------- #
class _SensorsPage(_Page):

    def build(self):
        signal = self.working.data[self.key]
        cs = signal.channel_set
        summary = I.sensor_summary(cs)
        self.body.addWidget(_kv_grid([
            ("sensors", summary["n_sensors"]),
            ("located", f"{summary['n_located']} / {summary['n_sensors']}"),
            ("dimensionality", "n/a" if summary["dim"] is None
             else f"{summary['dim']}-D"),
            ("coord_system", "set" if summary["coord_system"] else "unset")]))
        if summary["any_2d"]:
            warn = _muted("Some sensors are 2-D; BIDS EEG export needs 3-D "
                          "(topographic plotting still works).")
            warn.setStyleSheet(f"color: {medusa_style.current_theme().warning};")
            self.body.addWidget(warn)
        self.body.addWidget(_muted(
            "Physical transducers (electrodes.tsv). Leave x/y/z blank to mark a "
            "sensor unlocated; a blank z makes it 2-D. Sensors may be added or "
            "removed (a sensor still linked by a channel cannot be removed)."))

        headers = ["", "label", "x", "y", "z", "type", "location", "material",
                   "impedance"]
        table = QTableWidget(len(cs.sensors), len(headers))
        table.setHorizontalHeaderLabels(headers)
        _fit_table(table)
        for r, (label, sensor) in enumerate(list(cs.sensors.items())):
            table.setCellWidget(r, 0, self._remove_btn(cs, label))
            table.setCellWidget(r, 1, self._label_edit(cs, label))
            xe, ye, ze = (self._coord_edit(sensor, i) for i in range(3))
            for e in (xe, ye, ze):
                e._siblings = (xe, ye, ze)
                e._sensor = sensor
            table.setCellWidget(r, 2, xe)
            table.setCellWidget(r, 3, ye)
            table.setCellWidget(r, 4, ze)
            table.setCellWidget(r, 5, self._attr_edit(sensor, "sensor_type"))
            table.setCellWidget(r, 6, self._attr_edit(sensor, "location"))
            table.setCellWidget(r, 7, self._attr_edit(sensor, "material"))
            table.setCellWidget(r, 8, self._impedance_edit(sensor))
        self.fill(table)

        add = QPushButton(medusa_style.qt.icon("add"), "Add sensor…")
        add.clicked.connect(lambda: self._add_sensor(cs))
        self.body.addWidget(add)

        cbox, clayout = _section("coord_system (coordsystem.json)")
        self._coord_system = _DictEditor(cs.coord_system or {}, self.host)
        clayout.addWidget(self._coord_system)
        self.body.addWidget(cbox)

    def _remove_btn(self, cs, label):
        btn = QToolButton()
        btn.setIcon(medusa_style.qt.icon("delete_forever"))
        btn.setToolTip("Remove sensor")
        btn.clicked.connect(lambda: self._remove_sensor(cs, label))
        return btn

    def _label_edit(self, cs, label):
        edit = QLineEdit(label)

        def commit():
            new = edit.text().strip()
            if not new or new == label:
                edit.setText(label)
                return
            try:
                cs.rename_sensor(label, new)
            except ValueError as exc:
                _tint(edit, False, str(exc))
                return
            _tint(edit, True)
            self.host.rebuild_current()   # combos elsewhere reference sensor labels
            self.host.notify_change()
        edit.editingFinished.connect(commit)
        return edit

    def _coord_edit(self, sensor, axis):
        coords = sensor.coordinates
        text = ""
        if coords is not None and axis < coords.shape[0]:
            text = str(float(coords[axis]))
        edit = QLineEdit(text)
        edit.setPlaceholderText("xyz"[axis])
        edit.editingFinished.connect(lambda e=edit: self._commit_coords(e))
        return edit

    def _commit_coords(self, edit):
        xe, ye, ze = edit._siblings
        sensor = edit._sensor
        parsed = []
        ok = True
        for e in (xe, ye, ze):
            t = e.text().strip()
            if t == "":
                parsed.append(None)
                continue
            try:
                parsed.append(float(t))
            except ValueError:
                _tint(e, False, "not a number")
                ok = False
        if not ok:
            return
        for e in (xe, ye, ze):
            _tint(e, True)
        x, y, z = parsed
        if x is None or y is None:
            sensor.coordinates = None            # unlocated
        elif z is None:
            sensor.coordinates = np.array([x, y], dtype=float)   # 2-D
        else:
            sensor.coordinates = np.array([x, y, z], dtype=float)  # 3-D
        self.host.notify_change()

    def _attr_edit(self, sensor, attr):
        edit = QLineEdit("" if getattr(sensor, attr) is None
                         else str(getattr(sensor, attr)))
        edit.editingFinished.connect(
            lambda: (setattr(sensor, attr, edit.text().strip() or None),
                     self.host.notify_change()))
        return edit

    def _impedance_edit(self, sensor):
        edit = QLineEdit("" if sensor.impedance is None else str(sensor.impedance))
        edit.setPlaceholderText("kOhm")

        def commit():
            t = edit.text().strip()
            if t == "":
                sensor.impedance = None
            else:
                try:
                    sensor.impedance = float(t)
                except ValueError:
                    _tint(edit, False, "not a number")
                    return
            _tint(edit, True)
            self.host.notify_change()
        edit.editingFinished.connect(commit)
        return edit

    def _add_sensor(self, cs):
        label, ok = QInputDialog.getText(self, "Add sensor", "Sensor label:")
        label = label.strip()
        if not ok or not label:
            return
        if label in cs.sensors:
            QMessageBox.warning(self, "Add sensor",
                                f"sensor {label!r} already exists.")
            return
        cs.add_sensors(Sensor(label))
        self.host.rebuild_current()
        self.host.notify_change()

    def _remove_sensor(self, cs, label):
        linked = cs.channels_linking_sensor(label)
        if linked:
            QMessageBox.warning(
                self, "Remove sensor",
                f"sensor {label!r} is still linked by channel(s) {linked}; "
                "change those channels first.")
            return
        cs.sensors.pop(label, None)
        self.host.rebuild_current()
        self.host.notify_change()

    def flush(self):
        signal = self.working.data[self.key]
        signal.channel_set.coord_system = self._coord_system.values() or None


# --------------------------------------------------------------------------- #
# Events page (read-only timeline + editable descriptions)
# --------------------------------------------------------------------------- #
class _EventsPage(_Page):

    def build(self):
        ev = self.working.events
        if ev is None:
            self.body.addWidget(_muted("This recording has no events timeline."))
            return
        info = I.events_summary(ev, self.working.signals)
        cols = ", ".join(f"{c} ({d})" for c, d in info["columns"])
        pairs = [
            ("events", info["n_events"]),
            ("columns", cols),
            ("onset range", "n/a" if info["onset_min"] is None
             else f"{info['onset_min']:.3f} – {info['onset_max']:.3f} s"),
            ("recording duration", "n/a" if info["duration"] is None
             else f"{info['duration']:.3f} s")]
        self.body.addWidget(_kv_grid(pairs))
        if info["out_of_window"]:
            warn = _muted(f"{info['out_of_window']} event(s) end after the "
                          "recording duration.")
            warn.setStyleSheet(f"color: {medusa_style.current_theme().warning};")
            self.body.addWidget(warn)

        tbox, tlayout = _section("Timeline (read-only)")
        df = ev.to_dataframe()
        shown = min(len(df), _EVENTS_ROW_CAP)
        table = QTableWidget(shown, len(df.columns))
        table.setHorizontalHeaderLabels(list(df.columns))
        _fit_table(table)
        for r in range(shown):
            for c, col in enumerate(df.columns):
                table.setItem(r, c, _readonly_item(str(df.iloc[r][col])))
        tlayout.addWidget(table)
        if len(df) > shown:
            tlayout.addWidget(_muted(f"Showing first {shown} of {len(df)} rows."))
        self.fill(tbox)

        dbox, dlayout = _section("Column descriptions (events.json)")
        self._descriptions = _DictEditor(ev.descriptions, self.host)
        dlayout.addWidget(self._descriptions)
        self.body.addWidget(dbox)

    def flush(self):
        if self.working.events is not None and hasattr(self, "_descriptions"):
            self.working.events.descriptions = self._descriptions.values()


# --------------------------------------------------------------------------- #
# Experiment page
# --------------------------------------------------------------------------- #
class _ExperimentPage(_Page):

    def build(self):
        exp = self.working.experiment
        kind, detail = I.experiment_kind(exp)
        self._kind = kind
        if kind == "none":
            self.body.addWidget(_muted("No experiment metadata attached."))
            create = QPushButton(medusa_style.qt.icon("add"),
                                 "Create experiment dict")
            create.clicked.connect(self._create_dict)
            self.body.addWidget(create)
            return
        if kind == "component":
            self._component = exp
            self.body.addWidget(_muted(
                f"experiment is a {detail} component. Editing its serialized form; "
                "it is rebuilt via from_serializable_obj on Apply (kept read-only "
                "if that fails)."))
            self._editor = _DictEditor(exp.to_serializable_obj(), self.host)
            self.fill(self._editor)
            return
        if kind == "dict":
            self.body.addWidget(_muted("Free-form paradigm / setup metadata."))
            self._editor = _DictEditor(exp, self.host)
            self.fill(self._editor)
            return
        self.body.addWidget(_muted(
            f"experiment is a {detail}; not editable as a form (read-only)."))

    def _create_dict(self):
        self.working.set_experiment({})
        self.host.rebuild_current()
        self.host.notify_change()

    def flush(self):
        if self._kind == "dict":
            self.working.set_experiment(self._editor.values())
        elif self._kind == "component":
            # Best-effort round-trip; on failure keep the original component
            # silently (flush runs on every refresh -- no dialog spam). The page
            # header already states it may stay read-only.
            try:
                rebuilt = type(self._component).from_serializable_obj(
                    self._editor.values())
                self.working.set_experiment(rebuilt)
            except Exception:
                self.working.set_experiment(self._component)


# --------------------------------------------------------------------------- #
# The widget
# --------------------------------------------------------------------------- #
class RecordingInspectorWidget(QWidget):
    """Embeddable master-detail editor for a :class:`Recording`'s metadata.

    Parameters
    ----------
    recording : Recording, optional
        The recording to edit. Its metadata is committed **in place** on Apply;
        signal arrays are never touched. Pass ``None`` and call
        :meth:`set_recording` later.
    parent : QWidget, optional

    Signals
    -------
    dirty(bool)
        Emitted when the pending edits differ from the last applied state.
    applied(object)
        Emitted with the (mutated) recording after a successful Apply.
    validated(list)
        Emitted with the current list of :class:`.inspect.Issue` on every change.
    """
    dirty = Signal(bool)
    applied = Signal(object)
    validated = Signal(list)

    _PAGE_BUILDERS = {
        "identity": lambda h, k: _IdentityPage(h, h._working),
        "shared": lambda h, k: _SharedSidecarPage(h, h._working),
        "streams": lambda h, k: _StreamsOverviewPage(h, h._working),
    }

    def __init__(self, recording=None, *, parent=None):
        super().__init__(parent)
        self._live = None
        self._working = None
        self._snapshot = None
        self._current_page = None
        self._issues = []
        self._dirty = False

        outer = QVBoxLayout(self)
        self._summary = QLabel("")
        self._summary.setWordWrap(True)
        self._summary.setTextInteractionFlags(Qt.TextSelectableByMouse)
        strip = QFrame()
        strip.setFrameShape(QFrame.StyledPanel)
        strip_layout = QVBoxLayout(strip)
        strip_layout.setContentsMargins(8, 6, 8, 6)
        strip_layout.addWidget(self._summary)
        outer.addWidget(strip)

        splitter = QSplitter(Qt.Horizontal)
        self._nav = QTreeWidget()
        self._nav.setHeaderHidden(True)
        self._nav.setColumnCount(1)
        self._nav.setMinimumWidth(180)
        self._nav.currentItemChanged.connect(self._on_nav_changed)
        splitter.addWidget(self._nav)
        self._page_host = QWidget()
        self._page_layout = QVBoxLayout(self._page_host)
        self._page_layout.setContentsMargins(0, 0, 0, 0)
        splitter.addWidget(self._page_host)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([_NAV_WIDTH, _WINDOW_SIZE[0] - _NAV_WIDTH])
        outer.addWidget(splitter, 1)

        action_bar = QHBoxLayout()
        self._status = QLabel("")
        self._status.setWordWrap(True)
        self._revert_btn = QPushButton(medusa_style.qt.icon("refresh"), "Revert")
        self._revert_btn.clicked.connect(self.revert)
        self._apply_btn = QPushButton("Apply")
        self._apply_btn.clicked.connect(self.apply)
        action_bar.addWidget(self._status, 1)
        action_bar.addWidget(self._revert_btn, 0)
        action_bar.addWidget(self._apply_btn, 0)
        outer.addLayout(action_bar)

        if recording is not None:
            self.set_recording(recording)

    # -- public API ---------------------------------------------------------
    def set_recording(self, recording):
        """Load ``recording`` for editing (replaces any current one)."""
        self._live = recording
        self._working = I.clone(recording)
        self._snapshot = I.clone(recording)
        self.rebuild_all()

    def get_recording(self):
        """Return the live recording (mutated in place by :meth:`apply`)."""
        return self._live

    def is_dirty(self):
        """Whether there are un-applied edits."""
        self._flush_current()
        return I.metadata_state(self._working) != I.metadata_state(self._snapshot)

    def validate(self):
        """Return the current list of :class:`.inspect.Issue` (no write)."""
        self._flush_current()
        return I.validate_recording(self._working)

    def apply(self):
        """Validate the working clone and commit it into the live recording.

        Returns ``True`` on success; on any error the live object is left
        untouched and a message is shown.
        """
        self._flush_current()
        try:
            self._working.bids = I.rebuild_bids(self._working.bids)
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid identity", str(exc))
            return False
        issues = I.validate_recording(self._working)
        errors = [i for i in issues if i.severity == "error"]
        if errors:
            QMessageBox.warning(
                self, "Cannot apply",
                "Resolve these first:\n\n"
                + "\n".join(f"• {i.message}" for i in errors))
            return False
        I.swap_into(self._live, self._working)
        # Re-establish isolation: the swap made ``live`` share ``working``'s
        # objects, so a fresh clone is needed or further edits (and a later
        # Revert) would leak into / fail to roll back the live object.
        self._working = I.clone(self._live)
        self._snapshot = I.clone(self._live)
        self.rebuild_all()
        self.applied.emit(self._live)
        return True

    def revert(self):
        """Discard pending edits, restoring the last applied state."""
        self._working = I.clone(self._snapshot)
        self.rebuild_all()

    # -- host callbacks used by pages --------------------------------------
    def notify_change(self):
        """Light refresh after an in-page edit (dirty / validation / summary)."""
        self._refresh_state()

    def rebuild_current(self):
        """Rebuild the visible page from the working clone (structural change)."""
        item = self._nav.currentItem()
        if item is not None:
            self._show_page(item.data(0, Qt.UserRole), flush=False)
        self._refresh_state()

    def rebuild_all(self):
        """Rebuild the navigator and the visible page (load / revert / rename)."""
        selected = None
        item = self._nav.currentItem()
        if item is not None:
            selected = item.data(0, Qt.UserRole)
        self._build_nav()
        target = self._find_item(selected) or (
            self._nav.topLevelItem(0) if self._nav.topLevelItemCount() else None)
        if target is not None:
            self._nav.setCurrentItem(target)   # triggers _on_nav_changed
        self._refresh_state()

    # -- navigator ----------------------------------------------------------
    def _build_nav(self):
        self._nav.blockSignals(True)
        self._nav.clear()
        for node in I.outline(self._working):
            self._nav.addTopLevelItem(self._nav_item(node))
        self._nav.expandAll()
        self._nav.blockSignals(False)

    def _nav_item(self, node):
        base = node.label + (f"  [{node.badge}]" if node.badge else "")
        item = QTreeWidgetItem([base])
        item.setData(0, Qt.UserRole, node.node_id)
        item.setData(0, _ROLE_BASE, base)       # stable label for badge redraws
        for child in node.children:
            item.addChild(self._nav_item(child))
        return item

    def _find_item(self, node_id):
        if node_id is None:
            return None
        stack = [self._nav.topLevelItem(i)
                 for i in range(self._nav.topLevelItemCount())]
        while stack:
            item = stack.pop()
            if item.data(0, Qt.UserRole) == node_id:
                return item
            stack.extend(item.child(i) for i in range(item.childCount()))
        return None

    def _on_nav_changed(self, current, _previous):
        if current is not None:
            self._show_page(current.data(0, Qt.UserRole), flush=True)

    def _show_page(self, node_id, *, flush):
        if flush:
            self._flush_current()
        page = self._make_page(node_id)
        while self._page_layout.count():
            old = self._page_layout.takeAt(0)
            if old.widget():
                old.widget().deleteLater()
        self._current_page = page
        if page is not None:
            self._page_layout.addWidget(page)

    def _make_page(self, node_id):
        if node_id in self._PAGE_BUILDERS:
            return self._PAGE_BUILDERS[node_id](self, None)
        prefix, _, key = node_id.partition(":")
        pages = {"stream": _StreamPage, "sidecar": _SidecarPage,
                 "channels": _ChannelsPage, "sensors": _SensorsPage}
        if prefix == "events":
            return _EventsPage(self, self._working)
        if prefix == "experiment":
            return _ExperimentPage(self, self._working)
        if prefix in pages:
            return pages[prefix](self, self._working, key)
        return None

    def _flush_current(self):
        if self._current_page is not None:
            self._current_page.flush()

    # -- state refresh ------------------------------------------------------
    def _refresh_state(self):
        if self._working is None:
            return
        self._flush_current()
        self._issues = I.validate_recording(self._working)
        self._update_summary()
        self._update_nav_badges()
        self._update_action_bar()
        dirty = (I.metadata_state(self._working)
                 != I.metadata_state(self._snapshot))
        if dirty != self._dirty:
            self._dirty = dirty
            self.dirty.emit(dirty)
        self.validated.emit(self._issues)

    def _update_summary(self):
        s = I.recording_summary(self._working)
        chips = " · ".join(
            f"{st['key']}({st['datatype'] or '?'}"
            + (f",{st['n_channels']}ch,{st['fs']:g}Hz" if st["is_signal"] else "")
            + ")" for st in s["streams"]) or "no streams"
        events = "no events" if s["n_events"] is None else f"events {s['n_events']}"
        exp = s["experiment_kind"] if s["experiment_detail"] is None \
            else f"{s['experiment_kind']}·{s['experiment_detail']}"
        dot = "  ● unsaved" if self._dirty else ""
        self._summary.setText(
            f"{s['basename']}   —   streams: {chips}   —   {events}"
            f"   —   exp {exp}   —   schema {s['schema_version']}{dot}")

    def _update_nav_badges(self):
        by_node = {}
        for issue in self._issues:
            by_node.setdefault(issue.node_id, []).append(issue)
        stack = [self._nav.topLevelItem(i)
                 for i in range(self._nav.topLevelItemCount())]
        while stack:
            item = stack.pop()
            node_id = item.data(0, Qt.UserRole)
            issues = by_node.get(node_id, [])
            has_error = any(i.severity == "error" for i in issues)
            has_warn = any(i.severity == "warning" for i in issues)
            mark = "  ✗" if has_error else ("  ⚠" if has_warn else "")
            base = item.data(0, _ROLE_BASE) or item.text(0)
            item.setText(0, base + mark)
            stack.extend(item.child(i) for i in range(item.childCount()))

    def _update_action_bar(self):
        errors = [i for i in self._issues if i.severity == "error"]
        warnings = [i for i in self._issues if i.severity == "warning"]
        theme = medusa_style.current_theme()
        if errors:
            self._status.setText("✗ " + errors[0].message
                                 + (f"  (+{len(errors) - 1} more)"
                                    if len(errors) > 1 else ""))
            self._status.setStyleSheet(f"color: {theme.error};")
        elif warnings:
            self._status.setText("⚠ " + warnings[0].message
                                 + (f"  (+{len(warnings) - 1} more)"
                                    if len(warnings) > 1 else ""))
            self._status.setStyleSheet(f"color: {theme.warning};")
        else:
            self._status.setText("✓ no issues")
            self._status.setStyleSheet(f"color: {theme.text_secondary};")
        self._apply_btn.setEnabled(bool(self._working) and not errors
                                   and self._dirty)
        self._revert_btn.setEnabled(self._dirty)


# --------------------------------------------------------------------------- #
# Window + handle
# --------------------------------------------------------------------------- #
class RecordingInspectorWindow(QMainWindow):
    """Standalone window hosting a :class:`RecordingInspectorWidget`.

    Adds a slim toolbar (Revert / Validate / Expand / Collapse / Save…).
    """

    def __init__(self, recording=None, *, parent=None):
        super().__init__(parent)
        self.widget = RecordingInspectorWidget(recording)
        self.setCentralWidget(self.widget)
        self.setWindowTitle("Recording inspector")
        medusa_style.qt.set_window_icon(self)
        self.resize(*_WINDOW_SIZE)
        self._build_toolbar()
        self.widget.applied.connect(
            lambda rec: self.statusBar().showMessage("Applied metadata edits.",
                                                     4000))

    def _build_toolbar(self):
        bar = add_main_toolbar(self)
        ic = medusa_style.qt.icon
        bar.addAction(ic("refresh"), "Revert", self.widget.revert)
        bar.addAction(ic("science"), "Validate", self._validate)
        bar.addSeparator()
        bar.addAction(ic("unfold_more"), "Expand", self.widget._nav.expandAll)
        bar.addAction(ic("unfold_less"), "Collapse", self.widget._nav.collapseAll)
        bar.addSeparator()
        bar.addAction(ic("save_as"), "Save…", self._save)
        # Keep every action visible when the window is narrowed.
        pin_toolbar_width(bar)

    def _validate(self):
        issues = self.widget.validate()
        if not issues:
            QMessageBox.information(self, "Validate", "No issues found.")
            return
        lines = "\n".join(f"[{i.severity}] {i.node_id}: {i.message}"
                          for i in issues)
        QMessageBox.information(self, "Validation", lines)

    def _save(self):
        """Apply pending edits, then write the recording to disk."""
        if self.widget.is_dirty() and not self.widget.apply():
            return
        rec = self.widget.get_recording()
        path, _ = QFileDialog.getSaveFileName(
            self, "Save recording", rec.bids.basename() or "recording",
            "Recording (*.bson *.json *.mat *.h5 *.hdf5)")
        if not path:
            return
        try:
            rec.save(path)
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return
        self.statusBar().showMessage(f"Saved {path}", 5000)


class RecordingInspector:
    """Headless-friendly handle that owns the ``QApplication`` and the window.

    Reuses an existing ``QApplication`` when one is running, so it composes with a
    larger Qt app.

    Parameters
    ----------
    recording : Recording, optional
        The recording to inspect / edit (see :class:`RecordingInspectorWindow`).
    """

    def __init__(self, recording=None, **kwargs):
        self.app = medusa_style.qt.application(sys.argv)
        self.window = RecordingInspectorWindow(recording, **kwargs)

    @property
    def recording(self):
        """The live recording (mutated in place by Apply)."""
        return self.window.widget.get_recording()

    def show(self):
        """Show the window and run the event loop (blocks until closed)."""
        self.window.show()
        self.app.exec()
