"""Developer-mode editor for authoring a :class:`SettingsTree` *schema*.

Where :class:`~medusa.widgets.settings_tree.tree_viewer.SettingsTreeWidget`
(in :mod:`medusa.widgets.settings_tree.tree_viewer`) lets an *end user* edit
values, :class:`SettingsTreeBuilder` lets a *developer* design the schema
itself: add/remove/reorder items and groups and edit every field of each item
(key, type, default, info, numeric range, choice options, list element type).
Build from scratch, open an existing schema (JSON), save it, and preview it in
the end-user editor.

PySide6 is a core dependency of medusa-kernel.

Example
-------
>>> from medusa.widgets.settings_tree import SettingsTreeBuilder   # doctest: +SKIP
>>> from PySide6.QtWidgets import QApplication                     # doctest: +SKIP
>>> app = QApplication([])                                         # doctest: +SKIP
>>> builder = SettingsTreeBuilder()                                # doctest: +SKIP
>>> builder.show(); app.exec()                                     # doctest: +SKIP
>>> schema = builder.to_settings()       # the authored SettingsTree
"""


import medusa_style
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLineEdit,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from medusa.core.settings_tree import SettingsTree
from medusa.widgets._toolbar import add_main_toolbar, pin_toolbar_width
from medusa.widgets.settings_tree.tree_viewer import (
    TreeViewer,
    _build_editor,
    _read_editor,
)

__all__ = ["SettingsTreeBuilder"]

_WINDOW_SIZE = (1040, 620)   # default builder window size (px)
_EDITOR_BOX_H = 80           # fixed height (px) of the options / list-values boxes

# Developer-facing "type" of an item, and the editor each one maps to.
_TYPES = ["Group", "Group list", "Boolean", "Integer", "Float", "Text", "Choice", "List"]
_TYPE_TO_FORMAT = {"Boolean": "checkbox", "Integer": "spinbox",
                   "Float": "doublespinbox", "Text": "lineedit",
                   "Choice": "combobox"}
_SCALAR_TYPES = ["Text", "Integer", "Float", "Boolean"]  # for options/elements
_TYPE_DEFAULT = {"Boolean": False, "Integer": 0, "Float": 0.0, "Text": "",
                 "Choice": None, "List": []}


def _parse_scalar(text, type_name):
    """Parse one line of text into a value of the given developer type."""
    text = text.strip()
    if type_name == "Integer":
        return int(text)
    if type_name == "Float":
        return float(text)
    if type_name == "Boolean":
        return text.lower() in ("1", "true", "yes", "on")
    return text


def _scalar_type_of(value):
    if isinstance(value, bool):
        return "Boolean"
    if isinstance(value, int):
        return "Integer"
    if isinstance(value, float):
        return "Float"
    return "Text"


class SettingsTreeBuilder(QMainWindow):
    """A GUI to author/edit a :class:`SettingsTree` schema.

    Parameters
    ----------
    settings : SettingsTree, optional
        An existing schema to load for editing; a new empty schema otherwise.
    parent : QWidget, optional
        Qt parent.
    """

    def __init__(self, settings=None, parent=None):
        super().__init__(parent)
        self._tree = settings if isinstance(settings, SettingsTree) \
            else SettingsTree()
        self._node_of = {}      # QTreeWidgetItem -> node dict
        self._item_of = {}      # id(node)        -> QTreeWidgetItem
        self._parent_of = {}    # id(node)        -> parent node dict
        self._current_node = None
        self._loading = False
        self._preview = None

        self.setWindowTitle("Settings tree builder (developer mode)")
        self.resize(*_WINDOW_SIZE)
        self._build_toolbar()
        self._build_ui()
        self._rebuild_tree()

    # -- public API ---------------------------------------------------------
    def to_settings(self):
        """Return the authored schema as a :class:`SettingsTree`."""
        self._apply()
        return self._tree

    def load(self, settings):
        """Replace the edited schema with ``settings`` (a SettingsTree)."""
        self._tree = settings
        self._current_node = None
        self._rebuild_tree()

    # -- construction -------------------------------------------------------
    def _build_toolbar(self):
        bar = add_main_toolbar(self)
        # text, slot, medusa-style themed-icon name
        for text, slot, icon in (("New", self._new, "add"),
                                  ("Open…", self._open, "folder"),
                                  ("Save…", self._save, "save_as"),
                                  ("Preview", self._preview_, "visibility")):
            action = QAction(medusa_style.qt.icon(icon), text, self)
            action.triggered.connect(slot)
            bar.addAction(action)
        # Keep every action visible: pin the toolbar so its buttons never spill
        # into a "»" overflow menu when the window is narrowed.
        pin_toolbar_width(bar)

    def _build_ui(self):
        splitter = QSplitter()
        self.setCentralWidget(splitter)
        splitter.addWidget(self._build_structure_panel())
        splitter.addWidget(self._build_property_panel())
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        self.statusBar()

    def _build_structure_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)

        self.tree_widget = QTreeWidget()
        self.tree_widget.setHeaderLabels(["Key", "Type"])
        self.tree_widget.header().setSectionResizeMode(QHeaderView.Stretch)
        self.tree_widget.currentItemChanged.connect(self._on_selection_changed)
        layout.addWidget(self.tree_widget)

        buttons = QHBoxLayout()
        # text, slot, medusa-style themed-icon name, tooltip
        for text, slot, icon, tip in (
                ("Item", lambda: self._add("item"), "add_element", None),
                ("Group", lambda: self._add("group"), "add_session", None),
                ("Delete", self._delete, "delete_forever", None),
                ("", lambda: self._move(-1), "arrow_upward", "Move up"),
                ("", lambda: self._move(1), "arrow_downward", "Move down")):
            button = QPushButton(text)
            button.setIcon(medusa_style.qt.icon(icon))
            if tip is not None:
                button.setToolTip(tip)
            button.clicked.connect(slot)
            buttons.addWidget(button)
        layout.addLayout(buttons)
        return panel

    def _build_property_panel(self):
        box = QGroupBox("Properties")
        outer = QVBoxLayout(box)
        self._form = QFormLayout()
        outer.addLayout(self._form)

        self.key_edit = QLineEdit()
        self.type_combo = QComboBox()
        self.type_combo.addItems(_TYPES)
        self.info_edit = QLineEdit()

        self.opt_type_combo = QComboBox()
        self.opt_type_combo.addItems(_SCALAR_TYPES)
        self.options_edit = QPlainTextEdit()
        self.options_edit.setPlaceholderText("one option per line")
        self.options_edit.setFixedHeight(_EDITOR_BOX_H)

        self.elem_type_combo = QComboBox()
        self.elem_type_combo.addItems(_SCALAR_TYPES)
        self.list_values_edit = QPlainTextEdit()
        self.list_values_edit.setPlaceholderText("one element per line")
        self.list_values_edit.setFixedHeight(_EDITOR_BOX_H)

        self.default_holder = QWidget()
        QHBoxLayout(self.default_holder).setContentsMargins(0, 0, 0, 0)
        self._default_editor = None

        self.min_edit = QLineEdit()
        self.min_edit.setPlaceholderText("min (optional)")
        self.max_edit = QLineEdit()
        self.max_edit.setPlaceholderText("max (optional)")

        # name -> field widget (for show/hide + ordering)
        self._rows = [
            ("key", "Key", self.key_edit),
            ("type", "Type", self.type_combo),
            ("info", "Info", self.info_edit),
            ("opt_type", "Options type", self.opt_type_combo),
            ("options", "Options", self.options_edit),
            ("elem_type", "Element type", self.elem_type_combo),
            ("list_values", "Elements", self.list_values_edit),
            ("default", "Default", self.default_holder),
            ("min", "Min", self.min_edit),
            ("max", "Max", self.max_edit),
        ]
        for _, label, field in self._rows:
            self._form.addRow(label, field)

        apply_button = QPushButton("Apply changes")
        apply_button.clicked.connect(self._apply)
        outer.addWidget(apply_button)
        outer.addStretch()

        self.type_combo.currentTextChanged.connect(self._on_type_changed)
        self.opt_type_combo.currentTextChanged.connect(self._on_options_changed)
        self.options_edit.textChanged.connect(self._on_options_changed)

        self._set_panel_enabled(False)
        return box

    # -- structure tree -----------------------------------------------------
    def _rebuild_tree(self):
        self.tree_widget.clear()
        self._node_of.clear()
        self._item_of.clear()
        self._parent_of.clear()
        for node in self._tree.tree.get("items", []):
            self._add_tree_item(node, self.tree_widget, self._tree.tree)

    def _add_tree_item(self, node, parent_widget, parent_node):
        item = QTreeWidgetItem(parent_widget)
        if "element_group" in parent_node:              # a keyless group-list element
            item.setText(0, f"[{self._index_of(parent_node['items'], node)}]")
            item.setText(1, "Element")
        else:
            item.setText(0, str(node.get("key", "")))
            item.setText(1, self._node_type(node))
        self._node_of[item] = node
        self._item_of[id(node)] = item
        self._parent_of[id(node)] = parent_node
        for child in node.get("items", []):
            self._add_tree_item(child, item, node)
        item.setExpanded(True)
        return item

    def _refresh_row(self, node):
        item = self._item_of.get(id(node))
        if item is not None:
            item.setText(0, str(node.get("key", "")))
            item.setText(1, self._node_type(node))

    @staticmethod
    def _node_type(node):
        if "element_group" in node:
            return "Group list"
        if "items" in node:
            return "Group"
        if node.get("value_options") is not None:
            return "Choice"
        value = node.get("value", node.get("default"))
        if isinstance(value, list):
            return "List"
        return _scalar_type_of(value)

    def select_node(self, node):
        """Select ``node``'s row (used by tests/programmatic flows)."""
        item = self._item_of.get(id(node))
        if item is not None:
            self.tree_widget.setCurrentItem(item)

    # -- structural edits ---------------------------------------------------
    def _siblings(self, node):
        return self._parent_of[id(node)].get("items", [])

    @staticmethod
    def _index_of(siblings, node):
        """Position of ``node`` in ``siblings`` **by identity** (``is``), or ``-1``.

        Group-list elements are keyless and may be value-equal, so ``list.index`` /
        ``list.remove`` (value equality) would find the wrong one -- always match by identity.
        """
        for i, child in enumerate(siblings):
            if child is node:
                return i
        return -1

    def _enclosing_group_list(self, node):
        """The group-list ``node`` is (or lies inside), else ``None`` (walks up parents)."""
        while node is not None:
            if "element_group" in node:
                return node
            node = self._parent_of.get(id(node))
        return None

    def _add(self, kind):
        node = self._current_node
        group_list = self._enclosing_group_list(node)
        if group_list is not None:                       # inside a group-list: add an element
            element = SettingsTree(group_list).add_element().tree
            self._rebuild_tree()
            self.select_node(element)
            return
        if node is not None and "items" in node:         # add inside a group
            siblings, index = node["items"], len(node["items"])
        elif node is not None:                           # add after an item
            siblings = self._siblings(node)
            index = siblings.index(node) + 1
        else:                                            # add at the root
            siblings = self._tree.tree.setdefault("items", [])
            index = len(siblings)

        base = "new_group" if kind == "group" else "new_item"
        key = self._unique_key(siblings, base)
        new = {"key": key, "items": []} if kind == "group" \
            else {"key": key, "default": "", "value": ""}
        siblings.insert(index, new)
        self._rebuild_tree()
        self.select_node(new)

    def _delete(self):
        node = self._current_node
        if node is None:
            return
        siblings = self._siblings(node)
        index = self._index_of(siblings, node)
        if index >= 0:
            del siblings[index]
        self._current_node = None
        self._rebuild_tree()

    def _move(self, delta):
        node = self._current_node
        if node is None:
            return
        siblings = self._siblings(node)
        i = self._index_of(siblings, node)
        j = i + delta
        if 0 <= j < len(siblings):
            siblings[i], siblings[j] = siblings[j], siblings[i]
            self._rebuild_tree()
            self.select_node(node)

    @staticmethod
    def _unique_key(siblings, base):
        existing = {child.get("key") for child in siblings}
        if base not in existing:
            return base
        i = 1
        while f"{base}_{i}" in existing:
            i += 1
        return f"{base}_{i}"

    # -- property panel: load ----------------------------------------------
    def _on_selection_changed(self, current, _previous):
        node = self._node_of.get(current)
        if node is None:
            self._current_node = None
            self._set_panel_enabled(False)
            return
        self._load(node)

    def _load(self, node):
        self._loading = True
        try:
            self._current_node = node
            self._set_panel_enabled(True)
            self.key_edit.setText(str(node.get("key", "")))
            self.info_edit.setText(node.get("info", "") or "")
            type_name = self._node_type(node)
            self.type_combo.setCurrentText(type_name)

            options = node.get("value_options")
            if type_name == "Choice":
                opts = options or []
                self.opt_type_combo.setCurrentText(
                    _scalar_type_of(opts[0]) if opts else "Text")
                self.options_edit.setPlainText(
                    "\n".join(str(o) for o in opts))
            if type_name == "List":
                schema = node.get("element_schema") or {}
                self.elem_type_combo.setCurrentText(
                    self._format_to_type(schema.get("input_format")))
                self.list_values_edit.setPlainText(
                    "\n".join(str(v) for v in (node.get("value") or [])))
            if type_name in ("Integer", "Float"):
                low, high = node.get("value_range") or (None, None)
                self.min_edit.setText("" if low is None else str(low))
                self.max_edit.setText("" if high is None else str(high))

            self._rebuild_default_editor(
                type_name, node.get("value", node.get("default")), options)
            self._update_visibility(type_name)
        finally:
            self._loading = False

    @staticmethod
    def _format_to_type(fmt):
        for type_name, mapped in _TYPE_TO_FORMAT.items():
            if mapped == fmt and type_name in _SCALAR_TYPES:
                return type_name
        return "Text"

    def _on_type_changed(self, type_name):
        if self._loading:
            return
        options = self._read_options() if type_name == "Choice" else None
        self._rebuild_default_editor(type_name, None, options)
        self._update_visibility(type_name)

    def _on_options_changed(self):
        if self._loading or self.type_combo.currentText() != "Choice":
            return
        self._rebuild_default_editor("Choice", self._read_default_value(),
                                     self._read_options())

    def _rebuild_default_editor(self, type_name, value, options):
        layout = self.default_holder.layout()
        while layout.count():
            old = layout.takeAt(0).widget()
            if old is not None:
                old.setParent(None)
                old.deleteLater()
        fmt = _TYPE_TO_FORMAT.get(type_name)
        if fmt is None:                       # Group / List: no scalar default
            self._default_editor = None
            return
        if value is None:
            value = options[0] if (fmt == "combobox" and options) \
                else _TYPE_DEFAULT[type_name]
        editor = _build_editor(fmt, value, None, options)
        layout.addWidget(editor)
        self._default_editor = editor

    def _update_visibility(self, type_name):
        shown = {
            "Group": {"key", "type", "info"},
            "Group list": {"key", "type", "info"},
            "Boolean": {"key", "type", "info", "default"},
            "Integer": {"key", "type", "info", "default", "min", "max"},
            "Float": {"key", "type", "info", "default", "min", "max"},
            "Text": {"key", "type", "info", "default"},
            "Choice": {"key", "type", "info", "opt_type", "options", "default"},
            "List": {"key", "type", "info", "elem_type", "list_values"},
        }[type_name]
        for name, _, field in self._rows:
            self._form.setRowVisible(field, name in shown)

    # -- property panel: read / apply --------------------------------------
    def _read_default_value(self):
        return _read_editor(self._default_editor) \
            if self._default_editor is not None else None

    def _read_options(self):
        opt_type = self.opt_type_combo.currentText()
        return [_parse_scalar(line, opt_type)
                for line in self.options_edit.toPlainText().splitlines()
                if line.strip()]

    def _read_list_values(self):
        elem_type = self.elem_type_combo.currentText()
        return [_parse_scalar(line, elem_type)
                for line in self.list_values_edit.toPlainText().splitlines()
                if line.strip()]

    @staticmethod
    def _parse_bound(text, type_name):
        text = text.strip()
        if not text:
            return None
        return int(text) if type_name == "Integer" else float(text)

    def _apply(self):
        node = self._current_node
        if node is None:
            return
        key = self.key_edit.text().strip()
        if not key:
            return self._status("Key cannot be empty.")
        siblings = self._siblings(node)
        if any(child is not node and child.get("key") == key
               for child in siblings):
            return self._status(f"Duplicate key {key!r} at this level.")
        type_name = self.type_combo.currentText()
        info = self.info_edit.text().strip() or None
        try:
            new = self._build_node(key, type_name, info, node)
        except (TypeError, ValueError) as exc:
            return self._status(str(exc))
        node.clear()
        node.update(new)
        self._refresh_row(node)
        self._status(f"Applied {key!r}.")

    def _build_node(self, key, type_name, info, node):
        if type_name == "Group list":               # keep (or start) a group-list intact
            new = {"key": key, "input_format": "group_list",
                   "element_group": node.get("element_group", {"items": []}),
                   "items": node.get("items", []) if "element_group" in node else []}
            if info is not None:
                new["info"] = info
            return new
        if type_name == "Group":
            new = {"key": key, "items": node.get("items", [])}
            if info is not None:
                new["info"] = info
            return new

        kwargs = {"info": info}
        if type_name == "List":
            kwargs["value"] = self._read_list_values()
            kwargs["element_schema"] = {
                "input_format": _TYPE_TO_FORMAT[self.elem_type_combo
                                                .currentText()]}
        else:
            kwargs["value"] = self._read_default_value()
        if type_name == "Choice":
            options = self._read_options()
            if not options:
                raise ValueError("a Choice needs at least one option.")
            kwargs["value_options"] = options
        if type_name in ("Integer", "Float"):
            low = self._parse_bound(self.min_edit.text(), type_name)
            high = self._parse_bound(self.max_edit.text(), type_name)
            if low is not None or high is not None:
                kwargs["value_range"] = [low, high]
        # Validate by round-tripping through the canonical builder.
        child = SettingsTree().add_item(key, **kwargs)
        return child.to_serializable_obj()

    # -- file / preview -----------------------------------------------------
    def _new(self):
        self._tree = SettingsTree()
        self._current_node = None
        self._rebuild_tree()
        self._set_panel_enabled(False)

    def _open(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open settings schema", "", "JSON (*.json)")
        if path:
            self.load(SettingsTree.from_json(path))

    def _save(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save settings schema", "", "JSON (*.json)")
        if path:
            self.to_settings().to_json(path)
            self._status(f"Saved to {path}")

    def _preview_(self):
        self._preview = TreeViewer(self.to_settings())
        self._preview.setWindowTitle("Preview (end-user editor)")
        self._preview.show()

    # -- misc ---------------------------------------------------------------
    def _set_panel_enabled(self, enabled):
        for _, _, field in getattr(self, "_rows", []):
            field.setEnabled(enabled)

    def _status(self, message):
        self.statusBar().showMessage(message, 5000)
