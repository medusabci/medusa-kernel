"""Qt editor for a :class:`medusa.core.settings_tree.SettingsTree`.

:class:`SettingsTreeWidget` renders a settings tree as an editable
``QTreeWidget`` (a combobox/checkbox/spinbox/line-edit per leaf, an add/remove
editor for list values) and reads the edited tree back with
:meth:`SettingsTreeWidget.get_settings`. :class:`TreeViewer` is a standalone
window around it, and :class:`TreeSearchBar` is the (optional, reusable) find
bar.

This is the *end-user* (value-editing) half. To author or edit the schema
itself, see :class:`~medusa.widgets.settings_tree.tree_builder.SettingsTreeBuilder`
in :mod:`medusa.widgets.settings_tree.tree_builder`.

PySide6 is a core dependency of medusa-kernel.

Interaction model
-----------------
Actions live next to the content they affect rather than in global button rows:

* **Value editing** -- each leaf row carries its bare type-editor in the Value
  column (direct manipulation).
* **List editing** -- a list row shows a ``[a, b] (n)`` summary plus a ``+`` in
  its Value cell; an element's ``-`` appears only on the current/hovered row; a
  flat ``+ Add element`` sits at the list's bottom. ``Insert``/``Delete`` and a
  right-click menu do the same.
* **Reset to default** -- overridden leaves show in **bold**; a right-click
  ``Reset to default`` / ``Ctrl+Backspace`` (and group ``Reset section`` /
  toolbar ``Reset all``) restore the build-time defaults (``SettingsTree``).
* **Find** -- the search bar cycles matches (key/value/info) with ``F3`` /
  ``Shift+F3``, an "m of n" counter and ``Ctrl+F`` / ``Esc``.

Every editable row stores its node key and kind on the ``QTreeWidgetItem`` (Qt
user roles), so reading values back matches widget rows to schema nodes *by
key* -- robust to row reordering, info rows and the list "Add" button.
"""

import copy

import medusa_style
from PySide6.QtCore import QEvent, QSize, Qt
from PySide6.QtGui import QFontMetrics, QKeySequence, QPalette, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QPushButton,
    QSpinBox,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from medusa.core.settings_tree import SettingsTree, infer_input_format
from medusa.widgets._toolbar import add_main_toolbar, pin_toolbar_width

__all__ = ["SettingsTreeWidget", "TreeViewer", "TreeSearchBar"]

# QTreeWidgetItem user-role slots holding our per-row metadata.
_ROLE_KEY = int(Qt.UserRole)
_ROLE_KIND = int(Qt.UserRole) + 1

# Row kinds stored under _ROLE_KIND.
_KIND_GROUP = "group"          # a branch node (has children)
_KIND_LEAF = "leaf"            # a scalar value with a single editor
_KIND_LIST = "list"            # a list value (expands into element rows)
_KIND_ELEMENT = "element"      # one scalar element of a list value
_KIND_ADD = "add"             # the trailing "Add" button row of a list
_KIND_GROUP_LIST = "group_list"      # a group-list (repeated group; expands into element groups)
_KIND_GROUP_ELEMENT = "group_element"  # one element (a group of leaves) of a group-list

# QSpinBox is limited to signed 32-bit; QDoubleSpinBox gets a wide finite range.
_INT_MIN, _INT_MAX = -2_147_483_648, 2_147_483_647
_FLOAT_MIN, _FLOAT_MAX = -1.0e12, 1.0e12
_FLOAT_DECIMALS = 6
_SUMMARY_MAX = 40              # chars before a list summary is elided
_INFO_COL_WIDTH = 300         # default width (px) of the Info column
_WINDOW_SIZE = (1000, 600)    # default TreeViewer window size (px)


# ---------------------------------------------------------------------------
# Editor factory: one place mapping input_format -> editor widget (+ read-back)
# ---------------------------------------------------------------------------
def _apply_int_range(spin, value_range):
    low, high = (value_range or (None, None))
    spin.setRange(low if low is not None else _INT_MIN,
                  high if high is not None else _INT_MAX)


def _apply_float_range(spin, value_range):
    low, high = (value_range or (None, None))
    spin.setRange(float(low) if low is not None else _FLOAT_MIN,
                  float(high) if high is not None else _FLOAT_MAX)


def _build_checkbox(value, value_range, value_options):
    widget = QCheckBox()
    widget.setChecked(bool(value))
    return widget


def _build_spinbox(value, value_range, value_options):
    widget = QSpinBox()
    _apply_int_range(widget, value_range)
    widget.setValue(int(value))
    return widget


def _build_doublespinbox(value, value_range, value_options):
    widget = QDoubleSpinBox()
    widget.setDecimals(_FLOAT_DECIMALS)
    _apply_float_range(widget, value_range)
    widget.setValue(float(value))
    return widget


def _build_lineedit(value, value_range, value_options):
    widget = QLineEdit()
    widget.setText("" if value is None else str(value))
    return widget


def _build_combobox(value, value_range, value_options):
    widget = QComboBox()
    options = value_options or []
    for option in options:
        # Display as text but keep the original typed value as item data, so
        # an int/float option survives the round-trip instead of becoming str.
        widget.addItem(str(option), option)
    if value in options:
        widget.setCurrentIndex(options.index(value))
    return widget


#: input_format -> editor builder. ``list`` is handled structurally, not here.
_EDITOR_BUILDERS = {
    "checkbox": _build_checkbox,
    "spinbox": _build_spinbox,
    "doublespinbox": _build_doublespinbox,
    "lineedit": _build_lineedit,
    "combobox": _build_combobox,
}


def _build_editor(input_format, value, value_range, value_options):
    """Build the value-column editor for a scalar leaf/list element."""
    builder = _EDITOR_BUILDERS.get(input_format, _build_lineedit)
    return builder(value, value_range, value_options)


def _read_editor(widget):
    """Read the current value out of an editor built by :func:`_build_editor`."""
    if isinstance(widget, QComboBox):
        return widget.currentData()           # typed option, not its label
    if isinstance(widget, QCheckBox):
        return widget.isChecked()
    if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
        return widget.value()
    if isinstance(widget, QLineEdit):
        return widget.text()
    return None


def _editor_change_signal(widget):
    """The 'value changed' signal for an editor (for live override/summary)."""
    if isinstance(widget, QComboBox):
        return widget.currentIndexChanged
    if isinstance(widget, QCheckBox):
        return widget.toggled
    if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
        return widget.valueChanged
    if isinstance(widget, QLineEdit):
        return widget.textChanged
    return None


def _element_default(element_schema, sample=None):
    """A sensible default for a freshly added list element.

    Prefers the ``element_schema`` (options/format); otherwise mirrors the type
    of an existing element (``sample``) so a list of floats adds ``0.0`` rather
    than an empty string.
    """
    schema = element_schema or {}
    options = schema.get("value_options")
    if options:
        return options[0]
    fmt = schema.get("input_format")
    if fmt:
        return {"checkbox": False, "spinbox": 0, "doublespinbox": 0.0,
                "lineedit": "", "list": []}[fmt]
    if isinstance(sample, bool):
        return False
    if isinstance(sample, float):
        return 0.0
    if isinstance(sample, int):
        return 0
    if isinstance(sample, list):
        return []
    return ""


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------
class _TextIndex:
    """Maps searchable text (key, value, info) to its ``QTreeWidgetItem``."""

    def __init__(self):
        self._entries = []   # list[(lowercased_text, item)]

    def clear(self):
        self._entries.clear()

    def append(self, item, *texts):
        for text in texts:
            if text is not None:
                self._entries.append((str(text).lower(), item))

    def find(self, query):
        query = query.lower()
        seen, matches = set(), []
        for text, item in self._entries:
            if query in text and id(item) not in seen:
                seen.add(id(item))
                matches.append(item)
        return matches


class TreeSearchBar(QWidget):
    """A reusable find bar that cycles matches in a ``QTreeWidget``.

    Parameters
    ----------
    tree_widget : QTreeWidget
        The tree to highlight matches in.
    index : _TextIndex
        The text index built while populating the tree.
    parent : QWidget, optional
        Qt parent.
    reindex : callable, optional
        Called before each search to refresh ``index`` against the live tree
        (so edited/added element values are findable). Omit to search a static
        index.
    """

    def __init__(self, tree_widget, index, parent=None, reindex=None):
        super().__init__(parent)
        self._tree = tree_widget
        self._index = index
        self._reindex = reindex
        self._query = ""
        self._matches = []
        self._pos = 0

        self._box = QLineEdit()
        self._box.setPlaceholderText("Find key, value or info...")
        self._box.setClearButtonEnabled(True)
        self._box.returnPressed.connect(self.find_next)

        self._prev_btn = QToolButton()
        self._prev_btn.setIcon(medusa_style.qt.icon("arrow_upward"))
        self._prev_btn.setAutoRaise(True)
        self._prev_btn.setToolTip("Previous match (Shift+F3)")
        self._prev_btn.clicked.connect(self.find_prev)
        self._next_btn = QToolButton()
        self._next_btn.setIcon(medusa_style.qt.icon("arrow_downward"))
        self._next_btn.setAutoRaise(True)
        self._next_btn.setToolTip("Next match (F3)")
        self._next_btn.clicked.connect(self.find_next)
        self._counter = QLabel("")
        find_button = QPushButton(medusa_style.qt.icon("search"), "Find")
        find_button.clicked.connect(self.find_next)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._box)
        layout.addWidget(self._prev_btn)
        layout.addWidget(self._next_btn)
        layout.addWidget(self._counter)
        layout.addWidget(find_button)

    def focus(self):
        """Focus and select the search box (for a Ctrl+F shortcut)."""
        self._box.setFocus()
        self._box.selectAll()

    def clear(self):
        """Clear the query, selection state and any no-match tint."""
        self._box.clear()
        self._query = ""
        self._matches = []
        self._pos = 0
        self._box.setStyleSheet("")
        self._counter.setText("")

    def find_next(self):
        """Select the next match (re-querying when the text changed)."""
        self._search(forward=True)

    def find_prev(self):
        """Select the previous match (re-querying when the text changed)."""
        self._search(forward=False)

    def _search(self, forward):
        text = self._box.text().strip()
        if not text:
            self._counter.setText("")
            return
        if self._reindex is not None:
            self._reindex()
        if text != self._query:
            self._query = text
            self._matches = self._index.find(text)
            self._pos = 0 if forward else max(0, len(self._matches) - 1)
        elif self._matches:
            self._pos = (self._pos + (1 if forward else -1)) % len(self._matches)
        if self._matches:
            self._box.setStyleSheet("")
            self._tree.setCurrentItem(self._matches[self._pos])
            self._counter.setText(f"{self._pos + 1} of {len(self._matches)}")
        else:
            self._box.setStyleSheet(
                f"QLineEdit {{ color: "
                f"{medusa_style.current_theme().error}; }}")
            self._counter.setText("0 of 0")


# ---------------------------------------------------------------------------
# Info column word-wrap
# ---------------------------------------------------------------------------
class _InfoColumnDelegate(QStyledItemDelegate):
    """Word-wrapping delegate for the Info column.

    Qt's ``QTreeView.setWordWrap(True)`` wraps the glyphs but never grows the
    row, so a long Info string clips to one line. This delegate closes that gap:
    when :attr:`wrap` is on it reports the *wrapped* height for the Info column's
    current width (so the row grows to fit) and paints the text wrapped. When off
    it defers to the base delegate (single line, elided, with a tooltip).
    :meth:`SettingsTreeWidget.set_info_wrap` flips :attr:`wrap`.
    """

    _PAD = 6   # cell padding (px) mirrored into the width/height budget

    def __init__(self, tree):
        super().__init__(tree)
        self._tree = tree
        self.wrap = False

    def sizeHint(self, option, index):
        hint = super().sizeHint(option, index)
        if not self.wrap:
            return hint
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        width = self._tree.columnWidth(index.column()) - 2 * self._PAD
        if not opt.text or width <= 0:
            return hint
        rect = QFontMetrics(opt.font).boundingRect(
            0, 0, width, 1 << 20, Qt.TextWordWrap, opt.text)
        return QSize(hint.width(), max(hint.height(), rect.height() + 2 * self._PAD))

    def paint(self, painter, option, index):
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        if not self.wrap or not opt.text:
            super().paint(painter, option, index)
            return
        text, opt.text = opt.text, ""          # let the style draw only the chrome
        style = (opt.widget.style() if opt.widget else QApplication.style())
        style.drawControl(QStyle.CE_ItemViewItem, opt, painter, opt.widget)
        rect = style.subElementRect(QStyle.SE_ItemViewItemText, opt, opt.widget)
        selected = bool(opt.state & QStyle.State_Selected)
        painter.save()
        painter.setClipRect(opt.rect)
        painter.setFont(opt.font)
        painter.setPen(opt.palette.color(
            QPalette.HighlightedText if selected else QPalette.Text))
        painter.drawText(rect, Qt.TextWordWrap | Qt.AlignTop | Qt.AlignLeft, text)
        painter.restore()


# ---------------------------------------------------------------------------
# Editor widget
# ---------------------------------------------------------------------------
class SettingsTreeWidget(QWidget):
    """An editable view of a :class:`SettingsTree`.

    Parameters
    ----------
    jdata : SettingsTree or dict or list
        The schema to edit: a :class:`SettingsTree`, its serialized root node
        dict, or a bare list of item dicts.
    search : bool, default True
        Show the :class:`TreeSearchBar` above the tree.
    parent : QWidget, optional
        Qt parent.

    Notes
    -----
    Call :meth:`get_settings` to read the edited tree back as a new
    :class:`SettingsTree`, or :meth:`get_values` for a plain nested dict.
    """

    def __init__(self, jdata, search=True, parent=None, wrap_info=False):
        super().__init__(parent)
        self._source = self._as_root_node(jdata)
        self._index = _TextIndex()
        self._node_of = {}              # item -> source node dict
        self._baseline_of = {}          # leaf item -> initial editor read-back
        self._element_remove_btn = {}   # element item -> its "-" QToolButton
        self._summary_label_of = {}     # list item -> its summary QLabel
        self._hover_item = None
        self._wrap_info = False

        self.tree_widget = QTreeWidget()
        self.tree_widget.setHeaderLabels(["Key", "Value", "Info"])
        # The Value column fills the available width so the widget adapts to the
        # window; Key and Info stay user-resizable. When the columns no longer
        # fit (deep keys, a widened Info, a narrow window) a horizontal scrollbar
        # appears instead of squashing the content.
        header = self.tree_widget.header()
        header.setSectionResizeMode(0, QHeaderView.Interactive)   # Key
        header.setSectionResizeMode(1, QHeaderView.Stretch)       # Value -> fills
        header.setSectionResizeMode(2, QHeaderView.Interactive)   # Info
        header.setStretchLastSection(False)
        header.setMinimumSectionSize(50)
        # Word-wrap the Info column on demand (grows the row so nothing clips).
        self._info_delegate = _InfoColumnDelegate(self.tree_widget)
        self.tree_widget.setItemDelegateForColumn(2, self._info_delegate)
        header.sectionResized.connect(self._on_section_resized)
        self.tree_widget.setHorizontalScrollMode(
            QAbstractItemView.ScrollPerPixel)
        self.tree_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        # Selection highlight comes from the app-wide medusa-style theme (QSS +
        # QPalette); the tree only opts into alternating row backgrounds.
        self.tree_widget.setAlternatingRowColors(True)
        self.tree_widget.setMouseTracking(True)
        self.tree_widget.viewport().installEventFilter(self)
        self.tree_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree_widget.customContextMenuRequested.connect(self._on_context_menu)

        self._populate()
        # Connect AFTER populate so build-time current-item churn is ignored.
        self.tree_widget.currentItemChanged.connect(self._on_current_changed)

        layout = QVBoxLayout(self)
        self.search_bar = TreeSearchBar(
            self.tree_widget, self._index, reindex=self._reindex) \
            if search else None
        if self.search_bar is not None:
            layout.addWidget(self.search_bar)
        layout.addWidget(self.tree_widget)

        self._install_shortcuts()
        self.set_info_wrap(wrap_info)

    @staticmethod
    def _as_root_node(jdata):
        if isinstance(jdata, SettingsTree):
            return jdata.to_serializable_obj()
        if isinstance(jdata, list):
            return {"items": jdata}
        return jdata

    # -- building -----------------------------------------------------------
    def _populate(self):
        self.tree_widget.clear()
        self._node_of.clear()
        self._baseline_of.clear()
        self._element_remove_btn.clear()
        self._summary_label_of.clear()
        self._hover_item = None
        for node in self._source.get("items", []):
            self._add_node_row(node, self.tree_widget)
        self._reindex()
        self.tree_widget.resizeColumnToContents(0)   # Key fits its content
        self.tree_widget.setColumnWidth(2, _INFO_COL_WIDTH)  # Info default
        # (Value stretches to fill whatever width remains.)

    def _set_value_widget(self, item, widget):
        """Place an editor in the Value column, height-capped to its size hint so
        it stays compact and top-aligned when word-wrapped Info grows the row."""
        widget.setMaximumHeight(widget.sizeHint().height())
        self.tree_widget.setItemWidget(item, 1, widget)

    def _add_node_row(self, node, parent):
        key = node.get("key", "")
        value = node.get("value", node.get("default"))
        info = node.get("info")
        items = node.get("items")

        row = QTreeWidgetItem(parent)
        row.setText(0, str(key))
        row.setData(0, _ROLE_KEY, key)
        self._node_of[row] = node
        if info is not None:
            row.setText(2, str(info))
            row.setToolTip(2, str(info))

        if "element_group" in node:           # a group-list (has 'items' too: check this first)
            row.setData(0, _ROLE_KIND, _KIND_GROUP_LIST)
            self._set_bold(row, True)
            self._make_group_list_value_widget(row)
            for element in items or []:
                self._insert_group_element(row, row.childCount(), element)
            self._append_group_add_button(row)
            self._relabel_group_list(row)
        elif items is not None:
            row.setData(0, _ROLE_KIND, _KIND_GROUP)
            self._set_bold(row, True)
            row.setFirstColumnSpanned(True)   # empty Value/Info stop looking editable
            for child in items:
                self._add_node_row(child, row)
        elif isinstance(value, list):
            row.setData(0, _ROLE_KIND, _KIND_LIST)
            schema = node.get("element_schema") or {}
            self._make_list_value_widget(row, schema)
            self._build_list_rows(row, node.get("value") or [], schema)
        else:
            row.setData(0, _ROLE_KIND, _KIND_LEAF)
            fmt = node.get("input_format") \
                or infer_input_format(value, node.get("value_options"))
            editor = _build_editor(fmt, value, node.get("value_range"),
                                   node.get("value_options"))
            tooltip = self._editor_tooltip(node)
            if tooltip:
                editor.setToolTip(tooltip)
            self._set_value_widget(row, editor)
            self._baseline_of[row] = _read_editor(editor)
            self._connect_change(editor, row)

    @staticmethod
    def _editor_tooltip(node):
        parts = []
        if node.get("info"):
            parts.append(str(node["info"]))
        value_range = node.get("value_range")
        if value_range:
            parts.append(f"Range: [{value_range[0]}, {value_range[1]}]")
        value_options = node.get("value_options")
        if value_options:
            parts.append("Options: " + ", ".join(str(o) for o in value_options))
        return "\n".join(parts)

    @staticmethod
    def _set_bold(item, bold):
        font = item.font(0)
        font.setBold(bold)
        item.setFont(0, font)

    def _connect_change(self, editor, item):
        signal = _editor_change_signal(editor)
        if signal is not None:
            signal.connect(lambda *_: self._on_editor_changed(item))

    def _on_editor_changed(self, item):
        kind = item.data(0, _ROLE_KIND)
        if kind == _KIND_LEAF:
            self._refresh_override_marker(item)
        elif kind == _KIND_ELEMENT:
            parent = item.parent()
            if parent is not None:
                self._update_list_summary(parent)

    # -- list editing -------------------------------------------------------
    def _make_list_value_widget(self, list_item, element_schema):
        """Value-cell of a list row: a ``[a, b] (n)`` summary + a ``+`` button."""
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        summary = QLabel("")
        add = QToolButton()
        add.setIcon(medusa_style.qt.icon("add"))
        add.setAutoRaise(True)
        add.setToolTip("Add element")
        add.clicked.connect(
            lambda: self._add_list_element(list_item, element_schema))
        layout.addWidget(summary, 1)
        layout.addWidget(add, 0)
        self._set_value_widget(list_item, container)
        self._summary_label_of[list_item] = summary

    def _build_list_rows(self, list_item, elements, element_schema):
        for element in elements:
            self._insert_list_element(list_item, list_item.childCount(),
                                      element, element_schema)
        self._append_add_button(list_item, element_schema)
        self._relabel_list(list_item)

    def _insert_list_element(self, list_item, index, value, element_schema):
        row = QTreeWidgetItem()
        list_item.insertChild(index, row)
        self.tree_widget.setItemWidget(
            row, 0, self._make_element_key_widget(list_item, row))
        if isinstance(value, list):
            row.setData(0, _ROLE_KIND, _KIND_LIST)
            nested = element_schema.get("element_schema") or {} \
                if isinstance(element_schema, dict) else {}
            self._make_list_value_widget(row, nested)
            self._build_list_rows(row, value, nested)
        else:
            row.setData(0, _ROLE_KIND, _KIND_ELEMENT)
            schema = element_schema or {}
            fmt = schema.get("input_format") \
                or infer_input_format(value, schema.get("value_options"))
            editor = _build_editor(fmt, value, schema.get("value_range"),
                                   schema.get("value_options"))
            self._set_value_widget(row, editor)
            self._connect_change(editor, row)

    def _make_element_key_widget(self, list_item, element_row):
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        label = QLabel("")
        remove = QToolButton()
        remove.setIcon(medusa_style.qt.icon("remove"))
        remove.setAutoRaise(True)
        remove.setToolTip("Remove element")
        remove.setVisible(False)          # revealed only on the current/hovered row
        remove.clicked.connect(
            lambda: self._remove_list_element(list_item, element_row))
        layout.addWidget(label)
        layout.addStretch()
        layout.addWidget(remove)
        self._element_remove_btn[element_row] = remove
        return container

    def _append_add_button(self, list_item, element_schema):
        add_row = QTreeWidgetItem(list_item)
        add_row.setData(0, _ROLE_KIND, _KIND_ADD)
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        button = QToolButton()
        button.setIcon(medusa_style.qt.icon("add"))
        button.setText("Add element")
        button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        button.setAutoRaise(True)
        button.clicked.connect(
            lambda: self._add_list_element(list_item, element_schema))
        layout.addWidget(button)
        layout.addStretch()
        self.tree_widget.setItemWidget(add_row, 0, container)

    def _add_list_element(self, list_item, element_schema=None):
        if element_schema is None:
            node = self._node_of.get(list_item)
            element_schema = (node or {}).get("element_schema") or {}
        # Insert just before the trailing Add-button row.
        index = max(0, list_item.childCount() - 1)
        default = _element_default(element_schema,
                                   self._sample_element(list_item))
        self._insert_list_element(list_item, index, default, element_schema)
        self._relabel_list(list_item)
        list_item.setExpanded(True)

    def _sample_element(self, list_item):
        """The current value of the first real element, to type a new one."""
        for j in range(list_item.childCount()):
            child = list_item.child(j)
            kind = child.data(0, _ROLE_KIND)
            if kind == _KIND_ELEMENT:
                editor = self.tree_widget.itemWidget(child, 1)
                return _read_editor(editor) if editor is not None else None
            if kind == _KIND_LIST:
                return []
        return None

    def _remove_list_element(self, list_item, element_row):
        self._element_remove_btn.pop(element_row, None)
        list_item.removeChild(element_row)
        if list_item.data(0, _ROLE_KIND) == _KIND_GROUP_LIST:
            self._relabel_group_list(list_item)
        else:
            self._relabel_list(list_item)

    def _relabel_list(self, list_item):
        idx = 0
        for j in range(list_item.childCount()):
            child = list_item.child(j)
            if child.data(0, _ROLE_KIND) == _KIND_ADD:
                continue
            container = self.tree_widget.itemWidget(child, 0)
            if container is not None:
                label = container.findChild(QLabel)
                if label is not None:
                    label.setText(f"[{idx}]")
            idx += 1
        self._update_list_summary(list_item)

    def _update_list_summary(self, list_item):
        label = self._summary_label_of.get(list_item)
        if label is None:
            return
        values = self._read_list(list_item)
        inner = ", ".join(str(v) for v in values)
        if len(inner) > _SUMMARY_MAX:
            inner = inner[:_SUMMARY_MAX - 1] + "…"
        label.setText(f"[{inner}] ({len(values)})")

    # -- group-list editing (a repeated group; each element is an editable group) -----
    def _make_group_list_value_widget(self, list_item):
        """Value-cell of a group-list row: an element ``(n)`` count + a ``+`` add button."""
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        summary = QLabel("")
        add = QToolButton()
        add.setIcon(medusa_style.qt.icon("add"))
        add.setAutoRaise(True)
        add.setToolTip("Add element")
        add.clicked.connect(lambda: self._add_group_element(list_item))
        layout.addWidget(summary, 1)
        layout.addWidget(add, 0)
        self._set_value_widget(list_item, container)
        self._summary_label_of[list_item] = summary

    def _insert_group_element(self, list_item, index, element_node):
        """Insert one element: a ``[i]`` row (with a remove button) + its leaf editor rows."""
        element_row = QTreeWidgetItem()
        list_item.insertChild(index, element_row)
        element_row.setData(0, _ROLE_KIND, _KIND_GROUP_ELEMENT)
        self._node_of[element_row] = element_node
        self.tree_widget.setItemWidget(
            element_row, 0, self._make_element_key_widget(list_item, element_row))
        for leaf in element_node.get("items", []):
            self._add_node_row(leaf, element_row)
        return element_row

    def _append_group_add_button(self, list_item):
        add_row = QTreeWidgetItem(list_item)
        add_row.setData(0, _ROLE_KIND, _KIND_ADD)
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        button = QToolButton()
        button.setIcon(medusa_style.qt.icon("add"))
        button.setText("Add element")
        button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        button.setAutoRaise(True)
        button.clicked.connect(lambda: self._add_group_element(list_item))
        layout.addWidget(button)
        layout.addStretch()
        self.tree_widget.setItemWidget(add_row, 0, container)

    def _add_group_element(self, list_item):
        node = self._node_of.get(list_item)
        template = (node or {}).get("element_group") or {"items": []}
        element = copy.deepcopy(template)              # a fresh element (template + its defaults)
        index = max(0, list_item.childCount() - 1)     # before the trailing Add-button row
        self._insert_group_element(list_item, index, element)
        self._relabel_group_list(list_item)
        list_item.setExpanded(True)

    def _relabel_group_list(self, list_item):
        count = 0
        for j in range(list_item.childCount()):
            child = list_item.child(j)
            if child.data(0, _ROLE_KIND) != _KIND_GROUP_ELEMENT:
                continue
            container = self.tree_widget.itemWidget(child, 0)
            if container is not None:
                label = container.findChild(QLabel)
                if label is not None:
                    label.setText(f"[{count}]")
            count += 1
        summary = self._summary_label_of.get(list_item)
        if summary is not None:
            summary.setText(f"({count})")

    def _rebuild_group_list(self, list_item, node):
        while list_item.childCount():
            child = list_item.child(0)
            self._element_remove_btn.pop(child, None)
            list_item.removeChild(child)
        template = node.get("element_group") or {"items": []}
        for values in node.get("default") or []:
            element = copy.deepcopy(template)
            SettingsTree(element).update_from_dict(values)   # deep (nested groups/lists too)
            self._insert_group_element(list_item, list_item.childCount(), element)
        self._append_group_add_button(list_item)
        self._relabel_group_list(list_item)

    # -- override marker / reset -------------------------------------------
    def _is_overridden(self, item):
        node = self._node_of.get(item)
        if not node or "default" not in node:
            return False
        if item not in self._baseline_of:
            return False
        editor = self.tree_widget.itemWidget(item, 1)
        if editor is None:
            return False
        return _read_editor(editor) != self._baseline_of[item]

    def _refresh_override_marker(self, item):
        if item.data(0, _ROLE_KIND) == _KIND_LEAF:
            self._set_bold(item, self._is_overridden(item))

    def _reset_row(self, item):
        kind = item.data(0, _ROLE_KIND)
        if kind == _KIND_GROUP:
            for j in range(item.childCount()):
                self._reset_row(item.child(j))
        elif kind == _KIND_LIST:
            node = self._node_of.get(item)
            if node and "default" in node:
                self._rebuild_list(item, node)
        elif kind == _KIND_GROUP_LIST:
            node = self._node_of.get(item)
            if node and "default" in node:
                self._rebuild_group_list(item, node)
        elif kind == _KIND_LEAF:
            node = self._node_of.get(item)
            if node and "default" in node:
                self._reset_leaf(item, node)

    def _reset_leaf(self, item, node):
        default = node["default"]
        fmt = node.get("input_format") \
            or infer_input_format(default, node.get("value_options"))
        editor = _build_editor(fmt, default, node.get("value_range"),
                               node.get("value_options"))
        tooltip = self._editor_tooltip(node)
        if tooltip:
            editor.setToolTip(tooltip)
        self._set_value_widget(item, editor)
        self._connect_change(editor, item)
        self._baseline_of[item] = _read_editor(editor)
        self._refresh_override_marker(item)

    def _rebuild_list(self, list_item, node):
        while list_item.childCount():
            child = list_item.child(0)
            self._element_remove_btn.pop(child, None)
            list_item.removeChild(child)
        self._build_list_rows(list_item, node.get("default") or [],
                              node.get("element_schema") or {})

    def reset_all(self):
        """Restore every value to its build-time default."""
        for i in range(self.tree_widget.topLevelItemCount()):
            self._reset_row(self.tree_widget.topLevelItem(i))

    # -- view options -------------------------------------------------------
    def set_info_wrap(self, enabled):
        """Toggle the Info column between single-line (elided + tooltip) and
        word-wrapped (full text on multiple lines; the row grows to fit)."""
        self._wrap_info = bool(enabled)
        self._info_delegate.wrap = self._wrap_info
        # Stop the base delegate from eliding the Info text out from under ours.
        self.tree_widget.setTextElideMode(
            Qt.ElideNone if self._wrap_info else Qt.ElideRight)
        self.tree_widget.doItemsLayout()

    def _on_section_resized(self, index, _old, _new):
        """Re-flow wrapped Info rows when the Info column is resized."""
        if self._wrap_info and index == 2:
            self.tree_widget.doItemsLayout()

    def resize_columns_to_contents(self):
        """Size every column (Key, Value, Info) to fit its contents."""
        for column in range(self.tree_widget.columnCount()):
            self.tree_widget.resizeColumnToContents(column)

    # -- reveal-on-current/hover -------------------------------------------
    def _on_current_changed(self, current, previous):
        for item in (previous, current):
            if item is not None:
                self._update_reveal(item)
        if current is not None:
            self._refresh_override_marker(current)

    def _update_reveal(self, item):
        button = self._element_remove_btn.get(item)
        if button is not None:
            button.setVisible(item is self.tree_widget.currentItem()
                              or item is self._hover_item)

    def _set_hover(self, item):
        if item is self._hover_item:
            return
        previous, self._hover_item = self._hover_item, item
        for it in (previous, item):
            if it is not None:
                self._update_reveal(it)

    def eventFilter(self, obj, event):
        if obj is self.tree_widget.viewport():
            if event.type() == QEvent.MouseMove:
                self._set_hover(
                    self.tree_widget.itemAt(event.position().toPoint()))
            elif event.type() == QEvent.Leave:
                self._set_hover(None)
        return super().eventFilter(obj, event)

    # -- context menu -------------------------------------------------------
    def _on_context_menu(self, pos):
        item = self.tree_widget.itemAt(pos)
        if item is None:
            return
        menu = self._build_context_menu(item)
        if menu.actions():
            menu.exec(self.tree_widget.viewport().mapToGlobal(pos))

    def _build_context_menu(self, item):
        """Build (but do not show) the per-row-kind context menu."""
        menu = QMenu(self)
        kind = item.data(0, _ROLE_KIND)
        ic = medusa_style.qt.icon  # theme-recoloring SVGs
        if kind == _KIND_LEAF:
            act = menu.addAction(ic("refresh"), "Reset to default")
            act.setShortcut(QKeySequence("Ctrl+Backspace"))
            act.setEnabled(self._is_overridden(item))
            act.triggered.connect(lambda: self._reset_row(item))
        elif kind == _KIND_LIST:
            add = menu.addAction(ic("add"), "Add element")
            add.setShortcut(QKeySequence("Ins"))
            add.triggered.connect(lambda: self._add_list_element(item))
            node = self._node_of.get(item)
            if node and "default" in node:
                reset = menu.addAction(ic("refresh"), "Reset to default")
                reset.triggered.connect(lambda: self._reset_row(item))
        elif kind == _KIND_ELEMENT:
            rm = menu.addAction(ic("remove"), "Remove element")
            rm.setShortcut(QKeySequence("Del"))
            rm.triggered.connect(
                lambda: self._remove_list_element(item.parent(), item))
        elif kind == _KIND_GROUP_LIST:
            add = menu.addAction(ic("add"), "Add element")
            add.setShortcut(QKeySequence("Ins"))
            add.triggered.connect(lambda: self._add_group_element(item))
            node = self._node_of.get(item)
            if node and "default" in node:
                reset = menu.addAction(ic("refresh"), "Reset to default")
                reset.triggered.connect(lambda: self._reset_row(item))
        elif kind == _KIND_GROUP_ELEMENT:
            rm = menu.addAction(ic("remove"), "Remove element")
            rm.setShortcut(QKeySequence("Del"))
            rm.triggered.connect(
                lambda: self._remove_list_element(item.parent(), item))
        elif kind == _KIND_GROUP:
            menu.addAction(ic("unfold_more"), "Expand subtree",
                           lambda: self._set_subtree_expanded(item, True))
            menu.addAction(ic("unfold_less"), "Collapse subtree",
                           lambda: self._set_subtree_expanded(item, False))
            sec = menu.addAction(ic("refresh"), "Reset section")
            sec.triggered.connect(lambda: self._reset_row(item))
        return menu

    def _set_subtree_expanded(self, item, expanded):
        item.setExpanded(expanded)
        for j in range(item.childCount()):
            self._set_subtree_expanded(item.child(j), expanded)

    # -- keyboard -----------------------------------------------------------
    def _install_shortcuts(self):
        def shortcut(sequence, slot):
            sc = QShortcut(QKeySequence(sequence), self.tree_widget)
            sc.setContext(Qt.WidgetWithChildrenShortcut)
            sc.activated.connect(slot)

        shortcut("Ins", self._add_to_current_list)
        shortcut("Del", self._remove_current_element)
        shortcut("Ctrl+Backspace", self._reset_current)
        shortcut(QKeySequence.Find, self._focus_search)      # Ctrl+F
        shortcut("F3", self._find_next)
        shortcut("Shift+F3", self._find_prev)
        shortcut("Esc", self._clear_search)

    def _current_list(self):
        item = self.tree_widget.currentItem()
        if item is None:
            return None
        kind = item.data(0, _ROLE_KIND)
        if kind in (_KIND_LIST, _KIND_GROUP_LIST):
            return item
        if kind in (_KIND_ELEMENT, _KIND_GROUP_ELEMENT):
            return item.parent()
        return None

    def _add_to_current_list(self):
        target = self._current_list()
        if target is None:
            return
        if target.data(0, _ROLE_KIND) == _KIND_GROUP_LIST:
            self._add_group_element(target)
        else:
            self._add_list_element(target)

    def _remove_current_element(self):
        item = self.tree_widget.currentItem()
        if item is not None and item.data(0, _ROLE_KIND) in (
                _KIND_ELEMENT, _KIND_GROUP_ELEMENT):
            self._remove_list_element(item.parent(), item)

    def _reset_current(self):
        item = self.tree_widget.currentItem()
        if item is not None:
            self._reset_row(item)

    def _focus_search(self):
        if self.search_bar is not None:
            self.search_bar.focus()

    def _find_next(self):
        if self.search_bar is not None:
            self.search_bar.find_next()

    def _find_prev(self):
        if self.search_bar is not None:
            self.search_bar.find_prev()

    def _clear_search(self):
        if self.search_bar is not None:
            self.search_bar.clear()

    # -- search index -------------------------------------------------------
    def _reindex(self):
        """Rebuild the search index from the live tree (keys/info/values)."""
        self._index.clear()

        def walk(item):
            editor = self.tree_widget.itemWidget(item, 1)
            value = _read_editor(editor) if editor is not None else None
            self._index.append(item, item.text(0) or None, item.text(2) or None,
                               value if not isinstance(value, list) else None)
            for j in range(item.childCount()):
                walk(item.child(j))

        for i in range(self.tree_widget.topLevelItemCount()):
            walk(self.tree_widget.topLevelItem(i))

    # -- reading back -------------------------------------------------------
    def get_settings(self):
        """Return a new :class:`SettingsTree` with the edited values.

        The schema (info/format/constraints) is preserved; only ``value``
        fields are updated from the editors. Rows are matched to schema nodes by
        key, so the result is robust to row order and structural rows.
        """
        root = copy.deepcopy(self._source)
        by_key = {child.get("key"): child for child in root.get("items", [])}
        for i in range(self.tree_widget.topLevelItemCount()):
            item = self.tree_widget.topLevelItem(i)
            node = by_key.get(item.data(0, _ROLE_KEY))
            if node is not None:
                self._read_into_node(item, node)
        return SettingsTree(root)

    def get_values(self):
        """Return the edited values as a plain nested ``{key: value}`` dict."""
        return self.get_settings().to_dict()

    def _read_into_node(self, item, node):
        kind = item.data(0, _ROLE_KIND)
        if kind == _KIND_GROUP:
            by_key = {c.get("key"): c for c in node.get("items", [])}
            for j in range(item.childCount()):
                child_item = item.child(j)
                child_node = by_key.get(child_item.data(0, _ROLE_KEY))
                if child_node is not None:
                    self._read_into_node(child_item, child_node)
        elif kind == _KIND_GROUP_LIST:
            # rebuild the element list from the current GUI rows (adds/removes included);
            # each element is a fresh template clone with its leaf editors read in.
            template = node["element_group"]
            node["items"] = []
            for j in range(item.childCount()):
                element_item = item.child(j)
                if element_item.data(0, _ROLE_KIND) != _KIND_GROUP_ELEMENT:
                    continue                         # skip the trailing Add-button row
                element = copy.deepcopy(template)
                by_key = {c.get("key"): c for c in element.get("items", [])}
                for k in range(element_item.childCount()):
                    leaf_item = element_item.child(k)
                    leaf_node = by_key.get(leaf_item.data(0, _ROLE_KEY))
                    if leaf_node is not None:
                        self._read_into_node(leaf_item, leaf_node)
                node["items"].append(element)
        elif kind == _KIND_LIST:
            node["value"] = self._read_list(item)
        elif kind == _KIND_LEAF:
            editor = self.tree_widget.itemWidget(item, 1)
            if editor is not None:
                node["value"] = _read_editor(editor)

    def _read_list(self, list_item):
        values = []
        for j in range(list_item.childCount()):
            child = list_item.child(j)
            kind = child.data(0, _ROLE_KIND)
            if kind == _KIND_ADD:
                continue
            if kind == _KIND_LIST:
                values.append(self._read_list(child))
            else:
                editor = self.tree_widget.itemWidget(child, 1)
                if editor is not None:
                    values.append(_read_editor(editor))
        return values


class TreeViewer(QMainWindow):
    """A standalone window hosting a :class:`SettingsTreeWidget`.

    Adds a slim toolbar (Expand all / Collapse all / Reset all). The window is
    not shown until :meth:`show` is called.

    Parameters
    ----------
    jdata : SettingsTree or dict or list
        The schema to edit.
    parent : QWidget, optional
        Qt parent.
    """

    def __init__(self, jdata, parent=None):
        super().__init__(parent)
        self.widget = SettingsTreeWidget(jdata)
        self.setCentralWidget(self.widget)
        self.setWindowTitle("Settings tree viewer")
        self.resize(*_WINDOW_SIZE)
        self._build_toolbar()

    def _build_toolbar(self):
        bar = add_main_toolbar(self)
        tree = self.widget.tree_widget
        ic = medusa_style.qt.icon  # theme-recoloring SVGs
        bar.addAction(ic("unfold_more"), "Expand all", tree.expandAll)
        bar.addAction(ic("unfold_less"), "Collapse all", tree.collapseAll)
        bar.addAction(ic("refresh"), "Reset all", self.widget.reset_all)
        bar.addSeparator()
        bar.addAction(ic("view_column"), "Fit columns",
                      self.widget.resize_columns_to_contents)
        wrap = bar.addAction(ic("wrap_text"), "Wrap info")
        wrap.setCheckable(True)
        wrap.toggled.connect(self.widget.set_info_wrap)
        # Keep every action visible: pin the toolbar so its buttons never spill
        # into a "»" overflow menu when the window is narrowed.
        pin_toolbar_width(bar)

    def get_settings(self):
        """Return the edited :class:`SettingsTree` (see
        :meth:`SettingsTreeWidget.get_settings`)."""
        return self.widget.get_settings()
