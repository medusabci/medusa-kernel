"""Tests for the Qt settings-tree editor.

PySide6 is a core dependency, so these run everywhere; Qt uses the
``offscreen`` platform plugin so no display is needed.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QCheckBox,
    QComboBox,
    QHeaderView,
    QToolBar,
)

from medusa.core.settings_tree import SettingsTree  # noqa: E402
from medusa.widgets.settings_tree import (  # noqa: E402
    SettingsTreeWidget,
    TreeViewer,
)
from medusa.widgets.settings_tree.tree_viewer import (  # noqa: E402
    _KIND_GROUP_ELEMENT,
    _ROLE_KEY,
    _ROLE_KIND,
)


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def schema():
    s = SettingsTree()
    s.add_item("update_rate", value=0.2, info="Update rate (s)",
               value_range=[0, None])
    freq = s.add_group("frequency_filter", info="IIR filter")
    freq.add_item("apply", value=True)
    freq.add_item("type", value="highpass",
                  value_options=["highpass", "lowpass", "bandpass"])
    freq.add_item("order", value=5, value_options=[3, 5, 7])   # INT options
    freq.add_item("cutoff_freq", value=[1.0, 30.0],
                  element_schema={"value_range": [0, None]})
    return s


def _find(widget, key):
    """Return ``(editor_or_None, item)`` for the leaf/list row named ``key``."""
    tw = widget.tree_widget

    def walk(item):
        for i in range(item.childCount()):
            child = item.child(i)
            if child.text(0) == key:
                return tw.itemWidget(child, 1), child
            found = walk(child)
            if found:
                return found
        return None

    for i in range(tw.topLevelItemCount()):
        top = tw.topLevelItem(i)
        if top.text(0) == key:
            return tw.itemWidget(top, 1), top
        found = walk(top)
        if found:
            return found
    return None, None


class TestRoundTrip:
    def test_baseline_equals_source(self, qapp, schema):
        w = SettingsTreeWidget(schema)
        assert w.get_values() == schema.to_dict()

    def test_get_settings_returns_settingstree(self, qapp, schema):
        w = SettingsTreeWidget(schema)
        assert isinstance(w.get_settings(), SettingsTree)

    def test_editor_kinds(self, qapp, schema):
        w = SettingsTreeWidget(schema)
        assert isinstance(_find(w, "apply")[0], QCheckBox)
        assert isinstance(_find(w, "type")[0], QComboBox)
        assert isinstance(_find(w, "order")[0], QComboBox)   # has options

    def test_combobox_preserves_int_type(self, qapp, schema):
        w = SettingsTreeWidget(schema)
        editor, _ = _find(w, "order")
        editor.setCurrentIndex([3, 5, 7].index(7))
        out = w.get_values()["frequency_filter"]["order"]
        assert out == 7 and isinstance(out, int)

    def test_checkbox_edit(self, qapp, schema):
        w = SettingsTreeWidget(schema)
        _find(w, "apply")[0].setChecked(False)
        assert w.get_values()["frequency_filter"]["apply"] is False

    def test_schema_preserved_through_round_trip(self, qapp, schema):
        w = SettingsTreeWidget(schema)
        out = w.get_settings()
        # info/options survive (get_settings only updates values)
        assert out.get_item("frequency_filter", "type").tree["value_options"] \
            == ["highpass", "lowpass", "bandpass"]


class TestListEditing:
    def test_add_uses_element_type(self, qapp, schema):
        w = SettingsTreeWidget(schema)
        _, list_item = _find(w, "cutoff_freq")
        w._add_list_element(list_item, {"value_range": [0, None]})
        vals = w.get_values()["frequency_filter"]["cutoff_freq"]
        assert vals == [1.0, 30.0, 0.0]      # float, not ""

    def test_remove_element(self, qapp, schema):
        w = SettingsTreeWidget(schema)
        _, list_item = _find(w, "cutoff_freq")
        w._remove_list_element(list_item, list_item.child(0))
        assert w.get_values()["frequency_filter"]["cutoff_freq"] == [30.0]

    def test_nested_list_round_trip(self, qapp):
        s = SettingsTree()
        s.add_item("matrix", value=[[1, 2], [3, 4]])
        w = SettingsTreeWidget(s)
        assert w.get_values() == {"matrix": [[1, 2], [3, 4]]}


@pytest.fixture
def gl_schema():
    """A ``filterbank`` group-list (one default filter) for the widget tests."""
    s = SettingsTree()
    fb = s.add_group_list("filterbank", info="Parallel sub-band filters")
    elem = fb.element
    elem.add_item("filt_type", value="iir", value_options=["iir", "fir"])
    elem.add_item("band_type", value="bandpass",
                  value_options=["bandpass", "bandstop", "lowpass", "highpass"])
    elem.add_item("cutoff", value=[1.0, 70.0])
    elem.add_item("order", value=5, value_range=[1, None])
    fb.add_element()
    s.snapshot_defaults()
    return s


class TestGroupListEditing:
    DEFAULT = {"filt_type": "iir", "band_type": "bandpass",
               "cutoff": [1.0, 70.0], "order": 5}

    @staticmethod
    def _elements(gl):
        return [gl.child(j) for j in range(gl.childCount())
                if gl.child(j).data(0, _ROLE_KIND) == _KIND_GROUP_ELEMENT]

    def test_render_round_trip(self, qapp, gl_schema):
        w = SettingsTreeWidget(gl_schema)
        assert w.get_values() == {"filterbank": [self.DEFAULT]}

    def test_add_group_element_uses_template_defaults(self, qapp, gl_schema):
        w = SettingsTreeWidget(gl_schema)
        _, gl = _find(w, "filterbank")
        w._add_group_element(gl)
        vals = w.get_values()["filterbank"]
        assert len(vals) == 2 and vals[1] == self.DEFAULT

    def test_remove_group_element(self, qapp, gl_schema):
        w = SettingsTreeWidget(gl_schema)
        _, gl = _find(w, "filterbank")
        w._add_group_element(gl)
        w._remove_list_element(gl, self._elements(gl)[0])
        assert len(w.get_values()["filterbank"]) == 1

    def test_edit_element_leaf_round_trips(self, qapp, gl_schema):
        w = SettingsTreeWidget(gl_schema)
        _, gl = _find(w, "filterbank")
        elem = self._elements(gl)[0]
        order_row = next(elem.child(k) for k in range(elem.childCount())
                         if elem.child(k).data(0, _ROLE_KEY) == "order")
        w.tree_widget.itemWidget(order_row, 1).setValue(9)
        assert w.get_values()["filterbank"][0]["order"] == 9

    def test_reset_group_list_restores_default(self, qapp, gl_schema):
        w = SettingsTreeWidget(gl_schema)
        _, gl = _find(w, "filterbank")
        w._add_group_element(gl)
        w._reset_row(gl)
        assert w.get_values()["filterbank"] == [self.DEFAULT]

    def test_reset_element_with_nested_group(self, qapp):
        # reset must restore nested-group values in an element (not the template defaults)
        s = SettingsTree()
        fb = s.add_group_list("gl")
        band = fb.element.add_group("band")
        band.add_item("low", value=1.0)
        band.add_item("high", value=40.0)
        fb.add_element({"band": {"low": 8.0, "high": 12.0}})
        s.snapshot_defaults()
        w = SettingsTreeWidget(s)
        _, gl = _find(w, "gl")
        w._reset_row(gl)
        assert w.get_values() == {"gl": [{"band": {"low": 8.0, "high": 12.0}}]}


class TestSearch:
    def test_index_covers_value_and_info(self, qapp, schema):
        w = SettingsTreeWidget(schema)
        assert w._index.find("highpass")        # a value
        assert w._index.find("Update rate")     # an info string
        assert w._index.find("frequency_filter")  # a key

    def test_search_bar_no_match_is_non_blocking(self, qapp, schema):
        w = SettingsTreeWidget(schema)
        bar = w.search_bar
        bar._box.setText("definitely-absent")
        bar.find_next()                      # must not block or raise
        assert "color" in bar._box.styleSheet()   # tinted, no modal dialog
        # a subsequent match clears the tint
        bar._box.setText("order")
        bar.find_next()
        assert bar._box.styleSheet() == ""


class TestViewer:
    def test_viewer_does_not_show_on_init(self, qapp, schema):
        viewer = TreeViewer(schema)
        assert not viewer.isVisible()
        assert viewer.get_settings().to_dict() == schema.to_dict()


def _group(widget, key):
    tw = widget.tree_widget
    for i in range(tw.topLevelItemCount()):
        if tw.topLevelItem(i).text(0) == key:
            return tw.topLevelItem(i)
    return None


def _menu_labels(menu):
    return [a.text() for a in menu.actions()]


class TestContextMenu:
    def test_leaf_reset_disabled_until_overridden(self, qapp, schema):
        w = SettingsTreeWidget(schema)
        editor, item = _find(w, "update_rate")
        reset = [a for a in w._build_context_menu(item).actions()
                 if a.text() == "Reset to default"][0]
        assert not reset.isEnabled()                 # unchanged -> disabled
        editor.setValue(0.5)
        reset2 = [a for a in w._build_context_menu(item).actions()
                  if a.text() == "Reset to default"][0]
        assert reset2.isEnabled()                    # overridden -> enabled

    def test_list_menu_has_add(self, qapp, schema):
        w = SettingsTreeWidget(schema)
        _, list_item = _find(w, "cutoff_freq")
        assert "Add element" in _menu_labels(w._build_context_menu(list_item))

    def test_element_menu_has_remove(self, qapp, schema):
        w = SettingsTreeWidget(schema)
        _, list_item = _find(w, "cutoff_freq")
        assert "Remove element" in _menu_labels(
            w._build_context_menu(list_item.child(0)))

    def test_group_menu_has_section_actions(self, qapp, schema):
        w = SettingsTreeWidget(schema)
        labels = _menu_labels(w._build_context_menu(_group(w, "frequency_filter")))
        assert {"Expand subtree", "Collapse subtree", "Reset section"} <= set(labels)


class TestOverrideAndReset:
    def test_override_marker_is_live(self, qapp, schema):
        w = SettingsTreeWidget(schema)
        editor, item = _find(w, "update_rate")
        assert not item.font(0).bold()
        editor.setValue(0.5)
        assert w._is_overridden(item)
        assert item.font(0).bold()                   # live via change signal

    def test_reset_row_restores_default(self, qapp, schema):
        w = SettingsTreeWidget(schema)
        editor, item = _find(w, "update_rate")
        editor.setValue(0.5)
        w._reset_row(item)
        assert w.get_values()["update_rate"] == 0.2
        assert not w._is_overridden(item)
        assert not item.font(0).bold()

    def test_reset_section(self, qapp, schema):
        w = SettingsTreeWidget(schema)
        _find(w, "order")[0].setCurrentIndex([3, 5, 7].index(7))
        w._reset_row(_group(w, "frequency_filter"))
        assert w.get_values()["frequency_filter"]["order"] == 5

    def test_reset_all(self, qapp, schema):
        w = SettingsTreeWidget(schema)
        _find(w, "update_rate")[0].setValue(0.9)
        _find(w, "order")[0].setCurrentIndex([3, 5, 7].index(7))
        w.reset_all()
        values = w.get_values()
        assert values["update_rate"] == 0.2
        assert values["frequency_filter"]["order"] == 5


class TestListSummaryAndReveal:
    def test_summary_text(self, qapp, schema):
        w = SettingsTreeWidget(schema)
        _, list_item = _find(w, "cutoff_freq")
        assert w._summary_label_of[list_item].text() == "[1.0, 30.0] (2)"

    def test_summary_updates_on_add(self, qapp, schema):
        w = SettingsTreeWidget(schema)
        _, list_item = _find(w, "cutoff_freq")
        w._add_list_element(list_item, {"value_range": [0, None]})
        assert w._summary_label_of[list_item].text() == "[1.0, 30.0, 0.0] (3)"

    def test_remove_button_hidden_until_current(self, qapp, schema):
        # isHidden() (explicit-hidden flag) is the right check headless: the
        # window is never shown so isVisible() is always False.
        w = SettingsTreeWidget(schema)
        _, list_item = _find(w, "cutoff_freq")
        element = list_item.child(0)
        button = w._element_remove_btn[element]
        assert button.isHidden()                     # hidden at rest
        w.tree_widget.setCurrentItem(element)
        assert not button.isHidden()                 # revealed on current row


class TestKeyboardPaths:
    def test_add_to_current_list(self, qapp, schema):
        w = SettingsTreeWidget(schema)
        _, list_item = _find(w, "cutoff_freq")
        w.tree_widget.setCurrentItem(list_item)
        w._add_to_current_list()
        assert len(w.get_values()["frequency_filter"]["cutoff_freq"]) == 3

    def test_remove_current_element(self, qapp, schema):
        w = SettingsTreeWidget(schema)
        _, list_item = _find(w, "cutoff_freq")
        w.tree_widget.setCurrentItem(list_item.child(0))
        w._remove_current_element()
        assert w.get_values()["frequency_filter"]["cutoff_freq"] == [30.0]

    def test_remove_current_noop_on_leaf(self, qapp, schema):
        w = SettingsTreeWidget(schema)
        _, item = _find(w, "update_rate")
        w.tree_widget.setCurrentItem(item)
        w._remove_current_element()                  # must be a no-op, not crash
        assert "update_rate" in w.get_values()

    def test_reset_current(self, qapp, schema):
        w = SettingsTreeWidget(schema)
        editor, item = _find(w, "update_rate")
        editor.setValue(0.7)
        w.tree_widget.setCurrentItem(item)
        w._reset_current()
        assert w.get_values()["update_rate"] == 0.2


class TestSearchUpgrade:
    def test_find_prev_and_counter(self, qapp, schema):
        w = SettingsTreeWidget(schema)
        bar = w.search_bar
        bar._box.setText("a")                        # several matches
        bar.find_next()
        assert " of " in bar._counter.text()
        bar.find_prev()
        assert " of " in bar._counter.text()

    def test_search_reaches_edited_element_values(self, qapp, schema):
        w = SettingsTreeWidget(schema)
        _, list_item = _find(w, "cutoff_freq")
        w._add_list_element(list_item, {"value_range": [0, None]})
        w.tree_widget.itemWidget(list_item.child(2), 1).setValue(99.0)
        bar = w.search_bar
        bar._box.setText("99")
        bar.find_next()
        assert bar._box.styleSheet() == ""           # reindex found the element value


class TestViewerToolbar:
    def test_toolbar_actions_present(self, qapp, schema):
        viewer = TreeViewer(schema)
        labels = [a.text() for bar in viewer.findChildren(QToolBar)
                  for a in bar.actions()]
        assert {"Expand all", "Collapse all", "Reset all"} <= set(labels)

    def test_toolbar_reset_all(self, qapp, schema):
        viewer = TreeViewer(schema)
        _find(viewer.widget, "update_rate")[0].setValue(0.9)
        viewer.widget.reset_all()
        assert viewer.get_settings().to_dict()["update_rate"] == 0.2

    def test_toolbar_never_overflows(self, qapp, schema):
        # Narrowing the window must not push toolbar buttons into a "»" overflow
        # menu: the toolbar is pinned to hold all its buttons (which propagates to
        # the window's minimum width). See tests/widgets/test_toolbar.py for the
        # shared-helper mechanism.
        viewer = TreeViewer(schema)
        bar = viewer.findChildren(QToolBar)[0]
        assert bar.minimumWidth() >= bar.sizeHint().width()


class TestViewOptions:
    def test_value_fills_key_and_info_resizable(self, qapp, schema):
        # Value fills the width (adapts to the window); Key & Info are resizable.
        w = SettingsTreeWidget(schema)
        header = w.tree_widget.header()
        assert header.sectionResizeMode(0) == QHeaderView.Interactive
        assert header.sectionResizeMode(1) == QHeaderView.Stretch
        assert header.sectionResizeMode(2) == QHeaderView.Interactive
        assert not header.stretchLastSection()

    def test_horizontal_scrollbar_as_needed(self, qapp, schema):
        from PySide6.QtCore import Qt
        w = SettingsTreeWidget(schema)
        assert w.tree_widget.horizontalScrollBarPolicy() == Qt.ScrollBarAsNeeded

    def test_resize_columns_to_contents(self, qapp, schema):
        w = SettingsTreeWidget(schema)
        w.resize_columns_to_contents()           # must not raise
        assert w.tree_widget.columnWidth(0) > 0

    def test_info_wrap_toggle(self, qapp, schema):
        from PySide6.QtCore import Qt
        w = SettingsTreeWidget(schema)
        assert not w._info_delegate.wrap                       # one-line default
        assert w.tree_widget.textElideMode() == Qt.ElideRight
        w.set_info_wrap(True)
        assert w._info_delegate.wrap
        assert w.tree_widget.textElideMode() == Qt.ElideNone   # don't elide wrapped
        w.set_info_wrap(False)
        assert not w._info_delegate.wrap
        assert w.tree_widget.textElideMode() == Qt.ElideRight

    def test_wrap_info_constructor_arg(self, qapp, schema):
        w = SettingsTreeWidget(schema, wrap_info=True)
        assert w._info_delegate.wrap

    def test_wrap_grows_info_row(self, qapp):
        # The core of the fix: wrapping a long Info string must make the row
        # taller (so nothing clips), not stay one clipped line.
        from PySide6.QtWidgets import QStyleOptionViewItem
        s = SettingsTree()
        s.add_item("k", value=1, info="word " * 60)
        w = SettingsTreeWidget(s)
        w.tree_widget.setColumnWidth(2, 140)
        idx = w.tree_widget.indexFromItem(w.tree_widget.topLevelItem(0), 2)
        delegate = w._info_delegate
        delegate.wrap = False
        one_line = delegate.sizeHint(QStyleOptionViewItem(), idx).height()
        delegate.wrap = True
        wrapped = delegate.sizeHint(QStyleOptionViewItem(), idx).height()
        assert wrapped > one_line

    # NOTE: selection-highlight color is no longer set per-widget — it comes
    # from the app-wide medusa-style theme (QSS + QPalette via
    # medusa_style.qt.application), so there is nothing widget-local to assert.

    def test_toolbar_has_view_options(self, qapp, schema):
        viewer = TreeViewer(schema)
        labels = [a.text() for bar in viewer.findChildren(QToolBar)
                  for a in bar.actions()]
        assert "Fit columns" in labels and "Wrap info" in labels
