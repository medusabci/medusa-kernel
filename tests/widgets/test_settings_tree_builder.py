"""Tests for the developer-mode schema builder (Qt, offscreen)."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from medusa.core.settings_tree import SettingsTree  # noqa: E402
from medusa.widgets.settings_tree import SettingsTreeBuilder  # noqa: E402


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def schema():
    s = SettingsTree()
    s.add_item("rate", value=0.2, value_range=[0, None], info="update rate")
    grp = s.add_group("filter")
    grp.add_item("apply", value=True)
    grp.add_item("type", value="highpass",
                 value_options=["highpass", "lowpass"])
    grp.add_item("bands", value=[1.0, 30.0],
                 element_schema={"input_format": "doublespinbox"})
    return s


class TestLoad:
    def test_loads_existing_schema(self, qapp, schema):
        builder = SettingsTreeBuilder(schema)
        assert builder.to_settings().to_dict() == schema.to_dict()

    def test_structure_mirrors_tree(self, qapp, schema):
        builder = SettingsTreeBuilder(schema)
        top = builder.tree_widget
        assert top.topLevelItemCount() == 2
        assert top.topLevelItem(0).text(0) == "rate"
        assert top.topLevelItem(1).text(1) == "Group"      # type column

    def test_node_type_classification(self, qapp, schema):
        builder = SettingsTreeBuilder(schema)
        types = {builder._node_type(n): n
                 for n in [schema.get_item("rate").tree,
                           schema.get_item("filter").tree,
                           schema.get_item("filter", "apply").tree,
                           schema.get_item("filter", "type").tree,
                           schema.get_item("filter", "bands").tree]}
        assert set(types) == {"Float", "Group", "Boolean", "Choice", "List"}


class TestEditing:
    def test_edit_default_and_info(self, qapp, schema):
        builder = SettingsTreeBuilder(schema)
        node = schema.get_item("rate").tree
        builder.select_node(node)
        builder._default_editor.setValue(0.5)
        builder.info_edit.setText("new info")
        builder._apply()
        out = builder.to_settings()
        assert out.get_item_value("rate") == 0.5
        assert out.get_item("rate").info == "new info"

    def test_change_range(self, qapp, schema):
        builder = SettingsTreeBuilder(schema)
        builder.select_node(schema.get_item("rate").tree)
        builder.min_edit.setText("0")
        builder.max_edit.setText("10")
        builder._apply()
        assert builder.to_settings().get_item("rate").tree["value_range"] \
            == [0.0, 10.0]

    def test_edit_choice_options(self, qapp, schema):
        builder = SettingsTreeBuilder(schema)
        builder.select_node(schema.get_item("filter", "type").tree)
        builder.options_edit.setPlainText("highpass\nlowpass\nbandpass")
        builder._apply()
        out = builder.to_settings()
        assert out.get_item("filter", "type").tree["value_options"] == \
            ["highpass", "lowpass", "bandpass"]

    def test_rename_key(self, qapp, schema):
        builder = SettingsTreeBuilder(schema)
        builder.select_node(schema.get_item("rate").tree)
        builder.key_edit.setText("update_rate")
        builder._apply()
        assert "update_rate" in builder.to_settings().to_dict()

    def test_duplicate_key_rejected(self, qapp, schema):
        builder = SettingsTreeBuilder(schema)
        builder.select_node(schema.get_item("filter", "apply").tree)
        builder.key_edit.setText("type")          # sibling already 'type'
        builder._apply()
        # unchanged: still 'apply'
        assert "apply" in builder.to_settings().get_item("filter").to_dict()


class TestStructure:
    def test_add_item_at_root(self, qapp):
        builder = SettingsTreeBuilder()
        builder._add("item")
        assert builder.tree_widget.topLevelItemCount() == 1
        assert "new_item" in builder.to_settings().to_dict()

    def test_add_group_then_child(self, qapp):
        builder = SettingsTreeBuilder()
        builder._add("group")
        group_node = builder._tree.tree["items"][0]
        builder.select_node(group_node)
        builder._add("item")              # added inside the selected group
        assert builder._tree.tree["items"][0]["items"][0]["key"] == "new_item"

    def test_unique_keys_on_repeated_add(self, qapp):
        builder = SettingsTreeBuilder()
        builder._add("item")
        builder.tree_widget.setCurrentItem(None)
        builder._current_node = None
        builder._add("item")
        keys = [n["key"] for n in builder._tree.tree["items"]]
        assert keys == ["new_item", "new_item_1"]

    def test_delete(self, qapp, schema):
        builder = SettingsTreeBuilder(schema)
        builder.select_node(schema.get_item("rate").tree)
        builder._delete()
        assert "rate" not in builder.to_settings().to_dict()

    def test_move(self, qapp, schema):
        builder = SettingsTreeBuilder(schema)
        builder.select_node(schema.get_item("filter").tree)
        builder._move(-1)                 # move 'filter' above 'rate'
        keys = [n["key"] for n in builder._tree.tree["items"]]
        assert keys == ["filter", "rate"]


class TestRoundTripWithPreview:
    def test_build_from_scratch_round_trips(self, qapp, tmp_path):
        builder = SettingsTreeBuilder()
        builder._add("item")
        node = builder._tree.tree["items"][0]
        builder.select_node(node)
        builder.key_edit.setText("threshold")
        builder.type_combo.setCurrentText("Float")
        builder._default_editor.setValue(0.75)
        builder._apply()

        path = tmp_path / "schema.json"
        builder.to_settings().to_json(str(path))
        reloaded = SettingsTree.from_json(str(path))
        assert reloaded.to_dict() == {"threshold": 0.75}

    def test_preview_builds_enduser_widget(self, qapp, schema):
        builder = SettingsTreeBuilder(schema)
        builder._preview_()
        assert builder._preview is not None
        assert builder._preview.get_settings().to_dict() == schema.to_dict()
