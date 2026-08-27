"""Tests for :mod:`medusa.core.settings_tree` (Qt-free)."""

from __future__ import annotations

import warnings

import pytest

from medusa.core.settings_tree import (
    INPUT_FORMATS,
    SettingsTree,
    infer_input_format,
)


@pytest.fixture
def sample_tree():
    """A small representative schema used across tests."""
    s = SettingsTree()
    s.add_item("update_rate", value=0.2, info="Update rate (s)",
               value_range=[0, None])
    freq = s.add_group("frequency_filter", info="IIR filter")
    freq.add_item("apply", value=True)
    freq.add_item("type", value="highpass",
                  value_options=["highpass", "lowpass", "bandpass"])
    freq.add_item("order", value=5, value_range=[1, None])
    freq.add_item("cutoff_freq", value=[1.0],
                  element_schema={"value_range": [0, None]})
    return s


# ---------------------------------------------------------------------------
# infer_input_format
# ---------------------------------------------------------------------------
class TestInferInputFormat:
    @pytest.mark.parametrize("value, fmt", [
        (True, "checkbox"),          # bool before int
        (5, "spinbox"),
        (0.2, "doublespinbox"),
        ("hi", "lineedit"),
        ([1, 2], "list"),
        (None, None),
    ])
    def test_by_type(self, value, fmt):
        assert infer_input_format(value) == fmt

    def test_options_force_combobox(self):
        assert infer_input_format(5, [3, 5, 7]) == "combobox"

    def test_all_formats_are_known(self):
        for fmt in ("checkbox", "spinbox", "doublespinbox", "lineedit",
                    "combobox", "list"):
            assert fmt in INPUT_FORMATS


# ---------------------------------------------------------------------------
# Building + plain-dict bridge
# ---------------------------------------------------------------------------
class TestBuildAndToDict:
    def test_to_dict_nested(self, sample_tree):
        assert sample_tree.to_dict() == {
            "update_rate": 0.2,
            "frequency_filter": {
                "apply": True, "type": "highpass", "order": 5,
                "cutoff_freq": [1.0],
            },
        }

    def test_values_is_alias_of_to_dict(self, sample_tree):
        assert sample_tree.values() == sample_tree.to_dict()

    def test_from_dict_round_trip(self, sample_tree):
        cfg = sample_tree.to_dict()
        assert SettingsTree.from_dict(cfg).to_dict() == cfg

    def test_from_dict_infos(self):
        tree = SettingsTree.from_dict(
            {"a": 1, "g": {"b": 2}},
            infos={"a": "help a", "g": {"b": "help b"}})
        assert tree.get_item("a").info == "help a"
        assert tree.get_item("g", "b").info == "help b"

    def test_default_equals_value_on_build(self, sample_tree):
        order = sample_tree.get_item("frequency_filter", "order")
        assert order.value == order.default == 5

    def test_add_group_is_branch(self, sample_tree):
        assert sample_tree.get_item("frequency_filter").is_group
        assert not sample_tree.get_item("update_rate").is_group


# ---------------------------------------------------------------------------
# input_format: the C1 regression — explicit format must be honored
# ---------------------------------------------------------------------------
class TestInputFormat:
    def test_explicit_consistent_format_accepted(self):
        s = SettingsTree()
        # consistent with type -> not stored (widget infers), but no error
        s.add_item("order", value=5, input_format="spinbox")
        s.add_item("flag", value=True, input_format="checkbox")
        s.add_item("type", value="a", input_format="combobox",
                   value_options=["a", "b"])

    def test_override_format_is_stored(self):
        s = SettingsTree()
        # lineedit over a value-less item overrides the (None) inference
        s.add_item("name", input_format="lineedit")
        assert s.get_item("name").tree.get("input_format") == "lineedit"

    @pytest.mark.parametrize("value, fmt", [
        (5, "checkbox"),         # int is not bool
        ("a", "spinbox"),        # str is not int
        (1, "doublespinbox"),    # int is not float
    ])
    def test_inconsistent_format_raises(self, value, fmt):
        with pytest.raises(ValueError):
            SettingsTree().add_item("x", value=value, input_format=fmt)

    def test_combobox_without_options_raises(self):
        with pytest.raises(ValueError):
            SettingsTree().add_item("x", value="a", input_format="combobox")

    def test_unknown_format_raises(self):
        with pytest.raises(ValueError):
            SettingsTree().add_item("x", value="a", input_format="nope")


# ---------------------------------------------------------------------------
# Error policy: raise on programmer errors
# ---------------------------------------------------------------------------
class TestErrorPolicy:
    def test_bad_value_type_raises(self):
        with pytest.raises(TypeError):
            SettingsTree().add_item("x", value=object())

    def test_none_key_raises(self):
        with pytest.raises(TypeError):
            SettingsTree().add_item(None, value=1)

    def test_bad_info_raises(self):
        with pytest.raises(TypeError):
            SettingsTree().add_item("x", value=1, info=123)

    def test_bad_value_range_raises(self):
        with pytest.raises(ValueError):
            SettingsTree().add_item("x", value=1, value_range=[1])

    def test_duplicate_sibling_key_raises(self):
        s = SettingsTree()
        s.add_item("x", value=1)
        with pytest.raises(ValueError):
            s.add_item("x", value=2)

    def test_add_item_never_returns_none(self):
        # fluent chaining must not blow up
        s = SettingsTree()
        child = s.add_item("x", value=1)
        assert isinstance(child, SettingsTree)

    def test_out_of_range_value_warns_not_raises(self):
        with pytest.warns(UserWarning, match="outside value_range"):
            SettingsTree().add_item("x", value=-5, value_range=[0, None])


# ---------------------------------------------------------------------------
# Navigation / read accessors
# ---------------------------------------------------------------------------
class TestNavigation:
    def test_get_item_value_leaf(self, sample_tree):
        assert sample_tree.get_item_value("frequency_filter", "order") == 5

    def test_get_item_value_branch_returns_nested(self, sample_tree):
        assert sample_tree.get_item_value("frequency_filter") == \
            sample_tree.get_item("frequency_filter").to_dict()

    def test_get_item_value_missing_value_returns_default(self):
        s = SettingsTree()
        s.add_group("g")        # value-less branch is fine
        leaf = SettingsTree()
        leaf.add_item("empty")  # leaf with no value
        assert leaf.get_item_value("empty", default="x") == "x"

    def test_get_item_missing_key_raises(self, sample_tree):
        with pytest.raises(KeyError):
            sample_tree.get_item("frequency_filter", "missing")

    def test_dunders(self, sample_tree):
        assert len(sample_tree) == 2
        assert "frequency_filter" in sample_tree
        assert sample_tree["frequency_filter"]["order"].value == 5
        assert [c.key for c in sample_tree] == \
            ["update_rate", "frequency_filter"]
        assert "frequency_filter" in repr(sample_tree)

    def test_remove_item(self, sample_tree):
        sample_tree.remove_item("frequency_filter", "order")
        with pytest.raises(KeyError):
            sample_tree.get_item("frequency_filter", "order")


# ---------------------------------------------------------------------------
# Editing / reset / overrides
# ---------------------------------------------------------------------------
class TestEditing:
    def test_set_value_and_get(self, sample_tree):
        sample_tree.set_value("frequency_filter", "order", value=9)
        assert sample_tree.get_item_value("frequency_filter", "order") == 9

    def test_edit_changes_value_not_default(self, sample_tree):
        order = sample_tree.get_item("frequency_filter", "order")
        order.edit_item(value=9)
        assert order.value == 9 and order.default == 5

    def test_reset_restores_default(self, sample_tree):
        sample_tree.set_value("frequency_filter", "order", value=9)
        sample_tree.reset()
        assert sample_tree.get_item_value("frequency_filter", "order") == 5

    def test_user_overrides(self, sample_tree):
        sample_tree.set_value("frequency_filter", "order", value=9)
        assert sample_tree.user_overrides() == \
            {"frequency_filter": {"order": 9}}
        sample_tree.reset()
        assert sample_tree.user_overrides() == {}

    def test_value_property_setter(self, sample_tree):
        sample_tree["update_rate"].value = 0.5
        assert sample_tree.get_item_value("update_rate") == 0.5

    def test_update_from_dict(self, sample_tree):
        sample_tree.update_from_dict({"frequency_filter": {"order": 8}})
        assert sample_tree.get_item_value("frequency_filter", "order") == 8

    def test_update_from_dict_unknown_key_warns(self, sample_tree):
        with pytest.warns(UserWarning, match="unknown key"):
            sample_tree.update_from_dict({"nope": 1})


# ---------------------------------------------------------------------------
# validate / coerce
# ---------------------------------------------------------------------------
class TestValidateCoerce:
    def test_validate_clean(self, sample_tree):
        assert sample_tree.validate() == []

    def test_validate_range_violation(self, sample_tree):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sample_tree.set_value("frequency_filter", "order", value=-3)
        violations = sample_tree.validate()
        assert violations == [("frequency_filter.order", "-3 < min 1")]

    def test_validate_option_violation(self, sample_tree):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sample_tree.set_value("frequency_filter", "type", value="zzz")
        assert sample_tree.validate()[0][0] == "frequency_filter.type"

    def test_coerce_clamps_range(self, sample_tree):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sample_tree.set_value("frequency_filter", "order", value=-3)
        sample_tree.coerce()
        assert sample_tree.get_item_value("frequency_filter", "order") == 1
        assert sample_tree.validate() == []

    def test_coerce_snaps_options_to_default(self, sample_tree):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sample_tree.set_value("frequency_filter", "type", value="zzz")
        sample_tree.coerce()
        assert sample_tree.get_item_value("frequency_filter", "type") == \
            "highpass"


# ---------------------------------------------------------------------------
# Serialization (JSON only)
# ---------------------------------------------------------------------------
class TestSerialization:
    def test_serializable_round_trip(self, sample_tree):
        obj = sample_tree.to_serializable_obj()
        assert SettingsTree.from_serializable_obj(obj).to_dict() == \
            sample_tree.to_dict()

    def test_json_round_trip(self, sample_tree, tmp_path):
        path = tmp_path / "settings.json"
        sample_tree.to_json(str(path))
        loaded = SettingsTree.from_json(str(path))
        assert loaded.to_dict() == sample_tree.to_dict()

    def test_no_serializable_component_base(self):
        # clean-break: SettingsTree is standalone JSON, not the multi-format base
        assert not hasattr(SettingsTree, "save_to_mat")
        assert not hasattr(SettingsTree, "update_tree_from_widget")


# ---------------------------------------------------------------------------
# group-lists (a variable-length list of same-schema groups)
# ---------------------------------------------------------------------------
@pytest.fixture
def filterbank_tree():
    """A ``freq_filtering`` group holding a ``filterbank`` group-list (one default filter)."""
    s = SettingsTree()
    ff = s.add_group("freq_filtering")
    fb = ff.add_group_list("filterbank", info="Parallel sub-band filters")
    elem = fb.element
    elem.add_item("filt_type", value="iir", value_options=["iir", "fir"])
    elem.add_item("band_type", value="bandpass",
                  value_options=["bandpass", "bandstop", "lowpass", "highpass"])
    elem.add_item("cutoff", value=[1.0, 70.0])
    elem.add_item("order", value=5, value_range=[1, None])
    fb.add_element()
    s.snapshot_defaults()          # baseline the group-list default (as Configurable does)
    return s


class TestGroupList:
    DEFAULT = {"filt_type": "iir", "band_type": "bandpass",
               "cutoff": [1.0, 70.0], "order": 5}

    def test_is_group_list_and_to_dict_projects_to_list(self, filterbank_tree):
        fb = filterbank_tree.get_item("freq_filtering", "filterbank")
        assert fb.is_group_list and fb.is_group
        assert filterbank_tree.to_dict() == {"freq_filtering": {"filterbank": [self.DEFAULT]}}

    def test_add_element_clones_template_with_overrides(self, filterbank_tree):
        fb = filterbank_tree.get_item("freq_filtering", "filterbank")
        fb.add_element({"band_type": "bandstop", "cutoff": [48.0, 52.0], "order": 4})
        bank = filterbank_tree.to_dict()["freq_filtering"]["filterbank"]
        assert len(bank) == 2
        assert bank[1] == {"filt_type": "iir", "band_type": "bandstop",
                           "cutoff": [48.0, 52.0], "order": 4}

    def test_partial_override_fills_template_defaults(self, filterbank_tree):
        fb = filterbank_tree.get_item("freq_filtering", "filterbank")
        fb.set_elements([{"cutoff": [6.0, 40.0]}])          # only cutoff given
        assert fb.to_dict() == [{"filt_type": "iir", "band_type": "bandpass",
                                 "cutoff": [6.0, 40.0], "order": 5}]

    def test_remove_element(self, filterbank_tree):
        fb = filterbank_tree.get_item("freq_filtering", "filterbank")
        fb.add_element()
        fb.remove_element(0)
        assert len(fb.to_dict()) == 1

    def test_update_from_dict_rebuilds_elements(self, filterbank_tree):
        filterbank_tree.update_from_dict({"freq_filtering": {"filterbank": [
            {"band_type": "bandpass", "cutoff": [6.0, 40.0], "order": 4},
            {"filt_type": "fir", "band_type": "lowpass", "cutoff": 30.0, "order": 64}]}})
        bank = filterbank_tree.to_dict()["freq_filtering"]["filterbank"]
        assert len(bank) == 2 and bank[1]["filt_type"] == "fir"

    def test_update_from_dict_rejects_non_list(self, filterbank_tree):
        with pytest.raises(ValueError, match="group-list"):
            filterbank_tree.update_from_dict({"freq_filtering": {"filterbank": {"x": 1}}})

    def test_validate_checks_each_element_with_indexed_path(self, filterbank_tree):
        fb = filterbank_tree.get_item("freq_filtering", "filterbank")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fb.add_element({"band_type": "weird", "order": 0})
        paths = {p for p, _ in filterbank_tree.validate()}
        assert "freq_filtering.filterbank.1.band_type" in paths
        assert "freq_filtering.filterbank.1.order" in paths

    def test_reset_restores_default_elements(self, filterbank_tree):
        fb = filterbank_tree.get_item("freq_filtering", "filterbank")
        fb.set_elements([{"cutoff": [1.0, 10.0]}, {"cutoff": [10.0, 20.0]}])
        filterbank_tree.reset()
        assert filterbank_tree.to_dict()["freq_filtering"]["filterbank"] == [self.DEFAULT]

    def test_edit_time_add_element_preserves_default(self, filterbank_tree):
        # add_element at edit time must NOT corrupt the build-time default
        fb = filterbank_tree.get_item("freq_filtering", "filterbank")
        fb.add_element({"cutoff": [10.0, 20.0]})
        filterbank_tree.reset()
        assert filterbank_tree.to_dict()["freq_filtering"]["filterbank"] == [self.DEFAULT]

    def test_user_overrides_reports_group_list_edits(self, filterbank_tree):
        assert filterbank_tree.user_overrides() == {}          # unedited -> no override
        fb = filterbank_tree.get_item("freq_filtering", "filterbank")
        fb.add_element({"cutoff": [10.0, 20.0]})
        ov = filterbank_tree.user_overrides()
        assert len(ov["freq_filtering"]["filterbank"]) == 2

    def test_reset_no_default_is_noop(self):
        # a group-list never snapshot_defaults()'d resets to a no-op (current == default)
        s = SettingsTree()
        fb = s.add_group_list("gl")
        fb.element.add_item("x", value=1, value_range=[0, None])
        fb.add_element()
        s.reset()
        assert s.to_dict() == {"gl": [{"x": 1}]}

    def test_default_snapshot_is_deepcopied(self):
        # the captured default must not alias (share list objects with) live elements
        s = SettingsTree()
        fb = s.add_group_list("gl")
        fb.element.add_item("cutoff", value=[1.0, 2.0])
        fb.add_element()
        s.snapshot_defaults()
        s.to_dict()["gl"][0]["cutoff"].append(99.0)      # in-place mutation of a cfg list
        s.reset()
        assert s.to_dict() == {"gl": [{"cutoff": [1.0, 2.0]}]}

    def test_nested_group_list_gets_its_own_default(self):
        s = SettingsTree()
        outer = s.add_group_list("outer")
        inner = outer.element.add_group_list("inner")
        inner.element.add_item("x", value=1, value_range=[0, None])
        inner.add_element()
        outer.add_element()
        s.snapshot_defaults()
        inner_node = s.get_item("outer").elements[0].get_item("inner")
        assert inner_node.is_group_list and "default" in inner_node.tree

    def test_json_round_trip_preserves_group_list(self, filterbank_tree, tmp_path):
        filterbank_tree.get_item("freq_filtering", "filterbank").add_element(
            {"cutoff": [8.0, 30.0]})
        path = tmp_path / "s.json"
        filterbank_tree.to_json(str(path))
        loaded = SettingsTree.from_json(str(path))
        assert loaded.to_dict() == filterbank_tree.to_dict()
        assert loaded.get_item("freq_filtering", "filterbank").is_group_list

    def test_element_mutators_require_group_list(self):
        plain = SettingsTree().add_group("g")
        with pytest.raises(TypeError):
            plain.add_element()
        with pytest.raises(TypeError):
            _ = plain.element


# ---------------------------------------------------------------------------
# console rendering (describe / print_tree / __str__)
# ---------------------------------------------------------------------------
class TestConsoleRendering:
    def test_values_detail_lists_every_leaf(self, sample_tree):
        out = sample_tree.describe()
        assert out.splitlines()[0] == "SettingsTree"
        assert "update_rate = 0.2" in out
        assert "frequency_filter" in out and "type = 'highpass'" in out
        # 'values' shows values only: no constraints, defaults or help text
        assert "range:" not in out and "options:" not in out
        assert "Update rate" not in out

    def test_full_detail_adds_constraints_and_info(self, sample_tree):
        out = sample_tree.describe(detail="full")
        assert "[range: >= 0]" in out
        assert "options: 'highpass', 'lowpass', 'bandpass'" in out
        assert "# Update rate (s)" in out
        assert "# IIR filter" in out                     # groups show their info too

    def test_full_detail_shows_default_only_when_edited(self, sample_tree):
        assert "default:" not in sample_tree.describe(detail="full")
        sample_tree.set_value("frequency_filter", "order", value=7)
        out = sample_tree.describe(detail="full")
        assert "order = 7  [default: 5 | range: >= 1]" in out

    def test_range_formats(self):
        s = SettingsTree()
        s.add_item("both", value=5, value_range=[1, 10])
        s.add_item("upper", value=5, value_range=[None, 10])
        out = s.describe(detail="full")
        assert "[range: 1..10]" in out and "[range: <= 10]" in out

    def test_unset_leaf(self):
        s = SettingsTree()
        s.add_item("montage")
        assert "montage = <not set>" in s.describe()

    def test_subtree_title_defaults_to_its_key(self, sample_tree):
        out = sample_tree.get_item("frequency_filter").describe()
        assert out.splitlines()[0] == "frequency_filter"
        assert "update_rate" not in out                  # only this subtree

    def test_explicit_title(self, sample_tree):
        assert sample_tree.describe(title="My settings").startswith("My settings")

    def test_group_list_shows_indexed_elements(self, filterbank_tree):
        filterbank_tree.get_item("freq_filtering", "filterbank").add_element()
        out = filterbank_tree.describe()
        assert "filterbank  (group-list, 2 elements)" in out
        assert "[0]" in out and "[1]" in out
        assert out.count("filt_type = 'iir'") == 2

    def test_empty_group_list_shows_template_in_full_detail(self):
        s = SettingsTree()
        s.add_group_list("bands").element.add_item("low", value=1.0)
        assert "(group-list, 0 elements)" in s.describe()
        assert "<template>" not in s.describe()          # values: elements only
        full = s.describe(detail="full")
        assert "<template>" in full and "low = 1.0" in full

    def test_ascii_guides_and_no_ansi(self, sample_tree):
        out = sample_tree.describe()
        assert "\x1b[" not in out                        # no colour escapes
        assert out.isascii()                             # printable on any console
        assert "`-- " in out                             # ASCII tree guides

    def test_markup_in_values_is_not_interpreted(self):
        s = SettingsTree()
        s.add_item("pattern", value="[bold red]x[/]", info="[i]")
        out = s.describe(detail="full")
        assert "'[bold red]x[/]'" in out and "# [i]" in out

    def test_str_is_describe_with_defaults(self, sample_tree):
        assert str(sample_tree) == sample_tree.describe()
        assert not str(sample_tree).endswith("\n")

    def test_repr_stays_terse(self, sample_tree):
        assert "\n" not in repr(sample_tree)

    def test_print_tree_writes_to_stdout(self, sample_tree, capsys):
        sample_tree.print_tree()
        assert "update_rate" in capsys.readouterr().out

    def test_invalid_detail(self, sample_tree):
        with pytest.raises(ValueError, match="'values' or 'full'"):
            sample_tree.describe(detail="everything")
