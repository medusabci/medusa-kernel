"""Events (D3)."""
import pytest

from medusa.core.data import Events


def _ev():
    return Events(optional_columns={"trial_type": str, "value": int},
                  descriptions={"trial_type": "go or nogo"})


def test_append_and_typed_schema():
    ev = _ev().append([
        {"onset": 1.0, "duration": 0.5, "trial_type": "go", "value": 1},
        {"onset": 2.0, "duration": 0.5, "trial_type": "nogo", "value": 2}])
    assert ev.n_events == 2
    assert ev.df["value"].dtype.name == "int64"
    assert ev.df["trial_type"].tolist() == ["go", "nogo"]


def test_append_requires_exact_columns():
    with pytest.raises(ValueError):
        _ev().append({"onset": 1.0, "duration": 0.5, "trial_type": "go"})  # missing value


def test_reserved_and_unknown_description_rejected():
    with pytest.raises(ValueError):
        Events(optional_columns={"onset": float})            # reserved name
    with pytest.raises(ValueError):
        Events(optional_columns={"a": str}, descriptions={"b": "x"})  # unknown col


def test_select_filters_and_keeps_schema():
    ev = _ev().append([
        {"onset": 0.0, "duration": 0.1, "trial_type": "go", "value": 1},
        {"onset": 1.0, "duration": 0.1, "trial_type": "nogo", "value": 2},
        {"onset": 2.0, "duration": 0.1, "trial_type": "go", "value": 1}])
    sel = ev.select(trial_type="go")
    assert sel.n_events == 2 and sel.column_names == ev.column_names


def test_out_of_order_onset_warns_not_raised():
    ev = _ev()
    with pytest.warns(UserWarning):
        ev.append([{"onset": 2.0, "duration": 0.1, "trial_type": "go", "value": 1},
                   {"onset": 1.0, "duration": 0.1, "trial_type": "go", "value": 1}])
    assert ev.n_events == 2                         # appended, not rejected
    ev.sort()
    assert ev.df["onset"].tolist() == [1.0, 2.0]


@pytest.mark.parametrize("fmt", ["bson", "json", "mat"])
def test_roundtrip(tmp_path, roundtrip, fmt):
    ev = _ev().append([
        {"onset": 0.5, "duration": 0.2, "trial_type": "target", "value": 1},
        {"onset": 1.5, "duration": 0.2, "trial_type": "nontarget", "value": 0}])
    back = roundtrip(ev, tmp_path, fmt)
    assert back.df["trial_type"].tolist() == ["target", "nontarget"]
    assert back.df["value"].tolist() == [1, 0]


@pytest.mark.parametrize("fmt", ["bson", "json", "mat"])
def test_roundtrip_single_event(tmp_path, roundtrip, fmt):
    # A 1-row table collapses each column to a scalar through .mat -- now restored.
    ev = _ev().append({"onset": 0.5, "duration": 0.2, "trial_type": "go", "value": 1})
    back = roundtrip(ev, tmp_path, fmt)
    assert back.n_events == 1 and back.df["trial_type"].tolist() == ["go"]
