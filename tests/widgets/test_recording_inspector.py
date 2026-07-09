"""Tests for the Recording metadata inspector.

Split across the three layers: the new core engine mutators (Qt-free), the
``inspect`` adapter (Qt-free), and the Qt widget (offscreen platform). Modal
dialogs are patched to no-ops so the widget's error paths do not block headless.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QLineEdit  # noqa: E402

from medusa.core.data import ChannelSet, Signal  # noqa: E402
from medusa.core.data.channels import Channel, Sensor  # noqa: E402
from medusa.core.data.recording import Recording, BidsInfo  # noqa: E402
from medusa.core.data.events import Events  # noqa: E402
from medusa.core.data._bids import is_derived_sidecar_field  # noqa: E402
from medusa.widgets.recording_inspector import inspect as I  # noqa: E402
import medusa.widgets.recording_inspector.recording_inspector as ri  # noqa: E402
from medusa.widgets.recording_inspector import (  # noqa: E402
    RecordingInspectorWidget, RecordingInspectorWindow)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(autouse=True)
def _no_dialogs(monkeypatch):
    """Neutralise modal dialogs so headless error paths don't block."""
    class _Box:
        warning = information = critical = staticmethod(lambda *a, **k: None)
    monkeypatch.setattr(ri, "QMessageBox", _Box)


def _recording():
    cs = ChannelSet().add_unipolar_eeg_channels(
        ["Fz", "Cz", "Pz"], reference="M1", ground="AFz")
    cs.add_sensors([Sensor("VEOG_up"), Sensor("VEOG_dn")])
    cs.add_channels(Channel("VEOG", "VEOG", "uV", sensor="VEOG_up",
                            reference="VEOG_dn", reference_method="bipolar"))
    eeg = Signal(np.zeros((5000, 4)), fs=250.0, channel_set=cs)
    acc_cs = ChannelSet().add_channels(
        [Channel(f"a{ax}", "ACCEL", "m/s^2") for ax in "xyz"])
    acc = Signal(np.zeros((2000, 3)), fs=100.0, channel_set=acc_cs)
    rec = Recording(BidsInfo("01", session="1", task="rest", run=1,
                             participant={"age": 25}))
    rec.add_signal("eeg", eeg).add_signal("acc", acc)
    rec.set_sidecar("eeg", TaskName="resting", PowerLineFrequency=50.0)
    rec.set_sidecar("acc", TaskName="resting")
    ev = Events(optional_columns={"trial_type": str})
    ev.append([{"onset": 1.0, "duration": 0.5, "trial_type": "target"},
               {"onset": 2.0, "duration": 0.5, "trial_type": "nontarget"}])
    rec.set_events(ev)
    rec.set_experiment({"paradigm": "oddball"})
    return rec


# --------------------------------------------------------------------------- #
# Engine mutators (core, no Qt)
# --------------------------------------------------------------------------- #
def test_is_derived_sidecar_field():
    assert is_derived_sidecar_field("SamplingFrequency")
    assert is_derived_sidecar_field("RecordingDuration")
    assert is_derived_sidecar_field("EEGChannelCount")
    assert not is_derived_sidecar_field("MotionChannelCount")  # total, not mapped
    assert not is_derived_sidecar_field("TaskName")


def test_rename_channel_unique_no_cascade():
    cs = ChannelSet().add_unipolar_eeg_channels(["Fz", "Cz"], reference="M1")
    cs.rename_channel("Fz", "Fz1")
    assert cs.labels == ["Fz1", "Cz"]
    assert cs.get_channel("Fz1").sensor == "Fz"   # sensor link untouched
    with pytest.raises(ValueError):
        cs.rename_channel("Cz", "Fz1")


def test_rename_sensor_cascades():
    cs = ChannelSet().add_unipolar_eeg_channels(["Fz", "Cz"], reference="M1")
    assert cs.channels_linking_sensor("M1") == ["Fz", "Cz"]
    cs.rename_sensor("M1", "M1x")
    assert "M1x" in cs.sensors and "M1" not in cs.sensors
    assert cs.get_sensor("M1x").label == "M1x"
    assert all(cs.get_channel(u).reference == "M1x" for u in ("Fz", "Cz"))
    with pytest.raises(ValueError):
        cs.rename_sensor("Fz", "Cz")     # collision with an existing sensor


def test_recording_rename_data_preserves_draft():
    rec = _recording()
    draft = dict(rec.sidecars["eeg"])
    rec.rename_data("eeg", "amp2")
    assert "amp2" in rec.data and "eeg" not in rec.data
    assert rec.sidecars["amp2"] == draft          # draft kept, not reseeded
    with pytest.raises(KeyError):
        rec.rename_data("missing", "x")


# --------------------------------------------------------------------------- #
# Qt-free inspect layer
# --------------------------------------------------------------------------- #
def test_outline_and_summary():
    rec = _recording()
    ids = _node_ids(I.outline(rec))
    assert {"identity", "shared", "streams", "stream:eeg", "sidecar:eeg",
            "channels:eeg", "sensors:eeg", "events", "experiment"} <= set(ids)
    summ = I.recording_summary(rec)
    assert summ["basename"] == "sub-01_ses-1_task-rest_run-1"
    assert summ["n_events"] == 2


def test_sidecar_rows_mark_derived():
    rec = _recording()
    rows = I.sidecar_rows("eeg", rec.sidecars["eeg"], rec.signals["eeg"])
    derived = {r["field"]: r for r in rows if r["is_derived"]}
    assert derived["EEGChannelCount"]["derived_value"] == 3
    assert derived["SamplingFrequency"]["derived_value"] == 250.0


def test_validate_subject_required():
    rec = _recording()
    assert not [i for i in I.validate_recording(rec) if i.severity == "error"]
    rec.bids.subject = ""
    errors = [i for i in I.validate_recording(rec) if i.severity == "error"]
    assert any("subject" in e.message for e in errors)


def test_clone_and_dirty_projection():
    rec = _recording()
    c = I.clone(rec)
    assert I.metadata_state(c) == I.metadata_state(rec)
    c.bids.subject = "02"
    assert I.metadata_state(c) != I.metadata_state(rec)


def _node_ids(nodes):
    out = []
    for n in nodes:
        out.append(n.node_id)
        out += _node_ids(n.children)
    return out


# --------------------------------------------------------------------------- #
# Qt widget
# --------------------------------------------------------------------------- #
def test_builds_every_page(qapp):
    rec = _recording()
    w = RecordingInspectorWidget(rec)
    for nid in _node_ids(I.outline(w._working)):
        w._nav.setCurrentItem(w._find_item(nid))
        assert w._current_page is not None, nid


def test_label_edit_validates(qapp):
    edit = ri._LabelEdit("subject", "01", required=True)
    committed = []
    edit.committed.connect(committed.append)
    edit.setText("bad id")
    edit._on_finished()
    assert not committed                       # invalid -> no commit
    edit.setText("02")
    edit._on_finished()
    assert committed == ["02"]


def test_edit_subject_apply_in_place(qapp):
    rec = _recording()
    w = RecordingInspectorWidget(rec)
    w._nav.setCurrentItem(w._find_item("identity"))
    w._current_page._set_entity("subject", "02")
    assert w.is_dirty()
    assert w.apply() is True
    assert rec.bids.subject == "02"            # live object mutated in place
    assert not w.is_dirty()


def test_apply_blocked_on_missing_subject(qapp):
    rec = _recording()
    w = RecordingInspectorWidget(rec)
    w._working.bids.subject = ""
    w.notify_change()
    assert w.apply() is False                  # dialog patched to no-op
    assert rec.bids.subject == "01"            # unchanged


def test_shared_broadcast_and_revert(qapp):
    rec = _recording()
    w = RecordingInspectorWidget(rec)
    w._working.set_sidecar(None, Manufacturer="BrainProducts")
    w.notify_change()
    assert w.is_dirty()
    w.apply()
    assert rec.sidecars["eeg"]["Manufacturer"] == "BrainProducts"
    assert rec.sidecars["acc"]["Manufacturer"] == "BrainProducts"
    # revert restores the snapshot
    w._working.bids.subject = "99"
    w.notify_change()
    assert w.is_dirty()
    w.revert()
    assert not w.is_dirty()
    assert w._working.bids.subject == "01"


def test_post_apply_edits_isolated_until_next_apply(qapp):
    rec = _recording()
    w = RecordingInspectorWidget(rec)
    w._current_page  # noqa: B018  (ensure a page exists)
    w._working.bids.subject = "02"
    w.notify_change()
    assert w.apply() is True and rec.bids.subject == "02"
    # A second round of edits must NOT leak into the live object before Apply...
    w._working.bids.subject = "03"
    w.notify_change()
    assert rec.bids.subject == "02", "post-apply edit leaked into the live object"
    # ...and Revert must roll them back cleanly.
    w.revert()
    assert not w.is_dirty()
    assert w._working.bids.subject == "02" and rec.bids.subject == "02"


def test_rename_data_via_widget_apply(qapp):
    rec = _recording()
    w = RecordingInspectorWidget(rec)
    w._working.rename_data("eeg", "amp2")
    w.rebuild_all()
    w.notify_change()
    w.apply()
    assert "amp2" in rec.data and "amp2" in rec.sidecars


def test_participant_edit_via_dict_editor(qapp):
    rec = _recording()
    w = RecordingInspectorWidget(rec)
    w._nav.setCurrentItem(w._find_item("identity"))
    name, value = QLineEdit(), QLineEdit()
    name.setText("handedness")
    value.setText("right")
    w._current_page._participant._add(name, value)
    w.apply()
    assert rec.bids.participant["handedness"] == "right"


def test_window_constructs(qapp):
    win = RecordingInspectorWindow(_recording())
    assert win.widget.get_recording() is not None
