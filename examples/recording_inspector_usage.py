"""Explore and edit recording metadata — ``medusa.widgets.RecordingInspector``.

Run:  python examples/recording_inspector_usage.py

Opens the **Recording inspector**: a master-detail editor whose left tree mirrors
a :class:`~medusa.core.data.recording.Recording` (Identity / Shared sidecar /
Streams -> Sidecar - Channels - Sensors / Events / Experiment) and whose right
pane edits the selected node. It edits *metadata only* -- BIDS identity, sidecar
fields, channel labels/types/references, sensor positions, participant/scan and
experiment dicts -- never the sample arrays. A persistent summary strip reports
dims / duration / datatypes, requirement levels and derived (read-only) fields are
surfaced explicitly, and **Apply** commits the edits into the *same* recording
object in place after validation (**Save...** writes it to disk).

Two ways to use it, both shown below:

1. ``RecordingInspector(rec).show()`` -- a standalone window (owns the QApplication).
2. ``RecordingInspectorWidget`` embedded in your own Qt app, wired to its
   ``dirty`` / ``applied`` / ``validated`` signals (the medusa-platform story);
   see :func:`build_embedded_widget`.
"""
import numpy as np

from medusa.core.data import ChannelSet, Signal
from medusa.core.data.channels import Channel, Sensor
from medusa.core.data.events import Events
from medusa.core.data.recording import BidsInfo, Recording
from medusa.widgets import RecordingInspector, RecordingInspectorWidget


# --------------------------------------------------------------------------- #
# Build a realistic multimodal run to inspect: a 31-channel EEG stream (30 EEG +
# 1 bipolar VEOG, referenced to M1, ground AFz) plus a 3-axis accelerometer.
# Only metadata matters here, so the sample arrays are just noise.
# --------------------------------------------------------------------------- #
def make_recording() -> Recording:
    rng = np.random.default_rng(0)
    eeg_labels = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC5", "FC1",
                  "FC2", "FC6", "T7", "C3", "Cz", "C4", "T8", "CP5", "CP1",
                  "CP2", "CP6", "P7", "P3", "Pz", "P4", "P8", "PO3", "PO4",
                  "O1", "Oz", "O2"]
    cs = ChannelSet().add_unipolar_eeg_channels(
        eeg_labels, reference="M1", ground="AFz", reference_method="common")
    cs.add_sensors([Sensor("VEOG_up"), Sensor("VEOG_dn")])
    cs.add_channels(Channel("VEOG", "VEOG", "uV", sensor="VEOG_up",
                            reference="VEOG_dn", reference_method="bipolar"))
    eeg = Signal(rng.standard_normal((250 * 120, 31)), fs=250.0, channel_set=cs)

    acc_cs = ChannelSet().add_channels(
        [Channel(f"acc_{ax}", "ACCEL", "m/s^2") for ax in "xyz"])
    acc = Signal(rng.standard_normal((100 * 120, 3)), fs=100.0,
                 channel_set=acc_cs)

    rec = Recording(BidsInfo(
        "01", session="1", task="rest", run=1,
        participant={"age": 27, "sex": "F", "handedness": "R"},
        scan={"acq_time": "2026-07-06T10:00:00"}))
    rec.add_signal("eeg", eeg).add_signal("acc", acc)
    # A recording-wide field (broadcast to every stream sidecar) + eeg-only fields.
    rec.set_sidecar(None, TaskName="Resting state", InstitutionName="GIB")
    rec.set_sidecar("eeg", PowerLineFrequency=50.0, EEGReference="M1")

    events = Events(optional_columns={"trial_type": str},
                    descriptions={"trial_type": "Eyes open vs closed"})
    events.append([{"onset": 0.0, "duration": 60.0, "trial_type": "eyes_open"},
                   {"onset": 60.0, "duration": 60.0, "trial_type": "eyes_closed"}])
    rec.set_events(events)
    rec.set_experiment({"paradigm": "resting_state", "protocol_version": 2})
    return rec


# --------------------------------------------------------------------------- #
# (2) Embedding the widget in your own Qt app and reacting to its signals. The
# widget edits a working clone; `applied` fires (with the now-mutated recording)
# only after a successful, validated Apply, so a host can persist / refresh then.
# --------------------------------------------------------------------------- #
def build_embedded_widget(rec: Recording) -> RecordingInspectorWidget:
    widget = RecordingInspectorWidget(rec)
    widget.dirty.connect(
        lambda pending: print(f"  [signal] dirty -> {pending}"))
    widget.validated.connect(
        lambda issues: print(f"  [signal] validated -> {len(issues)} issue(s)"))
    widget.applied.connect(
        lambda r: print(f"  [signal] applied -> {r.bids.basename()}"))
    return widget


def interactive() -> None:
    rec = make_recording()
    print("Before:", repr(rec.bids))
    RecordingInspector(rec).show()   # blocks until the window is closed
    # Apply mutates `rec` in place, so edits are visible on the same object here.
    print("After: ", repr(rec.bids))
    print("Dirty edits are committed on Apply; unsaved edits are discarded on "
          "close.")


if __name__ == "__main__":
    interactive()
