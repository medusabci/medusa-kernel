"""Recording / BidsInfo + schema version (D4, D8)."""
import warnings

import numpy as np
import pytest

from medusa.core.data import BidsInfo, Recording, Signal, ChannelSet
from medusa.core.schema import SCHEMA_VERSION


# --------------------------------------------------------------------- BidsInfo
def test_bidsinfo_validation_and_basename():
    bids = BidsInfo("01", session="1", task="rest", run=1)
    assert bids.basename("eeg") == "sub-01_ses-1_task-rest_run-1_eeg"
    assert bids.basename("eeg", acquisition="amp2") == \
        "sub-01_ses-1_task-rest_acq-amp2_run-1_eeg"
    with pytest.raises(ValueError):
        BidsInfo("sub 01")                      # invalid label (space)
    with pytest.raises(ValueError):
        BidsInfo(None)                          # subject required


# --------------------------------------------------------------------- builders
def test_add_signal_seeds_and_autofills_sidecar(eeg_cs):
    rec = Recording(BidsInfo("01", task="rest"))
    rec.add_signal("eeg", Signal(np.zeros((500, 3)), fs=250.0, channel_set=eeg_cs))
    sc = rec.sidecars["eeg"]
    assert sc["SamplingFrequency"] == 250.0
    assert sc["EEGChannelCount"] == 3
    assert sc["RecordingDuration"] == 2.0


def test_set_sidecar_broadcast_and_remove(recording):
    recording.set_sidecar(None, TaskName="go/no-go")     # broadcast to all streams
    assert recording.sidecars["eeg"]["TaskName"] == "go/no-go"
    assert recording.sidecars["emg"]["TaskName"] == "go/no-go"
    recording.remove_data("emg")
    assert "emg" not in recording.data and "emg" not in recording.sidecars


def test_signals_and_get_by_type(recording):
    assert set(recording.signals) == {"eeg", "emg"}
    assert set(recording.get_signals_by_type("EMG")) == {"emg"}
    assert set(recording.get_signals_by_type("EEG")) == {"eeg"}


def test_data_keys_stay_in_sync(recording):
    assert set(recording.data) == set(recording.sidecars)


# ------------------------------------------------------------------- round-trip
@pytest.mark.parametrize("fmt", ["bson", "json", "mat", "h5"])
def test_roundtrip_all_formats(recording, tmp_path, roundtrip, fmt):
    back = roundtrip(recording, tmp_path, fmt)
    np.testing.assert_allclose(back.signals["eeg"].signal,
                               recording.signals["eeg"].signal)
    assert back.signals["emg"].signal.shape == (200, 2)
    assert back.events.df["trial_type"].tolist() == ["go", "nogo", "target"]
    assert back.experiment == recording.experiment           # incl. nested dict
    assert back.sidecars["eeg"]["Manufacturer"] == "Brain Products"
    assert back.bids.subject == "01" and back.bids.run == 1


def test_experiment_component_roundtrips(recording, tmp_path, roundtrip):
    # A SerializableComponent experiment is rebuilt via the dispatch tag.
    bids = recording.bids
    other = Recording(bids)
    other.add_signal("eeg", recording.signals["eeg"])
    other.set_experiment(recording.signals["eeg"].channel_set)   # any component
    back = roundtrip(other, tmp_path, "json")
    assert isinstance(back.experiment, ChannelSet)


# ----------------------------------------------------------------- schema (D8)
def test_serialized_dict_stamps_version(recording):
    assert recording.to_serializable_obj()["schema_version"] == SCHEMA_VERSION


@pytest.mark.parametrize("bad", [1, SCHEMA_VERSION + 1, None])
def test_unsupported_or_missing_version_raises(recording, bad):
    d = recording.to_serializable_obj()
    if bad is None:
        d.pop("schema_version")
    else:
        d["schema_version"] = bad
    with pytest.raises(ValueError):
        Recording.from_serializable_obj(d)


def test_missing_required_key_raises(recording):
    d = recording.to_serializable_obj()
    d.pop("bids")
    with pytest.raises(ValueError):
        Recording.from_serializable_obj(d)
