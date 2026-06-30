"""Shared fixtures for the ``core/data`` test suite (D9)."""
import warnings

import numpy as np
import pytest

from medusa.core.data import (BidsInfo, Channel, ChannelSet, Events, Recording,
                              Signal)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def eeg_cs():
    """A 3-channel unipolar EEG set (positioned 10-10 sensors)."""
    return ChannelSet().add_unipolar_eeg_channels(["Fz", "Cz", "Pz"])


@pytest.fixture
def emg_cs():
    """A 2-channel EMG set (no sensor positions)."""
    return ChannelSet().add_channels([Channel("EMG_left", "EMG", "uV"),
                                      Channel("EMG_right", "EMG", "uV")])


@pytest.fixture
def recording(rng, eeg_cs, emg_cs):
    """A small multi-stream recording with events, sidecar and experiment."""
    rec = Recording(BidsInfo("01", task="gonogo", run=1,
                             participant={"age": 30, "sex": "F"}))
    rec.add_signal("eeg", Signal(rng.standard_normal((300, 3)), fs=256.0,
                                 channel_set=eeg_cs))
    rec.add_signal("emg", Signal(rng.standard_normal((200, 2)), fs=1000.0,
                                 channel_set=emg_cs))
    events = Events(optional_columns={"trial_type": str, "value": int},
                    descriptions={"trial_type": "go or nogo"})
    events.append([{"onset": 0.5, "duration": 0.2, "trial_type": "go", "value": 1},
                   {"onset": 1.0, "duration": 0.2, "trial_type": "nogo", "value": 2},
                   {"onset": 1.5, "duration": 0.2, "trial_type": "target", "value": 3}])
    rec.set_events(events)
    rec.set_sidecar("eeg", Manufacturer="Brain Products", PowerLineFrequency=50)
    rec.set_experiment({"paradigm": "go/no-go",
                        "stimulus": {"go": "green circle", "nogo": "red X"}})
    return rec


@pytest.fixture
def roundtrip():
    """Save a component to ``fmt`` and load it back (warnings silenced)."""
    def _roundtrip(obj, tmp_path, fmt):
        path = str(tmp_path / f"obj.{fmt}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            obj.save(path)
            return type(obj).load(path)
    return _roundtrip
