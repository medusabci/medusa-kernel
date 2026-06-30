"""Recorder: buffering + HDF5 persistence (D5)."""
import gc

import numpy as np
import pytest

from medusa.core.data import (BidsInfo, ChannelSet, Events, Recorder,
                              Recording, RecordingData, Signal)


def _empty_rec(eeg_cs, fs=256.0):
    rec = Recording(BidsInfo("01", task="rest"))
    rec.add_signal("eeg", Signal.empty(fs=fs, channel_set=eeg_cs))
    return rec


# ----------------------------------------------------- buffer = the data
def test_buffer_is_the_recording_signal(eeg_cs, rng):
    rec = _empty_rec(eeg_cs)
    chunks = []
    with Recorder(rec, buffer=1.0) as r:                 # capacity 256, no file
        for _ in range(5):
            c = rng.standard_normal((100, 3)); chunks.append(c)
            r.append_data("eeg", c)
    full = np.vstack(chunks)                              # 500 total
    assert rec.signals["eeg"].n_samples == 256           # bounded to last 1 s
    np.testing.assert_allclose(rec.signals["eeg"].signal, full[-256:])
    assert rec.signals["eeg"].pick_types("EEG").n_channels == 3


def test_buffer_keeps_true_appended_times(eeg_cs):
    rec = _empty_rec(eeg_cs)
    t0 = 1000.0                                           # device clock offset
    with Recorder(rec, buffer=10.0) as r:
        for i in range(3):
            times = t0 + (np.arange(50) + i * 50) / 256.0
            r.append_data("eeg", np.zeros((50, 3)), times=times)
    assert rec.signals["eeg"].times[0] >= 1000.0         # real times, not arange/fs


def test_append_events_needs_a_file(eeg_cs):
    rec = _empty_rec(eeg_cs)
    rec.set_events(Events(optional_columns={"t": str}))
    with Recorder(rec, buffer=1.0) as r:                 # no file
        with pytest.raises(ValueError):
            r.append_events(onset=1.0, duration=0.1, t="x")


# ----------------------------------------------------- persistence + read
def test_record_and_read_back(eeg_cs, rng, tmp_path):
    rec = _empty_rec(eeg_cs)
    rec.set_events(Events(optional_columns={"trial_type": str}))
    path = str(tmp_path / "rec.h5")
    eeg = []
    with Recorder(rec, file=path, buffer=0.5) as r:      # cap 128; file is full
        for i in range(3):
            c = rng.standard_normal((100, 3)); eeg.append(c)
            r.append_data("eeg", c)
            r.append_events(onset=float(i), duration=0.1, trial_type="go")
    assert rec.signals["eeg"].n_samples == 128           # in-RAM buffer bounded
    loaded = Recording.load(path)
    assert loaded.signals["eeg"].signal.shape == (300, 3)  # file holds everything
    np.testing.assert_allclose(loaded.signals["eeg"].signal, np.vstack(eeg))
    assert loaded.events.n_events == 3


def test_persist_only_leaves_signals_empty(eeg_cs, rng, tmp_path):
    rec = _empty_rec(eeg_cs)
    path = str(tmp_path / "log.h5")
    with Recorder(rec, file=path, buffer=None) as r:
        r.append_data("eeg", rng.standard_normal((200, 3)))
    assert rec.signals["eeg"].n_samples == 0             # not buffered
    assert Recording.load(path).signals["eeg"].signal.shape == (200, 3)


def test_one_shot_save_equals_stream(eeg_cs, rng, tmp_path):
    rec = Recording(BidsInfo("02", task="mi"))
    rec.add_signal("eeg", Signal(rng.standard_normal((500, 3)), fs=128.0,
                                 channel_set=eeg_cs))
    path = str(tmp_path / "full.h5")
    rec.save(path)
    back = Recording.load(path)
    np.testing.assert_allclose(back.signals["eeg"].signal, rec.signals["eeg"].signal)
    assert rec.signals["eeg"].n_samples == 500           # save did not trim


# ----------------------------------------------------- robustness / guards
def test_crash_without_close_is_readable(eeg_cs, rng, tmp_path):
    rec = _empty_rec(eeg_cs)
    path = str(tmp_path / "crash.h5")
    r = Recorder(rec, file=path, buffer=2.0)
    r.append_data("eeg", rng.standard_normal((50, 3)))
    r.append_data("eeg", rng.standard_normal((50, 3)))
    del r                                                # no close()
    gc.collect()
    assert Recording.load(path).signals["eeg"].signal.shape == (100, 3)


def test_overwrite_guard(eeg_cs, tmp_path):
    rec = _empty_rec(eeg_cs)
    path = str(tmp_path / "x.h5")
    Recorder(rec, file=path, buffer=None).close()
    with pytest.raises((FileExistsError, OSError)):
        Recorder(rec, file=path, buffer=None)            # exists, overwrite=False


def test_option_guards(eeg_cs, tmp_path):
    rec = _empty_rec(eeg_cs)
    with pytest.raises(ValueError):
        Recorder(rec, buffer=None)                       # neither buffer nor file


def test_non_signal_data_rejected(tmp_path):
    class FakeVolume(RecordingData):
        def to_serializable_obj(self): return {}
        @classmethod
        def from_serializable_obj(cls, d): return cls()
    rec = Recording(BidsInfo("03"))
    rec.add_data("t1w", FakeVolume())
    with pytest.raises(NotImplementedError):
        Recorder(rec, buffer=1.0)
