"""Signal (D2)."""
import numpy as np
import pytest

from medusa.core.data import Channel, ChannelSet, Signal


def test_shape_properties_and_times_synthesis(eeg_cs):
    sig = Signal(np.zeros((500, 3)), fs=250.0, channel_set=eeg_cs)
    assert (sig.n_samples, sig.n_channels) == (500, 3)
    # times synthesized as arange(n)/fs when omitted
    np.testing.assert_allclose(sig.times, np.arange(500) / 250.0)


def test_validation_rejects_bad_shapes(eeg_cs):
    with pytest.raises(ValueError):
        Signal(np.zeros(500), fs=250.0, channel_set=eeg_cs)            # not 2-D
    with pytest.raises(ValueError):
        Signal(np.zeros((500, 2)), fs=250.0, channel_set=eeg_cs)       # cols != n_channels
    with pytest.raises(ValueError):
        Signal(np.zeros((500, 3)), fs=250.0, channel_set=eeg_cs,
               times=np.arange(499))                                   # times mismatch


def test_empty_declares_structure(eeg_cs):
    sig = Signal.empty(fs=256.0, channel_set=eeg_cs)
    assert sig.n_samples == 0 and sig.n_channels == 3


def test_pick_types_returns_sub_signal(rng):
    cs = ChannelSet().add_unipolar_eeg_channels(["Fz", "Cz"])
    cs.add_channels([Channel("EMG", "EMG", "uV")])
    sig = Signal(rng.standard_normal((100, 3)), fs=256.0, channel_set=cs)
    eeg = sig.pick_types("EEG")
    assert eeg.n_channels == 2 and eeg.n_samples == 100


@pytest.mark.parametrize("types,expected", [
    (["EEG"], "eeg"), (["EMG"], "emg"), (["ACCEL", "GYRO"], "motion"),
    (["TRIG"], None)])
def test_bids_datatype(types, expected):
    from medusa.core.data import Channel
    cs = ChannelSet().add_channels(
        [Channel(f"c{i}", t, "uV") for i, t in enumerate(types)])
    sig = Signal(np.zeros((10, len(types))), fs=100.0, channel_set=cs)
    assert sig.bids_datatype() == expected


@pytest.mark.parametrize("fmt", ["bson", "json", "mat"])
def test_roundtrip(eeg_cs, rng, tmp_path, roundtrip, fmt):
    sig = Signal(rng.standard_normal((120, 3)), fs=256.0, channel_set=eeg_cs)
    back = roundtrip(sig, tmp_path, fmt)
    assert back.signal.shape == (120, 3)
    np.testing.assert_allclose(back.signal, sig.signal)
    assert back.fs == 256.0


@pytest.mark.parametrize("fmt,shape", [
    ("mat", (1, 3)), ("mat", (50, 1)), ("bson", (1, 3)), ("json", (50, 1))])
def test_roundtrip_single_sample_or_channel(rng, tmp_path, roundtrip, fmt, shape):
    # 1-sample / 1-channel matrices collapse a size-1 axis through .mat -- the
    # reshape in from_serializable_obj restores 2-D.
    n_ch = shape[1]
    cs = ChannelSet().add_unipolar_eeg_channels(["Fz", "Cz", "Pz"][:n_ch])
    sig = Signal(rng.standard_normal(shape), fs=128.0, channel_set=cs)
    back = roundtrip(sig, tmp_path, fmt)
    assert back.signal.shape == shape
    np.testing.assert_allclose(back.signal, sig.signal)
