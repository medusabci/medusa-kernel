"""Channel / Sensor / ChannelSet (D1)."""
import numpy as np
import pytest

from medusa.core.data import Channel, ChannelSet, Sensor


def test_build_sensors_first_and_query():
    cs = (ChannelSet()
          .add_sensors([Sensor("Fz", coordinates=[0.0, 0.7, 0.7]),
                        Sensor("Cz", coordinates=[0.0, 0.0, 1.0])])
          .add_channels([Channel("Fz", "EEG", "uV", sensor="Fz"),
                         Channel("Cz", "EEG", "uV", sensor="Cz")]))
    assert cs.n_channels == 2
    assert list(cs.uids) == ["Fz", "Cz"]
    assert list(cs.types) == ["EEG", "EEG"]
    np.testing.assert_array_equal(cs.index(["Cz", "Fz"]), [1, 0])


def test_unknown_ch_type_coerced_to_other_with_warning():
    with pytest.warns(UserWarning):
        ch = Channel("X", "PHOTODIODE", "V")
    assert ch.ch_type == "OTHER"


def test_duplicate_uid_rejected():
    cs = ChannelSet().add_channels([Channel("A", "EEG", "uV")])
    with pytest.raises((ValueError, KeyError)):
        cs.add_channels([Channel("A", "EEG", "uV")])


def test_channel_links_to_unregistered_sensor_rejected():
    with pytest.raises((ValueError, KeyError)):
        ChannelSet().add_channels([Channel("A", "EEG", "uV", sensor="missing")])


def test_pick_and_subset_preserve_sensors():
    cs = ChannelSet().add_channels([Channel("A", "EEG", "uV"),
                                    Channel("B", "EMG", "uV"),
                                    Channel("C", "EEG", "uV")])
    subset, idx = cs.pick(types="EEG")
    assert list(subset.uids) == ["A", "C"]
    np.testing.assert_array_equal(idx, [0, 2])


@pytest.mark.parametrize("fmt", ["bson", "json", "mat"])
def test_roundtrip_multichannel(eeg_cs, tmp_path, roundtrip, fmt):
    back = roundtrip(eeg_cs, tmp_path, fmt)
    assert list(back.uids) == list(eeg_cs.uids)
    assert list(back.types) == list(eeg_cs.types)


@pytest.mark.parametrize("fmt", ["bson", "json", "mat"])
def test_roundtrip_single_channel(tmp_path, roundtrip, fmt):
    # A 1-channel/1-sensor set used to break .mat (squeeze collapse) -- now fixed.
    cs = ChannelSet().add_unipolar_eeg_channels(["Fz"])
    back = roundtrip(cs, tmp_path, fmt)
    assert list(back.uids) == ["Fz"]
    assert back.n_channels == 1
