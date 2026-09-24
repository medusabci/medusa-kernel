"""``segmentation.norm``: the baseline normalization the motor pipelines apply per trial."""
import numpy as np
import pytest

from medusa.pipelines.bci.motor_decoding import MICSPLDAPipeline
from medusa.pipelines.bci.motor_decoding.decoding._common import trial_segments

WINDOW = (0.0, 1000.0)


def _segments(rec, channels, *, baseline, norm):
    onsets = rec.events.df["onset"].to_numpy()
    return trial_segments(rec.signals["eeg"], onsets, channels=channels, apply_car=False,
                          filter_spec=MICSPLDAPipeline.default_settings().to_dict()["filter"],
                          window=WINDOW, baseline=baseline, target_fs=None, norm=norm)


def test_default_norm_is_dc():
    assert MICSPLDAPipeline.default_settings().to_dict()["segmentation"]["norm"] == "dc"


def test_z_norm_scales_each_channel_by_its_baseline(mi_recording, mi_channels):
    """With the baseline equal to the window, ``'z'`` gives every trial channel zero mean
    and unit standard deviation; ``'dc'`` only removes the mean."""
    z = _segments(mi_recording, mi_channels, baseline=WINDOW, norm="z")
    np.testing.assert_allclose(z.mean(axis=1), 0.0, atol=1e-9)
    np.testing.assert_allclose(z.std(axis=1), 1.0)
    dc = _segments(mi_recording, mi_channels, baseline=WINDOW, norm="dc")
    np.testing.assert_allclose(dc.mean(axis=1), 0.0, atol=1e-9)
    assert not np.allclose(dc.std(axis=1), 1.0)


def test_csp_lda_fits_with_z_norm(mi_recording, mi_channels):
    # n_filters <= n_channels: the fixture has three
    pipe = MICSPLDAPipeline(channels=mi_channels, csp={"n_filters": 2},
                            segmentation={"norm": "z"})
    assert pipe.fit([mi_recording]).predict(mi_recording).shape[0] == 16


def test_unknown_norm_is_rejected_at_fit(mi_recording, mi_channels):
    pipe = MICSPLDAPipeline(channels=mi_channels)
    with pytest.warns(UserWarning, match="value_options"):
        pipe.settings["segmentation"]["norm"].edit_item(value="minmax")
    with pytest.raises(ValueError, match="segmentation.norm"):
        pipe.fit([mi_recording])
