"""``segmentation.norm``: the baseline normalization the BWR pipelines apply.

With ``baseline_t`` on, every frame epoch is normalized per channel against its baseline
window: ``'dc'`` subtracts the baseline mean, ``'z'`` also divides by the baseline standard
deviation. With ``baseline_t`` off, ``norm`` is inert.
"""
import numpy as np
import pytest

from medusa.pipelines.bci.vep_spellers import generate_random_codebook, BWRLDAPipeline
from medusa.pipelines.bci.vep_spellers.data import SpellerData, cycle_arrays

FPS, FS, N_CMDS, N_FRAMES, N_CYCLES = 60.0, 256.0, 4, 84, 2
WINDOW = [0.0, 100.0]


@pytest.fixture
def recording(cvep_recording_factory):
    cmds = generate_random_codebook(N_CMDS, n_frames=N_FRAMES, seed=0)
    return cvep_recording_factory(cmds, list(cmds), fps=FPS, fs=FS, n_cycles=N_CYCLES,
                                  resp_amp=3.0, seed=1, mode="train")


def _lda(channels, *, baseline_t, norm):
    return BWRLDAPipeline(channels=list(channels), segmentation={
        "w_segment_t": WINDOW, "baseline_t": baseline_t, "norm": norm, "target_fs": None})


def _epochs(pipe, rec):
    """The per-frame epochs, ``(n_segments, n_samples, n_channels)`` (single band)."""
    cfg = pipe.cfg
    onsets, _, _, _ = cycle_arrays(rec.events)
    sd = SpellerData.from_recording(rec)
    feats = pipe._features(rec.signals[cfg["signal_key"]], onsets, sd.codes.shape[2],
                           sd.fps_resolution, cfg)
    return feats.reshape(len(feats), -1, len(cfg["channels"]))


def test_default_z_scores_each_segment_with_its_own_statistics():
    seg = BWRLDAPipeline.default_settings().to_dict()["segmentation"]
    assert seg["norm"] == "z"
    assert seg["baseline_t"] == seg["w_segment_t"]


@pytest.mark.parametrize("norm, expected_std", [("z", True), ("dc", False)])
def test_norm_uses_baseline_statistics_per_channel(recording, cvep_channels,
                                                   norm, expected_std):
    """With the baseline equal to the segment window, each epoch channel ends up with the
    statistics the mode sets: zero mean for both, unit standard deviation for ``'z'`` only."""
    epochs = _epochs(_lda(cvep_channels, baseline_t=WINDOW, norm=norm), recording)
    np.testing.assert_allclose(epochs.mean(axis=1), 0.0, atol=1e-9)
    unit_std = np.allclose(epochs.std(axis=1), 1.0)
    assert unit_std is expected_std


def test_norm_is_inert_without_baseline(recording, cvep_channels):
    dc = _epochs(_lda(cvep_channels, baseline_t=[], norm="dc"), recording)
    z = _epochs(_lda(cvep_channels, baseline_t=[], norm="z"), recording)
    np.testing.assert_array_equal(dc, z)


def test_unknown_norm_is_rejected_at_fit(recording, cvep_channels):
    pipe = _lda(cvep_channels, baseline_t=[-100.0, 0.0], norm="dc")
    with pytest.warns(UserWarning, match="value_options"):
        pipe.settings["segmentation"]["norm"].edit_item(value="minmax")
    with pytest.raises(ValueError, match="segmentation.norm"):
        pipe.fit([recording])


def test_eeg_inception_defaults_z_score_each_segment_with_its_own_statistics():
    pytest.importorskip("torch")
    pytest.importorskip("lightning")
    from medusa.pipelines.bci.vep_spellers.decoding import (
        bwr_eeg_inception_settings, mseq_cvep_settings, burst_cvep_settings)
    for s in (bwr_eeg_inception_settings(), mseq_cvep_settings(),
              burst_cvep_settings(w_segment_t=(0.0, 300.0))):
        seg = s.to_dict()["segmentation"]
        assert seg["norm"] == "z"
        assert seg["baseline_t"] == seg["w_segment_t"]
    # switched off, it keeps the segment window as the value it takes when switched on
    off = bwr_eeg_inception_settings(w_segment_t=(0.0, 300.0), baseline_t=None)
    assert off.to_dict()["segmentation"]["baseline_t"] is None
    off.set_enabled("segmentation", "baseline_t", enabled=True)
    assert off.to_dict()["segmentation"]["baseline_t"] == [0.0, 300.0]
