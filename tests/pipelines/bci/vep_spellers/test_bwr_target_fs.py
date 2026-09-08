"""``segmentation.target_fs`` and the rules it puts on ``fs`` and the filter bank.

Both BWR pipelines resample every epoch to ``segmentation.target_fs`` when it is set. Two
consequences are checked here, for :class:`BWRLDAPipeline` (torch-free) and for
:class:`BWREEGInceptionPipeline` (torch-gated, so those tests import it lazily):

* With ``target_fs`` set the native rate stops mattering -- filtering and segmentation use
  each recording's own ``fs`` and the epochs all come out at ``target_fs`` -- so a corpus
  may mix rates. With it unset the old equal-``fs`` rule still holds.
* Resampling to ``target_fs`` keeps only what is below ``target_fs / 2``, so the filter
  bank has to bound the signal under that already, and ``target_fs`` may not exceed the
  recording's rate.
"""
import numpy as np
import pytest

from medusa.pipelines.bci.vep_spellers import (
    generate_random_codebook, BWRLDAPipeline)

FPS, N_CMDS, N_FRAMES, N_CYCLES = 60.0, 4, 84, 2
#: The two native rates a mixed-rate corpus is built from.
FS_A, FS_B = 256.0, 600.0
TRAINING = {"max_epochs": 1, "batch_size": 256, "val_split": 0.2, "patience": 2,
            "verbose": "silent"}


@pytest.fixture
def recording_factory(cvep_recording_factory):
    """Build one synthetic c-VEP recording at the requested sampling rate."""
    cmds = generate_random_codebook(N_CMDS, n_frames=N_FRAMES, seed=0)
    uids = list(cmds)

    def make(fs, seed=1):
        return cvep_recording_factory(cmds, uids, fps=FPS, fs=fs, n_cycles=N_CYCLES,
                                      resp_amp=3.0, seed=seed, mode="train")
    return make


def _lda(channels, *, target_fs=None, cutoff=(1.0, 60.0), band_type="bandpass"):
    """A BWR-LDA pipeline with the filter bank and resampling rate set on its tree."""
    pipe = BWRLDAPipeline(channels=list(channels))
    if target_fs is not None:
        pipe.settings["segmentation"]["target_fs"].edit_item(value=target_fs)
    band = pipe.settings["freq_filtering"]["filterbank"].elements[0]
    band["cutoff"].edit_item(value=list(cutoff))
    band["band_type"].edit_item(value=band_type)
    return pipe


# --------------------------------------------------------------------------- #
# fs is only pinned when the epochs keep the native rate
# --------------------------------------------------------------------------- #
def test_mixed_native_rates_accepted_when_target_fs_is_set(recording_factory,
                                                           cvep_channels):
    pipe = _lda(cvep_channels, target_fs=128.0)
    pipe.check_consistency(recording_factory(FS_A))
    pipe.check_consistency(recording_factory(FS_B, seed=2))     # must not raise
    assert pipe.fs == FS_A, "the first rate seen is still recorded"


def test_mixed_native_rates_rejected_without_target_fs(recording_factory,
                                                       cvep_channels):
    pipe = _lda(cvep_channels)          # target_fs off -> epochs keep the native rate
    pipe.check_consistency(recording_factory(FS_A))
    with pytest.raises(ValueError, match="fs mismatch"):
        pipe.check_consistency(recording_factory(FS_B, seed=2))


# --------------------------------------------------------------------------- #
# target_fs has to be reachable: filter bank below the new Nyquist, and no upsampling
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("upper", [64.0, 70.0])
def test_cutoff_at_or_above_the_new_nyquist_is_rejected(recording_factory,
                                                        cvep_channels, upper):
    """target_fs=128 Hz keeps only what is under 64 Hz, so a band reaching it is cut."""
    pipe = _lda(cvep_channels, target_fs=128.0, cutoff=(1.0, upper))
    with pytest.raises(ValueError, match="Nyquist") as excinfo:
        pipe.check_consistency(recording_factory(FS_A))
    message = str(excinfo.value)
    assert str(upper) in message and "64.0" in message


@pytest.mark.parametrize("band_type", ["highpass", "bandstop"])
def test_a_filter_unbounded_above_is_rejected(recording_factory, cvep_channels,
                                              band_type):
    pipe = _lda(cvep_channels, target_fs=128.0, band_type=band_type)
    with pytest.raises(ValueError, match="bandpass or lowpass"):
        pipe.check_consistency(recording_factory(FS_A))


def test_a_cutoff_below_the_new_nyquist_is_accepted(recording_factory, cvep_channels):
    """The guard above must not fire on a bank that already fits."""
    pipe = _lda(cvep_channels, target_fs=128.0, cutoff=(1.0, 40.0))
    pipe.check_consistency(recording_factory(FS_A))              # must not raise


def test_target_fs_above_the_recording_rate_is_rejected(recording_factory,
                                                        cvep_channels):
    pipe = _lda(cvep_channels, target_fs=512.0, cutoff=(1.0, 40.0))
    with pytest.raises(ValueError, match="above the recording's fs"):
        pipe.check_consistency(recording_factory(FS_A))


def test_no_resampling_rules_apply_when_target_fs_is_off(recording_factory,
                                                         cvep_channels):
    """With the epochs at the native rate there is no new Nyquist to respect."""
    pipe = _lda(cvep_channels, cutoff=(1.0, 60.0), band_type="highpass")
    pipe.check_consistency(recording_factory(FS_A))              # must not raise


# --------------------------------------------------------------------------- #
# The deep pipeline: the same rules, driven through a real fit
# --------------------------------------------------------------------------- #
@pytest.fixture
def deep_pipeline_factory(cvep_channels):
    """Build a :class:`BWREEGInceptionPipeline`; skips when torch is not installed."""
    pytest.importorskip("torch")
    pytest.importorskip("lightning")
    from medusa.pipelines.bci.vep_spellers import (
        BWREEGInceptionPipeline, bwr_eeg_inception_settings)

    def make(*, target_fs, band=(1.0, 40.0)):
        settings = bwr_eeg_inception_settings(
            arch="eeg_inception_v2", band=band, order=5, w_segment_t=(0.0, 250.0),
            baseline_t=None, target_fs=target_fs, scales_ms=(100.0, 75.0, 50.0))
        return BWREEGInceptionPipeline(settings=settings, channels=list(cvep_channels),
                                       classifier={"training": dict(TRAINING)})
    return make


def test_deep_pipeline_fits_a_mixed_rate_corpus(deep_pipeline_factory,
                                                recording_factory):
    """256 Hz and 600 Hz recordings train one model once ``target_fs`` unifies them."""
    pipe = deep_pipeline_factory(target_fs=128.0)
    recordings = [recording_factory(FS_A), recording_factory(FS_B, seed=2)]

    pipe.fit(recordings)

    scores = pipe.predict(recordings[1])
    assert scores.shape[1] == N_CMDS
    assert np.isfinite(scores).all()


def test_deep_pipeline_still_rejects_a_mixed_rate_corpus_without_target_fs(
        deep_pipeline_factory, recording_factory):
    pipe = deep_pipeline_factory(target_fs=None)
    with pytest.raises(ValueError, match="fs mismatch"):
        pipe.fit([recording_factory(FS_A), recording_factory(FS_B, seed=2)])


def test_deep_pipeline_rejects_a_band_reaching_the_new_nyquist(deep_pipeline_factory,
                                                               recording_factory):
    pipe = deep_pipeline_factory(target_fs=128.0, band=(1.0, 64.0))
    with pytest.raises(ValueError, match="Nyquist"):
        pipe.fit([recording_factory(FS_A)])


def test_deep_pipeline_rejects_target_fs_above_the_recording_rate(deep_pipeline_factory,
                                                                  recording_factory):
    pipe = deep_pipeline_factory(target_fs=512.0)
    with pytest.raises(ValueError, match="above the recording's fs"):
        pipe.fit([recording_factory(FS_A)])
