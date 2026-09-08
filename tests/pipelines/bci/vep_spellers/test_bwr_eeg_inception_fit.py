"""End-to-end ``fit`` / ``predict`` for the deep BWR pipeline, on every architecture.

The unit tests elsewhere build a backbone straight from a settings group; this drives the
real path instead -- ``fit`` reads ``classifier.arch``, hands that architecture's own group
to :func:`~medusa.pipelines.bci._torch_backbones.build_backbone` along with the epoch shape
and the epoch rate, and trains. It is a **smoke** test: two training epochs on a small
synthetic recording says nothing about accuracy, only that the whole chain runs and returns
the cumulative score matrix the Layer-2 decoder expects.

Skipped on the no-extras CI job: this pipeline is torch-gated.
"""
import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("lightning")

from medusa.pipelines.bci._torch_backbones import ARCHITECTURES
from medusa.pipelines.bci.vep_spellers import (
    generate_random_codebook, BWREEGInceptionPipeline, bwr_eeg_inception_settings)

# Small on purpose: 4 codes x 2 cycles x 84 frames keeps the epoch count in the hundreds.
FPS, FS, N_CMDS, N_FRAMES, N_CYCLES = 60.0, 256.0, 4, 84, 2

#: 250 ms at the native 256 Hz -> 64 samples, and 100/75/50 ms -> 26/19/13 samples, so both
#: architectures are comfortably above EEG-Inception v1's degenerate-dimension floors.
SETTINGS = dict(band=(1.0, 40.0), order=5, w_segment_t=(0.0, 250.0), target_fs=None,
                scales_ms=(100.0, 75.0, 50.0))
TRAINING = {"max_epochs": 2, "batch_size": 256, "val_split": 0.2, "patience": 2,
            "verbose": "silent"}


@pytest.fixture
def cvep_train_test(cvep_recording_factory, cvep_channels):
    """A (train, test, channels) triple of synthetic random-code c-VEP recordings."""
    cmds = generate_random_codebook(N_CMDS, n_frames=N_FRAMES, seed=0)
    uids = list(cmds)
    train = cvep_recording_factory(cmds, uids, fps=FPS, fs=FS, n_cycles=N_CYCLES,
                                   resp_amp=3.0, seed=1, mode="train")
    test = cvep_recording_factory(cmds, uids, fps=FPS, fs=FS, n_cycles=N_CYCLES,
                                  resp_amp=3.0, seed=2, mode="test")
    return train, test, cvep_channels


def _pipeline(channels, arch):
    return BWREEGInceptionPipeline(
        settings=bwr_eeg_inception_settings(arch=arch, **SETTINGS),
        channels=channels, classifier={"training": dict(TRAINING)})


def _n_cycles(recording):
    df = recording.events.df
    return len(df[df["cycle_idx"].notna()])


@pytest.mark.parametrize("arch", list(ARCHITECTURES))
def test_fit_and_predict_through_every_architecture(cvep_train_test, arch):
    train, test, channels = cvep_train_test
    pipe = _pipeline(channels, arch).fit([train])

    scores = pipe.predict(test)
    assert scores.shape == (_n_cycles(test), len(test.experiment.command_uids))
    assert np.isfinite(scores).all()


@pytest.mark.parametrize("arch", list(ARCHITECTURES))
def test_a_fitted_pipeline_round_trips(cvep_train_test, arch, tmp_path):
    """The per-architecture group and its group-list must survive save/load untouched."""
    train, test, channels = cvep_train_test
    pipe = _pipeline(channels, arch).fit([train])
    expected = pipe.predict(test)

    path = tmp_path / "pipe.pkl"
    pipe.save(str(path))
    reloaded = BWREEGInceptionPipeline.load(str(path))

    assert reloaded.cfg["classifier"] == pipe.cfg["classifier"]
    np.testing.assert_allclose(reloaded.predict(test), expected, rtol=1e-5, atol=1e-6)


def _fitted_scores(channels, train, test, training):
    """Fit a fresh pipeline with these training settings and score the test recording."""
    pipe = BWREEGInceptionPipeline(
        settings=bwr_eeg_inception_settings(arch="eeg_inception_v2", **SETTINGS),
        channels=channels, classifier={"training": training})
    return pipe.fit([train]).predict(test)


def test_a_seeded_fit_is_reproducible(cvep_train_test):
    """``random_state`` has to cover the backbone's initial weights as well.

    The pipeline builds the backbone itself, before the estimator exists, so a seed that
    only reached the :class:`~medusa.ml.torch_models.classification.TorchClassifier` would
    still leave every run different. ``device='cpu'`` because CUDA kernels are not
    deterministic by default.
    """
    train, test, channels = cvep_train_test
    training = dict(TRAINING, random_state=0, device="cpu")
    np.testing.assert_allclose(_fitted_scores(channels, train, test, training),
                               _fitted_scores(channels, train, test, training),
                               rtol=1e-5, atol=1e-6)


def test_without_a_seed_two_fits_differ(cvep_train_test):
    """The default, and the guard that keeps the test above from passing on a
    seed-independent output."""
    train, test, channels = cvep_train_test
    training = dict(TRAINING, device="cpu")
    assert not np.allclose(_fitted_scores(channels, train, test, training),
                           _fitted_scores(channels, train, test, training),
                           rtol=1e-3, atol=1e-3)
