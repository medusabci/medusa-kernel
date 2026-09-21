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

from medusa.ml.torch_models.classification import TorchClassifier
from medusa.pipelines.bci._torch_backbones import ARCHITECTURES
from medusa.pipelines.bci.vep_spellers import (
    generate_random_codebook, BWREEGInceptionPipeline, bwr_eeg_inception_settings)
from medusa.pipelines.bci.vep_spellers.decoding.bwr_eeg_inception import (
    _validation_groups)

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


# --------------------------------------------------------------------------- #
# classifier.training.val_split_unit: what the validation split holds out
# --------------------------------------------------------------------------- #
class TestValidationGroups:
    """``_validation_groups`` maps the setting to one group id per frame epoch."""

    TRIAL = np.array([0, 0, 1, 1, 2, 2])            # 3 trials x 2 cycles

    def test_trial_gives_every_cycle_its_trial(self):
        ids = _validation_groups("trial", self.TRIAL, n_frames=4)
        np.testing.assert_array_equal(ids, np.repeat([0, 0, 1, 1, 2, 2], 4))

    def test_cycle_gives_every_cycle_its_own_id(self):
        ids = _validation_groups("cycle", self.TRIAL, n_frames=4)
        np.testing.assert_array_equal(ids, np.repeat(np.arange(6), 4))

    def test_frame_means_no_groups(self):
        assert _validation_groups("frame", self.TRIAL, n_frames=4) is None

    def test_unknown_unit_is_rejected(self):
        with pytest.raises(ValueError, match="val_split_unit"):
            _validation_groups("recording", self.TRIAL, n_frames=4)


def test_the_default_unit_is_trial_and_sits_next_to_val_split():
    settings = bwr_eeg_inception_settings()
    item = settings.get_item("classifier", "training", "val_split_unit")
    assert item.tree["value"] == "trial"
    assert item.tree["value_options"] == ["frame", "cycle", "trial"]
    keys = list(settings.to_dict()["classifier"]["training"])
    assert keys.index("val_split_unit") == keys.index("val_split") + 1
    assert keys[keys.index("val_split_unit") + 1] == "patience"    # the stopping block


@pytest.mark.parametrize("unit, n_groups, group_size", [
    ("trial", 2 * N_CMDS, N_CYCLES * N_FRAMES),      # 2 recordings x N_CMDS trials
    ("cycle", 2 * N_CMDS * N_CYCLES, N_FRAMES),
])
def test_fit_hands_the_classifier_one_group_per_unit(cvep_train_test, monkeypatch,
                                                     unit, n_groups, group_size):
    """Two recordings: trials must stay apart across them, and every group is whole."""
    train, test, channels = cvep_train_test
    seen = {}
    monkeypatch.setattr(TorchClassifier, "fit",
                        lambda self, X, y, groups=None: seen.update(groups=groups) or self)
    pipe = BWREEGInceptionPipeline(
        settings=bwr_eeg_inception_settings(arch="eeg_inception_v2", **SETTINGS),
        channels=channels, classifier={"training": dict(TRAINING, val_split_unit=unit)})
    pipe.fit([train, test])

    groups = seen["groups"]
    _, counts = np.unique(groups, return_counts=True)
    assert len(groups) == 2 * N_CMDS * N_CYCLES * N_FRAMES
    assert len(counts) == n_groups
    assert (counts == group_size).all()


def test_frame_unit_passes_no_groups(cvep_train_test, monkeypatch):
    train, _, channels = cvep_train_test
    seen = {}
    monkeypatch.setattr(TorchClassifier, "fit",
                        lambda self, X, y, groups=None: seen.update(groups=groups) or self)
    pipe = BWREEGInceptionPipeline(
        settings=bwr_eeg_inception_settings(arch="eeg_inception_v2", **SETTINGS),
        channels=channels, classifier={"training": dict(TRAINING, val_split_unit="frame")})
    pipe.fit([train])
    assert seen["groups"] is None
