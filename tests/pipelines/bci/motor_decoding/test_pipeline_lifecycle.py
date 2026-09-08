"""The stateful pipeline lifecycle: fit continues, restart forgets, and what guards it.

Driven through :class:`MIEEGInceptionPipeline` because it is the cheapest torch pipeline to
fit (16 synthetic trials, two epochs), but everything tested here lives in
:class:`~medusa.pipelines.torch_base.TorchPipeline` and
:class:`~medusa.pipelines.base.DecodingPipeline`, so it holds for every deep pipeline. The
shallow case -- a model with nothing to continue from -- is covered with the CSP+LDA sibling.

Skipped on the no-extras CI job: the deep pipeline is torch-gated.
"""
import warnings

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("lightning")

import torch

from medusa.pipelines.base import DecodingPipeline
from medusa.pipelines.bci.motor_decoding import MICSPLDAPipeline, MIEEGInceptionPipeline

#: 'cpu' so weight comparisons are exact and device-independent.
TRAINING = {"max_epochs": 2, "batch_size": 8, "val_split": 0.2, "patience": 2,
            "device": "cpu", "verbose": "silent"}


def _deep(channels, **classifier):
    return MIEEGInceptionPipeline(
        channels=channels, classifier={"training": dict(TRAINING), **classifier})


def _weights(module):
    return {name: p.detach().cpu().clone() for name, p in module.named_parameters()}


def _moved(before, module):
    return any(not torch.equal(before[n], p.detach().cpu())
               for n, p in module.named_parameters())


class TestContinuedFit:
    """``fit`` trains from whatever the pipeline holds."""

    def test_a_second_fit_keeps_the_same_model(self, mi_recording, mi_channels):
        pipe = _deep(mi_channels).fit([mi_recording])
        backbone, before = pipe.clf.backbone, _weights(pipe.clf.backbone)

        pipe.fit([mi_recording])

        assert pipe.clf.backbone is backbone        # not rebuilt
        assert _moved(before, pipe.clf.backbone)    # trained further
        assert len(pipe.clf.history_) == 2          # one entry per phase

    def test_restart_forgets_it(self, mi_recording, mi_channels):
        pipe = _deep(mi_channels).fit([mi_recording])
        backbone = pipe.clf.backbone

        pipe.restart()

        assert pipe.clf is None and pipe.fs is None and not pipe._fitted
        pipe.fit([mi_recording])
        assert pipe.clf.backbone is not backbone    # built from the settings again
        assert len(pipe.clf.history_) == 1


class TestResetHead:
    """Keep the learned features, train a new classifier on them."""

    def test_the_backbone_survives(self, mi_recording, mi_channels):
        pipe = _deep(mi_channels).fit([mi_recording])
        backbone, before = pipe.clf.backbone, _weights(pipe.clf.backbone)

        pipe.reset_head()

        assert pipe.clf.backbone is backbone and not _moved(before, pipe.clf.backbone)

    def test_the_pipeline_is_unfitted_until_the_next_fit(self, mi_recording, mi_channels):
        pipe = _deep(mi_channels).fit([mi_recording]).reset_head()
        with pytest.raises(RuntimeError, match="not fitted"):
            pipe.predict(mi_recording)

        pipe.fit([mi_recording])
        assert np.isfinite(pipe.predict(mi_recording)).all()


class TestSetBackbone:
    """Pretrained features enter a pipeline as a module, whatever trained them."""

    def test_it_installs_the_module_and_drops_the_head(self, mi_recording, mi_channels):
        source = _deep(mi_channels).fit([mi_recording])
        target = _deep(mi_channels)

        target.set_backbone(source.clf.backbone)

        assert target.clf.backbone is source.clf.backbone
        assert not target._fitted

    def test_the_next_fit_trains_a_head_on_them(self, mi_recording, mi_channels):
        source = _deep(mi_channels).fit([mi_recording])
        before = _weights(source.clf.backbone)

        target = _deep(mi_channels).set_backbone(source.clf.backbone)
        target.settings.update_from_dict(
            {"classifier": {"training": {"profile": "finetune"}}})
        target.fit([mi_recording])

        assert np.isfinite(target.predict(mi_recording)).all()
        # 'finetune' froze the backbone, so the features they share are untouched
        assert not _moved(before, source.clf.backbone)


class TestShallowPipeline:
    """A model with nothing to continue from still gets the same lifecycle."""

    def test_restart_clears_the_fitted_state(self, mi_recording, mi_channels):
        # n_filters <= n_channels: the fixture has three
        pipe = MICSPLDAPipeline(channels=mi_channels,
                                csp={"n_filters": 2}).fit([mi_recording])
        assert pipe._fitted

        pipe.restart()

        assert pipe.csp is None and pipe.clf is None
        assert pipe.fs is None and not pipe._fitted


class TestGuards:
    """A continued fit reads the live settings, which can disagree with the model."""

    def test_a_changed_epoch_window_raises(self, mi_recording, mi_channels):
        """Caught by the backbone itself: ``prepare_X`` validates the epoch shape."""
        pipe = _deep(mi_channels).fit([mi_recording])
        pipe.settings.update_from_dict(
            {"segmentation": {"w_segment_t": [500.0, 1500.0]}})

        with pytest.raises(ValueError, match="each epoch must be"):
            pipe.fit([mi_recording])

    def test_a_changed_architecture_raises(self, mi_recording, mi_channels):
        pipe = _deep(mi_channels).fit([mi_recording])
        pipe.settings.update_from_dict({"classifier": {"arch": "eeg_inception_v2"}})

        with pytest.raises(ValueError, match="restart"):
            pipe.fit([mi_recording])

    def test_a_changed_band_warns(self, mi_recording, mi_channels):
        """Same epoch shape, different signal -- nothing else would notice."""
        pipe = _deep(mi_channels).fit([mi_recording])
        pipe.settings.update_from_dict({"filter": {"cutoff": [8.0, 30.0]}})

        with pytest.warns(UserWarning, match="settings changed since the last fit"):
            pipe.fit([mi_recording])

    def test_changing_the_training_settings_is_silent(self, mi_recording, mi_channels):
        """A new phase is *supposed* to change these, so they are outside the guard."""
        pipe = _deep(mi_channels).fit([mi_recording])
        pipe.settings.update_from_dict(
            {"classifier": {"training": {"learning_rate": 1e-4, "max_epochs": 1}}})

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pipe.fit([mi_recording])

        assert not [w for w in caught if "settings changed" in str(w.message)]


class TestPersistence:
    """The bundle carries the model, and a loaded pipeline is still a live one."""

    def test_a_fitted_pipeline_round_trips_and_continues(self, mi_recording, mi_channels,
                                                         tmp_path):
        pipe = _deep(mi_channels).fit([mi_recording])
        expected = pipe.predict(mi_recording)
        path = tmp_path / "pipe.pkl"
        pipe.save(str(path))

        loaded = DecodingPipeline.load(str(path))

        np.testing.assert_allclose(loaded.predict(mi_recording), expected,
                                   rtol=1e-5, atol=1e-6)
        loaded.fit([mi_recording])                  # continues, does not rebuild
        assert len(loaded.clf.history_) == 2

    def test_the_backbone_survives_reset_head_and_a_save(self, mi_recording, mi_channels,
                                                         tmp_path):
        """The saved bundle keeps the classifier whenever there is one, fitted or not."""
        pipe = _deep(mi_channels).fit([mi_recording])
        before = _weights(pipe.clf.backbone)
        pipe.reset_head()
        path = tmp_path / "features.pkl"
        pipe.save(str(path))

        loaded = DecodingPipeline.load(str(path))

        assert loaded.clf is not None and not loaded._fitted
        assert not _moved(before, loaded.clf.backbone)
