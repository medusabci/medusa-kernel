"""``classifier.training.profile``: what is allowed to move in a training phase.

The profile is applied to the backbone before every fit, so the setting is authoritative
rather than only effective when it changes. ``train`` and ``finetune`` are pure
``requires_grad`` manipulation and live in
:class:`~medusa.pipelines.torch_base.TorchPipeline`; a pipeline that needs more declares
the names in ``TRAINING_PROFILES`` and implements them in ``_apply_training_profile``,
which the last two tests exercise.

The point of ``finetune`` is the asymmetry it produces: no weight moves, but the
normalization layers' running statistics still follow the new subject, because the module
stays in training mode.

Skipped on the no-extras CI job: the deep pipeline is torch-gated.
"""
import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("lightning")

import torch

from medusa.pipelines.bci.motor_decoding import MIEEGInceptionPipeline

#: 'cpu' so weight comparisons are exact and device-independent.
TRAINING = {"max_epochs": 2, "batch_size": 8, "val_split": 0.2, "patience": 2,
            "device": "cpu", "verbose": "silent"}


def _pipeline(channels, cls=MIEEGInceptionPipeline, **training):
    return cls(channels=channels,
               classifier={"training": dict(TRAINING, **training)})


def _set_profile(pipe, profile):
    pipe.settings.update_from_dict({"classifier": {"training": {"profile": profile}}})
    return pipe


def _params(module):
    return {name: p.detach().cpu().clone() for name, p in module.named_parameters()}


def _stats(module):
    """The normalization running statistics -- buffers, not parameters."""
    return {name: b.detach().cpu().clone() for name, b in module.named_buffers()
            if name.endswith(("running_mean", "running_var"))}


def _moved(before, current):
    return any(not torch.equal(before[n], current[n]) for n in before)


@pytest.fixture
def trained(mi_recording, mi_channels):
    """A pipeline already trained for one phase with the default 'train' profile."""
    return _pipeline(mi_channels).fit([mi_recording])


class TestFinetune:
    """Freeze the weights, let the statistics and the head follow the new data."""

    def test_no_backbone_weight_moves(self, trained, mi_recording):
        before = _params(trained.clf.backbone)

        _set_profile(trained, "finetune").fit([mi_recording])

        assert not _moved(before, _params(trained.clf.backbone))

    def test_the_normalization_statistics_still_adapt(self, trained, mi_recording):
        before = _stats(trained.clf.backbone)
        assert before, "the backbone should have normalization layers to adapt"

        _set_profile(trained, "finetune").fit([mi_recording])

        assert _moved(before, _stats(trained.clf.backbone))

    def test_the_head_trains(self, trained, mi_recording):
        before = _params(trained.clf.head_)

        _set_profile(trained, "finetune").fit([mi_recording])

        assert _moved(before, _params(trained.clf.head_))

    def test_it_warns_when_the_backbone_was_never_trained(self, mi_recording, mi_channels):
        pipe = _pipeline(mi_channels, profile="finetune")
        with pytest.warns(UserWarning, match="never been trained"):
            pipe.fit([mi_recording])


class TestTrain:
    """The default: everything moves."""

    def test_the_backbone_keeps_training(self, trained, mi_recording):
        before = _params(trained.clf.backbone)

        trained.fit([mi_recording])

        assert _moved(before, _params(trained.clf.backbone))

    def test_it_unfreezes_what_a_previous_phase_froze(self, trained, mi_recording):
        _set_profile(trained, "finetune").fit([mi_recording])
        assert not any(p.requires_grad for p in trained.clf.backbone.parameters())

        before = _params(trained.clf.backbone)
        _set_profile(trained, "train").fit([mi_recording])

        assert all(p.requires_grad for p in trained.clf.backbone.parameters())
        assert _moved(before, _params(trained.clf.backbone))


class TestCustom:
    """The escape hatch: the pipeline touches nothing it did not set."""

    def test_hand_set_flags_survive_the_next_fit(self, trained, mi_recording):
        for p in trained.clf.backbone.parameters():
            p.requires_grad = False
        before = _params(trained.clf.backbone)

        _set_profile(trained, "custom").fit([mi_recording])

        assert not any(p.requires_grad for p in trained.clf.backbone.parameters())
        assert not _moved(before, _params(trained.clf.backbone))


class TestUnknownProfile:
    """The settings tree rejects it first, listing what this pipeline offers."""

    def test_it_is_refused_before_training_starts(self, mi_recording, mi_channels):
        pipe = _pipeline(mi_channels)
        with pytest.warns(UserWarning):        # the tree flags the value as it is written
            _set_profile(pipe, "not_a_profile")

        with pytest.raises(ValueError, match="not_a_profile") as excinfo:
            pipe.fit([mi_recording])
        assert "train" in str(excinfo.value) and "custom" in str(excinfo.value)


class _ExtraProfilePipeline(MIEEGInceptionPipeline):
    """A pipeline that adds one profile of its own."""

    TRAINING_PROFILES = MIEEGInceptionPipeline.TRAINING_PROFILES + ("head_only",)

    def _apply_training_profile(self, profile, backbone, cfg):
        if profile == "head_only":
            self.applied = (profile, cfg["classifier"]["arch"])
            for p in backbone.parameters():
                p.requires_grad = False
            return
        super()._apply_training_profile(profile, backbone, cfg)


class _DeclaredOnlyPipeline(MIEEGInceptionPipeline):
    """Declares a profile but forgets to implement it."""

    TRAINING_PROFILES = MIEEGInceptionPipeline.TRAINING_PROFILES + ("declared_only",)


class TestPipelineSpecificProfiles:
    """A pipeline extends the vocabulary and implements the extra names."""

    def test_the_setting_offers_what_the_pipeline_implements(self):
        options = (_ExtraProfilePipeline.default_settings()
                   .get_item("classifier", "training", "profile").tree["value_options"])
        assert "head_only" in options
        assert "head_only" not in (MIEEGInceptionPipeline.default_settings()
                                   .get_item("classifier", "training", "profile")
                                   .tree["value_options"])

    def test_the_hook_receives_the_profile_and_the_config(self, mi_recording, mi_channels):
        pipe = _pipeline(mi_channels, cls=_ExtraProfilePipeline, profile="head_only")

        pipe.fit([mi_recording])

        assert pipe.applied == ("head_only", "eeg_inception_v1")
        assert not any(p.requires_grad for p in pipe.clf.backbone.parameters())
        assert np.isfinite(pipe.predict(mi_recording)).all()

    def test_declaring_a_profile_without_implementing_it_raises(self, mi_recording,
                                                                mi_channels):
        pipe = _pipeline(mi_channels, cls=_DeclaredOnlyPipeline, profile="declared_only")
        with pytest.raises(ValueError, match="does not implement"):
            pipe.fit([mi_recording])
