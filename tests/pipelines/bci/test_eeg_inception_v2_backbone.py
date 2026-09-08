"""Regression guard: the ``eeg_inception_v2`` arch must actually build.

Both deep pipelines used to pass an ``activation=`` keyword to
:class:`~medusa.ml.torch_models.backbones.eeg_inception_v2.EEGInceptionV2`, which has no such
parameter (it is hard-wired to ELU). Selecting the v2 architecture therefore raised
``TypeError`` deep inside ``fit`` -- in *both* pipelines, because each hand-maintained its own
copy of the same mapping. They now share
:mod:`~medusa.pipelines.bci._torch_backbones`, so these tests drive that one mapping through
each pipeline's own settings and build the backbone directly, with no training.

Skipped on the no-extras CI job: both pipelines are torch-gated.
"""
from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("lightning")

from medusa.ml.torch_models.backbones.eeg_inception import EEGInception
from medusa.ml.torch_models.backbones.eeg_inception_v2 import EEGInceptionV2
from medusa.pipelines.bci._torch_backbones import ARCHITECTURES, build_backbone
from medusa.pipelines.bci.motor_decoding import MIEEGInceptionPipeline
from medusa.pipelines.bci.vep_spellers import BWREEGInceptionPipeline

ARCHS = [("eeg_inception_v1", EEGInception), ("eeg_inception_v2", EEGInceptionV2)]

#: One case per deep pipeline: its class, channels, and an epoch length its own default
#: window and target_fs would produce.
CASES = [
    (BWREEGInceptionPipeline, ["Fz", "Cz", "Pz", "Oz"], 64),      # 500 ms @ 128 Hz
    (MIEEGInceptionPipeline, ["C3", "Cz", "C4"], 256),            # 2 s @ 128 Hz
]


def _all_keys(node):
    """Every key anywhere in a nested settings dict (groups and group-list elements)."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield key
            yield from _all_keys(value)
    elif isinstance(node, list):
        for value in node:
            yield from _all_keys(value)


@pytest.mark.parametrize("pipeline_cls, channels, n_samples", CASES)
@pytest.mark.parametrize("arch, backbone_cls", ARCHS)
def test_every_arch_builds(pipeline_cls, channels, n_samples, arch, backbone_cls):
    pipe = pipeline_cls(channels=channels, classifier={"arch": arch})
    cfg = pipe.cfg
    backbone = build_backbone(cfg["classifier"], input_samples=n_samples,
                              n_cha=len(channels),
                              rate=cfg["segmentation"]["target_fs"])
    assert isinstance(backbone, backbone_cls)
    assert backbone.backbone_features > 0


@pytest.mark.parametrize("pipeline_cls, channels, n_samples", CASES)
def test_no_activation_setting_survives(pipeline_cls, channels, n_samples):
    """Neither backbone takes an activation, so no classifier leaf may advertise one."""
    classifier = pipeline_cls.default_settings().to_dict()["classifier"]
    assert "activation" not in set(_all_keys(classifier))


@pytest.mark.parametrize("pipeline_cls, channels, n_samples", CASES)
def test_both_pipelines_share_one_architecture_schema(pipeline_cls, channels, n_samples):
    """One mapping, two consumers -- which is what stops the two copies drifting again."""
    classifier = pipeline_cls.default_settings().to_dict()["classifier"]
    assert set(ARCHITECTURES) <= set(classifier)
    assert classifier["arch"] in ARCHITECTURES
