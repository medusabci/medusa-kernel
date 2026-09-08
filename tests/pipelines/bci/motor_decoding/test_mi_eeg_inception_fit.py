"""End-to-end ``fit`` / ``predict`` for the deep motor pipeline, on every architecture.

The sibling of ``vep_spellers/test_bwr_eeg_inception_fit.py``: it drives the real path, where
``fit`` reads ``classifier.arch``, hands that architecture's own group to
:func:`~medusa.pipelines.bci._torch_backbones.build_backbone` with the epoch shape and the
epoch rate, and trains. A **smoke** test -- two training epochs on a synthetic recording says
nothing about accuracy, only that the chain runs and returns per-trial posteriors.

Skipped on the no-extras CI job: this pipeline is torch-gated.
"""
import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("lightning")

from medusa.pipelines.bci._torch_backbones import ARCHITECTURES
from medusa.pipelines.bci.motor_decoding import MIEEGInceptionPipeline
from medusa.pipelines.bci.trial_events import trial_arrays, trial_labels

TRAINING = {"max_epochs": 2, "batch_size": 8, "val_split": 0.2, "patience": 2,
            "verbose": "silent"}


def _pipeline(channels, arch):
    # The stock window and target_fs give 2 s @ 128 Hz = 256 samples, and the stock
    # 500/250/125 ms scales give 64/32/16, so both architectures build.
    return MIEEGInceptionPipeline(
        channels=channels,
        classifier={"arch": arch, "training": dict(TRAINING)})


@pytest.mark.parametrize("arch", list(ARCHITECTURES))
def test_fit_and_predict_through_every_architecture(mi_recording, mi_channels, arch):
    pipe = _pipeline(mi_channels, arch).fit([mi_recording])

    scores = pipe.predict(mi_recording)
    onsets, _, _ = trial_arrays(mi_recording.events)
    assert scores.shape == (len(onsets), len(np.unique(trial_labels(mi_recording))))
    assert np.isfinite(scores).all()
    np.testing.assert_allclose(scores.sum(axis=1), 1.0, rtol=1e-5)


@pytest.mark.parametrize("arch", list(ARCHITECTURES))
def test_a_fitted_pipeline_round_trips(mi_recording, mi_channels, arch, tmp_path):
    """The per-architecture group and its group-list must survive save/load untouched."""
    pipe = _pipeline(mi_channels, arch).fit([mi_recording])
    expected = pipe.predict(mi_recording)

    path = tmp_path / "pipe.pkl"
    pipe.save(str(path))
    reloaded = MIEEGInceptionPipeline.load(str(path))

    assert reloaded.cfg["classifier"] == pipe.cfg["classifier"]
    np.testing.assert_allclose(reloaded.predict(mi_recording), expected,
                               rtol=1e-5, atol=1e-6)
