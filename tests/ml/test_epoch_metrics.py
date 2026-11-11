"""``epoch_metrics``: validation scores recorded after every epoch, next to the losses.

The metrics are computed in the validation pass training already runs, so they must leave
the model untouched: a seeded fit gives the same weights with and without them. They need
a validation fold; without one, ``fit`` warns and records nothing.

Every test pins ``device='cpu'``: CUDA kernels are not deterministic by default.

Skipped on the no-extras CI job: torch / Lightning are optional.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("lightning")

import torch
import torch.nn as nn

from medusa.ml.torch_models.classification import TorchClassifier

N_SAMPLES, N_CHA = 16, 4     # tiny on purpose: these tests are about bookkeeping


class _Linear(nn.Module):
    """Minimal backbone: one linear layer over the flattened epoch."""

    input_layout = ("batch", "n_samples", "n_channels")
    backbone_features = 8

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(N_SAMPLES * N_CHA, self.backbone_features)

    def forward(self, x):
        return self.fc(x.flatten(1))

    def get_config(self):
        return {}


def _data(n=96, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, N_SAMPLES, N_CHA).astype("float32")
    y = rng.randint(0, 3, size=n)
    return X, y


def _fit(epoch_metrics=(), val_split=0.25):
    torch.manual_seed(0)                 # the backbone's initial weights
    clf = TorchClassifier(_Linear(), max_epochs=4, batch_size=16, val_split=val_split,
                          patience=10, device="cpu", verbose=0, random_state=0,
                          epoch_metrics=epoch_metrics)
    return clf.fit(*_data())


def test_default_records_the_losses_only():
    history = _fit().history_[-1]
    assert [key for key in history if key.endswith("_curve")] == \
        ["train_loss_curve", "val_loss_curve"]


def test_each_metric_has_one_validation_value_per_epoch():
    history = _fit(epoch_metrics=["accuracy", "balanced_accuracy"]).history_[-1]
    for name in ("val_accuracy_curve", "val_balanced_accuracy_curve"):
        curve = history[name]
        assert len(curve) == history["epochs"]
        assert all(0.0 <= value <= 1.0 for value in curve)


def test_metrics_do_not_change_the_model():
    plain = _fit()
    scored = _fit(epoch_metrics=["balanced_accuracy"])
    for name, weights in plain.backbone.state_dict().items():
        assert torch.equal(weights, scored.backbone.state_dict()[name])
    assert torch.equal(plain.head_.weight, scored.head_.weight)
    assert plain.history_[-1]["val_loss_curve"] == scored.history_[-1]["val_loss_curve"]


def test_unknown_metric_raises():
    with pytest.raises(ValueError, match="unknown epoch_metrics"):
        _fit(epoch_metrics=["f1"])


def test_metrics_without_a_validation_fold_warn_and_record_nothing():
    with pytest.warns(UserWarning, match="val_split is off"):
        history = _fit(epoch_metrics=["accuracy"], val_split=None).history_[-1]
    assert "val_accuracy_curve" not in history


def test_pipeline_setting_reaches_the_classifier():
    pytest.importorskip("rich")
    from medusa.pipelines.bci.vep_spellers import bwr_eeg_inception_settings
    from medusa.pipelines.torch_base import training_kwargs

    settings = bwr_eeg_inception_settings()
    assert training_kwargs(settings.to_dict()["classifier"]["training"])[
        "epoch_metrics"] == []
    settings.set_value("classifier", "training", "epoch_metrics",
                       value=["balanced_accuracy"])
    assert settings.validate() == []
    assert training_kwargs(settings.to_dict()["classifier"]["training"])[
        "epoch_metrics"] == ["balanced_accuracy"]
