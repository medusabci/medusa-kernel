"""Tests for ``min_delta``, the size of a loss drop that counts as an improvement.

Early stopping without it treats any drop as progress, so a run that is only wobbling
around its best value keeps resetting ``patience`` and spends every remaining epoch on
noise. ``min_delta`` puts a floor under what counts, and the pipelines expose it as a
training setting, so the value has to survive the whole way down: settings tree ->
``training_kwargs`` -> estimator -> Lightning's ``EarlyStopping``.

Skipped on the no-extras CI job: torch / Lightning are optional.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("lightning")

import torch.nn as nn
from sklearn.base import clone

from medusa.ml.torch_models import _engine
from medusa.ml.torch_models.backbones.eeg_inception_v2 import EEGInceptionV2
from medusa.ml.torch_models.classification import TorchClassifier
from medusa.pipelines.torch_base import training_kwargs

N_SAMPLES, N_CHA = 64, 8


def _backbone():
    # small temporal scales: they fit inside the 64-sample epoch, and keep these tests fast
    return EEGInceptionV2(input_samples=N_SAMPLES, n_cha=N_CHA,
                          temp_scales_samples=(15, 11, 7))


def _data(n=24, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, N_SAMPLES, N_CHA)).astype(np.float32)
    y = np.tile([0, 1], n // 2)
    return X, y


class TestTheEstimatorParameter:

    def test_it_defaults_to_accepting_any_drop(self):
        assert TorchClassifier(nn.Identity()).min_delta == 0.0

    def test_clone_preserves_it(self):
        assert clone(TorchClassifier(nn.Identity(), min_delta=0.001)).min_delta == 0.001

    def test_it_survives_save_and_load(self, tmp_path):
        clf = TorchClassifier(_backbone(), min_delta=0.001, max_epochs=1, device="cpu")
        path = tmp_path / "clf.pkl"
        clf.save(str(path))
        assert TorchClassifier.load(str(path)).min_delta == 0.001


class TestItReachesEarlyStopping:
    """The one line that does the work: the callback must be built with the value."""

    def test_the_callback_is_built_with_it(self, monkeypatch):
        captured = {}
        real = _engine.EarlyStopping

        def spy(**kwargs):
            captured.update(kwargs)
            return real(**kwargs)

        monkeypatch.setattr(_engine, "EarlyStopping", spy)
        X, y = _data()
        TorchClassifier(_backbone(), min_delta=0.25, patience=3, max_epochs=1,
                        val_split=0.25, device="cpu", verbose=0).fit(X, y)
        assert captured["min_delta"] == 0.25
        assert captured["patience"] == 3


class TestItReachesTheEstimatorFromTheSettings:

    def test_training_kwargs_maps_the_leaf(self):
        cfg = {"learning_rate": 0.01, "max_epochs": 100, "batch_size": 256,
               "class_weight": None, "val_split": 0.1, "patience": 5,
               "min_delta": 0.001, "random_state": None, "device": "auto",
               "verbose": "epoch"}
        assert training_kwargs(cfg)["min_delta"] == 0.001

    def test_the_vep_speller_pipeline_hands_it_over(self):
        from medusa.pipelines.bci.vep_spellers.decoding.bwr_eeg_inception import (
            bwr_eeg_inception_settings)
        cfg = bwr_eeg_inception_settings().to_dict()["classifier"]["training"]
        assert training_kwargs(cfg)["min_delta"] == 0.001
