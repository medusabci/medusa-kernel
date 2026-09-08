"""A second ``fit`` continues the model; ``reset_head`` is the way to start the head over.

The estimator owns the head, so that is what it can reset; the backbone belongs to whoever
built it and is trained in place across fits. Together these give multi-phase training
(pretrain, then adapt) without any mode argument: fit again to continue, call
``reset_head()`` first to train a new classifier on the features learned so far.

Skipped on the no-extras CI job: torch / Lightning are optional.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("lightning")

import torch
import torch.nn as nn

from medusa.ml.torch_models.classification import (
    TorchClassifier, TorchMultiTaskClassifier)

N_SAMPLES, N_CHA = 16, 4


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


def _data(n=64, n_classes=2, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, N_SAMPLES, N_CHA).astype("float32")
    y = np.tile(np.arange(n_classes), n // n_classes + 1)[:n]
    return X, y


def _clf(**kwargs):
    return TorchClassifier(_Linear(), max_epochs=2, batch_size=16, val_split=0.25,
                           patience=2, device="cpu", verbose=0, **kwargs)


def _weights(module):
    return {name: p.detach().clone() for name, p in module.named_parameters()}


def _moved(before, module):
    return any(not torch.equal(before[n], p) for n, p in module.named_parameters())


class TestContinuedFit:
    """Fitting twice trains one model further, rather than starting a second one."""

    def test_the_head_object_survives_a_second_fit(self):
        clf = _clf().fit(*_data())
        head = clf.head_
        clf.fit(*_data(seed=1))
        assert clf.head_ is head

    def test_the_backbone_and_head_keep_training(self):
        clf = _clf().fit(*_data())
        before_backbone, before_head = _weights(clf.backbone), _weights(clf.head_)

        clf.fit(*_data(seed=1))

        assert _moved(before_backbone, clf.backbone)
        assert _moved(before_head, clf.head_)

    def test_every_fit_appends_a_history_entry(self):
        clf = _clf().fit(*_data())
        assert len(clf.history_) == 1
        clf.fit(*_data(seed=1))
        assert len(clf.history_) == 2

    def test_changing_the_classes_is_refused(self):
        clf = _clf().fit(*_data())
        with pytest.raises(ValueError, match="reset_head"):
            clf.fit(*_data(n_classes=4))


class TestResetHead:
    """The explicit cold start for the part the estimator owns."""

    def test_it_drops_the_head_and_the_classes(self):
        clf = _clf().fit(*_data())
        clf.reset_head()
        assert not hasattr(clf, "head_") and not hasattr(clf, "classes_")

    def test_the_backbone_is_untouched(self):
        clf = _clf().fit(*_data())
        before = _weights(clf.backbone)
        clf.reset_head()
        assert not _moved(before, clf.backbone)

    def test_the_next_fit_builds_a_new_head(self):
        clf = _clf().fit(*_data())
        head = clf.head_
        clf.reset_head().fit(*_data())
        assert clf.head_ is not head

    def test_it_lets_the_classes_change(self):
        """The transfer case: same features, a different set of labels."""
        clf = _clf().fit(*_data())
        clf.reset_head().fit(*_data(n_classes=4))
        assert list(clf.classes_) == [0, 1, 2, 3]

    def test_it_is_harmless_before_any_fit(self):
        assert _clf().reset_head() is not None


class TestNothingTrainable:
    """A model whose every parameter is frozen is a configuration mistake, not a no-op."""

    def test_it_raises_instead_of_training_nothing(self):
        clf = _clf().fit(*_data())
        for p in list(clf.backbone.parameters()) + list(clf.head_.parameters()):
            p.requires_grad = False
        with pytest.raises(RuntimeError, match="nothing to train"):
            clf.fit(*_data(seed=1))

    def test_a_frozen_backbone_still_trains_the_head(self):
        clf = _clf().fit(*_data())
        for p in clf.backbone.parameters():
            p.requires_grad = False
        before_backbone, before_head = _weights(clf.backbone), _weights(clf.head_)

        clf.fit(*_data(seed=1))

        assert not _moved(before_backbone, clf.backbone)
        assert _moved(before_head, clf.head_)


class TestMultiTask:
    """The same rules per task, over ``heads_`` / ``classes_``."""

    @staticmethod
    def _tasks(seed=0):
        X, y = _data(seed=seed)
        return X, {"a": y, "b": y}

    def test_the_heads_survive_a_second_fit(self):
        mt = TorchMultiTaskClassifier(_Linear(), max_epochs=2, batch_size=16,
                                      device="cpu", verbose=0)
        mt.fit(*self._tasks())
        heads = mt.heads_
        mt.fit(*self._tasks(seed=1))
        assert mt.heads_ is heads

    def test_changing_the_tasks_is_refused(self):
        mt = TorchMultiTaskClassifier(_Linear(), max_epochs=2, batch_size=16,
                                      device="cpu", verbose=0)
        mt.fit(*self._tasks())
        X, y = self._tasks()
        with pytest.raises(ValueError, match="reset_head"):
            mt.fit(X, {"a": y["a"], "c": y["b"]})

    def test_reset_head_lets_the_tasks_change(self):
        mt = TorchMultiTaskClassifier(_Linear(), max_epochs=2, batch_size=16,
                                      device="cpu", verbose=0)
        mt.fit(*self._tasks())
        X, y = self._tasks()
        mt.reset_head().fit(X, {"c": y["a"]})
        assert list(mt.heads_) == ["c"]
