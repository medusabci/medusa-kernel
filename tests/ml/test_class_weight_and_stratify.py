"""Tests for the two knobs that make an unbalanced classification problem trainable.

``class_weight`` (off by default) re-weights the cross-entropy so a rare class is not cheap
to ignore, and ``val_split_stratify`` (on by default) keeps each class's share of the data
in the validation split that early stopping watches. A 3 %-target bit-wise-reconstruction problem
needs both: without the first the network can collapse onto the majority class, and without
the second the validation fold can end up with almost no targets.

Skipped on the no-extras CI job: torch / Lightning are optional.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("lightning")

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import clone
from torch.utils.data import TensorDataset

from medusa.ml.torch_models._engine import _stratified_split_indices
from medusa.ml.torch_models.backbones.eeg_inception_v2 import EEGInceptionV2
from medusa.ml.torch_models.classification import (
    ClassificationTask, TorchClassifier, _class_weights)

N_SAMPLES, N_CHA = 64, 8


def _backbone():
    # small temporal scales: they fit inside the 64-sample epoch, and keep these tests fast
    return EEGInceptionV2(input_samples=N_SAMPLES, n_cha=N_CHA,
                          temp_scales_samples=(15, 11, 7))


def _task(class_weight=None):
    """A ClassificationTask whose ``log`` is captured, so its steps run without a Trainer."""
    torch.manual_seed(0)
    task = ClassificationTask(_backbone(), 2, class_weight=class_weight).eval()
    task.logged = {}
    task.log = lambda name, value, **kwargs: task.logged.__setitem__(name, value)
    return task


def _batch():
    """One forward-ready batch: the backbone's ('batch', 'feature', 'n_samples',
    'n_channels') layout, i.e. what the estimator's ``_prepare`` would produce."""
    torch.manual_seed(0)
    return torch.randn(16, 1, N_SAMPLES, N_CHA), torch.tensor([0] * 14 + [1] * 2)


def _unbalanced(n=400, minority=0.05, seed=0):
    """``(X, y)`` with only ``minority`` of the observations in class 1."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, N_SAMPLES, N_CHA).astype("float32")
    y = (rng.random_sample(n) < minority).astype(int)
    y[:2] = [0, 1]                      # both classes are always present
    return X, y


class TestClassWeights:
    """``_class_weights`` maps the sklearn-style spec to one weight per class."""

    CLASSES = np.array([0, 1])
    Y_IDX = np.array([0] * 90 + [1] * 10)

    def test_none_means_unweighted(self):
        assert _class_weights(None, self.CLASSES, self.Y_IDX) is None

    def test_balanced_follows_the_sklearn_rule(self):
        weights = _class_weights("balanced", self.CLASSES, self.Y_IDX)
        # n_samples / (n_classes * count): 100 / (2 * 90) and 100 / (2 * 10)
        np.testing.assert_allclose(weights, [100 / 180, 100 / 20], rtol=1e-6)

    def test_balanced_gives_each_class_the_same_total_weight(self):
        weights = _class_weights("balanced", self.CLASSES, self.Y_IDX)
        counts = np.bincount(self.Y_IDX)
        np.testing.assert_allclose(weights * counts, [50.0, 50.0], rtol=1e-6)

    def test_a_dict_sets_the_weights_by_label(self):
        weights = _class_weights({0: 1.0, 1: 9.0}, self.CLASSES, self.Y_IDX)
        np.testing.assert_allclose(weights, [1.0, 9.0])

    def test_the_weights_follow_the_class_order_not_the_dict_order(self):
        classes = np.array(["left", "right"])
        y_idx = np.array([0, 0, 1, 1])
        weights = _class_weights({"right": 3.0, "left": 1.0}, classes, y_idx)
        np.testing.assert_allclose(weights, [1.0, 3.0])

    def test_an_unknown_spec_is_rejected(self):
        with pytest.raises(ValueError):
            _class_weights("inverse", self.CLASSES, self.Y_IDX)


class TestWeightedLoss:
    """The weights must actually reach the loss, in training *and* validation."""

    WEIGHT = [1.0, 7.0]

    def test_training_step_applies_the_weight(self):
        task, (x, y) = _task(self.WEIGHT), _batch()
        with torch.no_grad():
            logits = task(x)
            expected = F.cross_entropy(logits, y, weight=torch.tensor(self.WEIGHT))
            unweighted = F.cross_entropy(logits, y)
            got = task.training_step((x, y), 0)
        assert torch.allclose(got, expected)
        assert not torch.allclose(got, unweighted)   # the weight cannot be dropped

    def test_no_weight_is_the_plain_cross_entropy(self):
        task, (x, y) = _task(), _batch()
        with torch.no_grad():
            got = task.training_step((x, y), 0)
            assert torch.allclose(got, F.cross_entropy(task(x), y))
        assert task.class_weight is None

    def test_validation_is_weighted_the_same_way(self):
        """Otherwise early stopping would watch a different objective than training."""
        task, (x, y) = _task(self.WEIGHT), _batch()
        with torch.no_grad():
            task.validation_step((x, y), 0)
            expected = F.cross_entropy(task(x), y, weight=torch.tensor(self.WEIGHT))
        assert torch.allclose(task.logged["val_loss"], expected)

    def test_the_weight_is_a_buffer_so_it_follows_the_module(self):
        task = ClassificationTask(_backbone(), 2, class_weight=self.WEIGHT)
        assert "class_weight" in dict(task.named_buffers())


class TestStratifiedSplitIndices:
    """``_stratified_split_indices`` returns None whenever it cannot balance the split."""

    def test_keeps_every_class_on_both_sides(self):
        labels = np.array([0] * 90 + [1] * 10)
        train_idx, val_idx = _stratified_split_indices(100, 20, labels)
        assert len(val_idx) == 20 and len(train_idx) == 80
        assert set(train_idx) | set(val_idx) == set(range(100))
        assert not set(train_idx) & set(val_idx)
        assert labels[val_idx].sum() == 2          # 10 % of 20, the class's share
        assert labels[train_idx].sum() == 8

    def test_no_stratification_for_a_singleton_class(self):
        labels = np.array([0] * 99 + [1])
        assert _stratified_split_indices(100, 20, labels) is None

    def test_no_stratification_when_the_fold_is_smaller_than_the_classes(self):
        labels = np.repeat(np.arange(5), 20)
        assert _stratified_split_indices(100, 3, labels) is None

    def test_no_stratification_for_a_single_class(self):
        assert _stratified_split_indices(100, 20, np.zeros(100, int)) is None

    def test_mismatched_labels_are_ignored(self):
        assert _stratified_split_indices(100, 20, np.zeros(7, int)) is None


class TestSplitLoaders:
    """The estimator's own splitter, which is what ``fit`` calls."""

    LABELS = np.array([0] * 180 + [1] * 20)

    @staticmethod
    def _dataset(labels):
        return TensorDataset(torch.zeros(len(labels), 2), torch.as_tensor(labels))

    @staticmethod
    def _clf(**kwargs):
        # the backbone is untouched by the splitter, so keep it trivial
        return TorchClassifier(nn.Identity(), batch_size=16, **kwargs)

    @staticmethod
    def _val_labels(loader):
        return torch.cat([y for _, y in loader]).numpy()

    def test_stratified_by_default(self):
        clf = self._clf(val_split=0.2)
        _, val_loader = clf._loaders_from_dataset(self._dataset(self.LABELS),
                                                  labels=self.LABELS)
        assert self._val_labels(val_loader).sum() == 4      # 10 % of 40

    def test_stratify_off_falls_back_to_a_random_split(self):
        clf = self._clf(val_split=0.2, val_split_stratify=False)
        with warnings.catch_warnings():
            warnings.simplefilter("error")    # no warning when it was not asked for
            _, val_loader = clf._loaders_from_dataset(self._dataset(self.LABELS),
                                                      labels=self.LABELS)
        assert len(self._val_labels(val_loader)) == 40

    def test_an_impossible_split_warns_and_falls_back(self):
        labels = np.array([0] * 199 + [1])
        clf = self._clf(val_split=0.2)
        with pytest.warns(UserWarning, match="cannot stratify"):
            _, val_loader = clf._loaders_from_dataset(self._dataset(labels), labels=labels)
        assert len(self._val_labels(val_loader)) == 40

    def test_no_labels_means_no_stratification_and_no_warning(self):
        clf = self._clf(val_split=0.2)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _, val_loader = clf._loaders_from_dataset(self._dataset(self.LABELS))
        assert len(self._val_labels(val_loader)) == 40

    def test_without_val_split_there_is_no_validation_loader(self):
        labels = np.array([0] * 10 + [1] * 10)
        clf = self._clf(val_split=None)
        train_loader, val_loader = clf._loaders_from_dataset(self._dataset(labels),
                                                             labels=labels)
        assert val_loader is None
        assert len(torch.cat([y for _, y in train_loader])) == 20


class TestEstimatorParameters:
    """Both knobs are ordinary sklearn parameters: cloned, saved and reloaded."""

    def test_defaults_are_off_and_on(self):
        clf = TorchClassifier(nn.Identity())
        assert clf.class_weight is None      # opt-in
        assert clf.val_split_stratify is True     # opt-out

    def test_get_params_keeps_every_engine_parameter(self):
        """The explicit __init__ must not hide the inherited hyper-parameters."""
        params = TorchClassifier(nn.Identity()).get_params(deep=False)
        assert set(params) == {"backbone", "lr", "max_epochs", "batch_size",
                               "val_split", "val_split_stratify", "patience",
                               "device", "verbose", "class_weight",
                               "random_state"}

    def test_clone_preserves_them(self):
        clf = TorchClassifier(nn.Identity(), class_weight="balanced",
                              val_split_stratify=False)
        cloned = clone(clf)
        assert cloned.class_weight == "balanced"
        assert cloned.val_split_stratify is False

    def test_they_survive_save_and_load(self, tmp_path):
        clf = TorchClassifier(_backbone(), class_weight={0: 1.0, 1: 4.0},
                              val_split_stratify=False, max_epochs=1,
                              device="cpu")
        path = tmp_path / "clf.pkl"
        clf.save(str(path))
        reloaded = TorchClassifier.load(str(path))
        assert reloaded.class_weight == {0: 1.0, 1: 4.0}
        assert reloaded.val_split_stratify is False


class TestFitEndToEnd:
    """A short real fit, to prove the two knobs work together on an unbalanced set."""

    def test_balanced_weighting_trains_and_predicts_both_classes(self):
        X, y = _unbalanced()
        clf = TorchClassifier(_backbone(), class_weight="balanced", max_epochs=2,
                              batch_size=64, val_split=0.2, device="cpu", verbose=0)
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        assert proba.shape == (len(X), 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)
        assert list(clf.classes_) == [0, 1]

    def test_an_unweighted_fit_still_works(self):
        X, y = _unbalanced()
        clf = TorchClassifier(_backbone(), max_epochs=2, batch_size=64,
                              val_split=0.2, device="cpu", verbose=0)
        clf.fit(X, y)
        assert clf.predict(X).shape == (len(X),)
