"""Tests for the group-aware validation split (``fit(X, y, groups=...)``).

When observations are not independent -- epochs cut from overlapping windows of one trial
-- a random validation split puts near-copies of the training data into the fold, and the
validation loss that early stopping watches stops measuring generalisation. ``groups`` makes
the estimator hold out whole groups instead. These tests pin that no group ever sits on both
sides, the fallbacks, and that the argument reaches the split through ``fit``.

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
from torch.utils.data import TensorDataset

from medusa.ml.torch_models._engine import _group_split_indices
from medusa.ml.torch_models.backbones.eeg_inception_v2 import EEGInceptionV2
from medusa.ml.torch_models.classification import TorchClassifier

N_SAMPLES, N_CHA = 64, 8


def _groups(n_groups, size):
    """``n_groups`` groups of ``size`` consecutive observations each."""
    return np.repeat(np.arange(n_groups), size)


class TestGroupSplitIndices:

    def test_holds_out_whole_groups(self):
        groups = _groups(10, 20)
        train_idx, val_idx = _group_split_indices(200, 0.2, groups, random_state=0)
        assert set(train_idx) | set(val_idx) == set(range(200))
        assert not set(train_idx) & set(val_idx)
        assert not set(groups[train_idx]) & set(groups[val_idx])      # the whole point
        assert len(set(groups[val_idx])) == 2                          # int(10 * 0.2)

    def test_at_least_one_group_and_never_all(self):
        groups = _groups(3, 5)
        _, val_idx = _group_split_indices(15, 0.05, groups, random_state=0)
        assert len(set(groups[val_idx])) == 1
        train_idx, val_idx = _group_split_indices(15, 0.99, groups, random_state=0)
        assert len(set(groups[val_idx])) == 2 and len(set(groups[train_idx])) == 1

    def test_a_single_group_cannot_be_split(self):
        assert _group_split_indices(20, 0.2, np.zeros(20, int)) is None

    def test_a_length_mismatch_is_an_error(self):
        with pytest.raises(ValueError, match="groups has 7 entries"):
            _group_split_indices(20, 0.2, np.zeros(7, int))

    def test_the_seed_fixes_the_held_out_groups(self):
        groups = _groups(10, 4)
        a = _group_split_indices(40, 0.2, groups, random_state=3)
        b = _group_split_indices(40, 0.2, groups, random_state=3)
        assert a == b


class TestSplitLoadersWithGroups:
    """The estimator's own splitter, which is what ``fit`` calls."""

    @staticmethod
    def _dataset(n):
        # the observation index is stored as the "feature", so folds can be read back
        return TensorDataset(torch.arange(n).unsqueeze(1), torch.zeros(n, dtype=torch.long))

    @staticmethod
    def _clf(**kwargs):
        return TorchClassifier(nn.Identity(), batch_size=16, **kwargs)

    @staticmethod
    def _indices(loader):
        return torch.cat([x for x, _ in loader]).squeeze(1).numpy()

    def test_groups_are_disjoint_across_the_folds(self):
        groups = _groups(8, 25)
        train_loader, val_loader = self._clf(val_split=0.25)._loaders_from_dataset(
            self._dataset(200), groups=groups)
        train_groups = set(groups[self._indices(train_loader)])
        val_groups = set(groups[self._indices(val_loader)])
        assert not train_groups & val_groups
        assert len(val_groups) == 2 and len(train_groups) == 6

    def test_groups_take_precedence_over_stratification(self):
        """Labels are passed too (``fit`` always does), but the split is by group."""
        groups = _groups(5, 20)
        labels = np.tile([0] * 19 + [1], 5)         # 5 % targets: would be stratified
        clf = self._clf(val_split=0.2)
        with warnings.catch_warnings():
            warnings.simplefilter("error")          # and there is nothing to warn about
            train_loader, val_loader = clf._loaders_from_dataset(
                self._dataset(100), labels=labels, groups=groups)
        assert not set(groups[self._indices(train_loader)]) & set(
            groups[self._indices(val_loader)])

    def test_a_single_group_warns_and_falls_back(self):
        clf = self._clf(val_split=0.2)
        with pytest.warns(UserWarning, match="single group"):
            _, val_loader = clf._loaders_from_dataset(self._dataset(50),
                                                      groups=np.zeros(50, int))
        assert len(self._indices(val_loader)) == 10      # the observation-level split

    def test_a_length_mismatch_is_an_error(self):
        with pytest.raises(ValueError, match="groups has"):
            self._clf(val_split=0.2)._loaders_from_dataset(self._dataset(50),
                                                           groups=np.zeros(7, int))

    def test_without_val_split_groups_are_ignored(self):
        train_loader, val_loader = self._clf(val_split=None)._loaders_from_dataset(
            self._dataset(50), groups=_groups(5, 10))
        assert val_loader is None
        assert len(self._indices(train_loader)) == 50


def _backbone():
    # small temporal scales: they fit inside the 64-sample epoch, and keep these tests fast
    return EEGInceptionV2(input_samples=N_SAMPLES, n_cha=N_CHA,
                          temp_scales_samples=(15, 11, 7))


class TestFitWithGroups:

    def test_fit_accepts_groups(self):
        """A short real fit, with the epochs of every "trial" kept on one side."""
        rng = np.random.RandomState(0)
        X = rng.randn(120, N_SAMPLES, N_CHA).astype("float32")
        y = rng.randint(0, 2, size=120)
        clf = TorchClassifier(_backbone(), max_epochs=1, batch_size=32, val_split=0.2,
                              device="cpu", verbose=0)
        clf.fit(X, y, groups=_groups(6, 20))
        assert clf.predict(X).shape == (120,)

    def test_groups_must_match_x(self):
        X = np.zeros((10, N_SAMPLES, N_CHA), dtype="float32")
        clf = TorchClassifier(_backbone(), val_split=0.2, device="cpu", verbose=0)
        with pytest.raises(ValueError, match="groups has 3 entries"):
            clf.fit(X, np.arange(10) % 2, groups=np.zeros(3, int))
