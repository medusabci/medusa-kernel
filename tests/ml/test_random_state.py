"""``random_state``: the seed that makes a torch fit repeatable.

Without it, every fit draws its own validation split, batch order and dropout masks, so two
runs of the same experiment end with different models -- noise that blurs a comparison
between two configurations. With it, the whole fit is fixed. And because the seeding is
forked rather than global, a reproducible fit leaves the random numbers the calling script
draws afterwards exactly as they were.

Every test pins ``device='cpu'``: CUDA kernels are not deterministic by default, so on a GPU
two seeded runs are close but not bit-identical (as the estimator's docstring says).

Skipped on the no-extras CI job: torch / Lightning are optional.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("lightning")

import torch
import torch.nn as nn
from sklearn.base import clone

from medusa.ml.torch_models._engine import _stratified_split_indices, seeded_rng
from medusa.ml.torch_models.classification import TorchClassifier

N_SAMPLES, N_CHA = 16, 4     # tiny on purpose: these tests are about repeatability


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
        """No constructor arguments; needed to build the persistable bundle."""
        return {}


def _data(n=64, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, N_SAMPLES, N_CHA).astype("float32")
    y = (rng.random_sample(n) < 0.5).astype(int)
    y[:2] = [0, 1]                       # both classes are always present
    return X, y


def _fit(random_state, backbone_seed=0):
    """A fitted classifier whose backbone was built under ``backbone_seed``.

    The backbone's weights are drawn when the module is built, outside anything the
    estimator controls, so they are fixed here -- otherwise ``random_state`` alone could
    never make two runs agree.
    """
    with seeded_rng(backbone_seed):
        backbone = _Linear()
    clf = TorchClassifier(backbone, max_epochs=3, batch_size=16, val_split=0.25,
                          patience=2, device="cpu", verbose=0,
                          random_state=random_state)
    X, y = _data()
    return clf.fit(X, y)


def _proba(random_state, backbone_seed=0):
    X, _ = _data()
    return _fit(random_state, backbone_seed).predict_proba(X)


class TestSeededSplit:
    """The validation fold itself, which is what moves between repetitions today."""

    LABELS = np.array([0] * 90 + [1] * 10)

    def test_the_same_seed_gives_the_same_fold(self):
        first = _stratified_split_indices(100, 20, self.LABELS, random_state=7)
        second = _stratified_split_indices(100, 20, self.LABELS, random_state=7)
        assert first == second

    def test_another_seed_gives_another_fold(self):
        first = _stratified_split_indices(100, 20, self.LABELS, random_state=7)
        second = _stratified_split_indices(100, 20, self.LABELS, random_state=8)
        assert first != second

    def test_the_fold_is_still_stratified(self):
        _, val_idx = _stratified_split_indices(100, 20, self.LABELS, random_state=7)
        assert self.LABELS[val_idx].sum() == 2          # 10 % of 20, the class's share

    def test_the_estimator_passes_its_seed_down(self):
        """Two estimators with the same seed must split the same data the same way."""
        from torch.utils.data import TensorDataset

        dataset = TensorDataset(torch.zeros(len(self.LABELS), 2),
                                torch.as_tensor(self.LABELS))
        folds = []
        for _ in range(2):
            clf = TorchClassifier(_Linear(), batch_size=8, val_split=0.2, random_state=3)
            _, val_loader = clf._loaders_from_dataset(dataset, labels=self.LABELS)
            folds.append(torch.cat([y for _, y in val_loader]).numpy())
        np.testing.assert_array_equal(folds[0], folds[1])


class TestRepeatableFit:
    """A whole fit, end to end: same seed in, same model out."""

    def test_the_same_seed_gives_the_same_model(self):
        np.testing.assert_allclose(_proba(0), _proba(0), rtol=1e-6, atol=1e-6)

    def test_another_seed_gives_another_model(self):
        assert not np.allclose(_proba(0), _proba(1), rtol=1e-3, atol=1e-3)

    def test_without_a_seed_two_runs_differ(self):
        """The behaviour ``random_state`` exists to switch off; it stays the default."""
        assert not np.allclose(_proba(None), _proba(None), rtol=1e-3, atol=1e-3)

    def test_the_backbone_weights_still_have_to_be_seeded(self):
        """The seed cannot reach weights drawn before the estimator existed.

        This is why the deep pipelines wrap ``build_backbone`` in ``seeded_rng`` as well.
        """
        assert not np.allclose(_proba(0, backbone_seed=0), _proba(0, backbone_seed=1),
                               rtol=1e-3, atol=1e-3)


class TestNoGlobalSideEffect:
    """A reproducible fit must not reseed the script that called it."""

    def test_fitting_leaves_the_callers_random_numbers_alone(self):
        torch.manual_seed(123)
        expected = torch.randn(4)

        torch.manual_seed(123)
        _fit(random_state=0)
        assert torch.equal(torch.randn(4), expected)

    def test_an_unseeded_fit_is_left_alone_too(self):
        """``random_state=None`` must not fork or seed anything -- it is the default path."""
        torch.manual_seed(123)
        _fit(random_state=None)
        drawn = torch.randn(4)

        torch.manual_seed(123)
        assert not torch.equal(torch.randn(4), drawn)   # the stream really was consumed


class TestSklearnPlumbing:
    """``random_state`` is an ordinary estimator parameter, so it travels for free."""

    def test_it_is_a_get_params_entry(self):
        assert TorchClassifier(_Linear(), random_state=5).get_params()["random_state"] == 5

    def test_clone_keeps_it(self):
        assert clone(TorchClassifier(_Linear(), random_state=5)).random_state == 5

    def test_the_saved_bundle_keeps_it(self):
        assert TorchClassifier(
            _Linear(), random_state=5).to_pickleable_obj()["params"]["random_state"] == 5
