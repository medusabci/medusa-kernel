"""Smoke tests for the torch estimators' levelled training output (:mod:`_progress`).

These guard the fiddly, Lightning-version-sensitive progress-bar overrides: the clean
one-line-per-epoch output must never regress into the multi-line ``Validation DataLoader``
spam, silent must stay silent, and ``history_`` must carry the per-epoch curves.

Skipped on the no-extras CI job: torch / Lightning / rich are optional (users install torch
to match their hardware).
"""
from __future__ import annotations

import contextlib
import io

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("lightning")
pytest.importorskip("rich")

from medusa.ml.torch_models._progress import normalize_verbose
from medusa.ml.torch_models.backbones.eeg_inception_v2 import EEGInceptionV2
from medusa.ml.torch_models.classification import TorchClassifier


def _train_capture(verbose):
    """Fit a tiny classifier at the given verbosity; return (estimator, captured output)."""
    rng = np.random.RandomState(0)
    X = rng.randn(256, 64, 8).astype("float32")
    y = rng.randint(0, 2, size=256)
    clf = TorchClassifier(EEGInceptionV2(input_samples=64, n_cha=8),
                          max_epochs=3, batch_size=64, val_split=0.2,
                          device="cpu", verbose=verbose)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        clf.fit(X, y)
    return clf, buf.getvalue()


def test_normalize_verbose_levels():
    assert [normalize_verbose(i) for i in (0, 1, 2)] == [0, 1, 2]
    assert normalize_verbose("silent") == 0
    assert normalize_verbose("epoch") == 1
    assert normalize_verbose("FULL") == 2       # case-insensitive
    # bool is deliberately rejected (use 0/1/2 or 'silent'/'epoch'/'full').
    for bad in (True, False, 3, -1, "loud", 1.5, None):
        with pytest.raises(ValueError):
            normalize_verbose(bad)


def test_level1_is_clean_and_informative():
    clf, out = _train_capture("epoch")
    # No validation/sanity dataloader spam, by construction.
    assert "Validation" not in out
    assert "Sanity" not in out
    # Banner + one line per epoch + summary.
    assert "Training TorchClassifier" in out
    assert out.count("Epoch ") >= 3
    assert "Done:" in out
    # Framework noise suppressed.
    assert "GPU available" not in out
    assert "LeafSpec" not in out


def test_level0_is_silent_but_records_history():
    clf, out = _train_capture("silent")
    assert out.strip() == ""
    # history_ is still populated (EpochHistory runs at every level).
    assert len(clf.history_[-1]["train_loss_curve"]) == clf.history_[-1]["epochs"]


def test_history_is_one_entry_per_fit():
    """``history_`` is a list, so a multi-phase run keeps every phase's curves."""
    clf, _ = _train_capture(0)
    assert len(clf.history_) == 1

    rng = np.random.RandomState(1)
    clf.fit(rng.randn(64, 64, 8).astype("float32"), np.array([0, 1] * 32))
    assert len(clf.history_) == 2


def test_history_has_curves_and_best():
    clf, _ = _train_capture(0)
    h = clf.history_[-1]
    for key in ("monitor", "best_score", "best_epoch",
                "train_loss_curve", "val_loss_curve"):
        assert key in h
    assert h["monitor"] == "val_loss"
    assert len(h["train_loss_curve"]) == h["epochs"]
    assert len(h["val_loss_curve"]) == h["epochs"]
    assert 0 <= h["best_epoch"] < h["epochs"]
