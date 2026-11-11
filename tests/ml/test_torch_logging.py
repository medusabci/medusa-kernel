"""Smoke tests for the torch estimators' levelled training output (:mod:`_progress`).

These guard the fiddly, Lightning-version-sensitive progress-bar overrides: the clean
one-line-per-epoch output must never regress into the multi-line ``Validation DataLoader``
spam, a line must never be cut to the console width, silent must stay silent, and
``history_`` must carry the per-epoch curves.

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

from medusa.ml.torch_models._progress import (
    _format_seconds, _format_value, normalize_verbose)
from medusa.ml.torch_models.backbones.eeg_inception_v2 import EEGInceptionV2
from medusa.ml.torch_models.classification import (
    TorchClassifier, TorchMultiTaskClassifier)


def _data(n=256, seed=0):
    rng = np.random.RandomState(seed)
    return rng.randn(n, 64, 8).astype("float32"), rng.randint(0, 2, size=n)


def _train_capture(verbose, **settings):
    """Fit a tiny classifier at the given verbosity; return (estimator, captured output).

    ``settings`` override the classifier's defaults here (3 epochs, a 20 % validation fold).
    The output is captured from a stream that is not a terminal, as a log file is.
    """
    kwargs = dict(max_epochs=3, batch_size=64, val_split=0.2, device="cpu",
                  verbose=verbose)
    kwargs.update(settings)
    clf = TorchClassifier(EEGInceptionV2(input_samples=64, n_cha=8), **kwargs)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        clf.fit(*_data())
    return clf, buf.getvalue()


def _epoch_lines(out):
    return [line for line in out.splitlines() if line.startswith("Epoch ")]


def test_normalize_verbose_levels():
    assert [normalize_verbose(i) for i in (0, 1, 2)] == [0, 1, 2]
    assert normalize_verbose("silent") == 0
    assert normalize_verbose("epoch") == 1
    assert normalize_verbose("FULL") == 2       # case-insensitive
    # bool is deliberately rejected (use 0/1/2 or 'silent'/'epoch'/'full').
    for bad in (True, False, 3, -1, "loud", 1.5, None):
        with pytest.raises(ValueError):
            normalize_verbose(bad)


def test_formatters():
    assert _format_seconds(0.84) == "0.8s"
    assert _format_seconds(65) == "1m05s"
    assert _format_seconds(3725) == "1h02m"
    assert _format_value(0.45561) == "0.4556"
    assert _format_value(0.00012) == "1.20e-04"     # four decimals would show 0.0001
    assert _format_value(float("nan")) == "nan"


def test_level1_is_clean_and_informative():
    clf, out = _train_capture("epoch")
    # No validation/sanity dataloader spam, by construction.
    assert "Validation" not in out
    assert "Sanity" not in out
    # Banner + one line per epoch + summary, and no progress bar.
    assert "Training TorchClassifier" in out
    assert len(_epoch_lines(out)) == 3
    assert "100%" not in out
    assert "Finished 3/3 epochs in " in out
    assert "Best epoch " in out
    # Framework noise suppressed.
    assert "GPU available" not in out
    assert "LeafSpec" not in out
    # A stream that is not a terminal gets no colour or cursor codes.
    assert "\x1b" not in out


def test_banner_names_the_training_setup():
    _, out = _train_capture("epoch", lr=0.002)
    assert "205 train + 51 val observations | batch size 64" in out
    assert "up to 3 epochs | lr 0.002 | early stop on val_loss (patience 10)" in out

    _, out = _train_capture("epoch", val_split=None, min_delta=0.05)
    assert "256 train observations, no validation set" in out
    assert "early stop on train_loss (patience 10, min_delta 0.05)" in out


def test_epoch_line_is_never_cut():
    """The captured console is 80 columns wide, and the line is longer than that."""
    clf, out = _train_capture("epoch", epoch_metrics=["accuracy", "balanced_accuracy"])
    lines = _epoch_lines(out)
    assert len(lines) == clf.history_[-1]["epochs"]
    for line in lines:
        assert len(line) > 80
        assert "…" not in line
        for title in ("train_loss", "val_loss", "val_acc", "val_bal_acc"):
            assert f" {title} " in line


def test_epoch_line_reports_the_history_and_marks_the_best():
    clf, out = _train_capture("epoch", max_epochs=6)
    history = clf.history_[-1]
    val_loss = history["val_loss_curve"]
    lines = _epoch_lines(out)
    assert len(lines) == 6
    for epoch, line in enumerate(lines):
        assert line.startswith(f"Epoch {epoch + 1}/6 ")
        assert f"train_loss {history['train_loss_curve'][epoch]:.4f}" in line
        assert f"val_loss {val_loss[epoch]:.4f}" in line
        is_best = val_loss[epoch] < min(val_loss[:epoch], default=float("inf"))
        assert ("best" in line.split()) == is_best
    assert f"Best epoch {history['best_epoch'] + 1} (weights restored): val_loss " in out


def test_wait_counts_up_to_the_early_stop():
    """With a ``min_delta`` no epoch can beat, only the first one improves: the count on
    each line must reach ``patience`` on the very epoch Lightning stops the fit."""
    clf, out = _train_capture("epoch", max_epochs=10, patience=2, min_delta=10.0)
    history = clf.history_[-1]
    lines = _epoch_lines(out)
    assert history["stopped_early"]
    assert len(lines) == history["epochs"] == 3
    assert "wait" not in lines[0]
    assert lines[1].endswith("wait 1/2")
    assert lines[2].endswith("wait 2/2")
    assert "Early stop at epoch 3/10 after " in out


def test_without_validation_the_training_loss_is_monitored():
    _, out = _train_capture("epoch", val_split=None)
    lines = _epoch_lines(out)
    assert len(lines) == 3
    assert all("train_loss " in line and "val_loss" not in line for line in lines)
    assert "(weights restored): train_loss " in out


def test_multitask_fit_prints_the_same_log():
    X, y = _data()
    clf = TorchMultiTaskClassifier(EEGInceptionV2(input_samples=64, n_cha=8),
                                   max_epochs=2, batch_size=64, val_split=0.2,
                                   device="cpu", verbose="epoch")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        clf.fit(X, {"a": y, "b": 1 - y})
    out = buf.getvalue()
    assert "Training TorchMultiTaskClassifier" in out
    assert len(_epoch_lines(out)) == 2


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
