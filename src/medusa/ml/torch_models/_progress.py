"""Levelled, rich-rendered training output for the torch estimators.

Everything here shapes what a user *sees* during ``fit`` -- a single tidy line per epoch, a
one-line banner and summary, and suppression of Lightning's noisier defaults. Nothing here
changes what the model learns; it is pure presentation, kept out of the training engine
(:mod:`._engine`) so the engine stays about *running* the fit.

Three verbosity levels (see :func:`normalize_verbose`):

* **0 -- silent**: no bar, no banner, no summary; Lightning's chatter is suppressed.
* **1 -- epoch (default)**: one clean :class:`MedusaProgressBar` line per epoch (live
  progress + losses), a banner before and a summary after; Lightning's info logs and the
  ``num_workers`` warning are silenced.
* **2 -- full/debug**: no custom bar -- Lightning's stock progress bar, model summary and all
  its output, kept intentionally verbose for debugging.

The one-clean-line-per-epoch behaviour comes from subclassing the **base**
:class:`~lightning.pytorch.callbacks.ProgressBar` and overriding **only the train hooks**, so
Lightning never creates a validation or sanity progress bar -- that is what removes the
multi-line ``Validation DataLoader`` spam by construction, rather than trying to hide it.
"""

from __future__ import annotations

import contextlib
import logging
import warnings

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ProgressBar
from rich.console import Console
from rich.progress import (
    Progress, BarColumn, TextColumn, TaskProgressColumn, MofNCompleteColumn,
    TimeElapsedColumn)

__all__ = [
    "normalize_verbose",
    "MedusaProgressBar",
    "EpochHistory",
    "quiet_lightning",
    "print_banner",
    "print_summary",
]

#: The three verbosity levels, keyed by their canonical string name.
_LEVELS = {"silent": 0, "epoch": 1, "full": 2}


def normalize_verbose(v) -> int:
    """Map a verbosity ``0``/``1``/``2`` or ``'silent'``/``'epoch'``/``'full'`` to an ``int`` level.

    Accepts an ``int`` (0, 1 or 2) or a case-insensitive level name (:data:`_LEVELS`).
    ``bool`` is **rejected** on purpose -- pass ``0``/``1`` or the names, not ``True``/
    ``False``. Anything else raises :class:`ValueError` naming the valid set.
    """
    if isinstance(v, bool):
        raise ValueError(
            f"verbose does not accept bool; use 0/1/2 or {sorted(_LEVELS)}, got {v!r}.")
    if isinstance(v, int):
        if v in (0, 1, 2):
            return v
        raise ValueError(f"verbose int must be 0, 1 or 2, got {v}.")
    if isinstance(v, str):
        key = v.strip().lower()
        if key in _LEVELS:
            return _LEVELS[key]
        raise ValueError(
            f"verbose {v!r} not recognised; use 0/1/2 or one of {sorted(_LEVELS)}.")
    raise ValueError(
        f"verbose must be int (0-2) or str {sorted(_LEVELS)}, got {type(v).__name__}.")


class MedusaProgressBar(ProgressBar):
    """One tidy ``rich`` line per training epoch; no validation/sanity bars by construction.

    Subclasses the *base* :class:`~lightning.pytorch.callbacks.ProgressBar` and overrides
    **only** the train hooks, so Lightning never spins up a validation or sanity progress bar.
    Each epoch shows a live bar (``Epoch k/N`` + progress + per-epoch elapsed) that finalises
    with that epoch's losses and stays on its own line (Keras-style history). ``rich`` renders
    it: coloured and animated on a real terminal, one plain finalised line per epoch when the
    output is piped to a file or notebook.
    """

    def __init__(self, console: "Console | None" = None) -> None:
        super().__init__()
        self._enabled = True
        self.console = console or Console()
        self._prog: "Progress | None" = None
        self._task = None

    # Lightning calls enable()/disable() (e.g. to mute non-zero ranks under DDP).
    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    def _active(self, trainer) -> bool:
        return self._enabled and trainer.is_global_zero

    @staticmethod
    def _format_metrics(metrics: dict) -> str:
        """``rich`` markup for the epoch's metrics: ``train_loss`` first, then ``val_loss``."""
        order = {"train_loss": 0, "val_loss": 1}
        items = [(k, v) for k, v in metrics.items() if k != "v_num"]
        items.sort(key=lambda kv: order.get(kv[0], 2))
        return "  ".join(f"[green]{k}[/] {float(v):.3f}" for k, v in items)

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        if not self._active(trainer):
            return
        self._prog = Progress(
            TextColumn("[bold cyan]Epoch {task.fields[ep]}/{task.fields[tot]}[/]"),
            BarColumn(bar_width=24),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TextColumn("{task.fields[metrics]}"),
            console=self.console, transient=False)   # transient=False -> leave the line
        total = self.total_train_batches
        total = None if total == float("inf") else total
        self._prog.start()
        self._task = self._prog.add_task(
            "", total=total, ep=trainer.current_epoch + 1,
            tot=trainer.max_epochs, metrics="")

    def on_train_batch_end(self, trainer, pl_module, *args) -> None:
        if self._prog is not None and self._active(trainer):
            self._prog.update(self._task, advance=1)

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        # Validation (if any) has already run by now, so val_loss is populated.
        if self._prog is None or not self._active(trainer):
            return
        metrics = self._format_metrics(self.get_metrics(trainer, pl_module))
        self._prog.update(self._task, metrics=metrics)
        self._prog.stop()                            # commit the finished line
        self._prog = self._task = None


class EpochHistory(pl.Callback):
    """Record per-epoch train/val loss (and the best) for ``history_``; attached at every level.

    Decoupled from presentation so ``history_`` is uniform even in silent or headless runs (a
    tutorial can plot the loss curve after ``fit`` with no logger and no disk artifact).
    """

    def __init__(self) -> None:
        self.train_curve: list[float] = []
        self.val_curve: list[float] = []
        self.best_score: float = float("inf")
        self.best_epoch: int = -1

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        m = trainer.callback_metrics
        tl, vl = m.get("train_loss"), m.get("val_loss")
        self.train_curve.append(float(tl) if tl is not None else float("nan"))
        self.val_curve.append(float(vl) if vl is not None else float("nan"))
        monitored = vl if vl is not None else tl
        if monitored is not None and float(monitored) < self.best_score:
            self.best_score = float(monitored)
            self.best_epoch = int(trainer.current_epoch)


@contextlib.contextmanager
def quiet_lightning(enabled: bool):
    """Silence Lightning's INFO chatter and the ``num_workers`` warning while training.

    Raises the ``lightning.pytorch`` / ``lightning.fabric`` loggers to ``WARNING`` **at
    runtime** (Lightning resets them to ``INFO`` at import, so a pre-import ``setLevel`` is
    silently overwritten) and filters the dataloader ``num_workers`` / ``GPU available but not
    used`` :class:`PossibleUserWarning`\\ s. Everything is restored on exit, so no global
    logging state leaks; genuine warnings and errors still surface. A no-op when ``enabled`` is
    ``False`` (level 2 keeps the full output).
    """
    if not enabled:
        yield
        return
    loggers = [logging.getLogger(n) for n in ("lightning.pytorch", "lightning.fabric")]
    prev = [lg.level for lg in loggers]
    for lg in loggers:
        lg.setLevel(logging.WARNING)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*does not have many workers.*")
            warnings.filterwarnings("ignore", message=".*GPU available but not used.*")
            warnings.filterwarnings("ignore", message=".*LeafSpec.*")
            # Framework-internal churn the user cannot act on (e.g. Lightning's own
            # torch-pytree LeafSpec deprecation): silence deprecation/future warnings
            # raised from within lightning/* while training.
            for cat in (DeprecationWarning, FutureWarning, UserWarning):
                warnings.filterwarnings("ignore", category=cat, module=r"lightning\..*")
            yield
    finally:
        for lg, lv in zip(loggers, prev):
            lg.setLevel(lv)


def print_banner(console: Console, *, estimator: str, device, n_params: int,
                 n_train: int, n_val: "int | None", max_epochs: int, batch_size: int,
                 monitor: str, patience: int, phase: int = 1,
                 continuing: bool = False) -> None:
    """One-line-ish banner before training: what is being trained, on what data, and how.

    ``continuing`` marks a fit that starts from a model this estimator has already
    trained, rather than a fresh one. It is printed because it is otherwise invisible:
    reusing one estimator (or pipeline) across a loop keeps adding training to the same
    model, which quietly invalidates a comparison between configurations.
    """
    console.print(
        f"[bold]Training {estimator}[/] on [cyan]{device}[/] "
        f"([magenta]{n_params:,}[/] trainable params)")
    if continuing:
        console.print(
            f"  [yellow]phase {phase}[/] -- continuing an already-trained model "
            f"(restart() to start over)")
    data = (f"[magenta]{n_train:,}[/] training observations, "
            f"[magenta]{n_val:,}[/] for validation" if n_val is not None
            else f"[magenta]{n_train:,}[/] training observations (no validation set)")
    console.print(f"  {data}")
    console.print(
        f"  up to {max_epochs} epochs, batch size {batch_size}, "
        f"early stop on [green]{monitor}[/] (patience {patience})")


def print_summary(console: Console, *, epochs: int, stopped_early: bool, monitor: str,
                  best_score: float, best_epoch: int) -> None:
    """One-line summary after training: epochs run, early stop, and the best score."""
    tail = " (stopped early)" if stopped_early else ""
    best = (f", best [green]{monitor}[/] {best_score:.4f} @ epoch {best_epoch + 1}"
            if best_epoch >= 0 else "")
    console.print(f"[bold green]Done:[/] {epochs} epochs{tail}{best}")
