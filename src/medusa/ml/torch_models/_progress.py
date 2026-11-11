"""Levelled, rich-rendered training output for the torch estimators.

Everything here shapes what a user *sees* during ``fit`` -- one line per epoch, a short
banner and summary, and suppression of Lightning's noisier defaults. Nothing here changes
what the model learns; it is pure presentation, kept out of the training engine
(:mod:`._engine`) so the engine stays about *running* the fit.

Three verbosity levels (see :func:`normalize_verbose`):

* **0 -- silent**: no banner, no epoch lines, no summary; Lightning's chatter is suppressed.
* **1 -- epoch (default)**: one :class:`EpochPrinter` line per epoch (elapsed time, losses
  and validation scores, where early stopping stands), a banner before and a summary after;
  Lightning's progress bar, info logs and ``num_workers`` warning are off.
* **2 -- full/debug**: Lightning's stock progress bar, model summary and all its output,
  kept intentionally verbose for debugging.

Level 1 has no progress bar. Each line is printed once, when its epoch ends, so the output
is the same on a terminal, in an IDE console and in a log file.
"""

from __future__ import annotations

import contextlib
import logging
import time
import warnings

import lightning.pytorch as pl
from rich.console import Console

__all__ = [
    "normalize_verbose",
    "EpochPrinter",
    "EpochHistory",
    "quiet_lightning",
    "print_banner",
    "print_summary",
]

#: The three verbosity levels, keyed by their canonical string name.
_LEVELS = {"silent": 0, "epoch": 1, "full": 2}

#: Shorter titles for the long metric names in the printed lines. ``history_`` keeps the
#: full names; a metric that is not here is printed under its own name.
_SHORT_TITLES = {
    "val_accuracy": "val_acc",
    "val_balanced_accuracy": "val_bal_acc",
}


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


def _print(console: Console, text: str) -> None:
    """Print one line of ``rich`` markup whole.

    ``soft_wrap`` stops ``rich`` from wrapping or cutting the line at the console width,
    which it takes to be 80 columns when the output is not a terminal.
    """
    console.print(text, soft_wrap=True, highlight=False)


def _format_seconds(seconds: float) -> str:
    """``0.8s`` under a minute, ``1m05s`` under an hour, ``1h02m`` beyond."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"
    return f"{int(seconds // 3600)}h{int(seconds % 3600 // 60):02d}m"


def _format_value(value: float) -> str:
    """A loss or a score with four decimals; scientific notation where those would hide it."""
    if value == 0 or 1e-3 <= abs(value) < 1e4:
        return f"{value:.4f}"
    return f"{value:.2e}"                            # also prints 'nan' and 'inf' as they are


def _format_scores(scores: dict) -> list[str]:
    """``rich`` markup of each ``{metric: value}`` entry: a green title and its value."""
    return [f"[green]{_SHORT_TITLES.get(name, name)}[/] {_format_value(value)}"
            for name, value in scores.items()]


class EpochPrinter(pl.Callback):
    """Print one line when each training epoch ends.

    The line holds the time the epoch took (training plus validation), every metric the
    task logs with ``prog_bar=True``, and where early stopping stands::

        Epoch  6/12   0.8s  train_loss 0.0647  val_loss 0.0506  val_bal_acc 0.9806  best
        Epoch  7/12   0.8s  train_loss 0.0541  val_loss 0.0526  val_bal_acc 0.9806  wait 1/3

    ``best`` marks a new minimum of the monitored metric: the weights that are restored
    after the fit. ``wait k/patience`` counts the epochs that did not improve it by more
    than ``min_delta``; the fit stops when ``k`` reaches ``patience``.

    Parameters
    ----------
    console : rich.console.Console
        Where the lines are written.
    monitor : str
        The metric early stopping watches, lower being better (``'val_loss'``).
    patience : int
        Epochs without improvement that early stopping tolerates.
    min_delta : float
        Smallest drop of ``monitor`` that early stopping counts as an improvement.
    """

    def __init__(self, console: Console, *, monitor: str, patience: int,
                 min_delta: float) -> None:
        self.console = console
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.epoch_start_s = 0.0
        self.best_score = float("inf")               # lowest monitored value so far
        self.patience_score = float("inf")           # the value patience is counted from
        self.wait_count = 0

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        self.epoch_start_s = time.perf_counter()

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        # Validation (if any) has already run by now, so val_loss is populated.
        epoch_seconds = time.perf_counter() - self.epoch_start_s
        metrics = dict(trainer.progress_bar_metrics)

        # train_loss first, then val_loss, then the rest in the order the task logged them
        names = [name for name in ("train_loss", "val_loss") if name in metrics]
        names += [name for name in metrics if name not in names]
        scores = {name: metrics[name] for name in names}

        # Lightning runs EarlyStopping and ModelCheckpoint after this callback, so neither
        # has seen this epoch yet. Their two rules are applied here: any new minimum is the
        # new best checkpoint, and only a drop larger than min_delta resets the patience.
        score = metrics[self.monitor]
        status = []
        if score < self.best_score:
            self.best_score = score
            status.append("[bold green]best[/]")
        if score < self.patience_score - self.min_delta:
            self.patience_score = score
            self.wait_count = 0
        else:
            self.wait_count += 1
            status.append(f"[yellow]wait {self.wait_count}/{self.patience}[/]")

        # The epoch number is padded so that the lines of a fit stay aligned
        n_epochs = str(trainer.max_epochs)
        epoch = str(trainer.current_epoch + 1).rjust(len(n_epochs))
        label = (f"[bold cyan]Epoch {epoch}/{n_epochs}[/] "
                 f"[cyan]{_format_seconds(epoch_seconds).rjust(6)}[/]")
        _print(self.console, "  ".join([label] + _format_scores(scores) + status))


class EpochHistory(pl.Callback):
    """Record per-epoch train/val loss (and the best) for ``history_``; attached at every level.

    Decoupled from presentation so ``history_`` is uniform even in silent or headless runs (a
    tutorial can plot the loss curve after ``fit`` with no logger and no disk artifact).
    ``metric_names`` are the validation metrics the task logs as ``'val_<name>'``; each gets
    its own curve in ``metric_curves``.
    """

    def __init__(self, metric_names: "list[str] | None" = None) -> None:
        self.train_curve: list[float] = []
        self.val_curve: list[float] = []
        self.metric_curves: dict[str, list[float]] = {
            name: [] for name in (metric_names or [])}
        self.best_score: float = float("inf")
        self.best_epoch: int = -1

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        m = trainer.callback_metrics
        tl, vl = m.get("train_loss"), m.get("val_loss")
        self.train_curve.append(float(tl) if tl is not None else float("nan"))
        self.val_curve.append(float(vl) if vl is not None else float("nan"))
        for name, curve in self.metric_curves.items():
            value = m.get(f"val_{name}")
            curve.append(float(value) if value is not None else float("nan"))
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
                 lr: float, monitor: str, patience: int, min_delta: float = 0.0,
                 phase: int = 1, continuing: bool = False) -> None:
    """Short banner before training: what is being trained, on what data, and how.

    ``min_delta`` is only printed when it is set.

    ``continuing`` marks a fit that starts from a model this estimator has already
    trained, rather than a fresh one. It is printed because it is otherwise invisible:
    reusing one estimator (or pipeline) across a loop keeps adding training to the same
    model, which quietly invalidates a comparison between configurations.
    """
    _print(console,
           f"[bold]Training {estimator}[/] on [cyan]{device}[/] | "
           f"[magenta]{n_params:,}[/] trainable params")
    if continuing:
        _print(console,
               f"  [yellow]phase {phase}[/] -- continuing an already-trained model "
               f"(restart() to start over)")
    if n_val is None:
        data = f"[magenta]{n_train:,}[/] train observations, no validation set"
    else:
        data = f"[magenta]{n_train:,}[/] train + [magenta]{n_val:,}[/] val observations"
    _print(console, f"  {data} | batch size {batch_size}")
    tolerance = f", min_delta {min_delta:g}" if min_delta else ""
    _print(console,
           f"  up to {max_epochs} epochs | lr {lr:g} | early stop on "
           f"[green]{monitor}[/] (patience {patience}{tolerance})")


def print_summary(console: Console, *, epochs: int, max_epochs: int, stopped_early: bool,
                  fit_seconds: float, best_epoch: int, best_scores: dict) -> None:
    """Two lines after training: how it ended, and the epoch whose weights were kept.

    ``best_scores`` are the ``{metric: value}`` of the best epoch (``best_epoch``, counted
    from 0). The second line is skipped when no epoch was the best (``best_epoch`` is -1).
    """
    took = _format_seconds(fit_seconds)
    if stopped_early:
        _print(console, f"[bold]Early stop[/] at epoch {epochs}/{max_epochs} after {took}")
    else:
        _print(console, f"[bold]Finished[/] {epochs}/{max_epochs} epochs in {took}")
    if best_epoch >= 0:
        _print(console,
               f"[bold green]Best epoch {best_epoch + 1}[/] (weights restored): "
               + " | ".join(_format_scores(best_scores)))
