"""Private training engine shared by the torch estimators.

Everything here is an implementation detail of the public estimators in
:mod:`medusa.ml.torch_models.classification`: numpy/tensor conversion, device
resolution and the Lightning training run (``pl.Trainer`` + early stopping +
best-checkpoint restore). Nothing in this module is public API.
"""
from __future__ import annotations

import contextlib
import importlib
import tempfile
import warnings

from . import require_lightning

require_lightning()  # friendly error if torch / Lightning is missing

import numpy as np
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from rich.console import Console
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, random_split

from medusa.core.serialization import PickleableComponent
from ._progress import (
    normalize_verbose, MedusaProgressBar, EpochHistory, quiet_lightning,
    print_banner, print_summary)


def _resolve_device(device) -> torch.device:
    """Map ``'auto'``/``'cpu'``/``'cuda[:n]'``/``'mps'`` to a ``torch.device``."""
    if device in (None, 'auto'):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device)


def _trainer_target(dev: torch.device):
    """Return ``(accelerator, devices)`` for ``pl.Trainer`` from a device."""
    if dev.type == 'cuda':
        return 'gpu', ([dev.index] if dev.index is not None else 1)
    if dev.type == 'mps':
        return 'mps', 1
    return 'cpu', 1


@contextlib.contextmanager
def seeded_rng(seed, device=None):
    """Run the block with torch's RNG seeded, and put the RNG back afterwards.

    ``seed=None`` yields straight away: nothing is seeded and nothing is
    touched. With a seed, the CPU generator (and every CUDA generator, when the
    run is on GPU) is forked with :func:`torch.random.fork_rng`, seeded, and
    restored on exit. The fork is the point: seeding globally would make a
    reproducible ``fit`` silently reset the random numbers the calling script
    draws afterwards.

    Used by :meth:`_BaseTorchEstimator._rng_scope` around a whole ``fit``, and
    by the deep pipelines around the construction of the backbone, whose initial
    weights are drawn before any estimator exists.
    """
    if seed is None:
        yield
        return
    if _resolve_device(device).type == 'cuda':
        devices, device_type = range(torch.cuda.device_count()), 'cuda'
    else:
        devices, device_type = [], 'cpu'
    with torch.random.fork_rng(devices=devices, device_type=device_type):
        torch.manual_seed(int(seed))
        yield


def _input_layout(backbone) -> tuple:
    """Return ``backbone.input_layout`` or raise with the contract message."""
    try:
        return tuple(backbone.input_layout)
    except AttributeError:
        raise AttributeError(
            f"{type(backbone).__name__} must declare an 'input_layout' "
            "attribute (e.g. ('batch', 'feature', 'n_samples', 'n_channels')) "
            "— it is part of the backbone contract.") from None


def _prepare_X(X, input_layout: tuple) -> torch.Tensor:
    """Validate ``X`` against ``input_layout`` and convert it to a tensor.

    The array is used **exactly as given** — nothing is reshaped, inserted or
    reordered. ``X`` must already have one axis per name in the backbone's
    declared ``input_layout`` (same rank, ``'batch'`` first); the only work
    done here is casting to ``float32`` and returning a contiguous
    ``torch.Tensor``. A wrong rank raises :class:`ValueError` naming the
    expected layout, so the required shape is explicit to the caller.
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != len(input_layout):
        raise ValueError(
            f"X must have axes {input_layout} ({len(input_layout)}-D); got a "
            f"{X.ndim}-D array of shape {X.shape}. Reshape it to match the "
            "backbone's input_layout before fitting.")
    return torch.from_numpy(np.ascontiguousarray(X))


def _to_model_input(backbone, X) -> torch.Tensor:
    """User array → forward-ready tensor for ``backbone``.

    Delegates to the backbone's own ``prepare_X`` when it defines one — the
    backbone owns any architecture-specific input adaptation (feature axis,
    channel reordering, modality-specific layout). Backbones without one fall
    back to the strict generic path: validate the rank against
    ``input_layout`` and convert, with no reshaping.
    """
    prepare = getattr(backbone, 'prepare_X', None)
    if prepare is not None:
        return prepare(X)
    return _prepare_X(X, _input_layout(backbone))


def encode_labels(y):
    """Labels (categorical or one-hot) → ``(classes, int64 class indices)``.

    ``classes`` is the sorted array of unique labels (sklearn's ``classes_``
    convention); the returned indices index into it.
    """
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] > 1:  # one-hot
        y = y.argmax(axis=1)
    classes, y_idx = np.unique(y.reshape(-1), return_inverse=True)
    return classes, y_idx.astype(np.int64)


def _stratified_split_indices(n: int, n_val: int, labels, random_state=None):
    """``(train_idx, val_idx)`` keeping each class's share, or ``None``.

    ``None`` means the split cannot be stratified — fewer validation slots than
    classes, fewer training slots than classes, or a class with a single member
    — and the caller falls back to a random split. ``random_state`` fixes which
    observations land on each side, so the same data give the same split.
    """
    labels = np.asarray(labels).reshape(-1)
    if len(labels) != n:
        return None
    counts = np.unique(labels, return_counts=True)[1]
    n_classes = len(counts)
    if (n_classes < 2 or counts.min() < 2
            or n_val < n_classes or n - n_val < n_classes):
        return None
    train_idx, val_idx = train_test_split(
        np.arange(n), test_size=n_val, stratify=labels,
        random_state=random_state)
    return train_idx.tolist(), val_idx.tolist()


def _metric(trainer, name) -> float:
    value = trainer.callback_metrics.get(name)
    return float(value) if value is not None else float('nan')


def _n_observations(loader) -> int:
    """Number of observations behind ``loader``.

    A plain ``DataLoader`` exposes them through its dataset; a ``CombinedLoader``
    (multi-task training) wraps one loader per task, so its observations are the
    sum over the tasks.
    """
    flattened = getattr(loader, 'flattened', None)
    if flattened is not None:
        return sum(len(dl.dataset) for dl in flattened)
    return len(loader.dataset)


def _cpu_state_dict(module) -> dict:
    """``module.state_dict()`` with every tensor detached and on CPU.

    Persisting CPU tensors keeps the saved model portable across devices —
    a model fitted on CUDA reloads on a CPU-only machine and vice versa.
    """
    return {k: v.detach().cpu() for k, v in module.state_dict().items()}


def _import_class(module_name: str, qualname: str):
    """Import ``qualname`` from ``module_name`` (clear ImportError if missing)."""
    try:
        obj = importlib.import_module(module_name)
        for part in qualname.split('.'):
            obj = getattr(obj, part)
        return obj
    except (ImportError, AttributeError):
        raise ImportError(
            f"Could not import '{qualname}' from module '{module_name}'. The "
            "backbone and estimator classes must be importable to rebuild a "
            f"saved model; did you import '{module_name}' before loading?"
        ) from None


class _BaseTorchEstimator(BaseEstimator, PickleableComponent):
    """Shared hyperparameters + Lightning training run (private engine).

    A scikit-learn ``BaseEstimator`` (``get_params``/``clone``/
    ``cross_validate`` work) that also implements the
    :class:`~medusa.core.serialization.PickleableComponent` persistence
    contract: :meth:`to_pickleable_obj` / :meth:`from_pickleable_obj` are
    defined here, ``save`` is overridden below (tensor-efficient protocol) and
    ``load`` is inherited.

    ``fit`` **continues from the model the estimator already holds**: the backbone
    is held by reference and trained in place, and a second ``fit`` keeps training
    the same head as well. :meth:`reset_head` is the explicit cold start for the
    part the estimator owns; the backbone belongs to whoever built it.

    Subclasses build a task (``pl.LightningModule``) and its dataloaders in
    ``fit`` and hand them to :meth:`_run_training`; they own all predict
    semantics. A model is saved as a portable ``config + state_dict`` bundle —
    never a raw module pickle — so it reloads across devices and torch
    versions. Subclasses supply the fitted-state capture/restore hooks
    (:meth:`_capture_fitted_state` / :meth:`_restore_fitted_state`) and name
    their fitted sentinel attribute via ``_FITTED_ATTR``.

    Parameters
    ----------
    backbone : nn.Module
        Headless feature extractor (``backbone(x)`` → ``[N, F]``,
        ``backbone.backbone_features == F``) that declares ``input_layout``.
        Held by reference and trained in place; sklearn ``clone`` deep-copies
        it (so ``cross_validate`` refits a fresh copy per fold, starting from
        the backbone's current weights).
    lr, max_epochs, batch_size, patience : training hyperparameters.
    val_split : float or None
        Fraction in (0, 1) held out for validation-based early stopping and
        best-checkpoint restore. ``None`` monitors training loss instead.
    val_split_stratify : bool, default True
        Keep each class's share of the data in that validation split, instead of
        drawing it at random. It only applies when the estimator hands labels to
        the splitter (the classifiers do) and ``val_split`` is set; when a class
        is too small to appear on both sides, the split falls back to the random
        one and warns.
    device : {'auto', 'cpu', 'cuda', 'cuda:N', 'mps'}
        Resolved once at ``fit`` time and reused for inference.
    verbose : int | str, default 1
        Training-output verbosity level (see
        :func:`~medusa.ml.torch_models._progress.normalize_verbose`):
        ``0`` / ``'silent'`` -- no output; ``1`` / ``'epoch'`` (default) -- one clean line
        per epoch (live progress + losses) with a banner and summary; ``2`` / ``'full'`` --
        Lightning's full stock output (model summary, validation bars) for debugging.
    random_state : int or None, default None
        Seed that makes a fit repeatable. ``None`` (the default) leaves training
        stochastic: every call draws its own validation split, batch order and
        dropout masks, so two runs of the same experiment give slightly
        different models — noise that blurs a comparison between two
        configurations. An integer fixes all of them, so the same data and the
        same settings give the same model. The seed is applied inside
        :func:`torch.random.fork_rng`, so it never disturbs the random numbers
        the calling script draws after ``fit``.

        Two things stay outside it. The weights ``backbone`` already carries were
        drawn when that module was built, before the estimator existed, so seed
        that too (:func:`seeded_rng` around its construction, as the deep
        pipelines do). And CUDA kernels are not deterministic by default: on GPU
        two seeded runs come out very close but not bit-identical, unless you
        also set ``torch.use_deterministic_algorithms(True)``, which costs
        speed.
    """

    def __init__(self, backbone, *, lr=1e-3, max_epochs=100, batch_size=64,
                 val_split=None, val_split_stratify=True, patience=10,
                 device='auto', verbose=1, random_state=None):
        self.backbone = backbone
        self.lr = lr
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.val_split = val_split
        self.val_split_stratify = val_split_stratify
        self.patience = patience
        self.device = device
        self.verbose = verbose
        self.random_state = random_state

    # ---- model utilities ---- #

    def reset_head(self):
        """Drop the fitted head(s) and ``classes_``, keeping the trained backbone.

        The next ``fit`` builds a fresh head on the features learned so far — the
        transfer move: keep what the backbone knows, train a new classifier for a
        new subject or a new set of labels. Nothing happens if the estimator was
        never fitted.
        """
        for name in (self._FITTED_ATTR, 'classes_'):
            if name is not None and hasattr(self, name):
                delattr(self, name)
        return self

    def _prepare(self, X) -> torch.Tensor:
        """User array → forward-ready tensor (via the backbone's prepare_X)."""
        return _to_model_input(self.backbone, X)

    def _device(self) -> torch.device:
        """Inference device: the one fixed at ``fit`` time, else resolved now.

        Lets a freshly loaded model (whose transient ``device_`` was not
        persisted) pick a device on its first ``predict``/``encode``.
        """
        dev = getattr(self, 'device_', None)
        if dev is None:
            dev = self.device_ = _resolve_device(self.device)
        return dev

    def encode(self, X) -> np.ndarray:
        """Backbone features for ``X`` (works before or after ``fit``)."""
        dev = self._device()
        self.backbone.to(dev).eval()
        with torch.no_grad():
            return self.backbone(self._prepare(X).to(dev)).cpu().numpy()

    # ---- training ---- #

    def _rng_scope(self):
        """``random_state`` applied to one fit; a no-op when it is ``None``.

        Every ``fit`` runs its whole body inside this scope — building the task
        (the head's initial weights), splitting off the validation fold, and the
        training run itself — so a single seed covers the split, the batch order
        and the dropout masks.
        """
        return seeded_rng(self.random_state, self.device)

    def _loaders_from_dataset(self, dataset, labels=None):
        """Split one dataset into ``(train_loader, val_loader)`` per ``val_split``.

        With ``val_split_stratify`` on (the default) and ``labels`` given, the
        validation fold keeps each class's share of the data instead of being
        drawn at random. That matters as soon as the classes are unbalanced: a random
        split of, say, a 3 %-target set can leave the fold with almost no
        targets, and then the validation loss that early stopping watches says
        very little about the model. When the split cannot keep every class on
        both sides, it falls back to the random split and warns.
        """
        if not self.val_split:
            return DataLoader(dataset, batch_size=self.batch_size,
                              shuffle=True), None
        n_val = max(1, int(len(dataset) * self.val_split))
        split = (_stratified_split_indices(len(dataset), n_val, labels,
                                           random_state=self.random_state)
                 if self.val_split_stratify and labels is not None else None)
        if split is None:
            if self.val_split_stratify and labels is not None:
                warnings.warn(
                    f"cannot stratify a {n_val}-observation validation split of "
                    f"{len(dataset)} observations (a class is too small, or there "
                    f"are fewer slots than classes); splitting at random instead.",
                    UserWarning, stacklevel=3)
            train_ds, val_ds = random_split(
                dataset, [len(dataset) - n_val, n_val])
        else:
            train_idx, val_idx = split
            train_ds, val_ds = Subset(dataset, train_idx), Subset(dataset, val_idx)
        return (DataLoader(train_ds, batch_size=self.batch_size, shuffle=True),
                DataLoader(val_ds, batch_size=self.batch_size))

    def _run_training(self, task: pl.LightningModule, train_loader, val_loader):
        """Train ``task``; restore best weights, append to ``history_``.

        ``history_`` is a list with one entry per ``fit`` — a multi-phase run
        (pretrain, then finetune) keeps every phase's curves, and ``history_[-1]``
        is the phase that just ran.

        Output verbosity follows ``self.verbose`` (see
        :func:`~medusa.ml.torch_models._progress.normalize_verbose`): level 1 (default) shows
        one clean line per epoch via :class:`~medusa.ml.torch_models._progress.MedusaProgressBar`
        with a banner and summary, level 0 is silent, level 2 is Lightning's full stock output.
        Per-epoch loss curves are captured at every level by
        :class:`~medusa.ml.torch_models._progress.EpochHistory`.
        """
        self.device_ = _resolve_device(self.device)
        accelerator, devices = _trainer_target(self.device_)
        monitor = 'val_loss' if val_loader is not None else 'train_loss'
        level = normalize_verbose(self.verbose)

        early = EarlyStopping(monitor=monitor, mode='min', patience=self.patience)
        history = EpochHistory()
        console = Console() if level == 1 else None
        bar = MedusaProgressBar(console) if level == 1 else None
        with tempfile.TemporaryDirectory() as ckpt_dir:
            checkpoint = ModelCheckpoint(dirpath=ckpt_dir, monitor=monitor,
                                         mode='min', save_top_k=1)
            callbacks = [early, checkpoint, history] + ([bar] if bar is not None else [])
            with quiet_lightning(enabled=level <= 1):
                trainer = pl.Trainer(
                    max_epochs=self.max_epochs,
                    accelerator=accelerator,
                    devices=devices,
                    callbacks=callbacks,
                    logger=False,
                    enable_progress_bar=level >= 1,
                    enable_model_summary=level >= 2,
                    num_sanity_val_steps=0)
                if level == 1:
                    print_banner(
                        console,
                        estimator=type(self).__name__,
                        device=self.device_,
                        n_params=sum(p.numel() for p in task.parameters()
                                     if p.requires_grad),
                        n_train=_n_observations(train_loader),
                        n_val=(_n_observations(val_loader)
                               if val_loader is not None else None),
                        phase=len(getattr(self, 'history_', [])) + 1,
                        continuing=bool(getattr(self, 'history_', [])),
                        max_epochs=self.max_epochs,
                        batch_size=self.batch_size,
                        monitor=monitor,
                        patience=self.patience)
                trainer.fit(task, train_loader, val_loader)
            best_score = (float(checkpoint.best_model_score)
                          if checkpoint.best_model_score is not None else float('nan'))
            if checkpoint.best_model_path:  # restore best weights
                state = torch.load(
                    checkpoint.best_model_path,
                    map_location='cpu'
                )['state_dict']
                task.load_state_dict(state)

        self.backbone.to(self.device_)
        phase = {
            'epochs': int(trainer.current_epoch),
            'train_loss': _metric(trainer, 'train_loss'),
            'val_loss': (_metric(trainer, 'val_loss')
                         if val_loader is not None else float('nan')),
            'stopped_early': bool(early.stopped_epoch),
            'monitor': monitor,
            'best_score': best_score,
            'best_epoch': history.best_epoch,
            'train_loss_curve': history.train_curve,
            'val_loss_curve': history.val_curve,
        }
        self.history_ = getattr(self, 'history_', []) + [phase]
        if level == 1:
            print_summary(
                console,
                epochs=phase['epochs'],
                stopped_early=phase['stopped_early'],
                monitor=monitor,
                best_score=best_score,
                best_epoch=history.best_epoch)
        return task

    # ---- inference ---- #

    def _head_logits(self, X, head_module) -> np.ndarray:
        """``head_module(backbone(X))`` as numpy, on the inference device."""
        dev = self._device()
        self.backbone.to(dev).eval()
        head_module.to(dev).eval()
        with torch.no_grad():
            feats = self.backbone(self._prepare(X).to(dev))
            return head_module(feats).cpu().numpy()

    def _check_fitted(self):
        if not self._is_fitted():
            raise RuntimeError("estimator is not fitted; call fit() first.")

    # ---- persistence ---- #
    #: Name of the attribute holding the fitted head(s); set by subclasses.
    _FITTED_ATTR: str = None

    def _is_fitted(self) -> bool:
        return self._FITTED_ATTR is not None and hasattr(self, self._FITTED_ATTR)

    def _capture_fitted_state(self) -> dict:
        """Portable representation of the fitted state. Overridden per leaf.

        Returns whatever ``fit`` produces beyond the (separately persisted)
        backbone — e.g. a classifier's ``classes_`` + head weights, or ``{}``
        for an SSL encoder whose only product is the trained backbone itself.
        """
        raise NotImplementedError

    def _restore_fitted_state(self, state: dict):
        """Rebuild the fitted state from :meth:`_capture_fitted_state`. Per leaf."""
        raise NotImplementedError

    def to_pickleable_obj(self) -> dict:
        """Portable, dill-safe bundle: estimator + backbone + fitted state.

        The backbone is stored as its ``get_config()`` plus a CPU
        ``state_dict`` (rebuilt by import + ``load_state_dict`` on load), not as
        a live module — this is what keeps a saved model robust across devices
        and torch versions, and lets it nest inside a single-file pipeline.
        """
        backbone = self.backbone
        if not hasattr(backbone, 'get_config'):
            raise AttributeError(
                f"{type(backbone).__name__} must define get_config() to be "
                "persisted (constructor kwargs needed to rebuild it).")
        params = self.get_params(deep=False)
        params.pop('backbone', None)  # stored separately as config + weights
        obj = {
            'format_version': 2,
            'estimator': {'module': type(self).__module__,
                          'qualname': type(self).__qualname__},
            'params': params,
            'backbone': {'module': type(backbone).__module__,
                         'qualname': type(backbone).__qualname__,
                         'config': backbone.get_config(),
                         'state_dict': _cpu_state_dict(backbone)},
            'fitted': self._is_fitted(),
            'history': getattr(self, 'history_', []),
        }
        if obj['fitted']:
            obj['fitted_state'] = self._capture_fitted_state()
        return obj

    @classmethod
    def from_pickleable_obj(cls, obj: dict):
        """Rebuild the estimator from :meth:`to_pickleable_obj`."""
        bb_info = obj['backbone']
        backbone = _import_class(bb_info['module'], bb_info['qualname'])(
            **bb_info['config'])
        backbone.load_state_dict(bb_info['state_dict'])
        est_cls = _import_class(obj['estimator']['module'],
                                obj['estimator']['qualname'])
        est = est_cls(backbone, **obj['params'])
        est.history_ = obj.get('history') or []
        if obj['fitted']:
            est._restore_fitted_state(obj['fitted_state'])
        return est

    # Route *any* dill/pickle of a live estimator (e.g. nested inside a trained
    # pipeline) through the portable config+state_dict bundle, so it reloads across
    # devices/torch versions -- the same contract as ``save``, not a live-module pickle.
    def __getstate__(self) -> dict:
        return self.to_pickleable_obj()

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(type(self).from_pickleable_obj(state).__dict__)

    def save(self, path, protocol=None):
        """Save to a single dill file (CPU-portable ``config + state_dict``)."""
        import dill
        if protocol is None:
            protocol = dill.HIGHEST_PROTOCOL  # protocol 0 bloats tensors
        with open(path, 'wb') as f:
            dill.dump(self.to_pickleable_obj(), f, protocol=protocol)
