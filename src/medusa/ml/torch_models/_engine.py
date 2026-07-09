"""Private training engine shared by the torch estimators.

Everything here is an implementation detail of the public estimators in
:mod:`medusa.ml.torch_models.classification`: numpy/tensor conversion, device
resolution and the Lightning training run (``pl.Trainer`` + early stopping +
best-checkpoint restore). Nothing in this module is public API.
"""
from __future__ import annotations

import importlib
import tempfile

from . import require_lightning

require_lightning()  # friendly error if torch / Lightning is missing

import numpy as np
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.base import BaseEstimator
from torch.utils.data import DataLoader, random_split

from medusa.core.serialization import PickleableComponent


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


def _metric(trainer, name) -> float:
    value = trainer.callback_metrics.get(name)
    return float(value) if value is not None else float('nan')


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
    device : {'auto', 'cpu', 'cuda', 'cuda:N', 'mps'}
        Resolved once at ``fit`` time and reused for inference.
    verbose : bool
        Show the Lightning progress bar.
    """

    def __init__(self, backbone, *, lr=1e-3, max_epochs=100, batch_size=64,
                 val_split=None, patience=10, device='auto', verbose=True):
        self.backbone = backbone
        self.lr = lr
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.val_split = val_split
        self.patience = patience
        self.device = device
        self.verbose = verbose

    # ---- backbone utilities ---- #

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        return self

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True
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

    def _loaders_from_dataset(self, dataset):
        """Split one dataset into ``(train_loader, val_loader)`` per ``val_split``."""
        if self.val_split:
            n_val = max(1, int(len(dataset) * self.val_split))
            train_ds, val_ds = random_split(
                dataset, [len(dataset) - n_val, n_val])
            val_loader = DataLoader(val_ds, batch_size=self.batch_size)
        else:
            train_ds, val_loader = dataset, None
        train_loader = DataLoader(train_ds, batch_size=self.batch_size,
                                  shuffle=True)
        return train_loader, val_loader

    def _run_training(self, task: pl.LightningModule, train_loader, val_loader):
        """Train ``task``; restore best weights, stash ``history_``."""
        self.device_ = _resolve_device(self.device)
        accelerator, devices = _trainer_target(self.device_)
        monitor = 'val_loss' if val_loader is not None else 'train_loss'

        early = EarlyStopping(monitor=monitor, mode='min',
                              patience=self.patience)
        with tempfile.TemporaryDirectory() as ckpt_dir:
            checkpoint = ModelCheckpoint(dirpath=ckpt_dir, monitor=monitor,
                                         mode='min', save_top_k=1)
            trainer = pl.Trainer(
                max_epochs=self.max_epochs,
                accelerator=accelerator,
                devices=devices,
                callbacks=[early, checkpoint],
                logger=False,
                enable_progress_bar=self.verbose,
                enable_model_summary=False,
                num_sanity_val_steps=0)
            trainer.fit(task, train_loader, val_loader)
            if checkpoint.best_model_path:  # restore best weights
                state = torch.load(
                    checkpoint.best_model_path,
                    map_location='cpu'
                )['state_dict']
                task.load_state_dict(state)

        self.backbone.to(self.device_)
        self.history_ = {
            'epochs': int(trainer.current_epoch),
            'train_loss': _metric(trainer, 'train_loss'),
            'val_loss': (_metric(trainer, 'val_loss')
                         if val_loader is not None else float('nan')),
            'stopped_early': bool(early.stopped_epoch),
        }
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
            'history': getattr(self, 'history_', None),
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
        est.history_ = obj.get('history')
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
