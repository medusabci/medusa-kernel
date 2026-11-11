"""Classification paradigm: Lightning tasks + sklearn estimators (co-located).

The training view (Lightning ``LightningModule`` tasks) and the public sklearn
estimators for classification live together here — they change as a unit, and
this is the single ``classification.py`` for the paradigm. Other paradigms get
their own module the same way (a future ``regression.py`` / ``ssl.py``).

Tasks (internal — end users never instantiate these):

* :class:`ClassificationTask` — backbone + one linear head, cross-entropy.
* :class:`MultiTaskClassificationTask` — shared backbone + one head per task,
  summed cross-entropy.

Estimators (public API):

* :class:`TorchClassifier` — backbone + **one** softmax head. Plain sklearn
  classifier: ``fit(X, y)`` / ``predict(X)`` / ``predict_proba(X)`` /
  ``score(X, y)``; the head is sized from ``np.unique(y)`` at fit time, and
  ``class_weight`` re-weights the loss when the classes are unbalanced.
* :class:`TorchMultiTaskClassifier` — shared backbone + **one head per task**,
  trained jointly. ``y`` is a dict mapping task name to labels; ``X`` is either
  one array shared by all tasks or a dict of per-task arrays (e.g. one
  experiment per task). Predict takes the task name explicitly.

Both estimators inherit the training hyperparameters and the Lightning run from
the private engine (:mod:`._engine`); end users never touch ``pl.Trainer``.
"""
from __future__ import annotations

import warnings

from . import require_lightning

require_lightning()  # friendly error if torch / Lightning is missing

import numpy as np
import scipy.special
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import TensorDataset

from ._engine import _BaseTorchEstimator, _cpu_state_dict, encode_labels

# Public API: the sklearn estimators. The Lightning tasks
# (ClassificationTask / MultiTaskClassificationTask) are internal training
# machinery — end users never instantiate them.
__all__ = ['TorchClassifier', 'TorchMultiTaskClassifier']


def _adam(task, lr: float):
    """Adam over ``task``'s trainable parameters only.

    A frozen backbone (the ``finetune`` training profile) contributes none, so the
    optimizer holds state for the head alone. Training a model whose parameters are
    all frozen is a configuration mistake, not a no-op, so it raises.
    """
    params = [p for p in task.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError(
            "nothing to train: every parameter has requires_grad=False. Check the "
            "training profile of the pipeline, or unfreeze the head.")
    return torch.optim.Adam(params, lr=lr)


def _class_weights(class_weight, classes, y_idx):
    """Per-class loss weights, ordered as ``classes``, or ``None`` (unweighted).

    ``None`` — the default — leaves every sample counting the same.
    ``'balanced'`` applies scikit-learn's rule, ``n_samples / (n_classes *
    count)``, so each class contributes the same total weight however rare it
    is. A dict maps a label, in the original encoding of ``y``, to its weight.
    """
    if class_weight is None:
        return None
    weights = compute_class_weight(class_weight, classes=classes,
                                   y=classes[y_idx])
    return np.asarray(weights, dtype=np.float32)


#: The scores :class:`TorchClassifier` can record on the validation fold after
#: every epoch (its ``epoch_metrics``), each a function of ``(y_true, y_pred)``.
EPOCH_METRICS = {
    'accuracy': accuracy_score,
    'balanced_accuracy': balanced_accuracy_score,
}


# ---------------------------------------------------------------------------- #
# Lightning tasks (training view: loss + loop). Internal — the estimators below
# build these in ``fit`` and hand them to the engine; users don't instantiate.
# ---------------------------------------------------------------------------- #

class ClassificationTask(pl.LightningModule):
    """One linear head on a backbone, trained with cross-entropy.

    Batches are plain ``(x, y)`` pairs with ``y`` as int64 class indices.

    Parameters
    ----------
    backbone : nn.Module
        Headless feature extractor: ``backbone(x)`` returns ``[N, F]`` and
        exposes ``backbone_features == F``. Held by reference (training
        updates it in place).
    n_classes : int
        Output size of the linear head.
    lr : float, optional
        Adam learning rate. Default ``1e-3``.
    head : nn.Linear or None, optional
        An existing head to keep training (what a continued ``fit`` passes in).
        ``None`` -- the default -- builds a fresh ``nn.Linear`` of ``n_classes``
        outputs.
    class_weight : array-like or None, optional
        The **resolved** weights: one float per class, in class-index order.
        ``None`` (the default) weights every sample the same. This task takes
        numbers only — the ``'balanced'`` / ``{label: weight}`` spec users write
        is turned into this array by :func:`_class_weights`, in
        :meth:`TorchClassifier.fit`. Registered as a buffer, so it follows the
        module to its device.
    epoch_metrics : dict of {str: callable} or None, optional
        Scores of the validation fold, ``{name: fn(y_true, y_pred)}`` on class
        indices, logged as ``'val_<name>'`` at the end of every validation pass.
        ``None`` (the default) logs the loss only.
    """

    def __init__(self, backbone: nn.Module, n_classes: int, lr: float = 1e-3,
                 class_weight=None, head: "nn.Linear | None" = None,
                 epoch_metrics: "dict | None" = None):
        super().__init__()
        self.backbone = backbone
        self.head = (head if head is not None
                     else nn.Linear(backbone.backbone_features, n_classes))
        self.lr = lr
        self.register_buffer(
            'class_weight',
            None if class_weight is None
            else torch.as_tensor(class_weight, dtype=torch.float32))
        self.epoch_metrics = epoch_metrics or {}
        # The validation pass's true and predicted classes, kept until its end
        self._val_true, self._val_pred = [], []

    def forward(self, x):
        """Return logits: ``head(backbone(x))``."""
        return self.head(self.backbone(x))

    def _loss(self, logits, y):
        """Cross-entropy, weighted per class when ``class_weight`` is set.

        The one definition of the loss, for training and validation alike, so the
        two stay comparable and early stopping watches the objective the model is
        trained on.
        """
        return F.cross_entropy(logits, y, weight=self.class_weight)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self._loss(self(x), y)
        self.log('train_loss', loss, prog_bar=True, on_step=False,
                 on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        self.log('val_loss', self._loss(logits, y), prog_bar=True)
        if self.epoch_metrics:
            self._val_true.append(y.cpu())
            self._val_pred.append(logits.argmax(dim=1).cpu())

    def on_validation_epoch_end(self):
        # The metrics need the whole fold at once (balanced accuracy does not
        # average over batches), so they are scored here, not per batch
        if not self.epoch_metrics:
            return
        y_true = torch.cat(self._val_true).numpy()
        y_pred = torch.cat(self._val_pred).numpy()
        for name, score in self.epoch_metrics.items():
            self.log(f'val_{name}', float(score(y_true, y_pred)), prog_bar=True)
        self._val_true, self._val_pred = [], []

    def configure_optimizers(self):
        """Adam with PyTorch defaults over the trainable parameters."""
        return _adam(self, self.lr)


class MultiTaskClassificationTask(pl.LightningModule):
    """One linear head per task on a shared backbone; summed cross-entropy.

    Batches are dicts ``{task_name: (x, y)}`` (one entry per task), so the
    same step covers tasks trained on the **same** ``X`` and tasks trained on
    **different** datasets uniformly — the estimator feeds these via
    ``lightning...CombinedLoader``. Each task's features come from its own
    forward pass; the backbone receives the gradient of the summed per-task
    cross-entropies, so it is shaped jointly by every task/experiment.

    Parameters
    ----------
    backbone : nn.Module
        Headless feature extractor (see :class:`ClassificationTask`), shared
        by all heads.
    n_classes_per_task : dict of {str: int}
        Maps task name to number of classes; one ``nn.Linear(F, n_classes)``
        is created per entry.
    lr : float, optional
        Adam learning rate. Default ``1e-3``.
    heads : nn.ModuleDict or None, optional
        Existing heads to keep training (what a continued ``fit`` passes in).
        ``None`` -- the default -- builds one fresh head per task.
    """

    def __init__(self, backbone: nn.Module, n_classes_per_task: dict,
                 lr: float = 1e-3, heads: "nn.ModuleDict | None" = None):
        super().__init__()
        self.backbone = backbone
        self.heads = heads if heads is not None else nn.ModuleDict({
            name: nn.Linear(backbone.backbone_features, n_classes)
            for name, n_classes in n_classes_per_task.items()
        })
        self.lr = lr

    def forward(self, x, head: str):
        """Return logits for ``head``: ``head(backbone(x))``."""
        return self.heads[head](self.backbone(x))

    def _loss(self, batch) -> torch.Tensor:
        """Sum of per-task cross-entropy over a ``{task: (x, y)}`` batch."""
        return sum(F.cross_entropy(self(x, name), y)
                   for name, (x, y) in batch.items())

    @staticmethod
    def _batch_size(batch) -> int:
        return sum(x.shape[0] for x, _ in batch.values())

    def training_step(self, batch, batch_idx):
        loss = self._loss(batch)
        self.log('train_loss', loss, prog_bar=True, on_step=False,
                 on_epoch=True, batch_size=self._batch_size(batch))
        return loss

    def validation_step(self, batch, batch_idx):
        self.log('val_loss', self._loss(batch), prog_bar=True,
                 batch_size=self._batch_size(batch))

    def configure_optimizers(self):
        """Adam with PyTorch defaults over the trainable parameters."""
        return _adam(self, self.lr)


# ---------------------------------------------------------------------------- #
# sklearn estimators (public API). Constructor parameters come from the engine
# (:class:`._engine._BaseTorchEstimator`); only TorchClassifier adds one of its
# own (``class_weight``), so only it needs an __init__.
# ---------------------------------------------------------------------------- #

class TorchClassifier(ClassifierMixin, _BaseTorchEstimator):
    """Backbone + one softmax head, trained with cross-entropy.

    Everything about the head is inferred at ``fit`` time: ``classes_`` is
    ``np.unique(y)`` and sizes the head, and ``predict`` returns labels in the
    original encoding. ``score`` (accuracy) comes from sklearn's
    ``ClassifierMixin``. See :class:`._engine._BaseTorchEstimator` for the
    constructor parameters.

    Parameters
    ----------
    class_weight : None, 'balanced' or dict, optional
        How much each class counts in the loss. ``None`` (the default) counts
        every sample the same. ``'balanced'`` gives each class the same total
        weight, ``n_samples / (n_classes * count)``, which is what an unbalanced
        problem usually wants — a rare class stops being cheap to ignore. A dict
        ``{label: weight}`` sets the weights yourself, in the original encoding
        of ``y``. The weighting applies to the validation loss too, so early
        stopping watches the objective the model is trained on.
    epoch_metrics : sequence of str, optional
        Scores of the validation fold to record after every epoch, besides its
        loss: ``'accuracy'``, ``'balanced_accuracy'``. They are computed in the
        validation pass training already runs, so they cost almost nothing and
        never change the model, and they land in ``history_`` as
        ``'val_<metric>_curve'``. They need ``val_split``; without it there is no
        fold to score and ``fit`` warns. Empty (the default) records the losses
        only.

    Examples
    --------
    >>> clf = TorchClassifier(EEGInceptionV2(...), max_epochs=50, val_split=0.2)
    >>> clf.fit(X_train, y_train).score(X_test, y_test)

    A rare positive class, weighted up so training does not ignore it:

    >>> clf = TorchClassifier(EEGInceptionV2(...), class_weight='balanced')

    The validation balanced accuracy of every epoch, next to the losses:

    >>> clf = TorchClassifier(EEGInceptionV2(...), val_split=0.2,
    ...                       epoch_metrics=['balanced_accuracy'])
    >>> clf.fit(X_train, y_train)
    >>> clf.history_[-1]['val_balanced_accuracy_curve']     # one value per epoch
    """

    # The engine's parameters are written out again here because scikit-learn
    # builds ``get_params`` from this signature: a ``**kwargs`` would hide them
    # from ``clone()``, from ``cross_validate`` and from the saved bundle.
    def __init__(self, backbone, *, lr=1e-3, max_epochs=100, batch_size=64,
                 val_split=None, patience=10, min_delta=0.0, device='auto',
                 verbose=1, class_weight=None, random_state=None,
                 epoch_metrics=()):
        super().__init__(backbone, lr=lr, max_epochs=max_epochs,
                         batch_size=batch_size, val_split=val_split,
                         patience=patience, min_delta=min_delta, device=device,
                         verbose=verbose, random_state=random_state)
        self.class_weight = class_weight
        self.epoch_metrics = epoch_metrics

    def fit(self, X, y, groups=None):
        """Fit backbone and head on ``X`` (epochs) and ``y`` (labels).

        A second ``fit`` **continues** the model: the same backbone and the same head
        keep training. Call :meth:`reset_head` first to train a fresh classifier on
        the learned features instead (a new subject, or a new set of labels).

        Parameters
        ----------
        X : array-like
            Epochs, in the layout the backbone declares.
        y : array-like
            One label per epoch (categorical or one-hot).
        groups : array-like of shape (n_epochs,), optional
            Group id of every epoch (the trial it was cut from, say). When given and
            ``val_split`` is on, the validation fold holds out **whole groups**, so
            an epoch never has a near-copy of itself on the other side (see
            :meth:`_loaders_from_dataset`). ``None`` (the default) splits single
            epochs, stratified by label.

        Raises
        ------
        ValueError
            If ``X`` and ``y`` differ in length, or an ``epoch_metrics`` name is
            unknown.
        """
        classes, y_idx = encode_labels(y)
        if len(y_idx) != len(X):
            raise ValueError(
                f"X and y have different lengths: {len(X)} vs {len(y_idx)}.")
        head = self._head_to_continue(classes)
        metrics = self._epoch_metric_functions()
        self.classes_ = classes
        with self._rng_scope():   # head init, split, batch order and dropout
            dataset = TensorDataset(self._prepare(X), torch.as_tensor(y_idx))
            task = ClassificationTask(
                self.backbone,
                len(self.classes_),
                lr=self.lr,
                class_weight=_class_weights(self.class_weight, self.classes_,
                                            y_idx),
                head=head,
                epoch_metrics=metrics)
            train_loader, val_loader = self._loaders_from_dataset(
                dataset, labels=y_idx, groups=groups)
            self._run_training(task, train_loader, val_loader,
                               metric_names=list(metrics))
        self.head_ = task.head
        return self

    def _epoch_metric_functions(self) -> dict:
        """``{name: fn(y_true, y_pred)}`` for every ``epoch_metrics`` entry; empty,
        with a warning, when there is no validation fold to score."""
        unknown = [name for name in self.epoch_metrics if name not in EPOCH_METRICS]
        if unknown:
            raise ValueError(f"unknown epoch_metrics {unknown}; choose among "
                             f"{list(EPOCH_METRICS)}.")
        if self.epoch_metrics and not self.val_split:
            warnings.warn(
                f"epoch_metrics {list(self.epoch_metrics)} are scored on the "
                f"validation fold, and val_split is off: nothing is recorded.",
                UserWarning, stacklevel=3)
            return {}
        return {name: EPOCH_METRICS[name] for name in self.epoch_metrics}

    def _head_to_continue(self, classes):
        """The fitted head when it still fits ``classes``, else ``None`` or an error."""
        if not self._is_fitted():
            return None
        if not np.array_equal(self.classes_, classes):
            raise ValueError(
                f"this estimator is fitted on classes {list(self.classes_)}, but y "
                f"has {list(classes)}. A fitted head cannot change its classes: call "
                f"reset_head() to train a new one on the same backbone.")
        return self.head_

    def predict_proba(self, X) -> np.ndarray:
        """Class probabilities, columns ordered as ``classes_``."""
        self._check_fitted()
        return scipy.special.softmax(self._head_logits(X, self.head_), axis=1)

    def predict(self, X) -> np.ndarray:
        """Predicted labels, in the original encoding of ``y``."""
        proba = self.predict_proba(X)
        return self.classes_[proba.argmax(axis=1)]

    # ---- persistence ---- #
    _FITTED_ATTR = 'head_'

    def _capture_fitted_state(self) -> dict:
        return {'classes_': self.classes_, 'head': _capture_linear(self.head_)}

    def _restore_fitted_state(self, state: dict):
        self.classes_ = state['classes_']
        self.head_ = _restore_linear(state['head'])


class TorchMultiTaskClassifier(_BaseTorchEstimator):
    """Shared backbone + one softmax head per task, trained jointly.

    Each optimizer step backpropagates the sum of the per-task cross-entropies
    through the shared backbone, so every task shapes the backbone. Heads and
    their ``classes_`` are inferred from the ``y`` dict at ``fit`` time.

    ``X`` can be:

    * a single array shared by all tasks — multi-label learning on the same
      epochs (e.g. predict stimulus *and* attention from one EEG segment), or
    * a dict ``{task_name: X_task}`` — each task trained on its **own**
      dataset (e.g. one ERP experiment per task, with different epoch counts
      and different numbers of classes). All datasets must share the same
      per-epoch shape, since the backbone is fixed-size. Uneven dataset sizes
      are handled by cycling the smaller ones each epoch
      (``CombinedLoader(mode='max_size_cycle')``).

    This is deliberately **not** a plain sklearn classifier (``y`` is a dict
    and predictions are per task), so it does not get ``ClassifierMixin``;
    ``score`` returns a per-task accuracy dict instead. See
    :class:`._engine._BaseTorchEstimator` for the constructor parameters.

    Examples
    --------
    Same data, two labellings:

    >>> mt = TorchMultiTaskClassifier(EEGInceptionV2(...), val_split=0.2)
    >>> mt.fit(X, {'stimulus': y_stim, 'attention': y_att})

    Different experiment per task:

    >>> mt.fit({'exp1': X1, 'exp2': X2}, {'exp1': y1, 'exp2': y2})
    >>> mt.predict(X_test, 'exp1')
    """

    def fit(self, X, y: dict):
        """Fit all heads jointly.

        Parameters
        ----------
        X : array or dict of {task_name: array}
            One array shared by all tasks, or one array per task. When a dict,
            its keys must match ``y``'s.
        y : dict of {task_name: labels}
            Labels per task. Sizes each task's head from ``np.unique``.
        """
        if not isinstance(y, dict) or not y:
            raise ValueError(
                "y must be a non-empty dict {task_name: labels}; for a "
                "single task use TorchClassifier.")
        X_per_task = self._resolve_task_X(X, y)

        previous = getattr(self, 'classes_', None) if self._is_fitted() else None
        self.classes_ = {}
        datasets = {}
        for task_name, y_task in y.items():
            classes, y_idx = encode_labels(y_task)
            X_task = X_per_task[task_name]
            if len(y_idx) != len(X_task):
                raise ValueError(
                    f"task '{task_name}': X has {len(X_task)} epochs but y "
                    f"has {len(y_idx)} labels.")
            self.classes_[task_name] = classes
            datasets[task_name] = (TensorDataset(self._prepare(X_task),
                                                 torch.as_tensor(y_idx)), y_idx)

        heads = self._heads_to_continue(previous)
        with self._rng_scope():   # head init, splits, batch order and dropout
            train, val = {}, {}
            for task_name, (dataset, y_idx) in datasets.items():
                train[task_name], val_loader = self._loaders_from_dataset(
                    dataset, labels=y_idx)
                if val_loader is not None:
                    val[task_name] = val_loader
            train_loader = CombinedLoader(train, mode='max_size_cycle')
            val_loader = (CombinedLoader(val, mode='max_size_cycle')
                          if val else None)

            task = MultiTaskClassificationTask(
                self.backbone,
                {name: len(c) for name, c in self.classes_.items()}, lr=self.lr,
                heads=heads)
            self._run_training(task, train_loader, val_loader)
        self.heads_ = task.heads
        return self

    def _heads_to_continue(self, previous):
        """The fitted heads when they still fit the new tasks, else ``None`` or an error."""
        if previous is None:
            return None
        same = (set(previous) == set(self.classes_)
                and all(np.array_equal(previous[name], self.classes_[name])
                        for name in previous))
        if not same:
            raise ValueError(
                f"this estimator is fitted on tasks/classes "
                f"{ {k: list(v) for k, v in previous.items()} }, but y has "
                f"{ {k: list(v) for k, v in self.classes_.items()} }. Fitted heads "
                f"cannot change their tasks or classes: call reset_head() to train "
                f"new ones on the same backbone.")
        return self.heads_

    @staticmethod
    def _resolve_task_X(X, y: dict) -> dict:
        """Normalise ``X`` to ``{task_name: array}`` matching ``y``'s keys."""
        if isinstance(X, dict):
            if set(X) != set(y):
                raise ValueError(
                    f"X and y must have the same task names; got X={list(X)} "
                    f"vs y={list(y)}.")
            return X
        return {task_name: X for task_name in y}  # one array shared by all

    def _get_head(self, task_name: str):
        self._check_fitted()
        if task_name not in self.heads_:
            raise ValueError(f"unknown task '{task_name}'; fitted tasks: "
                             f"{list(self.heads_)}.")
        return self.heads_[task_name]

    def predict_proba(self, X, task_name: str) -> np.ndarray:
        """Class probabilities for one task, ordered as its ``classes_``."""
        logits = self._head_logits(X, self._get_head(task_name))
        return scipy.special.softmax(logits, axis=1)

    def predict(self, X, task_name: str) -> np.ndarray:
        """Predicted labels for one task, in its original encoding."""
        proba = self.predict_proba(X, task_name)
        return self.classes_[task_name][proba.argmax(axis=1)]

    def score(self, X, y: dict) -> dict:
        """Accuracy per task. ``X`` may be shared or a per-task dict."""
        X_per_task = self._resolve_task_X(X, y)
        return {name: float(np.mean(
            self.predict(X_per_task[name], name)
            == np.asarray(y_task).reshape(-1)))
            for name, y_task in y.items()}

    # ---- persistence ---- #
    _FITTED_ATTR = 'heads_'

    def _capture_fitted_state(self) -> dict:
        return {'classes_': self.classes_,
                'heads': {name: _capture_linear(head)
                          for name, head in self.heads_.items()}}

    def _restore_fitted_state(self, state: dict):
        self.classes_ = state['classes_']
        self.heads_ = torch.nn.ModuleDict(
            {name: _restore_linear(h) for name, h in state['heads'].items()})


def _capture_linear(linear) -> dict:
    """Portable representation of an ``nn.Linear`` head (sizes + CPU weights)."""
    return {'in_features': linear.in_features,
            'out_features': linear.out_features,
            'state_dict': _cpu_state_dict(linear)}


def _restore_linear(head: dict):
    """Rebuild an ``nn.Linear`` from :func:`_capture_linear`."""
    linear = torch.nn.Linear(head['in_features'], head['out_features'])
    linear.load_state_dict(head['state_dict'])
    return linear