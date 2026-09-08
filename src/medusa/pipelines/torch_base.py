"""Shared foundation for decoding pipelines whose model is a torch classifier.

:class:`TorchPipeline` is the base for any pipeline that decodes with a
:class:`~medusa.ml.torch_models.classification.TorchClassifier` -- a backbone plus a
classification head. It sits beside :class:`~medusa.pipelines.base.DecodingPipeline`
rather than inside it because that module deliberately imports neither Qt nor torch, so a
headless install and the shallow decoders (LDA, CCA, CSP) stay torch-free. Nothing here is
BCI-specific.

It gives a concrete pipeline three things:

* **the model lifecycle** -- build the backbone on the first ``fit``, keep training it on
  the next one, and the explicit ways back: :meth:`~TorchPipeline.restart`,
  :meth:`~TorchPipeline.reset_head`, :meth:`~TorchPipeline.set_backbone`;
* **training profiles** -- what is allowed to move in this phase, as a setting
  (``classifier.training.profile``), applied to the backbone before every fit;
* **the settings schema and persistence** -- :func:`add_training_settings` /
  :func:`training_kwargs` describe the ``TorchClassifier`` hyper-parameters once for every
  torch pipeline, and the save bundle (settings + fitted state) is implemented here.

A subclass supplies its feature path and one method, ``_build_backbone(cfg, X)``.

Training profiles
-----------------

===================  ==================  =========================  =======
profile              backbone weights    normalization statistics   head
===================  ==================  =========================  =======
``train`` (default)  train               update                     trains
``finetune``         frozen              update                     trains
``custom``           untouched           untouched                  trains
===================  ==================  =========================  =======

``finetune`` is exactly ``requires_grad = False`` over the backbone. The module still runs
in training mode, so the running statistics of its normalization layers follow the new
data -- the cheap and effective part of subject transfer -- while no weight moves. Its
dropout also stays active, which regularises the head; that differs from Keras-style
``trainable=False``, where a frozen layer is switched to inference mode as well.
``custom`` is the escape hatch: the pipeline touches nothing, so any ``requires_grad``
flags you set by hand on ``pipe.clf.backbone`` survive the next fit.

A pipeline that needs more (an architecture-specific recipe, a paradigm-specific one)
declares the names and implements them::

    class MyPipeline(TorchPipeline):
        TRAINING_PROFILES = TorchPipeline.TRAINING_PROFILES + ("spatial_only",)

        def _apply_training_profile(self, profile, backbone, cfg):
            if profile == "spatial_only":
                ...                                   # this architecture's own recipe
                return
            super()._apply_training_profile(profile, backbone, cfg)

The settings dropdown is built from ``cls.TRAINING_PROFILES``, so each pipeline offers
exactly what it implements, and the hook receives ``cfg`` -- a profile that only makes
sense for one architecture can branch on ``cfg["classifier"]["arch"]``.

Multi-phase training
--------------------

``fit`` continues, so pretrain-then-adapt is a sequence of fits with the settings changed
in between::

    pipe.fit(corpus)                                  # pretrain
    pipe.save("pretrained.pkl")

    pipe = DecodingPipeline.load("pretrained.pkl")
    pipe.reset_head()                                 # new subject => new head
    pipe.settings.update_from_dict({"classifier": {"training": {
        "profile": "finetune", "max_epochs": 30}}})
    pipe.fit(subject_recordings)                      # head + statistics adapt

    pipe.settings.update_from_dict({"classifier": {"training": {
        "profile": "train", "learning_rate": 1e-4, "max_epochs": 10}}})
    pipe.fit(subject_recordings)                      # polish everything, head kept

Optimizer state is not carried across phases (each ``fit`` starts a fresh Adam), which is
the usual choice for finetuning.
"""
from __future__ import annotations

import warnings
from abc import abstractmethod
from typing import TYPE_CHECKING, Self

from medusa.core.serialization import pack_pickleable, unpack_pickleable
from medusa.ml.torch_models._engine import seeded_rng
from medusa.ml.torch_models.classification import TorchClassifier
from medusa.pipelines.base import DecodingPipeline

if TYPE_CHECKING:
    from collections.abc import Sequence

    import torch.nn as nn
    from numpy.typing import NDArray

    from medusa.core.settings_tree import SettingsTree

__all__ = ["TorchPipeline", "TRAINING_PROFILES", "add_training_settings",
           "training_kwargs"]

#: Training profiles every torch pipeline understands. See the module docstring for what
#: each one moves; a pipeline that needs more extends this tuple and implements them in
#: :meth:`TorchPipeline._apply_training_profile`.
TRAINING_PROFILES = ("train", "finetune", "custom")


# --------------------------------------------------------------------------- #
# Settings: the TorchClassifier hyper-parameters, described once
# --------------------------------------------------------------------------- #
def add_training_settings(clf_group: "SettingsTree", *,
                          profiles: "Sequence[str]" = TRAINING_PROFILES,
                          max_epochs: int = 100, batch_size: int = 64,
                          learning_rate: float = 1e-3, patience: int = 10) -> None:
    """Add the ``training`` subgroup of
    :class:`~medusa.ml.torch_models.classification.TorchClassifier` hyper-parameters to a
    classifier settings group.

    Every torch pipeline uses this one schema, so the knobs and their help text cannot
    drift apart. The defaults that legitimately differ between paradigms (a VEP speller
    trains on far more, far shorter epochs than a motor-imagery decoder) are arguments.

    Parameters
    ----------
    clf_group : SettingsTree
        The ``classifier`` group to add the ``training`` subgroup to.
    profiles : sequence of str, optional
        The training profiles this pipeline implements, offered as the options of the
        ``profile`` item. Pass ``cls.TRAINING_PROFILES``.
    max_epochs, batch_size, learning_rate, patience : optional
        Per-pipeline defaults for the items of the same name.
    """
    tr = clf_group.add_group("training", info="TorchClassifier training hyper-parameters")
    tr.add_item("profile", value="train", value_options=list(profiles),
                info="What trains in this run: 'train' the whole model; 'finetune' only "
                     "the head, with the backbone frozen but its normalization "
                     "statistics still following the new data; 'custom' leaves the "
                     "backbone exactly as you configured it by hand")
    tr.add_item("max_epochs", value=int(max_epochs), value_range=[1, None],
                info="Maximum training epochs")
    tr.add_item("batch_size", value=int(batch_size), value_range=[1, None],
                info="Mini-batch size")
    tr.add_item("learning_rate", value=float(learning_rate), value_range=[0, None],
                info="Adam learning rate")
    tr.add_item("class_weight", value="balanced", optional=True, enabled=False,
                value_options=["balanced"],
                info="Weight the loss by class frequency, so a rare class counts as much "
                     "as a common one; switch it off to weight every observation the same")
    tr.add_item("val_split", value=0.2, optional=True, value_range=[0, 1],
                info="Validation fraction for early stopping; switch it off to train "
                     "without a validation split")
    tr.add_item("val_split_stratify", value=True,
                info="Keep each class's share of the data in the validation split "
                     "(only used when val_split is on)")
    tr.add_item("patience", value=int(patience), value_range=[1, None],
                info="Early-stopping patience (epochs)")
    tr.add_item("random_state", value=0, optional=True, enabled=False,
                value_range=[0, None],
                info="Seed that makes a fit repeatable: the backbone's initial weights, "
                     "the validation split, the batch order and the dropout masks. "
                     "Switch it off to let every run draw its own")
    tr.add_item("device", value="auto",
                info="Compute device ('auto', 'cpu', 'cuda', 'cuda:N', 'mps'); keep "
                     "'auto' so a saved model reloads on any host")
    tr.add_item("verbose", value="epoch", value_options=["silent", "epoch", "full"],
                info="Training log: 'silent' / 'epoch' (one line per epoch) / "
                     "'full' (Lightning debug)")


def training_kwargs(cfg_training: dict) -> dict:
    """Map a ``classifier.training`` config dict to ``TorchClassifier`` keyword arguments.

    ``profile`` is not among them: it describes what the *pipeline* does to the backbone
    before training, not how the estimator is built.
    """
    t = cfg_training
    return dict(lr=float(t["learning_rate"]), max_epochs=int(t["max_epochs"]),
                batch_size=int(t["batch_size"]), class_weight=t["class_weight"],
                val_split=t["val_split"] or None,
                val_split_stratify=t["val_split_stratify"],
                patience=int(t["patience"]), device=t["device"], verbose=t["verbose"],
                random_state=(None if t["random_state"] is None
                              else int(t["random_state"])))


# --------------------------------------------------------------------------- #
# The base pipeline
# --------------------------------------------------------------------------- #
class TorchPipeline(DecodingPipeline):
    """A :class:`~medusa.pipelines.base.DecodingPipeline` whose model is a
    :class:`~medusa.ml.torch_models.classification.TorchClassifier`.

    Holds the model lifecycle, the training profiles and the save bundle (see the module
    docstring). A subclass writes its settings schema, its feature path, its
    ``check_consistency``, and:

    * ``_build_backbone(cfg, X)`` -- build the backbone for this configuration, sized to
      the feature array of the first fit;
    * a ``fit`` that gathers ``X`` and ``y`` from the recordings and hands them to
      :meth:`_fit_classifier`.

    Everything else -- continuing a trained model, the profile, the guards, persistence --
    comes from here.
    """

    #: The estimator: a ``TorchClassifier`` wrapping the backbone. Built by the first
    #: ``fit``, kept afterwards, restored by ``load``.
    clf = None

    #: Profiles this pipeline understands; a subclass extends it and implements the extra
    #: names in :meth:`_apply_training_profile`.
    TRAINING_PROFILES = TRAINING_PROFILES

    #: True once the backbone has been trained (or installed by
    #: :meth:`set_backbone`); survives :meth:`reset_head`, cleared by :meth:`restart`.
    _backbone_trained = False

    #: The configuration of the last fit, to notice a setting that drifted since.
    _fit_cfg = None

    # ---- what a subclass provides ---- #
    @abstractmethod
    def _build_backbone(self, cfg: dict, X: "NDArray") -> "nn.Module":
        """Build the backbone for ``cfg``, sized to the feature array ``X``.

        Called once, on the first fit -- afterwards the pipeline keeps the backbone it
        has. ``X`` is the pipeline's own feature array, so the backbone can be sized to
        the data (epoch length, channel count) and can never desync from it.
        """

    @staticmethod
    def _training_cfg(cfg: dict) -> dict:
        """The training sub-configuration (``classifier.training`` by convention)."""
        return cfg["classifier"]["training"]

    # ---- lifecycle ---- #
    def restart(self) -> Self:
        """Forget the trained model; the next ``fit`` rebuilds it from the settings."""
        self.clf = None
        self._backbone_trained = False
        self._fit_cfg = None
        return super().restart()

    def reset_head(self) -> Self:
        """Drop the classifier layer, keep the trained backbone; return ``self``.

        The transfer move: the next ``fit`` trains a fresh head on the features learned
        so far. Use it for a new subject, or whenever the labels change -- a fitted head
        cannot change its classes. The pipeline counts as unfitted until that fit.
        """
        if self.clf is not None:
            self.clf.reset_head()
        self._fitted = False
        return self

    def set_backbone(self, backbone: "nn.Module") -> Self:
        """Install ``backbone`` as this pipeline's feature extractor; return ``self``.

        The way pretrained features enter a pipeline, whatever trained them: another
        pipeline (``pipe.set_backbone(other.clf.backbone)``, or
        ``DecodingPipeline.load(path).clf.backbone``), a multi-task estimator, a
        self-supervised encoder, a notebook. The head is not carried over, so the next
        ``fit`` trains a new one on these features.

        The backbone has to match what this pipeline's settings describe; it is checked
        on the first batch of the next fit, which is where the epoch shape is known.
        """
        self.clf = TorchClassifier(
            backbone, **training_kwargs(self._training_cfg(self.cfg)))
        self._backbone_trained = True
        self._fitted = False
        return self

    # ---- training ---- #
    def _fit_classifier(self, cfg: dict, X: "NDArray", y: "NDArray") -> Self:
        """Build or continue the classifier, train it on ``(X, y)``, return ``self``.

        The tail of every subclass's ``fit``: gather the features and labels, then hand
        them here.
        """
        self._ensure_classifier(cfg, X)
        self.clf.fit(X, y)
        self._backbone_trained = True
        self._fitted = True
        self._fit_cfg = cfg     # the configuration of the last *successful* fit
        return self

    def _ensure_classifier(self, cfg: dict, X: "NDArray") -> None:
        """Make ``self.clf`` ready for this fit: build it, or carry the trained one on.

        On a first fit the backbone is built from the settings (under the configured
        seed, so ``random_state`` covers its initial weights too). On a later fit the
        model is kept and only re-configured, so this phase's hyper-parameters take
        effect. The profile is applied either way, so the setting is authoritative on
        every fit rather than only when it changes.
        """
        training = self._training_cfg(cfg)
        if self.clf is None:
            with seeded_rng(training["random_state"], training["device"]):
                backbone = self._build_backbone(cfg, X)
            self.clf = TorchClassifier(backbone, **training_kwargs(training))
        else:
            self._check_architecture(cfg)
            self._warn_settings_drift(cfg)
            self.clf.set_params(**training_kwargs(training))
        self._set_training_profile(training["profile"], cfg)

    # ---- training profiles ---- #
    def _set_training_profile(self, profile: str, cfg: dict) -> None:
        """Validate ``profile`` and apply it to the backbone."""
        if profile not in self.TRAINING_PROFILES:
            raise ValueError(
                f"unknown training profile {profile!r} for {type(self).__name__}; "
                f"choose one of {list(self.TRAINING_PROFILES)}.")
        if profile == "finetune" and not self._backbone_trained:
            warnings.warn(
                f"{type(self).__name__}: the 'finetune' profile freezes the backbone, "
                f"but this one has never been trained -- the head would be fitted on "
                f"random features. Train it first (profile 'train'), or bring trained "
                f"features in with set_backbone() or load().",
                UserWarning, stacklevel=5)
        self._apply_training_profile(profile, self.clf.backbone, cfg)

    def _apply_training_profile(self, profile: str, backbone: "nn.Module",
                                cfg: dict) -> None:
        """Set ``requires_grad`` on ``backbone`` for ``profile``.

        The extension point: a subclass that declares extra profiles handles its own
        names here and calls ``super()`` for the rest. ``cfg`` is passed so a profile
        that only applies to one architecture can branch on ``classifier.arch``.
        """
        if profile == "custom":
            return
        if profile not in ("train", "finetune"):
            raise ValueError(
                f"{type(self).__name__} declares the training profile {profile!r} in "
                f"TRAINING_PROFILES but does not implement it; override "
                f"_apply_training_profile to handle it.")
        for p in backbone.parameters():
            p.requires_grad = profile == "train"

    # ---- guards ---- #
    def _check_architecture(self, cfg: dict) -> None:
        """Refuse to continue a model whose architecture setting changed since the fit.

        Only the architecture is checked here. The epoch shape is validated by the
        backbone itself, in ``prepare_X``, on the first batch of every fit and predict,
        with a message naming the shape it was built for -- so a changed segment window,
        resampling rate or channel list is already caught, in both directions.
        """
        was = (self._fit_cfg or {}).get("classifier", {}).get("arch")
        now = cfg.get("classifier", {}).get("arch")
        if was is not None and was != now:
            raise ValueError(
                f"this pipeline holds a model built as {was!r}, but classifier.arch is "
                f"now {now!r}. A fit continues the model it has and cannot change its "
                f"architecture: call restart() to build the new one from scratch.")

    def _warn_settings_drift(self, cfg: dict) -> None:
        """Warn when settings other than the training ones changed since the last fit.

        A continued fit reads the live settings tree, so it can disagree with the model
        it is continuing: a different band or baseline keeps the epoch shape and raises
        nothing, it just feeds the model a signal it was not trained on. The training
        group is exempt on purpose -- changing the profile, learning rate, epochs or seed
        between phases is the whole point.
        """
        if self._fit_cfg is None:
            return
        changed = _changed_keys(_without_training(self._fit_cfg),
                               _without_training(cfg))
        if changed:
            warnings.warn(
                f"{type(self).__name__}: settings changed since the last fit "
                f"({', '.join(changed)}), but this fit continues the model trained with "
                f"the previous ones. Call restart() to train from scratch with the new "
                f"settings.", UserWarning, stacklevel=5)

    # ---- persistence ---- #
    def to_pickleable_obj(self) -> dict:
        """Bundle the settings and the fitted state (the classifier packed portably).

        The classifier is saved whenever there is one, not only when the pipeline counts
        as fitted, so a pretrained backbone survives ``reset_head`` + ``save``.
        """
        return {"settings": self.settings.to_dict(), "fitted": self._fitted,
                "fs": self.fs, "backbone_trained": self._backbone_trained,
                "fit_cfg": self._fit_cfg,
                "clf": pack_pickleable(self.clf) if self.clf is not None else None}

    @classmethod
    def from_pickleable_obj(cls, obj: dict) -> Self:
        """Rebuild the pipeline from a bundle made by :meth:`to_pickleable_obj`."""
        self = cls(settings=obj["settings"])
        self.fs, self._fitted = obj["fs"], obj["fitted"]
        self._backbone_trained = obj.get("backbone_trained", obj["fitted"])
        self._fit_cfg = obj.get("fit_cfg")
        if obj["clf"] is not None:
            self.clf = unpack_pickleable(obj["clf"])
        return self


# --------------------------------------------------------------------------- #
# Small helpers for the drift warning
# --------------------------------------------------------------------------- #
def _without_training(cfg: dict) -> dict:
    """``cfg`` with ``classifier.training`` removed (it is meant to change per phase)."""
    out = dict(cfg)
    classifier = out.get("classifier")
    if isinstance(classifier, dict):
        out["classifier"] = {k: v for k, v in classifier.items() if k != "training"}
    return out


def _changed_keys(old: dict, new: dict, prefix: str = "") -> "list[str]":
    """Dotted names of the values that differ between two nested config dicts."""
    changed = []
    for key in dict.fromkeys(list(old) + list(new)):
        name = f"{prefix}{key}"
        a, b = old.get(key), new.get(key)
        if isinstance(a, dict) and isinstance(b, dict):
            changed += _changed_keys(a, b, f"{name}.")
        elif a != b:
            changed.append(name)
    return changed
