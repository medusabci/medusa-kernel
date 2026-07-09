"""Shared building blocks for every decoding pipeline.

A **decoding pipeline** is the trainable, stateful, saveable core of a paradigm
(Layer 1). It turns EEG into one *score* per segment. It is the only class hierarchy
in the package, on purpose. Application data are thin
:class:`~medusa.core.serialization.SerializableComponent` containers. Command
decoding (Layer 2, used only by VEP spellers) is a separate ``CommandDecoder``.
Offline *analysis pipelines* are plain functions (scripts), not classes.

This module holds:

* :class:`DecodingPipeline` -- the Layer-1 base. It builds on the
  :class:`~medusa.core.settings_tree.Configurable` mixin (``SettingsTree``-backed
  configuration) and :class:`~medusa.core.serialization.PickleableComponent`. It runs
  ``fit`` / ``predict`` to make scores, has matching online methods, validates each
  recording, and saves to a single file that a polymorphic
  :meth:`~DecodingPipeline.load` reads back.
* :func:`load_recordings` -- load many recordings, one at a time or all at once (there
  is no ``Dataset`` class).
* :func:`leave_one_recording_out` -- a leave-one-recording-out cross-validation helper
  that needs no indices.
* :func:`harmonize_channels` -- pick and reorder a stream to a target channel list, with
  a clear error message.

The base imports neither Qt nor torch. Shallow decoders (LDA, CCA) and a headless
install stay torch-free. Deep pipelines import torch from their own module.
"""
from __future__ import annotations

import importlib
from abc import abstractmethod
from collections.abc import Iterable, Iterator, Sequence
from typing import Self

import dill
from numpy.typing import NDArray

from medusa.core.serialization import (
    PickleableComponent, pack_pickleable, unpack_pickleable)
from medusa.core.settings_tree import Configurable, SettingsTree
from medusa.core.data.recording import Recording
from medusa.core.data.signal import Signal

__all__ = [
    "DecodingPipeline",
    "load_recordings",
    "leave_one_recording_out",
    "harmonize_channels",
    # re-exported for concrete pipelines that build their persistence bundle
    "pack_pickleable",
    "unpack_pickleable",
]


class DecodingPipeline(Configurable, PickleableComponent):
    """Base class for a Layer-1 decoding pipeline: trained EEG -> one score per segment.

    A concrete pipeline reads its configuration from :attr:`cfg` (backed by the live
    :attr:`settings` tree). It holds its *fitted sub-estimators* (sklearn classifiers,
    :class:`~medusa.signal.spatial_filtering.CSP`, torch estimators, and so on) and any
    *online state*. In :meth:`fit` and :meth:`predict` it calls the free functions in
    :mod:`medusa.signal` and :mod:`medusa.ml`. There is no composition graph, no
    ``configure`` / ``build`` lifecycle, and no reflection.

    Contract
    --------
    A subclass implements:

    * :meth:`default_settings` -- its configuration schema (from
      :class:`~medusa.core.settings_tree.Configurable`);
    * :meth:`fit`, :meth:`predict`, :meth:`check_consistency` -- its behaviour;
    * :meth:`to_pickleable_obj` / :meth:`from_pickleable_obj` -- what to save. The bundle
      must hold both ``settings`` **and** the fitted state, e.g.::

          def to_pickleable_obj(self):
              return {"settings": self.settings.to_dict(), "fitted": self._fitted,
                      "clf": pack_pickleable(self.clf), "fs": self.fs}

          @classmethod
          def from_pickleable_obj(cls, obj):
              self = cls(settings=obj["settings"])
              self.fs, self._fitted = obj["fs"], obj["fitted"]
              self.clf = unpack_pickleable(obj["clf"])
              return self

      Wrap nested estimators in :func:`~medusa.core.serialization.pack_pickleable` so a
      torch model can reload on any device. The base handles *how* the bundle reaches
      disk (a class tag, ``dill.HIGHEST_PROTOCOL``, and a polymorphic :meth:`load`). The
      subclass only decides *what* goes in the bundle.

    The **online** methods (:meth:`fit_online`, :meth:`predict_online`) and :meth:`reset`
    have safe defaults. Override them when the pipeline supports streaming.

    Output
    ------
    :meth:`predict` returns raw *scores* (an ndarray), not commands. For VEP spellers,
    a ``CommandDecoder`` maps scores to commands in a separate step. For endogenous
    paradigms, the scores are the final output. Store the output's degrees of freedom and
    its ``discrete`` / ``continuous`` meaning as pipeline metadata, not as a type.
    """

    def __init__(self, settings: "SettingsTree | dict | None" = None,
                 **overrides) -> None:
        super().__init__(settings, **overrides)
        self._fitted = False

    # ------------------------------------------------------------------
    # Contract -- offline (required)
    # ------------------------------------------------------------------
    @abstractmethod
    def fit(self, recordings: Iterable[Recording]) -> Self:
        """Train on an iterable of recordings and return ``self`` (sets ``self._fitted``).

        Call :meth:`check_consistency` on each recording before you use it. Iterate
        lazily, so only one raw recording sits in memory at a time (see
        :func:`load_recordings`).
        """

    @abstractmethod
    def predict(self, recording: Recording) -> NDArray:
        """Return the scores, one per segment, for one recording (offline inference)."""

    @abstractmethod
    def check_consistency(self, recording: Recording) -> None:
        """Check that this pipeline can use ``recording``; raise ``ValueError`` if not.

        This replaces the removed ``ConsistencyChecker`` DSL, one check per pipeline.
        Check the sampling rate, the required channels, the expected event columns, and
        so on. :meth:`fit` calls it on every recording. So if each recording matches the
        config, all recordings also match each other.
        """

    # ------------------------------------------------------------------
    # Contract -- online (optional; safe defaults)
    # ------------------------------------------------------------------
    def fit_online(self, signal: Signal, *, onsets: "NDArray | None" = None,
                   labels: "NDArray | None" = None) -> Self:
        """Calibrate or adapt from a live streaming window. Not implemented by default."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement online fitting (fit_online).")

    def predict_online(self, signal: Signal, *,
                       onsets: "NDArray | None" = None) -> NDArray:
        """Return scores for the current streaming window. Not implemented by default."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement online prediction "
            "(predict_online).")

    def reset(self) -> None:
        """Clear temporary online state, such as a causal filter. Does nothing by default."""

    # ------------------------------------------------------------------
    # Persistence: portable single-file bundle, class-tagged for polymorphic load
    # ------------------------------------------------------------------
    def save(self, path: str, protocol: "int | None" = None) -> None:
        """Save to a single dill file: the :meth:`to_pickleable_obj` bundle, tagged with the class.

        Uses ``dill.HIGHEST_PROTOCOL`` (protocol 0 makes arrays much larger). The tag
        lets :meth:`load` rebuild the right class without you naming it.
        """
        tagged = {"module_name": type(self).__module__,
                  "class_name": type(self).__name__,
                  "obj": self.to_pickleable_obj()}
        with open(path, "wb") as f:
            dill.dump(tagged, f,
                      protocol=dill.HIGHEST_PROTOCOL if protocol is None else protocol)

    @classmethod
    def load(cls, path: str) -> "DecodingPipeline":
        """Load a pipeline saved by :meth:`save`. This is **polymorphic**.

        ``DecodingPipeline.load(path)`` rebuilds the right pipeline without you naming
        it. If you call it on a subclass (``BWRLDAPipeline.load(path)``), it also checks
        that the file holds that class.
        """
        with open(path, "rb") as f:
            tagged = dill.load(f)
        module = importlib.import_module(tagged["module_name"])
        klass = getattr(module, tagged["class_name"])
        if cls is not DecodingPipeline and klass is not cls:
            raise TypeError(
                f"{path!r} holds a {klass.__name__}, not a {cls.__name__}; "
                f"use DecodingPipeline.load(...) for polymorphic loading.")
        return klass.from_pickleable_obj(tagged["obj"])


def load_recordings(paths: Iterable[str], *,
                    lazy: bool = True) -> "Iterator[Recording] | list[Recording]":
    """Load recordings from ``paths``. This is the lightweight stand-in for a ``Dataset``.

    ``lazy=True`` (the default) returns a generator that yields one :class:`Recording` at
    a time. Training over a large corpus then keeps only one raw recording in memory (peak
    RAM is about one recording plus the features gathered so far). ``lazy=False`` returns
    a list. Use it when you must go over the recordings more than once, for example in
    cross-validation (see :func:`leave_one_recording_out`).
    """
    if lazy:
        return (Recording.load(p) for p in paths)
    return [Recording.load(p) for p in paths]


def leave_one_recording_out(
        recordings: Sequence[Recording],
) -> "Iterator[tuple[list[Recording], Recording]]":
    """Yield ``(train, test)`` splits for leave-one-recording-out cross-validation.

    The recordings are copied into a list first, so the whole corpus stays in memory and
    each fold can reuse it. Prefer an eagerly-loaded corpus (``load_recordings(paths,
    lazy=False)``); a lazy generator would be fully realized here anyway.
    """
    recs = list(recordings)
    for i, test in enumerate(recs):
        yield recs[:i] + recs[i + 1:], test


def harmonize_channels(signal: Signal, channels: "list[str]") -> Signal:
    """Pick and reorder ``signal`` to match ``channels``.

    A thin wrapper over :meth:`Signal.pick <medusa.core.data.signal.Signal.pick>`. It
    turns the plain ``KeyError`` into a clear, useful message. It replaces the old
    ``Dataset.custom_operations_on_recordings`` channel harmonisation.

    Raises
    ------
    ValueError
        If any name in ``channels`` is not in ``signal.channel_set``.
    """
    missing = [c for c in channels if c not in signal.channel_set.labels]
    if missing:
        raise ValueError(
            f"signal is missing required channels {missing}; "
            f"available: {list(signal.channel_set.labels)}.")
    return signal.pick(channels)
