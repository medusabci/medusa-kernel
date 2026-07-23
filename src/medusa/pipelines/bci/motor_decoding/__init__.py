"""Motor decoding: ready-to-use pipelines for motor imagery and motor execution.

These pipelines are thin, opinionated workflows with sensible motor defaults, not a configurable
framework. The reusable parts -- frequency filters, Common Spatial Patterns, classifiers, deep
backbones -- live in :mod:`medusa.signal` / :mod:`medusa.ml`; a pipeline only wires them together
for a motor task and ships good defaults.

Motor trials are independent and labelled, so their timing and class live in ``Recording.events``
(see :mod:`~medusa.pipelines.bci.trial_events`) and there is no dedicated data class. Each pipeline
is a Layer-1 :class:`~medusa.pipelines.base.DecodingPipeline`: ``predict`` returns per-trial class
scores, and those scores are the output (there is no Layer-2 decoder).

* ``MICSPLDAPipeline`` -- CSP + regularised LDA (shallow, torch-free).
* ``MIEEGInceptionPipeline`` -- an EEG-Inception (v1/v2) convolutional classifier.
* ``MIEEGSymPipeline`` -- the hemisphere-symmetric EEGSym backbone.

Offline analysis (spectrograms, ERD/ERS, statistics) lives in the pure-NumPy
:mod:`~medusa.pipelines.bci.motor_decoding.analysis` module -- plain functions that turn trials
into arrays for plotting, separate from the decoding pipelines.

The two deep pipelines are torch-gated and re-exported **lazily** (PEP 562 ``__getattr__``), so
``import medusa.pipelines.bci.motor_decoding`` stays torch-free for headless installs.
"""
from medusa.pipelines.bci.motor_decoding import analysis
from medusa.pipelines.bci.motor_decoding.decoding import MICSPLDAPipeline

__all__ = [
    "analysis",
    "MICSPLDAPipeline",
    # deep, torch-gated; resolved lazily via __getattr__
    "MIEEGInceptionPipeline",
    "MIEEGSymPipeline",
]

_LAZY = {"MIEEGInceptionPipeline", "MIEEGSymPipeline"}


def __getattr__(name: str):
    """Resolve the torch-gated deep pipelines lazily (PEP 562), keeping this import torch-free."""
    if name in _LAZY:
        from medusa.pipelines.bci.motor_decoding import decoding
        return getattr(decoding, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}.")
