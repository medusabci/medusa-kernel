"""Motor-decoding Layer-1 pipelines (motor imagery, motor execution).

Several pipelines, so decoding is a package with one short module per pipeline plus a shared
``_common`` helper module. Each pipeline is a **direct**
:class:`~medusa.pipelines.base.DecodingPipeline` (their feature paths share nothing, so there is
no intermediate base). Every pipeline reads labelled trials from
:mod:`~medusa.pipelines.bci.trial_events` and returns per-trial class scores; there is no
Layer-2 decoder, so the scores are the output.

* :mod:`~medusa.pipelines.bci.motor_decoding.decoding.mi_csp_lda` -- ``MICSPLDAPipeline``:
  Common Spatial Patterns + regularised LDA (shallow, torch-free).
* :mod:`~medusa.pipelines.bci.motor_decoding.decoding.mi_eeg_inception` --
  ``MIEEGInceptionPipeline``: an EEG-Inception (v1/v2) convolutional classifier.
* :mod:`~medusa.pipelines.bci.motor_decoding.decoding.mi_eegsym` -- ``MIEEGSymPipeline``: the
  hemisphere-symmetric EEGSym backbone.

The two deep pipelines are **torch-gated**: they import torch through their modules, so they are
re-exported **lazily** here (PEP 562 ``__getattr__``). Importing this package stays torch-free for
headless installs; the deep classes load only when accessed.
"""
from __future__ import annotations

from medusa.pipelines.bci.motor_decoding.decoding.mi_csp_lda import MICSPLDAPipeline

__all__ = [
    "MICSPLDAPipeline",
    # deep, torch-gated; resolved lazily via __getattr__
    "MIEEGInceptionPipeline",
    "MIEEGSymPipeline",
]

#: Torch-gated deep pipelines -> the module they live in (imported on first access only).
_LAZY = {
    "MIEEGInceptionPipeline": "mi_eeg_inception",
    "MIEEGSymPipeline": "mi_eegsym",
}


def __getattr__(name: str):
    """Resolve the torch-gated deep pipelines lazily (PEP 562), keeping the package torch-free."""
    if name in _LAZY:
        import importlib
        module = importlib.import_module(
            f"medusa.pipelines.bci.motor_decoding.decoding.{_LAZY[name]}")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}.")
