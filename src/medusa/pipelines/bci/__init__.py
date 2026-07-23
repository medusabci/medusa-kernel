"""Brain-Computer Interface (BCI) paradigm pipelines.

Organised by *application domain* -- each package is a decoding strategy named for what it
is, and holds thin, ready-to-use pipelines with sensible defaults (the reusable algorithms
live below, in :mod:`medusa.signal` / :mod:`medusa.ml`):

- :mod:`~medusa.pipelines.bci.vep_spellers` -- visual-evoked-potential spellers (c-VEP,
  SSVEP, ERP/P300): a stimulation codebook plus the Layer-2 ``VEPCommandDecoder``.
- :mod:`~medusa.pipelines.bci.motor_decoding` -- motor imagery and motor execution: Layer-1
  trial-classification pipelines (the scores are the output; no command decoder).
- :mod:`~medusa.pipelines.bci.neurofeedback` -- continuous, baseline-referenced feature feedback
  (band power, connectivity): Layer-1 only, calibrated (not labelled), streaming a feedback trace.

Trial-based paradigms share one event contract, :mod:`~medusa.pipelines.bci.trial_events`
(one labelled row per trial), instead of a data class. Paradigm-agnostic performance metrics
live in :mod:`~medusa.pipelines.bci.performance` (``itr`` is re-exported for convenience).
"""

from medusa.pipelines.bci import (
    vep_spellers, motor_decoding, neurofeedback, trial_events, performance)
from medusa.pipelines.bci.performance import itr

__all__ = [
    "vep_spellers",
    "motor_decoding",
    "neurofeedback",
    "trial_events",
    "performance",
    "itr",
]
