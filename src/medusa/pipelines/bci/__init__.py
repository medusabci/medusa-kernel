"""Brain-Computer Interface (BCI) paradigm pipelines.

Organised by *signal-generation mode* -- the axis that mirrors the decoding stack:

- :mod:`~medusa.pipelines.bci.vep_spellers` -- visual-evoked-potential spellers, i.e.
  stimulus/event-locked paradigms (c-VEP, SSVEP, ERP/P300): a stimulation codebook plus the
  Layer-2 ``VEPCommandDecoder``.
- :mod:`~medusa.pipelines.bci.endogenous` -- self-generated paradigms (motor imagery,
  neurofeedback): Layer-1 decoding pipelines only (no codebook, no command decoder).

Paradigm-agnostic performance metrics live in
:mod:`~medusa.pipelines.bci.performance` (``itr`` is re-exported for convenience).
"""

from medusa.pipelines.bci import vep_spellers, endogenous, performance
from medusa.pipelines.bci.performance import itr

__all__ = [
    "vep_spellers",
    "endogenous",
    "performance",
    "itr",
]
