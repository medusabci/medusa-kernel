"""High-level analysis pipelines for medusa-kernel.

Pipelines assemble the kernel's signal-processing and machine-learning building
blocks into end-to-end, paradigm-specific workflows. Currently this package
exposes :mod:`~medusa.pipelines.bci`, which bundles the data containers,
processing methods and models for the supported BCI paradigms (c-VEP, ERP,
motor imagery, neurofeedback and SSVEP spellers).
"""

from medusa.pipelines import bci

__all__ = [
    "bci",
]
