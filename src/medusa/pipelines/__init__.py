"""High-level analysis pipelines for medusa-kernel.

Pipelines assemble the kernel's signal-processing and machine-learning building
blocks into end-to-end, application-specific workflows. The shared foundations
(:class:`~medusa.pipelines.base.DecodingPipeline` and the loading helpers) live in
:mod:`medusa.pipelines.base`; the domain flows live in subpackages
(:mod:`~medusa.pipelines.bci`, and future non-BCI siblings such as ``sleep`` /
``anesthesia``).
"""

from medusa.pipelines.base import (
    DecodingPipeline,
    load_recordings,
    leave_one_recording_out,
    harmonize_channels,
)
from medusa.pipelines import bci

__all__ = [
    "DecodingPipeline",
    "load_recordings",
    "leave_one_recording_out",
    "harmonize_channels",
    "bci",
]
