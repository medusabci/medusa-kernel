"""High-level analysis pipelines for medusa-kernel.

Pipelines assemble the kernel's signal-processing and machine-learning building
blocks into end-to-end, application-specific workflows. The shared foundations
(:class:`~medusa.pipelines.base.DecodingPipeline` and the loading helpers) live in
:mod:`medusa.pipelines.base`, with the torch-backed lifecycle
(:class:`~medusa.pipelines.torch_base.TorchPipeline`) beside it in
:mod:`medusa.pipelines.torch_base`; the domain flows live in subpackages
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
    "TorchPipeline",   # torch-gated; resolved lazily via __getattr__
    "load_recordings",
    "leave_one_recording_out",
    "harmonize_channels",
    "bci",
]

#: Torch-gated: :mod:`medusa.pipelines.torch_base` pulls PyTorch, so it is resolved on
#: first access by ``__getattr__`` below and importing this package stays torch-free.
_TORCH_GATED = ("TorchPipeline",)


def __getattr__(name: str):
    """Resolve the torch-backed base lazily (PEP 562), so a headless install still works."""
    if name in _TORCH_GATED:
        from medusa.pipelines import torch_base
        return getattr(torch_base, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}.")
