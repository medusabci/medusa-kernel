"""PyTorch + Lightning integration for medusa.ml (optional, torch-gated).

Everything under ``medusa.ml.torch`` needs PyTorch; the task/estimator layers
also need PyTorch-Lightning. medusa-kernel does **not** install them
automatically — their builds are coupled to the local CUDA / ROCm / MPS / CPU
stack and must be installed by the user to match the hardware. Importing this
subpackage triggers :func:`require_torch`, so a missing install fails early with
an actionable message instead of a bare ``ModuleNotFoundError`` from a submodule.

(The backbones stay pure-torch and import fine without Lightning; only the
``tasks`` / estimator layers call :func:`require_lightning`.)
"""
from __future__ import annotations

#: Versions validated for this medusa-kernel release (named in the install
#: errors below). Bump when the supported range changes.
RECOMMENDED_TORCH = "torch==2.12"
RECOMMENDED_LIGHTNING = "lightning==2.6"


class TorchNotInstalled(ImportError):
    """A torch-backed feature was used without PyTorch / Lightning installed."""


def require_torch():
    """Return the imported ``torch`` module, or raise :class:`TorchNotInstalled`."""
    try:
        import torch
    except ImportError as exc:
        raise TorchNotInstalled(
            "This feature needs PyTorch, which medusa-kernel does not install "
            "automatically (the build depends on your CUDA/ROCm/MPS/CPU setup). "
            f"Install the right build for your hardware (recommended "
            f"'{RECOMMENDED_TORCH}') from https://pytorch.org/get-started/."
        ) from exc
    return torch


def require_lightning():
    """Return ``lightning.pytorch`` (and ensure PyTorch is present too)."""
    require_torch()
    try:
        import lightning.pytorch as pl
    except ImportError as exc:
        raise TorchNotInstalled(
            "This feature needs PyTorch-Lightning, which medusa-kernel does not "
            "install automatically. Install it to match your PyTorch "
            f"(recommended '{RECOMMENDED_LIGHTNING}'): pip install "
            f"'{RECOMMENDED_LIGHTNING}'. See https://lightning.ai/."
        ) from exc
    return pl


# Fail fast on `import medusa.ml.torch[...]` when torch is absent, with the
# friendly message above instead of a bare ImportError from a submodule.
require_torch()

# Public API of the subpackage top level: the install-gating helpers defined
# here. The models themselves are NOT re-exported at this level on purpose —
# the backbones (``medusa.ml.torch_models.backbones``) import fine with only
# torch, while the estimators (``medusa.ml.torch_models.classification``)
# additionally call ``require_lightning()`` at import; re-exporting the latter
# here would force PyTorch-Lightning on anyone who only wants a backbone. Import
# models from their submodules, which carry their own ``__all__``.
__all__ = [
    "RECOMMENDED_TORCH",
    "RECOMMENDED_LIGHTNING",
    "TorchNotInstalled",
    "require_torch",
    "require_lightning",
]
