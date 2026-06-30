"""medusa-kernel: the signal-processing core of the MEDUSA© ecosystem.

This top-level package is intentionally minimal and headless-safe: importing
``medusa`` pulls in no Qt (PySide6) or torch. Reach for the functionality you
need from the subpackages directly — ``medusa.signal`` (filters, transforms,
segmentation, metrics), ``medusa.core`` (runtime data model, persistence,
pipelines), ``medusa.graph`` (graph-theoretic metrics), ``medusa.ml``
(machine-learning utilities and PyTorch models), ``medusa.plots`` (matplotlib
visualisations), ``medusa.pipelines`` (high-level BCI paradigms) and
``medusa.widgets`` (Qt-based interactive tools).
"""

from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _version("medusa-kernel")
except PackageNotFoundError:  # running from a source tree w/o install metadata
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
