"""Headless import smoke tests (K3).

Verify that the core surface of `medusa-kernel` is fully importable without
*loading* PySide6 and without a user-installed PyTorch.

Contract:

- The non-visual core (``medusa.core`` / ``medusa.signal`` / ``medusa.ml`` /
  ``medusa.graph``) must load WITHOUT importing PySide6, so a headless DSP /
  server install never pulls Qt. ``medusa.plots`` is exempt: it applies the
  light medusa-style theme on import, which initializes the theme manager and
  loads ``PySide6.QtCore`` (PySide6 is a core dependency) — the same QtCore
  import the first plot draw would trigger anyway. Only ``medusa.widgets`` pulls
  the heavy GUI modules (QtWidgets/QtGui).
- ``medusa.ml`` must import without ``torch``; the torch-only backends
  (``medusa.ml.torch_models``) import ``torch`` lazily and are out of scope for
  this smoke. PyTorch is *not* shipped as an extra — users install the right
  CUDA/ROCm/CPU build themselves (https://pytorch.org/get-started/).
- ``medusa.pipelines.*`` is excluded (a separate, deferred migration).

These tests will run in CI on the no-extras job (K3 / K7).
"""

from __future__ import annotations

import importlib

import pytest

# Core subpackages that must import WITHOUT pulling PySide6 (or torch) at module
# load time. PySide6 is now a core dependency (so it is installed), but it is
# heavy: only ``medusa.widgets`` should import it. ``medusa.pipelines.*`` is
# intentionally excluded (a separate, deferred BCI/ChannelSet migration).
HEADLESS_MODULES: list[str] = [
    "medusa",
    "medusa.core",
    "medusa.core.serialization",
    "medusa.core.data.recording",
    "medusa.core.data.signal",
    "medusa.core.legacy.experiment",
    "medusa.core.legacy.biosignals.eeg.eeg",
    "medusa.core.legacy.biosignals.ecg.ecg",
    "medusa.core.legacy.biosignals.eog.eog",
    "medusa.core.legacy.biosignals.emg.emg",
    "medusa.core.legacy.biosignals.nirs.nirs",
    "medusa.core.settings_tree",
    "medusa.signal",
    "medusa.signal.frequency_filtering",
    "medusa.signal.spatial_filtering",
    "medusa.signal.artifact_removal",
    "medusa.signal.segmentation",
    "medusa.signal.transforms",
    "medusa.signal.generators",
    "medusa.signal.orthogonalization",
    "medusa.signal.metrics",
    "medusa.signal.metrics.spectral",
    "medusa.signal.metrics.nonlinear",
    "medusa.signal.metrics.discriminability",
    "medusa.signal.metrics.connectivity",
    "medusa.graph",
    "medusa.plots",
    "medusa.ml",
]


@pytest.mark.parametrize("modname", HEADLESS_MODULES)
def test_headless_import(modname: str) -> None:
    """Each headless module imports without raising."""
    importlib.import_module(modname)


def test_no_pyside6_in_headless_modules() -> None:
    """After importing the NON-VISUAL headless surface in a fresh subprocess,
    ``PySide6`` must NOT be in ``sys.modules`` — a headless DSP / server / daemon
    using ``medusa.core`` / ``medusa.signal`` / ``medusa.ml`` must never pull Qt.

    ``medusa.plots`` is deliberately excluded: it applies the active medusa-style
    theme on import (the kernel defaults to light), which initializes the
    medusa-style theme manager — and that builds its ``theme_changed`` Qt
    ``Signal`` from ``PySide6.QtCore`` whenever PySide6 is installed (PySide6 is
    now a core dependency). The same QtCore import would happen on the first
    plot draw via ``current_theme()``, so loading it on the viz layer is
    expected; only the heavy GUI modules (QtWidgets/QtGui) stay out.

    A subprocess is required because sibling tests in the same pytest session
    legitimately instantiate Qt widgets, polluting the parent ``sys.modules``.
    """
    import subprocess
    import sys
    import textwrap

    non_visual = [m for m in HEADLESS_MODULES if m != "medusa.plots"]
    script = textwrap.dedent(
        f"""
        import importlib, sys, json
        for m in {non_visual!r}:
            importlib.import_module(m)
        leaked = sorted(m for m in sys.modules
                        if m == 'PySide6' or m.startswith('PySide6.'))
        print(json.dumps(leaked))
        """
    )
    # NOTE: feed the script via stdin instead of ``-c`` to side-step a
    # CPython 3.13.1 regression in ``linecache._register_code`` ("'str'
    # object has no attribute 'co_consts'") triggered by ``python -c``.
    proc = subprocess.run(
        [sys.executable, "-"],
        input=script,
        capture_output=True, text=True, check=True,
    )
    import json
    leaked = json.loads(proc.stdout.strip().splitlines()[-1])
    assert not leaked, (
        "PySide6 leaked into the headless import surface via: "
        f"{leaked[:5]}{'...' if len(leaked) > 5 else ''}. "
        "Move the offending Qt code under medusa.widgets."
    )



