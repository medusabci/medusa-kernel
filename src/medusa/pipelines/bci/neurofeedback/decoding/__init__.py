"""Neurofeedback Layer-1 pipelines (continuous, baseline-referenced feature feedback).

Several pipelines, so decoding is a package with one short module per pipeline plus a shared
``_common`` helper module. Each is a direct
:class:`~medusa.pipelines.base.DecodingPipeline` whose ``fit`` is an **unsupervised calibration**
(the baseline is the mean feature over the calibration windows) and whose ``predict`` streams a
continuous ``(n_windows,)`` feedback trace. There are no labels, no classifier and no Layer-2
decoder -- the trace is the output. Both pipelines are torch-free.

* :mod:`~medusa.pipelines.bci.neurofeedback.decoding.power` -- ``PowerNFTPipeline``: band power
  or band ratio.
* :mod:`~medusa.pipelines.bci.neurofeedback.decoding.connectivity` -- ``ConnectivityNFTPipeline``:
  a functional-connectivity graph metric (wPLI / AEC).
"""
from __future__ import annotations

from medusa.pipelines.bci.neurofeedback.decoding.power import PowerNFTPipeline
from medusa.pipelines.bci.neurofeedback.decoding.connectivity import ConnectivityNFTPipeline

__all__ = ["PowerNFTPipeline", "ConnectivityNFTPipeline"]
