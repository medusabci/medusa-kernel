"""Neurofeedback: continuous baseline-referenced feature feedback.

Neurofeedback is not trial decoding: there are no labelled events. A pipeline slides a window
over the ongoing signal, computes one feature per window (band power, a connectivity metric), and
streams it relative to a calibration baseline. So ``fit`` is an unsupervised calibration (it sets
the baseline from a rest recording), and ``predict`` returns the continuous feedback trace. These
are Layer-1 :class:`~medusa.pipelines.base.DecodingPipeline`\\ s only -- no labels, no classifier,
no command decoder.

* ``PowerNFTPipeline`` -- band power, or a band ratio (for example theta/beta).
* ``ConnectivityNFTPipeline`` -- a functional-connectivity graph metric (wPLI / AEC).

The reusable parts (filters, PSD, connectivity measures, graph metrics) live in
:mod:`medusa.signal` / :mod:`medusa.graph`; a pipeline only wires them together with sensible
defaults.
"""
from medusa.pipelines.bci.neurofeedback.decoding import (
    PowerNFTPipeline, ConnectivityNFTPipeline)

__all__ = ["PowerNFTPipeline", "ConnectivityNFTPipeline"]
