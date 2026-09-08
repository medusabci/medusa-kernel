"""Functional-connectivity neurofeedback pipeline.

Holds :class:`ConnectivityNFTPipeline`. It slides a window over the signal (band-passed to a
training band), builds a connectivity adjacency matrix (weighted phase-lag index or
amplitude-envelope correlation), reduces it to one scalar (global coupling or node strength),
and streams that relative to a calibration baseline. Layer-1 only, no labels, no classifier.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from medusa.core.settings_tree import SettingsTree
from medusa.core.data.recording import Recording
from medusa.signal.metrics.connectivity import wpli, aec
from medusa.graph.degree import degree

from medusa.pipelines.base import DecodingPipeline
from medusa.pipelines.bci.neurofeedback.decoding._common import (
    add_band_filter_settings, add_windowing_settings, add_reference_settings,
    apply_reference, calibrate_baseline, check_signal, feature_trace)

__all__ = ["ConnectivityNFTPipeline"]

#: Connectivity measures -> the adjacency function that builds the matrix.
_MEASURES = {"wpli": lambda w: wpli(w), "aec": lambda w: aec(w, ort=True)}


class ConnectivityNFTPipeline(DecodingPipeline):
    """Functional-connectivity neurofeedback.

    ``fit`` is the **calibration**: it computes the baseline as the mean connectivity feature
    over the windows of the calibration recordings -- unsupervised, no labels. ``predict`` slides
    the same window over a recording and returns the continuous ``(n_windows,)`` feedback trace,
    expressed relative to that baseline (see ``reference``). This is a Layer-1
    :class:`~medusa.pipelines.base.DecodingPipeline`; the trace is the output.

    Two choices shape the feature: the connectivity ``measure`` (``'wpli'`` weighted phase-lag
    index, or ``'aec'`` orthogonalised amplitude-envelope correlation) and the graph ``feature``
    (``'global_coupling'`` = the mean off-diagonal connectivity, or ``'strength'`` = the mean node
    degree). The ``filter`` band-passes to the training band before the connectivity is computed
    (connectivity is band-specific). Configure it through the live
    :attr:`~medusa.core.settings_tree.Configurable.settings` tree or with nested construction
    kwargs, for example ``ConnectivityNFTPipeline(channels=[...], measure="aec")``.
    """

    fs = None           # sampling rate adopted at fit (fitted state)
    baseline = None     # calibration baseline (mean feature over the calibration windows)

    @classmethod
    def default_settings(cls) -> SettingsTree:
        """A GUI-editable, levelled schema of the pipeline's configuration."""
        s = SettingsTree()
        s.add_item("channels", value=[], info="Channels to compute the feedback over (required)")
        s.add_item("signal_key", value="eeg", info="Recording stream key to use")
        s.add_item("car", value=True, info="Common-average reference before filtering")
        add_band_filter_settings(s, "filter", [8.0, 12.0], 5,
                                 "Training band-pass (connectivity is band-specific)")
        s.add_item("measure", value="wpli", value_options=list(_MEASURES),
                   info="Connectivity measure (wpli phase-lag, or aec envelope correlation)")
        s.add_item("feature", value="global_coupling",
                   value_options=["global_coupling", "strength"],
                   info="Graph reduction: mean off-diagonal coupling, or mean node degree")
        add_windowing_settings(s)
        add_reference_settings(s)
        return s

    def check_consistency(self, recording: Recording) -> None:
        """Check the signal, channels and ``fs`` (neurofeedback needs no events)."""
        self.fs = check_signal(recording, self.cfg, self.fs)

    def _window_feature(self, window: NDArray, fs: float, cfg: dict) -> float:
        """A single connectivity scalar for one window: build the adjacency, reduce to a graph metric."""
        # Pass the window as 3-D (1 epoch) -- the documented shape -- and take the single
        # epoch's ``(n_cha, n_cha)`` adjacency; no auto-promotion warning fires.
        adjacency = _MEASURES[cfg["measure"]](window[None])[0]
        if cfg["feature"] == "strength":
            return float(np.mean(degree(adjacency)))
        # global coupling: the mean of the lower-triangular (off-diagonal) connectivity.
        lower = adjacency[np.tril_indices(adjacency.shape[0], k=-1)]
        return float(np.nanmean(lower))

    def fit(self, recordings) -> "ConnectivityNFTPipeline":
        """Calibrate: set the baseline to the mean feature over the calibration windows."""
        self._check_settings()
        self.baseline = calibrate_baseline(
            recordings, check_consistency=self.check_consistency, cfg=self.cfg,
            window_feature=self._window_feature)
        self._fitted = True
        return self

    def predict(self, recording: Recording) -> NDArray:
        """The continuous ``(n_windows,)`` feedback trace, referenced to the baseline."""
        if not self._fitted:
            raise RuntimeError("pipeline is not fitted (calibrated); call fit() first.")
        self.check_consistency(recording)
        trace = feature_trace(recording, self.cfg, self._window_feature)
        return apply_reference(trace, self.baseline, self.cfg["reference"])

    def restart(self) -> "ConnectivityNFTPipeline":
        """Forget the calibration baseline; the next ``fit`` starts over."""
        self.baseline = None
        return super().restart()

    def to_pickleable_obj(self) -> dict:
        """Bundle the settings, the fitted flag, ``fs`` and the calibration baseline."""
        return {"settings": self.settings.to_dict(), "fitted": self._fitted, "fs": self.fs,
                "baseline": self.baseline}

    @classmethod
    def from_pickleable_obj(cls, obj: dict) -> "ConnectivityNFTPipeline":
        """Rebuild the pipeline from a bundle made by :meth:`to_pickleable_obj`."""
        self = cls(settings=obj["settings"])
        self.fs, self._fitted, self.baseline = obj["fs"], obj["fitted"], obj["baseline"]
        return self
