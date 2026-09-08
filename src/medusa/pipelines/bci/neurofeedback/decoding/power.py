"""Band-power neurofeedback pipeline.

Holds :class:`PowerNFTPipeline`. It slides a window over the signal, computes band power in a
training band (or a band ratio, e.g. theta/beta for attention training), and streams it relative
to a calibration baseline. Layer-1 only: there is no classifier, no labels, and no command
decoder -- the continuous feedback trace is the output.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from medusa.core.settings_tree import SettingsTree
from medusa.core.data.recording import Recording
from medusa.signal.transforms import power_spectral_density
from medusa.signal.metrics.spectral import band_power

from medusa.pipelines.base import DecodingPipeline
from medusa.pipelines.bci.neurofeedback.decoding._common import (
    add_band_filter_settings, add_windowing_settings, add_reference_settings,
    apply_reference, calibrate_baseline, check_signal, feature_trace)

__all__ = ["PowerNFTPipeline"]


class PowerNFTPipeline(DecodingPipeline):
    """Band-power (or band-ratio) neurofeedback.

    ``fit`` is the **calibration**: it computes the baseline as the mean band power over the
    windows of the calibration (rest) recordings -- unsupervised, no labels. ``predict`` slides
    the same window over a recording and returns the continuous ``(n_windows,)`` feedback trace,
    expressed relative to that baseline (see ``reference``). This is a Layer-1
    :class:`~medusa.pipelines.base.DecodingPipeline`; the trace is the output.

    ``mode='single'`` streams the power in ``band``; ``mode='ratio'`` streams
    ``band / band2`` (for example theta/beta). A broad ``filter`` cleans the signal first; the
    band selection itself is done on the PSD. Configure it through the live
    :attr:`~medusa.core.settings_tree.Configurable.settings` tree or with nested construction
    kwargs, for example ``PowerNFTPipeline(channels=["C3", "C4"], band=[8.0, 12.0])``.
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
        add_band_filter_settings(s, "filter", [1.0, 45.0], 5,
                                 "Broad preprocessing band-pass (drift / line-noise cleanup)")
        s.add_item("mode", value="single", value_options=["single", "ratio"],
                   info="'single' = power in 'band'; 'ratio' = band / band2")
        s.add_item("band", value=[8.0, 12.0], info="Training band (Hz)")
        s.add_item("band2", value=[13.0, 30.0], info="Denominator band for ratio mode (Hz)")
        s.add_item("power_type", value="absolute", value_options=["absolute", "relative"],
                   info="Absolute band power, or power relative to the whole spectrum")
        add_windowing_settings(s)
        add_reference_settings(s)
        return s

    def check_consistency(self, recording: Recording) -> None:
        """Check the signal, channels and ``fs`` (neurofeedback needs no events)."""
        self.fs = check_signal(recording, self.cfg, self.fs)

    def _window_feature(self, window: NDArray, fs: float, cfg: dict) -> float:
        """Band power (or band ratio) of one window, averaged over channels."""
        # Pass the window as 3-D (1 segment) so the PSD is (1, n_freq, n_cha) -- the shape
        # band_power expects -- and no auto-promotion warning fires.
        _, psd = power_spectral_density(window[None], fs)
        p1 = float(np.mean(band_power(psd, fs, tuple(cfg["band"]),
                                      power_type=cfg["power_type"])))
        if cfg["mode"] == "ratio":
            p2 = float(np.mean(band_power(psd, fs, tuple(cfg["band2"]),
                                          power_type=cfg["power_type"])))
            return p1 / p2
        return p1

    def fit(self, recordings) -> "PowerNFTPipeline":
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

    def restart(self) -> "PowerNFTPipeline":
        """Forget the calibration baseline; the next ``fit`` starts over."""
        self.baseline = None
        return super().restart()

    def to_pickleable_obj(self) -> dict:
        """Bundle the settings, the fitted flag, ``fs`` and the calibration baseline."""
        return {"settings": self.settings.to_dict(), "fitted": self._fitted, "fs": self.fs,
                "baseline": self.baseline}

    @classmethod
    def from_pickleable_obj(cls, obj: dict) -> "PowerNFTPipeline":
        """Rebuild the pipeline from a bundle made by :meth:`to_pickleable_obj`."""
        self = cls(settings=obj["settings"])
        self.fs, self._fitted, self.baseline = obj["fs"], obj["fitted"], obj["baseline"]
        return self
