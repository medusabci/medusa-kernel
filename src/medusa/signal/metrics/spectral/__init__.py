"""Spectral signal metrics — *what frequencies dominate?*
Signal -> scalar/vector extractions computed from a power spectral density
(or directly from the signal). For PSD computation itself see
``medusa.signal.transforms`` (transform: signal -> spectrum), not this
package (metric: signal -> scalar).
"""
from medusa.signal.metrics.spectral.band_power import band_power
from medusa.signal.metrics.spectral.median_frequency import median_frequency
from medusa.signal.metrics.spectral.spectral_edge_frequency import (
    spectral_edge_frequency,
)

__all__ = [
    "band_power",
    "median_frequency",
    "spectral_edge_frequency",
]
