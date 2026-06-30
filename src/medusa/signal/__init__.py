"""Signal-processing operations for biosignals (flat namespace).

The medusa-kernel signal toolbox exposes its operations *flat at the root* of
this package: frequency- and spatial-domain filtering, segmentation, artifact
removal, time-frequency transforms, orthogonalization and synthetic signal
generators. Metric families (spectral, nonlinear, discriminability,
connectivity) live under :mod:`medusa.signal.metrics`.

All operations follow the medusa-kernel signal-shape contract: continuous
signals are ``(n_samples, n_channels)`` (the ``'time'`` representation).
"""

from medusa.signal.artifact_removal import (
    ICA,
    ArtifactRegression,
    ICAData,
    reject_noisy_segments,
)
from medusa.signal.frequency_filtering import (
    FIRFilter,
    IIRFilter,
    display_filter,
)
from medusa.signal.generators import (
    ChirpSignalGenerator,
    ECGSignalGenerator,
    EEGSignalGenerator,
    EOGSignalGenerator,
    ImpulseSignalGenerator,
    RampSignalGenerator,
    SawtoothSignalGenerator,
    SignalGenerator,
    SinusoidalSignalGenerator,
    SquareSignalGenerator,
)
from medusa.signal.orthogonalization import signal_orthogonalization
from medusa.signal.segmentation import (
    EventSegmentFeasibility,
    check_event_segments_feasibility,
    normalize_segments,
    resample_segments,
    segment_signal,
    segment_signal_around_events,
    times_to_sample_indices,
)
from medusa.signal.spatial_filtering import (
    CCA,
    CSP,
    TRCA,
    LaplacianFilter,
    car,
)
from medusa.signal.transforms import (
    cross_cwt,
    cwt_spectrogram,
    fourier_spectrogram,
    hilbert,
    normalize_psd,
    power_spectral_density,
)

__all__ = [
    # Artifact removal
    "reject_noisy_segments",
    "ICA",
    "ICAData",
    "ArtifactRegression",
    # Frequency-domain filtering
    "FIRFilter",
    "IIRFilter",
    "display_filter",
    # Orthogonalization
    "signal_orthogonalization",
    # Segmentation
    "segment_signal",
    "segment_signal_around_events",
    "normalize_segments",
    "resample_segments",
    "EventSegmentFeasibility",
    "check_event_segments_feasibility",
    "times_to_sample_indices",
    # Spatial filtering
    "LaplacianFilter",
    "car",
    "CSP",
    "CCA",
    "TRCA",
    # Transforms (time-frequency)
    "hilbert",
    "power_spectral_density",
    "normalize_psd",
    "fourier_spectrogram",
    "cwt_spectrogram",
    "cross_cwt",
    # Synthetic generators
    "SignalGenerator",
    "SinusoidalSignalGenerator",
    "SquareSignalGenerator",
    "SawtoothSignalGenerator",
    "RampSignalGenerator",
    "ImpulseSignalGenerator",
    "ChirpSignalGenerator",
    "EEGSignalGenerator",
    "ECGSignalGenerator",
    "EOGSignalGenerator",
]
