"""EEG-specific data utilities: standard montages, coordinate readers and
coordinate-system conversions for building EEG channel sets.

This is the per-modality home the generic ``core/data/channels.py`` delegates to
(it never hard-codes montage/coordinate knowledge). Other modalities (EMG, NIRS,
...) will get sibling modules under ``core/data/``.
"""

from medusa.core.data.eeg.montages import (
    get_standard_montage,
    montage_10_20,
    montage_10_10,
    montage_10_05,
    get_standard_montage_labels,
    get_coordinates,
    resolve_label,
    project_to_2d,
    switch_coordinates_system,
    read_montage_file,
    get_camel_case_labels,
    SYNONYMS,
    REFERENCE_ELECTRODES,
)

__all__ = [
    "get_standard_montage",
    "montage_10_20",
    "montage_10_10",
    "montage_10_05",
    "get_standard_montage_labels",
    "get_coordinates",
    "resolve_label",
    "project_to_2d",
    "switch_coordinates_system",
    "read_montage_file",
    "get_camel_case_labels",
    "SYNONYMS",
    "REFERENCE_ELECTRODES",
]
