"""medusa-kernel core: serialization, configuration and the runtime data model.

This package gathers the foundational building blocks the rest of medusa-kernel
is built on and that consumers reach for most often:

* the persistence base classes
  (:class:`~medusa.core.serialization.SerializableComponent`,
  :class:`~medusa.core.serialization.PickleableComponent`),
* the typed, Qt-free configuration schema
  (:class:`~medusa.core.settings_tree.SettingsTree`),
* the on-disk recording schema version and its validator
  (:data:`~medusa.core.schema.SCHEMA_VERSION`,
  :func:`~medusa.core.schema.validate_recording_dict`),
* the profiling decorator (:func:`~medusa.core.profiling.perf_analysis`) and the
  signal dimension checker (:func:`~medusa.core.utils.check_data_dims`),
* and the BIDS-aligned runtime data types, re-exported from
  :mod:`medusa.core.data` (:class:`Recording`, :class:`Signal`, ...).

Importing this package is headless-safe: it pulls in no Qt (PySide6) or torch.
"""

from medusa.core.serialization import (
    SerializableComponent,
    PickleableComponent,
    pack_pickleable,
    unpack_pickleable,
)
from medusa.core.settings_tree import SettingsTree
from medusa.core.schema import SCHEMA_VERSION, validate_recording_dict
from medusa.core.profiling import perf_analysis
from medusa.core.utils import check_data_dims
from medusa.core.data import (
    Recording,
    BidsInfo,
    RecordingData,
    Signal,
    Events,
    ChannelSet,
    Channel,
    Sensor,
    Recorder,
)

__all__ = [
    # Serialization base classes
    "SerializableComponent",
    "PickleableComponent",
    "pack_pickleable",
    "unpack_pickleable",
    # Configuration schema
    "SettingsTree",
    # On-disk recording schema
    "SCHEMA_VERSION",
    "validate_recording_dict",
    # Profiling / signal utilities
    "perf_analysis",
    "check_data_dims",
    # Runtime data model (re-exported from medusa.core.data)
    "Recording",
    "BidsInfo",
    "RecordingData",
    "Signal",
    "Events",
    "ChannelSet",
    "Channel",
    "Sensor",
    "Recorder",
]
