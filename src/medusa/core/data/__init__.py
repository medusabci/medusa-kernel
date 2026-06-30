"""MEDUSA runtime data types (BIDS-aligned).

A :class:`~medusa.core.data.recording.Recording` groups everything captured in
one run: typed-channel :class:`~medusa.core.data.signal.Signal` streams, a BIDS
:class:`~medusa.core.data.events.Events` timeline and metadata. Channels and
sensors are described by :class:`~medusa.core.data.channels.ChannelSet`, and the
:class:`~medusa.core.data.streaming.Recorder` handles live acquisition and HDF5
persistence.
"""

from medusa.core.data.channels import (
    Channel,
    Sensor,
    ChannelSet,
    BIDS_CHANNEL_TYPES,
    REFERENCE_METHODS,
)
from medusa.core.data.recording_data import RecordingData
from medusa.core.data.signal import Signal
from medusa.core.data.events import Events
from medusa.core.data.recording import BidsInfo, Recording
from medusa.core.data.streaming import Recorder

__all__ = [
    "Channel",
    "Sensor",
    "ChannelSet",
    "BIDS_CHANNEL_TYPES",
    "REFERENCE_METHODS",
    "RecordingData",
    "Signal",
    "Events",
    "BidsInfo",
    "Recording",
    "Recorder",
]
