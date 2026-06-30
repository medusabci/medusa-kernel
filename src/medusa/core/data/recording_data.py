"""Abstract base for the modalities a
:class:`~medusa.core.data.recording.Recording` holds in its ``data`` dict.

Adds an optional ``bids_datatype()`` self-description hook on top of the
:class:`~medusa.core.serialization.SerializableComponent` contract. The only
concrete type is :class:`~medusa.core.data.signal.Signal`; other modalities
(imaging volumes, phenotype tables, ...) subclass ``RecordingData``. It lives in
its own module so :mod:`signal` can subclass it without a circular import with
:mod:`recording`.
"""

from medusa.core.serialization import SerializableComponent

__all__ = ["RecordingData"]


class RecordingData(SerializableComponent):
    """Base class for any recorded modality stored in ``Recording.data``."""

    def bids_datatype(self) -> "str | None":
        """The BIDS datatype this entry maps to (``'eeg'``, ``'meg'``, ...).

        Returns ``None`` when the modality does not imply a datatype. Subclasses
        override; the default is ``None``.
        """
        return None
