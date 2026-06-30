"""Legacy time-series base (1.x persistence compatibility).

This is the **compat** biosignal hierarchy, kept only so the 2.0 kernel can
*load* recordings written by Kernel 1.x:

    SerializableComponent  ->  Signal  ->  EEG / ECG / EMG / EOG / NIRS / ...

It is the 1.x ``BiosignalData`` (and its free-form ``CustomBiosignalData``,
re-exported here as :class:`CustomSignal`) under the post-K1 name. New code must
not build on it -- the 2.0 runtime model is the concrete
:class:`medusa.core.data.signal.Signal` (D2), driven by the BIDS-aligned
``ChannelSet`` (D1). The compat ``EEG``/``ECG``/... subclasses and the compat
:class:`~medusa.core.legacy.recording.Recording` import these bases; nothing in
``core/data/`` does.
"""

from abc import abstractmethod

import numpy as np

from medusa.core.serialization import SerializableComponent


class Signal(SerializableComponent):
    """Base class for every recorded time-series in MEDUSA.

    Provides the contract every modality shares: a sample rate, a time
    axis and an `[n_samples x n_channels]` data array, plus default
    serialization that round-trips numpy arrays and any nested
    `SerializableComponent` (e.g. a channel set).

    Parameters
    ----------
    times : array_like
        1D array of shape `[n_samples]` with the timestamp of every sample.
        Always coerced to ``np.ndarray``.
    signal : array_like
        2D array of shape `[n_samples x n_channels]` with the sampled data.
        Always coerced to ``np.ndarray``.
    fs : int or float
        Sampling rate in Hz.
    **kwargs
        Optional metadata attached as instance attributes. Subclasses
        typically capture a ``channel_set`` keyword here (or as an explicit
        parameter in their own ``__init__``).

    Notes
    -----
    All subclasses should call ``super().__init__(times, signal, fs, **kw)``
    so they inherit the structural validation and the default
    ``to_serializable_obj`` implementation. Override ``from_serializable_obj``
    when the modality stores nested serializable objects (typically a
    channel set) that need to be reconstructed before instantiation.
    """

    def __init__(self, times, signal, fs, **kwargs):
        self.times = np.asarray(times)
        self.signal = np.asarray(signal)
        self.fs = fs
        for key, value in kwargs.items():
            setattr(self, key, value)
        self._validate()

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------
    def _validate(self):
        """Structural sanity checks. Subclasses can extend or relax."""
        if self.signal.ndim != 2:
            raise ValueError(
                "`signal` must be a 2D array [n_samples x n_channels], "
                f"got shape {self.signal.shape}"
            )
        if self.times.ndim != 1:
            raise ValueError(
                "`times` must be a 1D array [n_samples], "
                f"got shape {self.times.shape}"
            )
        if self.times.shape[0] != self.signal.shape[0]:
            raise ValueError(
                "`times` and `signal` must share the first dimension "
                f"(got {self.times.shape[0]} vs {self.signal.shape[0]})"
            )

    # ------------------------------------------------------------------
    # SerializableComponent contract
    # ------------------------------------------------------------------
    def to_serializable_obj(self):
        """Default serialization.

        Numpy arrays are converted to lists, nested
        ``SerializableComponent`` instances are serialized via their own
        ``to_serializable_obj``, and everything else is passed through.
        """
        out = {}
        for key, value in self.__dict__.items():
            if isinstance(value, np.ndarray):
                out[key] = value.tolist()
            elif isinstance(value, SerializableComponent):
                out[key] = value.to_serializable_obj()
            else:
                out[key] = value
        return out

    @classmethod
    @abstractmethod
    def from_serializable_obj(cls, dict_data):
        """Reconstruct an instance from a serialized dict.

        Subclasses MUST implement this so that any nested serializable
        attribute (channel set, montage, ...) is reconstructed before
        the instance is built.
        """


class CustomSignal(Signal):
    """Free-form legacy biosignal (1.x ``CustomBiosignalData``).

    A catch-all for 1.x streams that do not map to a specific modality class:
    it stores whatever keyword arguments it is given and performs **no**
    structural validation (so it does not require the ``times``/``signal``/``fs``
    triple of the base :class:`Signal`). Used by
    :class:`~medusa.core.legacy.recording.Recording` to load custom biosignals
    and to flag them with a warning.
    """

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        # No _validate(): custom signals carry arbitrary, unchecked attributes.
        # (Base Signal.to_serializable_obj is inherited unchanged.)

    @classmethod
    def from_serializable_obj(cls, dict_data):
        return cls(**dict_data)