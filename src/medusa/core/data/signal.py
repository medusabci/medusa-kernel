"""Concrete, mixed-channel time-series container.

A :class:`Signal` is one acquisition stream: one device, one clock, one sampling
rate ``fs`` and one ``[n_samples x n_channels]`` matrix whose columns are typed
(EEG + EOG + EMG + TRIG ... in a single stream) through a
:class:`~medusa.core.data.channels.ChannelSet`. Modality is a per-channel property
(BIDS ``channels.tsv`` ``type``); unknown channels coerce to ``OTHER``/``MISC``.

A ``Signal`` carries no BIDS sidecar metadata -- the per-stream
``_<datatype>.json`` lives in ``Recording.sidecars``. Processing stays in free
functions over arrays; this container only structures data and offers channel
selection (:meth:`Signal.pick_types` / :meth:`Signal.pick`).
"""

from collections import Counter

import numpy as np
from numpy.typing import ArrayLike, NDArray

from medusa.core.serialization import SerializableComponent
from medusa.core.data.recording_data import RecordingData
from medusa.core.data._bids import datatype_for_channel_type
from medusa.core.data.channels import ChannelSet

__all__ = ["Signal"]


# Explicitly-modelled attributes; anything else passed at construction is a
# free-form metadata *extra* (stored flat on the instance and round-tripped).
# Single source of truth for the core/extra split (mirrors channels.py).
_SIGNAL_CORE = ("signal", "fs", "channel_set", "times")


class Signal(RecordingData):
    """One acquisition stream: a typed ``[n_samples x n_channels]`` matrix.

    Modality-agnostic: the per-column type lives in ``channel_set`` (channel ``i``
    <-> ``signal[:, i]``), so a single stream may mix EEG, EOG, EMG, TRIG, ...
    Several devices/rates in a run are sibling ``Signal``\\ s on the owning
    :class:`Recording`.

    Parameters
    ----------
    signal
        ``[n_samples x n_channels]`` samples. Coerced with ``np.asarray`` (dtype
        preserved).
    fs
        Sampling rate in Hz (one rate per stream).
    channel_set
        Column typing/identity; ``channel_set.n_channels`` must equal
        ``signal.shape[1]``.
    times
        ``[n_samples]`` per-sample timestamps in seconds against the recording
        origin. Kept as an explicit array so irregular / dropped-sample streams
        round-trip exactly. If ``None``, regular sampling is assumed and the
        axis is synthesised from ``t0 + arange(n_samples) / fs``.
    t0
        Time of the first sample (seconds); only used to synthesise ``times``
        when it is omitted.
    **metadata
        Extra attributes, stored flat on the instance and round-tripped.

    Raises
    ------
    ValueError
        If ``signal`` is not 2-D, ``times`` does not match the sample axis, or
        ``channel_set`` size does not match the number of columns.

    Examples
    --------
    >>> from medusa.core.data import ChannelSet
    >>> cs = ChannelSet().add_unipolar_eeg_channels(["Fz", "Cz", "Pz"])
    >>> import numpy as np
    >>> sig = Signal(np.zeros((1000, 3)), fs=250.0, channel_set=cs)
    >>> sig.n_samples, sig.n_channels
    (1000, 3)
    >>> sig.pick_types("EEG").n_channels
    3
    """

    def __init__(self, signal: ArrayLike, fs: float, channel_set: ChannelSet,
                 times: "ArrayLike | None" = None, t0: float = 0.0,
                 **metadata) -> None:
        self.signal = np.asarray(signal)
        self.fs = fs
        self.channel_set = channel_set
        self.times = np.asarray(
            times if times is not None
            else t0 + np.arange(self.signal.shape[0]) / fs)
        for k, v in metadata.items():
            setattr(self, k, v)
        self._validate()

    def _validate(self) -> None:
        if self.signal.ndim != 2:
            raise ValueError(
                "`signal` must be a 2-D array [n_samples x n_channels], got "
                f"shape {self.signal.shape}.")
        if self.times.ndim != 1 or self.times.shape[0] != self.signal.shape[0]:
            raise ValueError(
                "`times` must be a 1-D array [n_samples] matching `signal`'s "
                f"first axis (got times {self.times.shape}, signal "
                f"{self.signal.shape}).")
        if self.channel_set.n_channels != self.signal.shape[1]:
            raise ValueError(
                f"`channel_set` has {self.channel_set.n_channels} channels but "
                f"`signal` has {self.signal.shape[1]} columns; they must match "
                "(channel i <-> signal[:, i]).")

    @classmethod
    def empty(cls, fs: float, channel_set: ChannelSet,
              dtype: "np.dtype | type" = np.float64) -> "Signal":
        """A zero-sample :class:`Signal` declaring a stream's structure.

        The canonical way to hand a stream to
        :class:`~medusa.core.data.streaming.Recorder` before any samples
        exist: the matrix is ``[0 x channel_set.n_channels]`` (so ``fs`` and the
        column typing are fixed up front) and ``times`` is empty. Append the
        samples to the *writer*, not to this object.

        Examples
        --------
        >>> from medusa.core.data import ChannelSet
        >>> cs = ChannelSet().add_unipolar_eeg_channels(["Fz", "Cz"])
        >>> sig = Signal.empty(fs=256.0, channel_set=cs)
        >>> sig.n_samples, sig.n_channels
        (0, 2)
        """
        return cls(np.empty((0, channel_set.n_channels), dtype=dtype),
                   fs=fs, channel_set=channel_set)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def n_samples(self) -> int:
        """Number of samples (``signal``'s first axis)."""
        return self.signal.shape[0]

    @property
    def n_channels(self) -> int:
        """Number of channels (``signal``'s second axis)."""
        return self.signal.shape[1]

    def __repr__(self) -> str:
        return (f"Signal(n_samples={self.n_samples}, "
                f"n_channels={self.n_channels}, fs={self.fs}, "
                f"types={sorted(set(self.channel_set.types))})")

    # ------------------------------------------------------------------
    # Selection (bridge to the free-function processing layer)
    # ------------------------------------------------------------------
    def pick_types(self, channel_types: "str | list[str]") -> "Signal":
        """Return a sub-:class:`Signal` of the channels matching ``channel_types``.

        The canonical bridge to type-specific routines, e.g.
        ``signal.pick_types("EEG")`` before a montage operation. Channel order is
        preserved; sensors and ``coord_system`` are carried over intact.
        """
        subset, idx = self.channel_set.pick(types=channel_types)
        return self._subset(subset, idx)

    def pick(self, channel_names: "str | list[str]") -> "Signal":
        """Return a sub-:class:`Signal` of the named channels, in the given order.

        Selects (and may reorder) by channel ``label``; raises ``KeyError`` on an
        unknown name.
        """
        idx = self.channel_set.index(channel_names)
        return self._subset(self.channel_set.subset(idx), idx)

    def _subset(self, channel_set: ChannelSet, idx: NDArray) -> "Signal":
        """Build a sub-:class:`Signal` from a selected ``channel_set`` + column ``idx``."""
        idx = np.asarray(idx, dtype=int)
        extras = {k: v for k, v in vars(self).items() if k not in _SIGNAL_CORE}
        return Signal(self.signal[:, idx], self.fs, channel_set,
                      times=self.times, **extras)

    # ------------------------------------------------------------------
    # BIDS
    # ------------------------------------------------------------------
    def bids_datatype(self) -> str | None:
        """The stream's primary BIDS datatype, or ``None`` if undecidable.

        The dominant *primary* channel modality (``EEG`` -> ``"eeg"``,
        ``SEEG``/``ECOG`` -> ``"ieeg"``, ``MEG*`` -> ``"meg"``, ``NIRS*`` ->
        ``"nirs"``, ``EMG`` -> ``"emg"``, motion kinematics -> ``"motion"``);
        auxiliary channels ride along in ``channels.tsv``. ``None`` when no
        channel implies a datatype. The mapping lives in
        :mod:`medusa.core.data._bids`.
        """
        cands = [d for d in
                 (datatype_for_channel_type(t) for t in self.channel_set.types)
                 if d is not None]
        if not cands:
            return None
        return Counter(cands).most_common(1)[0][0]

    # ------------------------------------------------------------------
    # SerializableComponent contract
    # ------------------------------------------------------------------
    def to_serializable_obj(self) -> dict:
        obj = {
            "signal": self.signal.tolist(),
            "fs": self.fs,
            "channel_set": self.channel_set.to_serializable_obj(),
            "times": self.times.tolist(),
        }
        # Flat metadata extras -> round-tripped; arrays/components serialised too.
        for k, v in vars(self).items():
            if k in _SIGNAL_CORE:
                continue
            if isinstance(v, np.ndarray):
                obj[k] = v.tolist()
            elif isinstance(v, SerializableComponent):
                obj[k] = v.to_serializable_obj()
            else:
                obj[k] = v
        return obj

    @classmethod
    def from_serializable_obj(cls, data: dict) -> "Signal":
        data = dict(data)
        channel_set = data.pop("channel_set")
        if not isinstance(channel_set, ChannelSet):
            channel_set = ChannelSet.from_serializable_obj(channel_set)
        # A single-sample or single-channel matrix round-trips through .mat with a
        # size-1 axis squeezed away; restore 2-D [n_samples x n_channels] using the
        # (already-rebuilt) channel count, and keep `times` 1-D.
        signal = np.asarray(data.pop("signal"))
        if signal.ndim < 2:
            signal = signal.reshape(-1, channel_set.n_channels)
        fs = data.pop("fs")
        times = data.pop("times", None)
        if times is not None:
            times = np.atleast_1d(times)
        return cls(signal, fs, channel_set, times=times, **data)