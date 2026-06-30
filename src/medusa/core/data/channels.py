"""BIDS-aligned channel / sensor model for the MEDUSA runtime data hierarchy.

Two distinct concepts, mirroring the BIDS file split:

* a :class:`Channel` is a *data column* (one column of ``signal``; one row of
  ``channels.tsv``);
* a :class:`Sensor` is a *physical transducer* (electrode / optode / EMG sensor)
  with an optional position (``electrodes.tsv`` / ``optodes.tsv`` plus
  ``coordsystem.json``).

The mapping is **many-to-many**: a bipolar channel spans two sensors; a sensor may
feed several channels; reference/ground sensors are not data columns; ``TRIG``
columns have no sensor. :class:`ChannelSet` groups them. Channels link to sensors
by ``uid``, which doubles as the human label and the BIDS ``name`` and is unique
within a :class:`ChannelSet`.
"""

import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray
import pandas as pd

from medusa.core.serialization import SerializableComponent, ensure_list
# The channels.tsv `type` vocabulary lives in the centralised BIDS home
# (`medusa.core.data._bids`); re-exported here for backwards-compatible access.
from medusa.core.data._bids import BIDS_CHANNEL_TYPES

__all__ = ["Channel", "Sensor", "ChannelSet", "REFERENCE_METHODS"]

# Reference *scheme* vocabulary for ``Channel.reference_method``. ``None`` means
# unspecified / unreferenced. The scheme is orthogonal to the *operand*, held
# separately in ``Channel.reference`` as sensor ``uid``(s) or ``None``.
REFERENCE_METHODS = ("common", "average", "bipolar")

# Explicitly-modelled attribute names for each class. Anything else passed at
# construction is a free-form *extra*: stored flat on the instance (so it surfaces
# as an extra BIDS tsv column) and round-tripped. Single source of truth for the
# core/extra split.
_SENSOR_CORE = ("uid", "coordinates", "sensor_type", "location", "material",
                "impedance")
_CHANNEL_CORE = ("uid", "ch_type", "unit", "sensor", "reference",
                 "reference_method")


def _coerce_coordinates(coordinates: "ArrayLike | dict | None") -> "NDArray | None":
    """Normalise a coordinate input to a 1-D float array of length 2/3, or ``None``.

    Accepts ``None``, an array-like ``[x, y]`` / ``[x, y, z]``, or a mapping with
    ``x``/``y`` (and optional ``z``) keys.

    Raises
    ------
    ValueError
        If the coordinates do not have 2 or 3 entries.
    """
    if coordinates is None:
        return None
    if isinstance(coordinates, dict):
        keys = ["x", "y", "z"] if "z" in coordinates else ["x", "y"]
        coordinates = [coordinates[k] for k in keys]
    coordinates = np.asarray(coordinates, dtype=float).ravel()
    if coordinates.shape[0] not in (2, 3):
        raise ValueError(
            "`coordinates` must have 2 (x, y) or 3 (x, y, z) entries, got "
            f"{coordinates.shape[0]}"
        )
    return coordinates


class Sensor(SerializableComponent):
    """A physical transducer (electrode, optode, EMG sensor, ...).

    Not a data column: a device that optionally has a position. Channels link to
    it by ``uid``. Exports as one row of ``electrodes.tsv`` / ``optodes.tsv`` under
    :attr:`ChannelSet.coord_system`.

    Parameters
    ----------
    uid
        Unique identifier within the :class:`ChannelSet` (e.g. ``"C3"``); BIDS
        ``name``.
    coordinates
        Position ``[x, y]`` or ``[x, y, z]`` (or a ``{'x':.., 'y':..[, 'z':..]}``
        mapping). ``None`` when undigitised (EMG by landmark, reference clip).
    sensor_type
        *Physical* transducer kind -- the BIDS ``electrodes.tsv`` / ``optodes.tsv``
        ``type`` (``cup``/``ring``/``needle``/``surface`` ..., ``source``/
        ``detector`` for optodes). Not the modality (that is :attr:`Channel.ch_type`).
    location
        Anatomical/textual placement when there is no numeric position (e.g.
        ``"upper-quadriceps"``).
    material
        Electrode material (e.g. ``"Ag/AgCl"``); BIDS ``material``.
    impedance
        Measured impedance in kOhm; BIDS ``impedance``.
    **kwargs
        Extra attributes, stored **flat** and surfaced as extra ``electrodes.tsv``
        columns; round-tripped on (de)serialization.

    Examples
    --------
    >>> Sensor("C3", coordinates=[-0.3249, 0.0, 0.9211], material="Ag/AgCl")
    Sensor(uid='C3', ...)
    >>> Sensor("EMG-quad", location="upper-quadriceps", sensor_type="surface")
    Sensor(uid='EMG-quad', ...)
    """

    def __init__(self, uid: str, coordinates: "ArrayLike | dict | None" = None,
                 sensor_type: str | None = None, location: str | None = None,
                 material: str | None = None, impedance: float | None = None,
                 **kwargs) -> None:
        self.uid = str(uid)
        self.coordinates = _coerce_coordinates(coordinates)
        self.sensor_type = sensor_type
        self.location = location
        self.material = material
        self.impedance = impedance
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self) -> str:
        return (f"Sensor(uid={self.uid!r}, sensor_type={self.sensor_type!r}, "
                f"located={self.coordinates is not None})")

    def to_serializable_obj(self) -> dict:
        obj = {
            "uid": self.uid,
            "coordinates": (None if self.coordinates is None
                            else self.coordinates.tolist()),
            "sensor_type": self.sensor_type,
            "location": self.location,
            "material": self.material,
            "impedance": self.impedance,
        }
        # Flat extra attributes -> extra electrodes.tsv columns; round-tripped.
        obj.update({k: v for k, v in vars(self).items()
                    if k not in _SENSOR_CORE})
        return obj

    @classmethod
    def from_serializable_obj(cls, data: dict) -> "Sensor":
        data = dict(data)
        uid = data.pop("uid")
        return cls(
            uid,
            coordinates=data.pop("coordinates", None),
            sensor_type=data.pop("sensor_type", None),
            location=data.pop("location", None),
            material=data.pop("material", None),
            impedance=data.pop("impedance", None),
            **data,   # remaining keys are flat extras
        )

    def __eq__(self, other: object) -> bool:
        """Content equality (uid + position + all attributes); ``uid`` conflicts."""
        if not isinstance(other, Sensor):
            return NotImplemented
        return self.to_serializable_obj() == other.to_serializable_obj()

    __hash__ = None   # mutable; identified by uid within a ChannelSet (dict key)


class Channel(SerializableComponent):
    """A data column: one column of ``signal``; one row of ``channels.tsv``.

    Carries *what was measured* and references its transducer(s) by sensor ``uid``
    rather than storing a position, so bipolar/derived/position-less channels are
    first-class.

    Parameters
    ----------
    uid
        Unique identifier within the :class:`ChannelSet` (e.g. ``"C3"``,
        ``"F3-C3"``); BIDS ``name``.
    ch_type
        Channel modality -- the BIDS ``channels.tsv`` ``type``. Matched
        case-insensitively against :data:`BIDS_CHANNEL_TYPES`; unknown -> ``"OTHER"``
        with a warning.
    unit
        Physical unit of the column; BIDS ``units``.
    sensor
        ``uid`` of the active sensor (position source); must already be registered
        in the :class:`ChannelSet`. ``None`` for ``TRIG``/``MISC``/derived columns.
    reference
        Reference *operand* as sensor ``uid``(s): the reference electrode (common),
        the subtracted electrode (bipolar), or averaged electrodes (e.g.
        ``["M1", "M2"]``). ``None`` for CAR / unreferenced.
    reference_method
        Reference *scheme* -- one of :data:`REFERENCE_METHODS` or ``None``.
        Disambiguates a single-``uid`` operand (common vs bipolar).
    **kwargs
        Extra attributes, stored flat and surfaced as extra ``channels.tsv``
        columns; round-tripped on (de)serialization.

    Notes
    -----
    Scheme (``reference_method``) and operand (``reference``) combine as:

    | Scheme               | ``reference_method`` | ``reference``         |
    |----------------------|----------------------|-----------------------|
    | Common               | ``"common"``         | ``"M1"``              |
    | Bipolar              | ``"bipolar"``        | ``"C3"`` (subtracted) |
    | CAR (average of all) | ``"average"``        | ``None``              |
    | Linked mastoids      | ``"average"``        | ``["M1", "M2"]``      |
    | Unreferenced (TRIG)  | ``None``             | ``None``              |

    Examples
    --------
    >>> Channel("C3", "EEG", "uV", sensor="C3", reference="M1",
    ...         reference_method="common")
    Channel(uid='C3', ch_type='EEG', ...)
    >>> Channel("TRIG", "TRIG", "n/a")  # no sensor, no reference
    Channel(uid='TRIG', ch_type='TRIG', ...)
    """

    def __init__(self, uid: str, ch_type: str, unit: str,
                 sensor: str | None = None,
                 reference: "str | list[str] | None" = None,
                 reference_method: str | None = None, **kwargs) -> None:
        self.uid = str(uid)
        self.ch_type = self._validate_ch_type(ch_type)
        self.unit = unit
        self.sensor = sensor
        self.reference = reference
        self.reference_method = self._validate_reference_method(reference_method)
        for k, v in kwargs.items():
            setattr(self, k, v)

    @staticmethod
    def _validate_ch_type(ch_type: str | None) -> str:
        if ch_type is None:
            return "OTHER"
        norm = str(ch_type).upper()
        if norm not in BIDS_CHANNEL_TYPES:
            warnings.warn(
                f"Channel type {ch_type!r} is not in the BIDS channel-type "
                f"vocabulary; coercing to 'OTHER'. Known types: "
                f"{', '.join(BIDS_CHANNEL_TYPES)}.")
            return "OTHER"
        return norm

    @staticmethod
    def _validate_reference_method(reference_method: str | None) -> str | None:
        if reference_method is None:
            return None
        norm = str(reference_method).lower()
        if norm not in REFERENCE_METHODS:
            raise ValueError(
                f"reference_method {reference_method!r} is not one of "
                f"{REFERENCE_METHODS} (or None).")
        return norm

    def __repr__(self) -> str:
        return (f"Channel(uid={self.uid!r}, ch_type={self.ch_type!r}, "
                f"sensor={self.sensor!r}, reference={self.reference!r}, "
                f"reference_method={self.reference_method!r})")

    def to_serializable_obj(self) -> dict:
        obj = {
            "uid": self.uid,
            "ch_type": self.ch_type,
            "unit": self.unit,
            "sensor": self.sensor,
            "reference": self.reference,
            "reference_method": self.reference_method,
        }
        # Flat extra attributes -> extra channels.tsv columns; round-tripped.
        obj.update({k: v for k, v in vars(self).items()
                    if k not in _CHANNEL_CORE})
        return obj

    @classmethod
    def from_serializable_obj(cls, data: dict) -> "Channel":
        data = dict(data)
        uid = data.pop("uid")
        return cls(
            uid,
            ch_type=data.pop("ch_type", "EEG"),
            unit=data.pop("unit", "uV"),
            sensor=data.pop("sensor", None),
            reference=data.pop("reference", None),
            reference_method=data.pop("reference_method", None),
            **data,   # remaining keys are flat extras
        )


class ChannelSet(SerializableComponent):
    """An ordered set of :class:`Channel`s plus their :class:`Sensor`s (BIDS-aligned).

    Mirrors the BIDS three-file split: :attr:`channels` (``channels.tsv``, an
    ordered list where ``channels[i]`` <-> ``signal[:, i]``), :attr:`sensors`
    (``electrodes.tsv`` / ``optodes.tsv``, a ``{uid: Sensor}`` dict) and
    :attr:`coord_system` (``coordsystem.json``). The two collections are
    independent in size (no position for ``TRIG``; extra reference/ground sensors).

    Build **sensors-first**: register sensors with :meth:`add_sensors` (or let
    :meth:`add_unipolar_eeg_channels` mint them from a montage), then add columns
    with :meth:`add_channels` (both accept a single item or a list). A channel
    links to its sensor(s) by ``uid`` only; naming an unregistered sensor, or a
    duplicate channel ``uid``, raises on add, so the set is valid as soon as a
    builder returns (no finalisation step).

    Parameters
    ----------
    channels
        Data columns, in signal-column order.
    sensors
        Physical transducers (list or ``{uid: Sensor}`` dict); a conflicting uid
        raises.
    coord_system
        BIDS ``coordsystem.json`` content. :meth:`add_unipolar_eeg_channels`
        records the montage name in its ``description``.

    Raises
    ------
    ValueError
        On duplicate channel/sensor ``uid``s, or a channel linking to an
        unregistered sensor.

    Examples
    --------
    >>> cs = ChannelSet()
    >>> _ = cs.add_unipolar_eeg_channels(["Fz", "Cz", "Pz", "Oz"],
    ...                                  reference="M1", ground="AFz")
    >>> _ = cs.add_sensors([Sensor("VEOG_up"), Sensor("VEOG_down")])
    >>> _ = cs.add_channels(Channel("VEOG", "VEOG", "uV", sensor="VEOG_up",
    ...                             reference="VEOG_down",
    ...                             reference_method="bipolar"), index=2)
    >>> cs.uids
    ['Fz', 'Cz', 'VEOG', 'Pz', 'Oz']
    """

    def __init__(self, channels: "list[Channel] | None" = None,
                 sensors: "list[Sensor] | dict[str, Sensor] | None" = None,
                 coord_system: dict | None = None) -> None:
        self.channels = []
        self.sensors = {}   # {uid: Sensor}; uniqueness is structural
        self.coord_system = coord_system
        # Sensors first: a channel links to its sensor(s) by uid, so those
        # sensors must already be registered before the channels are added
        # (add_channels enforces this).
        if sensors is not None:
            seq = sensors.values() if isinstance(sensors, dict) else sensors
            for s in seq:
                self._register_sensor(s)
        if channels is not None:
            self.add_channels(channels)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    @staticmethod
    def _linked_sensor_uids(channel: Channel) -> "list[str]":
        """Sensor ``uid``s a channel links to (active ``sensor`` + ``reference``)."""
        uids = []
        if isinstance(channel.sensor, str):
            uids.append(channel.sensor)
        ref = channel.reference
        if isinstance(ref, str):
            uids.append(ref)
        elif isinstance(ref, (list, tuple)):
            uids.extend(ref)
        return [u for u in uids if u is not None]

    def _check_new_channel_uids(self, new_channels: "list[Channel]") -> None:
        """Raise if any incoming channel ``uid`` clashes (with existing or itself)."""
        existing = {c.uid for c in self.channels}
        seen = set()
        for c in new_channels:
            if not isinstance(c, Channel):
                raise TypeError("channels must be Channel instances")
            if c.uid in existing or c.uid in seen:
                raise ValueError(
                    f"Duplicate channel uid {c.uid!r} in ChannelSet; channel "
                    f"uids must be unique.")
            seen.add(c.uid)

    def _check_sensor_links(self, channel: Channel) -> None:
        """Raise if a channel names a sensor ``uid`` that is not registered."""
        missing = [u for u in self._linked_sensor_uids(channel)
                   if u not in self.sensors]
        if missing:
            raise ValueError(
                f"Channel {channel.uid!r} links to unregistered sensor(s) "
                f"{missing}; add them with add_sensors before adding the channel "
                f"(sensors must be registered first).")

    def _register_sensor(self, sensor: Sensor) -> None:
        """Register one sensor by ``uid``; equal re-add is a no-op, conflict raises."""
        if not isinstance(sensor, Sensor):
            raise TypeError("sensors must be Sensor instances")
        existing = self.sensors.get(sensor.uid)
        if existing is None:
            self.sensors[sensor.uid] = sensor
        elif existing != sensor:
            raise ValueError(
                f"Conflicting definition for sensor {sensor.uid!r}: a different "
                f"sensor with the same uid is already registered. Sensor uids "
                f"must identify a single physical transducer.")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def uids(self) -> "list[str]":
        """Channel ``uid``s, in column order."""
        return [c.uid for c in self.channels]

    @property
    def n_channels(self) -> int:
        """Number of channels (== number of signal columns)."""
        return len(self.channels)

    @property
    def types(self) -> "list[str]":
        """Per-channel ``ch_type``."""
        return [c.ch_type for c in self.channels]

    @property
    def reference_method(self) -> str | None:
        """Shared reference scheme: the common per-channel value, ``"mixed"``, or ``None``."""
        methods = {c.reference_method for c in self.channels
                   if c.reference_method is not None}
        if not methods:
            return None
        if len(methods) == 1:
            return next(iter(methods))
        return "mixed"

    # ------------------------------------------------------------------
    # Builders
    # ------------------------------------------------------------------
    def add_channels(self, channels: "Channel | list[Channel]",
                     index: int | None = None) -> "ChannelSet":
        """Add one data column or several (block insert at ``index``); returns ``self``.

        Accepts a single :class:`Channel` or a list of them. Every channel's named
        sensor(s) must already be registered (see :meth:`add_sensors`) and channel
        ``uid``s must be unique; both are checked before anything is inserted.

        Parameters
        ----------
        channels
            A :class:`Channel` or a list of them, in signal-column order
            (``sensor``/``reference`` name registered sensors).
        index
            Insert position (``list`` slice semantics); ``None`` appends. Channel
            order is signal-column order.

        Raises
        ------
        ValueError
            If a channel ``uid`` clashes, or a channel names an unregistered sensor.
        """
        channels = [channels] if isinstance(channels, Channel) else list(channels)
        self._check_new_channel_uids(channels)
        for c in channels:
            self._check_sensor_links(c)
        if index is None:
            self.channels.extend(channels)
        else:
            self.channels[index:index] = channels
        return self

    def add_sensors(self, sensors: "Sensor | list[Sensor]") -> "ChannelSet":
        """Register one :class:`Sensor` or several (before the channels that link to them).

        Accepts a single :class:`Sensor` or a list. Re-adding an equal sensor is a
        no-op; a different sensor under an existing ``uid`` raises. Returns ``self``.
        """
        for s in ([sensors] if isinstance(sensors, Sensor) else sensors):
            self._register_sensor(s)
        return self

    def add_unipolar_eeg_channels(
            self,
            uids: "list[str]",
            montage: "dict | None" = None,
            reference: "str | list[str] | None" = None,
            ground: str | None = None,
            reference_method: str | None = None,
            coord_system: dict | None = None,
            index: int | None = None,
    ) -> "ChannelSet":
        """Add a block of unipolar EEG channels and their located sensors.

        EEG shortcut for the additive API: builds one EEG :class:`Channel` + one
        located :class:`Sensor` per label, registers the reference/ground
        sensors, and sets :attr:`coord_system`. Coordinates resolve from the
        single standard 3-D master -- one per electrode, synonyms included -- so
        you just pass labels. Pass
        :func:`~medusa.core.data.eeg.get_standard_montage_labels` to add a whole
        standard system, or a custom ``montage`` dict to override the positions
        (a custom montage may be 2-D, which warns). Returns ``self``.

        Parameters
        ----------
        uids
            EEG channel/sensor labels (e.g. ``["Fz", "Cz", "Pz"]``, or
            ``get_standard_montage_labels("10-20")``).
        montage
            ``None`` (default): use the standard 3-D coordinates, resolved per
            label. A custom ``{label: coordinates}`` dict (2-D or 3-D): override
            the standard positions. A standard NAME is no longer accepted -- pass
            its labels via ``get_standard_montage_labels(...)`` instead.
        reference
            Reference operand: a label (common), a list (subset average, e.g.
            linked mastoids), or ``None`` (CAR / unreferenced). Each becomes a
            located reference sensor.
        ground
            Ground sensor label (a located sensor, not a channel).
        reference_method
            Scheme for every channel; if ``None``, inferred from ``reference``
            (list -> ``'average'``, single -> ``'common'``).
        coord_system
            ``coordsystem.json`` content to attach; a montage-describing default is
            used when ``None``.
        index
            Insert position for the block (append if ``None``).
        """
        if montage is None:
            # Coordinates come from the single standard 3-D master, resolved per
            # label (synonyms included). Pass get_standard_montage_labels(...) to
            # add a whole standard system. Lazy import keeps eeg off the path.
            from medusa.core.data.eeg import get_coordinates, resolve_label
            standard_coords = get_coordinates()
            montage_name = "standard"
            default_coord_system = {
                "EEGCoordinateSystem": "Other",
                "EEGCoordinateUnits": "n/a",
                "description": "MEDUSA standard 3D cartesian coordinates.",
            }

            def _coords_for(label):
                return standard_coords.get(resolve_label(label))
        elif isinstance(montage, dict):
            # Custom montage {label: {x,y[,z]}} or {label: [x,y[,z]]}; 2-D or 3-D.
            montage_data = {str(k).upper().strip(): v
                            for k, v in montage.items()}
            montage_name = "custom"
            default_coord_system = {
                "EEGCoordinateSystem": "Other",
                "EEGCoordinateUnits": "n/a",
                "description": "User-provided custom EEG montage.",
            }

            def _coords_for(label):
                # Raw entry (dict {x,y[,z]} / array-like / None); Sensor coerces
                # it to a 2-D or 3-D array.
                return montage_data.get(label.upper().strip())
        else:
            raise TypeError(
                "`montage` must be None (use the standard coordinates) or a "
                "custom {label: coordinates} dict. A standard NAME is no longer "
                "accepted -- pass its labels instead, e.g. "
                "add_unipolar_eeg_channels(get_standard_montage_labels('10-20')).")

        # Reference operand -> scheme.
        ref_uid = reference
        if isinstance(reference, str):
            ref_labels = [reference]
            reference_method = reference_method or "common"
        elif isinstance(reference, (list, tuple)):
            ref_labels = list(reference)
            reference_method = reference_method or "average"
        else:
            ref_labels = []  # None -> CAR / unreferenced

        new_sensors, new_channels = [], []
        for r in ref_labels:
            coords = _coords_for(r)
            if coords is None:
                warnings.warn(f"Reference {r!r} not in montage {montage_name!r}; "
                              f"added without coordinates.")
            new_sensors.append(Sensor(r, coordinates=coords,
                                      sensor_type="reference"))
        if ground is not None:
            coords = _coords_for(ground)
            if coords is None:
                warnings.warn(f"Ground {ground!r} not in montage {montage_name!r}; "
                              f"added without coordinates.")
            new_sensors.append(Sensor(ground, coordinates=coords,
                                      sensor_type="ground"))
        for uid in uids:
            coords = _coords_for(uid)
            if coords is None:
                warnings.warn(f"Channel {uid!r} not in montage {montage_name!r}; "
                              f"added without coordinates.")
            new_sensors.append(Sensor(uid, coordinates=coords))
            new_channels.append(Channel(uid, ch_type="EEG", unit="uV",
                                        sensor=uid, reference=ref_uid,
                                        reference_method=reference_method))

        # A custom montage may be 2-D; flag it once (not BIDS-EEG compliant).
        if montage_name == "custom" and any(
                s.coordinates is not None and s.coordinates.shape[0] == 2
                for s in new_sensors):
            warnings.warn(
                "Custom EEG montage has 2-D coordinates; BIDS EEG export "
                "requires 3-D (x, y, z). Topographic plotting still works.")

        seen = set()
        for s in new_sensors:   # de-dup within this block (e.g. a label used as
            if s.uid in seen:   # both a channel and the reference); first wins
                continue
            seen.add(s.uid)
            self._register_sensor(s)
        self.add_channels(new_channels, index=index)
        # The EEG block defines the head coordinate system for the whole set; the
        # montage name lives in its `description` (no EEG-specific attr on the set).
        self.coord_system = (coord_system if coord_system is not None
                             else default_coord_system)
        return self

    def get_channel(self, uid: str) -> Channel:
        """Return the :class:`Channel` with this ``uid`` (raises ``KeyError``)."""
        for c in self.channels:
            if c.uid == uid:
                return c
        raise KeyError(f"No channel with uid {uid!r}")

    def get_sensor(self, uid: str) -> Sensor:
        """Return the :class:`Sensor` with this ``uid`` (raises ``KeyError``)."""
        try:
            return self.sensors[uid]
        except KeyError:
            raise KeyError(f"No sensor with uid {uid!r}")

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------
    def index(self, uids: "str | list[str]") -> NDArray:
        """Return the column indices of ``uids``, in the given order (raises if unknown)."""
        if isinstance(uids, str):
            uids = [uids]
        pos = {c.uid: i for i, c in enumerate(self.channels)}
        missing = [u for u in uids if u not in pos]
        if missing:
            raise KeyError(f"Unknown channel uid(s): {missing}")
        return np.array([pos[u] for u in uids], dtype=int)

    def pick(self, types: "str | list[str] | None" = None,
             uids: "str | list[str] | None" = None) -> "tuple[ChannelSet, NDArray]":
        """Select a subset of channels by ``ch_type`` and/or ``uid``.

        ``uids`` is applied before ``types``.

        Returns
        -------
        subset : ChannelSet
            New set with the selected channels (sensors kept intact).
        idx : numpy.ndarray
            Selected column indices into the original set (to slice ``signal``).
        """
        if uids is not None:
            idx = self.index(uids)
        else:
            idx = np.arange(self.n_channels, dtype=int)
        if types is not None:
            if isinstance(types, str):
                types = [types]
            wanted = {t.upper() for t in types}
            idx = np.array([i for i in idx
                            if self.channels[i].ch_type in wanted], dtype=int)
        return self.subset(idx), idx

    def subset(self, idx: "NDArray | list[int]") -> "ChannelSet":
        """Return a new set with channels selected/reordered by ``idx``.

        ``sensors`` and ``coord_system`` are preserved; ``reference_method``
        re-derives from the selected channels.
        """
        idx = np.asarray(idx, dtype=int).ravel()
        return ChannelSet(
            channels=[self.channels[i] for i in idx],
            sensors=self.sensors,
            coord_system=self.coord_system,
        )

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------
    def get_positions(self, types: "str | list[str] | None" = "EEG") -> NDArray:
        """Active-sensor positions, shape ``(n_selected, dim)`` (``NaN`` where unresolved).

        Each (matching) channel resolves to its active ``sensor``'s coordinates; a
        bipolar channel returns its active sensor's position; channels with no /
        unlocated sensor or a dimensionality mismatch yield a ``NaN`` row. ``dim``
        is 3, or 2 if no sensor is 3-D. ``types=None`` keeps all channels.
        """
        if types is None:
            channels = self.channels
        else:
            if isinstance(types, str):
                types = [types]
            wanted = {t.upper() for t in types}
            channels = [c for c in self.channels if c.ch_type in wanted]
        dim = 3 if any(s.coordinates is not None and s.coordinates.shape[0] == 3
                       for s in self.sensors.values()) else 2
        sensors = self.sensors
        pos = np.full((len(channels), dim), np.nan, dtype=float)
        for i, cha in enumerate(channels):
            uid = cha.sensor
            s = sensors.get(uid) if isinstance(uid, str) else None
            if s is not None and s.coordinates is not None \
                    and s.coordinates.shape[0] == dim:
                pos[i] = s.coordinates
        return pos

    # ------------------------------------------------------------------
    # BIDS projection
    # ------------------------------------------------------------------
    def to_channels_dataframe(self) -> pd.DataFrame:
        """BIDS ``channels.tsv`` table: one row per channel.

        Columns ``name``/``type``/``units``/``reference`` (operand folded into the
        single BIDS column) plus any extra channel attributes; missing -> ``"n/a"``.
        """
        rows = []
        for c in self.channels:
            row = {
                "name": c.uid,
                "type": c.ch_type,
                "units": c.unit,
                "reference": self._bids_reference(c),
            }
            # Surface any extra (non-core) attributes set via **kwargs.
            row.update({k: v for k, v in vars(c).items()
                        if k not in _CHANNEL_CORE})
            rows.append(row)
        return pd.DataFrame(rows).where(lambda df: df.notna(), "n/a")

    @staticmethod
    def _bids_reference(c: Channel) -> str:
        """Fold (``reference_method``, ``reference``) into BIDS' single ``reference`` column."""
        ref = c.reference
        operand = ",".join(ref) if isinstance(ref, (list, tuple)) else ref
        if c.reference_method == "bipolar":
            return operand if operand else "bipolar"
        if c.reference_method == "average":
            return operand if operand else "average"
        if c.reference_method == "common":
            return operand if operand else "n/a"
        return operand if operand else "n/a"

    def to_sensors_dataframe(self) -> pd.DataFrame:
        """BIDS ``electrodes.tsv`` table: one row per sensor.

        Columns ``name``/``x``/``y``/``z``/``type``/``material``/``impedance``/
        ``location`` plus extras (the basis of ``optodes.tsv`` for NIRS); missing
        -> ``"n/a"``.
        """
        rows = []
        for s in self.sensors.values():
            coords = s.coordinates
            row = {
                "name": s.uid,
                "x": None if coords is None else float(coords[0]),
                "y": None if coords is None else float(coords[1]),
                "z": (float(coords[2]) if coords is not None
                      and coords.shape[0] == 3 else None),
                "type": s.sensor_type,
                "material": s.material,
                "impedance": s.impedance,
                "location": s.location,
            }
            # Surface any extra (non-core) attributes set via **kwargs.
            row.update({k: v for k, v in vars(s).items()
                        if k not in _SENSOR_CORE})
            rows.append(row)
        return pd.DataFrame(rows).where(lambda df: df.notna(), "n/a")

    # ------------------------------------------------------------------
    # SerializableComponent contract
    # ------------------------------------------------------------------
    def to_serializable_obj(self) -> dict:
        # `reference_method` is not stored: it is derived from the per-channel
        # scheme (which round-trips inside each Channel).
        return {
            "channels": [c.to_serializable_obj() for c in self.channels],
            "sensors": [s.to_serializable_obj() for s in self.sensors.values()],
            "coord_system": self.coord_system,
        }

    @classmethod
    def from_serializable_obj(cls, data: dict) -> "ChannelSet":
        data = dict(data)
        # ensure_list: a single channel/sensor round-trips through .mat as one
        # dict (squeeze), not a 1-element list -- wrap it back so iteration holds.
        return cls(
            channels=[Channel.from_serializable_obj(c)
                      for c in ensure_list(data.get("channels", []))],
            sensors=[Sensor.from_serializable_obj(s)
                     for s in ensure_list(data.get("sensors", []))],
            coord_system=data.get("coord_system"),
        )
