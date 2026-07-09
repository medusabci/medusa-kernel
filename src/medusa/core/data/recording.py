"""Top-level runtime container for one recording (one BIDS *run*).

A :class:`Recording` groups everything captured in a single run around a shared
time origin: BIDS identity (:class:`BidsInfo`), one or more data streams
(:class:`~medusa.core.data.signal.Signal`) in a heterogeneous ``data`` dict, the
per-stream ``_<datatype>.json`` sidecar drafts in a parallel ``sidecars`` dict,
the BIDS ``events`` timeline, and a free-form ``experiment`` slot for
paradigm/setup metadata that BIDS does not model.

Multimodal / multi-device runs are first-class: each device is its own entry in
``data`` (keyed by a caller label that doubles as the ``acq-<key>`` entity when
two streams share a datatype), with its own sidecar in ``sidecars[key]``.
"""

import numpy as np

from medusa.core.serialization import (
    SerializableComponent, tag_component, load_component)
from medusa.core.schema import SCHEMA_VERSION, validate_recording_dict
from medusa.core.data._bids import (
    ENTITY_PREFIXES, validate_label, validate_index, sidecar_template,
    CHANNEL_COUNT_TYPE, is_derived_sidecar_field)
from medusa.core.data.recording_data import RecordingData
from medusa.core.data.signal import Signal
from medusa.core.data.events import Events

__all__ = ["Recording", "BidsInfo"]


def _autofill_derivable(sidecar: dict, signal: "Signal") -> None:
    """Fill the *derivable* sidecar fields from ``signal`` (in place).

    ``SamplingFrequency``, ``RecordingDuration`` and the mappable
    ``<X>ChannelCount`` fields. These are a convenience for the GUI/inspection;
    a BIDS export recomputes them and wins, so a stale draft cannot corrupt the
    output.
    """
    types = list(signal.channel_set.types)
    for name in sidecar:
        if not is_derived_sidecar_field(name):   # SSOT: medusa.core.data._bids
            continue
        if name == "SamplingFrequency":
            sidecar[name] = signal.fs
        elif name == "RecordingDuration":
            sidecar[name] = signal.n_samples / signal.fs
        else:   # a mappable <X>ChannelCount (prefix guaranteed valid above)
            ch_type = CHANNEL_COUNT_TYPE[name[:-len("ChannelCount")]]
            sidecar[name] = int(sum(t == ch_type for t in types))


class BidsInfo(SerializableComponent):
    """Recording-WIDE BIDS metadata, grouped by the artifact it feeds.

    Entities build the filename/directory; ``participant`` feeds
    ``participants.tsv``; ``scan`` feeds ``scans.tsv``. Per-stream
    ``_<datatype>.json`` sidecars are **not** here -- they live in
    ``Recording.sidecars`` (one per ``Recording.data`` entry).

    Parameters
    ----------
    subject
        Required ``sub-<label>`` identifier.
    session, task, acquisition
        Optional ``ses``/``task``/``acq`` labels (``[0-9a-zA-Z+]+``).
    run
        Optional ``run`` index: a non-negative ``int`` or a digit string
        (a string preserves zero-padding, e.g. ``"01"``).
    participant
        Subject-level phenotype (``age``/``sex``/``handedness``/``group`` ...);
        an open set (``-> participants.tsv``).
    scan
        Per-scan metadata (``acq_time`` ...); an open set (``-> scans.tsv``).

    Raises
    ------
    ValueError
        On an invalid label/index or a missing ``subject``.
    """

    def __init__(self, subject, session=None, task=None, acquisition=None,
                 run=None, participant: "dict | None" = None,
                 scan: "dict | None" = None) -> None:
        self.subject = validate_label(subject, "subject", required=True)
        self.session = validate_label(session, "session")
        self.task = validate_label(task, "task")
        self.acquisition = validate_label(acquisition, "acquisition")
        self.run = validate_index(run, "run")
        self.participant = dict(participant) if participant else {}
        self.scan = dict(scan) if scan else {}

    @property
    def entities(self) -> "dict":
        """Present entities as ``{name: value}`` in BIDS filename order."""
        values = {"subject": self.subject, "session": self.session,
                  "task": self.task, "acquisition": self.acquisition,
                  "run": self.run}
        return {name: values[name] for name in ENTITY_PREFIXES
                if values[name] is not None}

    def basename(self, suffix: "str | None" = None,
                 acquisition: "str | None" = None) -> str:
        """BIDS basename from the present entities, in spec order.

        ``acquisition`` overrides the stored ``acq`` entity (used per-stream to
        disambiguate same-datatype data via the data key). ``suffix`` (e.g.
        ``"eeg"``, ``"events"``), if given, is appended last.

        Examples
        --------
        >>> BidsInfo("01", session="1", task="rest", run=1).basename("eeg")
        'sub-01_ses-1_task-rest_run-1_eeg'
        >>> BidsInfo("01", task="rest").basename("eeg", acquisition="amp2")
        'sub-01_task-rest_acq-amp2_eeg'
        """
        values = dict(self.entities)
        if acquisition is not None:
            values["acquisition"] = validate_label(acquisition, "acquisition")
        # Emit in canonical BIDS order (an injected `acquisition` must land before
        # `run`, regardless of where it was added to the dict).
        parts = [f"{ENTITY_PREFIXES[name]}-{values[name]}"
                 for name in ENTITY_PREFIXES if name in values]
        base = "_".join(parts)
        return f"{base}_{suffix}" if suffix else base

    def __repr__(self) -> str:
        return f"BidsInfo({self.basename() or 'sub-?'})"

    def to_serializable_obj(self) -> dict:
        return {"subject": self.subject, "session": self.session,
                "task": self.task, "acquisition": self.acquisition,
                "run": self.run, "participant": self.participant,
                "scan": self.scan}

    @classmethod
    def from_serializable_obj(cls, data: dict) -> "BidsInfo":
        data = dict(data)
        return cls(data.get("subject"), session=data.get("session"),
                   task=data.get("task"), acquisition=data.get("acquisition"),
                   run=data.get("run"), participant=data.get("participant"),
                   scan=data.get("scan"))


class Recording(SerializableComponent):
    """One run: BIDS identity + data streams + sidecars + events + experiment.

    Parameters
    ----------
    bids
        The :class:`BidsInfo` identifying this run.

    Notes
    -----
    All ``data``/``sidecars`` mutation funnels through :meth:`add_data` /
    :meth:`add_signal` / :meth:`remove_data` so the two dicts keep identical
    keys; editing the dicts directly bypasses that guarantee.

    Examples
    --------
    >>> from medusa.core.data import ChannelSet, Signal
    >>> cs = ChannelSet().add_unipolar_eeg_channels(["Fz", "Cz", "Pz"])
    >>> sig = Signal(np.zeros((500, 3)), fs=250.0, channel_set=cs)
    >>> rec = Recording(BidsInfo("01", task="rest")).add_signal("eeg", sig)
    >>> rec.sidecars["eeg"]["SamplingFrequency"], rec.sidecars["eeg"]["EEGChannelCount"]
    (250.0, 3)
    >>> rec.bids_basename("eeg")
    'sub-01_task-rest_eeg'
    """

    def __init__(self, bids: BidsInfo) -> None:
        if not isinstance(bids, BidsInfo):
            raise TypeError("`bids` must be a BidsInfo instance.")
        self.schema_version = SCHEMA_VERSION          # serialized recording schema version
        self.bids = bids
        self.data: "dict[str, RecordingData]" = {}    # heterogeneous; KEY = stream/acq label
        self.sidecars: "dict[str, dict]" = {}         # PARALLEL to data; one _<datatype>.json draft per stream
        self.events: "Events | None" = None
        self.experiment: "dict | SerializableComponent | None" = None

    # ------------------------------------------------------------------
    # Builders (keep data/sidecars keys in sync)
    # ------------------------------------------------------------------
    def add_data(self, key: str, data: RecordingData, *,
                 overwrite: bool = False) -> "Recording":
        """Register a modality entry under ``key`` and seed its sidecar draft.

        The sidecar is a ``None``-filled template for ``data.bids_datatype()``
        with the derivable fields auto-filled (for ``Signal``). Raises
        ``KeyError`` if ``key`` already exists and ``overwrite`` is ``False``.
        """
        if not isinstance(data, RecordingData):
            raise TypeError("`data` must be a RecordingData instance.")
        if key in self.data and not overwrite:
            raise KeyError(
                f"key {key!r} already exists; pass overwrite=True to replace it.")
        self.data[key] = data
        sidecar = sidecar_template(data.bids_datatype())
        if isinstance(data, Signal):
            _autofill_derivable(sidecar, data)
        self.sidecars[key] = sidecar
        return self

    def add_signal(self, key: str, signal: Signal, *,
                   overwrite: bool = False) -> "Recording":
        """Convenience :meth:`add_data` that requires a :class:`Signal`."""
        if not isinstance(signal, Signal):
            raise TypeError("`signal` must be a Signal instance.")
        return self.add_data(key, signal, overwrite=overwrite)

    def remove_data(self, key: str) -> "Recording":
        """Drop ``data[key]`` and its ``sidecars[key]`` together."""
        del self.data[key]
        self.sidecars.pop(key, None)
        return self

    def rename_data(self, old: str, new: str) -> "Recording":
        """Rename a data key from ``old`` to ``new``, keeping the two dicts in sync.

        The key doubles as the ``acq-<key>`` entity when two streams share a
        datatype, so ``new`` must be a valid BIDS label. Both ``data`` and
        ``sidecars`` are rebuilt in lock-step (insertion order preserved) and the
        **existing sidecar draft is kept** -- unlike ``remove_data`` + ``add_data``,
        which would reseed the sidecar from a fresh template. Raises ``KeyError``
        for an unknown ``old`` or an already-used ``new``.
        """
        if old not in self.data:
            raise KeyError(f"no data entry for key {old!r}.")
        new = validate_label(new, "data key", required=True)
        if new == old:
            return self
        if new in self.data:
            raise KeyError(
                f"key {new!r} already exists; choose an unused key.")
        self.data = {(new if k == old else k): v for k, v in self.data.items()}
        self.sidecars = {(new if k == old else k): v
                         for k, v in self.sidecars.items()}
        return self

    def set_sidecar(self, key: "str | None" = None, **fields) -> "Recording":
        """Set sidecar ``field=value`` pairs on one stream, or broadcast to all.

        ``key=None`` writes the fields to *every* stream's sidecar -- the way to
        set recording-wide fields (``TaskName``, ``Instructions`` ...) once
        (BIDS duplicates them per file anyway). Raises ``KeyError`` on an
        unknown ``key``.
        """
        keys = list(self.sidecars) if key is None else [key]
        for k in keys:
            if k not in self.sidecars:
                raise KeyError(f"no sidecar for key {k!r}.")
            self.sidecars[k].update(fields)
        return self

    def set_events(self, events: Events) -> "Recording":
        """Attach the BIDS events timeline."""
        if not isinstance(events, Events):
            raise TypeError("`events` must be an Events instance.")
        self.events = events
        return self

    def set_experiment(self,
                       experiment: "dict | SerializableComponent") -> "Recording":
        """Attach the free-form paradigm/setup metadata (dict or component)."""
        if not isinstance(experiment, (dict, SerializableComponent)):
            raise TypeError(
                "`experiment` must be a dict or a SerializableComponent.")
        self.experiment = experiment
        return self

    # ------------------------------------------------------------------
    # Queries / BIDS
    # ------------------------------------------------------------------
    @property
    def signals(self) -> "dict[str, Signal]":
        """The ``Signal``-typed subset of ``data`` (by key)."""
        return {k: v for k, v in self.data.items() if isinstance(v, Signal)}

    def get_signals_by_type(self, channel_type: str) -> "dict[str, Signal]":
        """Signals that contain at least one channel of ``channel_type``."""
        return {k: v for k, v in self.signals.items()
                if channel_type in v.channel_set.types}

    def bids_basename(self, suffix: "str | None" = None,
                      key: "str | None" = None) -> str:
        """BIDS basename for this run; ``key`` (a data label) sets ``acq-<key>``."""
        return self.bids.basename(suffix=suffix, acquisition=key)

    def __repr__(self) -> str:
        return (f"Recording({self.bids.basename() or 'sub-?'}, "
                f"data={list(self.data)}, "
                f"events={self.events is not None}, "
                f"experiment={self.experiment is not None})")

    # ------------------------------------------------------------------
    # SerializableComponent contract
    # ------------------------------------------------------------------
    def to_serializable_obj(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "bids": self.bids.to_serializable_obj(),
            "data": {k: tag_component(v) for k, v in self.data.items()},
            "sidecars": {k: dict(v) for k, v in self.sidecars.items()},
            "events": (self.events.to_serializable_obj()
                       if self.events is not None else None),
            "experiment": self._experiment_to_obj(),
        }

    @classmethod
    def from_serializable_obj(cls, data: dict) -> "Recording":
        data = dict(data)
        validate_recording_dict(data)             # version + required-keys check
        rec = cls(BidsInfo.from_serializable_obj(data["bids"]))
        rec.schema_version = data.get("schema_version", rec.schema_version)
        # Set data/sidecars directly (bypass builders) to preserve saved drafts.
        rec.data = {k: load_component(tagged)
                    for k, tagged in (data.get("data") or {}).items()}
        rec.sidecars = {k: dict(v)
                        for k, v in (data.get("sidecars") or {}).items()}
        events = data.get("events")
        rec.events = Events.from_serializable_obj(events) \
            if events is not None else None
        rec.experiment = cls._experiment_from_obj(data.get("experiment"))
        return rec

    def _experiment_to_obj(self):
        """Serialise ``experiment``: tag a component, wrap a plain dict."""
        if self.experiment is None:
            return None
        if isinstance(self.experiment, SerializableComponent):
            return tag_component(self.experiment)        # {module_name, class_name, component_data}
        return {"component_data": self.experiment}       # plain dict (no class tag)

    @staticmethod
    def _experiment_from_obj(obj):
        """Inverse of :meth:`_experiment_to_obj` (tag presence discriminates)."""
        if obj is None:
            return None
        if "module_name" in obj and "class_name" in obj:
            return load_component(obj)
        return obj["component_data"]

    # ------------------------------------------------------------------
    # Persistence: HDF5 is a Recording-specific format, not a universal one.
    # The base SerializableComponent handles the to_serializable_obj formats
    # (bson/json/mat/pickle); we extend save/load with `.h5`/`.hdf5` because only
    # an array-bearing component can store binary arrays natively (see streaming.py).
    # ------------------------------------------------------------------
    def save(self, path, data_format=None) -> None:
        """Save the recording; adds ``.h5``/``.hdf5`` to the base formats.

        HDF5 writes sample arrays as native (chunked, gzip) datasets and the rest
        as a ``/meta`` JSON blob -- the one-shot equivalent of the streaming
        :class:`~medusa.core.data.streaming.Recorder`, and a full peer of the
        ``.bson``/``.json``/``.mat`` exports. Other formats defer to
        :meth:`SerializableComponent.save`.
        """
        df = data_format if data_format is not None else path.split('.')[-1]
        if df in ('hdf5', 'h5'):
            from medusa.core.data.streaming import Recorder
            with Recorder(self, file=path, buffer=None, overwrite=True):
                pass  # __init__ writes the present samples; __exit__ closes
            return
        super().save(path, data_format)

    @classmethod
    def load(cls, path, data_format=None) -> "Recording":
        """Load a recording; adds ``.h5``/``.hdf5`` to the base formats.

        HDF5 is read by :func:`medusa.core.data.streaming.read_recording` (there is
        no reader class); other formats defer to :meth:`SerializableComponent.load`.
        """
        df = data_format if data_format is not None else path.split('.')[-1]
        if df in ('hdf5', 'h5'):
            from medusa.core.data.streaming import read_recording
            return read_recording(path)
        return super().load(path, data_format)
