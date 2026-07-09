"""Qt-free adapter/inspector layer for the :class:`RecordingInspector` widget.

The metadata-editor analog of ``erp_viewer/analysis.py``: it imports only
:mod:`medusa.core` and numpy (no Qt), so it is unit-testable headless and holds
every rule the GUI must obey in one place. Three concerns:

* **outline / summaries** -- ``outline`` builds the navigator tree; the ``*_summary``
  / ``*_info`` helpers compute the read-only info shown on each page.
* **row extractors** -- ``sidecar_rows`` / ``channel_rows`` / ``sensor_rows`` /
  ``shared_sidecar_rows`` flatten a stream into table rows, tagging BIDS requirement
  levels and *derived* (read-only) fields.
* **validation / commit** -- ``check_label`` / ``check_index`` for live per-field
  checks, ``validate_recording`` for the aggregate pass, and ``clone`` / ``swap_into``
  / ``metadata_state`` for the clone-first, staged, atomic commit model.

The widget edits a working **clone** of the caller's :class:`Recording` in place
through the real engine mutators; ``validate_recording`` is the backstop and
``swap_into`` performs the atomic hand-off on Apply.
"""

from collections import Counter
from dataclasses import dataclass, field

from medusa.core.data._bids import (
    CHANNEL_COUNT_TYPE,
    is_derived_sidecar_field,
    sidecar_fields,
    validate_index,
    validate_label,
)
from medusa.core.data.channels import _CHANNEL_CORE, _SENSOR_CORE
from medusa.core.data.recording import BidsInfo, Recording
from medusa.core.data.signal import Signal
from medusa.core.serialization import SerializableComponent, tag_component

__all__ = [
    "NavNode", "Issue",
    "outline", "recording_summary", "stream_info", "channelset_summary",
    "sensor_summary", "events_summary", "experiment_kind",
    "sidecar_rows", "shared_sidecar_rows", "channel_rows", "sensor_rows",
    "derived_sidecar_value",
    "check_label", "check_index", "validate_recording",
    "clone", "swap_into", "rebuild_bids", "metadata_state",
    "ENTITY_FIELDS",
]

#: Editable BIDS filename entities, in spec order. ``subject`` is required.
ENTITY_FIELDS = ("subject", "session", "task", "acquisition", "run")

#: The one recording-wide sidecar field that mirrors a filename entity (BIDS
#: duplicates it into every ``_<datatype>.json``); surfaced for the sync hint.
TASK_NAME_FIELD = "TaskName"


# --------------------------------------------------------------------------- #
# Navigator model
# --------------------------------------------------------------------------- #
@dataclass
class NavNode:
    """One row of the navigator tree.

    ``node_id`` is the stable key the widget maps to a stacked page (e.g.
    ``"identity"``, ``"stream:eeg"``, ``"channels:eeg"``). ``badge`` is a short
    read-only annotation shown after the label.
    """
    node_id: str
    label: str
    badge: str = ""
    children: list = field(default_factory=list)


@dataclass
class Issue:
    """A validation finding pinned to a navigator ``node_id``."""
    node_id: str
    severity: str   # "error" (blocks Apply) | "warning" (advisory)
    message: str


def outline(rec: Recording) -> "list[NavNode]":
    """Build the navigator tree mirroring the :class:`Recording` structure."""
    nodes = [NavNode("identity", "Identity")]
    if rec.data:
        nodes.append(NavNode("shared", "Shared sidecar"))
    streams = NavNode("streams", "Streams")
    for key, data in rec.data.items():
        if isinstance(data, Signal):
            info = stream_info(data)
            badge = f"{info['datatype'] or '?'}·{info['n_channels']}"
            stream = NavNode(f"stream:{key}", key, badge, [
                NavNode(f"sidecar:{key}", "Sidecar"),
                NavNode(f"channels:{key}", "Channels",
                        str(info["n_channels"])),
                NavNode(f"sensors:{key}", "Sensors",
                        str(len(data.channel_set.sensors))),
            ])
        else:   # non-Signal RecordingData: minimal read-only node
            stream = NavNode(f"stream:{key}", key, type(data).__name__)
        streams.children.append(stream)
    nodes.append(streams)
    nodes.append(NavNode(
        "events", "Events",
        str(rec.events.n_events) if rec.events is not None else "none"))
    kind, detail = experiment_kind(rec.experiment)
    exp_badge = kind if detail is None else f"{kind}·{detail}"
    nodes.append(NavNode("experiment", "Experiment", exp_badge))
    return nodes


# --------------------------------------------------------------------------- #
# Summaries (read-only info)
# --------------------------------------------------------------------------- #
def recording_summary(rec: Recording) -> dict:
    """Compact whole-recording summary for the persistent top strip."""
    streams = []
    for key, data in rec.data.items():
        if isinstance(data, Signal):
            info = stream_info(data)
            streams.append({
                "key": key, "datatype": info["datatype"],
                "n_channels": info["n_channels"], "fs": info["fs"],
                "duration": info["duration"], "is_signal": True})
        else:
            streams.append({"key": key, "datatype": data.bids_datatype(),
                            "n_channels": None, "fs": None, "duration": None,
                            "is_signal": False})
    kind, detail = experiment_kind(rec.experiment)
    return {
        "basename": rec.bids.basename() or "sub-?",
        "streams": streams,
        "n_events": rec.events.n_events if rec.events is not None else None,
        "experiment_kind": kind,
        "experiment_detail": detail,
        "schema_version": rec.schema_version,
    }


def stream_info(sig: Signal) -> dict:
    """Read-only info for one :class:`Signal` stream."""
    fs = float(sig.fs) if sig.fs else None
    duration = (sig.n_samples / fs) if fs else None
    return {
        "n_samples": sig.n_samples,
        "n_channels": sig.n_channels,
        "fs": fs,
        "duration": duration,
        "dtype": str(sig.signal.dtype),
        "datatype": sig.bids_datatype(),
        "type_counts": dict(Counter(sig.channel_set.types)),
    }


def channelset_summary(cs) -> dict:
    """Read-only info for a stream's :class:`ChannelSet` (channels view)."""
    return {
        "n_channels": cs.n_channels,
        "type_counts": dict(Counter(cs.types)),
        "reference_method": cs.reference_method,
    }


def sensor_summary(cs) -> dict:
    """Read-only info for a stream's sensors."""
    sensors = list(cs.sensors.values())
    located = [s for s in sensors if s.coordinates is not None]
    any_2d = any(s.coordinates is not None and s.coordinates.shape[0] == 2
                 for s in sensors)
    dim = 3 if any(s.coordinates is not None and s.coordinates.shape[0] == 3
                   for s in sensors) else (2 if located else None)
    return {
        "n_sensors": len(sensors),
        "n_located": len(located),
        "dim": dim,
        "any_2d": any_2d,
        "coord_system": cs.coord_system,
    }


def events_summary(ev, signals: "dict") -> dict:
    """Read-only info for the events timeline (+ onset-window sanity check)."""
    if ev is None:
        return {"n_events": 0, "columns": [], "onset_min": None,
                "onset_max": None, "out_of_window": 0, "duration": None}
    df = ev.df
    onset = df["onset"] if "onset" in df else None
    duration = _recording_duration(signals)
    out_of_window = 0
    if duration is not None and len(df):
        end = df["onset"].astype(float) + df["duration"].astype(float).fillna(0)
        out_of_window = int((end > duration + 1e-9).sum())
    return {
        "n_events": ev.n_events,
        "columns": [(c, str(df[c].dtype)) for c in ev.column_names],
        "onset_min": float(onset.min()) if onset is not None and len(df) else None,
        "onset_max": float(onset.max()) if onset is not None and len(df) else None,
        "out_of_window": out_of_window,
        "duration": duration,
    }


def _recording_duration(signals: "dict") -> "float | None":
    """Longest stream duration (seconds), or ``None`` if no timed signal."""
    durs = [s.n_samples / s.fs for s in signals.values() if s.fs]
    return max(durs) if durs else None


def experiment_kind(exp) -> "tuple[str, object]":
    """Classify ``experiment``: ``("none"|"dict"|"component", detail)``."""
    if exp is None:
        return "none", None
    if isinstance(exp, SerializableComponent):
        return "component", type(exp).__name__
    if isinstance(exp, dict):
        return "dict", len(exp)
    return "other", type(exp).__name__


# --------------------------------------------------------------------------- #
# Row extractors
# --------------------------------------------------------------------------- #
_LEVEL_ORDER = {"required": 0, "recommended": 1, "optional": 2, "extra": 3}


def derived_sidecar_value(field_name: str, signal: Signal):
    """Live-recompute a *derived* sidecar field from ``signal`` (else ``None``)."""
    if signal is None:
        return None
    if field_name == "SamplingFrequency":
        return signal.fs
    if field_name == "RecordingDuration":
        return signal.n_samples / signal.fs if signal.fs else None
    if field_name.endswith("ChannelCount"):
        ch_type = CHANNEL_COUNT_TYPE.get(field_name[:-len("ChannelCount")])
        if ch_type is not None:
            return int(sum(t == ch_type for t in signal.channel_set.types))
    return None


def sidecar_rows(datatype: "str | None", sidecar: dict,
                 signal: Signal) -> "list[dict]":
    """Ordered sidecar rows: schema fields (with levels) then extra keys.

    Each row is ``{field, level, is_derived, value, derived_value}``. Derived rows
    carry the live-recomputed ``derived_value`` and must render read-only.
    """
    levels = sidecar_fields(datatype)     # {field: requirement_level}, BIDS order
    rows = []
    for name, level in levels.items():
        rows.append(_sidecar_row(name, level, sidecar, signal))
    for name in sidecar:                  # extra (non-schema) keys the draft carries
        if name not in levels:
            rows.append(_sidecar_row(name, "extra", sidecar, signal))
    return rows


def _sidecar_row(name, level, sidecar, signal):
    derived = is_derived_sidecar_field(name)
    return {
        "field": name,
        "level": level,
        "is_derived": derived,
        "value": sidecar.get(name),
        "derived_value": derived_sidecar_value(name, signal) if derived else None,
    }


def shared_sidecar_rows(rec: Recording) -> "list[dict]":
    """Recording-wide sidecar rows: each non-derived field across all streams.

    ``{field, value, shared, divergent, present_in, n_streams}``. ``shared`` means
    every stream has the field with an equal value (editing broadcasts once);
    ``divergent`` means streams disagree (flagged, unify-for-all offered).
    """
    keys = [k for k, v in rec.data.items()]   # all stream keys (sidecars parallel)
    n = len(keys)
    names = []
    for key in keys:
        for name in rec.sidecars.get(key, {}):
            if not is_derived_sidecar_field(name) and name not in names:
                names.append(name)
    rows = []
    for name in names:
        present = [k for k in keys if name in rec.sidecars.get(k, {})]
        values = [rec.sidecars[k][name] for k in present]
        set_values = [v for v in values if v is not None]
        distinct = {_hashable(v) for v in values}
        shared = len(present) == n and len(distinct) == 1
        divergent = len({_hashable(v) for v in set_values}) > 1
        rows.append({
            "field": name,
            "value": set_values[0] if set_values else (values[0] if values else None),
            "shared": shared,
            "divergent": divergent,
            "any_set": bool(set_values),
            "present_in": len(present),
            "n_streams": n,
        })
    return rows


def _hashable(v):
    return tuple(v) if isinstance(v, (list, tuple)) else v


def channel_rows(cs) -> "list[dict]":
    """One row per channel (in column order) with its editable fields + extras."""
    rows = []
    for c in cs.channels:
        extras = {k: v for k, v in vars(c).items() if k not in _CHANNEL_CORE}
        rows.append({
            "label": c.label, "ch_type": c.ch_type, "unit": c.unit,
            "sensor": c.sensor, "reference": c.reference,
            "reference_method": c.reference_method, "extras": extras})
    return rows


def sensor_rows(cs) -> "list[dict]":
    """One row per sensor with its editable fields + extras."""
    rows = []
    for label, s in cs.sensors.items():
        coords = s.coordinates
        extras = {k: v for k, v in vars(s).items() if k not in _SENSOR_CORE}
        rows.append({
            "label": label,
            "located": coords is not None,
            "x": None if coords is None else float(coords[0]),
            "y": None if coords is None else float(coords[1]),
            "z": (float(coords[2]) if coords is not None
                  and coords.shape[0] == 3 else None),
            "sensor_type": s.sensor_type, "location": s.location,
            "material": s.material, "impedance": s.impedance, "extras": extras})
    return rows


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def check_label(value, name: str, *, required: bool = False) -> "str | None":
    """Live check of a BIDS *label* entity; returns an error message or ``None``.

    A blank/``None`` value is *unset* (valid) unless ``required``.
    """
    try:
        validate_label(value, name, required=required)
        return None
    except ValueError as exc:
        return str(exc)


def check_index(value, name: str) -> "str | None":
    """Live check of a BIDS *index* entity (``run``); message or ``None``."""
    try:
        validate_index(value, name)
        return None
    except ValueError as exc:
        return str(exc)


def validate_recording(rec: Recording) -> "list[Issue]":
    """Aggregate validation pass; an ``error`` blocks Apply, a ``warning`` advises."""
    issues: list[Issue] = []
    if not rec.bids.subject:
        issues.append(Issue("identity", "error", "subject is required."))
    for key, sig in rec.signals.items():
        cs = sig.channel_set
        # n_channels must stay locked to the signal matrix width.
        if cs.n_channels != sig.signal.shape[1]:
            issues.append(Issue(
                f"channels:{key}", "error",
                f"{cs.n_channels} channels but signal has {sig.signal.shape[1]} "
                "columns."))
        # Channel label uniqueness.
        dups = [u for u, c in Counter(cs.labels).items() if c > 1]
        if dups:
            issues.append(Issue(f"channels:{key}", "error",
                                f"duplicate channel label(s): {dups}."))
        # Reference / sensor operands must name a registered sensor.
        for c in cs.channels:
            missing = [u for u in cs._linked_sensor_labels(c)
                       if u not in cs.sensors]
            if missing:
                issues.append(Issue(
                    f"channels:{key}", "error",
                    f"channel {c.label!r} links unregistered sensor(s) {missing}."))
        # Advisory: custom 2-D montage is not BIDS-EEG compliant.
        if sensor_summary(cs)["any_2d"]:
            issues.append(Issue(
                f"sensors:{key}", "warning",
                "some sensors are 2-D; BIDS EEG export needs 3-D (topography ok)."))
    # Advisory: TaskName diverges across streams (BIDS duplicates it per file).
    task_names = {_hashable(s.get(TASK_NAME_FIELD))
                  for s in rec.sidecars.values()
                  if s.get(TASK_NAME_FIELD) is not None}
    if len(task_names) > 1:
        issues.append(Issue("shared", "warning",
                            f"TaskName differs across streams: {sorted(task_names)}."))
    # Advisory: events fall outside the recording window.
    ev_info = events_summary(rec.events, rec.signals)
    if ev_info["out_of_window"]:
        issues.append(Issue(
            "events", "warning",
            f"{ev_info['out_of_window']} event(s) end after the recording "
            "duration."))
    return issues


# --------------------------------------------------------------------------- #
# Commit helpers (clone-first, staged, atomic)
# --------------------------------------------------------------------------- #
def clone(rec: Recording) -> Recording:
    """A deep, re-validated copy via the serialization round-trip (the SSOT copy)."""
    return Recording.from_serializable_obj(rec.to_serializable_obj())


def rebuild_bids(bids: BidsInfo) -> BidsInfo:
    """Reconstruct ``bids`` so the constructor re-runs label/index validation.

    Editors write attributes directly (bypassing ``BidsInfo.__init__``); rebuilding
    from the serialized form on Apply re-validates atomically and carries any future
    field through. Raises ``ValueError`` on an invalid entity.
    """
    return BidsInfo.from_serializable_obj(bids.to_serializable_obj())


def swap_into(live: Recording, edited: Recording) -> None:
    """Copy ``edited``'s attributes into ``live`` in place (the atomic hand-off).

    A host holding the caller's reference sees one consistent update rather than a
    half-mutated object.
    """
    live.schema_version = edited.schema_version
    live.bids = edited.bids
    live.data = edited.data
    live.sidecars = edited.sidecars
    live.events = edited.events
    live.experiment = edited.experiment


def metadata_state(rec: Recording) -> dict:
    """A JSON-able projection of *only the editable metadata* (dirty-check basis).

    Excludes the sample/`times` arrays (never edited, expensive to compare) so the
    dirty signal reflects metadata edits alone.
    """
    return {
        "bids": rec.bids.to_serializable_obj(),
        "data_keys": list(rec.data),
        "sidecars": {k: dict(v) for k, v in rec.sidecars.items()},
        "channel_sets": {k: sig.channel_set.to_serializable_obj()
                         for k, sig in rec.signals.items()},
        "events_descriptions": (dict(rec.events.descriptions)
                                if rec.events is not None else None),
        "experiment": _experiment_state(rec.experiment),
    }


def _experiment_state(exp):
    if exp is None:
        return None
    if isinstance(exp, SerializableComponent):
        return tag_component(exp)
    return exp
