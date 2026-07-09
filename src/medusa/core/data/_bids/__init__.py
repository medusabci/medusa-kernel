"""Centralised BIDS-standard data for the MEDUSA runtime model.

Single home for everything derived from the BIDS specification, so no BIDS
constants are duplicated across ``core.data``:

* :data:`BIDS_CHANNEL_TYPES` -- the ``channels.tsv`` ``type`` controlled
  vocabulary (derived from the schema at import; ~5 ms).
* the entity grammar -- :data:`ENTITY_PREFIXES` (filename order) and
  :func:`validate_label` / :func:`validate_index` (BIDS label/index format).
* the per-datatype sidecar field tables -- :func:`sidecar_fields` /
  :func:`sidecar_template` / :data:`DATATYPES`.

The full schema is vendored as ``bids_schema.json`` (refreshed by
``_update_schema``); the small runtime table ``bids_sidecar_fields.json`` is
generated from it by ``_create_bids_sidecar_fields``. The full schema is *not*
held in memory unless :func:`load_schema` is called (BIDS I/O / validation).
"""

import json
import re
from pathlib import Path

_DIR = Path(__file__).parent
_SCHEMA_PATH = _DIR / "bids_schema.json"
_SIDECAR_FIELDS_PATH = _DIR / "bids_sidecar_fields.json"

_SCHEMA_CACHE: "dict | None" = None
_SIDECAR_FIELDS_CACHE: "dict | None" = None


def load_schema() -> dict:
    """Return the full vendored BIDS schema (parsed once, then cached).

    Heavyweight (~800 kB in memory); kept off the import path. Intended for BIDS
    I/O and schema validation, not for runtime vocabulary lookups.
    """
    global _SCHEMA_CACHE
    if _SCHEMA_CACHE is None:
        with open(_SCHEMA_PATH, encoding="utf-8") as f:
            _SCHEMA_CACHE = json.load(f)
    return _SCHEMA_CACHE


# --------------------------------------------------------------------------- #
# Channel-type vocabulary (channels.tsv `type`)
# --------------------------------------------------------------------------- #
def _read_channel_type_enum() -> "tuple[str, ...]":
    """Derive the channels.tsv ``type`` enum from the vendored schema.

    Read in a local scope so the 800 kB dict is freed once the 48-value enum is
    extracted (only :func:`load_schema` keeps the full schema resident).
    """
    with open(_SCHEMA_PATH, encoding="utf-8") as f:
        column = json.load(f)["objects"]["columns"]["type__channels"]
    spec = column.get("definition", column)
    return tuple(str(t).upper() for t in spec["enum"])


#: BIDS ``channels.tsv`` ``type`` controlled vocabulary (upper-case), BIDS v1.11.x.
BIDS_CHANNEL_TYPES = _read_channel_type_enum()


# --------------------------------------------------------------------------- #
# Channel-type semantics (MEDUSA-curated; NOT machine-derivable from the schema)
# --------------------------------------------------------------------------- #
# The schema has a single shared channels.tsv `type` enum and binds neither types
# to datatypes nor `*ChannelCount` fields to types (their descriptions are free
# text). These interpretations are therefore curated here -- the single home for
# them -- rather than scattered across signal.py / recording.py.

# Primary motion modalities (kinematic / IMU); auxiliary motion columns
# (LATENCY, MISC) are excluded, like TRIG/MISC for EEG.
_MOTION_CHANNEL_TYPES = frozenset({
    "ACCEL", "ANGACCEL", "GYRO", "JNTANG", "MAGN", "ORNT", "POS", "VEL"})


def datatype_for_channel_type(ch_type: str) -> "str | None":
    """BIDS datatype implied by a *primary* channel modality, or ``None``.

    Auxiliary channels (``EOG``/``ECG``/``TRIG``/``MISC``/``LATENCY`` ...) return
    ``None`` and ride along in the dominant stream's ``channels.tsv`` rather than
    defining a datatype. Drives ``Signal.bids_datatype`` (dominant-modality rule).
    """
    if ch_type == "EEG":
        return "eeg"
    if ch_type in ("SEEG", "ECOG"):
        return "ieeg"
    if ch_type.startswith("MEG"):
        return "meg"
    if ch_type.startswith("NIRS"):
        return "nirs"
    if ch_type == "EMG":
        return "emg"
    if ch_type in _MOTION_CHANNEL_TYPES:
        return "motion"
    return None


#: Sidecar ``<Prefix>ChannelCount`` field -> the ``channels.tsv`` ``type`` it
#: counts (for auto-filling derivable counts). Absent prefixes stay unset, e.g.
#: ``MotionChannelCount`` is a total, not a single type.
CHANNEL_COUNT_TYPE = {
    "EEG": "EEG", "ECG": "ECG", "EMG": "EMG", "EOG": "EOG",
    "MISC": "MISC", "Misc": "MISC", "Trigger": "TRIG",
    "SEEG": "SEEG", "ECOG": "ECOG",
    "ACCEL": "ACCEL", "ANGACCEL": "ANGACCEL", "GYRO": "GYRO",
    "JNTANG": "JNTANG", "LATENCY": "LATENCY", "MAGN": "MAGN",
    "ORNT": "ORNT", "POS": "POS", "VEL": "VEL",
}

#: Sidecar fields MEDUSA auto-fills from the ``Signal`` and a BIDS export
#: recomputes (see ``recording._autofill_derivable``). Not the mappable
#: ``<X>ChannelCount`` fields, which are handled by :func:`is_derived_sidecar_field`.
_DERIVED_SCALAR_FIELDS = ("SamplingFrequency", "RecordingDuration")


def is_derived_sidecar_field(field: str) -> bool:
    """Whether a sidecar ``field`` is *derived* from the ``Signal`` (not hand-set).

    Derived fields (``SamplingFrequency``, ``RecordingDuration`` and the mappable
    ``<X>ChannelCount`` counts) are auto-filled by ``recording._autofill_derivable``
    and a BIDS export recomputes them; a metadata editor must render them read-only.
    Single source of truth shared by the auto-fill and the editor so the two cannot
    drift. ``MotionChannelCount`` (a total, not a single-type count) is *not* derived.
    """
    if field in _DERIVED_SCALAR_FIELDS:
        return True
    if field.endswith("ChannelCount"):
        return CHANNEL_COUNT_TYPE.get(field[:-len("ChannelCount")]) is not None
    return False


# --------------------------------------------------------------------------- #
# Entity grammar (filename identity)
# --------------------------------------------------------------------------- #
#: BIDS filename entities in spec-fixed order (``acq`` precedes ``run``), mapping
#: the entity name to its filename prefix. Single source of truth for both
#: validation and basename construction.
ENTITY_PREFIXES = {
    "subject": "sub", "session": "ses", "task": "task",
    "acquisition": "acq", "run": "run",
}

#: BIDS *label* = one or more letters/digits/``+``; *index* = digits only.
LABEL_RE = re.compile(r"^[0-9a-zA-Z+]+$")
INDEX_RE = re.compile(r"^[0-9]+$")


def validate_label(value, name: str, *, required: bool = False):
    """Validate/normalise a BIDS *label* entity (``[0-9a-zA-Z+]+``)."""
    if value is None:
        if required:
            raise ValueError(f"`{name}` is required.")
        return None
    value = str(value)
    if not LABEL_RE.fullmatch(value):
        raise ValueError(
            f"`{name}`={value!r} is not a valid BIDS label "
            f"(letters, digits and '+' only; no spaces, '-' or '_').")
    return value


def validate_index(value, name: str):
    """Validate a BIDS *index* entity (``run``): an ``int`` or a digit string."""
    if value is None:
        return None
    if isinstance(value, bool):  # bool is an int subclass; reject explicitly
        raise ValueError(f"`{name}` must be an int or digit string, got a bool.")
    if isinstance(value, int):
        if value < 0:
            raise ValueError(f"`{name}`={value!r} must be non-negative.")
        return value
    value = str(value)
    if not INDEX_RE.fullmatch(value):
        raise ValueError(
            f"`{name}`={value!r} is not a valid BIDS index (digits only).")
    return value


# --------------------------------------------------------------------------- #
# Sidecar field tables (_<datatype>.json)
# --------------------------------------------------------------------------- #
def _sidecar_fields_table() -> dict:
    """Load (once) the vendored ``{datatype: {field: level}}`` table."""
    global _SIDECAR_FIELDS_CACHE
    if _SIDECAR_FIELDS_CACHE is None:
        with open(_SIDECAR_FIELDS_PATH, encoding="utf-8") as f:
            _SIDECAR_FIELDS_CACHE = json.load(f)
    return _SIDECAR_FIELDS_CACHE


def datatypes() -> "tuple[str, ...]":
    """BIDS datatypes MEDUSA can emit as a ``Signal`` stream (have a sidecar table)."""
    return tuple(_sidecar_fields_table())


def sidecar_fields(datatype: "str | None") -> dict:
    """``{field: requirement_level}`` for ``datatype`` (empty for unknown/``None``)."""
    if datatype is None:
        return {}
    return dict(_sidecar_fields_table().get(datatype, {}))


def sidecar_template(datatype: "str | None") -> dict:
    """A ``None``-filled ``_<datatype>.json`` template for ``datatype``.

    Field names (in BIDS order) come from the vendored table; values are ``None``
    ("unset"). An unknown/``None`` datatype yields an empty dict. The keys make
    the valid fields discoverable (REPL, save-recording GUI form).
    """
    return {field: None for field in sidecar_fields(datatype)}
