"""Versioned on-disk format for serialized
:class:`~medusa.core.data.recording.Recording` objects.

Every serialized recording carries a ``schema_version`` so a loader can tell which
format wrote it. :func:`validate_recording_dict` checks that version (and the
required top-level keys) before the dict is rebuilt, turning an incompatible file
into a clear error instead of a confusing failure deep in deserialization.

Versioning starts at ``2`` (the v2.0 BIDS-aligned model); there is no support for
pre-2.0 (1.x) files -- their structure is too heterogeneous to migrate reliably.
Migrations for *future* versions (``n -> n+1``) are added by the kernel release
that introduces them, branching at the ``version < SCHEMA_VERSION`` case below.
"""

__all__ = ["SCHEMA_VERSION", "validate_recording_dict"]

#: On-disk schema version stamped into every serialized ``Recording``.
SCHEMA_VERSION = 2

#: Top-level keys a serialized ``Recording`` dict must contain.
_REQUIRED_KEYS = ("bids", "data")


def validate_recording_dict(data: dict) -> None:
    """Validate a serialized ``Recording`` dict's version and shape.

    Called by :meth:`Recording.from_serializable_obj` before rebuilding, so an
    unreadable file fails at the door with a clear message.

    Parameters
    ----------
    data
        The dict produced by :meth:`Recording.to_serializable_obj`.

    Raises
    ------
    ValueError
        If ``schema_version`` is missing or older than :data:`SCHEMA_VERSION`
        (pre-2.0 / unsupported -- no migration), newer (written by a later kernel
        release), or if a required top-level key is absent.
    """
    version = data.get("schema_version")
    if version is None or version < SCHEMA_VERSION:
        raise ValueError(
            f"unsupported recording schema version {version!r}: this medusa-kernel "
            f"reads v{SCHEMA_VERSION}, and pre-2.0 (1.x) files are not supported.")
    if version > SCHEMA_VERSION:
        raise ValueError(
            f"recording schema v{version} was written by a newer medusa-kernel; "
            f"this release reads v{SCHEMA_VERSION} -- please upgrade medusa-kernel.")
    missing = [k for k in _REQUIRED_KEYS if k not in data]
    if missing:
        raise ValueError(
            f"serialized Recording is missing required key(s): {missing}.")