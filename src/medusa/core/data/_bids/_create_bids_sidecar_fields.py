"""Dev-time generator for the vendored BIDS sidecar field table.

Reads the vendored BIDS schema (``_bids/bids_schema.json``, refreshed by
``_update_schema``) and writes ``bids_sidecar_fields.json`` next to this script:
an ordered ``{datatype: {field: level}}`` map for the datatypes MEDUSA can emit
as a ``Signal`` stream. The package loads that small JSON at runtime (see
``_bids.__init__``); the full schema is not loaded at runtime.

Re-run this whenever the schema is updated::

    python -m medusa.core.data._bids._create_bids_sidecar_fields [schema.json]

Mirrors the ``eeg/_create_*`` montage generators: a small script that vendors a
data file, kept beside its output.
"""

import json
import sys
from pathlib import Path

# Datatypes MEDUSA can emit as a `Signal` stream (electrophysiology, NIRS, motion).
_DATATYPES = ("eeg", "ieeg", "meg", "nirs", "emg", "motion")

# Master schema vendored alongside this script (see _update_schema).
_DEFAULT_SCHEMA = Path(__file__).with_name("bids_schema.json")
_OUTPUT = Path(__file__).with_name("bids_sidecar_fields.json")


def _field_level(spec) -> str:
    """Requirement level of a schema field spec (a bare string or a dict)."""
    if isinstance(spec, str):
        return spec
    return spec.get("level", "optional")


def _datatype_fields(sidecar_rules: dict, datatype: str) -> "dict[str, str]":
    """Ordered ``{field: level}`` for one datatype, deduped across groups.

    A schema field name may carry a ``__<suffix>`` disambiguator (e.g.
    ``SamplingFrequency__nirs``); the real BIDS key is the part before ``__``.
    The first occurrence wins (groups are listed hardware -> task -> required
    -> recommended -> optional).
    """
    out: "dict[str, str]" = {}
    for group in sidecar_rules[datatype].values():
        for raw_name, spec in group.get("fields", {}).items():
            name = raw_name.split("__", 1)[0]
            if name not in out:
                out[name] = _field_level(spec)
    return out


def main(schema_path: Path = _DEFAULT_SCHEMA) -> None:
    schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    sidecar_rules = schema["rules"]["sidecars"]
    table = {dt: _datatype_fields(sidecar_rules, dt) for dt in _DATATYPES}
    _OUTPUT.write_text(
        json.dumps(table, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {_OUTPUT} "
          f"({', '.join(f'{dt}:{len(f)}' for dt, f in table.items())})")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else _DEFAULT_SCHEMA)