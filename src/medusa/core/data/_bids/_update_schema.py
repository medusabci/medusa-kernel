#!/usr/bin/env python3
"""Dev-time updater for the vendored BIDS schema.

Downloads the BIDS schema JSON from the bids-specification ReadTheDocs build and
saves it next to this script as ``bids_schema.json`` -- the single master copy of
the BIDS standard the kernel vendors. Everything else under ``_bids`` (channel
types, the per-datatype sidecar field table, ...) is derived from it.

No external dependencies beyond the Python standard library.

Usage::

    python -m medusa.core.data._bids._update_schema

    # A specific spec version or a BEP preview:
    python -m medusa.core.data._bids._update_schema \
        --schema-url https://bids-specification.readthedocs.io/en/v1.11.0/schema.json

After updating, regenerate the derived tables::

    python -m medusa.core.data._bids._create_bids_sidecar_fields
"""

import argparse
import json
import urllib.request
from pathlib import Path

SCHEMA_URL = "https://bids-specification.readthedocs.io/en/stable/schema.json"
_OUTPUT = Path(__file__).with_name("bids_schema.json")


def update_schema(url: str) -> None:
    """Download ``schema.json`` from ``url`` and vendor it as ``bids_schema.json``."""
    print(f"Fetching {url} ...")
    req = urllib.request.Request(
        url, headers={"User-Agent": "medusa-bids-updater/1.0"})
    with urllib.request.urlopen(req) as resp:
        schema = json.loads(resp.read())
    with open(_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)
        f.write("\n")
    print(f"  -> {_OUTPUT.name}: schema {schema.get('schema_version', '?')} / "
          f"BIDS {schema.get('bids_version', '?')}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--schema-url", default=SCHEMA_URL,
                        help=f"URL for schema.json (default: {SCHEMA_URL})")
    args = parser.parse_args()
    update_schema(args.schema_url)
    print("Done. Re-run _create_bids_sidecar_fields to refresh the derived tables.")


if __name__ == "__main__":
    main()
