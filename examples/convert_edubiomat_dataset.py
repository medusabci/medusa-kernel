"""Bulk-convert a mixed legacy (1.x) dataset to 2.0 ``Recording`` files.

Run:  python examples/convert_edubiomat_dataset.py

``examples/data/edubiomat/`` is a real drop from two different MEDUSA apps, mixed in
one tree:

* the **edubiomat** app -- an educational test (images or questions) recorded
  trial by trial;
* the **Recorder** app -- plain runs (baseline, chronometer on/off) annotated by
  hand with condition and event marks.

Both write the same ``.rec.*`` container, so the app is told apart by the *experiment*
inside the file, not by its name or extension:

* an edubiomat file has an experiment with ``app_settings["test"]`` naming a mode
  (``"images"`` / ``"questions"``) and a per-trial ``data`` list;
* a Recorder file has an experiment carrying the manual mark arrays
  (``conditions_times`` / ``events_times`` / ...).

Each kind then goes through its own converter --
:func:`~medusa.core.legacy.convert.edubiomat_recording_to_v2` or
:func:`~medusa.core.legacy.convert.recorder_recording_to_v2` -- and the result is
written under ``converted/``, mirroring the original folder structure. Output is
HDF5: it stores the sample arrays natively, so a converted run takes about a quarter
of the space of the ``.bson`` it came from.

The BIDS identity of each file comes from :func:`bids_entities` below. That is the
one dataset-specific piece: these legacy files carry no ``subject_id``, so the
participant is read off the folder name. Retune that function for a differently
organised drop.
"""
import warnings
from pathlib import Path

from medusa.core.legacy.convert import (EDUBIOMAT_TASK_LABELS,
                                        edubiomat_recording_to_v2,
                                        recorder_recording_to_v2)
from medusa.core.legacy.recording import Recording as LegacyRecording

SRC = Path(__file__).resolve().parent / "data" / "edubiomat"
DST = SRC / "converted"
FORMAT = "h5"          # any Recording.save format: h5 / bson / json / mat

# Mark arrays written by the Recorder app; any one of them identifies its experiment.
MARK_ATTRS = ("conditions_times", "conditions_labels",
              "events_times", "events_labels")


def detect_app(legacy_recording):
    """Name the app that wrote a legacy recording: ``"edubiomat"``, ``"recorder"``.

    Both apps store their configuration in a generic ``CustomExperimentData``, so the
    class name says nothing -- what differs is the payload. Returns ``None`` when the
    file matches neither app, so the caller can skip it instead of guessing.
    """
    for name in legacy_recording.experiments:
        experiment = getattr(legacy_recording, name)
        settings = getattr(experiment, "app_settings", None) or {}
        test = settings.get("test") if isinstance(settings, dict) else None
        if hasattr(experiment, "data") and test in EDUBIOMAT_TASK_LABELS:
            return "edubiomat"
        if any(hasattr(experiment, attr) for attr in MARK_ATTRS):
            return "recorder"
    return None


def bids_label(value):
    """Reduce a free-form name to a valid BIDS label (letters and digits only)."""
    return "".join(ch for ch in str(value) if ch.isalnum())


def bids_entities(rel_path, app):
    """BIDS entities for one file, read off its place in the tree.

    This dataset is laid out as ``<study>/<participant>/[<block>/]<name>.rec.*``,
    with a couple of loose files at the root that belong to no study. So the
    participant is the second directory level, and for a Recorder run the file name
    is the condition it recorded (``LINEA-BASE``, ``SI-CRONO``, ...) -- a better
    ``task`` than the converter's generic ``"rest"`` default. An edubiomat run needs
    no ``task``: the converter labels it after the mode the file holds.
    """
    folders = rel_path.parts[:-1]
    entities = {}
    if len(folders) >= 2:
        entities["subject"] = folders[1]      # converters sanitize this one
    if app == "recorder":
        entities["task"] = bids_label(rel_path.name.split(".rec.")[0])
    return entities


def convert(path):
    """Load one legacy file, convert it, and return the 2.0 recording (or ``None``)."""
    # Legacy files predate several module renames, and their custom streams have no
    # channel metadata; the reader warns about both on every load.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        legacy = LegacyRecording.load(str(path))

    app = detect_app(legacy)
    if app is None:
        return None, None

    entities = bids_entities(path.relative_to(SRC), app)
    converter = {"edubiomat": edubiomat_recording_to_v2,
                 "recorder": recorder_recording_to_v2}[app]
    return app, converter(legacy, **entities)


# Every legacy container in the tree, skipping anything already converted. The
# companion .edf / .xlsx exports next to them are ignored: the .rec.* file is the
# complete recording, and they are views of it.
sources = sorted(p for p in SRC.rglob("*.rec.*") if DST not in p.parents)
print(f"{len(sources)} legacy recordings under {SRC}\n")

converted, skipped = 0, []
for path in sources:
    rel = path.relative_to(SRC)
    app, recording = convert(path)
    if recording is None:
        skipped.append(rel)
        print(f"[skip] {rel}  (no edubiomat or recorder experiment)")
        continue

    out = DST / rel.parent / f"{rel.name.split('.rec.')[0]}.{FORMAT}"
    out.parent.mkdir(parents=True, exist_ok=True)
    recording.save(str(out))
    converted += 1

    streams = ", ".join(f"{key} ({signal.n_channels}ch @ {signal.fs:g}Hz)"
                        for key, signal in recording.data.items())
    n_events = 0 if recording.events is None else len(recording.events)
    print(f"[{app:9}] {rel}\n"
          f"{'':12}-> {out.relative_to(SRC)}  "
          f"({out.stat().st_size / 1e6:.1f} MB, {n_events} events)\n"
          f"{'':12}   {recording.bids.basename()} | {streams}")

print(f"\nConverted {converted} recordings into {DST}"
      + (f"; skipped {len(skipped)}." if skipped else "."))

# Read one back to show the round-trip is complete.
if converted:
    from medusa.core.data import Recording

    sample = sorted(DST.rglob(f"*.{FORMAT}"))[0]
    back = Recording.load(str(sample))
    print(f"\nReloaded {sample.relative_to(SRC)}:\n  {back}")
    if back.events is not None:
        print(back.events.to_dataframe().head(3).to_string(index=False))
