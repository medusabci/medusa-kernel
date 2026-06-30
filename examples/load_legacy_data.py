"""Load legacy (1.x) MEDUSA recordings -- one file per paradigm.

Run:  python examples/load_legacy_data.py

The 2.0 kernel still *reads* recordings written by Kernel 1.x. Those files predate
the BIDS-aligned ``medusa.core.data`` model, so they load through the legacy reader
``medusa.core.legacy.recording.Recording`` (kept for loading only -- there is no
automatic 1.x -> 2.0 migration). This example loads one bundled recording from each
paradigm folder under ``examples/data/`` and prints what came back.

A 1.x ``Recording`` exposes its parts through *name registries*: ``rec.biosignals``
and ``rec.experiments`` are lists of attribute names, and the objects themselves
are attributes on the recording (``rec.eeg``, ``rec.erpspellerdata``, ...).
"""
from pathlib import Path

from medusa.core.legacy.recording import Recording

DATA = Path(__file__).resolve().parent / "data"


def section(title):
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


# One recording per paradigm folder under examples/data/ (first .bson in each).
paradigm_dirs = sorted(p for p in DATA.iterdir()
                       if p.is_dir() and any(p.glob("*.bson")))

for pdir in paradigm_dirs:
    rec_path = sorted(pdir.glob("*.bson"))[0]
    section(f"{pdir.name}  ({rec_path.name})")

    rec = Recording.load(str(rec_path))

    print("subject / date :", getattr(rec, "subject_id", "?") or "(none)",
          "/", getattr(rec, "date", "?"))

    # Biosignals (EEG here): each name in `rec.biosignals` is an attribute.
    for name in rec.biosignals:
        bio = getattr(rec, name)
        n_samples, n_cha = bio.signal.shape
        print(f"biosignal '{name}' : {type(bio).__name__} | "
              f"{n_cha} ch x {n_samples} samples @ {bio.fs} Hz "
              f"(~{n_samples / bio.fs:.0f} s)")

    # Experiment (the paradigm configuration): one per recording.
    for name in rec.experiments:
        exp = getattr(rec, name)
        mode = getattr(exp, "mode", None)
        print(f"experiment '{name}' : {type(exp).__name__}"
              + (f" (mode={mode})" if mode else ""))

print(f"\nLoaded {len(paradigm_dirs)} legacy recordings (one per paradigm).")
