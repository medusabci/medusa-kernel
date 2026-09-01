"""Decode real 2-class motor imagery (left vs right hand) with ``MICSPLDAPipeline``.

Run:  python examples/motor_decoding_mi_usage.py

The motor-decoding counterpart of the ``vep_spellers_*`` examples, on the bundled real
recordings in ``examples/data/mi/`` (a 16-channel, 256 Hz motor-imagery session; left- vs
right-hand imagery). Unlike a speller there is no codebook and no command decoder: each trial
is an independent, labelled segment, and the pipeline classifies it directly.

Steps:

1. load the legacy ``.mi.bson`` runs and convert them to 2.0 recordings with
   :func:`~medusa.core.legacy.convert.mi_recording_to_v2` -- each cue becomes one
   :class:`~medusa.core.data.events.Events` row (``trial_idx`` + integer ``label``), the
   shared trial contract of :mod:`~medusa.pipelines.bci.trial_events`;
2. train :class:`~medusa.pipelines.bci.motor_decoding.MICSPLDAPipeline` (8-30 Hz band-pass +
   Common Spatial Patterns + regularised LDA) on the calibration runs;
3. save it and reload it with the polymorphic
   :meth:`~medusa.pipelines.base.DecodingPipeline.load`;
4. decode the held-out test runs -- ``predict`` returns the per-trial ``(n_trials, n_classes)``
   class scores, whose argmax is the decoded class -- and report accuracy + a confusion matrix;
5. cross-validate with :func:`~medusa.pipelines.base.leave_one_recording_out` for a robust
   estimate (a single 4/4 split over 80 trials is noisy).
"""
import glob
import os
import tempfile

import numpy as np
import matplotlib
matplotlib.use("Agg")            # headless-safe; drop this line to show the figure live
import matplotlib.pyplot as plt

from medusa.core.legacy.recording import Recording as LegacyRecording
from medusa.core.legacy.convert import mi_recording_to_v2
from medusa.pipelines.base import DecodingPipeline, leave_one_recording_out
from medusa.pipelines.bci.motor_decoding import MICSPLDAPipeline
from medusa.pipelines.bci.trial_events import trial_labels

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "mi")

#: A central montage for sensorimotor-rhythm decoding (a subset of the 16-channel cap). CSP
#: learns the left/right spatial contrast, so CAR stays off (it would rank-deficient the CSP).
CHANNELS = ["F3", "FZ", "F4", "C3", "CZ", "C4", "CPZ", "P3", "PZ", "P4"]


def make_pipeline():
    """A fresh CSP + regularised-LDA motor decoder with sensible mu/beta defaults.

    The motor imagery is read in a 0.5-2.5 s window after each cue (the movement-imagery
    burst, before the later feedback phase dilutes it), band-passed to 8-30 Hz; four CSP
    filters + log-variance features feed a regularised LDA.
    """
    return MICSPLDAPipeline(
        channels=CHANNELS,
        filter={"cutoff": [8.0, 30.0], "order": 5},
        segmentation={"w_segment_t": [500.0, 2500.0], "baseline_t": [], "target_fs": 64.0},
        csp={"n_filters": 4, "selection": "extremes"})


def decode(pipe, recording):
    """Return ``(y_true, y_pred)`` for one recording: argmax of the per-trial class scores."""
    y_pred = pipe.clf.classes_[pipe.predict(recording).argmax(axis=1)]
    return trial_labels(recording), y_pred


# --------------------------------------------------------------------------- #
# 1) Load + convert the calibration (train) and held-out (test) runs. Every cue
#    becomes one labelled trial event; motor decoding needs no data class.
# --------------------------------------------------------------------------- #
train = [mi_recording_to_v2(LegacyRecording.load(p))
         for p in sorted(glob.glob(os.path.join(DATA, "train_*.mi.bson")))]
test = [mi_recording_to_v2(LegacyRecording.load(p))
        for p in sorted(glob.glob(os.path.join(DATA, "test_*.mi.bson")))]

label_names = train[0].experiment["label_names"]         # {0: 'Left_hand', 1: 'Right_hand'}
fs = train[0].signals["eeg"].fs
n_train = sum(len(trial_labels(r)) for r in train)
n_test = sum(len(trial_labels(r)) for r in test)
print(f"Loaded {len(train)} calibration ({n_train} trials) + {len(test)} test ({n_test} "
      f"trials) runs; {len(CHANNELS)} channels @ {fs:g} Hz. Classes: {label_names}")

# --------------------------------------------------------------------------- #
# 2) Train on the calibration runs, then 3) save + reload (portable single-file
#    bundle: settings + fitted CSP + LDA).
# --------------------------------------------------------------------------- #
pipe = make_pipeline().fit(train)
path = os.path.join(tempfile.gettempdir(), "mi_csp_lda.pkl")
pipe.save(path)
pipe = DecodingPipeline.load(path)                       # polymorphic reload
print(f"Trained + saved + reloaded {type(pipe).__name__} "
      f"(CSP {pipe.cfg['csp']['n_filters']} filters + rLDA).")

# --------------------------------------------------------------------------- #
# 4) Decode the held-out runs: per-run + pooled accuracy and a confusion matrix.
# --------------------------------------------------------------------------- #
run_names = [os.path.basename(p).replace(".mi.bson", "")
             for p in sorted(glob.glob(os.path.join(DATA, "test_*.mi.bson")))]
run_acc, y_true, y_pred = [], [], []
print("\nHeld-out decoding (train on calibration, test on the 4 held-out runs)")
print("=" * 46)
for name, rec in zip(run_names, test):
    yt, yp = decode(pipe, rec)
    run_acc.append(100.0 * np.mean(yt == yp))
    y_true.append(yt)
    y_pred.append(yp)
    print(f"  {name:<20} {run_acc[-1]:5.1f}%  ({len(yt)} trials)")
y_true, y_pred = np.concatenate(y_true), np.concatenate(y_pred)
overall = 100.0 * np.mean(y_true == y_pred)
print("-" * 46)
print(f"  {'pooled':<20} {overall:5.1f}%  ({len(y_true)} trials)")
print("=" * 46)

classes = list(pipe.clf.classes_)
names = [label_names.get(int(c), str(c)) for c in classes]
confusion = np.zeros((len(classes), len(classes)), dtype=int)
for t, p in zip(y_true, y_pred):
    confusion[classes.index(t), classes.index(p)] += 1
print("\nConfusion matrix (rows = true, cols = predicted):")
print("              " + "  ".join(f"{n[:9]:>9}" for n in names))
for i, n in enumerate(names):
    print(f"  {n[:11]:<11} " + "  ".join(f"{confusion[i, j]:9d}" for j in range(len(names))))

# --------------------------------------------------------------------------- #
# 5) Robust estimate: leave-one-recording-out over all 8 runs (every run tested once).
# --------------------------------------------------------------------------- #
cv_true, cv_pred = [], []
for tr, te in leave_one_recording_out(train + test):
    yt, yp = decode(make_pipeline().fit(tr), te)
    cv_true.append(yt)
    cv_pred.append(yp)
cv_true, cv_pred = np.concatenate(cv_true), np.concatenate(cv_pred)
print(f"\nLeave-one-recording-out over all 8 runs: "
      f"{100.0 * np.mean(cv_true == cv_pred):.1f}%  ({len(cv_true)} trials)")

# --------------------------------------------------------------------------- #
# 6) Figure: per-run held-out accuracy bars + the pooled confusion matrix.
# --------------------------------------------------------------------------- #
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
ax1.bar(run_names, run_acc, color="#1f77b4")
ax1.axhline(overall, color="#d62728", ls="--", label=f"pooled {overall:.0f}%")
ax1.axhline(50.0, color="gray", ls=":", label="chance")
ax1.set_ylabel("accuracy (%)")
ax1.set_title("Held-out motor-imagery accuracy per run")
ax1.set_ylim(0, 105)
ax1.legend(fontsize=8)
ax1.tick_params(axis="x", labelrotation=30)

ax2.imshow(confusion, cmap="Blues")
ax2.set_xticks(range(len(names)), names, rotation=30, ha="right")
ax2.set_yticks(range(len(names)), names)
ax2.set_xlabel("predicted")
ax2.set_ylabel("true")
ax2.set_title("Pooled confusion matrix")
for i in range(len(names)):
    for j in range(len(names)):
        ax2.text(j, i, confusion[i, j], ha="center", va="center",
                 color="white" if confusion[i, j] > confusion.max() / 2 else "black")
fig.tight_layout()
out = os.path.join(HERE, "motor_decoding_mi_accuracy.png")
fig.savefig(out, dpi=130)
print(f"\nSaved figure -> {out}")
