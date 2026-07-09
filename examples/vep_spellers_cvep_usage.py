"""Compare c-VEP decoders on the bundled speller data: BWR vs template matching,
single band-pass vs filter bank.

Run:  python examples/vep_spellers_cvep_usage.py

c-VEP is the one speller paradigm where **both** scoring families apply to the *same*
data, so it is the natural place to compare them head to head:

* **BWR** (:class:`~medusa.pipelines.bci.vep_spellers.BWRLDAPipeline`) -- classify every
  code frame with an LDA, then correlate the frame scores against each command's shifted
  m-sequence.
* **TM** (:class:`~medusa.pipelines.bci.vep_spellers.TMCCAPipeline`,
  ``reference='template'``) -- learn one coherent-average template and a spatial filter
  from the calibration runs, then correlate the (1-D projected) cycle EEG against each
  command's circularly-shifted template.

For each method we also compare two front-ends:

* a **single band-pass** 1-70 Hz, and
* a **filter bank** of three sub-bands (1-70, 10-70, 30-70 Hz), fused before the
  classifier for BWR and combined with FBCCA weights for TM.

That makes a 2x2 grid. Every pipeline trains on the same calibration runs and decodes the
same test runs; we report decoding accuracy as a function of the number of stimulation
cycles (the standard c-VEP performance curve) as a table and a saved figure.
"""
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")            # headless-safe; drop this line to show the figure live
import matplotlib.pyplot as plt

from medusa.core.legacy.recording import Recording as LegacyRecording
from medusa.core.legacy.convert import cvep_recording_to_v2
from medusa.pipelines.bci.vep_spellers import (
    BWRLDAPipeline, TMCCAPipeline, VEPCommandDecoder,
    command_decoding_accuracy_per_cycle)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "cvep")


def band(cutoff, order=7):
    """A single band-pass IIR filter spec for the ``freq_filtering`` filter bank."""
    return {
        "filt_type": "iir",
        "band_type": "bandpass",
        "cutoff": list(cutoff),
        "order": order
    }


#: The two front-ends compared for both methods: one band vs a three-band bank.
FILTERINGS = {
    "single band 1-70 Hz": [band((1.0, 70.0))],
    "filter bank 1/10/30-70 Hz": [
        band((1.0, 70.0)),
        band((10.0, 70.0)),
        band((30.0, 70.0))
    ],
}
METHODS = ("BWR", "TM")

# --------------------------------------------------------------------------- #
# 1) Load + convert the calibration (train) and test runs, once. A c-VEP
#    calibration attends a single command, so the converter auto-derives each
#    calibration trial's target (no labels needed for training).
# --------------------------------------------------------------------------- #
train = [cvep_recording_to_v2(LegacyRecording.load(p))
         for p in sorted(glob.glob(os.path.join(DATA, "cvep_train*.cvep.bson")))]
test = [cvep_recording_to_v2(LegacyRecording.load(p))
        for p in sorted(glob.glob(os.path.join(DATA, "cvep_test*.cvep.bson")))]
channels = list(train[0].signals["eeg"].channel_set.labels)     # use the full montage
exp = train[0].experiment
print(f"Loaded {len(train)} calibration + {len(test)} test runs "
      f"({exp.fps_resolution:g} fps, {len(channels)} channels; test codebook = "
      f"{len(test[0].experiment.commands_info)} commands = shifts of one m-sequence).")

# The test target letters live in cvep_labels.txt (they are not stored in the recording).
# Map them to command uids via each command's displayed content and attach them once, so
# every configuration decodes against the same ground truth.
labels = open(os.path.join(DATA, "cvep_labels.txt")).read().strip()
cursor = 0
for rec in test:
    sd = rec.experiment
    content_to_uid = {cmd.content: uid for uid, cmd in sd.commands_info.items()}
    n_trials = len(sd.trial_available_cmmds)
    sd.spell_target = [content_to_uid[c] for c in labels[cursor:cursor + n_trials]]
    cursor += n_trials
print(f"Test target: {labels!r} ({cursor} trials across {len(test)} runs).")


def make_pipeline(method, filterbank):
    """Build a fresh BWR or TM pipeline for the given method and filter bank."""
    if method == "BWR":
        return BWRLDAPipeline(
            channels=channels,
            freq_filtering={"filterbank": filterbank},
            epoching={"w_segment_t": [0.0, 250.0], "baseline_t": [], "target_fs": 0.0})
    return TMCCAPipeline(
        channels=channels,
        freq_filtering={"filterbank": filterbank},
        reference={"mode": "template"})


def accuracy_curve(pipe):
    """Pooled decoding accuracy (%) as a function of #cycles, over all test runs.

    Decodes every test run, pools the per-trial cumulative selections into one set of
    trials, and returns ``command_decoding_accuracy_per_cycle`` in percent (entry ``k`` is
    the accuracy after ``k + 1`` cycles).
    """
    decoder = VEPCommandDecoder()
    per_cycle, target, base = {}, {}, 0
    for rec in test:
        sd = rec.experiment
        pc = decoder.decode(pipe.predict(rec), sd,
                            rec.events)["selected_commands_per_cycle"]
        for t in pc:                                    # re-key trials globally
            per_cycle[base + t] = pc[t]
            target[base + t] = sd.spell_target[t]
        base += len(pc)
    return 100.0 * command_decoding_accuracy_per_cycle(per_cycle, target)


# --------------------------------------------------------------------------- #
# 2) Train each of the four (method x filtering) pipelines on the calibration
#    runs and decode the pooled test trials into an accuracy-vs-cycles curve.
# --------------------------------------------------------------------------- #
print("\nTraining + decoding the 2x2 grid ...")
results = {}                                            # (method, filtering) -> curve
for method in METHODS:
    for fname, fb in FILTERINGS.items():
        pipe = make_pipeline(method, fb).fit(train)
        results[(method, fname)] = accuracy_curve(pipe)
        print(f"  done: {method:3s} | {fname}")

# --------------------------------------------------------------------------- #
# 3) Report: a compact summary table, the full per-cycle curves, and a figure.
# --------------------------------------------------------------------------- #
def cycles_to_full(curve):
    """First #cycles reaching 100% accuracy, or None if it never does."""
    hit = np.where(curve >= 100.0)[0]
    return int(hit[0]) + 1 if len(hit) else None


print("\nc-VEP decoding accuracy, pooled over test trials")
print("=" * 60)
print(f"{'method':<4} {'filtering':<26} {'1 cyc':>6} {'final':>6} {'->100%':>7}")
print("-" * 60)
for (method, fname), curve in results.items():
    full = cycles_to_full(curve)
    print(f"{method:<4} {fname:<26} {curve[0]:5.0f}% {curve[-1]:5.0f}% "
          f"{(str(full) if full else '--'):>7}")
print("=" * 60)

print("\nAccuracy (%) per number of cycles:")
for (method, fname), curve in results.items():
    print(f"  {method:3s} | {fname:<26}: "
          + " ".join(f"{a:3.0f}" for a in curve))

# One figure: colour = method, line style = filtering, so the 2x2 grid reads at a glance.
fig, ax = plt.subplots(figsize=(7.5, 4.5))
colors = {"BWR": "#1f77b4", "TM": "#d62728"}
styles = {"single band 1-70 Hz": "-", "filter bank 1/10/30-70 Hz": "--"}
for (method, fname), curve in results.items():
    x = np.arange(1, len(curve) + 1)
    ax.plot(x, curve, styles[fname], color=colors[method], marker="o", ms=4,
            label=f"{method} | {fname}")
ax.set_xlabel("number of stimulation cycles")
ax.set_ylabel("decoding accuracy (%)")
ax.set_title("c-VEP: BWR vs template matching, single band vs filter bank")
ax.set_ylim(0, 105)
ax.margins(x=0.02)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8, loc="lower right")
fig.tight_layout()
out = os.path.join(HERE, "vep_spellers_cvep_comparison.png")
fig.savefig(out, dpi=130)
print(f"\nSaved comparison figure -> {out}")
