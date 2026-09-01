"""Decode a legacy RCP / P300 speller with bit-wise reconstruction, comparing a shallow
LDA classifier against the deep EEG-Inception one.

Run:  python examples/vep_spellers_rcp_usage.py

The ERP/P300 sibling of ``vep_spellers_cvep_usage.py``. A row-column (RCP) speller flashes
rows and columns and re-randomises their order every cycle, so -- unlike c-VEP -- each
command owns a *multi-code* ``(n_codes, n_frames)`` codebook (one display-frame code per
cycle, picked by the events' ``code_idx``). Everything else matches c-VEP BWR: a classifier
scores each flash frame (target response present or not), then each command's score is the
correlation of its flashing code with those frame scores.

**Only BWR is shown here, on purpose.** Template matching (``TMCCAPipeline``) averages the
per-cycle EEG coherently, which assumes every cycle repeats the *same* stimulation pattern
-- true for c-VEP / SSVEP, but false for an RCP speller that reshuffles its flash order
each cycle, so coherent averaging would smear the time-locked P300. BWR, which epochs each
flash by its own onset, is the correct scoring family for re-randomised RCP.

Two BWR frame classifiers are compared: the shallow
:class:`~medusa.pipelines.bci.vep_spellers.BWRLDAPipeline` (LDA) and, when PyTorch is
installed, the deep :class:`~medusa.pipelines.bci.vep_spellers.BWREEGInceptionPipeline`
(EEG-Inception, ``arch='eeg_inception_v2'``) -- P300 is EEG-Inception's original design
target. Same BWR strategy, same epoch window; only the frame classifier changes. If PyTorch /
Lightning are absent the deep pipeline is skipped and the LDA walk-through still runs.

**Expect the deep net to lose here.** This bundled set has only 3 calibration runs (18
trials), far too few to fit a convolutional model, so the shrinkage-regularised LDA is the
stronger choice on *this* data -- the deep pipeline needs many more calibration trials to pay
off (where it does, it wins clearly -- see ``vep_spellers_cvep_usage.py``, which has more
data per command). The comparison is kept because choosing the classifier for the data at
hand is the point; it is not a claim that EEG-Inception is worse for P300 in general.

Walk-through: load + convert -> configure + train LDA -> save/reload one portable file ->
decode the test runs (spelled text per run) -> add the deep EEG-Inception decoder ->
report accuracy vs #cycles for both (table + figure).
"""
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")            # headless-safe; drop this line to show the figure live
import matplotlib.pyplot as plt

from medusa.core.legacy.recording import Recording as LegacyRecording
from medusa.core.legacy.convert import rcp_recording_to_v2
from medusa.pipelines.base import DecodingPipeline
from medusa.pipelines.bci.vep_spellers import (
    BWRLDAPipeline, VEPCommandDecoder, SpellerData,
    command_decoding_accuracy_per_cycle)

# Deep BWR (EEG-Inception) is torch-gated: import it only if PyTorch + Lightning are
# installed, otherwise the example still runs the shallow LDA walk-through. The class is
# resolved lazily, so this import is what triggers (and here tolerates) the torch check.
try:
    from medusa.pipelines.bci.vep_spellers import BWREEGInceptionPipeline
    HAS_TORCH = True
except ImportError:
    BWREEGInceptionPipeline = None
    HAS_TORCH = False

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "rcp_speller")

# --------------------------------------------------------------------------- #
# 1) Load + convert the calibration (train) runs. The legacy file stores the
#    per-trial target command, so the converter fills ``spell_target`` for us.
#    (Legacy runs were not frame-locked, so the converter quantizes each flash to
#    whole 60 Hz frames -- a best effort; pass ``fps`` / ``t_stim`` to override.)
# --------------------------------------------------------------------------- #
train = [rcp_recording_to_v2(LegacyRecording.load(p))
         for p in sorted(glob.glob(os.path.join(DATA, "S1_TRAIN_*.rcp.bson")))]
sd0 = SpellerData.from_recording(train[0])
n_cmd, n_codes, n_frames = sd0.codes.shape
print(f"Loaded {len(train)} calibration runs ({n_cmd} commands, {sd0.fps_resolution:g} fps). "
      f"Each command carries a ({n_codes}, {n_frames}) multi-code codebook -- one "
      f"display-frame code per stimulation cycle.")

# --------------------------------------------------------------------------- #
# 2) Configure + train the bit-wise-reconstruction pipeline (shallow LDA). A P300
#    speller wants a longer epoch (the ERP unfolds over ~800 ms) and a low-pass band.
# --------------------------------------------------------------------------- #
channels = list(train[0].signals["eeg"].channel_set.labels)    # use the full montage
pipe = BWRLDAPipeline(
    channels=channels,
    freq_filtering={"filterbank": [
        {"filt_type": "iir", "band_type": "bandpass", "cutoff": [0.5, 16.0], "order": 5}]},
    segmentation={"w_segment_t": [0.0, 800.0], "baseline_t": [-200.0, 0.0], "target_fs": 20.0},
)
print(f"\nPipeline config: {pipe.cfg}")
pipe.fit(train)
print("Trained BWRLDAPipeline on the calibration runs.")

# --------------------------------------------------------------------------- #
# 3) Persist + reload the trained pipeline (one portable file, CPU-safe).
# --------------------------------------------------------------------------- #
model_path = os.path.join(HERE, "rcp_bwr_lda.pkl")
pipe.save(model_path)
pipe = DecodingPipeline.load(model_path)                       # polymorphic load
print(f"Saved + reloaded the pipeline ({type(pipe).__name__}).")

# --------------------------------------------------------------------------- #
# 4) Decode the test runs. ``decode`` returns the cumulative selection after each cycle,
#    so we can show both the spelled text per run and the accuracy vs #cycles curve
#    (pooled over runs). The number of cycles is the standard speller performance axis.
#    The decode + pool step is model-agnostic, so we reuse it for both classifiers.
# --------------------------------------------------------------------------- #
decoder = VEPCommandDecoder()
test_paths = sorted(glob.glob(os.path.join(DATA, "S1_TEST_*.rcp.bson")))
test = [rcp_recording_to_v2(LegacyRecording.load(p)) for p in test_paths]


def pooled_per_cycle(pipe, verbose=False):
    """Decode every test run with ``pipe``; pool the per-cycle selections across runs.

    Returns ``(per_cycle, target)`` re-keyed to global trial indices (ready for
    ``command_decoding_accuracy_per_cycle``). With ``verbose`` it also prints the spelled
    text per run (first cycle vs all cycles).
    """
    per_cycle, target, base = {}, {}, 0
    for path, rec in zip(test_paths, test):
        sd = SpellerData.from_recording(rec)
        n_trials = len(sd.spell_target)
        tgt = {t: sd.spell_target[t] for t in range(n_trials)}
        pc = decoder.decode(pipe.predict(rec), sd,
                            rec.events)["selected_commands_per_cycle"]
        if verbose:
            n_cyc = int(rec.events.df["cycle_idx"].max()) + 1
            tgt_text = "".join(sd.commands_info[u].content for u in sd.spell_target)
            first = "".join(sd.commands_info[pc[t][0]].content for t in range(n_trials))
            final = "".join(sd.commands_info[pc[t][n_cyc - 1]].content
                            for t in range(n_trials))
            acc = command_decoding_accuracy_per_cycle(pc, tgt)
            print(f"{os.path.basename(path)}  ({n_trials} trials x {n_cyc} cycles)")
            print(f"  target          : {tgt_text!r}")
            print(f"  after  1 cycle  : {first!r}  ({100 * acc[0]:.0f}%)")
            print(f"  after {n_cyc:2d} cycles : {final!r}  ({100 * acc[-1]:.0f}%)")
        for t in pc:                                           # re-key trials globally
            per_cycle[base + t] = pc[t]
            target[base + t] = tgt[t]
        base += n_trials
    return per_cycle, target


print("\nPer-run spelling (BWR-LDA)")
print("=" * 62)
lda_pc, lda_target = pooled_per_cycle(pipe, verbose=True)
print("=" * 62)
os.remove(model_path)

curves = {"BWR-LDA": 100.0 * command_decoding_accuracy_per_cycle(lda_pc, lda_target)}
n_test_trials = len(lda_target)

# --------------------------------------------------------------------------- #
# 5) Swap the LDA frame classifier for a deep EEG-Inception one (same BWR strategy,
#    same epoch window) and decode the same test runs. P300 is EEG-Inception's design
#    target; a P300 epoch wants a longer window (~800 ms), which we resample to 128 Hz
#    for the conv net. Training is kept brief with early stopping so the example runs on
#    CPU; skipped cleanly when torch is unavailable.
# --------------------------------------------------------------------------- #
if HAS_TORCH:
    print("\nTraining the deep BWR pipeline (EEG-Inception v2; CPU, may take a minute) ...")
    dl_pipe = BWREEGInceptionPipeline(
        channels=channels,
        freq_filtering={"filterbank": [
            {"filt_type": "iir", "band_type": "bandpass", "cutoff": [0.5, 16.0], "order": 5}]},
        segmentation={"w_segment_t": [0.0, 800.0], "baseline_t": [-200.0, 0.0],
                  "target_fs": 128.0},
        classifier={"arch": "eeg_inception_v2",
                    "training": {"max_epochs": 25, "batch_size": 512,
                                 "val_split": 0.2, "patience": 5, "verbose": "epoch"}})
    dl_pipe.fit(train)
    dl_pc, dl_target = pooled_per_cycle(dl_pipe)
    curves["BWR-EEGInception"] = 100.0 * command_decoding_accuracy_per_cycle(dl_pc, dl_target)
    print("Decoded the test runs with the deep pipeline.")
else:
    print("\nSkipping deep BWR (EEG-Inception): PyTorch / Lightning not installed.")

# --------------------------------------------------------------------------- #
# 6) Pooled accuracy vs #cycles for each classifier: a compact summary + a figure.
# --------------------------------------------------------------------------- #
def cycles_to_full(curve):
    """First #cycles reaching 100% accuracy, or None if it never does."""
    hit = np.where(curve >= 100.0)[0]
    return int(hit[0]) + 1 if len(hit) else None


print(f"\nRCP / P300 decoding accuracy, pooled over {n_test_trials} test trials")
print("=" * 52)
print(f"{'classifier':<18}{'1 cyc':>7}{'final':>7}{'->100%':>9}")
print("-" * 52)
for name, curve in curves.items():
    full = cycles_to_full(curve)
    print(f"{name:<18}{curve[0]:6.0f}%{curve[-1]:6.0f}%{(str(full) if full else '--'):>9}")
print("=" * 52)

print("\nAccuracy (%) per number of cycles:")
for name, curve in curves.items():
    print(f"  {name:<18}: " + " ".join(f"{a:3.0f}" for a in curve))

if "BWR-EEGInception" in curves and curves["BWR-EEGInception"][-1] < curves["BWR-LDA"][-1]:
    print(f"\nNote: the deep pipeline trails the LDA here "
          f"({curves['BWR-EEGInception'][-1]:.0f}% vs {curves['BWR-LDA'][-1]:.0f}% final) -- "
          f"{len(train)} calibration runs is too few to fit a conv net. Deep BWR pays off "
          f"with more data (see vep_spellers_cvep_usage.py, where it beats the LDA).")

fig, ax = plt.subplots(figsize=(7, 4.2))
colors = {"BWR-LDA": "#1f77b4", "BWR-EEGInception": "#2ca02c"}
for name, curve in curves.items():
    ax.plot(np.arange(1, len(curve) + 1), curve, "-o", color=colors[name], ms=4, label=name)
ax.set_xlabel("number of stimulation cycles")
ax.set_ylabel("decoding accuracy (%)")
ax.set_title("RCP / P300 speller: BWR decoding (LDA vs EEG-Inception)")
ax.set_ylim(0, 105)
ax.margins(x=0.02)
ax.grid(True, alpha=0.3)
if len(curves) > 1:
    ax.legend(loc="lower right")
fig.tight_layout()
out = os.path.join(HERE, "vep_spellers_rcp_accuracy.png")
fig.savefig(out, dpi=130)
print(f"\nSaved figure -> {out}")
