"""Decode a legacy RCP / P300 speller with bit-wise reconstruction (BWRLDAPipeline).

Run:  python examples/vep_spellers_rcp_usage.py

The ERP/P300 sibling of ``vep_spellers_cvep_usage.py``. A row-column (RCP) speller flashes
rows and columns and re-randomises their order every cycle, so -- unlike c-VEP -- each
command owns a *multi-code* ``(n_codes, n_frames)`` codebook (one display-frame code per
cycle, picked by the events' ``code_idx``). Everything else matches c-VEP BWR: an LDA
classifies each flash frame (target response present or not), then each command's score is
the correlation of its flashing code with those frame scores.

**Only BWR is shown here, on purpose.** Template matching (``TMCCAPipeline``) averages the
per-cycle EEG coherently, which assumes every cycle repeats the *same* stimulation pattern
-- true for c-VEP / SSVEP, but false for an RCP speller that reshuffles its flash order
each cycle, so coherent averaging would smear the time-locked P300. BWR, which epochs each
flash by its own onset, is the correct scoring family for re-randomised RCP.

Walk-through: load + convert -> configure + train -> save/reload one portable file ->
decode the test runs -> report spelled text and accuracy vs #cycles (table + figure).
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
    epoching={"w_segment_t": [0.0, 800.0], "baseline_t": [-200.0, 0.0], "target_fs": 20.0},
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
# --------------------------------------------------------------------------- #
decoder = VEPCommandDecoder()
per_cycle, target, base = {}, {}, 0                            # pooled across runs

print("\nPer-run spelling")
print("=" * 62)
for path in sorted(glob.glob(os.path.join(DATA, "S1_TEST_*.rcp.bson"))):
    rec = rcp_recording_to_v2(LegacyRecording.load(path))
    sd = SpellerData.from_recording(rec)
    n_trials = len(sd.spell_target)
    tgt = {t: sd.spell_target[t] for t in range(n_trials)}
    tgt_text = "".join(sd.commands_info[u].content for u in sd.spell_target)

    pc = decoder.decode(pipe.predict(rec), sd, rec.events)["selected_commands_per_cycle"]
    n_cyc = int(rec.events.df["cycle_idx"].max()) + 1
    first = "".join(sd.commands_info[pc[t][0]].content for t in range(n_trials))
    final = "".join(sd.commands_info[pc[t][n_cyc - 1]].content for t in range(n_trials))
    acc = command_decoding_accuracy_per_cycle(pc, tgt)

    print(f"{os.path.basename(path)}  ({n_trials} trials x {n_cyc} cycles)")
    print(f"  target          : {tgt_text!r}")
    print(f"  after  1 cycle  : {first!r}  ({100 * acc[0]:.0f}%)")
    print(f"  after {n_cyc:2d} cycles : {final!r}  ({100 * acc[-1]:.0f}%)")

    for t in pc:                                               # re-key trials globally
        per_cycle[base + t] = pc[t]
        target[base + t] = tgt[t]
    base += n_trials
print("=" * 62)

# --------------------------------------------------------------------------- #
# 5) Pooled accuracy vs #cycles: a compact summary + a figure.
# --------------------------------------------------------------------------- #
os.remove(model_path)
curve = 100.0 * command_decoding_accuracy_per_cycle(per_cycle, target)
hit = np.where(curve >= 100.0)[0]
to_full = int(hit[0]) + 1 if len(hit) else None

print(f"\nPooled over {base} test trials:")
print("  accuracy (%) / #cycles: " + " ".join(f"{a:3.0f}" for a in curve))
print(f"  1 cycle: {curve[0]:.0f}%   final: {curve[-1]:.0f}%   "
      f"cycles ->100%: {to_full if to_full else '--'}")

fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(np.arange(1, len(curve) + 1), curve, "-o", color="#1f77b4", ms=4)
ax.set_xlabel("number of stimulation cycles")
ax.set_ylabel("decoding accuracy (%)")
ax.set_title("RCP / P300 speller: bit-wise-reconstruction decoding accuracy")
ax.set_ylim(0, 105)
ax.margins(x=0.02)
ax.grid(True, alpha=0.3)
fig.tight_layout()
out = os.path.join(HERE, "vep_spellers_rcp_accuracy.png")
fig.savefig(out, dpi=130)
print(f"\nSaved figure -> {out}")
