"""Decode a (synthetic) SSVEP speller with the calibration-free ``TMCCAPipeline``.

Run:  python examples/vep_spellers_ssvep_usage.py

The template-matching sibling of ``vep_spellers_cvep_usage.py`` / ``vep_spellers_rcp_usage.py``.
Where those train a classifier, SSVEP needs **no calibration**: each command flickers at a
fixed frequency, and canonical correlation analysis (CCA) scores a cycle's EEG against every
command's sine/cosine harmonic reference -- the best match wins. So there is no training
data here; we just:

1. build an SSVEP codebook with
   :func:`~medusa.pipelines.bci.vep_spellers.generate_freq_codebook` (frame-locked flicker
   frequencies, one per command, stored in ``CommandInfo.extra['stim_freq']``);
2. synthesise an EEG recording whose occipital channels carry the attended command's flicker
   (fundamental + 2nd harmonic) buried in noise, several trials per command;
3. decode it with :class:`~medusa.pipelines.bci.vep_spellers.TMCCAPipeline` (no ``fit``) plus
   the paradigm-agnostic :class:`~medusa.pipelines.bci.vep_spellers.VEPCommandDecoder`, and
   report accuracy as a function of the number of cycles (table + figure).

**Only the calibration-free TM harmonics reference is shown**, since it is the SSVEP idiom:
``TMCCAPipeline.predict`` returns the cumulative ``(n_cycles, n_commands)`` canonical
correlations -- evidence accrues by coherently averaging the cycle segments -- and the
decoder just argmaxes them, exactly as for the BWR pipelines.
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")            # headless-safe; drop this line to show the figure live
import matplotlib.pyplot as plt

from medusa.core.data import BidsInfo, Recording, Signal, Events, ChannelSet
from medusa.pipelines.bci.vep_spellers import (
    generate_freq_codebook, TMCCAPipeline, zerocal_ssvep_settings,
    VEPCommandDecoder, SpellerData,
    command_decoding_accuracy_per_cycle)

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(0)

FPS = 60.0                  # display refresh rate the codebook is frame-locked to
FS = 250.0                  # EEG sampling rate
N_CMDS = 4                  # commands (one flicker frequency each)
N_REPS = 8                  # trials per command (more trials -> smoother accuracy curve)
T_STIM = 1.0                # seconds of stimulation per cycle
N_CYCLES = 8                # repetitions per trial
RESP_AMP = 0.25             # SSVEP response amplitude (low, so accuracy accrues over cycles)
CHANNELS = ["O1", "OZ", "O2", "POZ", "PO3", "PO4", "PZ", "CZ"]
OCCIPITAL = [c for c in CHANNELS if c != "CZ"]     # where the SSVEP response lives

# --------------------------------------------------------------------------- #
# 1) SSVEP codebook: N_CMDS commands at frame-locked frequencies in 8-15 Hz.
#    Each command carries its flicker frequency in extra['stim_freq'] -- all CCA needs.
# --------------------------------------------------------------------------- #
commands = generate_freq_codebook(N_CMDS, freq_range=(8.0, 15.0), t_stim=T_STIM,
                                  fps_resolution=FPS)
uids = list(commands)
freqs = {uid: cmd.extra["stim_freq"] for uid, cmd in commands.items()}
n_frames = np.asarray(commands[uids[0]].code).shape[1]      # = round(T_STIM * FPS)
print(f"Codebook: {N_CMDS} commands at frame-locked frequencies "
      f"{[round(freqs[u], 2) for u in uids]} Hz ({n_frames} frames/cycle).")

# --------------------------------------------------------------------------- #
# 2) Synthesise the EEG: N_REPS trials per command, N_CYCLES cycles each. During every
#    cycle the occipital channels carry the target's flicker (fundamental + 2nd harmonic,
#    phase-locked to the cycle onset) plus background noise.
# --------------------------------------------------------------------------- #
spell_target = list(uids) * N_REPS                         # each command attended N_REPS times
n_trials = len(spell_target)
spc = int(round(T_STIM * FS))                             # EEG samples per cycle
local_t = np.arange(spc) / FS
n_used = n_trials * N_CYCLES * spc
signal = 2.0 * rng.standard_normal((n_used + spc, len(CHANNELS)))   # +1 cycle tail pad
occ = [CHANNELS.index(c) for c in OCCIPITAL]

events = Events(optional_columns={"trial_idx": "Int64", "cycle_idx": "Int64",
                                  "code_idx": "Int64"})
rows = []
for trial, tgt in enumerate(spell_target):
    f = freqs[tgt]
    wave = np.sin(2 * np.pi * f * local_t) + 0.5 * np.sin(2 * np.pi * 2 * f * local_t)
    for cycle in range(N_CYCLES):
        i0 = (trial * N_CYCLES + cycle) * spc
        signal[i0:i0 + spc, occ] += RESP_AMP * wave[:, None]   # SSVEP response
        rows.append({"onset": i0 / FS, "duration": T_STIM,
                     "trial_idx": trial, "cycle_idx": cycle, "code_idx": 0})
events.append(rows)

channel_set = ChannelSet()
channel_set.add_unipolar_eeg_channels(CHANNELS)
rec = Recording(BidsInfo(subject="synthetic", task="ssvep"))
rec.add_signal("eeg", Signal(signal, fs=FS, channel_set=channel_set))
rec.set_events(events)
SpellerData(mode="test", paradigm_conf={}, commands_info=commands,
            fps_resolution=FPS, spell_target=spell_target).to_recording(rec)
print(f"Synthesised {n_trials} trials x {N_CYCLES} cycles of EEG "
      f"({len(CHANNELS)} channels @ {FS:g} Hz).")

# --------------------------------------------------------------------------- #
# 3) Decode -- calibration-free: construct and predict, no fit(). TMCCAPipeline.predict
#    returns the cumulative (n_cycles, n_commands) canonical correlations; the decoder
#    argmaxes them per cycle, just like the BWR pipelines.
# --------------------------------------------------------------------------- #
pipe = TMCCAPipeline(settings=zerocal_ssvep_settings(), channels=CHANNELS)
decoder = VEPCommandDecoder()

result = decoder.decode(pipe.predict(rec), SpellerData.from_recording(rec), rec.events)
per_cycle = result["selected_commands_per_cycle"]          # {trial: {cycle: uid}}
target = {t: spell_target[t] for t in range(n_trials)}
curve = 100.0 * command_decoding_accuracy_per_cycle(per_cycle, target)

# --------------------------------------------------------------------------- #
# 4) Report: accuracy vs #cycles as a table and a figure.
# --------------------------------------------------------------------------- #
hit = np.where(curve >= 100.0)[0]
to_full = int(hit[0]) + 1 if len(hit) else None

print(f"\nSSVEP CCA decoding ({n_trials} trials x {N_CYCLES} cycles), no calibration")
print("=" * 56)
print("  accuracy (%) / #cycles: " + " ".join(f"{a:3.0f}" for a in curve))
print(f"  1 cycle: {curve[0]:.0f}%   final: {curve[-1]:.0f}%   "
      f"cycles ->100%: {to_full if to_full else '--'}")
print("=" * 56)

fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(np.arange(1, len(curve) + 1), curve, "-o", color="#d62728", ms=4)
ax.set_xlabel("number of stimulation cycles")
ax.set_ylabel("decoding accuracy (%)")
ax.set_title("SSVEP speller: calibration-free CCA (harmonics) accuracy")
ax.set_ylim(0, 105)
ax.margins(x=0.02)
ax.grid(True, alpha=0.3)
fig.tight_layout()
out = os.path.join(HERE, "vep_spellers_ssvep_accuracy.png")
fig.savefig(out, dpi=130)
print(f"\nSaved figure -> {out}")
