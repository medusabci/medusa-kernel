"""Decode the bundled c-VEP speller with the calibrated template ``TMCCAPipeline``.

Run:  python examples/cvep_tmcca_usage.py

The **template-matching** counterpart of ``cvep_evoked_usage.py`` (which decodes the same
data with bit-wise reconstruction). A c-VEP calibration attends a single command -- the base
m-sequence -- and every other command's code is a circular shift of it. So:

1. ``TMCCAPipeline(reference={"mode": "template"})`` learns, from the calibration runs, one template
   for the attended command **and** a spatial filter that projects it to 1-D (a full
   multichannel template would let CCA align any command, washing out discrimination);
2. at test time it decodes all 16 commands by circularly shifting that learned 1-D template by
   each command's code lag and correlating it with the (1-D projected) EEG -- the classic
   c-VEP circular-shifting decoder;
3. ``TMCCAPipeline.predict`` returns the cumulative ``(n_cycles, n_commands)`` correlations and
   the paradigm-agnostic ``VEPCommandDecoder`` selects the command per cycle -- exactly as for
   the BWR pipelines.
"""
import glob
import os

from medusa.core.legacy.recording import Recording as LegacyRecording
from medusa.core.legacy.convert import cvep_recording_to_v2
from medusa.pipelines.base import DecodingPipeline
from medusa.pipelines.bci.evoked import (
    TMCCAPipeline, VEPCommandDecoder, SpellerData,
    command_decoding_accuracy_per_cycle)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "cvep")

# --------------------------------------------------------------------------- #
# 1) Load + convert the calibration runs (a single attended command) and learn
#    the template + spatial filter. reference='template' needs the per-trial target,
#    which the c-VEP converter fills from the file.
# --------------------------------------------------------------------------- #
train = [cvep_recording_to_v2(LegacyRecording.load(p))
         for p in sorted(glob.glob(os.path.join(DATA, "cvep_train*.cvep.bson")))]
channels = list(train[0].signals["eeg"].channel_set.labels)
pipe = TMCCAPipeline(
    channels=channels,
    freq_filtering={
        "filterbank": [
            {
                "filt_type": "iir",
                "band_type": "bandpass",
                "cutoff": [1.0, 70.0],
                "order": 7}]},
    reference={
        "mode": "template"}
)
pipe.fit(train)
print(f"Trained template TMCCAPipeline on {len(train)} calibration runs "
      f"(the attended command(s) pooled into {len(pipe.templates)} shift-family template(s)).")

# --------------------------------------------------------------------------- #
# 2) Persist + reload (one portable file; the learned templates travel with it).
# --------------------------------------------------------------------------- #
model_path = os.path.join(HERE, "cvep_tmcca.pkl")
pipe.save(model_path)
pipe = DecodingPipeline.load(model_path)                       # polymorphic load
print(f"Saved + reloaded the pipeline ({type(pipe).__name__}).\n")

# --------------------------------------------------------------------------- #
# 3) Decode the test runs (16 commands = circular shifts of the trained base).
# --------------------------------------------------------------------------- #
labels = open(os.path.join(DATA, "cvep_labels.txt")).read().strip()
decoder = VEPCommandDecoder()
cursor = 0
for path in sorted(glob.glob(os.path.join(DATA, "cvep_test*.cvep.bson"))):
    rec = cvep_recording_to_v2(LegacyRecording.load(path))
    sd = SpellerData.from_recording(rec)
    content_to_uid = {cmd.content: uid for uid, cmd in sd.commands_info.items()}
    n_trials = len(sd.trial_available_cmmds)
    target_text = labels[cursor:cursor + n_trials]
    cursor += n_trials
    sd.spell_target = [content_to_uid[ch] for ch in target_text]
    target = {t: sd.spell_target[t] for t in range(n_trials)}

    # cumulative (n_cycles, n_commands) correlations (Layer 1) -> command selection (Layer 2)
    result = decoder.decode(pipe.predict(rec), sd, rec.events)
    per_cycle = result["selected_commands_per_cycle"]
    n_cycles = int(rec.events.df["cycle_idx"].max()) + 1
    acc_per_cycle = command_decoding_accuracy_per_cycle(per_cycle, target)

    first = "".join(sd.commands_info[per_cycle[t][0]].content for t in range(n_trials))
    final = "".join(sd.commands_info[per_cycle[t][n_cycles - 1]].content
                    for t in range(n_trials))

    print(f"\n{os.path.basename(path)}  ({n_trials} trials x {n_cycles} cycles)")
    print(f"  target           : {target_text!r}")
    print(f"  after  1 cycle   : {first!r}  ({acc_per_cycle[0]:.0%})")
    print(f"  after {n_cycles:2d} cycles  : {final!r}  ({acc_per_cycle[-1]:.0%})")
    print("  accuracy / #cycles: "
          + "  ".join(f"{c}:{a:.0%}" for c, a in enumerate(acc_per_cycle, start=1)))

os.remove(model_path)
