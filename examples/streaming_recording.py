"""Stream a ``Recording`` to disk during acquisition, then round-trip the formats.

Run:  python examples/streaming_recording.py

The live-acquisition story (what medusa-platform's continuous-recording loop
drives) and the persistence story, end to end:

  1. Declare the run's STRUCTURE up front -- empty streams (``Signal.empty``),
     events schema, experiment, sidecars -- before any samples exist.
  2. Stream sample chunks + events with a ``Recorder``: each append grows the
     recording's own ``Signal`` (bounded to the buffer length, for real-time
     processing) AND persists to one ``.h5`` (crash-safe).
  3. Read the file back with ``Recording.load`` -- one read path, any format.
  4. Re-save that recording across formats (``bson``/``json``/``mat``/``h5``) and
     load each back: HDF5 is a full peer of the multiplatform exports.

Key ideas: the buffer *is the data* -- ``recording.signals["eeg"]`` is a live,
bounded ``Signal`` (the last ``buffer`` seconds), so you process it directly
(``.pick_types("EEG")``) and ``n_samples`` is the buffered count. The complete
stream goes to the file; ``Recording.load`` brings it all back. Persistence is
optional: ``Recorder(rec, buffer=2.0)`` with no ``file`` is a pure real-time
buffer. ``times`` is stored on disk only for streams you append it to (EMG here);
a stream appended without it (EEG) re-synthesizes ``arange/fs`` on read.
"""
import tempfile
import warnings
from pathlib import Path

import numpy as np

from medusa.core.data import (BidsInfo, Recording, Recorder, Signal, Events,
                              Channel, ChannelSet)


def section(title):
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


rng = np.random.default_rng(0)
TMP = Path(tempfile.gettempdir())

FS_EEG = 256.0            # device 1 (EEG amplifier)
FS_EMG = 1000.0           # device 2 (EMG), a different rate
BLOCK = 0.25             # seconds delivered per acquisition cycle
N_BLOCKS = 8             # -> 2.0 s total
STIMULI = [(0.5, "go"), (1.0, "nogo"), (1.5, "go")]   # (onset_s, trial_type)


# ---------------------------------------------------------------------------- #
# 1. Declare the run's structure BEFORE acquisition (empty streams)
# ---------------------------------------------------------------------------- #
# `Signal.empty` fixes a stream's fs + channel typing with zero samples; the
# Recorder lays the file out from this. The events schema and experiment are set
# now too, so the writer knows every column/field up front.
section("1. Declare structure (empty streams, schema, sidecars)")

eeg_cs = ChannelSet().add_unipolar_eeg_channels(
    ["Fz", "Cz", "Pz", "POZ", "Oz"], reference="Cz")
emg_cs = ChannelSet().add_channels([Channel("EMG_left", "EMG", "uV"),
                                   Channel("EMG_right", "EMG", "uV")])

rec = Recording(BidsInfo(subject="01", task="gonogo", run=1,
                         participant={"age": 27, "sex": "F"}))
rec.add_signal("eeg", Signal.empty(fs=FS_EEG, channel_set=eeg_cs))
rec.add_signal("emg", Signal.empty(fs=FS_EMG, channel_set=emg_cs))
rec.set_events(Events(optional_columns={"trial_type": str, "value": int},
                      descriptions={"trial_type": "go or nogo"}))
rec.set_experiment({"paradigm": "go/no-go", "monitor_refresh_hz": 60})

# Per-stream sidecars: derivable fields auto-fill; device fields set by hand.
rec.set_sidecar("eeg", Manufacturer="Brain Products", PowerLineFrequency=50)
rec.set_sidecar("emg", Manufacturer="Delsys", PowerLineFrequency=50)
rec.set_sidecar(None, TaskName="go/no-go")          # broadcast to all streams

print("streams         :", list(rec.data))
print("eeg samples now :", rec.signals["eeg"].n_samples, "(empty template)")
print("emg samples now :", rec.signals["emg"].n_samples, "(empty template)")


# ---------------------------------------------------------------------------- #
# 2. Stream: bounded buffer in the Signals (real-time) + continuous file
# ---------------------------------------------------------------------------- #
# buffer=1.0 keeps the recording's own Signals at the last 1 s; passing `file`
# also persists the full stream to the .h5. EEG is appended WITHOUT times
# (regular -> synthesized); EMG WITH per-sample times, which the buffer keeps
# verbatim (true timestamps -> precise sync). Each cycle we process the live
# buffer directly -- `rec.signals["eeg"]` IS the recent window.
section("2. Stream: bounded buffer in the Signals + file (Recorder)")

h5_path = TMP / "sub-01_task-gonogo_run-1_streamed.h5"
n_eeg_blk = int(BLOCK * FS_EEG)     # 64 EEG samples per cycle
n_emg_blk = int(BLOCK * FS_EMG)     # 250 EMG samples per cycle
pending = list(STIMULI)
rt_rms = []

with Recorder(rec, file=str(h5_path), buffer=1.0, overwrite=True) as recorder:
    for blk in range(N_BLOCKS):
        t_block = blk * BLOCK                       # block start, shared timebase

        # Device 1: EEG block (regular sampling, no explicit times). Channel count
        # comes from the channel set, so changing eeg_cs above just works.
        recorder.append_data(
            "eeg", 20.0 * rng.standard_normal((n_eeg_blk, eeg_cs.n_channels)))

        # Device 2: EMG block, with its own per-sample timestamps.
        emg_times = t_block + np.arange(n_emg_blk) / FS_EMG
        recorder.append_data(
            "emg", 50.0 * rng.standard_normal((n_emg_blk, emg_cs.n_channels)),
            times=emg_times)

        # Events: emit any stimulus whose onset falls in this block.
        while pending and pending[0][0] < t_block + BLOCK:
            onset, trial = pending.pop(0)
            recorder.append_events(onset=onset, duration=0.2,
                                   trial_type=trial, value=1 if trial == "go" else 2)

        # Real-time step: process the live buffer -- rec.signals["eeg"] is the
        # last 1 s, a Signal you use directly.
        window = rec.signals["eeg"]
        rt_rms.append(float(np.sqrt(np.mean(window.pick_types("EEG").signal ** 2))))

    print(f"appended {N_BLOCKS} blocks; processed {len(rt_rms)} live windows")
    print("buffered EEG    :", rec.signals["eeg"].n_samples,
          "samples (= last 1 s; bounded) | EEG RMS ~", round(rt_rms[-1], 1))

print("file written    :", h5_path.name)
print("in-memory buffer :", rec.signals["eeg"].n_samples, "EEG samples (last 1 s);",
      "the FULL stream is on disk")


# ---------------------------------------------------------------------------- #
# 3. Read it back (one read path: Recording.load)
# ---------------------------------------------------------------------------- #
section("3. Read back (Recording.load)")

loaded = Recording.load(str(h5_path))
exp_eeg = int(N_BLOCKS * n_eeg_blk)
exp_emg = int(N_BLOCKS * n_emg_blk)
print("eeg shape       :", loaded.signals["eeg"].signal.shape, "->", exp_eeg, "samples")
print("emg shape       :", loaded.signals["emg"].signal.shape, "->", exp_emg, "samples")
# EEG times were NOT stored -> synthesized; EMG times WERE stored -> exact.
print("eeg times       : synthesized arange/fs ->",
      np.allclose(loaded.signals["eeg"].times, np.arange(exp_eeg) / FS_EEG))
print("emg times       : stored, monotonic ->",
      bool(np.all(np.diff(loaded.signals["emg"].times) > 0)))
print("events          :", loaded.events.n_events, "rows")
print(loaded.events.df.to_string(index=False))
print("experiment      :", loaded.experiment)
print("eeg sidecar     :", {k: loaded.sidecars["eeg"][k] for k in
                           ("SamplingFrequency", "EEGChannelCount",
                            "Manufacturer", "TaskName")})


# ---------------------------------------------------------------------------- #
# 4. Same recording, every format -- and VERIFY none of them alter the data
# ---------------------------------------------------------------------------- #
# `save`/`load` pick the format from the extension. bson/json/mat back the
# multiplatform Platform<->Analyzer interchange; h5 adds the binary/streaming
# path. We assert value-level equality, not just shapes -- including the
# variable-length event labels ("go"/"nogo"), which is exactly what tripped up
# .mat before (it right-pads char matrices; now stored as length-preserving cells).
section("4. Save / load across formats (verify lossless)")

ref_eeg = loaded.signals["eeg"].signal
ref_trials = list(loaded.events.df["trial_type"])
for fmt in ("bson", "json", "mat", "h5"):
    path = TMP / f"sub-01_task-gonogo_run-1_recording.{fmt}"
    with warnings.catch_warnings():            # .mat warns on None-stripping
        warnings.simplefilter("ignore")
        loaded.save(str(path))
        back = Recording.load(str(path))
    lossless = (np.allclose(back.signals["eeg"].signal, ref_eeg)
                and np.allclose(back.signals["emg"].times,
                                loaded.signals["emg"].times)
                and list(back.events.df["trial_type"]) == ref_trials
                and back.experiment == loaded.experiment)
    size_kb = path.stat().st_size / 1024
    print(f"  .{fmt:4s}  {size_kb:8.1f} KB   "
          f"{'lossless' if lossless else 'ALTERED DATA'}   "
          f"labels={list(back.events.df['trial_type'])}")
    path.unlink()

h5_path.unlink()
print("\nDone: streamed to HDF5, read back, and round-tripped every format.")
