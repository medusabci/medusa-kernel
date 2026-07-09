"""Build a multimodal ``Recording`` from scratch with ``medusa.core.data``.

Run:  python examples/recording_from_scratch.py

One run, several devices, one shared time origin:

  1. Recording identity (``BidsInfo``).
  2. An EEG stream (device 1) that also carries EOG channels.
  3. An EMG stream (device 2) -- a different device, its own sampling rate.
  4. A phototransistor stream -- a signal BIDS has no datatype for.
  5. Per-stream sidecars: derivable fields auto-filled, device fields set by hand.
  6. Events timeline + free-form experiment metadata.
  7. Save / load round-trip.

Recurring ideas: each device is its own ``Signal`` in ``recording.data`` (keyed by
a label), with its own ``_<datatype>.json`` draft in ``recording.sidecars``;
modality is a per-channel property, so a single EEG stream mixes EEG + EOG, and a
non-BIDS sensor is still just a ``Signal``.
"""
import warnings
import tempfile
from pathlib import Path

import numpy as np

from medusa.core.data import (BidsInfo, Recording, Signal, Events,
                              Channel, Sensor, ChannelSet)


OUT = Path(__file__).resolve().parent / "fake_recordings"
OUT.mkdir(parents=True, exist_ok=True)

def section(title):
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


rng = np.random.default_rng(0)
DURATION = 4.0          # seconds; every stream shares t0 = 0
FS_EEG = 256.0          # device 1 (EEG amplifier)
FS_FAST = 2000.0        # device 2 (EMG) and the phototransistor


# ---------------------------------------------------------------------------- #
# 1. Recording identity
# ---------------------------------------------------------------------------- #
# `BidsInfo` holds the recording-WIDE BIDS metadata. Entities are validated as
# BIDS labels (letters/digits only): the human task name "go/no-go" is not a valid
# label, so the entity is `task-gonogo` and the readable name goes in the sidecar
# `TaskName` later.
section("1. Recording identity (BidsInfo)")

rec = Recording(BidsInfo(
    subject="01", session="01", task="gonogo", run=1,
    participant={"age": 27, "sex": "F", "handedness": "right"}))
print("recording   :", rec)
print("basename    :", rec.bids.basename())


# ---------------------------------------------------------------------------- #
# 2. EEG stream (device 1): EEG + EOG channels in one Signal
# ---------------------------------------------------------------------------- #
# Modality is per channel, so a single stream mixes EEG and EOG. Build the EEG
# block from a standard montage, then add two bipolar EOG channels.
section("2. EEG + EOG stream (device 1)")

eeg_cs = ChannelSet()
eeg_cs.add_unipolar_eeg_channels(
    ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4"],
    reference="Cz", ground="Fpz")
# Bipolar EOG: register the two electrodes of each derivation, then the channel.
eeg_cs.add_sensors([Sensor("VEOG_up"), Sensor("VEOG_down"),
                   Sensor("HEOG_l"), Sensor("HEOG_r")])
eeg_cs.add_channels(Channel("VEOG", "EOG", "uV", sensor="VEOG_up",
                           reference="VEOG_down", reference_method="bipolar"))
eeg_cs.add_channels(Channel("HEOG", "EOG", "uV", sensor="HEOG_l",
                           reference="HEOG_r", reference_method="bipolar"))

n_eeg = int(DURATION * FS_EEG)
eeg_sig = Signal(20.0 * rng.standard_normal((n_eeg, eeg_cs.n_channels)),
                 fs=FS_EEG, channel_set=eeg_cs)
print("channels    :", eeg_cs.labels)
print("modalities  :", eeg_cs.types)
print("datatype    :", eeg_sig.bids_datatype(), "| shape:", eeg_sig.signal.shape)


# ---------------------------------------------------------------------------- #
# 3. EMG stream (device 2): a different device, a different rate
# ---------------------------------------------------------------------------- #
# Surface EMG placed by anatomy (no digitised position -> `location`). Two bipolar
# channels, sampled faster than the EEG. It is a sibling Signal, not a subclass.
section("3. EMG stream (device 2)")

emg_cs = ChannelSet()
emg_cs.add_sensors([
    Sensor("EMG_R_act", location="right extensor digitorum", sensor_type="surface"),
    Sensor("EMG_R_ref", location="right ulnar styloid", sensor_type="surface"),
    Sensor("EMG_L_act", location="left extensor digitorum", sensor_type="surface"),
    Sensor("EMG_L_ref", location="left ulnar styloid", sensor_type="surface")])
emg_cs.add_channels(Channel("EMG_right", "EMG", "uV", sensor="EMG_R_act",
                           reference="EMG_R_ref", reference_method="bipolar"))
emg_cs.add_channels(Channel("EMG_left", "EMG", "uV", sensor="EMG_L_act",
                           reference="EMG_L_ref", reference_method="bipolar"))

n_fast = int(DURATION * FS_FAST)
emg_sig = Signal(50.0 * rng.standard_normal((n_fast, emg_cs.n_channels)),
                 fs=FS_FAST, channel_set=emg_cs)
print("channels    :", emg_cs.labels, "| located by anatomy, not coordinates")
print("datatype    :", emg_sig.bids_datatype(), "| shape:", emg_sig.signal.shape)


# ---------------------------------------------------------------------------- #
# 4. Phototransistor stream: a signal BIDS has no datatype for
# ---------------------------------------------------------------------------- #
# A light sensor on the monitor marks true stimulus onset. "PHOTOTRANSISTOR" is
# not a BIDS channel type, so MEDUSA coerces it to OTHER (warns) -- and the stream
# has no BIDS datatype. It is still a first-class Signal in the recording.
section("4. Phototransistor stream (custom, not in BIDS)")

photo_cs = ChannelSet()
photo_cs.add_sensors(Sensor("photosensor", location="top-left of stimulus monitor",
                           sensor_type="phototransistor"))
with warnings.catch_warnings(record=True) as caught:   # expected coercion warning
    warnings.simplefilter("always")
    photo_cs.add_channels(Channel(
        "photodiode", "PHOTOTRANSISTOR", "V", sensor="photosensor",
        description="Light sensor over the stimulus monitor (true stim onset)"))

# Synthetic trace: a quiet baseline with a light pulse while each stimulus is shown.
photo = 0.05 * rng.standard_normal((n_fast, 1))
for onset in (0.50, 1.75, 3.10):
    start = int(onset * FS_FAST)
    photo[start:start + int(0.20 * FS_FAST)] += 3.0
photo_sig = Signal(photo, fs=FS_FAST, channel_set=photo_cs)
print("coerced type:", photo_cs.types, "(unknown -> OTHER)")
print("datatype    :", photo_sig.bids_datatype(), "(None: BIDS has no datatype)")


# ---------------------------------------------------------------------------- #
# 5. Add the streams; inspect / fill the per-stream sidecars
# ---------------------------------------------------------------------------- #
# `add_signal` registers data[key] AND seeds sidecars[key]: a None-filled template
# for the stream's datatype, with derivable fields auto-filled. The non-BIDS
# stream gets an empty sidecar. Device-specific fields (which differ per device!)
# are then set by hand; recording-wide fields broadcast to every stream.
section("5. Streams + per-stream sidecars")

rec.add_signal("eeg", eeg_sig)      # device 1
rec.add_signal("emg", emg_sig)      # device 2
rec.add_signal("photo", photo_sig)  # non-BIDS auxiliary

print("eeg auto-filled :", {k: rec.sidecars["eeg"][k] for k in
                            ("SamplingFrequency", "EEGChannelCount",
                             "EOGChannelCount", "RecordingDuration")})
print("emg auto-filled :", {k: rec.sidecars["emg"][k] for k in
                            ("SamplingFrequency", "EMGChannelCount")})
print("photo sidecar   :", rec.sidecars["photo"], "(empty -> not a BIDS datatype)")

# Each device has its own Manufacturer -- exactly why sidecars are per stream.
rec.set_sidecar("eeg", Manufacturer="Brain Products",
                ManufacturersModelName="actiCHamp Plus",
                PowerLineFrequency=50, EEGReference="Cz")
rec.set_sidecar("emg", Manufacturer="Delsys", ManufacturersModelName="Trigno",
                PowerLineFrequency=50)
# key=None broadcasts a recording-wide field to every stream at once.
rec.set_sidecar(None, TaskName="go/no-go")
print("eeg Manufacturer:", rec.sidecars["eeg"]["Manufacturer"],
      "| emg Manufacturer:", rec.sidecars["emg"]["Manufacturer"])
print("TaskName (broadcast) on eeg:", rec.sidecars["eeg"]["TaskName"])


# ---------------------------------------------------------------------------- #
# 6. Events timeline + experiment metadata
# ---------------------------------------------------------------------------- #
# `events` is the BIDS events.tsv timeline (onsets in the shared t0 timebase).
# `experiment` is a free-form slot for paradigm info BIDS does not model -- a plain
# dict here (it may also be any SerializableComponent).
section("6. Events + experiment")

events = Events(optional_columns={"trial_type": str, "value": int},
                descriptions={"trial_type": "go or nogo", "value": "stimulus code"})
events.append([{"onset": 0.50, "duration": 0.20, "trial_type": "go", "value": 1},
               {"onset": 1.75, "duration": 0.20, "trial_type": "nogo", "value": 2},
               {"onset": 3.10, "duration": 0.20, "trial_type": "go", "value": 1}])
rec.set_events(events)
rec.set_experiment({"paradigm": "go/no-go", "n_trials": 3,
                    "stimulus": {"go": "green circle", "nogo": "red circle"},
                    "monitor_refresh_hz": 60})
print("events          :", rec.events.n_events, "rows")
print(rec.events.df.to_string(index=False))
print("experiment      :", rec.experiment)


# ---------------------------------------------------------------------------- #
# 7. Queries, BIDS names and a save / load round-trip
# ---------------------------------------------------------------------------- #
section("7. Queries + serialization round-trip")

print("streams         :", list(rec.data))
print("EEG-bearing     :", list(rec.get_signals_by_type("EEG")))
print("EMG-bearing     :", list(rec.get_signals_by_type("EMG")))
print("EEG file base   :", rec.bids_basename("eeg"))
print("EMG file base   :", rec.bids_basename("emg"))

path = OUT / "sub-01_task-gonogo_recording.json"
print(path)
rec.save(str(path))
loaded = Recording.load(str(path))
print("\nreloaded streams         :", list(loaded.data))
print("eeg shape / Manufacturer :", loaded.signals["eeg"].signal.shape,
      "/", loaded.sidecars["eeg"]["Manufacturer"])
print("events / experiment kept :", loaded.events.n_events, "/",
      loaded.experiment["paradigm"])
