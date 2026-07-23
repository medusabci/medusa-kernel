"""Interactive ERP explorer on real P300 data -- ``medusa.widgets.erp_viewer.ERPViewer``.

Run:  python examples/erp_viewer_usage.py

Loads the bundled legacy **RCP / P300 speller** recordings (the same data decoded in
``vep_spellers_rcp_usage.py``), builds the **target (attended) flash ERP** -- the classic
centro-parietal P300 -- and opens the :class:`~medusa.widgets.erp_viewer.ERPViewer` on it.

Where the epochs come from
--------------------------
An RCP speller flashes rows and columns of a command matrix. A flash is *attended* when it
lights the command the user is spelling. Each command carries a per-frame codebook (1 while
one of its groups flashes, 0 otherwise), so the **rising edges** of the trial target's code
are exactly the attended-flash onsets. We epoch the EEG around those onsets (-200..800 ms),
after a common-average reference and a 0.5--16 Hz band-pass, pooling every calibration and
test run into one ``(n_epochs, n_samples, n_channels)`` stack. Averaging attended flashes
yields the P300 (a positive deflection ~300--450 ms, largest over the parietal channels of
this posterior montage: Pz, POz, CPz, PO7/PO8, Oz). Only flash *onsets* are used, not every
"on" frame -- a flash is sustained for ~11 frames, so onset-locking keeps the ERP crisp.

In the viewer
-------------
  * pick channels (left panel -- its options follow the active tab) and switch the
    **Temporal View** between *Split* (one ERP per channel, ``optimal_grid`` layout, shared
    axes) and *Mean* (a single ERP averaged over the selected channels), the *Mean* spread
    shown either as an error/CI band **or** as a per-channel overlay (thin translucent
    channels under the thick mean);
  * open the **Spatial View** and drag the time slider (or press play) to sweep the scalp
    topography of the P300, with a synchronized ERP panel below -- either a channel overlay
    or a shaded summary (the *ERP plot* control) + a time cursor;
  * tune the shared analysis (error band, baseline, smoothing, amplitude limits) and the
    topography (interpolation, colormap, color limits);
  * export any panel, the ERP grid, the mean ERP, and the evolving topography GIF (with a
    live preview + a choice of frame size / DPI, and whether to include the ERP panel).
"""
import glob
import os

import numpy as np

from medusa.core.legacy.recording import Recording as LegacyRecording
from medusa.core.legacy.convert import rcp_recording_to_v2
from medusa.pipelines.bci.vep_spellers import SpellerData
from medusa.signal.spatial_filtering import car
from medusa.signal.frequency_filtering import IIRFilter
from medusa.signal.segmentation import (
    segment_signal_around_events, check_event_segments_feasibility)
from medusa.widgets.erp_viewer import ERPViewer

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "rcp_speller")

W_MS = (-200.0, 800.0)          # epoch window (ms) around each flash onset
BASELINE_S = (-0.2, 0.0)        # pre-stimulus baseline (the viewer applies it)
BAND = (0.5, 16.0)              # P300 band-pass (Hz)


def target_flash_onsets(recording, sd):
    """Onset time (s) of every flash that lit the trial's target command.

    The target command's code is 1 while a group containing it flashes, so its
    ``0 -> 1`` transitions (rising edges) are the attended-flash onsets. Frame ``f`` of a
    cycle occurs at ``cycle_onset + f / fps_resolution``.
    """
    codes = sd.codes                                   # (n_commands, n_codes, n_frames)
    row = {uid: i for i, uid in enumerate(sd.command_uids)}
    df = recording.events.df
    df = df[df["cycle_idx"].notna()]                   # one row per stimulation cycle
    onsets = []
    for onset, trial, code_idx in zip(df["onset"].to_numpy(float),
                                      df["trial_idx"].to_numpy(int),
                                      df["code_idx"].to_numpy(int)):
        pattern = codes[row[str(sd.spell_target[trial])], code_idx].astype(int)
        edges = np.flatnonzero(np.diff(pattern, prepend=0) == 1)
        onsets.append(onset + edges / sd.fps_resolution)
    return np.concatenate(onsets) if onsets else np.empty(0)


# --------------------------------------------------------------------------- #
# 1. Load + convert every bundled RCP run and epoch the attended-flash P300
# --------------------------------------------------------------------------- #
paths = sorted(glob.glob(os.path.join(DATA, "*.rcp.bson")))
recordings = [rcp_recording_to_v2(LegacyRecording.load(p)) for p in paths]

channel_set = recordings[0].signals["eeg"].channel_set
fs = recordings[0].signals["eeg"].fs

epochs = []
for rec in recordings:
    sig = rec.signals["eeg"]
    sd = SpellerData.from_recording(rec)
    x = car(sig.signal)                                # common-average reference
    x = IIRFilter(5, BAND, "bandpass").fit_transform(x, sig.fs)   # 0.5-16 Hz band-pass
    onsets = target_flash_onsets(rec, sd)
    valid = check_event_segments_feasibility(sig.times, onsets, sig.fs, W_MS).valid
    epochs.append(segment_signal_around_events(
        sig.times, x, onsets[valid], sig.fs, W_MS))    # (n_flashes, n_samples, n_cha)
epochs = np.concatenate(epochs)

# Time base matching the segmentation's sample grid (-200..800 ms -> seconds).
w_s = np.round(np.asarray(W_MS) * fs / 1000).astype(int)
times = np.arange(w_s[0], w_s[1]) / fs

print(f"epochs: {epochs.shape} (attended flashes, samples, channels) | fs: {fs:g} Hz | "
      f"epoch [{times[0]:.2f}, {times[-1]:.2f}] s | {len(recordings)} runs")


# --------------------------------------------------------------------------- #
# 2. Interactive: open the viewer (blocks until the window is closed)
# --------------------------------------------------------------------------- #
viewer = ERPViewer(
    epochs, times=times, channel_set=channel_set,
    baseline=BASELINE_S,             # pre-stimulus baseline correction
    error="ci95",                    # 95% confidence band
    mode="split",                    # start in per-channel split mode
    amplitude_unit="µV", time_unit="s")
print("opening ERPViewer (close the window to exit)...")
viewer.show()                        # blocks until closed
