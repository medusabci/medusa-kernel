"""Channels-by-time heatmaps with ``medusa.plots.plot_timeheatmap`` / ``TimeHeatmapPlot``.

Run:  python examples/plots_timeheatmap.py

Builds figures from a *fake* multichannel EEG signal to show the heatmap of
time-distributed data, in its two modes and two rendering methods:

  1. **channel rows** (2-D signal): each channel is one image row, color =
     amplitude.
  2. **per-channel images** (3-D + ``y_values``): each channel is its own 2-D
     image band, stacked in one axes with the ``y_values`` (e.g. frequency) ticks
     drawn in place per band. The common case is a spectrogram.
  3. ``method="imshow"`` (fast, assumes equidistant samples) vs
     ``method="pcolormesh"`` (accurate — honors irregular timestamps, e.g. samples
     lost in transmission).
  4. ``TimeHeatmapPlot`` (stateful): draw once, then ``set_data`` per window — the
     streaming / blit path.

Every plot takes a caller-created ``ax`` (no hidden figsize). Figures are written
to ``examples/plots_timeheatmap_figures/`` and also opened in the Qt
``PlotVisualizer``.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # build figures off-screen; the Qt viewer (§5) displays them

import matplotlib.pyplot as plt
import medusa_style
import numpy as np
import pandas as pd
from scipy.signal import spectrogram

import medusa.plots as mp

medusa_style.apply()  # theme matplotlib with the MEDUSA look for the whole script

OUT = Path(__file__).resolve().parent / "plots_timeheatmap_figures"
OUT.mkdir(parents=True, exist_ok=True)

figures: list = []


def section(title):
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


# --------------------------------------------------------------------------- #
# 0. A fake multichannel signal + its per-channel spectrograms
# --------------------------------------------------------------------------- #
# Five EEG-like channels with distinct rhythms and a posterior 10 Hz alpha burst
# between 2-4 s (so the spectrograms have something to show).
section("0. Fake signal + spectrograms")

rng = np.random.default_rng(7)
fs = 256.0
seconds = 6.0
n = int(fs * seconds)
t = np.arange(n) / fs
eeg_labels = ["Fz", "Cz", "Pz", "O1", "O2"]
n_eeg = len(eeg_labels)

rhythms = [6.0, 8.0, 10.0, 11.0, 12.0]
signal = np.column_stack([np.sin(2 * np.pi * f * t) * (6 + 3 * i)
                          for i, f in enumerate(rhythms)])
signal += rng.standard_normal((n, n_eeg)) * 4.0
burst = (t > 2.0) & (t < 4.0)
signal[burst] += 18.0 * np.sin(2 * np.pi * 10.0 * t[burst])[:, None]

# Per-channel spectrograms -> (n_freqs, n_times, n_channels), in dB for contrast.
specs = []
for i in range(n_eeg):
    f, ts, sxx = spectrogram(signal[:, i], fs=fs, nperseg=128, noverlap=112)
    keep = f <= 40.0
    specs.append(10.0 * np.log10(sxx[keep] + 1e-12))
power = np.stack(specs, axis=-1)
freqs = f[keep]
print("signal:", signal.shape, "| power:", power.shape, "| freqs:", freqs.shape)


# --------------------------------------------------------------------------- #
# 1. Channel rows: a 2-D signal as one image row per channel (+ events)
# --------------------------------------------------------------------------- #
section("1. plot_timeheatmap — channel rows (amplitude) + events")

# The 5 EEG channels as image rows, color = amplitude (a 2-D signal -> one row per
# channel). Diverging map centered on 0. The same automatic Events overlay as the
# line plot is drawn ON TOP of the image: a zero-duration cue -> onset line, the
# 2-4 s alpha burst -> shaded span, colored by the 'label' column.
events = pd.DataFrame({
    "onset":    [1.0,   2.0,    4.5],
    "duration": [0.0,   2.0,    0.0],
    "label":    ["cue", "task", "cue"],
})
fig, ax = plt.subplots(figsize=(11, 3.2))
mp.plot_timeheatmap(
    signal, ax,
    fs=fs,
    cha_labels=eeg_labels,
    clim=(-25, 25),
    cmap="RdBu_r",
    events=events,
    event_hue="label",
    colorbar=True,
    cbar_label="uV"
)
ax.set_title("plot_timeheatmap — channel rows (amplitude) + events")
mp.save_figure(fig, OUT / "heatmap_rows.png")
figures.append(fig)
print("saved heatmap_rows.png")


# --------------------------------------------------------------------------- #
# 2. Per-channel images: a spectrogram per channel (stacked bands)
# --------------------------------------------------------------------------- #
section("2. plot_timeheatmap — per-channel spectrograms (stacked bands)")

# The channels are stacked as bands in ONE axes (a small gap between them, like
# several subplots); each band keeps its own frequency scale, with the `y_values`
# ticks drawn in place, so you can read the frequencies. The contract is only "a
# per-channel 2-D map", so a scalogram or anything else fits too. The same Events
# overlay works here: the cue lines + task span are drawn across all bands.
fig, ax = plt.subplots(figsize=(11, 5))
mp.plot_timeheatmap(
    power, ax,
    times=ts,
    y_values=freqs,
    cha_labels=eeg_labels,
    events=events,
    event_hue="label",
    colorbar=True,
    cbar_label="power (dB)"
)
ax.set_title("plot_timeheatmap — per-channel spectrograms (stacked bands, y_values=freqs)")
mp.save_figure(fig, OUT / "heatmap_images.png")
figures.append(fig)
print("saved heatmap_images.png")


# --------------------------------------------------------------------------- #
# 3. Accurate rendering of a non-equidistant time axis (pcolormesh)
# --------------------------------------------------------------------------- #
section("3. method='imshow' (fast) vs method='pcolormesh' (accurate)")

# When samples are lost in transmission the time axis is no longer equidistant.
# The default method="imshow" maps columns uniformly and would misplace them;
# method="pcolormesh" honors the real `times`, so each column lands at its true
# instant. Here we drop the spectrogram columns between 2.5-3.5 s to mimic a
# dropout -- `ts_gap` has a jump.
keep_t = (ts < 2.5) | (ts > 3.5)
power_gap, ts_gap = power[:, keep_t, :], ts[keep_t]
fig, ax = plt.subplots(figsize=(11, 5))
mp.plot_timeheatmap(
    power_gap, ax,
    times=ts_gap,
    y_values=freqs,
    cha_labels=eeg_labels,
    method="pcolormesh",          # accurate placement of non-uniform columns
    colorbar=True,
    cbar_label="power (dB)"
)
ax.set_title("plot_timeheatmap — method='pcolormesh' (dropout at 2.5-3.5 s "
             "kept at its true time)")
mp.save_figure(fig, OUT / "heatmap_pcolormesh.png")
figures.append(fig)
print("saved heatmap_pcolormesh.png")


# --------------------------------------------------------------------------- #
# 4. Stateful class: draw once, stream windows with set_data (animation path)
# --------------------------------------------------------------------------- #
# The class is the real-time / animation path. Construct it once (the images +
# per-band axes are built on the first set_data), then call set_data() per window.
# With method="imshow" the images are updated in place and the updated artists
# are returned for a host's blit loop. Here we sweep 2 s windows.
section("4. TimeHeatmapPlot.set_data() over windows (streaming pattern)")

fig, ax = plt.subplots(figsize=(11, 5))
hm = mp.TimeHeatmapPlot(ax, y_values=freqs, cha_labels=eeg_labels)
win = 40                                         # spectrogram columns per window
images0, updated = None, []
n_win = power.shape[1] // win
for k in range(n_win):
    chunk = power[:, k * win:(k + 1) * win, :]
    updated = hm.set_data(chunk, times=ts[k * win:(k + 1) * win])  # in-place update
    if images0 is None:
        images0 = hm.artists["images"]
ax.set_title("TimeHeatmapPlot.set_data() over windows (last window shown)")
mp.save_figure(fig, OUT / "heatmap_streaming.png")
figures.append(fig)
print(f"streamed {n_win} windows | images reused across frames: "
      f"{hm.artists['images'] is images0} | updated artists/frame: {len(updated)}")

# Tip: for an irregular live stream use method="pcolormesh" (the mesh is rebuilt
# each set_data, which is slower but places every column at its true timestamp).

# This script called medusa_style.apply() up top (session-wide). For a localized
# effect that restores the previous rcParams afterwards, wrap a block in
# ``with plt.style.context(medusa_style.mpl.rcparams()): ...`` (scoped; no
# global mutation).


section("Figures saved")
pngs = sorted(OUT.glob("*.png"))
print(f"{len(pngs)} PNG(s) in {OUT}:")
for p in pngs:
    print("  -", p.name)


# --------------------------------------------------------------------------- #
# 5. Browse the figures in the Qt PlotVisualizer
# --------------------------------------------------------------------------- #
section("5. Browse the figures in the Qt PlotVisualizer")

try:
    from medusa.widgets.plot_visualizer import PlotVisualizer
except ImportError as exc:
    print(f"Qt viewer could not open ({exc}); skipping.")
else:
    viz = PlotVisualizer()
    for fig in figures:
        viz.add_figure(fig)
    print(f"opening PlotVisualizer with {len(figures)} figures "
          "(Previous/Next to browse, Export to save; close to exit)...")
    viz.show_figs()      # blocks until you close the window
