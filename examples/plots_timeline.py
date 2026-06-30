"""Stacked multichannel traces with ``medusa.plots.plot_timeline`` / ``TimeLinePlot``.

Run:  python examples/plots_timeline.py

Builds figures from a *fake* multichannel EEG signal to show the line view:

  1. ``plot_timeline`` (one-shot): stacked per-channel traces with an amplitude
     scale bar, a pre-scaled trigger channel, a ``time_grid`` of vertical
     gridlines, and an automatic Events overlay (onset lines + shaded spans).
     The returned ``artists`` let you restyle.
  2. ``fs`` is optional (sample-index x-axis), and a 3-D ``(n_segments, n_samples,
     n_channels)`` input is concatenated with a boundary line per join.
  3. ``TimeLinePlot`` (stateful): draw once, then ``set_data`` per window updates
     the lines in place — the streaming / blit path.

Channel separation is a single ``offset`` knob, in signal units (auto-scaled to
``6 x`` the median per-channel std when left ``None``).

Every plot takes a caller-created ``ax`` (no hidden figsize). Figures are written
to ``examples/plots_timeline_figures/`` and also opened in the Qt
``PlotVisualizer``; set ``MEDUSA_EXAMPLE_HEADLESS=1`` to only write the PNGs.
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # build figures off-screen; the Qt viewer (§4) displays them

import matplotlib.pyplot as plt
import medusa_style
import numpy as np
import pandas as pd

import medusa.plots as mp

medusa_style.apply()  # theme matplotlib with the MEDUSA look for the whole script

OUT = Path(__file__).resolve().parent / "plots_timeline_figures"
OUT.mkdir(parents=True, exist_ok=True)

figures: list = []
HEADLESS = bool(os.environ.get("MEDUSA_EXAMPLE_HEADLESS"))


def section(title):
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


# --------------------------------------------------------------------------- #
# 0. A fake multichannel signal + a trigger channel + an events table
# --------------------------------------------------------------------------- #
# Five EEG-like channels with distinct rhythms and a posterior 10 Hz alpha burst
# between 2-4 s, a 0/1 TRIGGER channel with short pulses, and an Events table: two
# zero-duration cues (-> onset lines) and one task block (-> shaded span).
section("0. Fake signal + trigger + events")

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

trigger = np.zeros(n)
for onset in (1.0, 4.5):
    s = int(onset * fs)
    trigger[s:s + int(0.1 * fs)] = 1.0

events = pd.DataFrame({
    "onset":    [1.0,   2.0,    4.5],
    "duration": [0.0,   2.0,    0.0],
    "label":    ["cue", "task", "cue"],
})
print("signal:", signal.shape, "| fs:", fs, "| events:", len(events))


# --------------------------------------------------------------------------- #
# 1. plot_timeline (one-shot): traces + trigger + time grid + events
# --------------------------------------------------------------------------- #
section("1. plot_timeline — traces + trigger + time grid + events")

# The 5 EEG channels + the TRIGGER as a 6th column. There is no special "step"
# mode: pre-scale the 0/1 trigger to a comparable amplitude so it stacks like any
# other channel. A time_grid draws vertical gridlines every second; the Events
# overlay is automatic (cue lines + task span), colored by the 'label' column.
trigger_scaled = trigger * np.median(np.std(signal, axis=0))   # ~1 channel-std
line_data = np.column_stack([signal, trigger_scaled])
line_labels = eeg_labels + ["TRIG"]
fig, ax = plt.subplots(figsize=(11, 5))
_, artists = mp.plot_timeline(
    line_data, ax,
    fs=fs,
    cha_labels=line_labels,
    time_grid=1.0,
    events=events,
    event_hue="label"
)
artists["lines"][3].set_color(medusa_style.categorical_color(1))   # restyle O1
ax.set_title("plot_timeline — traces + trigger + time grid + events")
mp.save_figure(fig, OUT / "timeline.png")
figures.append(fig)
print("saved timeline.png")


# --------------------------------------------------------------------------- #
# 2. fs is optional (sample-index axis) + segmented input (boundary lines)
# --------------------------------------------------------------------------- #
section("2. fs-optional + segmented input")

# With neither fs nor times, the x-axis is the sample index -- fs is NOT required
# (e.g. for channels with no fixed rate).
fig, ax = plt.subplots(figsize=(8, 3))
mp.plot_timeline(signal[:512], ax, cha_labels=eeg_labels)
ax.set_title("fs optional -> x-axis is the sample index")
mp.save_figure(fig, OUT / "timeline_samples.png")
figures.append(fig)

# A 3-D (n_segments, n_samples, n_channels) array (e.g. epochs) is concatenated
# along time, with a boundary line drawn at each segment join.
seg_len = int(fs)                                # 1 s epochs
n_seg = 4
segments = signal[:n_seg * seg_len].reshape(n_seg, seg_len, n_eeg)
fig, ax = plt.subplots(figsize=(11, 4))
_, artists = mp.plot_timeline(segments, ax, fs=fs, cha_labels=eeg_labels)
ax.set_title(f"segmented input -> {len(artists['boundaries'])} boundary lines")
mp.save_figure(fig, OUT / "timeline_segments.png")
figures.append(fig)
print("saved 2 figures (fs-optional + segmented)")


# --------------------------------------------------------------------------- #
# 3. Stateful class: draw once, stream windows with set_data (animation path)
# --------------------------------------------------------------------------- #
# The class is the real-time / animation path. Construct it once (the furniture is
# built on the first set_data), then call set_data() per window -- only the line
# data is replaced in place; it returns the updated artists for a host's blit
# loop. Here we sweep consecutive 1 s windows.
section("3. TimeLinePlot.set_data() over windows (streaming pattern)")

fig, ax = plt.subplots(figsize=(9, 4))
tl = mp.TimeLinePlot(ax, fs=fs, cha_labels=eeg_labels)
win = int(fs)
lines0, updated = None, []
for k in range(int(seconds)):
    seg = signal[k * win:(k + 1) * win]
    updated = tl.set_data(seg)                   # in-place line update
    if lines0 is None:
        lines0 = tl.artists["lines"]
ax.set_title("TimeLinePlot.set_data() over 1 s windows (last window shown)")
mp.save_figure(fig, OUT / "timeline_streaming.png")
figures.append(fig)
print(f"streamed {int(seconds)} windows | lines reused across frames: "
      f"{tl.artists['lines'] is lines0} | updated artists/frame: {len(updated)}")

# For a true live view a host keeps animated=True and BLITS the updated artists
# each frame (no full redraw). The flag is all the engine needs to expose:
fig, ax = plt.subplots(figsize=(9, 4))
live = mp.TimeLinePlot(ax, fs=fs, cha_labels=eeg_labels, animated=True)
updated = live.set_data(signal[:win])
print("blit-ready (line artists animated):",
      all(a.get_animated() for a in updated))

# A matplotlib FuncAnimation would drive the SAME engine (commented so the example
# needs no animation writer):
#
#     from matplotlib.animation import FuncAnimation
#     fig_a, ax_a = plt.subplots()
#     live = mp.TimeLinePlot(ax_a, fs=fs, cha_labels=eeg_labels, animated=True)
#     def update(k):
#         seg = signal[(k % 6) * win:((k % 6) + 1) * win]
#         return live.set_data(seg)               # -> updated artists
#     anim = FuncAnimation(fig_a, update, frames=12, blit=True, interval=300)

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
# 4. Browse the figures in the Qt PlotVisualizer
# --------------------------------------------------------------------------- #
section("4. Browse the figures in the Qt PlotVisualizer")

if HEADLESS:
    print("MEDUSA_EXAMPLE_HEADLESS set -> not opening the viewer.")
else:
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
