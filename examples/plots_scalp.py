"""Publication-ready scalp plots with ``medusa.plots`` (topography + connectivity).

Run:  python examples/plots_scalp.py

Builds figures from a *fake* multichannel EEG signal to show the two ways to use
the module — and when to reach for each:

  1. Free functions  -- ``plot_topography`` / ``plot_connectivity`` / ``plot_scalp``:
     one-shot, return ``(ax, artists)`` (restyle the named artists; ``colorbar=True``
     for the scale; ``save_figure`` for publication export, transparent bg). The
     90% / notebook / paper case.
  2. Stateful classes -- ``TopographicPlot`` / ``ConnectivityPlot``: draw the head
     ONCE and update the data layer in place via ``set_data`` (the real-time /
     animation path; ``set_data`` returns the updated artists for a host's
     blit loop).
  3. The MEDUSA style -- the ``medusa_style`` package (the ecosystem SSOT):
     ``medusa_style.apply()`` to theme matplotlib, ``medusa_style.categorical_color(i)``
     for series colors, ``medusa_style.mpl.sequential_cmap()`` / ``diverging_cmap()``.

Every plot accepts ``ax=None`` (a fresh square axes) or an explicit ``ax`` so it
composes into ``plt.subplots()``. All figures are written as PNGs to
``examples/plots_scalp_figures/`` and also opened in the Qt ``PlotVisualizer``
(Previous/Next to browse, Export to re-save).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # build figures off-screen; the Qt viewer (§5) displays them

import matplotlib.pyplot as plt
import medusa_style
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

import medusa.plots as mp
from medusa.core.data import ChannelSet, Signal

medusa_style.apply()  # theme matplotlib with the MEDUSA look for the whole script

# Figures are written next to this script so you can open them directly.
OUT = Path(__file__).resolve().parent / "plots_scalp_figures"
OUT.mkdir(parents=True, exist_ok=True)

# Collected for the Qt PlotVisualizer at the end.
figures: list = []


def section(title):
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


# --------------------------------------------------------------------------- #
# 0. A fake EEG signal on a standard 10-20 montage
# --------------------------------------------------------------------------- #
# A 19-channel 10-10 set and a fake Signal: a shared 10 Hz alpha rhythm whose
# amplitude grows toward the BACK of the head (posterior alpha), plus per-channel
# noise. From the signal we derive the two inputs the scalp plots consume:
#   - a per-channel scalar (alpha-band power)  -> topography
#   - a channel x channel matrix (correlation) -> connectivity
section("0. Fake EEG signal + 10-10 montage")

LABELS = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC6', 'C3', 'Cz',
          'C4', 'CP5', 'CP6', 'P3', 'Pz', 'P4', 'O1', 'O2']
fs = 250.0
channel_set = ChannelSet()
channel_set.add_unipolar_eeg_channels(LABELS, reference=None)


def make_fake_eeg(channel_set, fs, seconds=4.0, seed=7):
    rng = np.random.default_rng(seed)
    n = int(fs * seconds)
    t = np.arange(n) / fs
    y = channel_set.get_positions('EEG')[:, 1]            # +y is anterior
    posterior = (y.max() - y) / np.ptp(y)                 # 1 occipital .. 0 frontal
    alpha = np.sin(2 * np.pi * 10.0 * t)                  # shared 10 Hz rhythm
    x = (posterior[None, :] * 25.0) * alpha[:, None]      # spatially-weighted alpha
    x += rng.standard_normal((n, channel_set.n_channels)) * 6.0   # noise (uV)
    return x


signal = Signal(make_fake_eeg(channel_set, fs), fs=fs, channel_set=channel_set)
alpha_power = signal.signal.var(axis=0)                  # per-channel band power
adjacency = np.corrcoef(signal.signal.T)                 # channel x channel
print("signal:", signal.signal.shape, "| fs:", signal.fs,
      "| n_channels:", signal.channel_set.n_channels)
print("alpha power per channel:", np.round(alpha_power, 1))


# --------------------------------------------------------------------------- #
# 1. Free functions: one-shot, publication-ready
# --------------------------------------------------------------------------- #
section("1. Free functions (plot_topography / plot_connectivity / plot_scalp)")

# You create the axes (no hidden figsize -- transparency); the plot draws into it
# and returns (ax, artists). colorbar=True adds the scale; save_figure writes PNG.
fig, ax = plt.subplots(figsize=(4, 4))
mp.plot_topography(alpha_power, channel_set, ax, show_labels=True,
                   colorbar=True, cbar_label='alpha power (a.u.)')
ax.set_title('Alpha power topography')
mp.save_figure(fig, OUT / 'topography_interp.png', transparent=True)
figures.append(fig)

# Discrete (per-channel discs) variant, signed values on a diverging colormap.
signed = alpha_power - alpha_power.mean()
fig, ax = plt.subplots(figsize=(4, 4))
mp.plot_topography(signed, channel_set, ax, interpolate=False, cmap='RdBu_r',
                   show_labels=True, colorbar=True,
                   cbar_label='alpha power (centered)')
ax.set_title('Discrete topography (signed)')
mp.save_figure(fig, OUT / 'topography_discrete.png')
figures.append(fig)

# Connectivity: the strongest 15% of correlations (|weight| >= 85th percentile).
fig, ax = plt.subplots(figsize=(4, 4))
mp.plot_connectivity(adjacency, channel_set, ax, threshold=85,
                     show_labels=True, colorbar=True, cbar_label='correlation')
ax.set_title('Connectivity (top 15%)')
mp.save_figure(fig, OUT / 'connectivity.png')
figures.append(fig)

# The bare head substrate (no data) -- e.g. to compose your own overlay on top.
fig, ax = plt.subplots(figsize=(4, 4))
mp.plot_scalp(channel_set, ax, show_labels=True)
ax.set_title('Head substrate')
mp.save_figure(fig, OUT / 'head.png')
figures.append(fig)
print("saved 4 one-shot figures")


# --------------------------------------------------------------------------- #
# 2. Multi-panel: 4 topographies + 4 connectivity, ONE colorbar per row
# --------------------------------------------------------------------------- #
# Four consecutive 1 s windows give four related maps per row. Because every plot
# takes an explicit `ax`, they drop straight into a 2x4 plt.subplots() grid. A
# *shared* row colorbar is only honest if every panel in the row uses the SAME
# color limits, so we compute one `clim` per row and pass it to each plot; the
# colorbar is then built once from a ScalarMappable on that shared scale and told
# to steal space from all four of the row's axes (`fig.colorbar(ax=row_axes)`).
section("2. 2x4 panel (topographies + connectivity), one colorbar per row")

win = int(fs)
windows = [signal.signal[k * win:(k + 1) * win] for k in range(4)]
powers = [w.var(axis=0) for w in windows]                # 4 topography inputs
adjs = [np.corrcoef(w.T) for w in windows]               # 4 connectivity inputs

# Shared color limits per row (computed over all four windows).
topo_clim = (0.0, max(p.max() for p in powers))
tri = np.triu_indices(channel_set.n_channels, 1)
corr = np.concatenate([a[tri] for a in adjs])
conn_clim = (float(corr.min()), float(corr.max()))
seq, div = medusa_style.mpl.sequential_cmap(), medusa_style.mpl.diverging_cmap()

with plt.style.context(medusa_style.mpl.rcparams()):
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.6), layout='constrained')
    for k in range(4):
        mp.plot_topography(powers[k], channel_set, ax=axes[0, k],
                           cmap=seq, clim=topo_clim)
        axes[0, k].set_title(f"{k}–{k + 1} s")
        mp.plot_connectivity(adjs[k], channel_set, ax=axes[1, k],
                             cmap=div, clim=conn_clim, threshold=85)
    axes[0, 0].set_ylabel("Band power", labelpad=10)
    axes[1, 0].set_ylabel("Connectivity", labelpad=10)

    # One shared colorbar per row, spanning all four of the row's axes.
    for row_axes, clim, cmap, label in ((axes[0], topo_clim, seq, "power"),
                                        (axes[1], conn_clim, div, "r")):
        sm = ScalarMappable(Normalize(*clim), cmap=cmap)
        sm.set_array([])
        fig.colorbar(sm, ax=row_axes.tolist(), label=label, fraction=0.04)
    mp.save_figure(fig, OUT / 'panel.png')
figures.append(fig)
print("saved 2x4 panel (4 topographies + 4 connectivity, one colorbar per row)")


# --------------------------------------------------------------------------- #
# 3. Stateful class: draw the head once, stream frames with set_data
# --------------------------------------------------------------------------- #
# The class is the real-time / animation path. Construct it once (the head is
# drawn here), then call set_data() per frame -- only the value layer is replaced
# in place. set_data() returns the updated artists for a host's blit loop.
# Here we sweep alpha power over consecutive 1 s windows, saving a frame each.
section("3. TopographicPlot.set_data() over time windows (animation pattern)")

win = int(fs)                                    # 1 s windows
fig, ax = plt.subplots(figsize=(4, 4))
topo = mp.TopographicPlot(channel_set, ax=ax, clim=(0, alpha_power.max()))
head_artists = topo.artists                      # the head is built ONCE, here
n_frames = signal.signal.shape[0] // win
updated: list = []
for k in range(n_frames):
    seg = signal.signal[k * win:(k + 1) * win]
    updated = topo.set_data(seg.var(axis=0))     # in-place value-layer update
    mp.save_figure(topo.ax.figure, OUT / f'frame_{k}.png')
figures.append(fig)              # the final window's topography
print(f"saved {n_frames} frames | head reused across frames: "
      f"{topo.artists is head_artists} | updated artists/frame: {len(updated)}")

# For a true live view a host keeps animated=True and BLITS the updated artists
# each frame (no full redraw). The flag is all the engine needs to expose:
_, live_ax = plt.subplots(figsize=(4, 4))
live = mp.TopographicPlot(channel_set, live_ax, animated=True)
updated = live.set_data(alpha_power)
print("blit-ready (value artists animated):",
      all(a.get_animated() for a in updated))

# A matplotlib FuncAnimation would drive the SAME engine (kept commented so the
# example needs no animation writer):
#
#     from matplotlib.animation import FuncAnimation
#     fig_a, ax_a = plt.subplots()
#     t = mp.TopographicPlot(channel_set, ax=ax_a,
#                            clim=(0, alpha_power.max()), animated=True)
#     def update(k):
#         seg = signal.signal[(k % n_frames) * win:(k % n_frames + 1) * win]
#         return t.set_data(seg.var(axis=0))         # -> updated artists
#     anim = FuncAnimation(fig_a, update, frames=8, blit=True, interval=300)


# --------------------------------------------------------------------------- #
# 4. The MEDUSA style + palette helpers
# --------------------------------------------------------------------------- #
# Identity helpers for your own figures: a scoped style context, the brand
# categorical cycle (no jet/rainbow), and semantic colormaps by role.
section("4. Style helpers (medusa_style.apply / categorical_color / colormaps)")

print("categorical (first 4):",
      [medusa_style.categorical_color(i) for i in range(4)])
print("colormaps -> sequential:", medusa_style.mpl.sequential_cmap().name,
      "| diverging:", medusa_style.mpl.diverging_cmap().name)

with plt.style.context(medusa_style.mpl.rcparams()):
    fig, ax = plt.subplots(figsize=(6, 3))
    t = np.arange(signal.signal.shape[0]) / fs
    for i, ch in enumerate(['O1', 'O2', 'Fp1']):       # auto categorical colors
        ax.plot(t, signal.signal[:, channel_set.index(ch)[0]] + i * 80, label=ch)
    ax.set(xlabel='time (s)', ylabel='amplitude (uV, offset)',
           title='Fake EEG traces (brand categorical cycle)')
    ax.legend(ncol=3)
    mp.save_figure(fig, OUT / 'traces_style.png', transparent=True)
figures.append(fig)


section("Figures saved")
pngs = sorted(OUT.glob('*.png'))
print(f"{len(pngs)} PNG(s) in {OUT}:")
for p in pngs:
    print("  -", p.name)


# --------------------------------------------------------------------------- #
# 5. Browse the figures in the Qt PlotVisualizer
# --------------------------------------------------------------------------- #
# PlotVisualizer (medusa.widgets, PySide6 -- a core dependency) is a small Qt
# window that pages through several figures (Previous/Next) and can re-export
# them. It owns the QApplication and show_figs() BLOCKS until the window closes,
# so we built the figures on the Agg backend above and open the viewer here.
section("5. Browse the figures in the Qt PlotVisualizer")

try:
    from medusa.widgets.plot_visualizer import PlotVisualizer
except ImportError as exc:
    print(f"Qt viewer could not open ({exc}); skipping.")
else:
    viz = PlotVisualizer()
    for f in figures:
        viz.add_figure(f)
    print(f"opening PlotVisualizer with {len(figures)} figures "
          "(Previous/Next to browse, Export to save; close to exit)...")
    viz.show_figs()      # blocks until you close the window
