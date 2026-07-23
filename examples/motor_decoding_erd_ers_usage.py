"""Offline motor-imagery analysis on real data: per-class ERD/ERS, topographies, optimal band.

Run:  python examples/motor_decoding_erd_ers_usage.py

Explores the bundled real motor-imagery recordings in ``examples/data/mi/`` (a 16-channel,
256 Hz session; left- vs right-hand imagery) with the pure-NumPy
:mod:`medusa.pipelines.bci.motor_decoding.analysis` module. Motor imagery modulates the
sensorimotor mu/beta rhythm, and the two hands should differ **spatially** (lateralized) and in
**which frequency band** carries the difference -- both of which vary by subject. So we:

1. cut trials and split them by class (left / right hand);
2. **per-class ERD/ERS spectrograms** over C3/CZ/C4 -- a 3x2 grid (rows = channels, columns =
   classes) to read the time-frequency difference between the hands directly
   (:func:`~medusa.pipelines.bci.motor_decoding.analysis.spectrogram` +
   :func:`~medusa.pipelines.bci.motor_decoding.analysis.erd_ers`);
3. **band topographies** -- for each band of interest (mu, SMR, low/high beta) a scalp map of the
   left-vs-right discriminability (signed r-squared,
   :func:`~medusa.pipelines.bci.motor_decoding.analysis.class_discriminability` on
   :func:`~medusa.pipelines.bci.motor_decoding.analysis.trial_band_power`);
4. **the subject's optimal band** -- the frequency where the classes separate best, found from the
   r-squared spectrum (:func:`~medusa.pipelines.bci.motor_decoding.analysis.discriminability_spectrum`
   + :func:`~medusa.pipelines.bci.motor_decoding.analysis.optimal_band`), instead of a fixed preset.

Two figures are saved: the ERD/ERS spectrogram grid, and the band topographies + r-squared spectrum.
"""
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")            # headless-safe; drop this line to show the figures live
import matplotlib.pyplot as plt
import medusa_style

from medusa.core.legacy.recording import Recording as LegacyRecording
from medusa.core.legacy.convert import mi_recording_to_v2
from medusa.plots import plot_topography
from medusa.pipelines.bci.motor_decoding import analysis as mia

# medusa_style is the ecosystem styling SSOT: theme every figure and take colors/colormaps
# from it rather than hardcoding matplotlib defaults (as medusa.plots does internally).
medusa_style.apply()
DIVERGING = medusa_style.mpl.diverging_cmap()    # ERD/ERS + signed-r2 maps (centred at 0)
SEQUENTIAL = medusa_style.mpl.sequential_cmap()  # absolute-power maps
FG = medusa_style.current_theme().plot_fg        # chrome: onset lines, band edges

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "mi")

MAIN_CHANNELS = ["C3", "CZ", "C4"]          # sensorimotor channels for the spectrogram grid
WINDOW = (-4.0, 5.0)                        # epoch window (s) relative to each cue
BASELINE = (-3.0, -1.0)                     # pre-cue reference window (s)
POST = (0.5, 3.5)                           # imagery period analysed for discriminability
PSD_FREQS = (1.0, 30.0)                     # frequency range shown in the
# per-channel PSD plot (Hz)
#: Class code -> curve colour for the per-channel PSD plot, from the medusa_style palette.
CLASS_COLORS = {0: medusa_style.categorical_color(0),
                1: medusa_style.categorical_color(1)}
#: The candidate bands of interest for hand motor imagery (Hz).
BANDS = {
    "mu (8-13)": (8.0, 13.0),
    "smr (12-15)": (12.0, 15.0),
    "low beta (13-20)": (13.0, 20.0),
    "high beta (20-30)": (20.0, 30.0)
}

# --------------------------------------------------------------------------- #
# 1) Load every run (all channels, for the topographies) and pool the trials.
# --------------------------------------------------------------------------- #
recordings = [mi_recording_to_v2(LegacyRecording.load(p))
              for p in sorted(glob.glob(os.path.join(DATA, "*.mi.bson")))]
channel_set = recordings[0].signals["eeg"].channel_set
channels = list(channel_set.labels)
label_names = recordings[0].experiment["label_names"]        # {0: 'Left_hand', 1: 'Right_hand'}

epochs, labels, fs = [], [], None
for rec in recordings:
    ep, y, fs = mia.motor_trials(rec, channels=channels, window=WINDOW)
    epochs.append(ep)
    labels.append(y)
epochs, labels = np.concatenate(epochs), np.concatenate(labels)
classes = sorted(np.unique(labels[~np.isnan(labels)]))
main_idx = [channels.index(c) for c in MAIN_CHANNELS]
print(f"Loaded {len(recordings)} runs -> {len(epochs)} trials, {len(channels)} channels @ {fs:g} Hz.")
print(f"Classes: {[label_names[int(c)] for c in classes]} "
      f"({[int((labels == c).sum()) for c in classes]} trials).")

# --------------------------------------------------------------------------- #
# 2) Per-class ERD/ERS spectrograms (percent change vs the pre-cue baseline).
# --------------------------------------------------------------------------- #
power, times, freqs = mia.spectrogram(epochs, fs, epoch_start=WINDOW[0],
                                      time_window=1.0, overlap=0.9, freq_range=(5.0, 30.0))
erds = {c: mia.erd_ers(power[labels == c], times, baseline=BASELINE, ref_mode="classic")
        for c in classes}                                    # {class: (n_freqs, n_times, n_cha)}

lim = float(np.percentile([np.abs(erds[c][:, :, main_idx]) for c in classes], 98))
fig1, axes = plt.subplots(len(MAIN_CHANNELS), len(classes), figsize=(9, 8),
                          sharex=True, sharey=True)
for r, (ch, ci) in enumerate(zip(MAIN_CHANNELS, main_idx)):
    for col, c in enumerate(classes):
        ax = axes[r, col]
        mesh = ax.pcolormesh(times, freqs, erds[c][:, :, ci], cmap=DIVERGING,
                             vmin=-lim, vmax=lim, shading="auto")
        ax.axvline(0.0, color=FG, lw=0.8, ls="--")           # cue onset
        for edge in (8, 13, 15, 20, 30):                     # band boundaries
            ax.axhline(edge, color=FG, lw=0.4, alpha=0.2)
        if r == 0:
            ax.set_title(label_names[int(c)])
        if col == 0:
            ax.set_ylabel(f"{ch}\nfrequency (Hz)")
        if r == len(MAIN_CHANNELS) - 1:
            ax.set_xlabel("time from cue (s)")
fig1.colorbar(mesh, ax=axes, fraction=0.04, pad=0.02, label="ERD/ERS (%)")
fig1.suptitle("Motor imagery (real data): ERD/ERS per class and channel")
out1 = os.path.join(HERE, "motor_decoding_erd_ers_spectrograms.png")
fig1.savefig(out1, dpi=130)

# --------------------------------------------------------------------------- #
# 3) + 4) Discriminability: rank the candidate bands, find the subject's optimal band.
# --------------------------------------------------------------------------- #
band_r2 = {}                                                 # {band: signed r^2 per channel}
for name, band in BANDS.items():
    feat = mia.trial_band_power(epochs, fs, band, epoch_start=WINDOW[0], window=POST)
    band_r2[name] = mia.class_discriminability(feat, labels)

r2_spec, r2_freqs = mia.discriminability_spectrum(
    epochs, labels, fs, epoch_start=WINDOW[0], window=POST, freq_range=(5.0, 30.0))
opt_lo, opt_hi = mia.optimal_band(r2_spec, r2_freqs, channels=main_idx)
opt_feat = mia.trial_band_power(epochs, fs, (opt_lo, opt_hi), epoch_start=WINDOW[0], window=POST)
opt_r2 = mia.class_discriminability(opt_feat, labels)

print("\nLeft-vs-right discriminability by band (mean |r2| over C3/CZ/C4)")
print("=" * 56)
ranked = sorted(BANDS, key=lambda n: np.abs(band_r2[n][main_idx]).mean(), reverse=True)
for name in ranked:
    print(f"  {name:<18} mean |r2| {np.abs(band_r2[name][main_idx]).mean():.3f}")
print("-" * 56)
print(f"  optimal band {opt_lo:.1f}-{opt_hi:.1f} Hz   mean |r2| "
      f"{np.abs(opt_r2[main_idx]).mean():.3f}  (data-driven, subject-specific)")
print("=" * 56)

# --------------------------------------------------------------------------- #
# Figure 2: band topographies (signed r^2 per channel) + the r^2 spectrum.
# --------------------------------------------------------------------------- #
topos = list(BANDS.items()) + [(f"optimal\n{opt_lo:.0f}-{opt_hi:.0f} Hz", (opt_lo, opt_hi))]
topo_r2 = list(band_r2.values()) + [opt_r2]
tlim = float(np.abs(np.array(topo_r2)).max())

fig2 = plt.figure(figsize=(13, 6))
gs = fig2.add_gridspec(2, len(topos), height_ratios=[1.1, 1.0], hspace=0.35)
for i, ((name, _band), r2) in enumerate(zip(topos, topo_r2)):
    ax = fig2.add_subplot(gs[0, i])
    plot_topography(r2, channel_set, ax, cmap=DIVERGING, clim=(-tlim, tlim),
                    colorbar=(i == len(topos) - 1), cbar_label="signed r2")
    ax.set_title(name, fontsize=9)

axp = fig2.add_subplot(gs[1, :])
profile = np.abs(r2_spec[:, main_idx]).mean(axis=1)          # discriminability over sensorimotor
axp.plot(r2_freqs, profile, color=FG, lw=1.5)
for name, (lo, hi) in BANDS.items():
    axp.axvspan(lo, hi, alpha=0.08, color=medusa_style.categorical_color(0))
axp.axvspan(opt_lo, opt_hi, alpha=0.25, color=medusa_style.categorical_color(1),
            label=f"optimal {opt_lo:.0f}-{opt_hi:.0f} Hz")
axp.set_xlabel("frequency (Hz)")
axp.set_ylabel("mean |r2| over C3/CZ/C4")
axp.set_title("Left-vs-right discriminability spectrum (shaded = preset bands, highlighted = optimal)")
axp.legend(fontsize=8, loc="upper right")
axp.margins(x=0.01)
fig2.suptitle("Motor imagery (real data): spatial + spectral class differences")
out2 = os.path.join(HERE, "motor_decoding_erd_ers_topographies.png")
fig2.savefig(out2, dpi=130, bbox_inches="tight")

# --------------------------------------------------------------------------- #
# Figure 3: per-channel PSD by class (one panel per sensorimotor channel).
# --------------------------------------------------------------------------- #
# Welch PSD of each trial over the imagery window, averaged per class (mean +/- SEM band).
# A contralateral mu ERD shows up as a lower curve for one hand than the other.
psd, psd_freqs = mia.trial_psd(epochs, fs, epoch_start=WINDOW[0], window=POST,
                               freq_range=PSD_FREQS)            # (n_trials, n_freqs, n_channels)
fig3, axes3 = plt.subplots(1, len(MAIN_CHANNELS), figsize=(12, 4), sharex=True, sharey=True)
axes3 = np.atleast_1d(axes3)
for ax, ch, ci in zip(axes3, MAIN_CHANNELS, main_idx):
    for c in classes:
        m = psd[labels == c][:, :, ci]                          # (n_class_trials, n_freqs)
        mean, sem = m.mean(axis=0), m.std(axis=0) / np.sqrt(len(m))
        ax.plot(psd_freqs, mean, color=CLASS_COLORS[int(c)], lw=1.5, label=label_names[int(c)])
        ax.fill_between(psd_freqs, np.clip(mean - sem, 1e-12, None), mean + sem,
                        color=CLASS_COLORS[int(c)], alpha=0.2)
    ax.axvspan(opt_lo, opt_hi, alpha=0.12, color=FG,
               label=f"optimal {opt_lo:.0f}-{opt_hi:.0f} Hz")
    ax.set_yscale("log")
    ax.set_title(ch)
    ax.set_xlabel("frequency (Hz)")
    ax.margins(x=0.01)
axes3[0].set_ylabel("PSD (log scale)")
axes3[0].legend(fontsize=8, loc="upper right")
fig3.suptitle("Motor imagery (real data): per-channel PSD by class (imagery window)")
out3 = os.path.join(HERE, "motor_decoding_erd_ers_psd_by_class.png")
fig3.savefig(out3, dpi=130, bbox_inches="tight")

# --------------------------------------------------------------------------- #
# Figure 4: scalp topography of optimal-band power for each class (shared scale).
# --------------------------------------------------------------------------- #
opt_power = mia.trial_band_power(epochs, fs, (opt_lo, opt_hi),
                                 epoch_start=WINDOW[0], window=POST)   # (n_trials, n_channels)
power_by_class = {c: opt_power[labels == c].mean(axis=0) for c in classes}   # {class: (n_cha,)}
pmin = min(float(v.min()) for v in power_by_class.values())
pmax = max(float(v.max()) for v in power_by_class.values())

fig4, axes4 = plt.subplots(1, len(classes), figsize=(9, 4))
axes4 = np.atleast_1d(axes4)
for i, (ax, c) in enumerate(zip(axes4, classes)):
    plot_topography(power_by_class[c], channel_set, ax, cmap=SEQUENTIAL, clim=(pmin, pmax),
                    colorbar=(i == len(classes) - 1),
                    cbar_label=f"mean power {opt_lo:.0f}-{opt_hi:.0f} Hz")
    ax.set_title(label_names[int(c)])
fig4.suptitle(f"Motor imagery (real data): optimal-band ({opt_lo:.0f}-{opt_hi:.0f} Hz) power per class")
out4 = os.path.join(HERE, "motor_decoding_erd_ers_optband_power.png")
fig4.savefig(out4, dpi=130, bbox_inches="tight")

print(f"\nSaved -> {out1}\n         {out2}\n         {out3}\n         {out4}")
print("Note: real single-subject EEG is noisy; the CSP + LDA pipeline in "
      "motor_decoding_mi_usage.py\ndecodes these runs far better than any single fixed band, which "
      "is why spatial filtering is used.")
