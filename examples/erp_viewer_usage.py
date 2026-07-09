"""Interactive ERP explorer — ``medusa.widgets.erp_viewer.ERPViewer``.

Run:  python examples/erp_viewer_usage.py

Builds a *fake* P300-style dataset — 60 epochs of a 16-channel 10-20 EEG montage,
each a centro-parietal positive deflection ~300 ms after stimulus onset, buried in
1/f-like noise with trial-to-trial amplitude jitter — and opens the
:class:`~medusa.widgets.erp_viewer.ERPViewer` on it. In the window you can:

  * pick channels (left panel — its options follow the active tab) and switch the
    **Temporal View** between *Split* (one ERP per channel, ``optimal_grid``
    layout, shared axes) and *Mean* (a single ERP averaged over the selected
    channels), the *Mean* spread shown either as an error/CI band **or** as a
    per-channel overlay (thin translucent channels under the thick mean);
  * open the **Spatial View** and drag the time slider (or press play) to sweep the
    scalp topography of the ERP, with a synchronized ERP panel below — either a
    channel overlay or a shaded summary (the *ERP plot* control) + a time cursor;
  * tune the shared analysis (error band, baseline, smoothing, amplitude limits)
    and the topography (interpolation, colormap, color limits);
  * export any panel, the ERP grid, the mean ERP, and the evolving topography GIF
    (with a live preview + a choice of frame size / DPI, and whether to include the
    ERP panel beneath the head).
"""
import numpy as np

from medusa.core.data import ChannelSet
from medusa.widgets.erp_viewer import ERPViewer

# --------------------------------------------------------------------------- #
# 1. A fake P300 dataset on a 10-20 montage
# --------------------------------------------------------------------------- #
FS = 256.0
LABELS = ["Fp1", "Fp2", "F3", "Fz", "F4", "C3", "Cz", "C4",
          "P3", "Pz", "P4", "O1", "Oz", "O2", "T7", "T8"]
channel_set = ChannelSet()
channel_set.add_unipolar_eeg_channels(LABELS, reference=None)

times = np.arange(-0.2, 0.8, 1.0 / FS)          # epoch: -200 .. +800 ms
n_samples = times.shape[0]
n_segments = 60

rng = np.random.default_rng(7)
# Spatial profile: the P300 grows toward the back of the head (centro-parietal).
y = channel_set.get_positions("EEG")[:, 1]      # +y is anterior
posterior = (y.max() - y) / np.ptp(y)           # 1 occipital .. 0 frontal
p300 = np.exp(-0.5 * ((times - 0.30) / 0.06) ** 2)      # bump at 300 ms
amp = 8.0 * posterior                                    # per-channel µV

epochs = np.empty((n_segments, n_samples, len(LABELS)))
for s in range(n_segments):
    noise = rng.standard_normal((n_samples, len(LABELS))) * 6.0
    jitter = rng.normal(1.0, 0.25)              # trial amplitude variability
    epochs[s] = noise + jitter * amp[None, :] * p300[:, None]

print(f"epochs: {epochs.shape} (segments, samples, channels) | fs: {FS} Hz | "
      f"epoch [{times[0]:.2f}, {times[-1]:.2f}] s")


# --------------------------------------------------------------------------- #
# 2a. Interactive: open the viewer (blocks until the window is closed)
# --------------------------------------------------------------------------- #
viewer = ERPViewer(
    epochs, times=times, channel_set=channel_set,
    baseline=(-0.2, 0.0),        # pre-stimulus baseline correction
    error="ci95",                # 95% confidence band
    mode="split",                # start in per-channel split mode
    amplitude_unit="µV", time_unit="s")
print("opening ERPViewer (close the window to exit)...")
viewer.show()                    # blocks until closed
