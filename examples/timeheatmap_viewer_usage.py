"""Interactive per-channel heatmap viewer — ``medusa.widgets.time_viewer.TimeHeatmapViewer``.

Run:  python examples/timeheatmap_viewer_usage.py

Opens a per-channel **heatmap** over a long fake recording: one spectrogram
(time-frequency) per channel, stacked as bands in a single axes, with 3 frequency
ticks per band by default (its lower edge, midpoint and upper edge) and a
*Freq ticks/ch* control to show more. A BIDS event overlay is drawn over the
image — onset lines (``duration == 0``) and shaded condition spans
(``duration > 0``), coloured/legended by ``trial_type`` exactly like the timeline
viewer (the colour *scale* of the heatmap itself is the colorbar on the right).

Browse the record with the draggable overview scrubber, keyboard navigation
(←/→ time, ↑/↓ channels, +/- colour contrast) and one-click export; all chrome
uses the active ``medusa_style`` theme.
"""
import numpy as np
from scipy.signal import spectrogram

from medusa.core.data.events import Events
from medusa.widgets.time_viewer import TimeHeatmapViewer

rng = np.random.default_rng(0)
fs, n_cha, dur = 250.0, 16, 120.0
n = int(fs * dur)
labels = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4",
          "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6"]
raw = rng.standard_normal((n, n_cha))

# Spans (duration > 0) for conditions, instantaneous markers (duration == 0) for
# events, coloured by the categorical `trial_type` hue — the same BIDS overlay the
# timeline viewer takes, drawn on top of the spectrogram image.
events = Events(optional_columns={"trial_type": str})
events.append([
    {"onset": 5.0, "duration": 35.0, "trial_type": "Eyes closed"},
    {"onset": 10.0, "duration": 0.0, "trial_type": "Blink"},
    {"onset": 30.0, "duration": 0.0, "trial_type": "Movement"},
    {"onset": 45.0, "duration": 55.0, "trial_type": "Eyes open"},
    {"onset": 60.0, "duration": 0.0, "trial_type": "Blink"},
    {"onset": 95.0, "duration": 0.0, "trial_type": "Movement"},
])

# --------------------------------------------------------------------------- #
# One spectrogram per channel (0–40 Hz)
# --------------------------------------------------------------------------- #
freqs, times, sxx = spectrogram(raw.T, fs=fs, nperseg=256, noverlap=192)
band = freqs <= 40.0
# scipy returns (n_channels, n_freqs, n_times); the heatmap wants
# (n_freqs, n_times, n_channels).
power = 10 * np.log10(sxx[:, band, :].transpose(1, 2, 0) + 1e-12)

heatmap = TimeHeatmapViewer(cha_labels=labels, channels_visible=6)
heatmap.add_timeheatmap(power, y_values=freqs[band], times=times,
                        events=events, event_hue="trial_type")

heatmap.show()   # blocks until the window is closed
