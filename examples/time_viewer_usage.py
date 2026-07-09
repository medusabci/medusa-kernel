"""Interactive multichannel viewers — ``medusa.widgets.time_viewer``.

Run:  python examples/time_viewer_usage.py

Opens two windows over a long fake recording:

  1. a stacked-trace **timeline** (:class:`TimeLineViewer`) — raw vs a 10 Hz
     "filtered" copy overlaid on the same baselines, with a BIDS event overlay
     (onset lines + shaded condition spans, coloured by ``trial_type``);
  2. a per-channel **heatmap** (:class:`TimeHeatmapViewer`) — a spectrogram per
     channel (time-frequency), with 3 frequency ticks per band by default (its
     lower edge, midpoint and upper edge) and a *Freq ticks/ch* control to show
     more.

Both browse the record with a draggable scrubber, keyboard navigation and
one-click export; all chrome uses the active ``medusa_style`` theme.
"""
import numpy as np
from scipy.signal import spectrogram

from medusa.core.data.events import Events
from medusa.widgets.time_viewer import TimeHeatmapViewer, TimeLineViewer

rng = np.random.default_rng(0)
fs, n_cha, dur = 250.0, 16, 120.0
n = int(fs * dur)
labels = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4",
          "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6"]

# --------------------------------------------------------------------------- #
# 1. Timeline: raw + filtered + an events overlay
# --------------------------------------------------------------------------- #
t_raw = np.arange(n) / fs
raw = rng.standard_normal((n, n_cha))
filtered = raw + 0.6 * np.sin(2 * np.pi * 10 * (t_raw + 5))[:, None]

# Spans (duration > 0) for conditions, instantaneous markers (duration == 0) for
# events, coloured by the categorical `trial_type` hue.
events = Events(optional_columns={"trial_type": str})
events.append([
    {"onset": 5.0, "duration": 35.0, "trial_type": "Eyes closed"},
    {"onset": 10.0, "duration": 0.0, "trial_type": "Blink"},
    {"onset": 30.0, "duration": 0.0, "trial_type": "Movement"},
    {"onset": 45.0, "duration": 55.0, "trial_type": "Eyes open"},
    {"onset": 60.0, "duration": 0.0, "trial_type": "Blink"},
    {"onset": 95.0, "duration": 0.0, "trial_type": "Movement"},
])

timeline = TimeLineViewer(cha_labels=labels, channels_visible=6,
                          amplitude_unit="µV")
timeline.add_timeline(raw, times=t_raw, label="raw",
                      events=events, event_hue="trial_type")
timeline.add_timeline(filtered, times=t_raw, label="filtered")  # same channels

# --------------------------------------------------------------------------- #
# 2. Heatmap: one spectrogram per channel (0–40 Hz)
# --------------------------------------------------------------------------- #
freqs, times, sxx = spectrogram(raw.T, fs=fs, nperseg=256, noverlap=192)
band = freqs <= 40.0
# scipy returns (n_channels, n_freqs, n_times); the heatmap wants
# (n_freqs, n_times, n_channels).
power = 10 * np.log10(sxx[:, band, :].transpose(1, 2, 0) + 1e-12)
heatmap = TimeHeatmapViewer(cha_labels=labels, channels_visible=6)
heatmap.add_timeheatmap(power, y_values=freqs[band], times=times)

# Both viewers share one QApplication: show both windows, run one loop.
timeline.window.show()
heatmap.window.show()
timeline.app.exec()
