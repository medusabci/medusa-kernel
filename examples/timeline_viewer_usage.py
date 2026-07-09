"""Interactive stacked-trace viewer — ``medusa.widgets.time_viewer.TimeLineViewer``.

Run:  python examples/timeline_viewer_usage.py

Opens a stacked-trace **timeline** over a long fake recording: the raw signal and
a 10 Hz "filtered" copy overlaid on the same channel baselines (distinct colours,
named in the legend), with a BIDS event overlay — onset lines (``duration == 0``)
and shaded condition spans (``duration > 0``), coloured by ``trial_type``.

Browse the record with the draggable overview scrubber, keyboard navigation
(←/→ time, ↑/↓ channels, +/- amplitude gain) and one-click export; all chrome
uses the active ``medusa_style`` theme.
"""
import numpy as np

from medusa.core.data.events import Events
from medusa.widgets.time_viewer import TimeLineViewer

rng = np.random.default_rng(0)
fs, n_cha, dur = 250.0, 16, 120.0
n = int(fs * dur)
labels = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4",
          "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6"]

# --------------------------------------------------------------------------- #
# Raw + a 10 Hz "filtered" copy + an events overlay
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

timeline.show()   # blocks until the window is closed
