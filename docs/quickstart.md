# Quickstart

A five-minute, end-to-end taste of `medusa-kernel`: synthesise an EEG
recording, band-pass it to the alpha band, measure alpha power per channel and
draw a scalp topography — all on plain NumPy arrays.

```python
import matplotlib.pyplot as plt
import medusa_style

from medusa.signal.generators import EEGSignalGenerator
from medusa.signal.frequency_filtering import IIRFilter
from medusa.signal.transforms import power_spectral_density
from medusa.signal.metrics.spectral import band_power
from medusa.core.data import ChannelSet
from medusa.plots import plot_topography, save_figure

# 1. Synthesise 10 s of 8-channel EEG with a strong ~10 Hz alpha rhythm.
fs = 250.0
labels = ["F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2"]
gen = EEGSignalGenerator(fs=fs, oscillations=[(10.0, 15.0)], seed=0)
signal = gen.get_chunk(10.0, len(labels))            # (2500, 8) = (n_samples, n_channels)

# 2. Zero-phase band-pass to the alpha band (8-13 Hz).
iir = IIRFilter(order=4, cutoff=(8.0, 13.0), btype="bandpass")
alpha = iir.fit_transform(signal, fs=fs)             # (2500, 8)

# 3. Power spectral density (Welch). Spectral functions take the canonical
#    3-D shape (n_segments, n_samples, n_channels), so add a segment axis.
f, psd = power_spectral_density(alpha[None], fs=fs)  # (1, n_frequencies, 8)

# 4. Absolute alpha-band power per channel.
power = band_power(psd, fs, target_band=(8.0, 13.0)) # (1, 8) = (n_segments, n_channels)

# 5. Scalp topography. Build a located channel set from the labels, theme the
#    figure with medusa-style, draw into a caller-supplied Axes, and export.
channels = ChannelSet().add_unipolar_eeg_channels(labels)
medusa_style.apply()
fig, ax = plt.subplots()
ax, artists = plot_topography(power[0], channels, ax, colorbar=True, cbar_label="µV²")
ax.set_title("Alpha-band power")
save_figure(fig, "alpha_topography.png")
```

## What just happened

1. **Synthetic signal.** `EEGSignalGenerator` builds biologically plausible EEG
   (a `1/f` background plus narrow-band rhythms). `get_chunk(duration, n_channels)`
   returns the canonical **continuous** time shape `(n_samples, n_channels)`.

2. **Filtering.** `IIRFilter` wraps a SciPy Butterworth design. It operates on
   continuous 2-D `(n_samples, n_channels)` data; `fit_transform` designs the
   filter at `fs` and applies it zero-phase (`sosfiltfilt`). For the streaming
   path use `filt_method="sosfilt"` and pass `n_channels` to `fit`. To filter
   pre-segmented data `(n_segments, n_samples, n_channels)`, loop over the
   segment axis.

3. **Spectrum & metric.** `power_spectral_density` (Welch) and the
   `signal.metrics.*` functions work on the **segmented** shape
   `(n_segments, n_samples, n_channels)` — that is why we pass `alpha[None]`.
   `band_power` returns one scalar per segment and channel.

4. **Plotting.** Every function in [`medusa.plots`](api/medusa/plots/index.md)
   takes plain arrays plus a caller-supplied `ax` and returns `(ax, artists)`,
   so you stay in control of the figure. The visual identity (palette,
   colormaps, fonts) lives in the shared
   [`medusa_style`](https://github.com/medusabci/medusa-style) package — call
   `medusa_style.apply()` to theme new figures, or `medusa_style.use_theme("dark")`
   to switch theme. `save_figure` writes a tight, optionally transparent export.

!!! tip "Shapes"
    `medusa-kernel` is strict and predictable about array shapes. The single
    rule — `(n_segments, n_samples, n_channels)` for time data, channels always
    last — is enforced by `check_data_dims` at every entry point.

## Where to next

- [Tutorials](tutorials/index.md) — runnable notebooks (recordings & streaming,
  visualization, connectivity, artifact rejection, training a PyTorch classifier).
- [API reference](api/medusa/index.md) — every public module.
