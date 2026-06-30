# MEDUSA© Kernel

**medusa-kernel** is a Python library for biomedical signal processing and
machine learning, with a focus on electrophysiology (EEG, MEG, ECG, EMG, EOG,
NIRS). It provides a complete, transparent toolbox of filters, transforms,
metrics, connectivity measures, graph-theoretic analyses, deep-learning models,
visualizations, and a BIDS-aligned data model for recordings.

- **Website:** https://www.medusabci.com/
- **Documentation:** https://docs.medusabci.com/kernel/

---

## Installation

```bash
pip install medusa-kernel
```

medusa-kernel requires **Python 3.13+** and runs on Linux, macOS, and Windows.
Everything is pure Python (no compilers or native binaries), so a single wheel
works everywhere. The signal-processing, data, plotting, and GUI layers all work
out of the box.

Optional extras:

```bash
pip install "medusa-kernel[dev]"    # test + lint toolchain
pip install "medusa-kernel[docs]"   # documentation toolchain
pip install "medusa-kernel[all]"    # dev + docs
```

**Deep learning is opt-in.** PyTorch is *not* installed automatically, because
its wheels are tied to your CUDA / ROCm / MPS / CPU stack. Install the build that
matches your machine from https://pytorch.org/get-started/. For the deep-learning
estimators you also need PyTorch Lightning:

```bash
pip install torch lightning   # pick the torch build for your platform
```

---

## Quick start

A complete chain — synthesize a multichannel EEG signal, filter it, estimate its
spectrum, compute alpha-band power per channel, and draw a scalp topography:

```python
import matplotlib.pyplot as plt

from medusa.signal.generators import EEGSignalGenerator
from medusa.signal import IIRFilter, power_spectral_density
from medusa.signal.metrics.spectral import band_power
from medusa.core.data import ChannelSet
from medusa.plots import plot_topography

labels = ["Fp1", "Fp2", "C3", "C4", "P3", "P4", "O1", "O2"]
fs = 250.0

# 1. Synthesize 4 s of 8-channel EEG
signal = EEGSignalGenerator(fs=fs, seed=0).get_chunk(duration=4.0,
                                                     n_channels=len(labels))

# 2. Band-pass filter, 1–40 Hz (4th-order Butterworth, zero-phase)
signal = IIRFilter(order=4, cutoff=(1.0, 40.0), btype="bandpass").fit_transform(
    signal, fs=fs)

# 3. Power spectral density (Welch)
f, psd = power_spectral_density(signal, fs=fs)

# 4. Absolute alpha-band (8–13 Hz) power per channel
alpha = band_power(psd, fs=fs, band=(8.0, 13.0))

# 5. Scalp topography
channel_set = ChannelSet().add_unipolar_eeg_channels(labels)
fig, ax = plt.subplots()
plot_topography(alpha[0], channel_set, ax, colorbar=True)
plt.show()
```

Every routine takes its inputs explicitly — an array, a sampling rate, a few
parameters — and returns arrays or scalars. There is nothing hidden in a
container object, so any step can be reused, tested, or swapped in isolation.

### The signal model

Signal processing functions operate on plain NumPy arrays with channels last:

| Representation | Shape |
| --- | --- |
| Time-domain | `(n_segments, n_samples, n_channels)` |
| Power spectral density | `(n_segments, n_frequencies, n_channels)` |
| Time–frequency | `(n_segments, n_frequencies, n_times, n_channels)` |

`n_segments` is the number of independent windows/epochs (use `1` for a single
continuous recording). As a convenience, single-segment functions also accept a
2-D `(n_samples, n_channels)` array and return a result with the segment axis
removed, so interactive and streaming use stays terse.

---

## Working with recordings

The data model is BIDS-aligned. A `Signal` is one acquisition stream — a
`(n_samples, n_channels)` matrix with a sampling rate, a `ChannelSet`, and a time
vector. Per-channel modality (EEG, EOG, …) lives in the `ChannelSet`, so a single
`Signal` can hold mixed channel types. A `Recording` groups one or more named
streams for a single run, together with an event timeline and metadata.

```python
import numpy as np
from medusa.core.data import ChannelSet, Signal, Recording, BidsInfo

channel_set = ChannelSet().add_unipolar_eeg_channels(["Fz", "Cz", "Pz"])
signal = Signal(np.zeros((1000, 3)), fs=250.0, channel_set=channel_set)

recording = Recording(BidsInfo("01", task="rest")).add_signal("eeg", signal)
recording.save("rest.h5")
loaded = Recording.load("rest.h5")
```

Recordings can be saved to several formats, chosen from the file extension:
`bson` (compact binary), `json` (human-readable), `mat` (MATLAB), and `h5` /
`hdf5` (chunked, compressed, append-friendly). Channel positions resolve from
bundled standard EEG montages (10-20, 10-10, 10-05) by label.

---

## Deep learning

Deep-learning models are built as scikit-learn–style estimators over a PyTorch
core, and are imported on demand so the rest of the library stays
PyTorch-free:

```python
from medusa.ml.torch_models.backbones import EEGNet
from medusa.ml.torch_models.classification import TorchClassifier

clf = TorchClassifier(EEGNet(n_cha=8, samples=128), max_epochs=50, val_split=0.2)
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
```

Backbones (`EEGInception`, `EEGInceptionV2`, `EEGNet`, `EEGSym`) are plain feature
extractors; `TorchClassifier` adds a classification head and exposes the familiar
`fit` / `predict` / `predict_proba` / `score` interface. Trained estimators save to
a single portable file that reloads across devices (CPU ↔ GPU). If PyTorch is not
installed, importing these modules raises a clear error telling you what to
install.

---

## What's inside

| Package | Contents |
| --- | --- |
| `medusa.core` | Foundation: the BIDS-aligned data model (`Signal`, `ChannelSet`, `Recording`, `Events`), serialization, and shared utilities. |
| `medusa.signal` | Array operations: frequency and spatial filters, segmentation, transforms (PSD, spectrograms, Hilbert), artifact removal, orthogonalization, and signal generators. |
| `medusa.signal.metrics` | Signal metrics by family: `spectral`, `nonlinear`, `discriminability`, `connectivity`. |
| `medusa.graph` | Graph-theoretic metrics over weighted adjacency matrices. |
| `medusa.ml` | scikit-learn–style machine-learning and deep-learning estimators (PyTorch, opt-in). |
| `medusa.plots` | Matplotlib visualizations: scalp topographies, connectivity maps, time series, time–frequency heatmaps, and summary plots. |
| `medusa.widgets` | Interactive GUI tools (figure browser, time-series viewer, settings editors). |

Visual identity (colors, fonts, themes) comes from the companion
[`medusa-style`](https://www.medusabci.com/) package, so every plot and widget
shares a consistent look.

---

## License

medusa-kernel is released under the **Apache License 2.0**. See `LICENSE` and
`NOTICE` for details.
