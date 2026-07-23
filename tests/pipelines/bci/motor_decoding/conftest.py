"""Shared fixtures for the motor-decoding analysis tests.

A fully-synthetic motor-imagery recording with **lateralized mu ERD**: every trial has a class
(0 or 1); during the trial the mu rhythm (10 Hz) desynchronizes over the contralateral channel
(C3 for class 0, C4 for class 1), the textbook motor-imagery signature. This gives the analysis
functions a real ERD to find and a real class difference to separate.
"""
import numpy as np
import pytest

from medusa.core.data import BidsInfo, Recording, Signal, Events, ChannelSet

FS = 250.0
CHANNELS = ["C3", "CZ", "C4"]
MU_HZ = 10.0
#: Channel that desynchronizes for each class (contralateral to the imagined hand).
ERD_CHANNEL = {0: "C3", 1: "C4"}


def make_mi_recording(labels, *, fs=FS, trial_s=4.0, iti_s=2.0, erd_depth=0.45, seed=0):
    """Synthesise a lateralized-ERD motor-imagery :class:`Recording` from a list of class labels.

    A 10 Hz mu rhythm runs on every channel. During each trial (from its onset, for ``trial_s``
    seconds) the class's ERD channel loses a fraction ``erd_depth`` of its mu amplitude, so band
    power drops there -- an ERD. Trials are spaced by ``iti_s`` seconds of rest (mu at full
    amplitude), which serves as the pre-onset baseline. Events carry one row per trial with
    ``trial_idx`` and ``label``.
    """
    rng = np.random.default_rng(seed)
    n_trials = len(labels)
    spt, ipt = int(round(trial_s * fs)), int(round(iti_s * fs))
    total = n_trials * (spt + ipt) + ipt                    # trailing rest pad
    t = np.arange(total) / fs
    mu = np.sin(2 * np.pi * MU_HZ * t)

    signal = 0.3 * rng.standard_normal((total, len(CHANNELS)))
    signal += 0.6 * mu[:, None]                             # baseline mu on every channel

    events = Events(optional_columns={"trial_idx": "Int64", "label": "Int64"})
    rows = []
    for i, label in enumerate(labels):
        onset = i * (spt + ipt) + ipt                       # trial starts after a rest gap
        col = CHANNELS.index(ERD_CHANNEL[int(label)])
        seg = slice(onset, onset + spt)
        signal[seg, col] -= erd_depth * 0.6 * mu[seg]       # cancel part of the mu -> ERD
        rows.append({"onset": onset / fs, "duration": trial_s,
                     "trial_idx": i, "label": int(label)})
    events.append(rows)

    channel_set = ChannelSet()
    channel_set.add_unipolar_eeg_channels(list(CHANNELS))
    rec = Recording(BidsInfo(subject="synthetic", task="mi"))
    rec.add_signal("eeg", Signal(signal, fs=fs, channel_set=channel_set))
    rec.set_events(events)
    return rec


@pytest.fixture
def mi_channels():
    return list(CHANNELS)


@pytest.fixture
def mi_recording():
    """A synthetic MI recording: 16 trials, classes alternating 0/1, lateralized mu ERD."""
    labels = [0, 1] * 8
    return make_mi_recording(labels, seed=1)


@pytest.fixture
def mi_recording_factory():
    return make_mi_recording
