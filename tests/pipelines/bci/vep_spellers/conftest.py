"""Shared fixtures for the VEP-speller pipeline tests.

Everything here is **fully synthetic** (no bundled data): a small helper builds a c-VEP
:class:`~medusa.core.data.recording.Recording` whose EEG carries a bit-wise VEP response,
so the bit-wise-reconstruction (BWR) and template-matching decoders have real structure to
learn. The response model is deliberately simple and code-agnostic: at every frame whose
code bit is ``1`` (a flash) the occipital channels get an evoked bump; ``0`` frames get
nothing. Any codebook -- m-sequence, Gold, or random -- drives it the same way, which is
exactly what lets us test random-code c-VEP decoding against the same machinery.
"""
import numpy as np
import pytest

from medusa.core.data import BidsInfo, Recording, Signal, Events, ChannelSet
from medusa.pipelines.bci.vep_spellers import SpellerData

#: A small occipital-ish montage; the first few carry the response, the rest are noise.
CHANNELS = ["O1", "OZ", "O2", "POZ", "PO3", "PO4", "PZ", "CZ"]
RESPONSE_CHANNELS = ["O1", "OZ", "O2", "POZ"]


def make_cvep_recording(commands_info, spell_target, *, fps=60.0, fs=256.0,
                        n_cycles=10, channels=CHANNELS, resp_amp=2.5,
                        resp_ms=60.0, seed=0, mode="test"):
    """Synthesise a c-VEP :class:`Recording` from a codebook and per-trial targets.

    Each command's first code (``codes[:, 0, :]``) drives the stimulation. For every trial,
    over ``n_cycles`` repetitions, the frames where the *target's* code is ``1`` inject a
    short evoked bump (a raised-cosine transient spanning ``resp_ms``) into
    ``RESPONSE_CHANNELS``, on top of unit Gaussian noise. The transient is short so a frame's
    epoch reflects whether *that* frame flashed (a long kernel would smear neighbouring
    frames together and destroy the bit-wise label). Events carry one row per cycle
    (``trial_idx``/``cycle_idx``/``code_idx=0``), matching ``SPELLER_EVENT_COLUMNS``.

    Returns the :class:`Recording` with a :class:`SpellerData` attached (``spell_target``
    set, so it can drive both calibration and scoring).
    """
    rng = np.random.default_rng(seed)
    uids = list(commands_info)
    codes = np.array([np.asarray(c.code)[0] for c in commands_info.values()])  # (n_cmd, n_frames)
    n_frames = codes.shape[1]

    frame_samples = int(round(fs / fps))
    cycle_samples = n_frames * frame_samples
    n_trials = len(spell_target)
    w_samples = max(1, int(round(resp_ms / 1000.0 * fs)))
    kernel = resp_amp * (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(w_samples) / w_samples))

    total = n_trials * n_cycles * cycle_samples + cycle_samples  # one cycle of tail pad
    signal = rng.standard_normal((total, len(channels)))
    resp_cols = [channels.index(c) for c in RESPONSE_CHANNELS if c in channels]

    events = Events(optional_columns={"trial_idx": "Int64", "cycle_idx": "Int64",
                                      "code_idx": "Int64"})
    rows = []
    for trial, tgt in enumerate(spell_target):
        code = codes[uids.index(str(tgt))]
        for cycle in range(n_cycles):
            c0 = (trial * n_cycles + cycle) * cycle_samples
            for f in np.flatnonzero(code):
                f0 = c0 + int(f) * frame_samples
                seg = signal[f0:f0 + w_samples, :]
                seg[:, resp_cols] += kernel[:len(seg), None]
            rows.append({"onset": c0 / fs, "duration": cycle_samples / fs,
                         "trial_idx": trial, "cycle_idx": cycle, "code_idx": 0})
    events.append(rows)

    channel_set = ChannelSet()
    channel_set.add_unipolar_eeg_channels(list(channels))
    rec = Recording(BidsInfo(subject="synthetic", task="cvep"))
    rec.add_signal("eeg", Signal(signal, fs=fs, channel_set=channel_set))
    rec.set_events(events)
    SpellerData(mode=mode, paradigm_conf={}, commands_info=commands_info,
                fps_resolution=fps, spell_target=[str(t) for t in spell_target]
                ).to_recording(rec)
    return rec


@pytest.fixture
def cvep_channels():
    """The synthetic montage used by the c-VEP fixtures."""
    return list(CHANNELS)


def make_ssvep_recording(commands_info, spell_target, *, fps=60.0, fs=250.0,
                         n_cycles=8, t_stim=1.0, channels=CHANNELS, resp_amp=0.25,
                         seed=0, mode="test"):
    """Synthesise an SSVEP :class:`Recording` from a frequency codebook and per-trial targets.

    Each command flickers at its ``extra['stim_freq']``. For every trial, over ``n_cycles``
    repetitions, the occipital channels carry the *target's* flicker (fundamental + 2nd
    harmonic, phase-locked to the cycle onset) on top of Gaussian noise. This mirrors the
    ``vep_spellers_ssvep_usage.py`` example and drives all three ``reference`` modes.
    """
    rng = np.random.default_rng(seed)
    freqs = {uid: float(cmd.extra["stim_freq"]) for uid, cmd in commands_info.items()}
    occ = [channels.index(c) for c in RESPONSE_CHANNELS if c in channels]
    spc = int(round(t_stim * fs))
    local_t = np.arange(spc) / fs
    n_trials = len(spell_target)

    signal = 2.0 * rng.standard_normal((n_trials * n_cycles * spc + spc, len(channels)))
    events = Events(optional_columns={"trial_idx": "Int64", "cycle_idx": "Int64",
                                      "code_idx": "Int64"})
    rows = []
    for trial, tgt in enumerate(spell_target):
        f = freqs[str(tgt)]
        wave = np.sin(2 * np.pi * f * local_t) + 0.5 * np.sin(2 * np.pi * 2 * f * local_t)
        for cycle in range(n_cycles):
            i0 = (trial * n_cycles + cycle) * spc
            signal[i0:i0 + spc, occ] += resp_amp * wave[:, None]
            rows.append({"onset": i0 / fs, "duration": t_stim,
                         "trial_idx": trial, "cycle_idx": cycle, "code_idx": 0})
    events.append(rows)

    channel_set = ChannelSet()
    channel_set.add_unipolar_eeg_channels(list(channels))
    rec = Recording(BidsInfo(subject="synthetic", task="ssvep"))
    rec.add_signal("eeg", Signal(signal, fs=fs, channel_set=channel_set))
    rec.set_events(events)
    SpellerData(mode=mode, paradigm_conf={}, commands_info=commands_info,
                fps_resolution=fps, spell_target=[str(t) for t in spell_target]
                ).to_recording(rec)
    return rec


@pytest.fixture
def cvep_recording_factory():
    """Return :func:`make_cvep_recording` so tests can synthesise c-VEP recordings."""
    return make_cvep_recording


@pytest.fixture
def ssvep_recording_factory():
    """Return :func:`make_ssvep_recording` so tests can synthesise SSVEP recordings."""
    return make_ssvep_recording
