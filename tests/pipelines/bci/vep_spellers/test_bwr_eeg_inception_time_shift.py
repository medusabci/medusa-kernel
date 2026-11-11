"""Time shifts inside :class:`BWREEGInceptionPipeline`.

These tests check the wiring, not the accuracy, so the network is never trained:

* ``fit`` must cut each recording's cycles at its recorded onsets plus its time shift
  (plus its target's raster latency, when that correction is on);
* ``estimate_time_shift`` must return the shift where the *target's* score peaks, without
  the raster latency;
* ``predict`` and ``estimate_time_shift`` must filter a recording once, however many
  shifts they try.

Skipped on the no-extras CI job: this pipeline is torch-gated.
"""
import dataclasses

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("lightning")

import medusa.pipelines.bci.vep_spellers.decoding.bwr_eeg_inception as bwr_eeg_inception
from medusa.pipelines.bci.vep_spellers import (
    BWREEGInceptionPipeline,
    bwr_eeg_inception_settings,
    command_raster_latencies,
    cycle_arrays,
    generate_random_codebook,
)

FPS, FS, N_FRAMES, N_CYCLES = 60.0, 256.0, 30, 2
SETTINGS = dict(band=(1.0, 40.0), order=5, w_segment_t=(0.0, 250.0),
                baseline_t=(0.0, 250.0), target_fs=None)
SCAN_FRACTION = 0.95

#: A 2 x 2 grid: commands 0 and 1 on the top row, 2 and 3 on the bottom one.
GRID = {"0": (0.25, 0.25), "1": (0.75, 0.25), "2": (0.25, 0.75), "3": (0.75, 0.75)}
#: A single command at the centre, like a calibration run.
CENTRE = {"0": (0.5, 0.5)}


def _recording(factory, spell_target, positions=GRID, seed=0):
    """A c-VEP recording with one command per entry of ``positions``."""
    cmds = generate_random_codebook(len(positions), n_frames=N_FRAMES, seed=0)
    cmds = {uid: dataclasses.replace(cmd, screen_pos=positions[uid])
            for uid, cmd in cmds.items()}
    return factory(cmds, spell_target, fps=FPS, fs=FS, n_cycles=N_CYCLES, seed=seed,
                   mode="train")


def _pipeline(channels, **overrides):
    return BWREEGInceptionPipeline(settings=bwr_eeg_inception_settings(**SETTINGS),
                                   channels=channels, **overrides)


def _latencies(recording, scan_fraction):
    """Raster latency of each command (s); zeros when the correction is off."""
    sd = recording.experiment
    if scan_fraction:
        return command_raster_latencies(sd, scan_fraction)
    return np.zeros(len(sd.command_uids))


def _stub_peaked_scoring(pipe, recording, peaks, latencies):
    """Replace the scoring pass with a stub whose scores peak at known time shifts.

    ``peaks`` maps each command uid to ``(peak_shift, height)``. A command's score is
    ``height - |onset shift - (peak_shift + latency)|``, where the onset shift is the mean
    shift between the onsets the pass gets and the recorded ones. So read from the pass
    at its own raster latency, a command's score peaks at time shift ``peak_shift``.
    The pipeline is marked fitted: the stub replaces the classifier.
    """
    recorded_onsets = cycle_arrays(recording.events)[0]
    uids = recording.experiment.command_uids
    peak_shift = np.array([peaks[uid][0] for uid in uids])
    height = np.array([peaks[uid][1] for uid in uids])

    def command_scores(filtered_signal, sd, onsets, trial_idx, cycle_idx, code_idx):
        onset_shift = float(np.mean(onsets - recorded_onsets))
        row = height - np.abs(onset_shift - (peak_shift + latencies))
        return np.tile(row, (len(onsets), 1))

    pipe._command_scores = command_scores
    pipe._fitted = True


def _stub_flat_scoring(pipe):
    """Replace the scoring pass with a stub that scores every command 1 at every shift."""
    def command_scores(filtered_signal, sd, onsets, trial_idx, cycle_idx, code_idx):
        return np.ones((len(onsets), len(sd.command_uids)))

    pipe._command_scores = command_scores
    pipe._fitted = True


def _stub_training(pipe):
    """Replace feature extraction and training; return the list of cycle onsets ``fit``
    cuts each recording at."""
    seen_onsets = []

    def features(signal, cycle_onsets, n_frames, fps, cfg):
        seen_onsets.append(np.array(cycle_onsets))
        return np.zeros((len(cycle_onsets) * n_frames, 4, 1))

    def fit_classifier(cfg, X, y, groups=None):
        return pipe                                     # no training

    pipe._features = features
    pipe._fit_classifier = fit_classifier
    return seen_onsets


class _RandomClassifier:
    """Stands in for the trained network: random frame probabilities."""
    classes_ = np.array([0, 1])

    def predict_proba(self, X):
        p = np.random.default_rng(0).random(len(X))
        return np.column_stack([1 - p, p])


# --------------------------------------------------------------------------- #
# fit
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("scan_fraction", [None, SCAN_FRACTION])
def test_fit_cuts_each_recording_at_its_time_shift(cvep_recording_factory, cvep_channels,
                                                   scan_fraction):
    recordings = [_recording(cvep_recording_factory, ["0", "2"], seed=0),
                  _recording(cvep_recording_factory, ["3", "1"], seed=1)]
    time_shifts = [0.010, -0.020]
    pipe = _pipeline(cvep_channels, raster_scan_fraction=scan_fraction)
    seen_onsets = _stub_training(pipe)

    pipe.fit(recordings, time_shifts=time_shifts)

    for rec, time_shift, seen in zip(recordings, time_shifts, seen_onsets):
        sd = rec.experiment
        recorded_onsets, trial_idx, _, _ = cycle_arrays(rec.events)
        latency_of = dict(zip(sd.command_uids, _latencies(rec, scan_fraction)))
        target_latency = [latency_of[sd.spell_target[t]] for t in trial_idx]
        np.testing.assert_allclose(seen, recorded_onsets + time_shift + target_latency)
        # The recording itself is not modified
        np.testing.assert_array_equal(cycle_arrays(rec.events)[0], recorded_onsets)


@pytest.mark.parametrize("time_shifts", [[0.01], 0.01, [[0.01, 0.02]]])
def test_fit_needs_one_time_shift_per_recording(cvep_recording_factory, cvep_channels,
                                                time_shifts):
    recordings = [_recording(cvep_recording_factory, ["0"], seed=0),
                  _recording(cvep_recording_factory, ["1"], seed=1)]
    pipe = _pipeline(cvep_channels)
    _stub_training(pipe)

    with pytest.raises(ValueError, match="one value per recording"):
        pipe.fit(recordings, time_shifts=time_shifts)


def test_fit_rejects_a_non_finite_time_shift(cvep_recording_factory, cvep_channels):
    recordings = [_recording(cvep_recording_factory, ["0"], seed=0),
                  _recording(cvep_recording_factory, ["1"], seed=1)]
    pipe = _pipeline(cvep_channels)
    _stub_training(pipe)

    with pytest.raises(ValueError, match="finite"):
        pipe.fit(recordings, time_shifts=[0.0, np.nan])


# --------------------------------------------------------------------------- #
# estimate_time_shift
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("scan_fraction", [None, SCAN_FRACTION])
def test_estimate_finds_the_targets_shift_without_the_raster_latency(
        cvep_recording_factory, cvep_channels, scan_fraction):
    # Every trial targets command 2 (bottom row). The other commands peak elsewhere, and
    # higher, so only the target's score can give its shift.
    recording = _recording(cvep_recording_factory, ["2", "2"])
    pipe = _pipeline(cvep_channels, raster_scan_fraction=scan_fraction)
    peaks = {"0": (-3 / FS, 5.0), "1": (-3 / FS, 5.0), "2": (2 / FS, 1.0),
             "3": (1 / FS, 5.0)}
    _stub_peaked_scoring(pipe, recording, peaks, _latencies(recording, scan_fraction))

    time_shift, candidate_shifts, target_scores = pipe.estimate_time_shift(
        recording, max_time_shift=4 / FS)

    assert time_shift == pytest.approx(2 / FS)
    # Every candidate, one sample apart and ascending, with the target's score at it
    np.testing.assert_allclose(candidate_shifts, np.arange(-4, 5) / FS)
    np.testing.assert_allclose(target_scores, 1.0 - np.abs(candidate_shifts - 2 / FS))


def test_estimate_works_with_a_single_command(cvep_recording_factory, cvep_channels):
    recording = _recording(cvep_recording_factory, ["0", "0"], positions=CENTRE)
    pipe = _pipeline(cvep_channels, raster_scan_fraction=SCAN_FRACTION)
    _stub_peaked_scoring(pipe, recording, {"0": (-3 / FS, 1.0)},
                         _latencies(recording, SCAN_FRACTION))

    time_shift, _, _ = pipe.estimate_time_shift(recording, max_time_shift=5 / FS)

    assert time_shift == pytest.approx(-3 / FS)


def test_estimate_tries_whole_samples_and_ties_keep_the_centre(cvep_recording_factory,
                                                               cvep_channels):
    recording = _recording(cvep_recording_factory, ["0", "3"])
    pipe = _pipeline(cvep_channels)
    _stub_flat_scoring(pipe)

    time_shift, candidate_shifts, _ = pipe.estimate_time_shift(
        recording, max_time_shift=3.4 / FS, center_time_shift=0.010)

    # 3.4 samples rounds to 3 on each side; the centre is one of the candidates
    np.testing.assert_allclose(candidate_shifts, 0.010 + np.arange(-3, 4) / FS)
    assert time_shift == pytest.approx(0.010)


def test_estimate_needs_the_target_of_every_trial(cvep_recording_factory, cvep_channels):
    recording = _recording(cvep_recording_factory, ["0", "3"])
    recording.experiment.spell_target[1] = None
    pipe = _pipeline(cvep_channels)
    _stub_flat_scoring(pipe)

    with pytest.raises(ValueError, match="trial 1 has no target"):
        pipe.estimate_time_shift(recording, max_time_shift=2 / FS)


def test_estimate_needs_a_fitted_pipeline(cvep_recording_factory, cvep_channels):
    recording = _recording(cvep_recording_factory, ["0"])

    with pytest.raises(RuntimeError, match="not fitted"):
        _pipeline(cvep_channels).estimate_time_shift(recording, max_time_shift=2 / FS)


def test_estimate_rejects_a_negative_search_width(cvep_recording_factory, cvep_channels):
    recording = _recording(cvep_recording_factory, ["0"])
    pipe = _pipeline(cvep_channels)
    _stub_flat_scoring(pipe)

    with pytest.raises(ValueError, match="max_time_shift"):
        pipe.estimate_time_shift(recording, max_time_shift=-2 / FS)


# --------------------------------------------------------------------------- #
# Filtering once
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("method", ["predict", "estimate_time_shift"])
def test_a_recording_is_filtered_once(cvep_recording_factory, cvep_channels, monkeypatch,
                                      method):
    recording = _recording(cvep_recording_factory, ["0", "2"])
    pipe = _pipeline(cvep_channels, raster_scan_fraction=SCAN_FRACTION)
    pipe.clf = _RandomClassifier()
    pipe._fitted = True
    calls = []
    real_filter = bwr_eeg_inception.apply_notch_and_filterbank

    def counting_filter(*args, **kwargs):
        calls.append(1)
        return real_filter(*args, **kwargs)

    monkeypatch.setattr(bwr_eeg_inception, "apply_notch_and_filterbank", counting_filter)

    # 5 candidate shifts x 2 screen rows = 10 scoring passes. The shifts are all
    # positive, so every segment of the first cycle stays inside the recording.
    getattr(pipe, method)(recording, max_time_shift=2 / FS, center_time_shift=3 / FS)

    assert len(calls) == 1
