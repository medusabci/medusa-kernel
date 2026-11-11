"""Raster-latency correction inside :class:`BWREEGInceptionPipeline`.

These tests check the wiring, not the accuracy, so the network is never trained:

* ``fit`` must cut the calibration epochs at onsets shifted by each target's latency;
* ``predict`` must score each command at its own latency, with one scoring pass per
  distinct latency, on top of the time-shift search.

The scoring pass is replaced by a stub that scores every command by the onset shift it
was given, so the scores show which pass each command came from.

Skipped on the no-extras CI job: this pipeline is torch-gated.
"""
import dataclasses

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("lightning")

from medusa.pipelines.bci.vep_spellers import (
    BWREEGInceptionPipeline,
    bwr_eeg_inception_settings,
    command_raster_latencies,
    cycle_arrays,
    generate_random_codebook,
)

FPS, FS, N_FRAMES, N_CYCLES = 60.0, 256.0, 30, 2
SETTINGS = dict(band=(1.0, 40.0), order=5, w_segment_t=(0.0, 250.0), target_fs=None)
SCAN_FRACTION = 0.95

#: A 2 x 2 grid: commands 0 and 1 on the top row, 2 and 3 on the bottom one.
GRID = {"0": (0.25, 0.25), "1": (0.75, 0.25), "2": (0.25, 0.75), "3": (0.75, 0.75)}


@pytest.fixture
def recording(cvep_recording_factory):
    """A 4-command c-VEP recording on the 2 x 2 GRID, one trial per command."""
    cmds = generate_random_codebook(len(GRID), n_frames=N_FRAMES, seed=0)
    cmds = {uid: dataclasses.replace(cmd, screen_pos=GRID[uid]) for uid, cmd in cmds.items()}
    return cvep_recording_factory(cmds, list(cmds), fps=FPS, fs=FS, n_cycles=N_CYCLES,
                                  mode="train")


def _pipeline(channels, **overrides):
    return BWREEGInceptionPipeline(settings=bwr_eeg_inception_settings(**SETTINGS),
                                   channels=channels, **overrides)


def _stub_scoring(pipe, recording):
    """Replace the scoring pass with a stub; return the list of onset shifts it gets.

    The stub scores every command with the mean shift between the onsets it was given
    and the recorded ones. The pipeline is marked fitted: the stub replaces the classifier.
    """
    recorded_onsets = cycle_arrays(recording.events)[0]
    shifts = []

    def command_scores(recording, sd, onsets, trial_idx, cycle_idx, code_idx):
        shift = float(np.mean(onsets - recorded_onsets))
        shifts.append(shift)
        return np.full((len(onsets), len(sd.command_uids)), shift)

    pipe._command_scores = command_scores
    pipe._fitted = True
    return shifts


def _latencies(recording):
    return command_raster_latencies(recording.experiment, SCAN_FRACTION)


# --------------------------------------------------------------------------- #
# Setting
# --------------------------------------------------------------------------- #
def test_the_correction_is_off_by_default(cvep_channels):
    assert _pipeline(cvep_channels).cfg["raster_scan_fraction"] is None
    pipe = _pipeline(cvep_channels, raster_scan_fraction=SCAN_FRACTION)
    assert pipe.cfg["raster_scan_fraction"] == SCAN_FRACTION


# --------------------------------------------------------------------------- #
# predict
# --------------------------------------------------------------------------- #
def test_off_runs_one_pass_at_the_recorded_onsets(recording, cvep_channels):
    pipe = _pipeline(cvep_channels)
    shifts = _stub_scoring(pipe, recording)

    scores, time_shift = pipe.predict(recording)

    assert shifts == [0.0]
    assert time_shift == 0.0
    np.testing.assert_array_equal(scores, 0.0)


def test_on_scores_each_command_at_its_own_latency(recording, cvep_channels):
    pipe = _pipeline(cvep_channels, raster_scan_fraction=SCAN_FRACTION)
    shifts = _stub_scoring(pipe, recording)
    latencies = _latencies(recording)

    scores, _ = pipe.predict(recording)

    # One pass per screen row (2), not per command (4)
    np.testing.assert_allclose(sorted(shifts), np.unique(latencies))
    # Each command's scores come from the pass at its own latency
    np.testing.assert_allclose(scores, np.tile(latencies, (len(scores), 1)))


def test_time_shift_and_raster_latency_add_up(recording, cvep_channels):
    pipe = _pipeline(cvep_channels, raster_scan_fraction=SCAN_FRACTION)
    shifts = _stub_scoring(pipe, recording)
    time_shifts = np.linspace(-2 / FS, 2 / FS, 5)       # 5 candidates, one sample apart

    _, time_shift = pipe.predict(recording, max_time_shift=2 / FS)

    # Every candidate time shift runs one pass per distinct latency
    expected = sorted(t + lat for t in time_shifts for lat in np.unique(_latencies(recording)))
    np.testing.assert_allclose(sorted(shifts), expected)
    # The stub makes every candidate tie, and ties keep the centre. The returned shift
    # does not include the raster latencies.
    assert time_shift == pytest.approx(0.0)


def test_on_needs_every_screen_pos(recording, cvep_channels):
    pipe = _pipeline(cvep_channels, raster_scan_fraction=SCAN_FRACTION)
    _stub_scoring(pipe, recording)
    recording.experiment.commands_info["2"].screen_pos = None

    with pytest.raises(ValueError, match="screen_pos"):
        pipe.predict(recording)


# --------------------------------------------------------------------------- #
# fit
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("scan_fraction", [None, SCAN_FRACTION])
def test_fit_cuts_each_cycle_at_its_targets_latency(recording, cvep_channels,
                                                    scan_fraction):
    pipe = _pipeline(cvep_channels, raster_scan_fraction=scan_fraction)
    seen_onsets = []

    def features(signal, cycle_onsets, n_frames, fps, cfg):
        seen_onsets.append(np.array(cycle_onsets))
        return np.zeros((len(cycle_onsets) * n_frames, 4, 1))

    def fit_classifier(cfg, X, y, groups=None):
        return pipe                                     # no training

    pipe._features = features
    pipe._fit_classifier = fit_classifier
    recorded_onsets, trial_idx, _, _ = cycle_arrays(recording.events)

    pipe.fit([recording])

    sd = recording.experiment
    if scan_fraction:
        latency_of = dict(zip(sd.command_uids, _latencies(recording)))
        expected = recorded_onsets + [latency_of[sd.spell_target[t]] for t in trial_idx]
    else:
        expected = recorded_onsets
    np.testing.assert_allclose(seen_onsets[0], expected)
    # The recording itself is not modified
    np.testing.assert_array_equal(cycle_arrays(recording.events)[0], recorded_onsets)
