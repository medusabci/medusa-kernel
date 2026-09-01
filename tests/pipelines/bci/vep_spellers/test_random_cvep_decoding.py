"""End-to-end decoding of **random-code** c-VEP spellers.

Random ("noise") c-VEP gives every command an independent random code, with no
circular-shift or frequency structure. This locks in that the code-agnostic decoders handle
it: bit-wise reconstruction (:class:`BWRLDAPipeline`, the natural fit -- it reconstructs the
frame response and correlates each command's code, needing no per-command template) and
calibrated template matching (:class:`TMCCAPipeline`, ``reference='template'`` -- each
distinct random code learns its own template). Both are exercised on the same synthetic
recording, with a fixed-cycle accuracy curve and a fitted-pipeline persistence round-trip.
"""
import numpy as np
import pytest

from medusa.pipelines.base import DecodingPipeline
from medusa.pipelines.bci.vep_spellers import (
    generate_random_codebook, BWRLDAPipeline, TMCCAPipeline, VEPCommandDecoder,
    command_decoding_accuracy_per_cycle)

# Locked synthesis parameters (see conftest): 6 random codes of 126 frames decode cleanly.
# Longer codes lower random cross-correlation; 126 frames is the point where all six codes
# are separable for both the reconstruct-then-correlate BWR and the template decoder.
FPS, FS, N_CMDS, N_FRAMES, N_CYCLES = 60.0, 256.0, 6, 126, 6
BANDPASS = {"filterbank": [
    {"filt_type": "iir", "band_type": "bandpass", "cutoff": [1.0, 40.0], "order": 5}]}


@pytest.fixture
def random_cvep_train_test(cvep_recording_factory, cvep_channels):
    """A (train, test, channels) triple of synthetic random-code c-VEP recordings.

    Calibration attends every command three times (so template matching can learn each
    distinct code); the test spells every command twice.
    """
    cmds = generate_random_codebook(N_CMDS, n_frames=N_FRAMES, seed=0)
    uids = list(cmds)
    train = cvep_recording_factory(cmds, uids * 3, fps=FPS, fs=FS, n_cycles=N_CYCLES,
                                   resp_amp=3.0, seed=1, mode="train")
    test = cvep_recording_factory(cmds, uids + uids[::-1], fps=FPS, fs=FS,
                                  n_cycles=N_CYCLES, resp_amp=3.0, seed=2, mode="test")
    return train, test, cvep_channels


def _accuracy_curve(pipe, test):
    """Per-cycle decoding accuracy (%) of ``pipe`` on one test recording."""
    sd = test.experiment
    result = VEPCommandDecoder().decode(pipe.predict(test), sd, test.events)
    target = {t: sd.spell_target[t] for t in range(len(sd.spell_target))}
    return 100.0 * command_decoding_accuracy_per_cycle(
        result["selected_commands_per_cycle"], target)


def test_bwr_decodes_random_codes(random_cvep_train_test):
    """BWR-LDA reconstructs the frame response and correlates each command's random code."""
    train, test, channels = random_cvep_train_test
    pipe = BWRLDAPipeline(
        channels=channels, freq_filtering=BANDPASS,
        segmentation={"w_segment_t": [0.0, 80.0], "baseline_t": [], "target_fs": None}).fit([train])
    curve = _accuracy_curve(pipe, test)
    assert curve[0] >= 90.0                       # already strong after one cycle
    assert curve[-1] == 100.0                     # perfect by the final cycle


def test_template_matching_decodes_random_codes(random_cvep_train_test):
    """Calibrated template matching learns one template per distinct random code."""
    train, test, channels = random_cvep_train_test
    pipe = TMCCAPipeline(
        channels=channels, freq_filtering=BANDPASS, reference={"mode": "calibrated_template"}).fit([train])
    curve = _accuracy_curve(pipe, test)
    assert curve[-1] == 100.0


def test_predict_shape_matches_cycles(random_cvep_train_test):
    """`predict` returns the cumulative ``(n_cycles, n_commands)`` score matrix."""
    train, test, channels = random_cvep_train_test
    pipe = BWRLDAPipeline(
        channels=channels, freq_filtering=BANDPASS,
        segmentation={"w_segment_t": [0.0, 80.0], "baseline_t": [], "target_fs": None}).fit([train])
    scores = pipe.predict(test)
    n_cycle_rows = test.events.df["cycle_idx"].notna().sum()
    assert scores.shape == (n_cycle_rows, N_CMDS)


def test_bwr_random_persistence_round_trip(random_cvep_train_test, tmp_path):
    """A fitted BWR pipeline saves and reloads to identical predictions (polymorphic load)."""
    train, test, channels = random_cvep_train_test
    pipe = BWRLDAPipeline(
        channels=channels, freq_filtering=BANDPASS,
        segmentation={"w_segment_t": [0.0, 80.0], "baseline_t": [], "target_fs": None}).fit([train])
    before = pipe.predict(test)

    path = tmp_path / "bwr_random.pkl"
    pipe.save(str(path))
    reloaded = DecodingPipeline.load(str(path))    # polymorphic: no class named
    assert isinstance(reloaded, BWRLDAPipeline)
    np.testing.assert_allclose(reloaded.predict(test), before)


def test_template_random_persistence_round_trip(random_cvep_train_test, tmp_path):
    """A fitted template-matching pipeline round-trips its learned templates."""
    train, test, channels = random_cvep_train_test
    pipe = TMCCAPipeline(
        channels=channels, freq_filtering=BANDPASS, reference={"mode": "calibrated_template"}).fit([train])
    before = pipe.predict(test)

    path = tmp_path / "tm_random.pkl"
    pipe.save(str(path))
    reloaded = DecodingPipeline.load(str(path))
    assert isinstance(reloaded, TMCCAPipeline)
    np.testing.assert_allclose(reloaded.predict(test), before)
