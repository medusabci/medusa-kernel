"""SSVEP decoding across the three ``reference`` modes, focused on ``mixed_harmonics_template``.

``mixed_harmonics_template`` is the extended-CCA (eCCA) mode: it fuses the calibration-free
synthetic harmonics with the calibrated subject template. These tests lock that it decodes
synthetic SSVEP, that fusing helps over the calibration-free harmonics alone, its validation
and persistence, and the properties of the :func:`_ecca_score` core.
"""
import warnings

import numpy as np
import pytest

from medusa.pipelines.base import DecodingPipeline
from medusa.pipelines.bci.vep_spellers import (
    generate_freq_codebook, generate_random_codebook, TMCCAPipeline, cycle_arrays,
    select_commands,
    command_decoding_accuracy_per_cycle)
from medusa.pipelines.bci.vep_spellers.decoding.template_matching import (
    _ecca_score, _cca_reference)

FPS, FS, N_CMDS, N_CYCLES = 60.0, 250.0, 4, 8
BANDPASS = {"filterbank": [
    {"filt_type": "iir", "band_type": "bandpass", "cutoff": [6.0, 40.0], "order": 5}]}


@pytest.fixture
def ssvep_train_test(ssvep_recording_factory):
    """A (train, test, channels) triple of synthetic SSVEP recordings (each command attended)."""
    with warnings.catch_warnings():                  # a few freqs snap to frame-lockable values
        warnings.simplefilter("ignore")
        cmds = generate_freq_codebook(N_CMDS, freq_range=(8.0, 15.0), t_stim=1.0,
                                      fps_resolution=FPS)
    uids = list(cmds)
    from tests.pipelines.bci.vep_spellers.conftest import CHANNELS
    train = ssvep_recording_factory(cmds, uids * 3, fps=FPS, fs=FS, n_cycles=N_CYCLES,
                                    resp_amp=0.25, seed=1, mode="train")
    test = ssvep_recording_factory(cmds, uids + uids[::-1], fps=FPS, fs=FS,
                                   n_cycles=N_CYCLES, resp_amp=0.25, seed=2, mode="test")
    return train, test, list(CHANNELS)


def _curve(pipe, test):
    sd = test.experiment
    _, trial, cycle, _ = cycle_arrays(test.events)
    _, per_cycle, _ = select_commands(pipe.predict(test), sd.command_uids, trial, cycle,
                                      sd.trial_available_cmmds)
    target = {t: sd.spell_target[t] for t in range(len(sd.spell_target))}
    return 100.0 * command_decoding_accuracy_per_cycle(per_cycle, target)


def _pipe(channels, mode):
    return TMCCAPipeline(channels=channels, freq_filtering=BANDPASS,
                         reference={"mode": mode, "n_harmonics": 3})


class TestEccaDecoding:

    def test_mixed_decodes_ssvep(self, ssvep_train_test):
        train, test, channels = ssvep_train_test
        pipe = _pipe(channels, "mixed_harmonics_template").fit([train])
        assert _curve(pipe, test)[-1] == 100.0

    def test_mixed_beats_calibration_free_harmonics(self, ssvep_train_test):
        """Fusing the template with the harmonics does at least as well as harmonics alone."""
        train, test, channels = ssvep_train_test
        harmonics = _pipe(channels, "synthetic_harmonics").fit([])
        mixed = _pipe(channels, "mixed_harmonics_template").fit([train])
        h_curve, m_curve = _curve(harmonics, test), _curve(mixed, test)
        assert h_curve[-1] < 100.0                   # calibration-free leaves accuracy on the table
        assert m_curve[-1] > h_curve[-1]             # fusion recovers it

    def test_mixed_persistence_round_trip(self, ssvep_train_test, tmp_path):
        train, test, channels = ssvep_train_test
        pipe = _pipe(channels, "mixed_harmonics_template").fit([train])
        before = pipe.predict(test)
        path = tmp_path / "ecca.pkl"
        pipe.save(str(path))
        reloaded = DecodingPipeline.load(str(path))
        assert isinstance(reloaded, TMCCAPipeline)
        np.testing.assert_allclose(reloaded.predict(test), before)


class TestEccaValidation:

    def test_needs_fit(self, ssvep_train_test):
        _, test, channels = ssvep_train_test
        with pytest.raises(RuntimeError, match="not fitted"):
            _pipe(channels, "mixed_harmonics_template").predict(test)

    def test_needs_stim_freq(self, cvep_recording_factory, cvep_channels):
        """eCCA needs a per-command frequency; a random-code c-VEP recording has none."""
        cmds = generate_random_codebook(N_CMDS, n_frames=60, seed=0)
        rec = cvep_recording_factory(cmds, list(cmds), n_cycles=2, resp_amp=0.0, seed=0)
        with pytest.raises(ValueError, match="stim_freq"):
            _pipe(cvep_channels, "mixed_harmonics_template").fit([rec])

    def test_unknown_mode_rejected(self, ssvep_train_test):
        train, _, channels = ssvep_train_test
        with pytest.raises(ValueError, match="reference.mode"):
            _pipe(channels, "not_a_mode").fit([train])


class TestEccaScore:
    """Properties of the eCCA fusion core (exact values need 4 CCA fits; test behaviour)."""

    @staticmethod
    def _flicker(freq, amp, seed, n=250, fs=250.0, n_ch=4):
        rng = np.random.default_rng(seed)
        x = rng.standard_normal((n, n_ch))
        t = np.arange(n) / fs
        x[:, :2] += amp * np.sin(2 * np.pi * freq * t)[:, None]
        return x

    def test_matching_scores_higher_than_off_frequency(self):
        y = _cca_reference(10.0, 250, 250.0, 3)
        template = self._flicker(10.0, 1.0, seed=0)
        seg_match = self._flicker(10.0, 1.0, seed=1)         # same 10 Hz flicker
        seg_off = self._flicker(13.0, 1.0, seed=2)           # a different frequency
        assert _ecca_score(seg_match, y, template) > _ecca_score(seg_off, y, template)

    def test_returns_finite_on_degenerate_input(self):
        y = _cca_reference(10.0, 250, 250.0, 3)
        template = self._flicker(10.0, 1.0, seed=0)
        score = _ecca_score(np.zeros((250, 4)), y, template)  # constant segment
        assert np.isfinite(score)
