"""Legacy-load tests for 1.x BCI paradigm recordings.

The 2.0 kernel must still *load* recordings written by Kernel 1.x for every BCI
paradigm. The paradigm ``ExperimentData`` subclasses needed for this live as
lightweight, processing-free copies under :mod:`medusa.core.legacy.bci`, and the
1.x ``medusa.bci.*`` module names are remapped to them in
:data:`medusa.core.legacy.recording._LEGACY_MODULE_MAP`.

Two layers are covered:

* a fast, self-contained round-trip of each paradigm data class (no data files);
* an integration load of the 1.x recordings bundled under
  ``tests/core/legacy/data`` (skipped only if those files are missing).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from medusa.core.legacy.bci import (
    CVEPSpellerData,
    ERPSpellerData,
    MIData,
    NeurofeedbackData,
    SSVEPSpellerData,
)
from medusa.core.legacy.recording import Recording

# 1.x recordings live next to this test (one per paradigm), NOT in tutorials.
_LEGACY_DATA = Path(__file__).resolve().parent / "data"


# ---------------------------------------------------------------------------
# Data-class round-trips (no files): construct -> serialize -> reconstruct
# ---------------------------------------------------------------------------

class TestParadigmDataRoundTrip:

    def test_cvep_speller_data(self):
        d = CVEPSpellerData(
            mode="train", paradigm_conf=[[[[0, 1]]]],
            commands_info=[{0: {}}], onsets=[0.1, 0.2],
            command_idx=[0], unit_idx=[0], level_idx=[0], matrix_idx=[0],
            cycle_idx=[0], trial_idx=[0], spell_result=[0], fps_resolution=60,
        )
        r = CVEPSpellerData.from_serializable_obj(d.to_serializable_obj())
        assert r.mode == "train"
        assert r.fps_resolution == 60

    def test_mi_data(self):
        d = MIData(mode="train", onsets=[0.1, 0.2], w_trial_t=[500, 4000],
                   mi_labels=[0, 1], mi_labels_info={0: "left", 1: "right"})
        r = MIData.from_serializable_obj(d.to_serializable_obj())
        assert r.mode == "train"
        np.testing.assert_array_equal(r.mi_labels, [0, 1])

    def test_erp_speller_data(self):
        # test mode keeps construction simple; the train-mode compute_*_labels
        # path is exercised by the real-file integration load below.
        paradigm_conf = [[[[[0, 1], [2, 3]], [[0, 2], [1, 3]]]]]
        d = ERPSpellerData(
            mode="test", paradigm_conf=paradigm_conf,
            commands_info=[{0: {}, 1: {}, 2: {}, 3: {}}],
            onsets=[0.1, 0.2, 0.3, 0.4], batch_idx=[0, 1, 0, 1],
            group_idx=[0, 0, 1, 1], unit_idx=[0, 0, 0, 0],
            level_idx=[0, 0, 0, 0], matrix_idx=[0, 0, 0, 0],
            sequence_idx=[0, 0, 0, 0], trial_idx=[0, 0, 0, 0],
            spell_result=[0], control_state_result=[1],
        )
        r = ERPSpellerData.from_serializable_obj(d.to_serializable_obj())
        assert r.mode == "test"
        assert r.erp_labels is None  # not computed in test mode

    def test_ssvep_speller_data(self):
        d = SSVEPSpellerData(
            mode="train", paradigm_conf=[[[[0, 1]]]], commands_info=[{0: {}}],
            onsets=[0.1], unit_idx=[0], level_idx=[0], matrix_idx=[0],
            trial_idx=[0], cmd_model=None, csd_model=None, spell_result=[0],
            control_state_result=[1], fps_resolution=60, stim_time=4.0,
            stim_freq_range=[8.0, 10.0],
        )
        r = SSVEPSpellerData.from_serializable_obj(d.to_serializable_obj())
        assert r.mode == "train"

    def test_neurofeedback_data(self):
        d = NeurofeedbackData(
            run_onsets=[0.0], run_durations=[10.0], run_success=[True],
            run_pauses=[0], run_restarts=[0], medusa_nft_app_settings={},
            nft_values=[1.0, 2.0], nft_times=[0.1, 0.2], nft_baseline=0.5,
        )
        r = NeurofeedbackData.from_serializable_obj(d.to_serializable_obj())
        np.testing.assert_array_equal(r.nft_values, [1.0, 2.0])


# ---------------------------------------------------------------------------
# Integration: load the bundled 1.x recordings (tests/core/legacy/data)
# ---------------------------------------------------------------------------

# (filename under tests/core/legacy/data, expected ExperimentData class)
_DATA_FILES = [
    ("rec.cvep.bson", CVEPSpellerData),
    ("rec.mi.bson", MIData),
    ("rec.rcp.bson", ERPSpellerData),
    ("rec.rec.bson", None),  # CustomExperimentData; just load it
]


@pytest.mark.parametrize("relpath, exp_cls", _DATA_FILES)
def test_load_legacy_recording(relpath, exp_cls):
    path = _LEGACY_DATA / relpath
    if not path.exists():
        pytest.skip(f"legacy recording not present: {relpath}")
    rec = Recording.load(str(path))
    assert isinstance(rec, Recording)
    # EEG biosignal reconstructs with a channel set.
    eeg = getattr(rec, list(rec.biosignals)[0])
    assert eeg.signal.shape[1] == eeg.channel_set.n_cha
    assert len(eeg.times) == eeg.signal.shape[0]
    # Experiment reconstructs to the expected paradigm class.
    if exp_cls is not None:
        exp = getattr(rec, list(rec.experiments)[0])
        assert isinstance(exp, exp_cls)
        assert exp.mode in ("train", "test", "guided test")


def test_legacy_module_map_points_at_legacy_paradigms():
    """The 1.x ``medusa.bci.*`` paradigm modules remap into ``core.legacy.bci``
    (the processing-free copies), never the heavy ``pipelines.bci.*``."""
    from medusa.core.legacy.recording import _LEGACY_MODULE_MAP

    for legacy_name in ("medusa.bci.cvep_spellers", "medusa.bci.erp_spellers",
                        "medusa.bci.mi_paradigms", "medusa.bci.ssvep_spellers",
                        "medusa.bci.nft_paradigms"):
        target = _LEGACY_MODULE_MAP[legacy_name]
        assert target.startswith("medusa.core.legacy.bci")
