"""`SpellerData` serialisation round-trips across bson / json / mat.

The Phase-3 data-class requirement: a speller codebook (with a random c-VEP code, a
per-trial available-command list, and a spell target) must survive every cross-platform
format losslessly, so a run saved from the GUI reloads identically for offline analysis.
"""
import numpy as np
import pytest

from medusa.pipelines.bci.vep_spellers import (
    generate_random_codebook, SpellerData)


def _speller_data():
    """A representative random-code SpellerData: codebook + availability + targets."""
    cmds = generate_random_codebook(4, n_frames=63, seed=0)
    uids = list(cmds)
    return SpellerData(
        mode="test", paradigm_conf={"matrix": "4x1"}, commands_info=cmds,
        fps_resolution=60.0,
        trial_available_cmmds=[uids, uids[:2], uids],   # variable per-trial availability
        spell_target=[uids[0], uids[1], uids[3]])


@pytest.mark.parametrize("fmt", ["bson", "json", "mat"])
def test_speller_data_round_trip(fmt, tmp_path):
    sd = _speller_data()
    path = tmp_path / f"speller.{fmt}"
    sd.save(str(path))
    back = SpellerData.load(str(path))

    assert back.mode == sd.mode
    assert float(back.fps_resolution) == sd.fps_resolution
    assert back.command_uids == sd.command_uids
    np.testing.assert_array_equal(back.codes, sd.codes)
    assert [str(t) for t in back.spell_target] == [str(t) for t in sd.spell_target]
    # availability regroups back to the same nested list-of-lists
    assert [[str(u) for u in row] for row in back.trial_available_cmmds] == \
        [[str(u) for u in row] for row in sd.trial_available_cmmds]


def test_command_extra_survives_round_trip(tmp_path):
    """The paradigm-specific ``extra`` (here ``code_index``) is preserved."""
    sd = _speller_data()
    path = tmp_path / "speller.json"
    sd.save(str(path))
    back = SpellerData.load(str(path))
    assert [c.extra["code_index"] for c in back.commands_info.values()] == \
        [c.extra["code_index"] for c in sd.commands_info.values()]
