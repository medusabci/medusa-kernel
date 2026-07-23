"""Golden tests for the Layer-2 command decoder (pure selector) and metrics.

``VEPCommandDecoder`` does no fitting: it argmaxes the cumulative score row over a trial's
*available* commands, with optional early stopping. These lock the selection rule, the
per-trial availability restriction, the accuracy metrics, and the online ``step``.
"""
import numpy as np
import pytest

from medusa.pipelines.bci.vep_spellers import (
    select_commands, command_decoding_accuracy, command_decoding_accuracy_per_cycle,
    VEPCommandDecoder, SpellerData, CommandInfo)


def _speller_data(uids, trial_available_cmmds=None, spell_target=None):
    """A minimal SpellerData with 1-frame dummy codes, for the selector/decoder tests."""
    cmds = {u: CommandInfo(uid=u, code=[[1, 0]]) for u in uids}
    return SpellerData(mode="test", paradigm_conf={}, commands_info=cmds,
                       fps_resolution=60.0, trial_available_cmmds=trial_available_cmmds,
                       spell_target=spell_target)


# --------------------------------------------------------------------------- #
# select_commands
# --------------------------------------------------------------------------- #
class TestSelectCommands:

    def test_argmax_per_cycle_and_final(self):
        cycle_scores = np.array([[0.1, 0.9], [0.8, 0.2]])        # cycle 0 -> B, cycle 1 -> A
        selected, per_cycle, scores = select_commands(
            cycle_scores, ["A", "B"], np.array([0, 0]), np.array([0, 1]))
        assert per_cycle[0] == {0: "B", 1: "A"}
        assert selected[0] == "A"                                # final = last cycle
        np.testing.assert_allclose(scores[0][0], [0.1, 0.9])

    def test_availability_restricts_choice(self):
        """A trial only selects among its available commands, even if another scores higher."""
        cycle_scores = np.array([[0.1, 0.9]])                    # B scores higher globally
        selected, per_cycle, _ = select_commands(
            cycle_scores, ["A", "B"], np.array([0]), np.array([0]),
            trial_available_cmmds=[["A"]])                        # only A available in trial 0
        assert selected[0] == "A"

    def test_two_trials_independent(self):
        cycle_scores = np.array([[0.9, 0.1], [0.2, 0.8]])
        selected, _, _ = select_commands(
            cycle_scores, ["A", "B"], np.array([0, 1]), np.array([0, 0]))
        assert selected == {0: "A", 1: "B"}

    def test_out_of_order_cycles_sorted(self):
        # rows given as cycle 1 then cycle 0; the final selection must use cycle 1
        cycle_scores = np.array([[0.2, 0.8], [0.9, 0.1]])        # row0=cyc1 -> B, row1=cyc0 -> A
        selected, per_cycle, _ = select_commands(
            cycle_scores, ["A", "B"], np.array([0, 0]), np.array([1, 0]))
        assert per_cycle[0] == {0: "A", 1: "B"}
        assert selected[0] == "B"


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def test_command_decoding_accuracy():
    assert command_decoding_accuracy({0: "A", 1: "B"}, {0: "A", 1: "A"}) == 0.5
    assert command_decoding_accuracy({0: "A"}, {0: "A"}) == 1.0


def test_command_decoding_accuracy_ignores_untargeted_trials():
    assert command_decoding_accuracy({0: "A", 1: "B"}, {0: "A"}) == 1.0


def test_accuracy_per_cycle():
    per_cycle = {0: {0: "A", 1: "A"}, 1: {0: "B", 1: "A"}}
    target = {0: "A", 1: "A"}
    acc = command_decoding_accuracy_per_cycle(per_cycle, target)
    # cycle 0: A right, B wrong -> 0.5 ; cycle 1: both A -> 1.0
    np.testing.assert_allclose(acc, [0.5, 1.0])


# --------------------------------------------------------------------------- #
# VEPCommandDecoder
# --------------------------------------------------------------------------- #
def test_decoder_decode_matches_select_commands(cvep_recording_factory):
    """`decode` is `select_commands` wired to a recording's events + SpellerData."""
    cmds = {"0": CommandInfo(uid="0", code=[[1, 0, 1, 0]]),
            "1": CommandInfo(uid="1", code=[[0, 1, 0, 1]])}
    rec = cvep_recording_factory(cmds, ["0", "1"], n_cycles=2, resp_amp=0.0, seed=0)
    n_cycle_rows = int(rec.events.df["cycle_idx"].notna().sum())
    rng = np.random.default_rng(0)
    cycle_scores = rng.standard_normal((n_cycle_rows, 2))

    out = VEPCommandDecoder().decode(cycle_scores, rec.experiment, rec.events)

    trial = rec.events.df["trial_idx"].to_numpy(int)
    cycle = rec.events.df["cycle_idx"].to_numpy(int)
    selected, per_cycle, _ = select_commands(cycle_scores, ["0", "1"], trial, cycle)
    assert out["selected_commands"] == selected
    assert out["selected_commands_per_cycle"] == per_cycle


def test_decoder_step_early_stopping():
    sd = _speller_data(["A", "B"])
    decoder = VEPCommandDecoder(stop_corr=0.5)
    out = decoder.step(np.array([0.2, 0.9]), sd)
    assert out["selected"] == "B"
    assert out["stop"] is True                                   # 0.9 >= 0.5

    decoder_high = VEPCommandDecoder(stop_corr=0.95)
    assert decoder_high.step(np.array([0.2, 0.9]), sd)["stop"] is False


def test_decoder_step_off_by_default():
    sd = _speller_data(["A", "B"])
    out = VEPCommandDecoder().step(np.array([0.2, 0.9]), sd)     # stop_corr default 0 -> off
    assert out["selected"] == "B"
    assert out["stop"] is False


def test_decoder_step_respects_availability():
    sd = _speller_data(["A", "B"], trial_available_cmmds=[["A"]])
    out = VEPCommandDecoder().step(np.array([0.1, 0.9]), sd, trial=0)
    assert out["selected"] == "A"                                # B unavailable
