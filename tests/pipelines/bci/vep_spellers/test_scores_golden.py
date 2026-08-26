"""Golden tests for the pure Layer-1 scoring functions.

These lock the *math* of the family accumulators before the ``_old`` reference code is
deleted. Where a value can be computed by hand it is pinned exactly; the cumulative
structure is also cross-checked against an independent textbook-Pearson oracle. A separate
test pins, and surfaces, the deliberate divergence between the new c-VEP template score
(``|Pearson|``) and the ``_old`` signed-uncentered-cosine it replaced.
"""
import numpy as np
import pytest
from scipy.stats import pearsonr

from medusa.pipelines.bci.vep_spellers import (
    bwr_command_scores, tm_command_scores, bwr_labels, CommandInfo)
from medusa.pipelines.bci.vep_spellers.decoding.template_matching import (
    _pearson_signed)


# --------------------------------------------------------------------------- #
# bwr_command_scores
# --------------------------------------------------------------------------- #
class TestBwrCommandScores:

    def test_single_cycle_exact(self):
        """One cycle, two commands: the matching command scores 1, the orthogonal one 0."""
        codes = np.array([[[1, 1, 0, 0]], [[1, 0, 1, 0]]])       # (2 cmd, 1 code, 4 frames)
        frame_scores = np.array([1.0, 1.0, 0.0, 0.0])            # matches command 0
        out = bwr_command_scores(frame_scores, codes,
                                 cycle_trial=np.array([0]), cycle_idx=np.array([0]),
                                 cycle_code_idx=np.array([0]))
        np.testing.assert_allclose(out, [[1.0, 0.0]], atol=1e-12)

    def test_sign_inverted_scores_one(self):
        """BWR ranks by ``|corr|``: a fully sign-inverted reconstruction still scores 1."""
        codes = np.array([[[1, 1, 0, 0]]])
        inverted = np.array([0.0, 0.0, 1.0, 1.0])
        out = bwr_command_scores(inverted, codes, np.array([0]), np.array([0]), np.array([0]))
        np.testing.assert_allclose(out, [[1.0]], atol=1e-12)

    def test_constant_code_is_neg_inf(self):
        """A command whose code is constant over the used cycles gets ``-inf`` (never chosen)."""
        codes = np.array([[[1, 1, 1, 1]]])                       # constant -> zero variance
        out = bwr_command_scores(np.array([1.0, 0.0, 1.0, 0.0]), codes,
                                 np.array([0]), np.array([0]), np.array([0]))
        assert out[0, 0] == -np.inf

    def test_constant_scores_is_neg_inf(self):
        codes = np.array([[[1, 0, 1, 0]]])
        out = bwr_command_scores(np.array([2.0, 2.0, 2.0, 2.0]), codes,
                                 np.array([0]), np.array([0]), np.array([0]))
        assert out[0, 0] == -np.inf

    def test_wrong_length_raises(self):
        codes = np.array([[[1, 0, 1, 0]]])
        with pytest.raises(ValueError):
            bwr_command_scores(np.array([1.0, 0.0]), codes,   # 2 != 1 cycle * 4 frames
                               np.array([0]), np.array([0]), np.array([0]))

    def test_cumulative_matches_pearson_oracle(self):
        """Two trials, multi-cycle: match an independent concatenate-then-|Pearson| oracle."""
        rng = np.random.default_rng(0)
        codes = rng.integers(0, 2, size=(3, 1, 5))               # 3 cmd, 1 code, 5 frames
        # 2 trials x 2 cycles, deliberately out of cycle order to exercise the sort
        cycle_trial = np.array([0, 1, 0, 1])
        cycle_idx = np.array([1, 0, 0, 1])
        cycle_code = np.array([0, 0, 0, 0])
        frame_scores = rng.standard_normal(4 * 5)

        out = bwr_command_scores(frame_scores, codes, cycle_trial, cycle_idx, cycle_code)
        expected = _bwr_oracle(frame_scores, codes, cycle_trial, cycle_idx, cycle_code)
        np.testing.assert_allclose(out, expected, atol=1e-10)


def _bwr_oracle(frame_scores, codes, cycle_trial, cycle_idx, cycle_code):
    """Independent reference: per trial, cumulative concatenate-then-|Pearson| in cycle order."""
    n_cmd, _, n_frames = codes.shape
    n_cycles = len(cycle_idx)
    by_cycle = np.asarray(frame_scores, float).reshape(n_cycles, n_frames)
    out = np.full((n_cycles, n_cmd), -np.inf)
    for t in np.unique(cycle_trial):
        rows = np.where(cycle_trial == t)[0]
        order = rows[np.argsort(cycle_idx[rows], kind="stable")]
        for i in range(len(order)):
            used = order[:i + 1]
            s = by_cycle[used].ravel()
            for k in range(n_cmd):
                exp = codes[k, cycle_code[used]].ravel()
                if exp.min() == exp.max() or s.min() == s.max():
                    out[order[i], k] = -np.inf
                else:
                    out[order[i], k] = abs(pearsonr(exp, s)[0])
    return out


# --------------------------------------------------------------------------- #
# tm_command_scores
# --------------------------------------------------------------------------- #
class TestTmCommandScores:

    def test_coherent_average_is_cumulative(self):
        """Row i scores the mean of the used cycles' segments (coherent averaging)."""
        segments = np.array([[[5.0]], [[1.0]]])                  # (2 cycles, 1 sample, 1 ch)
        out = tm_command_scores(segments, lambda avg: np.array([avg[0, 0]]),
                                cycle_trial=np.array([0, 0]), cycle_idx=np.array([0, 1]))
        # cycle 0 alone -> 5 ; cycles 0+1 averaged -> 3
        np.testing.assert_allclose(out, [[5.0], [3.0]])

    def test_score_fn_maps_to_commands(self):
        segments = np.ones((2, 5, 3))
        out = tm_command_scores(segments, lambda avg: avg.mean(axis=(0, 1)) * np.array([1.0, 2.0]),
                                cycle_trial=np.array([0, 0]), cycle_idx=np.array([0, 1]))
        np.testing.assert_allclose(out, [[1.0, 2.0], [1.0, 2.0]])
        assert int(out[-1].argmax()) == 1

    def test_respects_cycle_order(self):
        """Cycles given out of order are averaged in cycle_idx order, not array order."""
        segments = np.array([[[1.0]], [[5.0]]])                  # array order: 1 then 5
        out = tm_command_scores(segments, lambda avg: np.array([avg[0, 0]]),
                                cycle_trial=np.array([0, 0]),
                                cycle_idx=np.array([1, 0]))       # true order: seg[1] then seg[0]
        np.testing.assert_allclose(out[np.array([1, 0])], [[5.0], [3.0]])


# --------------------------------------------------------------------------- #
# bwr_labels
# --------------------------------------------------------------------------- #
def test_bwr_labels_concatenates_target_code(cvep_recording_factory):
    """Per-frame labels are the target command's code repeated over the trial's cycles."""
    cmds = {"0": CommandInfo(uid="0", code=[[1, 0, 1, 1]]),
            "1": CommandInfo(uid="1", code=[[0, 1, 0, 0]])}
    rec = cvep_recording_factory(cmds, ["0"], n_cycles=2, resp_amp=0.0, seed=0)
    np.testing.assert_array_equal(bwr_labels(rec), [1, 0, 1, 1, 1, 0, 1, 1])


def test_bwr_labels_needs_spell_target(cvep_recording_factory):
    cmds = {"0": CommandInfo(uid="0", code=[[1, 0]]),
            "1": CommandInfo(uid="1", code=[[0, 1]])}
    rec = cvep_recording_factory(cmds, ["0"], n_cycles=1, resp_amp=0.0, seed=0)
    rec.experiment.spell_target = None
    with pytest.raises(ValueError):
        bwr_labels(rec)


# --------------------------------------------------------------------------- #
# c-VEP template scoring: new |Pearson| vs the _old signed-uncentered-cosine
# --------------------------------------------------------------------------- #
def _old_cosine(a, b):
    """The ``_old`` c-VEP template score: dot(a,b)/sqrt(dot(a,a)*dot(b,b)); signed, uncentered."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b)))


class TestCvepTemplateDivergence:
    """Pin the new score and document how it differs from what ``_old`` computed.

    The new score is a **centred, signed** correlation. It keeps ``_old``'s sign (an
    anti-correlated template must rank last, not first) and adds the mean subtraction that
    ``_old``'s uncentered cosine lacked, so the two still disagree on the numbers.
    """

    def test_new_is_signed_pearson(self):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.array([4.0, 3.0, 2.0, 1.0])
        assert _pearson_signed(a, b) == pytest.approx(pearsonr(a, b)[0])          # == -1.0
        assert _pearson_signed(a, b) < 0

    @pytest.mark.parametrize("n_samples", [4, 64, 134, 250])
    def test_agrees_with_scipy_on_random_input(self, n_samples):
        """scipy is the oracle; the hand-rolled version exists only to be fast.

        ``_pearson_signed`` is two dot products instead of ``scipy.stats.pearsonr`` because it
        runs thousands of times per decode (about a third of the wall clock) and scipy also
        computes a p-value we never use. This pins that the shortcut costs no accuracy.
        """
        rng = np.random.default_rng(n_samples)
        for _ in range(50):
            a = rng.standard_normal(n_samples)
            b = rng.standard_normal(n_samples)
            assert _pearson_signed(a, b) == pytest.approx(pearsonr(a, b)[0], abs=1e-12)
            # scaling and shifting either input must not move a correlation
            assert _pearson_signed(3.0 * a + 7.0, b) == pytest.approx(
                pearsonr(a, b)[0], abs=1e-12)

    def test_constant_input_abstains_instead_of_returning_nan(self):
        """Where scipy/numpy give nan (and warn), we return 0: that view simply abstains."""
        constant = np.ones(64)
        rng = np.random.default_rng(0)
        assert _pearson_signed(constant, rng.standard_normal(64)) == 0.0
        assert _pearson_signed(constant, constant) == 0.0

    def test_diverges_from_old_uncentered_cosine(self):
        """Same inputs, different numbers: the mean-subtraction changes the score."""
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.array([4.0, 3.0, 2.0, 1.0])
        assert _pearson_signed(a, b) == pytest.approx(-1.0)                       # Pearson = -1
        assert _old_cosine(a, b) == pytest.approx(20.0 / 30.0)            # cosine = +0.667
        assert not np.isclose(_pearson_signed(a, b), _old_cosine(a, b))

    def test_anti_correlated_template_ranks_last(self):
        """An inverted template is evidence AGAINST the command, so it must score below zero.

        This is the whole reason the sign is kept: with ``|Pearson|`` the segment below scored
        1.0 and outranked every honest match, which is how a wrong c-VEP lag could win.
        """
        template = np.array([1.0, -1.0, 1.0, -1.0])
        segment = np.array([-1.0, 1.0, -1.0, 1.0])
        assert _pearson_signed(template, segment) == pytest.approx(-1.0)
        assert _old_cosine(template, segment) == pytest.approx(-1.0)      # _old agreed
        # an unrelated segment carries no evidence, and must still beat an inverted one
        unrelated = np.array([1.0, 1.0, -1.0, -1.0])
        assert _pearson_signed(template, unrelated) > _pearson_signed(template, segment)

    def test_sign_survives_the_spatial_filter_polarity(self):
        """``w`` has an arbitrary sign, but it is applied to both sides, so the score is stable."""
        rng = np.random.default_rng(0)
        segment = rng.standard_normal((64, 4))
        template = rng.standard_normal((64, 4))
        w = rng.standard_normal(4)
        assert _pearson_signed(segment @ w, template @ w) ==             pytest.approx(_pearson_signed(segment @ -w, template @ -w))
