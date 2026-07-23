"""Codebook generator tests, with a focus on the random-code c-VEP generator."""
import numpy as np
import pytest

from medusa.pipelines.bci.vep_spellers import (
    generate_random_codebook, generate_mseq_codebook, CommandInfo)


def _codes(commands_info):
    """The ``(n_commands, n_frames)`` first-code matrix of a ``{uid: CommandInfo}`` dict."""
    return np.array([np.asarray(c.code)[0] for c in commands_info.values()])


class TestRandomCodebook:
    """`generate_random_codebook` — the net-new random ("noise") c-VEP generator."""

    def test_shape_and_uids(self):
        cmds = generate_random_codebook(6, n_frames=126, seed=0)
        assert list(cmds) == [str(i) for i in range(6)]
        assert all(isinstance(c, CommandInfo) for c in cmds.values())
        assert _codes(cmds).shape == (6, 126)
        # code is stored as the canonical 2-D (n_codes, n_frames)
        assert np.asarray(cmds["0"].code).shape == (1, 126)

    def test_binary_and_non_constant(self):
        codes = _codes(generate_random_codebook(6, n_frames=126, seed=0))
        assert set(np.unique(codes).tolist()) <= {0, 1}
        # every code varies (a constant code has zero variance and is undecodable)
        assert np.all(codes.min(axis=1) != codes.max(axis=1))

    def test_codes_are_distinct(self):
        codes = _codes(generate_random_codebook(20, n_frames=126, seed=3))
        assert len({row.tobytes() for row in codes}) == 20

    def test_reproducible_with_seed(self):
        a = _codes(generate_random_codebook(6, n_frames=126, seed=42))
        b = _codes(generate_random_codebook(6, n_frames=126, seed=42))
        c = _codes(generate_random_codebook(6, n_frames=126, seed=43))
        assert np.array_equal(a, b)
        assert not np.array_equal(a, c)

    def test_accepts_generator_as_seed(self):
        rng = np.random.default_rng(0)
        codes = _codes(generate_random_codebook(4, n_frames=63, seed=rng))
        assert codes.shape == (4, 63)

    def test_base_n_levels(self):
        codes = _codes(generate_random_codebook(4, n_frames=80, base=4, seed=1))
        assert set(np.unique(codes).tolist()) <= {0, 1, 2, 3}
        assert np.unique(codes).max() >= 2          # actually multi-level

    def test_extra_carries_code_index(self):
        cmds = generate_random_codebook(3, n_frames=63, seed=0)
        assert [c.extra["code_index"] for c in cmds.values()] == [0, 1, 2]

    @pytest.mark.parametrize("kwargs", [
        {"n_commands": 0, "n_frames": 63},
        {"n_commands": 3, "n_frames": 0},
        {"n_commands": 3, "n_frames": 63, "base": 1},
        {"n_commands": 3, "n_frames": 63, "p": 0.0},
        {"n_commands": 3, "n_frames": 63, "p": 1.5},
        {"n_commands": 1000, "n_frames": 3},        # cannot draw enough distinct codes
    ])
    def test_invalid_arguments_raise(self, kwargs):
        with pytest.raises(ValueError):
            generate_random_codebook(**kwargs)

    def test_low_cross_correlation_vs_short_codes(self):
        """Longer random codes have lower peak mutual cross-correlation (~1/sqrt(L))."""
        def peak_xcorr(codes):
            x = codes - codes.mean(axis=1, keepdims=True)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            g = np.abs(x @ x.T)
            np.fill_diagonal(g, 0.0)
            return g.max()
        short = peak_xcorr(_codes(generate_random_codebook(6, n_frames=32, seed=0)).astype(float))
        long = peak_xcorr(_codes(generate_random_codebook(6, n_frames=252, seed=0)).astype(float))
        assert long < short


def test_random_and_mseq_have_same_container_shape():
    """A random codebook drops into the same SpellerData machinery as an m-sequence one."""
    rnd = generate_random_codebook(6, n_frames=63, seed=0)
    mseq = generate_mseq_codebook(6, polynomial=[0, 0, 0, 0, 1, 1])
    assert _codes(rnd).shape[0] == _codes(mseq).shape[0]
    assert np.asarray(rnd["0"].code).ndim == np.asarray(mseq["0"].code).ndim == 2
