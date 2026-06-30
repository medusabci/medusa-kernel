"""Tests for :mod:`medusa.signal.metrics.nonlinear.sample_entropy`.

Uses small signals so the O(N²) pdist stays cheap on CI runners.
"""

from __future__ import annotations

import numpy as np
import pytest

from medusa.signal.metrics.nonlinear.sample_entropy import sample_entropy


def test_sample_entropy_white_noise_greater_than_sinusoid():
    """White noise must be more irregular than a pure sinusoid."""
    rng = np.random.default_rng(42)
    fs = 128
    n = 600
    t = np.arange(n) / fs
    sinus = np.sin(2 * np.pi * 5 * t)[None, :, None]      # [1, n, 1]
    noise = rng.standard_normal((1, n, 1))
    sampen_sin = sample_entropy(sinus, m=2, r=0.2)
    sampen_noi = sample_entropy(noise, m=2, r=0.2)
    assert sampen_sin.shape == (1, 1)
    assert sampen_noi.shape == (1, 1)
    assert sampen_noi[0, 0] > sampen_sin[0, 0]


def test_sample_entropy_invalid_dist_type_raises():
    rng = np.random.default_rng(0)
    sig = rng.standard_normal((1, 200, 1))
    with pytest.raises(ValueError):
        sample_entropy(sig, m=2, r=0.2, dist_type="not-a-metric")


def test_sample_entropy_m_larger_than_signal_raises():
    rng = np.random.default_rng(0)
    sig = rng.standard_normal((1, 50, 1))
    with pytest.raises(ValueError):
        sample_entropy(sig, m=100, r=0.2)

