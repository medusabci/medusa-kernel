"""Shared fixtures for the ``signal`` test suite."""
import numpy as np
import pytest


@pytest.fixture
def synthetic_signal_4ch():
    """A 4-channel broadband-noise signal and its sampling rate.

    Returns ``(signal, fs)`` where ``signal`` has the canonical ``time`` shape
    ``(n_samples, n_channels)`` with ``n_channels == 4``. The samples are
    broadband Gaussian noise so that a narrow band-pass measurably reduces the
    variance (see ``test_iir_bandpass_filter_reduces_variance``).
    """
    fs = 250.0
    signal = np.random.default_rng(42).standard_normal((1000, 4))
    return signal, fs