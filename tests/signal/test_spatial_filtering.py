"""Tests for :mod:`medusa.signal.spatial_filtering`."""

from __future__ import annotations

import numpy as np
import pytest

from medusa.signal import spatial_filtering as sf


def test_car_zero_mean_per_sample(synthetic_signal_4ch):
    signal, _ = synthetic_signal_4ch
    out = sf.car(signal)
    assert out.shape == signal.shape
    # CAR removes the per-sample mean across channels.
    np.testing.assert_allclose(out.mean(axis=1), 0.0, atol=1e-12)


# --------------------------------------------------------------------------- #
# CSP: fit/project use the `signal`/`labels` argument names.
# --------------------------------------------------------------------------- #
def test_csp_fit_project_signal_labels_kwargs():
    rng = np.random.default_rng(0)
    signal = rng.standard_normal((20, 200, 4))
    labels = np.array([0, 1] * 10)
    csp = sf.CSP(n_filters=2)
    csp.fit(signal=signal, labels=labels)
    out = csp.project(signal=signal)
    assert out.shape == (20, 200, 2)


def test_csp_project_requires_3d_signal():
    rng = np.random.default_rng(0)
    csp = sf.CSP(n_filters=2)
    csp.fit(signal=rng.standard_normal((20, 200, 4)),
            labels=np.array([0, 1] * 10))
    with pytest.raises(ValueError):
        csp.project(signal=rng.standard_normal((200, 4)))


@pytest.mark.parametrize("n_filters", [5, 0, -1])
def test_csp_rejects_a_filter_count_it_cannot_honour(n_filters):
    """CSP builds one filter per channel: asking for more used to die deep inside the
    'extremes' selection loop with ``IndexError: pop from empty list``, and to silently
    return fewer filters with 'eigenvalues'."""
    rng = np.random.default_rng(0)
    csp = sf.CSP(n_filters=n_filters)
    with pytest.raises(ValueError, match="n_filters must be between 1"):
        csp.fit(signal=rng.standard_normal((20, 200, 4)),
                labels=np.array([0, 1] * 10))


def test_csp_accepts_one_filter_per_channel():
    rng = np.random.default_rng(0)
    csp = sf.CSP(n_filters=4)
    csp.fit(signal=rng.standard_normal((20, 200, 4)), labels=np.array([0, 1] * 10))
    assert csp.project(signal=rng.standard_normal((20, 200, 4))).shape == (20, 200, 4)


def test_csp_handles_more_than_two_classes():
    """The multiclass branch (joint diagonalization + mutual information ranking)."""
    rng = np.random.default_rng(0)
    signal, labels = rng.standard_normal((30, 200, 4)), np.array([0, 1, 2] * 10)
    csp = sf.CSP(n_filters=2, selection="eigenvalues")
    csp.fit(signal=signal, labels=labels)

    assert csp.project(signal=signal).shape == (30, 200, 2)
    assert csp.filters.shape == (4, 4) and csp.patterns.shape == (4, 4)


# --------------------------------------------------------------------------- #
# CCA: fit uses `signal`/`reference`; project/canoncorr use `signal`.
# --------------------------------------------------------------------------- #
def test_cca_fit_project_signal_reference_kwargs():
    rng = np.random.default_rng(1)
    signal = rng.standard_normal((500, 4))
    reference = rng.standard_normal((500, 4))
    cca = sf.CCA()
    cca.fit(signal=signal, reference=reference)
    out = cca.project(signal=signal, filter_idx=0, projection='wx')
    assert out.shape == (500,)


def test_cca_project_unknown_projection_raises():
    rng = np.random.default_rng(1)
    cca = sf.CCA()
    cca.fit(signal=rng.standard_normal((100, 3)),
            reference=rng.standard_normal((100, 3)))
    with pytest.raises(ValueError):
        cca.project(signal=rng.standard_normal((100, 3)), projection='bad')


def test_canoncorr_signal_reference():
    rng = np.random.default_rng(2)
    a, b, r = sf.CCA.canoncorr(rng.random((10, 4)), rng.random((10, 4)))
    assert a.shape == (4, 4)
    assert b.shape == (4, 4)
    assert r.shape == (4,)


# --------------------------------------------------------------------------- #
# TRCA: fit/project use the `signal` argument name.
# --------------------------------------------------------------------------- #
def test_trca_fit_project_signal_kwarg():
    rng = np.random.default_rng(3)
    segments = rng.standard_normal((10, 200, 4))
    trca = sf.TRCA()
    trca.fit(signal=segments)
    out = trca.project(signal=rng.standard_normal((200, 4)))
    assert out.shape == (200,)


# --------------------------------------------------------------------------- #
# LaplacianFilter: apply_lp uses the `signal` argument name.
# --------------------------------------------------------------------------- #
def test_laplacian_filter_apply_lp_signal():
    from medusa.core.legacy.biosignals.eeg.eeg import EEGChannelSet

    channel_set = EEGChannelSet()
    channel_set.set_standard_montage(
        ['Fz', 'Cz', 'Pz', 'C3', 'C4', 'P3', 'P4', 'Oz'], montage='10-05')
    lp = sf.LaplacianFilter(channel_set, mode='auto')
    lp.fit_lp(['CZ'])  # set_standard_montage upper-cases the labels
    signal = np.random.default_rng(0).standard_normal((100, 8))
    out = lp.apply_lp(signal=signal)
    assert out.shape == (100, 1)


def test_laplacian_filter_requires_five_channels():
    from medusa.core.legacy.biosignals.eeg.eeg import EEGChannelSet

    channel_set = EEGChannelSet()
    channel_set.set_standard_montage(['Fz', 'Cz', 'Pz'], montage='10-05')
    with pytest.raises(ValueError):
        sf.LaplacianFilter(channel_set, mode='auto')
