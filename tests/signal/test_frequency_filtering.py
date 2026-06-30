"""Tests for :mod:`medusa.signal.frequency_filtering`.

Frequency filters are restricted to the canonical 2-D ``time``
representation ``(n_samples, n_channels)``. 1-D ``(n_samples,)``
inputs are promoted internally and the output is squeezed back to the
caller's ``ndim``. 3-D pre-segmented inputs are rejected; callers are
expected to iterate over the segment axis themselves.
"""

from __future__ import annotations

import numpy as np
import pytest

from medusa.signal import frequency_filtering as ff


# ---------------------------------------------------------------------------
# 2-D canonical contract.
# ---------------------------------------------------------------------------


def test_iir_bandpass_filter_reduces_variance(synthetic_signal_4ch):
    signal, fs = synthetic_signal_4ch
    filt = ff.IIRFilter(
        order=4, cutoff=(8.0, 12.0), btype="bandpass",
        filt_method="sosfiltfilt",
    )
    out = filt.fit_transform(signal.copy(), fs)
    assert out.shape == signal.shape
    assert np.isfinite(out).all()
    # Narrow band-pass on a noisy broadband signal must drop variance.
    assert out.var() < signal.var()


def test_iir_filter_fit_then_transform_consistency(synthetic_signal_4ch):
    signal, fs = synthetic_signal_4ch
    filt = ff.IIRFilter(
        order=4, cutoff=1.0, btype="highpass",
        filt_method="sosfiltfilt",
    )
    filt.fit(fs)
    out = filt.transform(signal.copy())
    assert out.shape == signal.shape
    assert np.isfinite(out).all()


def test_fir_lowpass_filter_runs(synthetic_signal_4ch):
    signal, fs = synthetic_signal_4ch
    filt = ff.FIRFilter(
        order=64, cutoff=20.0, btype="lowpass", width=None,
    )
    out = filt.fit_transform(signal.copy(), fs)
    assert out.shape == signal.shape
    assert np.isfinite(out).all()


# ---------------------------------------------------------------------------
# 1-D inputs: promoted to canonical (n_samples, 1) and squeezed back.
# ---------------------------------------------------------------------------


def test_iir_preserves_1d_input_shape():
    fs = 250.0
    signal = np.random.default_rng(3).standard_normal(500)
    filt = ff.IIRFilter(order=4, cutoff=20.0, btype="lowpass")
    out = filt.fit_transform(signal, fs)
    assert out.shape == signal.shape
    assert np.isfinite(out).all()


def test_fir_preserves_1d_input_shape():
    fs = 250.0
    signal = np.random.default_rng(4).standard_normal(500)
    filt = ff.FIRFilter(order=65, cutoff=20.0, btype="lowpass")
    out = filt.fit_transform(signal, fs)
    assert out.shape == signal.shape
    assert np.isfinite(out).all()


# ---------------------------------------------------------------------------
# 3-D pre-segmented inputs are rejected — only the 2-D 'time' rep is valid.
# ---------------------------------------------------------------------------


def test_iir_rejects_3d_signals():
    rng = np.random.default_rng(0)
    signal = rng.standard_normal((2, 500, 4))
    filt = ff.IIRFilter(order=4, cutoff=(8.0, 12.0), btype="bandpass")
    with pytest.raises(ValueError):
        filt.fit_transform(signal, fs=250.0)


def test_fir_rejects_3d_signals():
    rng = np.random.default_rng(2)
    signal = rng.standard_normal((2, 1024, 3))
    filt = ff.FIRFilter(order=65, cutoff=20.0, btype="lowpass")
    with pytest.raises(ValueError):
        filt.fit_transform(signal, fs=250.0)


# ---------------------------------------------------------------------------
# Streaming filter: IIRFilter(filt_method='sosfilt') needs n_channels.
# ---------------------------------------------------------------------------


def test_iir_streaming_requires_n_channels():
    filt = ff.IIRFilter(
        order=4, cutoff=20.0, btype="lowpass", filt_method="sosfilt",
    )
    with pytest.raises(ValueError, match="n_channels"):
        filt.fit(fs=250.0)


def test_iir_streaming_fit_with_n_channels(synthetic_signal_4ch):
    signal, fs = synthetic_signal_4ch
    filt = ff.IIRFilter(
        order=4, cutoff=20.0, btype="lowpass", filt_method="sosfilt",
    )
    filt.fit(fs, n_channels=signal.shape[1])
    out = filt.transform(signal.copy())
    assert out.shape == signal.shape
    assert np.isfinite(out).all()


def test_iir_streaming_fit_transform_infers_n_channels(synthetic_signal_4ch):
    signal, fs = synthetic_signal_4ch
    filt = ff.IIRFilter(
        order=4, cutoff=20.0, btype="lowpass", filt_method="sosfilt",
    )
    out = filt.fit_transform(signal.copy(), fs)
    assert out.shape == signal.shape
    assert np.isfinite(out).all()


# ---------------------------------------------------------------------------
# Filter-method validation: unsupported strings raise.
# ---------------------------------------------------------------------------


def test_iir_unsupported_filt_method_raises(synthetic_signal_4ch):
    signal, fs = synthetic_signal_4ch
    filt = ff.IIRFilter(
        order=4, cutoff=20.0, btype="lowpass",
        filt_method="bogus",  # type: ignore[arg-type]
    )
    filt.sos = np.zeros((1, 6))  # bypass fit() so transform() reaches dispatch
    with pytest.raises(ValueError, match="Unsupported IIR filter method"):
        filt.transform(signal)


def test_fir_unsupported_filt_method_raises(synthetic_signal_4ch):
    signal, fs = synthetic_signal_4ch
    filt = ff.FIRFilter(
        order=64, cutoff=20.0, btype="lowpass",
        filt_method="bogus",  # type: ignore[arg-type]
    )
    filt.b, filt.a = np.zeros(65), [1.0]
    with pytest.raises(ValueError, match="Unsupported FIR filter method"):
        filt.transform(signal)

