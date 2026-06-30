"""Tests for :mod:`medusa.core.utils` (K9.B)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from medusa.core.utils import check_data_dims


# ---------------------------------------------------------------------------
# Promotion table: each (rep_type, input_ndim) cell of the K9.B contract.
# ---------------------------------------------------------------------------

class TestRepTypeTime:
    """``rep_type='time'`` — canonical 2-D ``(n_samples, n_channels)``."""

    def test_1d_promotes_to_2d(self):
        with pytest.warns(UserWarning, match=r"promoted to \(1000, 1\)"):
            out, inserted = check_data_dims(np.zeros(1000), rep_type='time')
        assert out.shape == (1000, 1)
        assert inserted == (1,)

    def test_2d_unchanged_no_warning(self):
        arr = np.zeros((1000, 16))
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            out, inserted = check_data_dims(arr, rep_type='time')
        assert out.shape == (1000, 16)
        assert inserted == ()
        assert np.shares_memory(out, arr)

    def test_3d_raises(self):
        with pytest.raises(
            ValueError, match=r"not acceptable for rep_type='time'"
        ):
            check_data_dims(np.zeros((5, 1000, 16)), rep_type='time')

    def test_4d_raises(self):
        with pytest.raises(
            ValueError, match=r"not acceptable for rep_type='time'"
        ):
            check_data_dims(np.zeros((2, 5, 1000, 16)), rep_type='time')


class TestRepTypeTimeSegments:
    """``rep_type='time_segments'`` — canonical 3-D
    ``(n_segments, n_samples, n_channels)``."""

    def test_1d_promotes_to_3d(self):
        with pytest.warns(UserWarning, match=r"promoted to \(1, 1000, 1\)"):
            out, inserted = check_data_dims(
                np.zeros(1000), rep_type='time_segments'
            )
        assert out.shape == (1, 1000, 1)
        assert inserted == (0, 2)

    def test_2d_promotes_to_3d(self):
        with pytest.warns(UserWarning, match=r"promoted to \(1, 1000, 16\)"):
            out, inserted = check_data_dims(
                np.zeros((1000, 16)), rep_type='time_segments'
            )
        assert out.shape == (1, 1000, 16)
        assert inserted == (0,)

    def test_3d_unchanged_no_warning(self):
        arr = np.zeros((5, 1000, 16))
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            out, inserted = check_data_dims(arr, rep_type='time_segments')
        assert out.shape == (5, 1000, 16)
        assert inserted == ()
        assert np.shares_memory(out, arr)

    def test_4d_raises(self):
        with pytest.raises(
            ValueError, match=r"not acceptable for rep_type='time_segments'"
        ):
            check_data_dims(
                np.zeros((2, 5, 1000, 16)), rep_type='time_segments'
            )


class TestRepTypeFreq:
    """``rep_type='freq'`` — canonical 2-D ``(n_frequencies, n_channels)``."""

    def test_1d_promotes_to_2d(self):
        with pytest.warns(UserWarning, match=r"promoted to \(129, 1\)"):
            out, inserted = check_data_dims(np.zeros(129), rep_type='freq')
        assert out.shape == (129, 1)
        assert inserted == (1,)

    def test_2d_unchanged_no_warning(self):
        arr = np.zeros((129, 8))
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            out, inserted = check_data_dims(arr, rep_type='freq')
        assert out.shape == (129, 8)
        assert inserted == ()

    def test_3d_raises(self):
        with pytest.raises(
            ValueError, match=r"not acceptable for rep_type='freq'"
        ):
            check_data_dims(np.zeros((4, 129, 8)), rep_type='freq')


class TestRepTypeFreqSegments:
    """``rep_type='freq_segments'`` — canonical 3-D
    ``(n_segments, n_frequencies, n_channels)``."""

    def test_1d_promotes_to_3d(self):
        with pytest.warns(UserWarning, match=r"promoted to \(1, 129, 1\)"):
            out, inserted = check_data_dims(
                np.zeros(129), rep_type='freq_segments'
            )
        assert out.shape == (1, 129, 1)
        assert inserted == (0, 2)

    def test_2d_promotes_to_3d(self):
        with pytest.warns(UserWarning, match=r"promoted to \(1, 129, 8\)"):
            out, inserted = check_data_dims(
                np.zeros((129, 8)), rep_type='freq_segments'
            )
        assert out.shape == (1, 129, 8)
        assert inserted == (0,)

    def test_3d_unchanged_no_warning(self):
        arr = np.zeros((4, 129, 8))
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            out, inserted = check_data_dims(arr, rep_type='freq_segments')
        assert out.shape == (4, 129, 8)
        assert inserted == ()

    def test_4d_raises(self):
        with pytest.raises(
            ValueError, match=r"not acceptable for rep_type='freq_segments'"
        ):
            check_data_dims(
                np.zeros((2, 4, 129, 8)), rep_type='freq_segments'
            )


class TestRepTypeTimeFreq:
    """``rep_type='time_freq'`` — canonical 3-D
    ``(n_frequencies, n_times, n_channels)``."""

    def test_2d_promotes_to_3d(self):
        with pytest.warns(UserWarning, match=r"promoted to \(32, 100, 1\)"):
            out, inserted = check_data_dims(
                np.zeros((32, 100)), rep_type='time_freq'
            )
        assert out.shape == (32, 100, 1)
        assert inserted == (2,)

    def test_3d_unchanged_no_warning(self):
        arr = np.zeros((32, 100, 8))
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            out, inserted = check_data_dims(arr, rep_type='time_freq')
        assert out.shape == (32, 100, 8)
        assert inserted == ()

    def test_1d_raises(self):
        with pytest.raises(
            ValueError, match=r"not acceptable for rep_type='time_freq'"
        ):
            check_data_dims(np.zeros(100), rep_type='time_freq')

    def test_4d_raises(self):
        with pytest.raises(
            ValueError, match=r"not acceptable for rep_type='time_freq'"
        ):
            check_data_dims(np.zeros((4, 32, 100, 8)), rep_type='time_freq')


class TestRepTypeTimeFreqSegments:
    """``rep_type='time_freq_segments'`` — canonical 4-D
    ``(n_segments, n_frequencies, n_times, n_channels)``."""

    def test_2d_promotes_to_4d(self):
        with pytest.warns(
            UserWarning, match=r"promoted to \(1, 32, 100, 1\)"
        ):
            out, inserted = check_data_dims(
                np.zeros((32, 100)), rep_type='time_freq_segments'
            )
        assert out.shape == (1, 32, 100, 1)
        assert inserted == (0, 3)

    def test_3d_promotes_to_4d(self):
        with pytest.warns(
            UserWarning, match=r"promoted to \(1, 32, 100, 8\)"
        ):
            out, inserted = check_data_dims(
                np.zeros((32, 100, 8)), rep_type='time_freq_segments'
            )
        assert out.shape == (1, 32, 100, 8)
        assert inserted == (0,)

    def test_4d_unchanged_no_warning(self):
        arr = np.zeros((4, 32, 100, 8))
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            out, inserted = check_data_dims(
                arr, rep_type='time_freq_segments'
            )
        assert out.shape == (4, 32, 100, 8)
        assert inserted == ()

    def test_1d_raises(self):
        with pytest.raises(
            ValueError,
            match=r"not acceptable for rep_type='time_freq_segments'",
        ):
            check_data_dims(np.zeros(100), rep_type='time_freq_segments')


# ---------------------------------------------------------------------------
# rep_type validation: legacy v1 names are gone (no aliases).
# ---------------------------------------------------------------------------

class TestRepTypeValidation:
    @pytest.mark.parametrize(
        "legacy",
        [
            "time-series", "time-heatmap", "psd", "tfr",
            "TIME", "Time", "time-freq",
        ],
    )
    def test_legacy_or_unknown_rep_type_raises(self, legacy):
        with pytest.raises(ValueError, match=r"unknown rep_type"):
            check_data_dims(np.zeros((1000, 16)), rep_type=legacy)


# ---------------------------------------------------------------------------
# Round-trip via ``inserted``: caller's ndim is preserved end-to-end.
# ---------------------------------------------------------------------------

class TestInsertedRoundTrip:
    def test_squeeze_inserted_restores_1d(self):
        data = np.arange(1000.0)
        out, inserted = check_data_dims(data, rep_type='time')
        restored = np.squeeze(out, axis=inserted) if inserted else out
        assert restored.shape == data.shape
        np.testing.assert_array_equal(restored, data)

    def test_squeeze_inserted_restores_2d_under_time_segments(self):
        data = np.arange(2000.0).reshape(1000, 2)
        out, inserted = check_data_dims(data, rep_type='time_segments')
        restored = np.squeeze(out, axis=inserted) if inserted else out
        assert restored.shape == data.shape
        np.testing.assert_array_equal(restored, data)

    def test_canonical_input_inserted_is_empty(self):
        arr = np.zeros((3, 100, 4))
        out, inserted = check_data_dims(arr, rep_type='time_segments')
        assert inserted == ()
        assert np.shares_memory(out, arr)


# ---------------------------------------------------------------------------
# Promotion-warning details: stacklevel must point to the caller line.
# ---------------------------------------------------------------------------

class TestPromotionWarning:
    def test_warning_stacklevel_points_to_caller(self, recwarn):
        check_data_dims(np.zeros(100), rep_type='time')
        assert len(recwarn) == 1
        w = recwarn[0]
        assert __file__.endswith(w.filename) or w.filename.endswith(
            "test_utils.py"
        )

    def test_warning_message_includes_original_and_promoted_shapes(self):
        with pytest.warns(UserWarning) as captured:
            check_data_dims(np.zeros((128, 4)), rep_type='freq_segments')
        msgs = [str(w.message) for w in captured]
        assert any("(128, 4)" in m and "(1, 128, 4)" in m for m in msgs)


# ---------------------------------------------------------------------------
# Dtype preservation.
# ---------------------------------------------------------------------------

class TestDtypePreservation:
    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32])
    def test_dtype_unchanged_after_promotion(self, dtype):
        arr = np.zeros((1000, 4), dtype=dtype)
        out, _ = check_data_dims(arr, rep_type='time_segments')
        assert out.dtype == dtype
