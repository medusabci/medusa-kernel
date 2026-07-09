"""Qt-free ERP analysis backend for :class:`~medusa.widgets.erp_viewer.ERPViewer`.

:class:`ERPAnalyzer` owns the epoched signal ``(n_segments, n_samples,
n_channels)`` and turns it into the arrays the viewer renders, while **caching
the expensive/shared computations** so interaction stays responsive:

* the segment-averaged ERP ``(n_samples, n_channels)`` is computed once and
  reused by every topography frame (a slider move is then a single array index);
* per-channel *processed* segments (baseline + smoothing applied) are cached for
  the current selection so cosmetic redraws never recompute them.

No matplotlib and no Qt here: the class is a pure-NumPy model that the rendering
layer (:mod:`~medusa.widgets.erp_viewer.rendering`) and the Qt layer drive.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from medusa.plots.erp import _smooth_time

__all__ = ["ERPAnalyzer"]


class ERPAnalyzer:
    """Cached ERP computations over an epoched signal.

    Parameters
    ----------
    data
        Epoched signal ``(n_segments, n_samples, n_channels)``.
    fs
        Sampling rate (Hz). Optional if ``times`` is given.
    times
        Explicit per-sample time vector ``(n_samples,)`` or ``(t_start, t_end)``
        epoch limits. Derived from ``fs`` when ``None``; the sample index when
        both are ``None``.
    channel_set
        :class:`~medusa.core.data.channels.ChannelSet` for the spatial view. The
        column order must match ``data``'s channel axis. ``None`` disables the
        topography.
    cha_labels
        Per-channel names (defaults to ``channel_set.labels``, else ``Ch1..``).

    Attributes
    ----------
    data : numpy.ndarray
    times : numpy.ndarray
    n_segments, n_samples, n_channels : int
    """

    def __init__(self, data: ArrayLike, *, fs: float | None = None,
                 times: ArrayLike | None = None, channel_set=None,
                 cha_labels: ArrayLike | None = None) -> None:
        data = np.asarray(data, dtype=float)
        if data.ndim != 3:
            raise ValueError(
                "data must be a 3-D epoched signal (n_segments, n_samples, "
                f"n_channels); got shape {data.shape}.")
        self.data = data
        self.n_segments, self.n_samples, self.n_channels = data.shape
        self.times = self._resolve_times(fs, times)
        self.fs = float(fs) if fs is not None else self._infer_fs()
        self.channel_set = channel_set
        self.cha_labels = self._resolve_labels(cha_labels, channel_set)

        self._baseline: tuple[float, float] | None = None
        self._smooth: int | None = None
        self._erp_mean: NDArray | None = None            # (n_samples, n_channels)
        self._seg_cache: tuple[tuple, NDArray] | None = None

    # -- construction helpers --------------------------------------------
    def _resolve_times(self, fs, times) -> NDArray:
        if times is not None:
            t = np.asarray(times, dtype=float).ravel()
            if t.shape[0] == 2 and self.n_samples != 2:  # epoch limits
                return np.linspace(t[0], t[1], self.n_samples)
            if t.shape[0] != self.n_samples:
                raise ValueError(
                    f"times has length {t.shape[0]}; expected {self.n_samples} "
                    f"or 2 (epoch limits).")
            return t
        if fs is not None:
            return np.arange(self.n_samples) / float(fs)
        return np.arange(self.n_samples, dtype=float)

    def _infer_fs(self) -> float | None:
        if self.n_samples < 2:
            return None
        dt = float(np.median(np.diff(self.times)))
        return 1.0 / dt if dt > 0 else None

    def _resolve_labels(self, cha_labels, channel_set) -> list[str]:
        if cha_labels is not None:
            labels = [str(c) for c in cha_labels]
        elif channel_set is not None:
            labels = list(channel_set.labels)
        else:
            labels = [f"Ch{i + 1}" for i in range(self.n_channels)]
        if len(labels) != self.n_channels:
            raise ValueError(
                f"cha_labels has length {len(labels)}; expected "
                f"{self.n_channels} channels.")
        return labels

    # -- configuration (invalidates the caches) --------------------------
    @property
    def baseline(self) -> tuple[float, float] | None:
        return self._baseline

    def set_baseline(self, interval: tuple[float, float] | None) -> None:
        """Set (or clear) the ``(t_lo, t_hi)`` baseline window and invalidate caches."""
        self._baseline = (None if interval is None
                          else (float(interval[0]), float(interval[1])))
        self._invalidate()

    @property
    def smoothing(self) -> int | None:
        return self._smooth

    def set_smoothing(self, window: int | None) -> None:
        """Set (or clear) the moving-average window (samples) and invalidate caches."""
        self._smooth = int(window) if window and int(window) >= 2 else None
        self._invalidate()

    def _invalidate(self) -> None:
        self._erp_mean = None
        self._seg_cache = None

    # -- baseline mask ---------------------------------------------------
    def _baseline_mask(self) -> NDArray | None:
        if self._baseline is None:
            return None
        lo, hi = self._baseline
        mask = (self.times >= lo) & (self.times <= hi)
        if not mask.any():
            raise ValueError(
                f"baseline window {self._baseline} selects no samples in the "
                f"epoch [{self.times[0]:.4g}, {self.times[-1]:.4g}].")
        return mask

    # -- cached ERP mean (drives the topography) -------------------------
    @property
    def erp_mean(self) -> NDArray:
        """Segment-averaged ERP ``(n_samples, n_channels)`` (baseline + smoothing).

        Computed directly from the per-sample mean — baseline correction of the
        mean equals per-segment correction then averaging, and smoothing is
        linear — so this needs no per-segment intermediate and is cached until
        the baseline/smoothing changes.
        """
        if self._erp_mean is None:
            m = self.data.mean(axis=0)                    # (n_samples, n_channels)
            mask = self._baseline_mask()
            if mask is not None:
                m = m - m[mask].mean(axis=0, keepdims=True)
            if self._smooth:
                # _smooth_time filters axis 1; add a leading axis and drop it.
                m = _smooth_time(m[None], self._smooth)[0]
            self._erp_mean = m
        return self._erp_mean

    # -- processed segments (drive the temporal error bands) -------------
    def processed_segments(self, idxs: ArrayLike) -> NDArray:
        """Baseline+smoothed segments for channels ``idxs``: ``(n_seg, n_samp, n_sel)``.

        Cached for the current ``(idxs, baseline, smoothing)`` so cosmetic redraws
        (line width, colors, error type) never re-slice or re-filter the epochs.
        """
        idxs = np.asarray(idxs, dtype=int).ravel()
        key = (idxs.tobytes(), self._baseline, self._smooth)
        if self._seg_cache is not None and self._seg_cache[0] == key:
            return self._seg_cache[1]
        p = self.data[:, :, idxs].astype(float, copy=True)
        mask = self._baseline_mask()
        if mask is not None:
            p = p - p[:, mask, :].mean(axis=1, keepdims=True)
        if self._smooth:
            p = _smooth_time(p, self._smooth)
        self._seg_cache = (key, p)
        return p

    def channel_mean_segments(self, idxs: ArrayLike) -> NDArray:
        """Segments averaged across the selected channels: ``(n_seg, n_samp)``.

        Feeds *Mean* mode (error-band style): averaging across channels within each
        segment keeps the segment axis, so the band still reflects trial-to-trial
        variability while the mean equals "average over segments then over channels".
        """
        return self.processed_segments(idxs).mean(axis=2)

    # -- topography helpers ----------------------------------------------
    def topography_values(self, t_idx: int,
                          idxs: ArrayLike | None = None) -> NDArray:
        """ERP amplitude at sample ``t_idx`` for all (or the selected) channels."""
        row = self.erp_mean[int(t_idx)]
        return row if idxs is None else row[np.asarray(idxs, dtype=int)]

    def topo_channel_set(self, idxs: ArrayLike | None = None):
        """The channel set restricted to the selected channels (all when ``None``)."""
        if idxs is None or self.channel_set is None:
            return self.channel_set
        return self.channel_set.subset(np.asarray(idxs, dtype=int))

    def time_index(self, t: float) -> int:
        """Nearest sample index to time ``t``."""
        return int(np.argmin(np.abs(self.times - float(t))))

    def amplitude_range(self, idxs: ArrayLike | None = None) -> tuple[float, float]:
        """``(min, max)`` of the ERP mean over all (or selected) channels."""
        m = self.erp_mean if idxs is None else self.erp_mean[:, np.asarray(idxs, int)]
        return float(np.nanmin(m)), float(np.nanmax(m))

    def symmetric_clim(self, pad: float = 1.0) -> tuple[float, float]:
        """Symmetric ``(-v, v)`` color limits (``v = pad · max|ERP|``) for signed maps.

        Fixed across the whole epoch so topography colors are comparable frame to
        frame while the slider (or the GIF) sweeps time.
        """
        v = float(np.nanmax(np.abs(self.erp_mean))) * float(pad)
        v = v if v > 0 else 1.0
        return (-v, v)

    @property
    def located_mask(self) -> NDArray:
        """Boolean per-channel mask: which channels have a resolved 2-D/3-D position."""
        if self.channel_set is None:
            return np.zeros(self.n_channels, dtype=bool)
        pos = self.channel_set.get_positions(types=None)
        return np.isfinite(pos).all(axis=1)

    def any_located(self, idxs: ArrayLike | None = None) -> bool:
        """Whether any of the channels (all, or the selected ``idxs``) is located."""
        mask = self.located_mask
        if idxs is None:
            return bool(mask.any())
        idxs = np.asarray(idxs, dtype=int)
        return bool(mask[idxs].any()) if idxs.size else False

    @property
    def has_positions(self) -> bool:
        """Whether the channel set has at least one located sensor (spatial view)."""
        return self.any_located()
