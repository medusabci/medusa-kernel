"""Frequency-domain filtering primitives (FIR / IIR)."""

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy import signal as scipy_signal

from medusa.core.utils import check_data_dims

BAND_TYPES = Literal['lowpass', 'highpass', 'bandpass', 'bandstop']
FIRMethods = Literal['filtfilt', 'lfilter']
IIRMethods = Literal['sosfiltfilt', 'sosfilt']

_Cutoff = float | tuple[float, float] | list[float] | NDArray

__all__ = [
    "FIRFilter",
    "IIRFilter",
    "display_filter",
    "BAND_TYPES",
    "FIRMethods",
    "IIRMethods"
]


class FIRFilter:
    """FIR filter wrapper around :func:`scipy.signal.firwin`.

    Accepts the canonical time representation
    ``(n_samples, n_channels)``. 1-D inputs ``(n_samples,)`` are
    promoted internally via :func:`medusa.core.utils.check_data_dims`
    and the output is squeezed back to the caller's ``ndim`` — input
    shape is preserved end-to-end.

    To filter pre-segmented data
    ``(n_segments, n_samples, n_channels)``, iterate over the segment
    axis at the call site; this module only operates on the continuous
    time representation.

    Parameters
    ----------
    order :
        Length of the filter (number of taps). Must be odd if a passband
        includes the Nyquist frequency.
    cutoff :
        Cutoff frequency (Hz) or array of cutoff frequencies (band edges).
        Cutoff frequencies must be strictly between 0 and ``fs / 2``.
    btype :
        ``'lowpass'``, ``'highpass'``, ``'bandpass'`` or ``'bandstop'``.
        Controls the ``pass_zero`` argument of
        :func:`scipy.signal.firwin`.
    width :
        Approximate width of the transition region (Hz) for Kaiser FIR
        design. When given, ``window`` is ignored.
    window :
        Window function used by :func:`scipy.signal.firwin`. Default
        ``'hamming'``.
    scale :
        If True, scale the coefficients so the response is exactly unity
        at a characteristic frequency (DC, Nyquist or the centre of the
        first passband — see :func:`scipy.signal.firwin`).
    filt_method :
        ``'filtfilt'`` (zero-phase, default) or ``'lfilter'`` (causal).

    Examples
    --------
    >>> import numpy as np
    >>> from medusa.signal.frequency_filtering import FIRFilter
    >>> sig = np.random.default_rng(0).standard_normal((1024, 4))
    >>> filt = FIRFilter(order=65, cutoff=20.0, btype='lowpass')
    >>> out = filt.fit_transform(sig, fs=250.0)
    >>> out.shape
    (1024, 4)
    """

    def __init__(
        self,
        order: int,
        cutoff: _Cutoff,
        btype: BAND_TYPES,
        width: float | None = None,
        window: str = 'hamming',
        scale: bool = True,
        filt_method: FIRMethods = 'filtfilt',
    ) -> None:
        self.btype: BAND_TYPES = btype
        self.order = order
        self.cutoff = cutoff
        self.width = width
        self.window = window
        self.scale = scale
        self.filt_method: FIRMethods = filt_method

        # Filled by fit().
        self.fs: float | None = None
        self.b: NDArray | None = None
        self.a: list[float] | None = None

    def display(self) -> None:
        """Plot the magnitude and phase response of the fitted filter."""
        display_filter(self.b, self.a, self.fs)

    def fit(self, fs: float) -> None:
        """Design the FIR coefficients for the given sampling frequency.

        Parameters
        ----------
        fs :
            Sampling frequency in Hz. Each frequency in ``cutoff`` must
            satisfy ``0 < cutoff < fs / 2``.
        """
        self.fs = fs
        self.b = scipy_signal.firwin(
            numtaps=self.order,
            cutoff=self.cutoff,
            width=self.width,
            window=self.window,
            pass_zero=self.btype,
            scale=self.scale,
            fs=self.fs,
        )
        self.a = [1.0]

    def transform(
        self,
        signal: NDArray
    ) -> NDArray:
        """Apply the fitted filter to ``signal`` along the sample axis.

        Parameters
        ----------
        signal :
            Input array. Shape must be ``(n_samples,)`` or
            ``(n_samples, n_channels)``. 1-D inputs are promoted
            internally to the canonical 2-D ``time`` shape by
            :func:`medusa.core.utils.check_data_dims`.

        Returns
        -------
        NDArray
            Filtered signal with the same shape as ``signal``: the
            length-1 channel axis inserted by promotion of a 1-D input
            is squeezed out before returning.
        """
        canonical, inserted = check_data_dims(signal, rep_type='time')
        if self.filt_method == 'filtfilt':
            out = scipy_signal.filtfilt(self.b, self.a, canonical, axis=0)
        elif self.filt_method == 'lfilter':
            out = scipy_signal.lfilter(self.b, self.a, canonical, axis=0)
        else:
            raise ValueError(
                f"Unsupported FIR filter method {self.filt_method!r}. "
                f"Choose 'filtfilt' or 'lfilter'."
            )
        return np.squeeze(out, axis=inserted) if inserted else out

    def fit_transform(
        self,
        signal: NDArray,
        fs: float
    ) -> NDArray:
        """Design the FIR filter at ``fs`` and apply it to ``signal``.

        Parameters
        ----------
        signal :
            Input signal; see :meth:`transform` for the shape contract.
        fs :
            Sampling frequency in Hz.

        Returns
        -------
        NDArray
            Filtered signal with the same shape as ``signal``.
        """
        self.fit(fs)
        return self.transform(signal)


class IIRFilter:
    """IIR Butterworth filter wrapper around :func:`scipy.signal.butter`.

    Accepts the canonical time representation
    ``(n_samples, n_channels)``. 1-D inputs ``(n_samples,)`` are
    promoted internally via :func:`medusa.core.utils.check_data_dims`
    and the output is squeezed back to the caller's ``ndim``.

    To filter pre-segmented data
    ``(n_segments, n_samples, n_channels)``, iterate over the segment
    axis at the call site; this module only operates on the continuous
    time representation.

    The streaming path (``filt_method='sosfilt'``) keeps a per-channel
    state in :attr:`zi` between successive :meth:`transform` calls.
    ``n_channels`` must therefore be passed to :meth:`fit` (or
    inferred via :meth:`fit_transform`) so the initial state can be
    allocated.

    Parameters
    ----------
    order :
        Filter order. The actual number of coefficients depends on the
        band type.
    cutoff :
        Cutoff frequency (Hz) for low/highpass, or ``(f_low, f_high)`` for
        band/bandstop. Must be strictly between 0 and ``fs / 2``.
    btype :
        ``'lowpass'``, ``'highpass'``, ``'bandpass'`` or ``'bandstop'``.
    filt_method :
        ``'sosfiltfilt'`` (zero-phase, offline; default) or ``'sosfilt'``
        (causal, suitable for streaming). See class docstring for the
        streaming-state contract.

    Examples
    --------
    >>> import numpy as np
    >>> from medusa.signal.frequency_filtering import IIRFilter
    >>> sig = np.random.default_rng(0).standard_normal((1024, 4))
    >>> filt = IIRFilter(order=4, cutoff=(8.0, 12.0), btype='bandpass')
    >>> out = filt.fit_transform(sig, fs=250.0)
    >>> out.shape
    (1024, 4)
    """

    def __init__(
        self,
        order: int,
        cutoff: _Cutoff,
        btype: BAND_TYPES,
        filt_method: IIRMethods = 'sosfiltfilt',
    ) -> None:
        self.btype: BAND_TYPES = btype
        self.order = order
        self.cutoff = cutoff
        self.filt_method: IIRMethods = filt_method

        # Filled by fit().
        self.fs: float | None = None
        self.sos: NDArray | None = None
        self.zi: NDArray | None = None

    def display(self) -> None:
        """Plot the magnitude and phase response of the fitted filter."""
        b, a = scipy_signal.sos2tf(self.sos)
        display_filter(b, a, self.fs)

    def fit(
        self,
        fs: float,
        n_channels: int | None = None
    ) -> None:
        """Design the SOS coefficients for the given sampling frequency.

        Parameters
        ----------
        fs :
            Sampling frequency in Hz. Each frequency in ``cutoff`` must
            satisfy ``0 < cutoff < fs / 2``.
        n_channels :
            Number of channels in the input signal. Required only for
            streaming filtering (``filt_method='sosfilt'``) so the
            initial state can be allocated; ignored when
            ``filt_method='sosfiltfilt'``.

        Raises
        ------
        ValueError
            If ``filt_method='sosfilt'`` and ``n_channels`` is not given.
        """
        self.fs = fs
        self.sos = scipy_signal.butter(
            N=self.order,
            Wn=self.cutoff,
            btype=self.btype,
            analog=False,
            output='sos',
            fs=self.fs,
        )
        if self.filt_method == 'sosfilt':
            if n_channels is None:
                raise ValueError(
                    "n_channels is required when filt_method='sosfilt' "
                    "to allocate the streaming filter's initial state."
                )
            zi = scipy_signal.sosfilt_zi(self.sos)  # (n_sections, 2)
            # Store zi as (n_sections, 2, n_channels): no segments
            # axis in the streaming 'time' representation.
            self.zi = np.broadcast_to(
                zi[:, :, None],
                (zi.shape[0], 2, n_channels),
            ).copy()

    def transform(
        self,
        signal: NDArray
    ) -> NDArray:
        """Apply the fitted filter to ``signal`` along the sample axis.

        Parameters
        ----------
        signal :
            Input array. Shape must be ``(n_samples,)`` or
            ``(n_samples, n_channels)``. 1-D inputs are promoted
            internally to the canonical 2-D ``time`` shape by
            :func:`medusa.core.utils.check_data_dims`. For the
            streaming path (``filt_method='sosfilt'``) the per-channel
            ``zi`` state allocated in :meth:`fit` keeps its alignment
            across successive calls.

        Returns
        -------
        NDArray
            Filtered signal with the same shape as ``signal``: the
            length-1 channel axis inserted by promotion of a 1-D input
            is squeezed out before returning.
        """
        canonical, inserted = check_data_dims(signal, rep_type='time')
        if self.filt_method == 'sosfiltfilt':
            out = scipy_signal.sosfiltfilt(self.sos, canonical, axis=0)
        elif self.filt_method == 'sosfilt':
            out, zo = scipy_signal.sosfilt(
                self.sos, canonical, axis=0, zi=self.zi,
            )
            self.zi = zo
        else:
            raise ValueError(
                f"Unsupported IIR filter method {self.filt_method!r}. "
                f"Choose 'sosfiltfilt' or 'sosfilt'."
            )
        return np.squeeze(out, axis=inserted) if inserted else out

    def fit_transform(
        self,
        signal: NDArray,
        fs: float,
    ) -> NDArray:
        """Design the IIR filter at ``fs`` and apply it to ``signal``.

        For the streaming path (``filt_method='sosfilt'``) the channel
        count is inferred from ``signal`` so the per-channel initial
        state can be allocated transparently.

        Parameters
        ----------
        signal :
            Input signal; see :meth:`transform` for the shape contract.
        fs :
            Sampling frequency in Hz.

        Returns
        -------
        NDArray
            Filtered signal with the same shape as ``signal``.
        """
        if self.filt_method == 'sosfilt':
            # Peek at the channel count under the canonical 'time' rep
            # so fit() can allocate the initial state.
            canonical, _ = check_data_dims(signal, rep_type='time')
            self.fit(fs, n_channels=canonical.shape[-1])
        else:
            self.fit(fs)
        return self.transform(signal)


def display_filter(b: NDArray, a: NDArray, fs: float) -> None:
    """Plot the magnitude (dB) and phase (deg) response of a filter.

    Parameters
    ----------
    b :
        Numerator coefficients of the filter.
    a :
        Denominator coefficients of the filter.
    fs :
        Sampling frequency in Hz.

    Examples
    --------
    >>> from scipy.signal import butter
    >>> from medusa.signal.frequency_filtering import display_filter
    >>> b, a = butter(4, 0.1, btype='lowpass')
    >>> display_filter(b, a, fs=250.0)  # doctest: +SKIP
    """
    w, h = scipy_signal.freqz(b, a)
    freq = w * fs / (2 * np.pi)

    fig, ax = plt.subplots(2, 1, figsize=(8, 6))

    ax[0].plot(freq, 20 * np.log10(abs(h)), color='blue')
    ax[0].set_title("Frequency Response")
    ax[0].set_ylabel("Amplitude (dB)", color='blue')
    ax[0].set_xlim([0, fs / 2])
    ax[0].set_ylim([-50, 1])
    ax[0].grid()

    ax[1].plot(freq, np.unwrap(np.angle(h)) * 180 / np.pi, color='green')
    ax[1].set_ylabel("Angle (degrees)", color='green')
    ax[1].set_xlabel("Frequency (Hz)")
    ax[1].set_xlim([0, fs / 2])
    ax[1].set_yticks([-90, -60, -30, 0, 30, 60, 90])
    ax[1].set_ylim([-90, 90])
    ax[1].grid()

    plt.show()
