import numpy as np
from numpy.typing import NDArray

from medusa.core.utils import check_data_dims

__all__ = [
    "signal_orthogonalization",
]


def signal_orthogonalization(
    signal_1: NDArray,
    signal_2: NDArray | None = None,
) -> NDArray:
    """Orthogonalise each channel of ``signal_1`` w.r.t. every channel of
    ``signal_2`` using NumPy (CPU).

    For every pair of channels ``(i, j)`` the component of channel ``i`` of
    ``signal_2`` that is linearly explained by channel ``j`` of ``signal_1`` is
    removed, following the power-envelope-correlation procedure of O'Neill et
    al. (2015). The diagonal of the channel-by-channel output (``i == j``) is
    set to the original ``signal_1``.

    Parameters
    ----------
    signal_1 :
        Shape (n_segments, n_samples, n_channels). First M/EEG signal. 2-D
        ``(n_samples, n_channels)`` and 1-D ``(n_samples,)`` arrays are promoted
        by :func:`medusa.core.utils.check_data_dims`.
    signal_2 :
        Shape (n_segments, n_samples, n_channels). Second M/EEG signal. If
        ``None``, ``signal_2`` is set equal to ``signal_1``. Must share the
        number of segments and samples with ``signal_1``.

    Returns
    -------
    NDArray
        Shape (n_segments, n_samples, n_channels, n_channels). Orthogonalised
        signals. Axis 0 is the segment, axis 1 the sample, axis 2 the channel of
        ``signal_2`` being orthogonalised and axis 3 the channel of ``signal_1``
        it is orthogonalised against. Length-1 axes inserted while promoting an
        under-dimensioned ``signal_1`` are squeezed back out so the caller's
        leading/channel dimensionality is preserved.

    Raises
    ------
    ValueError
        If the number of segments and samples of ``signal_1`` and ``signal_2``
        do not match.

    References
    ----------
    O'Neill, G. C., Barratt, E. L., Hunt, B. A., Tewarie, P. K., & Brookes,
    M. J. (2015). Measuring electrophysiological connectivity by power envelope
    correlation: a technical review on MEG methods. Physics in Medicine &
    Biology, 60(21), R271.

    Examples
    --------
    >>> import numpy as np
    >>> from medusa.signal.orthogonalization import signal_orthogonalization
    >>> sig = np.random.default_rng(0).standard_normal((1, 1000, 8))
    >>> signal_orthogonalization(sig, sig).shape
    (1, 1000, 8, 8)
    """
    signal_1 = np.asarray(signal_1)
    signal_2 = signal_1 if signal_2 is None else np.asarray(signal_2)

    # Set canonical dimensions (n_segments, n_samples, n_channels)
    signal_1, inserted = check_data_dims(signal_1, rep_type='time_segments')
    signal_2, _ = check_data_dims(signal_2, rep_type='time_segments')

    if signal_1.shape[:2] != signal_2.shape[:2]:
        raise ValueError("signal_1 and signal_2 must share the number of "
                         "segments and samples")

    signal_1 = np.ascontiguousarray(signal_1)
    signal_2 = np.ascontiguousarray(signal_2)

    n_chan_1 = signal_1.shape[2]
    n_chan_2 = signal_2.shape[2]
    n_samp_1 = signal_1.shape[1]
    n_seg_1 = signal_1.shape[0]

    signal_1_chan = np.transpose(
        np.repeat(signal_1, repeats=n_chan_1, axis=2), (0, 2, 1))
    signal_2_chan = np.transpose(
        np.tile(signal_2, (1, 1, n_chan_2)), (0, 2, 1))

    # Add extra dimensions
    signal_1_chan = signal_1_chan[:, :, np.newaxis, :]
    signal_2_chan = signal_2_chan[:, :, np.newaxis, :]

    # Calculate pseudo inverse
    inv_signal_1_chan = np.repeat(
        np.linalg.pinv(np.transpose(signal_1, (0, 2, 1))[:, :, np.newaxis, :]),
        repeats=n_chan_1, axis=1)
    beta = np.matmul(signal_2_chan, inv_signal_1_chan)

    signal_ort = np.transpose(
        np.reshape(signal_2_chan - beta * signal_1_chan,
                   (n_seg_1, n_chan_2, n_chan_1, n_samp_1), order='F'),
        (0, 3, 1, 2))

    signal_ort[:, :, np.arange(n_chan_2), np.arange(n_chan_1)] = signal_1

    if inserted:
        signal_ort = np.squeeze(signal_ort, axis=inserted)

    return signal_ort

