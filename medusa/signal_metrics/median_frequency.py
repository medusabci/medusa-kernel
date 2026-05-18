import numpy as np
from medusa.utils import check_dimensions
from .spectral_edge_frequency import spectral_edge_frequency

def median_frequency(psd, fs, target_band=None, eps=1e-20):
    """
    Compute the Median Frequency (MF) of a PSD within a frequency band.

    This function calculates the frequency below which 50% of the total power
    in the specified band lies. It is a convenience wrapper around
    :func:`spectral_edge_frequency` with ``percentile=50``.

    Parameters
    ----------
    psd : numpy.ndarray
        Power Spectral Density array with shape ``[n_epochs, n_freqs, n_channels]``.
        If dimensions are missing, use `np.newaxis` to expand them (e.g.,
        ``[n_freqs] -> [1, n_freqs, 1]``).
    fs : int or float
        Sampling frequency in Hz. Used to construct the frequency axis, assuming
        the PSD corresponds to the range ``[0, fs/2]``.
    target_band : tuple or list, optional
        Frequency band limits ``(low_freq, high_freq)`` in Hz used for the
        calculation. If None, it defaults to the full range ``(0, fs/2)``.
    eps : float, optional
        Small epsilon value passed to `spectral_edge_frequency` to detect
        near-zero power. Defaults to 1e-20.

    Returns
    -------
    median_freqs : numpy.ndarray
        Median Frequency values for each epoch and channel with shape
        ``[n_epochs, n_channels]``.

    See Also
    --------
    spectral_edge_frequency : Function used internally with percentile=50.

    Examples
    --------
    >>> import numpy as np
    >>> from medusa.signal_metrics.median_frequency import median_frequency
    >>> fs = 256
    >>> # 10 epochs, 129 frequency bins, 16 channels
    >>> psd = np.random.rand(10, 129, 16)
    >>> mf = median_frequency(psd, fs, target_band=[1, 50])
    >>> mf.shape
    (10, 16)
    """
    # To numpy arrays
    return spectral_edge_frequency(
        psd,
        fs,
        percentile=50,
        target_band=target_band,
        eps=eps
    )