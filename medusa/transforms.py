import numpy as np
from scipy.signal import welch as welch_sp
from scipy.signal import hilbert as hilbert_sp
from medusa.utils import check_dimensions


def hilbert(signal):
    """This method implements the Hilbert transform.

    Parameters
    ----------
    signal :  numpy.ndarray
        Input signal with shape [n_epochs x n_samples x n_channels].

    Returns
    -------
    hilb : numpy 3D matrix
        Analytic signal of x [n_epochs, n_samples, n_channels].
    """
    # Check dimensions
    signal = np.asarray(signal)
    signal = check_dimensions(signal)
    # Check errors
    if np.iscomplexobj(signal):
        raise ValueError("Signal must be real.")

    # Old tensorflow implementation
    # if tensorflow_integration.check_tf_config(autoconfig=True) and flag:
    #
    #     # Run the fft on the columns, not the rows.
    #     signal = tf.convert_to_tensor(signal, dtype=tf.complex128)
    #     signal = tf.transpose(tf.signal.fft(tf.transpose(signal)))
    #
    #     # Coeficients
    #     h = np.zeros(n)
    #     if (n > 0) and (2*np.fix(n/2) == n):
    #         # Even and nonempty
    #         h[0:int(n/2+1)] = 1
    #         h[1:int(n/2)] *= 2
    #     elif n > 0:
    #         # Odd and nonempty
    #         h[0] = 1
    #         h[1:int((n+1)/2)] = 2
    #
    #     tf_h = tf.constant(h, name='h', dtype=tf.float64)
    #     if len(signal.shape) == 2:
    #         reps = tf.Tensor.get_shape(signal).as_list()[-1]
    #         hs = tf.stack([tf_h]*reps, -1)
    #     elif len(signal.shape) == 1:
    #         hs = tf_h
    #     else:
    #         raise NotImplementedError
    #
    #     xc = signal * tf.complex(hs, tf.zeros_like(hs))
    #     return tf.transpose(tf.signal.ifft(tf.transpose(xc)))

    return hilbert_sp(signal, axis=1)


def power_spectral_density(signal, fs, segment_pct=80, overlap_pct=50,
                           window='boxcar'):
    """This method allows to compute the one-sided power spectral density (PSD)
    by means of Welch's periodogram method. This method wraps around
    scipy.signal.welch method to compute the PSD, allowing to pass epoched
    signals and defining the segment length and overlap in percentage,
    simplifying the use for specific purposes. For more advanced configurations,
    use the original scipy (or equivalent) function.

    Parameters
    ----------
    signal : numpy nd array
        Signal with shape [n_epochs x n_samples x n_channels].
    fs : int
        Sampling frequency of the signal
    segment_pct: float
        Percentage of the signal (n_samples) used to calculate the FFT. Default:
        80% of the signal.
    overlap_pct: float
        Percentage of overlap (n_samples) for the Welch method. Default: 50% of
        the signal.
    window:
        Desired window to use. See scipy.signal.welch docs for more details

    Returns
    -------
    f : numpy 1D array
        Array of sampled frequencies.
    psd: numpy 2D array
        PSD of the signal with shape [n_epochs, n_samples, n_channels]
    """

    # Check signal dimensions
    signal = check_dimensions(signal)
    # Get the number of samples for the PSD length
    n_samp = signal.shape[1]
    # Get nperseg and noverlap
    nperseg = n_samp * segment_pct / 100
    noverlap = n_samp * overlap_pct / 100
    # Compute the PSD
    f, psd = welch_sp(signal, fs=fs, window=window, nperseg=nperseg,
                      noverlap=noverlap, axis=1)
    return f, psd


def normalize_psd(psd, norm='rel'):
    """Normalizes the PSD using different methods.

    Parameters
    ----------
    psd: numpy array or list
        Power Spectral Density (PSD) of the signal with shape [samples],
        [samples x channels] or [epochs x samples x channels]. It assumes PSD
        is the one-sided spectrum.
    norm: string
        Normalization to be performed. Choose z for z-score or rel for
        relative power.

    """
    # To numpy arrays
    psd = np.array(psd)

    # Check errors
    if len(psd.shape) != 3:
        raise Exception('Parameter psd must have shape [n_epochs x n_samples x '
                        'n_channels]')

    if norm == 'rel':
        p = np.sum(psd, axis=1, keepdims=True)
        psd_norm = psd / p
    elif norm == 'z':
        m = np.mean(psd, keepdims=True, axis=1)
        s = np.std(psd, keepdims=True, axis=1)
        psd_norm = (psd - m) / s
    else:
        raise Exception('Unknown normalization. Choose z or rel')

    return psd_norm


