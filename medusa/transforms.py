import tensorflow as tf
import numpy as np
from medusa import tensorflow_integration
from scipy.signal import hilbert as hilbert_sp


def hilbert(x, flag=0):
    """This method implements the Hilbert transform.

    Parameters
    ----------
    x :  numpy 2D matrix
        MEEG Signal. [n_samples x n_channels].
    flag : bool
        If True, if forces using Tensorflow. It is not recommended as it is MUCH
         slower.

    Returns
    -------
    hilb : numpy 2D matrix
        Analytic signal of x.
    """
    x = np.asarray(x)
    if np.iscomplexobj(x):
        raise ValueError("x must be real.")
    n = x.shape[0]
    if n == 0:
        raise ValueError("Incorrect dimensions along axis 0")

    if tensorflow_integration.check_tf_config(autoconfig=True) and flag:

        # Run the fft on the columns, not the rows.
        x = tf.convert_to_tensor(x, dtype=tf.complex128)
        x = tf.transpose(tf.signal.fft(tf.transpose(x)))

        # Coeficients
        h = np.zeros(n)
        if (n > 0) and (2*np.fix(n/2) == n):
            # Even and nonempty
            h[0:int(n/2+1)] = 1
            h[1:int(n/2)] *= 2
        elif n > 0:
            # Odd and nonempty
            h[0] = 1
            h[1:int((n+1)/2)] = 2

        tf_h = tf.constant(h, name='h', dtype=tf.float64)
        if len(x.shape) == 2:
            reps = tf.Tensor.get_shape(x).as_list()[-1]
            hs = tf.stack([tf_h]*reps, -1)
        elif len(x.shape) == 1:
            hs = tf_h
        else:
            raise NotImplementedError

        xc = x * tf.complex(hs, tf.zeros_like(hs))
        return tf.transpose(tf.signal.ifft(tf.transpose(xc)))
    else:
        return hilbert_sp(x, axis=0)


def power_spectral_density(signal, fs, epoch_len=None):
    """This method allows to compute the power spectral density by means of
    Welch's periodogram method.

    Parameters
    ----------
    signal : numpy 2D matrix
        Signal. [n_samples x n_channels].
    fs : int
        Sampling frequency of the signal
    epoch_len : int or None
        Length of the epochs in which divide the signal. If None,
        the power spectral density of the entire signal will be calculated.

    Returns
    -------
    f : numpy 1D array
        Array of sample frequencies.

    psd: numpy 2D array
        PSD of M/EEG Signal. [n_epochs, n_samples, n_channels]
    """

    if len(signal.shape) < 2:
        signal = signal[ ..., np.newaxis]

    if epoch_len is not None:
        if not isinstance(epoch_len,int):
            raise ValueError("Epoch length must be a integer"
                             "value.")
        if epoch_len > signal.shape[0]:
            raise ValueError("Epoch length must be shorter than"
                             "signal duration")
    else:
        epoch_len = signal.shape[0]

    signal_epoched = medusa.get_epochs(signal, epoch_len)

    if len(signal_epoched.shape) < 3:
        signal_epoched = signal_epoched[np.newaxis, ...]

    # Estimating the PSD
    # Get the number of samples for the PSD length
    n_samp = signal_epoched.shape[1]
    # Compute the PSD
    f, psd = spsig.welch(signal_epoched, fs=fs, window='boxcar',
                         nperseg=n_samp, noverlap=0, axis=-2)

    return f,psd