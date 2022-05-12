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
