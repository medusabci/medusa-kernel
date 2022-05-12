import tensorflow as tf
import numpy as np
from medusa import tensorflow_integration


def hilbert(x, flag=0):
    """Hilbert transform based on scipy implementation adapted for numba.

    Parameters
    ----------
    x : array_like
        Time series. SamplesXChannels.
    flag : bool
        To force using Tensorflow. MUCH slower.

    Returns
    -------
    xa : ndarray
        Analytic signal of x.
    """
    from medusa import tensorflow_integration
    from scipy.signal import hilbert as hilbert_sp
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


if __name__ == '__main__':

    import time
    import matplotlib.pyplot as plt
    from scipy.signal import hilbert as hilbert_sp
    import scipy

    tensorflow_integration.config_tensorflow()

    duration = 100
    fs = 100.0
    samples = int(fs * duration)
    t = np.arange(samples) / fs

    signal = np.sin(np.pi*t) * np.sin(16*np.pi*t)
    signal = signal.reshape(-1, 1)

    mdict = {"signal": signal}
    scipy.io.savemat('C:/Users/GIB/Desktop/data_orig', mdict)

    # Analytic signal using scipy
    t0_sp = time.time()
    analytic_signal_sp = hilbert(signal)
    t_sp = time.time() - t0_sp

    mdict = {"signal": analytic_signal_sp}
    scipy.io.savemat('C:/Users/GIB/Desktop/data_sp', mdict)

    # Analytic signal using tensorflow
    t0_tf = time.time()
    analytic_signal_tf = hilbert(signal, 1)
    t_tf = time.time() - t0_tf

    aa = analytic_signal_tf.numpy()

    mdict = {"signal": aa}
    scipy.io.savemat('C:/Users/GIB/Desktop/data_tf', mdict)

    print("Runtime hilbert transform (scipy): %.10f" % t_sp)
    print("Runtime hilbert transform (tensorflow): %.10f" % t_tf)

    # Plot scipy implementation
    amplitude_envelope = np.abs(analytic_signal_sp)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal_sp))
    instantaneous_frequency = (np.diff(instantaneous_phase, axis=0) /
                               (2.0 * np.pi) * fs)
    fig = plt.figure()
    ax0 = fig.add_subplot(211)
    ax0.plot(t, signal, label='signal')
    ax0.plot(t, amplitude_envelope, label='envelope')
    ax0.set_xlabel("time in seconds")
    ax0.legend()
    ax1 = fig.add_subplot(212)
    ax1.plot(t[1:], instantaneous_frequency)
    ax1.set_xlabel("time in seconds")
    ax1.set_ylim(0.0, 120.0)
    fig.suptitle('Scipy implementation')
    plt.show()

    # Plot tf implementation
    amplitude_envelope = np.abs(analytic_signal_tf)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal_tf))
    instantaneous_frequency = (np.diff(instantaneous_phase, axis=0) /
                               (2.0 * np.pi) * fs)
    fig = plt.figure()
    ax0 = fig.add_subplot(211)
    ax0.plot(t, signal, label='signal')
    ax0.plot(t, amplitude_envelope, label='envelope')
    ax0.set_xlabel("time in seconds")
    ax0.legend()
    ax1 = fig.add_subplot(212)
    ax1.plot(t[1:], instantaneous_frequency)
    ax1.set_xlabel("time in seconds")
    ax1.set_ylim(0.0, 120.0)
    fig.suptitle('Tensorflow implementation')
    plt.show()
