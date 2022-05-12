import scipy.signal as sp_signal
import numpy as np
from medusa import hilbert
from numba import jit
import warnings


@jit(nopython=True, cache=True, parallel=True)
def reshape_angles_loops(phase_data):
    n_cha = phase_data.shape[0]

    m = np.empty((phase_data.shape[0] * phase_data.shape[0],
                 phase_data.shape[1]))
    for i in range(n_cha):
        for j in range(n_cha):
            m[n_cha * i + j] = phase_data[i] - phase_data[j]

    n = np.empty((phase_data.shape[0] * phase_data.shape[0]))
    for i in range(m.shape[0]):
        n[i] = np.mean(np.sign(np.sin(m[i])))
    pli_vector = np.absolute(n)
    pli = np.reshape(pli_vector, (n_cha, n_cha))

    plv_vector = np.divide(
        np.absolute(np.sum(np.exp(1j * m), axis=1)),
        phase_data.shape[1])
    plv = np.reshape(plv_vector, (n_cha, n_cha))

    imz = np.sin(m)
    num = np.empty((phase_data.shape[0] * phase_data.shape[0]))
    den = np.empty((phase_data.shape[0] * phase_data.shape[0]))
    for i in range(m.shape[0]):
        num[i] = np.absolute(
                    np.mean(np.multiply(np.absolute(imz[i]), np.sign(imz[i]))))
        den[i] = np.mean(np.absolute(imz[i]))
    wpli_vector = np.divide(num, den)
    wpli = np.reshape(wpli_vector, (n_cha, n_cha))

    return pli, wpli, plv


def __phase_connectivity_numba(data):
    # ERROR CHECK
    if type(data) != np.ndarray:
        raise ValueError("Parameter data must be of type numpy.ndarray")
    if data.shape[0] < data.shape[1]:
        warnings.warn("Warning: Signal dimensions flipped out. If you have more samples than channels, comment this "
                      "line")
        data = data.T

    # VARIABLE INITIALIZATION
    num_chan = data.shape[1]

    # CONNECTIVITY CALCULATION
    phase_data = np.transpose(np.angle(sp_signal.hilbert(np.transpose(data))))
    phase_data = np.ascontiguousarray(phase_data.T)
    # angles_1 = np.reshape(np.tile(phase_data, (num_chan, 1)),
    #                       (len(phase_data), num_chan * num_chan),
    #                       order='F')
    # angles_2 = np.tile(phase_data, (1, num_chan))

    # pli_vector = abs(np.mean(np.sign(np.sin(angles_1 - angles_2)), axis=0))

    pli, wpli, plv = reshape_angles_loops(phase_data)

    return pli, wpli, plv


def __phase_connectivity_cpu(data):
    # ERROR CHECK
    if type(data) != np.ndarray:
        raise ValueError("Parameter data must be of type numpy.ndarray")
    if data.shape[0] < data.shape[1]:
        warnings.warn("Warning: Signal dimensions flipped out. If you have more samples than channels, comment this "
                      "line")
        data = data.T

    # VARIABLE INITIALIZATION
    num_chan = data.shape[1]

    # CONNECTIVITY CALCULATION
    phase_data = np.transpose(np.angle(sp_signal.hilbert(np.transpose(data))))
    phase_data = np.ascontiguousarray(phase_data)
    angles_1 = np.reshape(np.tile(phase_data, (num_chan, 1)),
                          (len(phase_data), num_chan * num_chan),
                          order='F')
    angles_2 = np.tile(phase_data, (1, num_chan))

    pli_vector = abs(np.mean(np.sign(np.sin(angles_1 - angles_2)), axis=0))
    pli = np.reshape(pli_vector, (num_chan, num_chan), order='F')

    plv_vector = np.divide(
        abs(np.sum(np.exp(1j * (angles_1 - angles_2)), axis=0)),
        data.shape[0])
    plv = np.reshape(plv_vector, (num_chan, num_chan), order='F')

    imz = np.sin(angles_1 - angles_2)
    with np.errstate(divide='ignore', invalid='ignore'):
        wpli_vector = np.divide(
            abs(np.mean(np.multiply(abs(imz), np.sign(imz)), axis=0)),
            np.mean(abs(imz), axis=0)
        )
    wpli = np.reshape(wpli_vector, (num_chan, num_chan), order='F')

    return pli, wpli, plv


def __phase_connectivity_gpu(data):
    import tensorflow as tf
    # ERROR CHECK
    if type(data) != np.ndarray:
        raise ValueError("Parameter data must be of type numpy.ndarray")
    if data.shape[0] < data.shape[1]:
        warnings.warn("Warning: Signal dimensions flipped out. If you have more samples than channels, comment this "
                      "line")
        data = data.T

    # VARIABLE INITIALIZATION
    num_chan = data.shape[1]

    # CONNECTIVITY CALCULATION
    phase_data = tf.math.angle(hilbert.hilbert(data))

    angles_1 = tf.transpose(
                    tf.reshape(
                        tf.transpose(tf.tile(phase_data, (num_chan, 1))),
                        (num_chan * num_chan, len(phase_data)))
                )
    angles_2 = tf.tile(phase_data, (1, num_chan))

    pli_vector = tf.math.abs(
                    tf.math.reduce_mean(
                        tf.math.sign(
                            tf.math.sin(tf.math.subtract(angles_1, angles_2))),
                        axis=0))
    pli = tf.reshape(pli_vector, (num_chan, num_chan))

    plv_vector = tf.math.divide(
                    tf.math.abs(
                        tf.math.reduce_sum(
                            tf.math.exp(
                                tf.math.scalar_mul(
                                    1j,
                                    tf.cast(
                                        tf.math.subtract(angles_1, angles_2),
                                        'complex64'))),
                            axis=0)),
                    data.shape[0])
    plv = tf.reshape(plv_vector, (num_chan, num_chan))

    imz = tf.math.sin(tf.math.subtract(angles_1, angles_2))
    wpli_vector = tf.math.divide(
                    tf.math.abs(tf.math.reduce_mean(
                        tf.math.multiply(
                            tf.math.abs(imz),
                            tf.math.sign(imz)),
                        axis=0)),
                    tf.math.reduce_mean(tf.math.abs(imz), axis=0))
    wpli = tf.reshape(wpli_vector, (num_chan, num_chan))

    return pli, wpli, plv


def phase_connectivity(data):
    """This function calculates three phase-based connectivity parameters
        - Phase locking value (PLV): Mormann, 2000, Physica D: Nonlinear
        Phenomena, DOI: 10.1016/S0167-2789(00)00087-7. ¡CAUTION! Sensitive to
        volume conduction effects
        - Phase lag index (PLI): Nolte, 2007, Human Brain Mapping, DOI:
        10.1016/j.clinph.2004.04.029
        - weighted Phase Lag Index (wPLI): Vinck, 2011, NeuroImage, DOI:
        10.1016/j.neuroimage.2011.01.055

    Parameters
    ----------
    data : numpy 2D matrix
        MEEG Signal. SamplesXChannel.

    Returns
    -------
    pli : numpy 2D matrix
        Array of size ChannelsXChannels containing PLI values.
    wpli : numpy 2D matrix
        Array of size ChannelsXChannels containing wPLI values.
    plv : numpy 2D matrix
        Array of size ChannelsXChannels containing PLV values.

    """
    from medusa import tensorflow_integration
    # ERROR CHECK
    if not np.issubdtype(data.dtype, np.number):
        raise ValueError('data matrix contains non-numeric values')

    if tensorflow_integration.check_tf_config(autoconfig=True):
        pli, wpli, plv = __phase_connectivity_gpu(data)
    else:
        pli, wpli, plv = __phase_connectivity_cpu(data)

    return pli, wpli, plv


if __name__ == "__main__":
    import scipy.io
    import time
    import matplotlib.pyplot as plt
    mat = scipy.io.loadmat('Q:/VRG/0001_Control.mat')
    vector = np.array(mat["signal"])[0:20, 0:5000]

    t0 = time.time()
    pli, wpli, plv = __phase_connectivity_cpu(vector)
    t1 = time.time()
    plt.imshow(pli)
    plt.clim(0, 0.3)
    plt.show()
    plt.imshow(wpli)
    plt.clim(0, 0.3)
    plt.show()
    plt.imshow(plv)
    plt.clim(0, 0.3)
    plt.show()
    t2 = time.time()
    pli, wpli, plv = __phase_connectivity_gpu(vector)
    t3 = time.time()
    plt.imshow(pli)
    plt.clim(0, 0.3)
    plt.show()
    plt.imshow(wpli)
    plt.clim(0, 0.3)
    plt.show()
    plt.imshow(plv)
    plt.clim(0, 0.3)
    plt.show()
    t4 = time.time()
    pli, wpli, plv = __phase_connectivity_numba(vector)
    t5 = time.time()
    plt.imshow(pli)
    plt.clim(0, 0.3)
    plt.show()
    plt.imshow(wpli)
    plt.clim(0, 0.3)
    plt.show()
    plt.imshow(plv)
    plt.clim(0, 0.3)
    plt.show()

    print('Time CPU: ', t1 - t0)
    print('Time GPU: ', t3 - t2)
    print('Time Numba: ', t5 - t4)
