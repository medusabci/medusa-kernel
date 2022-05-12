import numpy as np
# from numba import jit
import warnings


# @jit(nopython=True, cache=True)
# def __signal_orthogonalization_numba(signal_1, signal_2):
#     """Piece of code to calculate the signal orthogonalization matrix using
#     numba
#     """
#     # Create C contiguous numpy arrays to increase performance
#     signal_1 = signal_1.copy()
#     signal_2 = signal_2.copy()
#
#     # Init
#     n_samp = signal_1.shape[0]
#     n_cha_s1 = signal_1.shape[1]
#     n_cha_s2 = signal_1.shape[1]
#
#     signal_ort = np.empty((n_samp, n_cha_s2, n_cha_s1))
#     signal_ort[:] = np.nan
#
#     # Orthogonalization
#     for cha1 in range(n_cha_s1):
#         for cha2 in range(n_cha_s2):
#             signal_1_cha = signal_1[:, cha1]
#             signal_2_cha = signal_2[:, cha2]
#             if not np.array_equal(signal_1_cha, signal_2_cha):
#                 signal_1_cha_inv = np.linalg.pinv(signal_1_cha.reshape(-1, 1))
#                 signal_1_cha_inv = signal_1_cha_inv.reshape(
#                     signal_1_cha_inv.shape[1]
#                 )
#                 beta = np.dot(signal_2_cha, signal_1_cha_inv)
#                 signal_ort[:, cha2, cha1] = \
#                     signal_2_cha - beta * signal_1_cha
#             else:
#                 signal_ort[:, cha2, cha1] = signal_1_cha
#     return signal_ort


def signal_orthogonalization_cpu(signal_1, signal_2):
    """
    This function ortogonalizes each channel of signal_1 regarding all the
    channels in signal_2. Based in O'Neill et al. 2015

    Parameters
    ----------
    signal_1 : numpy 2D matrix
        MEEG Signal. SamplesXChannel
    signal_2 : numpy 2D matrix
        MEEG Signal. SamplesXChannel

    Returns
    -------
    signal_ort : np.ndarray
        MEEG ortogonalised signals. Shape [samples x channels x channels].
        The first dimension are the samples, the third is the base channel
        used to ortogonalize the other channels and the second dimension are
        the ortogonalized channels regarding the third dimension

    """
    # ERROR CHECK
    if len(signal_1) != len(signal_2):
        raise ValueError("Signals must have the same length")
    if (type(signal_1) != np.ndarray) or (type(signal_2) != np.ndarray):
        raise ValueError("Parameter signal_1 and signal_2 must be of type "
                         "numpy.ndarray")
    if signal_1.shape[0] < signal_1.shape[1]:
        warnings.warn("Warning: Signal dimensions flipped out. If you have more samples than channels, comment this "
                      "line")
        signal_1 = signal_1.T
    if signal_2.shape[0] < signal_2.shape[1]:
        warnings.warn(
            "Warning: Signal dimensions flipped out. If you have more samples than channels, comment this "
            "line")
        signal_2 = signal_2.T

    # VARIABLE INITIALIZE
    signal_ort = np.empty((signal_1.shape[0], signal_2.shape[1], signal_1.shape[1]))
    signal_ort[:] = np.nan

    # ORTHOGONALIZATION PROCESS
    for chan_1 in range(0, signal_1.shape[1]):
        for chan_2 in range(0, signal_2.shape[1]):
            signal_1_chan = signal_1[:, chan_1]
            signal_2_chan = signal_2[:, chan_2]
            if not np.array_equal(signal_1_chan, signal_2_chan):
                beta = np.matmul(signal_2_chan[None], np.linalg.pinv(signal_1_chan[None]))
                signal_ort[:, chan_2, chan_1] = signal_2_chan - beta * signal_1_chan
            else:
                signal_ort[:, chan_2, chan_1] = signal_1_chan

    # signal_ort = __signal_orthogonalization_numba(signal_1, signal_2)

    return signal_ort


def signal_orthogonalization_gpu(signal_1, signal_2):
    """
    This function ortogonalizes each channel of signal_1 regarding all the 
    channels in signal_2. Based in O'Neill et al. 2015
    
    Parameters
    ----------
    signal_1 : numpy 2D matrix
        MEEG Signal. SamplesXChannel.
    signal_2 : numpy 2D matrix
        MEEG Signal. SamplesXChannel.

    Returns
    -------
    signal_ort : numpy 3D matrix
        MEEG ortogonalised signals. Samples x Channel x Channel.
        
        The first dimension are the samples, the third is the base channel
        used to ortogonalize the other channels and the second dimension are
        the ortogonalized channels regarding the third dimension
    """

    import tensorflow as tf
    # ERROR CHECK
    if len(signal_1) != len(signal_2):
        raise ValueError("Signals must have the same length")
    if (type(signal_1) != np.ndarray) or (type(signal_2) != np.ndarray):
        raise ValueError("Parameter signal_1 and signal_2 must be of type "
                         "numpy.ndarray")
        
    # VARIABLE INITIALIZE
    signal_ort = tf.zeros((len(signal_1), len(signal_2[0]), len(signal_1[0])),
                          dtype=tf.dtypes.float64)
    i1 = tf.range(0, len(signal_1))
    i2 = tf.ones_like(i1)
    i3 = tf.ones_like(i1)
            
    # ORTHOGONALIZATION PROCESS
    signal_1_tf = tf.convert_to_tensor(signal_1)
    signal_2_tf = tf.convert_to_tensor(signal_2)
    
    for chan_1 in range(0, len(signal_1[0])):

        for chan_2 in range(0, len(signal_2[0])):

            signal_1_chan = tf.slice(signal_1_tf,
                                     [0, chan_1],
                                     [len(signal_1), 1])
            signal_2_chan = tf.slice(signal_2_tf,
                                     [0, chan_2],
                                     [len(signal_2), 1])
            if not np.array_equal(signal_1_chan.numpy(), signal_2_chan.numpy()):
                beta = tf.linalg.matmul(
                    tf.transpose(signal_2_chan),
                    tf.transpose(tf.linalg.pinv(signal_1_chan))
                )
                tmp = tf.math.subtract(
                    signal_2_chan, tf.math.multiply(beta,signal_1_chan)
                )
                idx = tf.stack([i1, chan_2 * i2, chan_1 * i3], axis=-1)                
                signal_ort = tf.tensor_scatter_nd_update(
                    signal_ort, idx, tf.squeeze(tmp)
                )
            else:
                idx = tf.stack([i1, chan_2 * i2, chan_1 * i3], axis=-1)
                signal_ort = tf.tensor_scatter_nd_update(
                    signal_ort, idx, tf.squeeze(signal_1_chan)
                )
    return signal_ort.numpy()
