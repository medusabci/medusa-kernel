import numpy as np
import warnings
from medusa.utils import check_dimensions


def signal_orthogonalization_cpu(signal_1, signal_2):

    """ This method implements the ortogonalization of each channel of signal_1
    regarding all the channels in signal_2 using CPU

    REFERENCES: O’Neill, G. C., Barratt, E. L., Hunt, B. A., Tewarie, P. K., &
    Brookes, M.J. (2015). Measuring electrophysiological connectivity by power
    envelope correlation: a technical review on MEG methods. Physics in Medicine
    & Biology, 60(21), R271.

    Parameters
    ----------
    signal_1 : numpy 2D matrix
        First MEEG Signal. Allowed dimensions: [n_epochs, n_samples, n_channels],
        [n_samples, n_channels] and [n_samples].
    signal_2 : numpy 2D matrix
        Second MEEG Signal. If empty signal_2 will be set to be equal to
        signal_1. Allowed dimensions: [n_epochs, n_samples, n_channels],
        [n_samples, n_channels] and [n_samples].

    Returns
    -------
    signal_ort : numpy 3D matrix
        MEEG ortogonalised signals. The first dimension is epochs. The second
        dimension is samples. The fourth dimension is the channel of first signal,
        and the second dimension is regarding which channel of second signal it
        has been orthogonalized the channel in third dimension.
        [n_epochs, n_samples, n_channels, n_channels].

    """
    # Error check
    if signal_2 is None:
        signal_2 = signal_1
    if len(signal_1) != len(signal_2):
        raise ValueError("Signals must have the same length")
    if (type(signal_1) != np.ndarray) or (type(signal_2) != np.ndarray):
        raise ValueError("Parameter signal_1 and signal_2 must be of type "
                         "numpy.ndarray")

    # Set correct dimensions
    signal_1 = np.ascontiguousarray(check_dimensions(signal_1))
    signal_2 = np.ascontiguousarray(check_dimensions(signal_2))

    n_chan_1 = signal_1.shape[2]
    n_chan_2 = signal_2.shape[2]
    n_samp_1 = signal_1.shape[1]
    n_epo_1 = signal_1.shape[0]

    signal_1_chan = np.transpose(np.repeat(signal_1, repeats=n_chan_1, axis=2), (0,2,1))
    signal_2_chan = np.transpose(np.tile(signal_2, (1, 1, n_chan_2)),(0,2,1))

    # Add extra dimensions
    signal_1_chan = signal_1_chan[:,:,np.newaxis,:]
    signal_2_chan = signal_2_chan[:,:,np.newaxis,:]

    # Calculate pseudo inverse
    inv_signal_1_chan = np.repeat(np.linalg.pinv(np.transpose(signal_1,(0,2,1))[:, :,
                                       np.newaxis, :]),repeats=n_chan_1,axis=1)
    beta = np.matmul(signal_2_chan, inv_signal_1_chan)

    signal_ort = np.transpose(np.reshape(signal_2_chan - beta * signal_1_chan,
                            (n_epo_1,n_chan_2,n_chan_1,n_samp_1),order='F'),(0,3,1,2))

    signal_ort[:,:,np.arange(n_chan_2),np.arange(n_chan_1)] = signal_1

    return signal_ort

def signal_orthogonalization_gpu(signal_1, signal_2):
    """
    This function ortogonalizes each channel of signal_1 regarding all the
    channels in signal_2. Based in O'Neill et al. 2015

    Parameters
    ----------
    signal_1 : numpy 3D matrix
        First MEEG Signal. Allowed dimensions: [n_epochs, n_samples, n_channels],
        [n_samples, n_channels] and [n_samples].
    signal_2 : numpy 3D matrix
        Second MEEG Signal. If empty signal_2 will be set to be equal to
        signal_1. Allowed dimensions: [n_epochs, n_samples, n_channels],
        [n_samples, n_channels] and [n_samples].

    Returns
    -------
    signal_ort : numpy 3D matrix
        MEEG ortogonalized signals. The first dimension is epochs. The second
        dimension is samples. The fourth dimension is the channel of first signal,
        and the second dimension is regarding which channel of second signal it
        has been orthogonalized the channel in third dimension.
        [n_epochs, n_samples, n_channels, n_channels].
    """
    import tensorflow as tf

    # Error check
    if signal_2 is None:
        signal_2 = signal_1
    if len(signal_1) != len(signal_2):
        raise ValueError("Signals must have the same length")
    if (type(signal_1) != np.ndarray) or (type(signal_2) != np.ndarray):
        raise ValueError("Parameter signal_1 and signal_2 must be of type "
                         "numpy.ndarray")

    # Set correct dimensions
    signal_1 = np.ascontiguousarray(check_dimensions(signal_1))
    signal_2 = np.ascontiguousarray(check_dimensions(signal_2))

    n_chan_1 = signal_1.shape[2]
    n_chan_2 = signal_2.shape[2]
    n_samp_1 = signal_1.shape[1]
    n_epo_1 = signal_1.shape[0]

    signal_1_chan = tf.transpose(tf.repeat(signal_1, repeats=n_chan_1, axis=2),
                                 perm=[0, 2, 1])
    signal_2_chan = tf.transpose(tf.tile(signal_2,(1,1,n_chan_2)),perm=[0,2,1])

    # Add extra dimensions
    signal_1_chan = signal_1_chan[:,:,tf.newaxis,:]
    signal_2_chan = signal_2_chan[:,:,tf.newaxis,:]

    # Calculate pseudo inverse
    inv_signal_1_chan = tf.repeat(
        tf.linalg.pinv(tf.transpose(signal_1, perm=[0, 2, 1])[:, :,
                       np.newaxis, :]), repeats=n_chan_1, axis=1)
    beta = np.matmul(signal_2_chan, inv_signal_1_chan)

    signal_ort = tf.transpose(tf.transpose(tf.reshape(tf.transpose(signal_2_chan - beta *
                                         signal_1_chan,perm=[0,2,3,1]),(n_epo_1,
                                                                        n_samp_1,
                                                                        n_chan_2,
                                                                        n_chan_1)),
                              perm =(0,3,2,1)),perm=[0,3,1,2]).numpy()

    signal_ort[:,:,tf.range(n_chan_2),tf.range(n_chan_1)] = signal_1

    return signal_ort


def signal_orthogonalization_gpu_old(signal_1, signal_2):
    """
    DEPRECATED: This functions performs the orthogonalization of the signal
    but it is not vectorized nor allow epochs as inputs. Its use is not
    recommended.

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
    # Error check
    if signal_2 is None:
        signal_2 = signal_1
    if len(signal_1) != len(signal_2):
        raise ValueError("Signals must have the same length")
    if (type(signal_1) != np.ndarray) or (type(signal_2) != np.ndarray):
        raise ValueError("Parameter signal_1 and signal_2 must be of type "
                         "numpy.ndarray")

    # Variable initialization
    signal_ort = tf.zeros((len(signal_1), len(signal_2[0]), len(signal_1[0])),
                          dtype=tf.dtypes.float64)
    i1 = tf.range(0, len(signal_1))
    i2 = tf.ones_like(i1)
    i3 = tf.ones_like(i1)

    # Orthogonalization process
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
