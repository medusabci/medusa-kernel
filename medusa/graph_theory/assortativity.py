# Built-in imports
import warinings, os

# External imports
import numpy as np

# Medusa imports
from medusa.graph_theory import degree
from medusa import tensorflow_integration

# Extras
if os.environ.get("MEDUSA_EXTRAS_GPU_TF") == "1":
    import tensorflow as tf


def __assort_gpu(W):
    """
    Calculates the assortativity using GPU

    Parameters
    ----------
    W : numpy 2D matrix
        Graph matrix. ChannelsXChannels.
        
    Returns
    -------
    global_assort : numpy array
        Global assortativity
        
    """

    deg = degree.__degree_gpu(W)
    aux = tf.linalg.band_part(W, 0, -1) - tf.linalg.band_part(W, 0, 0)
    ind = tf.where(aux > 0)
    K = ind.shape[0]  # Equal to N choose 2 is there is no 0
    deg_i = tf.gather(deg, tf.transpose(ind[:, 0]))
    deg_j = tf.gather(deg, tf.transpose(ind[:, 1]))

    num_1 = tf.math.divide(tf.reduce_sum(tf.math.multiply(deg_i, deg_j)), K)
    num_2 = tf.math.multiply(tf.math.add(deg_i, deg_j), 0.5)
    num_2 = tf.math.square(tf.math.divide(tf.reduce_sum(num_2), K))
    num = num_1 - num_2

    den_1 = tf.math.multiply(
        tf.math.add(tf.math.square(deg_i), tf.math.square(deg_j)), 0.5)
    den_1 = tf.math.divide(tf.reduce_sum(den_1), K)
    den_2 = num_2
    den = den_1 - den_2

    global_assort = tf.math.divide(num, den)
    return global_assort


def __assort_cpu(W):
    """
    Calculates the assortativity using CPU

    Parameters
    ----------
    W : numpy 2D matrix
        Graph matrix. ChannelsXChannels.
        
    Returns
    -------
    global_assort : numpy array
        Global assortativity
        
    """

    deg = degree.__degree_cpu(W)
    ind = np.triu_indices(W.shape[0], 1, W.shape[1])
    K = ind[0].shape
    deg_i = deg[ind[0]]
    deg_j = deg[ind[1]]

    num_1 = np.sum(deg_i * deg_j) / K
    num_2 = (np.sum(0.5 * (deg_i + deg_j)) / K) ** 2
    num = num_1 - num_2

    den_1 = np.sum(0.5 * (deg_i ** 2 + deg_j ** 2)) / K
    den_2 = num_2
    den = den_1 - den_2

    global_assort = num / den

    return global_assort


def assortativity(W, mode):
    """
    Calculates the assortativity, which is a preference of nodes to attach to 
    other nodes that are somehow similar to them

    Parameters
    ----------
    W : numpy 2D matrix
        Graph matrix. ChannelsXChannels.
    mode : string
        GPU or CPU

    Returns
    -------
    global_assort : numpy array
        Global assortativity

    """
    if W.shape[0] is not W.shape[1]:
        raise ValueError('W matrix must be square')

    if not np.issubdtype(W.dtype, np.number):
        raise ValueError('W matrix contains non-numeric values')

    if mode == 'GPU' and os.environ.get("MEDUSA_EXTRAS_GPU_TF") == "1" and \
            tensorflow_integration.check_tf_config(autoconfig=True):
        global_assort = __assort_gpu(W)
    else:
        global_assort = __assort_cpu(W)

    return global_assort
