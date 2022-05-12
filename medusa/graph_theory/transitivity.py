import tensorflow as tf
import numpy as np

def __trans_gpu(W):
    """
    Calculates the transitivity using GPU

    Parameters
    ----------
    W : numpy 2D matrix
        Graph matrix. ChannelsXChannels.
        
    Returns
    -------
    global_trans : int
        Global transitivity.
        
    """
    K = tf.reduce_sum(tf.where(W != 0,1,0),axis = 1)
    triples = tf.reduce_sum(tf.math.multiply(K,tf.math.subtract(K,1)))
    triangles = tf.math.pow(W,1/3)
    triangles = tf.linalg.matmul(tf.linalg.matmul(triangles,triangles),triangles)
    triangles = tf.linalg.tensor_diag_part(triangles)
    global_trans = tf.math.divide(tf.reduce_sum(triangles),tf.cast(triples,dtype=tf.float64))
        
    return global_trans


def __trans_cpu(W):
    """
    Calculates the transitivity using CPU

    Parameters
    ----------
    W : numpy 2D matrix
        Graph matrix. ChannelsXChannels.
        
    Returns
    -------
    global_trans : int
        Global transitivity.
        
    """
    
    K = np.sum(np.where(W != 0,1,0),axis = 1)
    triples = np.sum(K * (K-1))
    triangles = np.diag(np.linalg.matrix_power(W**(1/3),3))
    global_trans = np.sum(triangles) / triples
            
    return global_trans


def transitivity(W,mode):
    """
    Calculates the transitivity, which is the number of triangles divided by 
    the number of triples.

    Parameters
    ----------
    W : numpy 2D matrix
        Graph matrix. ChannelsXChannels.
    mode : string
        GPU or CPU
        
    Returns
    -------
    global_trans : int
        Global transitivity.

    """
    if W.shape[0] is not W.shape[1]:
        raise ValueError('W matrix must be square')
        
    if not np.issubdtype(W.dtype, np.number):
        raise ValueError('W matrix contains non-numeric values')        
      
    if mode == 'CPU':
        global_trans = __trans_cpu(W)
    elif mode == 'GPU':
        global_trans = __trans_gpu(W)
    else:
        raise ValueError('Unknown mode')
        
    return global_trans

# import scipy.io as rmat

# data = rmat.loadmat('D:/OneDrive - Universidad de Valladolid/Scripts/testPython/graphTest.mat')
# W = data['W']
# aa = transitivity(W,'CPU')
