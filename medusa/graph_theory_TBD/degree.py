import tensorflow as tf
import numpy as np


def __aux_symm_triu_gpu(W):
    N = tf.shape(W)
    aux = tf.subtract(tf.linalg.band_part(tf.ones(N), 0, -1),tf.linalg.band_part(tf.ones(N), 0, 0))
    W = tf.math.multiply(W,tf.cast(aux,tf.float64))
    W = tf.math.add(W,tf.transpose(W))
    W = tf.math.divide(tf.math.reduce_sum(W,axis=0),2)
    return W


def __aux_no_match_gpu(W):
    in_degree = tf.math.reduce_sum(W,axis=0)
    out_degree = tf.math.reduce_sum(W,axis=1)
    W = tf.math.add(in_degree, out_degree)
    return W    


def __degree_gpu(W):
    """
    Calculates node degree (also called strength in weighted networks) using GPU

    Parameters
    ----------
    W : numpy 2D matrix
        Graph matrix. ChannelsXChannels.
        
    Returns
    -------
    nodal_degree : numpy array
        Nodal degree.      

    """
    W = tf.math.divide(tf.math.round(tf.math.multiply(W,10000000000)),10000000000)
    W = W - tf.cast(tf.eye(tf.shape(W)[0]),tf.float64)    
    
    check_symmetry = tf.reduce_all(tf.math.equal(W,tf.transpose(W)))
    check_symmetry = tf.cond(
        tf.math.reduce_sum(tf.subtract(tf.linalg.band_part(W, -1, 0),tf.linalg.band_part(W, 0, 0))) == 0,
        lambda: 1, lambda: check_symmetry)
    check_symmetry = tf.cond(
        tf.reduce_all(tf.math.equal(W,-tf.transpose(W))),
        lambda: 2, lambda: check_symmetry)
        
    nodal_degree = tf.switch_case(tf.cast(check_symmetry,tf.int32), 
                branch_fns={0: lambda: __aux_no_match_gpu(W), 1: lambda: __aux_symm_triu_gpu(W), 2: lambda: -tf.math.reduce_sum(W,axis=0)})     
      
    return nodal_degree


def __aux_symm_triu_cpu(W):
    N = np.shape(W)[0]
    aux = np.ones((N,N))
    aux = np.triu(aux,k=1)
    W = W * aux
    W = W + np.transpose(W)
    W = np.sum(W,axis=0) / 2
    return W


def __aux_no_match_cpu(W):
    in_degree = np.sum(W,axis=0)
    out_degree = np.sum(W,axis=1)
    W = in_degree + out_degree
    return W    


def __degree_cpu(W):
    """
    Calculates node degree (also called strength in weighted networks) using CPU

    Parameters
    ----------
    W : numpy 2D matrix
        Graph matrix. ChannelsXChannels.
        
    Returns
    -------
    nodal_degree : numpy array
        Nodal degree.      

    """
    W = np.divide(np.round(W * 10000000000),10000000000)
    W = W - np.diag(np.diag(W))    
    
    check_symmetry = (W.transpose() == W).all() # if symmetric
    
    if (W == np.triu(W)).all(): # if upper triangular
        check_symmetry = 1
        
    if (W.transpose() == -W).all(): # if anti-symmetric
        check_symmetry = 2
        
    if check_symmetry == 0:
        nodal_degree = __aux_no_match_cpu(W)
    elif check_symmetry == 1:
        nodal_degree =  __aux_symm_triu_cpu(W)
    elif check_symmetry == 2:
        nodal_degree = -np.sum(W,axis=0)
          
    return nodal_degree


def degree(W,mode):
    """
    Calculates the graph degree.

    Parameters
    ----------
    W : numpy 2D matrix
        Graph matrix. ChannelsXChannels.
    mode : string
        GPU or CPU
        
    Returns
    -------
    nodal_degree : numpy array
        Nodal degree.      

    """
    if W.shape[0] is not W.shape[1]:
        raise ValueError('W matrix must be square')
        
    if not np.issubdtype(W.dtype, np.number):
        raise ValueError('W matrix contains non-numeric values')        
      
    if mode == 'CPU':
        nodal_degree = __degree_cpu(W)
    elif mode == 'GPU':
        nodal_degree = __degree_gpu(W)
    else:
        raise ValueError('Unknown mode')

    return nodal_degree