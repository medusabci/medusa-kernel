import tensorflow as tf
import numpy as np
from medusa.graph_theory import degree

def __density_gpu(W):
    """
    Calculates the graph density using GPU.

    Parameters
    ----------
    W : numpy 2D matrix
        Graph matrix. ChannelsXChannels.
        
    Returns
    -------
    global_den : int
        Global density.   
    nodal_den : numpy array
        Nodal density. 

    """    
    
    check_symmetry = tf.reduce_all(tf.math.equal(W,tf.transpose(W)))
    check_symmetry = tf.cond(
        tf.math.reduce_sum(tf.subtract(tf.linalg.band_part(W, -1, 0),tf.linalg.band_part(W, 0, 0))) == 0,
        lambda: 1, lambda: check_symmetry)
    check_symmetry = tf.cond(
        tf.reduce_all(tf.math.equal(W,-tf.transpose(W))),
        lambda: 2, lambda: check_symmetry)
    
    N = W.shape[0]
    deg = degree.degree(W,'GPU')
    
    norm_value = tf.switch_case(tf.cast(check_symmetry,tf.int32), 
                branch_fns={0: lambda: tf.math.multiply(N,tf.math.subtract(N,1)),
                            1: lambda: tf.math.divide(tf.math.multiply(N,tf.math.subtract(N,1)),2), 
                            2: lambda: tf.math.multiply(N,tf.math.subtract(N,1))})

    nodal_den = tf.divide(deg,norm_value)
    global_den = tf.divide(tf.math.reduce_sum(deg),norm_value)     
      
    return global_den,nodal_den


def __density_cpu(W):
    """
    Calculates the graph density using CPU.

    Parameters
    ----------
    W : numpy 2D matrix
        Graph matrix. ChannelsXChannels.
        
    Returns
    -------
    global_den : int
        Global density.   
    nodal_den : numpy array
        Nodal density.    

    """    
    
    check_symmetry = (W.transpose() == W).all() # if symmetric
    
    if (W == np.triu(W)).all(): # if upper triangular
        check_symmetry = 1
        
    if (W.transpose() == -W).all(): # if anti-symmetric
        check_symmetry = 2
    
    N = W.shape[0]
    deg = degree.degree(W,'CPU')
    
    if check_symmetry == 0 or check_symmetry == 2:
        norm_value = N*(N-1)
    elif check_symmetry == 1:
        norm_value = ((N*(N-1))/2)
        
    nodal_den = np.divide(deg,norm_value)
    global_den = np.divide(np.sum(deg),norm_value)
      
    return global_den,nodal_den


def density(W,mode):
    """
    Calculates the graph density.

    Parameters
    ----------
    W : numpy 2D matrix
        Graph matrix. ChannelsXChannels.
    mode : string
        GPU or CPU
        
    Returns
    -------
    global_den : int
        Global density.   
    nodal_den : numpy array
        Nodal density. 

    """
    if W.shape[0] is not W.shape[1]:
        raise ValueError('W matrix must be square')
        
    if not np.issubdtype(W.dtype, np.number):
        raise ValueError('W matrix contains non-numeric values')        
      
    if mode == 'CPU':
        global_den,nodal_den = __density_cpu(W)
    elif mode == 'GPU':
        global_den,nodal_den = __density_gpu(W)
    else:
        raise ValueError('Unknown mode')
        
    return global_den, nodal_den
