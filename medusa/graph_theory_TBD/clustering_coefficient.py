import tensorflow as tf
import numpy as np

def __clustering_gpu(W):
    """
    Calculates the ClC of echa node of the graph contained in the matrix W using GPU

    It is calculated as follows:
              ___   ___  
              \     \    Wik * Wli * Wkl
              /__   /__  
              k~=i (l~=k & l~=i)
    ClC = -----------------------------
              ___   ___  
              \     \    Wik * Wil 
              /__   /__  
              k~=i (l~=k & l~=i)

    C.J. Stam, et ál. (2009). "Graph theoretical analysis of 
    magnetoencephalographic functional connectivity in Alzheimer’s 
    disease". Brain, 132:213–224.

    Parameters
    ----------
    W : numpy 2D matrix
        Graph matrix. ChannelsXChannels.
        
    Returns
    -------
    nodal_clc : numpy array
        Nodal clustering coefficient
        
    """

    A = tf.where(W>0,1,0)
    S = tf.math.add(tf.math.pow(W, 1/3), tf.math.pow(tf.transpose(W), 1/3))
    K = tf.math.reduce_sum(tf.math.add(A,tf.transpose(A)),axis=1)
    cyc_3 = tf.math.divide(tf.linalg.tensor_diag_part(tf.linalg.matmul(tf.linalg.matmul(S,S),S)),2)
    K = tf.where(cyc_3==0,np.inf,tf.cast(K,tf.float32))
    f1 = tf.math.subtract(K,1)
    f2 = tf.math.multiply(2,tf.linalg.tensor_diag_part(tf.linalg.matmul(A,A)))
    f3 = tf.math.multiply(K,f1)
    CYC_3 = tf.math.subtract(f3,tf.cast(f2,tf.float32))
    nodal_clc = tf.math.divide(tf.cast(cyc_3, tf.float32),CYC_3)
        
    return nodal_clc


def __clustering_cpu(W):
    """
    Calculates the ClC of echa node of the graph contained in the matrix W using CPU

    Parameters
    ----------
    W : numpy 2D matrix
        Graph matrix. ChannelsXChannels.
        
    Returns
    -------
    nodal_clc : numpy array
        Nodal clustering coefficient
        
    """
    
    if not np.issubdtype(W.dtype, np.number):
        raise ValueError('W matrix contains non-numeric values')     

    A = np.ones(W.shape)
    A[W<=0] = 0
    S = np.power(W, 1/3) + np.power(np.transpose(W), 1/3)
    K = np.sum(A + np.transpose(A),axis=1)
    cyc_3 = np.diag(np.linalg.matrix_power(S,3)) / 2
    K[cyc_3 == 0] = np.inf
    f1 = np.multiply(K,(K - 1))
    f2 = 2 * np.diag(np.linalg.matrix_power(A,2))
    CYC_3 = f1 - f2
    nodal_clc = cyc_3 / CYC_3
        
    return nodal_clc


def clustering_coefficient(W,mode):
    """
    Calculates the clustering coefficient.

    Parameters
    ----------
    W : numpy 2D matrix
        Graph matrix. ChannelsXChannels.
    mode : string
        GPU or CPU
        
    Returns
    -------
    nodal_clc : numpy array
        Nodal clustering coefficient

    """
    if W.shape[0] is not W.shape[1]:
        raise ValueError('W matrix must be square')
      
    if mode == 'CPU':
        nodal_clc = __clustering_cpu(W)
    elif mode == 'GPU':
        nodal_clc = __clustering_gpu(W)
    else:
        raise ValueError('Unknown mode')
        
    return nodal_clc
