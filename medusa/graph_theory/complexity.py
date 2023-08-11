# Built-in imports
import warnings, os

# External imports
import numpy as np
import scipy.special as sps

# Medusa imports
from medusa import tensorflow_integration

# Extras
if os.environ.get("MEDUSA_EXTRAS_GPU_TF") == "1":
    import tensorflow as tf
    import tensorflow_probability as tfp

def __divergency_cpu(W):
    """
    Calculates the graph divergency. Its the entropic distance (as euclidean
    distance) between a uniform weight distribution (random graph) and the 
    network under study using CPU
    
    Parameters
    ----------
    W : numpy 2D matrix
        Graph matrix. ChannelsXChannels.

    Returns
    -------
    D : numpy array
        Network divergency.
        
    """    
    N = W.shape[0]
    ind = np.argwhere(np.triu(np.ones((N,N)),k=1))
    W_wei = np.sum(W[ind[:,0],ind[:,1]]) # Tri Up indices
    
    if not W.all() == 0:
        p = W[ind[:,0],ind[:,1]] / W_wei # Weights normalising
    
        p_sort = np.sort(p,axis=0)[::-1] # Weight sorting (descending)
        
        p_equi_sort = 1 / len(p_sort) * np.ones((1,len(p_sort))) # Full network is the
        #equilibrium network (same weight for all the edges)
        
        # Normalised euclidean distance
        c = sps.comb(N,2) # Number of connections without autoloops
        D = np.sqrt(c / (c - 1)) * np.sqrt(np.nansum(
                np.power(np.squeeze(np.asarray(p_equi_sort))-np.squeeze(np.asarray(p_sort)),2)))
    else: # Empty graph
        D = 0 
        
    return D


def __graph_entropy_cpu(W):
    """
    Calculates the Shannon entropy of a weighted graph using CPU
    
    Parameters
    ----------
    W : numpy 2D matrix
        Graph matrix. ChannelsXChannels.

    Returns
    -------
    H : numpy array
        Network entropy.
        
    """     
    N = W.shape[0]
    ind = np.argwhere(np.triu(np.ones((N,N)),k=1))
    W_wei = np.sum(W[ind[:,0],ind[:,1]]) # Tri Up indices

    if not W.all() == 0:
        p = W[ind[:,0],ind[:,1]] / W_wei # Weights normalising

        H = -1 * np.sum(p[p > 0] * np.log2(p[p>0])) # Shannon entropy calculation

        # Normalising H to range [0 1]
        K = np.log2(p.shape[0])
        H = H/K
    else: # Empty or full graph
        H = 1

    return H         
    

def __complexity_cpu(W):
    """
    Calculates the Shannon Graph Complexity of a graph that node i belong to 
    one of the network shortest paths using CPU
    
    Note: W must be converted to a connection-length matrix. It is common to 
    obtain it via mapping from weight to length. BC is normalised to the 
    range [0 1] as BC/[(N-1)(N-2)]
        
    Parameters
    ----------
    W : numpy 2D matrix
        Graph matrix. ChannelsXChannels.

    Returns
    -------
    C : numpy array
        Network complexity.
        
    """
    tmp_1 = __divergency_cpu(W)
    tmp_2 = __graph_entropy_cpu(W)
    C = tmp_1 * tmp_2
                
    return C

def __aux_divergency(W,N,ind,W_wei):
    p = tf.math.divide(tf.gather_nd(W,tf.cast(ind,tf.int32)),W_wei) # Weights normalising

    p_sort = tf.sort(p,axis=0,direction='DESCENDING') # Weight sorting (descending)
    
    p_equi_sort = tf.math.multiply(tf.math.divide(1,p_sort.shape[0]),tf.ones((1,p_sort.shape[0]),dtype=tf.float64)) # Full network is the
    #equilibrium network (same weight for all the edges)
    
    # Normalised euclidean distance
    N = tf.cast(N,tf.float32)
    c1 = tfp.distributions.Binomial(total_count=N, probs=0.5) # Binomial distr.
    c2 = tf.math.multiply(tf.math.pow(0.5,2),tf.math.pow((1-0.5),tf.math.subtract(N,tf.constant(2,dtype=tf.float32))))
    c = tf.math.round(tf.math.divide(c1.prob(2),c2)) # N choose 2
    f1 = tf.math.sqrt(tf.math.divide(c,tf.math.subtract(c,1.0)))
    f2 = tf.math.pow(tf.math.subtract(tf.reshape(p_equi_sort,[-1]),tf.reshape(p_sort,[-1])),2)
    f2 = tf.math.sqrt(tf.experimental.numpy.nansum(f2,dtype=tf.float32))
    D = tf.math.multiply(f1,f2)
    
    return D
        
        
def __divergency_gpu(W):
    """
    Calculates the graph divergency. Its the entropic distance (as euclidean
    distance) between a uniform weight distribution (random graph) and the 
    network under study using GPU
    
    Parameters
    ----------
    W : numpy 2D matrix
        Graph matrix. ChannelsXChannels.

    Returns
    -------
    D : numpy array
        Network divergency.
        
    """    
    W = tf.convert_to_tensor(W)
    N = W.shape[0]
    
    ind = tf.where(tf.linalg.band_part(tf.ones((N,N)), 0, -1) - tf.linalg.band_part(tf.ones((N,N)), 0, 0))
    W_wei = tf.reduce_sum(tf.gather_nd(W,ind)) # Tri Up indices
    
    D = tf.cond(tf.reduce_sum(W)!=0, lambda: __aux_divergency(W,N,ind,W_wei),lambda: tf.constant(0))
        
    return D


def __aux_entropy(W,ind,W_wei):
    p = tf.math.divide(tf.gather_nd(W,tf.cast(ind,tf.int32)),W_wei) # Weights normalising
    
    H1 = tf.constant(-1,dtype=tf.float64)
    H2 = tf.gather_nd(p,tf.where(p>0))
    H3 = tf.experimental.numpy.log2(tf.transpose(tf.gather_nd(p,tf.where(p>0))))
    H4 = tf.reduce_sum(tf.math.multiply(H2,H3))

    H = tf.math.multiply(H1,H4)

    K = tf.experimental.numpy.log2(p.shape[0])
    
    H = tf.math.divide(H,K)
    return H

def __graph_entropy_gpu(W):
    """
    Calculates the Shannon entropy of a weighted graph using GPU
    
    Parameters
    ----------
    W : numpy 2D matrix
        Graph matrix. ChannelsXChannels.

    Returns
    -------
    H : numpy array
        Network entropy.
        
    """     
    W = tf.convert_to_tensor(W)
    N = W.shape[0]
    
    ind = tf.where(tf.linalg.band_part(tf.ones((N,N)), 0, -1) - tf.linalg.band_part(tf.ones((N,N)), 0, 0))
    W_wei = tf.reduce_sum(tf.gather_nd(W,ind)) # Tri Up indices
    
    H = tf.cond(tf.reduce_sum(W)!=0, lambda: __aux_entropy(W,ind,W_wei),lambda: tf.constant(1))

    return H         
    

def __complexity_gpu(W):
    """
    Calculates the Shannon Graph Complexity of a graph that node i belong to 
    one of the network shortest paths using GPU
    
    Note: W must be converted to a connection-length matrix. It is common to 
    obtain it via mapping from weight to length. BC is normalised to the 
    range [0 1] as BC/[(N-1)(N-2)]
        
    Parameters
    ----------
    W : numpy 2D matrix
        Graph matrix. ChannelsXChannels.

    Returns
    -------
    C : numpy array
        Network complexity.
        
    """
    tmp_1 = __divergency_gpu(W)
    tmp_2 = __graph_entropy_gpu(W)
    C = tf.math.multiply(tmp_1,tf.cast(tmp_2,tf.float32))
                
    return C


def complexity(W,mode):
    """
    Calculates the graph complexity.

    Parameters
    ----------
    W : numpy 2D matrix
        Graph matrix. ChannelsXChannels.
    mode : string
        GPU or CPU
        
    Returns
    -------
    global_comp : numpy array
        Global complexity.

    """
    if W.shape[0] is not W.shape[1]:
        raise ValueError('W matrix must be square')
        
    if not np.issubdtype(W.dtype, np.number):
        raise ValueError('W matrix contains non-numeric values')        

    if mode == 'GPU' and os.environ.get("MEDUSA_EXTRAS_GPU_TF") == "1" and \
            tensorflow_integration.check_tf_config(autoconfig=True):
        global_comp = __complexity_gpu(W)
    else:
        global_comp = __complexity_cpu(W)

    return global_comp
 