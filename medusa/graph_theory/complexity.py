# Built-in imports
import warnings, os

# External imports
import numpy as np
import scipy.special as sps


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


def complexity(W):
    """
    Calculates the graph complexity.

    Parameters
    ----------
    W : numpy 2D matrix
        Graph matrix. ChannelsXChannels.
        
    Returns
    -------
    global_comp : numpy array
        Global complexity.

    """
    if W.shape[0] is not W.shape[1]:
        raise ValueError('W matrix must be square')
        
    if not np.issubdtype(W.dtype, np.number):
        raise ValueError('W matrix contains non-numeric values')        

    global_comp = __complexity_cpu(W)

    return global_comp
 