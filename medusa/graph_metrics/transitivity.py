# Built-in imports
import warnings, os

# External imports
import numpy as np


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


def transitivity(W):
    """
    Calculates the transitivity, which is the number of triangles divided by 
    the number of triples.

    Parameters
    ----------
    W : numpy 2D matrix
        Graph matrix. ChannelsXChannels.
        
    Returns
    -------
    global_trans : int
        Global transitivity.

    """
    if W.shape[0] is not W.shape[1]:
        raise ValueError('W matrix must be square')
        
    if not np.issubdtype(W.dtype, np.number):
        raise ValueError('W matrix contains non-numeric values')

    global_trans = __trans_cpu(W)
    return global_trans
