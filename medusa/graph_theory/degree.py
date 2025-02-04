# Built-in imports
import warnings, os

# External imports
import numpy as np

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


def degree(W):
    """
    Calculates the graph degree.

    Parameters
    ----------
    W : numpy 2D matrix
        Graph matrix. ChannelsXChannels.
    Returns
    -------
    nodal_degree : numpy array
        Nodal degree.      

    """
    if W.shape[0] is not W.shape[1]:
        raise ValueError('W matrix must be square')
        
    if not np.issubdtype(W.dtype, np.number):
        raise ValueError('W matrix contains non-numeric values')

    nodal_degree = __degree_cpu(W)
    return nodal_degree