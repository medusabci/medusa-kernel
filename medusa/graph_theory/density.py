# Built-in imports
import warnings, os

# External imports
import numpy as np

# Medusa imports
from medusa.graph_theory import degree


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
    deg = degree.__degree_cpu(W)
    
    if check_symmetry == 0 or check_symmetry == 2:
        norm_value = N*(N-1)
    elif check_symmetry == 1:
        norm_value = ((N*(N-1))/2)
        
    nodal_den = np.divide(deg,norm_value)
    global_den = np.divide(np.sum(deg),norm_value)
      
    return global_den,nodal_den


def density(W):
    """
    Calculates the graph density.

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
    if W.shape[0] is not W.shape[1]:
        raise ValueError('W matrix must be square')
        
    if not np.issubdtype(W.dtype, np.number):
        raise ValueError('W matrix contains non-numeric values')        

    global_den,nodal_den = __density_cpu(W)
        
    return global_den, nodal_den
