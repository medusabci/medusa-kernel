# Built-in imports
import warnings, os

# External imports
import numpy as np

# Medusa imports
from medusa.graph_theory import degree
from medusa import pytorch_integration


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


def assortativity(W):
    """
    Calculates the assortativity, which is a preference of nodes to attach to 
    other nodes that are somehow similar to them

    Parameters
    ----------
    W : numpy 2D matrix
        Graph matrix. ChannelsXChannels.

    Returns
    -------
    global_assort : numpy array
        Global assortativity

    """
    if W.shape[0] is not W.shape[1]:
        raise ValueError('W matrix must be square')

    if not np.issubdtype(W.dtype, np.number):
        raise ValueError('W matrix contains non-numeric values')

    global_assort = __assort_cpu(W)

    return global_assort
