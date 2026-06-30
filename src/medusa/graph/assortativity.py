import numpy as np
from numpy.typing import NDArray

# Medusa imports
from medusa.graph import degree

__all__ = ["assortativity"]


def __assort(W: NDArray) -> float:
    """
    Calculates the assortativity.

    Parameters
    ----------
    W : numpy 2D matrix
        Graph matrix. ChannelsXChannels.

    Returns
    -------
    global_assort : numpy array
        Global assortativity

    """
    deg = degree.__degree(W)
    ind = np.triu_indices(W.shape[0], 1, W.shape[1])
    K = ind[0].shape[0]
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


def assortativity(W: NDArray) -> float:
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

    Examples
    --------
    >>> import numpy as np
    >>> from medusa.graph.assortativity import assortativity
    >>> W = np.array([[0., 1., 2.],
    ...               [1., 0., 1.],
    ...               [2., 1., 0.]])
    >>> isinstance(float(assortativity(W)), float)
    True

    """
    W = np.asarray(W)
    if W.shape[0] != W.shape[1]:
        raise ValueError('W matrix must be square')

    if not np.issubdtype(W.dtype, np.number):
        raise ValueError('W matrix contains non-numeric values')

    global_assort = __assort(W)

    return global_assort
