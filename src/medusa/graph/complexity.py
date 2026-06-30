import numpy as np
import scipy.special as sps
from numpy.typing import NDArray

__all__ = ["complexity"]


def __divergency(W: NDArray) -> float:
    """
    Calculates the graph divergency. Its the entropic distance (as euclidean
    distance) between a uniform weight distribution (random graph) and the
    graph under study.

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
    ind = np.argwhere(np.triu(np.ones((N, N)), k=1))
    W_wei = np.sum(W[ind[:, 0], ind[:, 1]])  # Tri Up indices

    if not W.all() == 0:
        p = W[ind[:, 0], ind[:, 1]] / W_wei  # Weights normalising

        p_sort = np.sort(p, axis=0)[::-1]  # Weight sorting (descending)

        # Full graph is the equilibrium graph (same weight for all the edges)
        p_equi_sort = 1 / len(p_sort) * np.ones((1, len(p_sort)))

        # Normalised euclidean distance
        c = sps.comb(N, 2)  # Number of connections without autoloops
        D = np.sqrt(c / (c - 1)) * np.sqrt(np.nansum(
            np.power(np.squeeze(np.asarray(p_equi_sort)) -
                     np.squeeze(np.asarray(p_sort)), 2)))
    else:  # Empty graph
        D = 0

    return D


def __graph_entropy(W: NDArray) -> float:
    """
    Calculates the Shannon entropy of a weighted graph.

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
    ind = np.argwhere(np.triu(np.ones((N, N)), k=1))
    W_wei = np.sum(W[ind[:, 0], ind[:, 1]])  # Tri Up indices

    if not W.all() == 0:
        p = W[ind[:, 0], ind[:, 1]] / W_wei  # Weights normalising

        H = -1 * np.sum(p[p > 0] * np.log2(p[p > 0]))  # Shannon entropy

        # Normalising H to range [0 1]
        K = np.log2(p.shape[0])
        H = H / K
    else:  # Empty or full graph
        H = 1

    return H


def __complexity(W: NDArray) -> float:
    """
    Calculates the Shannon Graph Complexity of a weighted graph.

    Parameters
    ----------
    W : numpy 2D matrix
        Graph matrix. ChannelsXChannels.

    Returns
    -------
    C : numpy array
        Network complexity.

    """
    tmp_1 = __divergency(W)
    tmp_2 = __graph_entropy(W)
    C = tmp_1 * tmp_2

    return C


def complexity(W: NDArray) -> float:
    """
    Calculates the complexity.

    Parameters
    ----------
    W : numpy 2D matrix
        Graph matrix. ChannelsXChannels.

    Returns
    -------
    global_comp : numpy array
        Global complexity.

    Examples
    --------
    >>> import numpy as np
    >>> from medusa.graph.complexity import complexity
    >>> W = np.array([[0., 1., 2.],
    ...               [1., 0., 1.],
    ...               [2., 1., 0.]])
    >>> isinstance(float(complexity(W)), float)
    True

    """
    W = np.asarray(W)
    if W.shape[0] != W.shape[1]:
        raise ValueError('W matrix must be square')

    if not np.issubdtype(W.dtype, np.number):
        raise ValueError('W matrix contains non-numeric values')

    global_comp = __complexity(W)

    return global_comp
