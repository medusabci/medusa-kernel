import numpy as np
from numpy.typing import NDArray

__all__ = ["transitivity"]


def __trans(W: NDArray) -> float:
    """
    Calculates the transitivity.

    Parameters
    ----------
    W : numpy 2D matrix
        Graph matrix. ChannelsXChannels.

    Returns
    -------
    global_trans : int
        Global transitivity.

    """
    K = np.sum(np.where(W != 0, 1, 0), axis=1)
    triples = np.sum(K * (K - 1))
    triangles = np.diag(np.linalg.matrix_power(W ** (1 / 3), 3))
    global_trans = np.sum(triangles) / triples

    return global_trans


def transitivity(W: NDArray) -> float:
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

    Examples
    --------
    >>> import numpy as np
    >>> from medusa.graph.transitivity import transitivity
    >>> W = np.array([[0., 1., 2.],
    ...               [1., 0., 1.],
    ...               [2., 1., 0.]])
    >>> isinstance(float(transitivity(W)), float)
    True

    """
    W = np.asarray(W)
    if W.shape[0] != W.shape[1]:
        raise ValueError('W matrix must be square')

    if not np.issubdtype(W.dtype, np.number):
        raise ValueError('W matrix contains non-numeric values')

    global_trans = __trans(W)
    return global_trans
