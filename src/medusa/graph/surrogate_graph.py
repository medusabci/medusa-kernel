import numpy as np
from numpy.typing import NDArray

__all__ = ["surrogate_graph"]


def surrogate_graph(W: NDArray) -> NDArray:
    """
    Builds a surrogate graph by randomly shuffling the upper-triangle edge
    weights of ``W`` and mirroring them to keep the result symmetric.

    Parameters
    ----------
    W : numpy 2D matrix
        Graph matrix. ChannelsXChannels.

    Returns
    -------
    surrog_matrix : numpy array
        Surrogate graph.

    Examples
    --------
    >>> import numpy as np
    >>> from medusa.graph.surrogate_graph import surrogate_graph
    >>> W = np.array([[0., 1., 2.],
    ...               [1., 0., 1.],
    ...               [2., 1., 0.]])
    >>> surrogate_graph(W).shape
    (3, 3)

    """
    W = np.asarray(W)
    if W.shape[0] != W.shape[1]:
        raise ValueError('W matrix must be square')

    if not np.issubdtype(W.dtype, np.number):
        raise ValueError('W matrix contains non-numeric values')

    surrog_matrix = np.zeros((W.shape[0], W.shape[1]))
    idx_up = np.argwhere(np.triu(W, k=1))
    val_up = W[idx_up[:, 0], idx_up[:, 1]]
    val_up_surrog = val_up[np.random.permutation(val_up.shape[0])]
    surrog_matrix[idx_up[:, 0], idx_up[:, 1]] = val_up_surrog
    surrog_matrix = surrog_matrix + np.transpose(surrog_matrix)

    return surrog_matrix
