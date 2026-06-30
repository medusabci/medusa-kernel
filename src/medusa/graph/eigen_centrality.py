import numpy as np
from numpy.typing import NDArray

__all__ = ["eigen_centrality"]


def eigen_centrality(W: NDArray) -> NDArray:
    """
    Calculates the eigenvector centrality, which is a centrality measure based
    on the adjacency matrix eigenvectors

    Parameters
    ----------
    W : numpy 2D matrix
        Graph matrix. ChannelsXChannels.

    Returns
    -------
    nodal_eig : numpy array
        Nodal eigenvector centrality

    Examples
    --------
    >>> import numpy as np
    >>> from medusa.graph.eigen_centrality import eigen_centrality
    >>> W = np.array([[0., 1., 2.],
    ...               [1., 0., 1.],
    ...               [2., 1., 0.]])
    >>> eigen_centrality(W).shape
    (3, 1)

    """
    W = np.asarray(W)
    if W.shape[0] != W.shape[1]:
        raise ValueError('W matrix must be square')

    if not np.issubdtype(W.dtype, np.number):
        raise ValueError('W matrix contains non-numeric values')

    D, V = np.linalg.eig(W)

    idx = np.where(D == np.max(D))[0]

    nodal_eig = abs(V[:, idx])

    return nodal_eig
