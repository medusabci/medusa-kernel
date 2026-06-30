import numpy as np
from numpy.typing import NDArray

__all__ = ["degree"]


def __aux_symm_triu(W: NDArray) -> NDArray:
    """Nodal degree of a symmetric/upper-triangular adjacency matrix."""
    N = np.shape(W)[0]
    aux = np.ones((N, N))
    aux = np.triu(aux, k=1)
    W = W * aux
    W = W + np.transpose(W)
    W = np.sum(W, axis=0) / 2
    return W


def __aux_no_match(W: NDArray) -> NDArray:
    """Nodal degree of a directed (non-symmetric) adjacency matrix."""
    in_degree = np.sum(W, axis=0)
    out_degree = np.sum(W, axis=1)
    W = in_degree + out_degree
    return W


def __degree(W: NDArray) -> NDArray:
    """
    Calculates node degree (also called strength in weighted networks).

    Parameters
    ----------
    W : numpy 2D matrix
        Graph matrix. ChannelsXChannels.

    Returns
    -------
    nodal_degree : numpy array
        Nodal degree.

    """
    W = np.divide(np.round(W * 10000000000), 10000000000)
    W = W - np.diag(np.diag(W))

    check_symmetry = (W.transpose() == W).all()  # if symmetric

    if (W == np.triu(W)).all():  # if upper triangular
        check_symmetry = 1

    if (W.transpose() == -W).all():  # if anti-symmetric
        check_symmetry = 2

    if check_symmetry == 0:
        nodal_degree = __aux_no_match(W)
    elif check_symmetry == 1:
        nodal_degree = __aux_symm_triu(W)
    elif check_symmetry == 2:
        nodal_degree = -np.sum(W, axis=0)
    return nodal_degree


def degree(W: NDArray) -> NDArray:
    """
    Calculates the degree.

    Parameters
    ----------
    W : numpy 2D matrix
        Graph matrix. ChannelsXChannels.

    Returns
    -------
    nodal_degree : numpy array
        Nodal degree.

    Examples
    --------
    >>> import numpy as np
    >>> from medusa.graph.degree import degree
    >>> W = np.array([[0., 1., 2.],
    ...               [1., 0., 1.],
    ...               [2., 1., 0.]])
    >>> degree(W).shape
    (3,)

    """
    W = np.asarray(W)
    if W.shape[0] != W.shape[1]:
        raise ValueError('W matrix must be square')

    if not np.issubdtype(W.dtype, np.number):
        raise ValueError('W matrix contains non-numeric values')

    nodal_degree = __degree(W)
    return nodal_degree
