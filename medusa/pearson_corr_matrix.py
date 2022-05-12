import numpy as np


def pearson_corr_matrix(A, B):
    """
    This function calculates the Pearson correlation row wise (correlates each
    row in A with each row in B). The input is SamplesXChannels, but the
    matrices are transposed in a first step.

    Parameters
    ----------
    A : np.ndarray
        First variable to be correlated with shape [samples x channels]
    B : np.ndarray
        Second variable to be correlated with shape [samples x channels]

    Returns
    -------
    : numpy 2D matrix
        Array of size ChannelsXChannels (A.shape[0] x B.shape[0]) containing
        the correlation values
    """

    A = np.transpose(A)
    B = np.transpose(B)

    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]

    ssA = (A_mA**2).sum(1)
    ssB = (B_mB**2).sum(1)

    corr_values = np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None],ssB[None]))

    return corr_values
