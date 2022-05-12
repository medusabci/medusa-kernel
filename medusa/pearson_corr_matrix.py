import numpy as np


def pearson_corr_matrix(a, b):
    """ This method implements the Pearson correlation row wise (correlates each
    row in a with each row in b). The input is [n_samples x n_channels], but the
    matrices are transposed in a first step.

    Parameters
    ----------
    a : numpy 2D matrix
        First variable to be correlated. [n_samples x n_channels].
    b : numpy 2D matrix
        Second variable to be correlated. [n_samples x n_channels].

    Returns
    -------
    corr_values : numpy 2D square matrix
        matrix containing the correlation values. [n_channels x n_channels].

    """
    a = np.transpose(a)
    b = np.transpose(b)

    A_mA = a - a.mean(1)[:, None]
    B_mB = b - b.mean(1)[:, None]

    ssA = (A_mA ** 2).sum(1)
    ssB = (B_mB ** 2).sum(1)

    corr_values = np.dot(A_mA, B_mB.T) / np.sqrt(
        np.dot(ssA[:, None], ssB[None]))

    return corr_values
