# Built-in imports
import math
import ctypes
import os

# External imports
import numpy as np
from scipy.spatial.distance import pdist
from scipy.signal import decimate

# Medusa imports
from medusa.components import ThreadWithReturnValue
from medusa.utils import check_dimensions

def sample_entropy(signal, m, r, dist_type='chebyshev'):
    """
    This method implements the sample entropy (SampEn) of a time-series signal.

    SampEn is a nonlinear measure of signal irregularity. It evaluates the conditional
    probability that sequences of `m` samples that are similar (within a tolerance `r`)
    remain similar when one more point is added. SampEn assigns higher values to more
    irregular time sequences. It has two tuning parameters: the sequence length (m) and the
    tolerance (r)

    References
    ----------
    Richman, J. S., & Moorman, J. R. (2000). Physiological
    time-series analysis using approximate entropy and sample entropy. American
    Journal of Physiology-Heart and Circulatory Physiology.

    Parameters
    ----------
    signal : numpy.ndarray
        Input signal with shape [n_epochs, n_samples, n_channels].
    m : int
        Embedding dimension (sequence length). Determines the length of subsequences
        compared in the signal.

    r : float
        Tolerance value for considering matches, usually a proportion of the standard
        deviation of the signal (e.g., 0.2).

    dist_type : str, optional
        Distance metric used for comparing subsequences. Must be one of the distance
        metrics supported by `scipy.spatial.distance.pdist`. Default is 'chebyshev'.


    Returns
    -------
    sampen : numpy.ndarray
        SampEn value with shape [n_epochs, n_channels]

    Notes
    -----
    - If the number of matches (A or B) is zero, SampEn returns the theoretical upper
      limit using the formula: -log(2 / ((N - m - 1) * (N - m))).
    - Valid SampEn range:
        - Lower bound: 0 (perfect regularity)
        - Upper bound: log(N - m) + log(N - m - 1) - log(2)
    - This implementation uses threading for parallel computation across channels.

    Examples
    --------
    >>> import numpy as np
    >>> from medusa.signal_metrics.sample_entropy import sample_entropy

    >>> # Simulate a signal: 3 epochs, 1000 samples, 2 channels
    >>> signal = np.random.randn(3, 1000, 2)
    >>> m = 2
    >>> r = 0.2
    >>> sampen_values = sample_entropy(signal, m, r, dist_type='chebyshev')

    >>> print(sampen_values.shape)
    (3, 2)

    >>> print(sampen_values)
    [[1.23 1.18]
     [1.35 1.42]
     [1.01 1.07]]
    """

    # Check dimensions
    signal = check_dimensions(signal)

    # Check Errors
    if m > signal.shape[1]:
        raise ValueError('Embedding dimension must be smaller than the signal '
                         'length (m<N).')
    if not isinstance(dist_type, str):
        raise ValueError('Distance type must be a string.')
    if dist_type not in ['braycurtis', 'canberra', 'chebyshev', 'cityblock',
                         'correlation', 'cosine', 'dice', 'euclidean',
                         'hamming', 'jaccard', 'jensenshannon', 'kulsinski',
                         'mahalanobis', 'matching', 'minkowski',
                         'rogerstanimoto', 'russellrao', 'seuclidean',
                         'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']:
        raise ValueError(
            'Distance type unknown. Please, check allowed distances'
            'in pdist function from scipy.spatial.distance module.')

    # Useful parameters
    n_epo = signal.shape[0]
    N = signal.shape[1]
    n_channels = signal.shape[2]
    sigma = np.std(signal, axis=1)
    templates_m = []
    templates_m_plus_one = []
    B, A, value = np.empty((n_epo, n_channels)), np.empty((n_epo, n_channels)),\
                  np.empty((n_epo, n_channels))

    # Calculate B values
    for i in range(N - m + 1):
        templates_m.append(signal[:, i:i + m, :])
    templates_m = np.array(templates_m)
    for e_idx in range(n_epo):
        w_threads = []
        for ch_idx in range(n_channels):
            t = ThreadWithReturnValue(
                target=pdist,
                args=(templates_m[:, e_idx, :, ch_idx], dist_type,))
            w_threads.append(t)
            t.start()
        for th_idx, thread in enumerate(w_threads):
            B[e_idx, th_idx] = np.sum(thread.join() <= sigma[0, th_idx] * r)

    # Check if there is any B = 0
    zeros_idx = np.where(B == 0)
    value[zeros_idx] = math.inf

    # Calculate A values
    m += 1
    for i in range(N - m + 1):
        templates_m_plus_one.append(signal[:, i:i + m, :])
    templates_m_plus_one = np.array(templates_m_plus_one)
    for e_idx in range(n_epo):
        w_threads = []
        for ch_idx in range(n_channels):
            t = ThreadWithReturnValue(
                target=pdist, args=(templates_m_plus_one[:, e_idx, :, ch_idx],
                                    dist_type,))
            w_threads.append(t)
            t.start()
        for th_idx, thread in enumerate(w_threads):
            A[e_idx, th_idx] = np.sum(thread.join() <= sigma[0, th_idx] * r)

    # Check if there is any A = 0
    zeros_idx = np.where(A == 0)
    value[zeros_idx] = math.inf

    non_inf = np.where(value != math.inf)
    value[non_inf] = -np.log((A[non_inf] / B[non_inf])*((N - m + 1) / (N - m - 1)))

    # If there is infinity values
    inf_indx = np.where(value == math.inf)
    value[inf_indx] = -np.log(2 / ((N - m - 1) * (N - m)))
    return value