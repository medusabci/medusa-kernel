# Built-in imports
import math

# External imports
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import pdist

# Medusa imports
from medusa.core.utils import ThreadWithReturnValue, check_data_dims

__all__ = ["sample_entropy"]


def sample_entropy(signal: NDArray, m: int, r: float,
                   dist_type: str = 'chebyshev') -> NDArray:
    """Sample entropy (SampEn) of a time-series signal.

    SampEn is a nonlinear measure of signal irregularity. It evaluates the
    conditional probability that sequences of ``m`` samples that are similar
    (within a tolerance ``r``) remain similar when one more point is added.
    SampEn assigns higher values to more irregular time sequences. It has two
    tuning parameters: the sequence length ``m`` and the tolerance ``r``.

    Parameters
    ----------
    signal :
        (n_segments, n_samples, n_channels). Input time-series signal. A 2-D
        ``(n_samples, n_channels)`` array is accepted and promoted; the
        leading segment axis is squeezed back out of the result.
    m :
        Embedding dimension (sequence length). Determines the length of the
        subsequences compared in the signal.
    r :
        Tolerance for considering matches, usually a proportion of the
        standard deviation of the signal (e.g. ``0.2``).
    dist_type :
        Distance metric used to compare subsequences. Must be one of the
        metrics supported by :func:`scipy.spatial.distance.pdist`. Default is
        ``'chebyshev'``.

    Returns
    -------
    NDArray
        (n_segments, n_channels). SampEn value for each segment and channel.
        For a 2-D input the segment axis is dropped, yielding
        ``(n_channels,)``.

    Raises
    ------
    ValueError
        If ``m`` is not smaller than the number of samples, if ``dist_type``
        is not a string, or if ``dist_type`` is not a distance metric
        supported by :func:`scipy.spatial.distance.pdist`.

    References
    ----------
    Richman, J. S., & Moorman, J. R. (2000). Physiological time-series
    analysis using approximate entropy and sample entropy. American Journal of
    Physiology-Heart and Circulatory Physiology.

    Notes
    -----
    - If the number of matches (A or B) is zero, SampEn returns the
      theoretical upper limit ``-log(2 / ((N - m - 1) * (N - m)))``.
    - Valid SampEn range: lower bound 0 (perfect regularity); upper bound
      ``log(N - m) + log(N - m - 1) - log(2)``.
    - This implementation uses threading for parallel computation across
      channels.

    Examples
    --------
    >>> import numpy as np
    >>> from medusa.signal.metrics.nonlinear.sample_entropy import (
    ...     sample_entropy,
    ... )
    >>> signal = np.random.randn(3, 1000, 2)  # 3 segments, 1000 samples, 2 ch
    >>> sampen = sample_entropy(signal, m=2, r=0.2, dist_type='chebyshev')
    >>> sampen.shape
    (3, 2)
    """

    # Check dimensions
    signal, inserted = check_data_dims(signal, rep_type='time_segments')

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

    # Squeeze back the inserted segment axis so a 2-D caller gets a
    # de-segmented result
    if 0 in inserted:
        value = np.squeeze(value, axis=0)
    return value
