import math
import numpy as np
from scipy.spatial.distance import pdist

# TODO More Entropy Estimator Algorithms Must Be Implemented

def sample_entropy(signal,m,r,dist_type='chebyshev', result = None, scale = None):


    # Check Errors
    if m > len(signal):
        raise ValueError('Embedding dimension must be smaller than the signal length (m<N).')
    if len(signal) != signal.size:
        raise ValueError('The signal parameter must be a [Nx1] vector.')
    if not isinstance(dist_type, str):
        raise ValueError('Distance type must be a string.')
    if dist_type not in ['braycurtis', 'canberra', 'chebyshev', 'cityblock',
                         'correlation', 'cosine', 'dice', 'euclidean', 'hamming',
                         'jaccard', 'jensenshannon', 'kulsinski', 'mahalanobis',
                         'matching', 'minkowski', 'rogerstanimoto', 'russellrao',
                         'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']:
        raise ValueError('Distance type unknown.')

    # Useful parameters
    N = len(signal)
    sigma = np.std(signal)
    templates_m = []
    templates_m_plus_one = []
    signal = np.squeeze(signal)

    for i in range(N - m + 1):
        templates_m.append(signal[i:i + m])

    B = np.sum(pdist(templates_m, metric=dist_type) <= sigma * r)
    if B == 0:
        value = math.inf
    else:
        m += 1
        for i in range(N - m + 1):
            templates_m_plus_one.append(signal[i:i + m])
        A = np.sum(pdist(templates_m_plus_one, metric=dist_type) <= sigma * r)

        if A == 0:
            value = math.inf


        else:

            value = -np.log((A / B) * ((N - m + 1) / (N - m - 1)))

    """IF A = 0 or B = 0, SamEn would return an infinite value. 
    However, the lowest non-zero conditional probability that SampEn should
    report is A/B = 2/[(N-m-1)*(N-m)]"""

    if math.isinf(value):

        """Note: SampEn has the following limits:
                - Lower bound: 0 
                - Upper bound : log(N-m) + log(N-m-1) - log(2)"""

        value = -np.log(2/((N-m-1)*(N-m)))

    if result is not None:
        result[scale-1] = value

    return value



if __name__ == '__main__':
    np.random.seed(26)
    x = np.random.rand(5000)
    result=sample_entropy(x,m=2,r=0.2)
    print(result)
