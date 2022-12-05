import numpy as np
from medusa import components
from numpy import linalg as nlinalg
from scipy import linalg as slinalg
import warnings


class LaplacianFilter(components.ProcessingMethod):
    """
    Class for fitting and applying Laplacian Filter to EEG Signals.
    A channel set from EEGChannelSet class must have been defined before calling
    LaplacianFilter class.

    This class implements the second order Hjorth's approximation of the Laplacian
    surface for a spatial-discrete system [1].

    It counts with two different modes:
    - Auto: First, the location of the channel to be filtered is identified.
            This allows us to determine the number of surrounding electrodes to
            be taken into account when calculating the laplacian surface
            (i.e., an electrode located in the center of the assembly is
            not the same as an electrode located in a corner). Then, apply the
            Laplacian surface correction taking into account the distance at
            which each electrode is located. It should take into account that
            this mode only applies the laplacian surface to the closest
            electrodes, so for next-nearest neighbours [2] the custom mode should be
            used.
    - Custom: In this mode, a list containing the labels of the channels to be
              used to calculate the Laplacian surface of each channel to be
              filtered must be defined. This allows the design of long
              distance filters [2].

    References
    [1] Claudio Carvalhaes, J. Acacio de Barros, The surface Laplacian technique
        in EEG: Theory and methods, International Journal of Psychophysiology,
        Volume 97, Issue 3, 2015, Pages 174-188.
    [2] Dennis J McFarland, Lynn M. McCabe, Stephen V. David, Jonathan R. Wolpaw,
        Spatial filter selection for EEG-based communication, Electroencephalography
        and clinical Neurophysiology, Volume 193, 1997, Pages 386-394.
    """

    def __init__(self, channel_set, mode='auto'):
        super().__init__(apply_lp=['s_filtered'])
        """
        Constructor of class LaplacianFilter

        Parameters
        ----------
        channel_set: EEGChannelSet
            EEGChannelSet instance
        mode: str {'auto'|'custom'}
        """
        # Check Channel Set is initialized
        if not channel_set.channels:
            raise Exception('Cannot compute the nearest neighbors if channel set '
                            'is not initialized!')

        # Chech if there are enough channels to perform filtering
        if len(channel_set.l_cha) < 5:
            raise Exception('There are not enough channels to perform Laplacian'
                            'surface filtering. ')
        # Parameters
        self.channel_set = channel_set
        self.l_cha = channel_set.l_cha
        self.n_cha = channel_set.n_cha
        self.mode = mode

        # Variables
        self.dist_matrix = None
        self.lp_filter = []
        self.idx_cha_to_filter = None
        self.dist_weights = []

    def fit_lp(self, l_cha_to_filter, l_cha_laplace=None):
        """
        Fits the Laplacian Filter depending on the mode chosen

        Parameters
        ----------
        l_cha_to_filter: list of strings
            List [N x 1] containing the labels of the channels to filter. Used in
            both filtering modes.
        l_cha_laplace: list
            List of lists [N x M] containing the labels of the channels to
            compute the laplace filter for channel in position Ni of
            l_cha_to_filter. Only used in mode custom.
        """

        self.dist_matrix = self.channel_set.compute_dist_matrix()
        self.idx_cha_to_filter = np.empty(len(l_cha_to_filter))

        if self.mode == 'auto':
            for i in range(len(l_cha_to_filter)):
                # Get the indexes of channels to filter
                self.idx_cha_to_filter[i] = self.l_cha.index(l_cha_to_filter[i])
                nearest_channels = np.unique(np.round(np.sort(self.dist_matrix
                                                    [self.idx_cha_to_filter[i].astype(int),:])[:5],2))

                # Check the num of channels that can be taken to perform Laplacian
                # It is fully surrounded by other channels
                if len(nearest_channels) == 2 or len(nearest_channels) == 3:
                    n_cha_lp = 4
                # It is in one side
                elif len(nearest_channels) == 4:
                    n_cha_lp = 3
                # It is in a corner
                elif len(nearest_channels) == 5:
                    n_cha_lp = 2

                # Get the closest n channels
                self.lp_filter.append(np.argsort(self.dist_matrix[self.
                                                           idx_cha_to_filter[i].astype(int), :])
                                                [1:n_cha_lp + 1])

                # Get the distances of the n channels
                self.dist_weights.append(1./np.sort(self.dist_matrix[self.
                                                           idx_cha_to_filter[i].astype(int), :])
                                                [1:n_cha_lp + 1])

            self.lp_filter = np.array(self.lp_filter,dtype=object)
            self.dist_weights = np.array(self.dist_weights, dtype=object)

        elif self.mode == 'custom':
            # Check Errors
            if l_cha_to_filter is None:
                raise ValueError("[LaplacianFilter] In 'custom' mode is necessary to "
                                 "set the labels of the channels to filter in"
                                 "'l_cha_to_filter'")

            if l_cha_laplace is None:
                raise ValueError("[LaplacianFilter] In 'custom' mode is necessary to "
                                 "set the labels of the channels to compute the "
                                 "Laplacian filter in 'l_cha_to_filter'.")

            if len(l_cha_to_filter) != len(l_cha_laplace):
                raise ValueError("[LaplacianFilter] In 'custom' mode is necessary to"
                                 "define as many list with channel labels to compute"
                                 "the Laplacian filter as channels to apply such"
                                 "filter.")

            for i in range(len(l_cha_to_filter)):
                # Get the channel indexes
                self.idx_cha_to_filter[i] = self.l_cha.index(l_cha_to_filter[i])
                # Get the channel indexes for Laplacian filtering
                self.lp_filter.append(np.array([self.l_cha.index(x)
                                                      for x in l_cha_laplace[i]]))
                self.dist_weights.append(1./self.dist_matrix[self.idx_cha_to_filter[i].astype(int),
                                                           np.array([self.l_cha.index(x)
                                               for x in l_cha_laplace[i]])])

    def apply_lp(self, signal):
        """
        Applies Laplacian filter to an EEG signal

        Parameters
        ----------
        signal: np.ndarray
            Array of EEG signal with shape [N_samples x N_channels]

        Returns
        -------
        s_filtered: np.ndarray
        Filtered EEG signal with shape [N_samples x len(l_cha_to_filter)].
        """
        # Check dimensions
        if signal.shape[1] != len(self.l_cha):
            raise ValueError('Dimensions of s in axis 1 must match the number '
                             'of channels')

        s_filtered = np.empty((signal.shape[0],len( self.idx_cha_to_filter)))

        for i, index in enumerate(self.idx_cha_to_filter.astype(int)):
            s_filtered[:,i] = signal[:, index] - np.average(signal[:,
                                                        self.lp_filter[i].astype(int)],
                                                        weights=self.dist_weights[i],
                                                            axis=1)
        return s_filtered


def car(signal):
    """Implementation of the common average reference (CAR) spatial filter.

    This class applies a CAR filter over a signal with shape [samples x
    channels]. The CAR filtering substracts the averaged signal of all
    electrodes to every single one, following the expression below:

            X_car = X - (1/N)*sum(X,2),

    where X is the EEG signal with dimensions [samples x channels], N is the
    total number of channels, and sum(~,2) denotes the sum over the second
    dimension (i.e., over the channels, returning a [samples x 1] signal). In
    practice, this operation can be implemented as a matrix multiplication:

            X_car = (M * X')',

    where X is the original signal [samples x ch], and M is the spatial
    filter matrix, composed by:

                | 1-(1/N)   -1/N    -1/N  |
            M = | -1/N    1-(1/N)  -1/N   |
                | -1/N     -1/N   1-(1/N) |

    Parameters
    ----------
    signal: np.ndarray
        EEG raw signal with dimensions [samples x ch]. Note that the TRIGGER
        channel must be DELETED before calling this function. In other
        words, eeg_signal must not contain the TRIGGER channel, only the EEG
         channels.

    Returns
    -------
    signal: np.array
        Filtered signal

    """
    # Number of channels
    n_cha = signal.shape[1]
    if n_cha > 1:
        # Create the CAR spatial filter matrix
        m = np.ones([n_cha, n_cha]) * (-1 / float(n_cha))
        np.fill_diagonal(m, 1 - (1 / float(n_cha)))
        signal = np.dot(signal, m)
    return signal


class CSP(components.ProcessingMethod):
    """ The class CSP performs a Common Spatial Pattern filtering.

    Attributes
    ----------
    filters : {(…, M, M) numpy.ndarray, (…, M, M) matrix}
            Mixing matrix (spatial filters are stored in columns).
    eigenvalues : (…, M) numpy.ndarray
        Eigenvalues of w.
    patterns : numpy.ndarray
        De-mixing matrix (activation patterns are stored in columns).
    """

    def __init__(self, n_filters=4):
        """
        n_filter : int or None
            Number of most discriminant CSP filters to decompose the signal
            into (must be less or equal to the number of channels in your
            signal).
            If int it will return that number of filters.
            If None it will return all calculated filters."""

        self.filters = None  # Mixing matrix (spatial filters)
        self.eigenvalues = None  # Eigenvalues
        self.patterns = None  # De-mixing matrix
        self.n_filters = n_filters

    def fit(self, X, y):
        """Train Common Spatial Patterns (CSP) filters

            Train Common Spatial Patterns (CSP) filters with support to >2
            classes based on support multiclass CSP by means of approximate
            joint diagonalization. In this case, the spatial filter selection
            is achieved according to [1].

            Parameters
            ----------
            X : numpy.ndarray, [n_trials, samples, channels]
                Epoched data of shape (n_trials, samples, channels)

            y : numpy.ndarray, [n_trials,]
                Labels for epoched data of shape (n_trials,)

            References
            ----------
            [1] Grosse-Wentrup, Moritz, and Martin Buss. "Multiclass common
            spatial patterns and information theoretic feature extraction."
            Biomedical Engineering, IEEE Transactions on 55, no. 8 (2008):
            1991-2000.
        """
        n_classes = np.unique(y)
        # Covariance matrices
        cov = []
        for c in n_classes:
            cov.append(np.cov(X[y == c].reshape(-1, X.shape[-1]).T))
        cov = np.array(cov)
        # Classic implementation for 2-class
        if len(n_classes) == 2:
            # Solve the eigenvalue problem
            eigenvalues, filters = slinalg.eigh(cov[0], cov[0] + cov[1])

            # Indexes for sorting eigenvectors (w)
            ix_sorted = np.argsort(np.abs(eigenvalues - 0.5))[::-1]
        # Implementation for >2-classes
        elif len(n_classes) > 2:
            # Approximate joint diagonalization based on jacobi angle
            filters, d = self.adj_pham(x=cov)
            filters = filters.T
            # Normalization
            # Mean covariance
            cmean = np.average(cov, axis=0)
            for i in range(filters.shape[1]):
                temp = np.dot(np.dot(filters[:, i].T, cmean), filters[:, i])
                filters[:, i] /= np.sqrt(temp)
            # We calculate the probability of each class
            info = []
            prob_class = [np.mean(y == c) for c in n_classes]
            for j in range(filters.shape[1]):
                patterns, b = 0, 0
                for i, c in enumerate(n_classes):
                    temp = np.dot(np.dot(filters[:, j].T, cov[i]), filters[:, j])
                    patterns += prob_class[i] * np.log(np.sqrt(temp))
                    b += prob_class[i] * (temp ** 2 - 1)
                mutual_info = - (patterns + (3.0 / 16) * (b ** 2))
                info.append(mutual_info)
            eigenvalues = info
            # Indexes for sorting eigenvectors (w)
            ix_sorted = np.argsort(info)[::-1]
        else:
            raise ValueError("Number of classes must be  >= 2")
        # Sort eigenvectors (w)
        filters = filters[:, ix_sorted]
        eigenvalues = [eigenvalues[i] for i in ix_sorted]
        eigenvalues = np.diag(eigenvalues)
        # Activation spatial patterns
        # patterns = slinalg.pinv2(filters).T   # deprecated and removed in scipy 1.9.0
        patterns = slinalg.pinv(filters).T

        # Attribute storing of number of filters
        self.filters = filters[:, :self.n_filters].T
        self.eigenvalues = eigenvalues
        self.patterns = patterns[:, :self.n_filters].T

    def project(self, X):
        """ This method projects the input data X with the spatial filters W

        Parameters
        ----------
        X : numpy.ndarray [n_trials, samples, channels]
            Input data (dimensions [n_trials, samples, channels])

        Returns
        -------
        numpy.ndarray [n_trials, n_filters, samples]
            Array with the epochs of signal projected in the CSP space
        """
        if self.filters is None:
            raise Exception("CSP must be fitted first")
        return np.matmul(self.filters, X.transpose((0, 2, 1)))

    def adj_pham(self, x, eps=1e-6, n_iter_max=15):
        """Approximate joint diagonalization based on pham's algorithm.
            This is a direct implementation of the PHAM's AJD algorithm [1].
            Extracted from pyriemann module:
            http://github.com/alexandrebarachant/pyRiemann

            Parameters
            ----------
            x : ndarray, shape (n_trials, n_channels, n_channels)
                A set of covariance matrices to diagonalize
            eps : float (default 1e-6)
                tolerance for stopping criterion.
            n_iter_max : int (default 15)
                The maximum number of iteration to reach convergence.

            Returns
            -------
            v : numpy.ndarray, [n_channels, n_channels]
                Diagonalizer
            d : numpy.ndarray, [n_trials, n_channels, n_channels]
                Set of quasi diagonal matrices

            References
            ----------
            [1] Pham, Dinh Tuan. "Joint approximate diagonalization of positive
            definite Hermitian matrices." SIAM Journal on Matrix Analysis and
            Applications 22, no. 4 (2001): 1136-1152.
            """
        n_epochs = x.shape[0]

        # Reshape input matrix
        a = np.concatenate(x, axis=0).T

        # Init variables
        n_times, n_m = a.shape
        v = np.eye(n_times)
        epsilon = n_times * (n_times - 1) * eps

        for it in range(n_iter_max):
            decr = 0
            for ii in range(1, n_times):
                for jj in range(ii):
                    Ii = np.arange(ii, n_m, n_times)
                    Ij = np.arange(jj, n_m, n_times)

                    c1 = a[ii, Ii]
                    c2 = a[jj, Ij]

                    g12 = np.mean(a[ii, Ij] / c1)
                    g21 = np.mean(a[ii, Ij] / c2)

                    omega21 = np.mean(c1 / c2)
                    omega12 = np.mean(c2 / c1)
                    omega = np.sqrt(omega12 * omega21)

                    tmp = np.sqrt(omega21 / omega12)
                    tmp1 = (tmp * g12 + g21) / (omega + 1)
                    tmp2 = (tmp * g12 - g21) / max(omega - 1, 1e-9)

                    h12 = tmp1 + tmp2
                    h21 = np.conj((tmp1 - tmp2) / tmp)

                    decr += n_epochs * (g12 * np.conj(h12) + g21 * h21) / 2.0

                    tmp = 1 + 1.j * 0.5 * np.imag(h12 * h21)
                    tmp = np.real(tmp + np.sqrt(tmp ** 2 - h12 * h21))
                    tau = np.array([[1, -h12 / tmp], [-h21 / tmp, 1]])

                    a[[ii, jj], :] = np.dot(tau, a[[ii, jj], :])
                    tmp = np.c_[a[:, Ii], a[:, Ij]]
                    tmp = np.reshape(tmp, (n_times * n_epochs, 2), order='F')
                    tmp = np.dot(tmp, tau.T)

                    tmp = np.reshape(tmp, (n_times, n_epochs * 2), order='F')
                    a[:, Ii] = tmp[:, :n_epochs]
                    a[:, Ij] = tmp[:, n_epochs:]
                    v[[ii, jj], :] = np.dot(tau, v[[ii, jj], :])
            if decr < epsilon:
                break
        d = np.reshape(a, (n_times, -1, n_times)).transpose(1, 0, 2)
        return v, d

    def to_dict(self):
        return self.__dict__

    @staticmethod
    def from_dict(dict_data):
        csp = CSP()
        csp.filters = dict_data['filters']
        csp.eigenvalues = dict_data['eigenvalues']
        csp.patterns = dict_data['patterns']
        csp.n_filters = dict_data['n_filters']
        return csp


class CCA(components.ProcessingMethod):
    """
    The class CCA performs a Canonical Correlation Analysis filtering. First,
    function fit() sould be called to train the spatial filters. Then, spatial
    filters could be used to project testing data. After fit(), the following
    attributes are computed:

    Attributes
    ----------
    wx : {(channels, no_filters) ndarray}
        Mixing matrix for projecting the data, where spatial filters are stored
        in columns.
    wy : {(channels, no_filters) ndarray}
        Mixing matrix for projecting the reference, where spatial filters are
        stored in columns.
    r : {(channels, ) ndarray}
        Sample canonical correlations.
    """

    def __init__(self):
        self.wx = None
        self.wy = None
        self.r = None

    def fit(self, x, y):
        """
        Fits the CCA spatial filters given the two input matrices data and
        reference, storing relevant parameters.
        Parameters
        ----------
        x : {(samples, channels) ndarray}
            First input matrix, usually data with concatenated epochs.
        y : {(samples, channels) ndarray}
            Second input matrix, usually the data reference. The number of
            samples must match the samples of the data matrix. Repeat the
            reference if necessary before calling this function.
        """
        [self.wx, self.wy, self.r] = self.canoncorr(x, y)

    def project(self, data, filter_idx=(0), projection='wy'):
        """
        Projects the input data matrix using the given spatial filter. Note that
        the spatial filter will have dimensions [no. channels x no. channels].

        Parameters
        ----------
        data : {(samples, channels) ndarray}
            Testing data matrix. The number of channels must be the same as the
            no. channels used to train.
        filter_idx : {int}, optional
            Indexes of the spatial filters to be used. Since filters are sorted by
            their importance in terms of correlation, the default filter (0) is
            the one that yields highest correlation.
        projection: {str}, optional
            Canonical coefficients to be used in projection. By default, the
            function uses Wy. Typically, if data is averaged, Wy must be used;
            otherwise, if data is not averaged and just concatenated, Wx must
            be used.
        Returns
        -------
        projected_data : {(samples, ) ndarray}
            Projected data along the selected spatial filter.
        """
        if projection.lower() == 'wy':
            return np.matmul(data, self.wy[:, filter_idx])
        elif projection.lower() == 'wx':
            return np.matmul(data, self.wx[:, filter_idx])
        else:
            raise Exception('[CCA] Unknown projection %s' % str(projection))

    @staticmethod
    def canoncorr(X, Y):
        """
        Computes the canonical correlation analysis (CCA) for the data matrices
        X (dimensions N-by-P1) and Y (dimensions N-by-P2). X and Y must have the
        same number of observations (rows) but can have different numbers of
        variables (cols). The j-th columns of A and B contain the canonial
        coefficients, i.e. the linear combination of variables making up the
        j-th canoncial variable for X and Y, respectively. If X or Y are less
        than full rank, canoncorr gives a warning and returns zeros in the rows
        of A or B corresponding to dependent columns of X or Y. Final dimension
        D is computed as D = min(rank_X, rank_Y).
        Notes
        ----------
        This method is adapted from the MATLAB function 'canoncorr'. Check
        that file in case of conflicts, doubts or additional information.
        Parameters
        ----------
        X : {(N, P1) ndarray}
            Input matrix with dimensions N-by-P1. Rows are observations and cols
            are variables.
        Y : {(N, P2) ndarray}
            Input matrix with dimensions N-by-P1. Rows are observations and cols
            are variables.
        Returns
        -------
        A : {(P1, D) ndarray}
            Sample canonical coefficients for the variables in X. The j-th
            column of A contains the linear combination of variables that makes
            up the j-th canonical variable for X. If X is less than full rank,
            A will have zeros in the rows corresponding to dependent cols of X.
        B : {(P2, D) ndarray}
            Sample canonical coefficients for the variables in Y. The j-th
            column of B contains the linear combination of variables that makes
            up the j-th canonical variable for Y. If Y is less than full rank,
            B will have zeros in the rows corresponding to dependent cols of Y.
        r : {(D,) ndarray}
            Sample canonical correlations. The j-th element of r is the
            correlation between the h-th columns of the canonical scores for
            the variables in X and Y.
        References
        -------
        [1] Krzanowski, W.J., Principles of Multivariate Analysis,
        Oxford University Press, Oxford, 1988.
        [2] Seber, G.A.F., Multivariate Observations, Wiley, New York, 1984.
        Example
        --------
        >>> import numpy as np
        >>> X = np.random.rand(10, 4)
        >>> Y = np.random.rand(10, 4)
        >>> A, B, r = canoncorr(X, Y)
        """
        # X dims
        if len(X.shape) == 1:
            n = X.shape[0]
            p1 = 1
            X = X.reshape(n, 1)
        else:
            [n, p1] = X.shape

        # Check the input size
        if Y.shape[0] != n:
            raise ValueError('[canoncorr] Input size mismatch')
        elif n == 1:
            raise ValueError('[canoncorr] Not enough data')

        # Y dims
        if len(Y.shape) == 1:
            p2 = 1
            Y = Y.reshape(n, 1)
        else:
            p2 = Y.shape[1]

        # Center the variables
        X = X - np.mean(X, 0)
        Y = Y - np.mean(Y, 0)

        # Factor the inputs and find a full rank set of columns if necessary
        [Q1, T11, perm1] = slinalg.qr(X, mode='economic', pivoting=True)
        rankX = nlinalg.matrix_rank(T11)
        if rankX == 0:
            raise ValueError('[canoncorr] Rank of X is 0. Invalid Data')
        elif rankX < p1:
            warnings.warn('[canoncorr] Data X is not full rank')
            Q1 = Q1[:, 0:rankX]
            T11 = T11[0:rankX, 0:rankX]
        [Q2, T22, perm2] = slinalg.qr(Y, mode='economic', pivoting=True)
        rankY = nlinalg.matrix_rank(T22)
        if rankY == 0:
            raise ValueError('[canoncorr] Rank of Y is 0. Invalid Data')
        elif rankY < p2:
            warnings.warn('[canoncorr] Data Y is not full rank')
            Q2 = Q2[:, 0:rankY]
            T22 = T22[0:rankY, 0:rankY]

        # Compute canonical coefficients and canonical correlations. For
        # rankX > rank Y, the economy-size version ignores the extra columns
        # in L and rows in D. For rankX < rankY, need to ignore extra columns
        # in M and explicitly. Normalize A and B to give U and V unit variance
        d = np.min([rankX, rankY])
        [L, D, M] = nlinalg.svd(np.matmul(Q1.T, Q2))
        M = M.T
        A = nlinalg.lstsq(T11, L[:, 0:d], rcond=None)[0] * np.sqrt(n - 1)
        B = nlinalg.lstsq(T22, M[:, 0:d], rcond=None)[0] * np.sqrt(n - 1)
        r = D[0:d]
        r[r < 0] = 0  # remove roundoff errors
        r[r > 1] = 1

        # Put coefficients back to their full size and their correct order
        A = np.concatenate((A, np.zeros((p1 - rankX, d))), axis=0)
        A = A[np.argsort(perm1), :]
        B = np.concatenate((B, np.zeros((p2 - rankY, d))), axis=0)
        B = B[np.argsort(perm2), :]

        return A, B, r

    def to_dict(self):
        return self.__dict__

    @staticmethod
    def from_dict(dict_data):
        cca = CCA()
        cca.wx = dict_data['wx']
        cca.wy = dict_data['wy']
        cca.r = dict_data['r']
        return cca
