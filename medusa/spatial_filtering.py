import warnings
from copy import copy
import numpy as np
from numpy import linalg as nlinalg
from scipy import linalg as slinalg
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from medusa import components
from medusa.plots.head_plots import TopographicPlot

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
    """ Common Spatial Pattern filtering.

    Attributes
    ----------
    filters : {(…, M, M) numpy.ndarray, (…, M, M) matrix}
            Mixing matrix (spatial filters are stored in columns).
    eigenvalues : (…, M) numpy.ndarray
        Eigenvalues of w.
    patterns : numpy.ndarray
        De-mixing matrix (activation patterns are stored in columns).
    """

    def __init__(self, n_filters=4, selection="extremes"):
        """

        Parameters
        ----------
        n_filters : int or None
            Number of filters to select. Use None to return all filters
            (default: 4).
        selection : basestring
            Selection method:
            - "extremes" (default): classic method that takes the filters
            from the extremes, which belong to both classes separately. This
            method cannot be applied in problems with more than 2 classes.
            - "eigenvalues": the eigenvalues are sorted and the highest ones
            are eligible to be selected.

        Attributes
        ----------
        filters: np.ndarray (n_channels, n_channels)
            Mixing matrix, or forward model (spatial filters are stored in
            rows).
        patterns: np.ndarray (n_channels, n_channels)
            De-mixing matrix, or backward model (patterns are stored in rows).
        eigenvalues: np.ndarray (n_channels, )
            Eigenvalues associated to each filter and pattern.
        sel_idxs: np.ndarray (n_filters, )
            Selected indexes to get the desired filters and patterns.
        sel_filters: np.ndarray (n_filters, n_channels)
            Selected spatial filters (stored in rows).
        sel_patterns: np.ndarray (n_filters, n_channels)
            Selected patterns (stored in rows).
        sel_eigenvalues: np.ndarray (n_filters, )
            Selected eigenvalues.
        """
        self.n_filters = n_filters      # Number of filters to choose
        self.selection = selection      # Selection method
        if self.n_filters is None:
            self.selection = "none"

        # Generated
        self.filters = None             # Mixing matrix (spatial filters)
        self.patterns = None            # De-mixing matrix
        self.eigenvalues = None         # Eigenvalues
        self.sel_idxs = None            # Selected indexes
        self.sel_filters = None         # Selected spatial filters
        self.sel_patterns = None        # Selected patterns
        self.sel_eigenvalues = None     # Selected eigenvalues
        self.is_fitted = False

    def fit(self, X, y):
        """ Method to train the CSP.

        This code is based on [1] for the 2-class problem, and based on [2]
        for the > 2-class problem.

        Parameters
        ----------
        X : numpy.ndarray (n_epochs, n_samples, n_channels)
            Epoched data of shape (n_epochs, n_samples, n_channels)

        y : numpy.ndarray (n_epochs, )
            Labels for epoched data of shape (n_epochs, )

        References
        ----------
        [1]  Blankertz, B., Tomioka, R., Lemm, S., Kawanabe, M., & Muller,
        K. R. (2007). Optimizing spatial filters for robust EEG single-trial
        analysis. IEEE Signal processing magazine, 25(1), 41-56.
        [2] Grosse-Wentrup, Moritz, and Martin Buss. "Multiclass common
        spatial patterns and information theoretic feature extraction."
        Biomedical Engineering, IEEE Transactions on 55, no. 8 (2008):
        1991-2000.
        """
        # Error detection
        n_classes = np.unique(y)
        if len(n_classes) > 2 and self.selection == "extremes":
            raise ValueError("Cannot use 'extremes' selection if the data has "
                             "more than 2 classes (%i classess found)!"
                             % len(n_classes))

        # Covariance matrices
        cov = []
        for c in n_classes:
            cov.append(np.cov(X[y == c].reshape(-1, X.shape[-1]).T))
        cov = np.array(cov)  # dimensions [n_classes x n_cha x n_cha]

        # Classic implementation for 2 classes
        if len(n_classes) == 2:
            # Solve the eigenvalue problem
            self.eigenvalues, eigenvectors = slinalg.eigh(cov[0], cov.sum(0))

            # Indexes for sorting eigenvectors (w)
            if self.selection == "eigenvalues":
                # Automatic sorting using eigenvalues
                self.sel_idxs = np.argsort(np.abs(self.eigenvalues - 0.5))[::-1]
                self.sel_idxs = self.sel_idxs[:self.n_filters]
            # if self.selection == "ratio-of-means":
            #     proj0 = [np.dot(eigenvectors.T, trial.T) for trial in X[
            #         y == n_classes[0]]]
            #     proj0 = np.transpose(np.array(proj0), (0, 2, 1))
            #     # epochs x filters x channels
            if self.selection == "extremes":
                # Automatic selection using extremes for both classes
                self.sel_idxs = list()
                ids = np.arange(len(self.eigenvalues)).tolist()
                start = False
                while len(self.sel_idxs) < self.n_filters:
                    if start:
                        self.sel_idxs.append(ids.pop(0))
                    else:
                        self.sel_idxs.append((ids.pop(len(ids) - 1)))
                    start = not start

        # Implementation for more than 2 classes
        elif len(n_classes) > 2:
            # Approximate joint diagonalization based on jacobi angle
            filters, d = self._adj_pham(x=cov)
            filters = filters.T
            # Normalization
            # Mean covariance
            cmean = np.average(cov, axis=0)
            for i in range(filters.shape[1]):
                temp = np.dot(np.dot(filters[:, i].T, cmean), filters[:, i])
                filters[:, i] /= np.sqrt(temp)
            # We calculate the probability of each class
            self.eigenvalues = []
            prob_class = [np.mean(y == c) for c in n_classes]
            for j in range(filters.shape[1]):
                patterns, b = 0, 0
                for i, c in enumerate(n_classes):
                    temp = np.dot(np.dot(filters[:, j].T, cov[i]),
                                  filters[:, j])
                    patterns += prob_class[i] * np.log(np.sqrt(temp))
                    b += prob_class[i] * (temp ** 2 - 1)
                mutual_info = - (patterns + (3.0 / 16) * (b ** 2))
                self.eigenvalues.append(mutual_info)

            # Indexes for sorting eigenvalues (w)
            if self.selection == "eigenvalues":
                self.sel_idxs = np.argsort(self.eigenvalues)[::-1]
                self.sel_idxs = self.sel_idxs[:self.n_filters]
        else:
            raise ValueError("Number of classes must be  >= 2")

        # Get all the spatial filters, patterns and eigenvalues (non-sorted)
        self.filters = eigenvectors.T
        self.patterns = slinalg.pinv(eigenvectors)
        self.eigenvalues = np.array(self.eigenvalues)

        # Get the selected spatial filters, patterns and eigenvalues
        if self.sel_idxs is not None:
            self.sel_filters = self.filters[self.sel_idxs, :]
            self.sel_patterns = self.patterns[self.sel_idxs, :]
            self.sel_eigenvalues = self.eigenvalues[self.sel_idxs]
        else:
            self.sel_filters = self.filters
            self.sel_patterns = self.patterns
            self.sel_eigenvalues = self.eigenvalues
        self.is_fitted = True

    def project(self, X):
        """ Projects the input data X with the selected spatial filters.

        Parameters
        ----------
        X : numpy.ndarray (n_epochs, n_samples, n_channels)
            Epoched data of shape (n_epochs, n_samples, n_channels).

        Returns
        -------
        numpy.ndarray (n_epochs, n_filters, n_channels)
            Array with the epochs of signal projected in the CSP space.
        """
        if len(X.shape) != 3:
            raise Exception("X must be 3-dimensional (n_epochs x n_samples x "
                            "n_channels!")
        if not self.is_fitted:
            raise Exception("CSP must be fitted first")

        # Project each trial separately
        projection = [np.dot(self.sel_filters, trial.T) for trial in X]
        projection = np.transpose(np.array(projection), (0, 2, 1))
        return projection

    @staticmethod
    def _adj_pham(x, eps=1e-6, n_iter_max=15):
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
        dict_ = copy(self.__dict__)
        for key, value in dict_.items():
            if isinstance(value, np.ndarray):
                dict_[key] = value.tolist()
        return dict_

    @staticmethod
    def from_dict(dict_data):
        csp = CSP()
        csp.n_filters = dict_data['n_filters']
        csp.selection = dict_data['selection']
        csp.filters = np.array(dict_data['filters'])
        csp.patterns = np.array(dict_data['patterns'])
        csp.eigenvalues = np.array(dict_data['eigenvalues'])
        csp.sel_idxs = np.array(dict_data['sel_idxs'])
        csp.sel_filters = np.array(dict_data['sel_filters'])
        csp.sel_patterns = np.array(dict_data['sel_patterns'])
        csp.sel_eigenvalues = np.array(dict_data['sel_eigenvalues'])
        return csp

    def plot(self, channel_set, figure=None, plot_filters=False,
             plot_patterns=True, topo_settings=None, show=False,
             plot_eig=True, only_selected=True):
        # Error detection and initialization
        if not plot_patterns and not plot_filters:
            raise Exception("Cannot plot CSP if plot_filters and "
                            "plot_patterns are both None")
        if figure is None:
            figure = plt.figure(figsize=(7.5, 3), dpi=300)
        if len(channel_set.l_cha) != self.sel_filters.shape[1]:
            raise Exception("The number of channels (%i) must be the same as "
                            "the number of channels used to train the CSP ("
                            "%i)" % (len(channel_set.l_cha),
                                     self.sel_filters.shape[1]))
        if topo_settings is None:
            topo_settings = {
                "head_radius": 1.0,
                "head_line_width": 2,
                "interp_contour_width": 1,
                "interp_points": 500,
            }
        if only_selected:
            sel_patterns = self.sel_patterns
            sel_filters = self.sel_filters
            sel_eigenvalues = self.sel_eigenvalues
        else:
            sel_patterns = self.patterns
            sel_filters = self.filters
            sel_eigenvalues = self.eigenvalues

        # Parameters
        n_row = 2 if plot_patterns and plot_filters else 1
        max_f = 0
        max_p = 0
        for i in range(sel_filters.shape[0]):
            if np.max(np.abs(sel_patterns[i, :])) > max_p:
                max_p = np.max(np.abs(sel_patterns[i, :]))
            if np.max(np.abs(sel_filters[i, :])) > max_f:
                max_f = np.max(np.abs(sel_filters[i, :]))

        # Plot filters
        j = 0
        if plot_filters:
            for j in range(sel_filters.shape[0]):
                ax = figure.add_subplot(n_row, sel_filters.shape[0], j + 1)
                topo_settings["clim"] = (-max_f, max_f)
                topo_settings["cmap"] = "RdBu"
                topo = TopographicPlot(axes=ax, channel_set=channel_set,
                                       **topo_settings)
                topo.update(values=sel_filters[j, :])
                ax.set_title("Filter %i" % j)
                if plot_eig and not plot_patterns:
                    ax.set_xlabel('Eig: %.3f' % sel_eigenvalues[j])
                # Colorbar
                if j == sel_filters.shape[0] - 1:
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes('right', size='5%', pad=0.05)
                    cbar = figure.colorbar(
                        topo.plot_handles["color-mesh"], cax=cax,
                        orientation='vertical')
            j += 1

        # Plot patterns
        if plot_patterns:
            for i in range(sel_filters.shape[0]):
                ax = figure.add_subplot(n_row, sel_filters.shape[0], j + i + 1)
                topo_settings["clim"] = (-max_p, max_p)
                topo_settings["cmap"] = "PiYG"
                topo = TopographicPlot(axes=ax, channel_set=channel_set,
                                       **topo_settings)
                topo.update(values=sel_patterns[i, :])
                ax.set_title("Pattern %i" % i)
                if plot_eig:
                    ax.set_xlabel('Eig: %.3f' % sel_eigenvalues[i])
                # Colorbar
                if i == sel_filters.shape[0] - 1:
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes('right', size='5%', pad=0.05)
                    cbar = figure.colorbar(
                        topo.plot_handles["color-mesh"], cax=cax,
                        orientation='vertical')
        plt.suptitle("CSP")

        # Show?
        if show:
            plt.show()
        return figure


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
