import warnings
from copy import copy
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy import linalg as nlinalg
from numpy.typing import NDArray
from scipy import linalg as slinalg

if TYPE_CHECKING:
    from medusa.core.legacy.biosignals.eeg.eeg import EEGChannelSet
    from medusa.plots import TopographicPlot

__all__ = [
    "LaplacianFilter",
    "car",
    "CSP",
    "CCA",
    "TRCA",
]


class LaplacianFilter:
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

    def __init__(self, channel_set: "EEGChannelSet", mode: str = 'auto') -> None:
        """Constructor of class LaplacianFilter.

        Parameters
        ----------
        channel_set :
            Initialised channel set defining the montage. Must expose the
            ``channels``, ``l_cha`` and ``n_cha`` attributes and the
            ``compute_dist_matrix`` method (the legacy ``EEGChannelSet``).
        mode :
            ``'auto'`` (nearest-neighbour selection) or ``'custom'`` (the
            Laplacian neighbours of every channel are supplied explicitly to
            :meth:`fit_lp`).

        Raises
        ------
        ValueError
            If ``channel_set`` has no channels, or if it has fewer than five
            channels (the minimum required to estimate a Laplacian surface).
        """
        # Check Channel Set is initialized
        if not channel_set.channels:
            raise ValueError('Cannot compute the nearest neighbors if channel '
                             'set is not initialized!')

        # Chech if there are enough channels to perform filtering
        if len(channel_set.l_cha) < 5:
            raise ValueError('There are not enough channels to perform '
                             'Laplacian surface filtering.')
        # Parameters
        self.channel_set = channel_set
        self.l_cha = channel_set.l_cha
        self.n_channels = channel_set.n_cha
        self.mode = mode

        # Variables
        self.dist_matrix = None
        self.lp_filter = []
        self.idx_cha_to_filter = None
        self.dist_weights = []

    def fit_lp(
        self,
        l_cha_to_filter: list[str],
        l_cha_laplace: list[list[str]] | None = None,
    ) -> None:
        """Fit the Laplacian filter depending on the mode chosen.

        Parameters
        ----------
        l_cha_to_filter :
            ``(n_channels_to_filter,)`` labels of the channels to filter. Used
            in both filtering modes.
        l_cha_laplace :
            ``(n_channels_to_filter, n_neighbours)`` labels of the channels
            used to compute the Laplacian surface of each channel in
            ``l_cha_to_filter``. Only used in ``'custom'`` mode.

        Raises
        ------
        ValueError
            In ``'custom'`` mode, if ``l_cha_to_filter`` or ``l_cha_laplace``
            is ``None``, or if their lengths do not match.
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

    def apply_lp(self, signal: NDArray) -> NDArray:
        """Apply the fitted Laplacian filter to an EEG signal.

        Parameters
        ----------
        signal :
            ``(n_samples, n_channels)`` EEG signal. The channel axis must
            match the montage passed to the constructor.

        Returns
        -------
        NDArray
            ``(n_samples, n_channels_to_filter)`` Laplacian-filtered signal,
            one column per channel in ``l_cha_to_filter``.

        Raises
        ------
        ValueError
            If the channel axis of ``signal`` does not match the number of
            channels in the montage.

        Examples
        --------
        >>> import numpy as np
        >>> from medusa.core.legacy.biosignals.eeg.eeg import EEGChannelSet
        >>> from medusa.signal.spatial_filtering import LaplacianFilter
        >>> channel_set = EEGChannelSet()
        >>> channel_set.set_standard_montage(
        ...     ['Fz', 'Cz', 'Pz', 'C3', 'C4', 'P3', 'P4', 'Oz'],
        ...     montage='10-05')
        >>> lp = LaplacianFilter(channel_set, mode='auto')
        >>> lp.fit_lp(['CZ'])  # set_standard_montage upper-cases the labels
        >>> signal = np.random.default_rng(0).standard_normal((100, 8))
        >>> lp.apply_lp(signal).shape
        (100, 1)
        """
        signal = np.asarray(signal)
        # Check dimensions
        if signal.shape[1] != len(self.l_cha):
            raise ValueError('Dimensions of signal in axis 1 must match the '
                             'number of channels')

        s_filtered = np.empty((signal.shape[0], len(self.idx_cha_to_filter)))

        for i, index in enumerate(self.idx_cha_to_filter.astype(int)):
            s_filtered[:,i] = signal[:, index] - np.average(signal[:,
                                                        self.lp_filter[i].astype(int)],
                                                        weights=self.dist_weights[i],
                                                            axis=1)
        return s_filtered


def car(signal: NDArray) -> NDArray:
    """Apply the Common Average Reference (CAR) spatial filter.

    Subtracts the per-sample mean across channels from every channel:

    .. math::

        X_\\mathrm{CAR}[s, c] = X[s, c] - \\frac{1}{N} \\sum_{k=0}^{N-1} X[s, k]

    where ``N = n_channels``. Equivalent to the matrix form
    ``X_CAR = X @ M`` with ``M = I - (1/N) 1 1^T``.

    Parameters
    ----------
    signal :
        Either:

        - 3-D ``(n_segments, n_samples, n_channels)`` (canonical K9.B
          time-domain shape).
        - 2-D ``(n_samples, n_channels)`` for streaming / single-segment
          callers; output has the same 2-D shape.

        The TRIGGER channel, if any, must be removed before calling.

    Returns
    -------
    NDArray
        Same shape as the input. Output dtype matches the input dtype after
        the subtraction (integer inputs are promoted by NumPy's usual rules).

    Raises
    ------
    ValueError
        If ``signal`` is not 2-D or 3-D.

    Examples
    --------
    >>> import numpy as np
    >>> from medusa.signal.spatial_filtering import car
    >>> sig = np.random.default_rng(0).standard_normal((1, 1000, 8))
    >>> car(sig).shape
    (1, 1000, 8)
    >>> # Single-segment / streaming form preserves the 2-D shape:
    >>> car(np.zeros((1000, 8))).shape
    (1000, 8)
    """
    arr = np.asarray(signal)
    if arr.ndim not in (2, 3):
        raise ValueError(
            f"car expects a 2-D (n_samples, n_channels) or 3-D "
            f"(n_segments, n_samples, n_channels) array; got "
            f"shape {arr.shape}."
        )
    if arr.shape[-1] <= 1:
        return arr
    return arr - arr.mean(axis=-1, keepdims=True)


class CSP:
    """ Common Spatial Pattern filtering.

    Attributes
    ----------
    filters : {(…, M, M) numpy.ndarray, (…, M, M) matrix}
            Mixing matrix (spatial filters are stored in columns).
    eigenvalues : (…, M) numpy.ndarray
        Eigenvalues of w.
    patterns : numpy.ndarray
        De-mixing matrix (activation patterns are stored in columns).

    Examples
    --------
    >>> import numpy as np
    >>> from medusa.signal.spatial_filtering import CSP
    >>> rng = np.random.default_rng(0)
    >>> signal = rng.standard_normal((20, 200, 4))
    >>> labels = np.array([0, 1] * 10)
    >>> csp = CSP(n_filters=2)
    >>> csp.fit(signal, labels)
    >>> csp.project(signal).shape
    (20, 200, 2)
    """

    def __init__(
        self,
        n_filters: int | None = 4,
        selection: str = "extremes",
    ) -> None:
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

    def fit(self, signal: NDArray, labels: NDArray) -> None:
        """ Method to train the CSP.

        This code is based on [1] for the 2-class problem, and based on [2]
        for the > 2-class problem.

        Parameters
        ----------
        signal :
            ``(n_segments, n_samples, n_channels)`` segmented data.
        labels :
            ``(n_segments,)`` class label of every segment.

        Raises
        ------
        ValueError
            If ``selection='extremes'`` is requested with more than two
            classes, or if fewer than two classes are present.

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
        signal = np.asarray(signal)
        labels = np.asarray(labels)
        # Error detection
        n_classes = np.unique(labels)
        if len(n_classes) > 2 and self.selection == "extremes":
            raise ValueError("Cannot use 'extremes' selection if the data has "
                             f"more than 2 classes ({len(n_classes)} classes "
                             "found)!")

        # Covariance matrices
        cov = []
        for c in n_classes:
            cov.append(np.cov(signal[labels == c].reshape(
                -1, signal.shape[-1]).T))
        cov = np.asarray(cov)  # dimensions [n_classes x n_channels x n_channels]

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
            #     proj0 = [np.dot(eigenvectors.T, trial.T) for trial in signal[
            #         labels == n_classes[0]]]
            #     proj0 = np.transpose(np.asarray(proj0), (0, 2, 1))
            #     # segments x filters x channels
            if self.selection == "extremes":
                # Automatic selection using extremes for both classes
                self.sel_idxs = list()
                ids = np.arange(len(self.eigenvalues)).tolist()
                start = False
                while len(self.sel_idxs) < self.n_filters:
                    if start:
                        self.sel_idxs.append(ids.pop(0))
                    else:
                        self.sel_idxs.append(ids.pop(len(ids) - 1))
                    start = not start

        # Implementation for more than 2 classes
        elif len(n_classes) > 2:
            # Approximate joint diagonalization based on jacobi angle
            filters, d = self._adj_pham(cov_set=cov)
            filters = filters.T
            # Normalization
            # Mean covariance
            cmean = np.average(cov, axis=0)
            for i in range(filters.shape[1]):
                temp = np.dot(np.dot(filters[:, i].T, cmean), filters[:, i])
                filters[:, i] /= np.sqrt(temp)
            # We calculate the probability of each class
            self.eigenvalues = []
            prob_class = [np.mean(labels == c) for c in n_classes]
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
        self.eigenvalues = np.asarray(self.eigenvalues)

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

    def project(self, signal: NDArray) -> NDArray:
        """ Projects the input ``signal`` with the selected spatial filters.

        Parameters
        ----------
        signal :
            ``(n_segments, n_samples, n_channels)`` segmented data.

        Returns
        -------
        NDArray
            ``(n_segments, n_samples, n_filters)`` segments projected onto the
            CSP space.

        Raises
        ------
        ValueError
            If ``signal`` is not 3-dimensional, or if the CSP has not been
            fitted yet.
        """
        signal = np.asarray(signal)
        if len(signal.shape) != 3:
            raise ValueError("signal must be 3-dimensional (n_segments x "
                             "n_samples x n_channels)!")
        if not self.is_fitted:
            raise ValueError("CSP must be fitted first")

        # Project each segment separately
        projection = [np.dot(self.sel_filters, segment.T) for segment in signal]
        projection = np.transpose(np.asarray(projection), (0, 2, 1))
        return projection

    @staticmethod
    def _adj_pham(
        cov_set: NDArray,
        eps: float = 1e-6,
        n_iter_max: int = 15,
    ) -> tuple[NDArray, NDArray]:
        """Approximate joint diagonalization based on pham's algorithm.
            This is a direct implementation of the PHAM's AJD algorithm [1].
            Extracted from pyriemann module:
            http://github.com/alexandrebarachant/pyRiemann

            Parameters
            ----------
            cov_set :
                ``(n_segments, n_channels, n_channels)`` set of covariance
                matrices to diagonalize.
            eps :
                Tolerance for the stopping criterion.
            n_iter_max :
                The maximum number of iterations to reach convergence.

            Returns
            -------
            v : numpy.ndarray, [n_channels, n_channels]
                Diagonalizer
            d : numpy.ndarray, [n_segments, n_channels, n_channels]
                Set of quasi diagonal matrices

            References
            ----------
            [1] Pham, Dinh Tuan. "Joint approximate diagonalization of positive
            definite Hermitian matrices." SIAM Journal on Matrix Analysis and
            Applications 22, no. 4 (2001): 1136-1152.
            """
        n_segments = cov_set.shape[0]

        # Reshape input matrix
        a = np.concatenate(cov_set, axis=0).T

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

                    decr += n_segments * (g12 * np.conj(h12) + g21 * h21) / 2.0

                    tmp = 1 + 1.j * 0.5 * np.imag(h12 * h21)
                    tmp = np.real(tmp + np.sqrt(tmp ** 2 - h12 * h21))
                    tau = np.array([[1, -h12 / tmp], [-h21 / tmp, 1]])

                    a[[ii, jj], :] = np.dot(tau, a[[ii, jj], :])
                    tmp = np.c_[a[:, Ii], a[:, Ij]]
                    tmp = np.reshape(tmp, (n_times * n_segments, 2), order='F')
                    tmp = np.dot(tmp, tau.T)

                    tmp = np.reshape(tmp, (n_times, n_segments * 2), order='F')
                    a[:, Ii] = tmp[:, :n_segments]
                    a[:, Ij] = tmp[:, n_segments:]
                    v[[ii, jj], :] = np.dot(tau, v[[ii, jj], :])
            if decr < epsilon:
                break
        d = np.reshape(a, (n_times, -1, n_times)).transpose(1, 0, 2)
        return v, d

    def to_dict(self) -> dict:
        dict_ = copy(self.__dict__)
        for key, value in dict_.items():
            if isinstance(value, np.ndarray):
                dict_[key] = value.tolist()
        return dict_

    @staticmethod
    def from_dict(dict_data: dict) -> "CSP":
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

    def plot(
        self,
        channel_set: "EEGChannelSet",
        figure: "plt.Figure | None" = None,
        plot_filters: bool = False,
        plot_patterns: bool = True,
        topo_settings: dict | None = None,
        show: bool = False,
        plot_eig: bool = True,
        only_selected: bool = True,
    ) -> "plt.Figure":
        # Late import: plotting deps are optional (headless-safe).
        from medusa.core.data import ChannelSet
        from medusa.plots import TopographicPlot

        # Error detection and initialization
        if not plot_patterns and not plot_filters:
            raise ValueError("Cannot plot CSP if plot_filters and "
                             "plot_patterns are both None")
        if figure is None:
            figure = plt.figure(figsize=(7.5, 3), dpi=300)
        if len(channel_set.l_cha) != self.sel_filters.shape[1]:
            raise ValueError(
                f"The number of channels ({len(channel_set.l_cha)}) must be "
                f"the same as the number of channels used to train the CSP "
                f"({self.sel_filters.shape[1]})")
        if topo_settings is None:
            topo_settings = {
                "head_radius": 1.0,
                "head_line_width": 2,
                "interp_contour_width": 1,
                "interp_points": 500,
            }

        # The new plots package needs the new ChannelSet (positions/uids); rebuild
        # one from the channel labels, bridging a legacy EEGChannelSet argument.
        topo_channels = ChannelSet().add_unipolar_eeg_channels(channel_set.l_cha)

        def _topo(
            ax: "plt.Axes",
            values: NDArray,
            clim: tuple[float, float],
            cmap: str,
        ) -> "TopographicPlot":
            """Build + render a topography with the medusa.plots API, mapping the
            legacy ``topo_settings`` keys onto the new TopographicPlot params."""
            tp = TopographicPlot(
                topo_channels, ax, cmap=cmap, clim=clim,
                radius=topo_settings.get("head_radius", 1.0),
                line_width=topo_settings.get("head_line_width", 2),
                interp_resolution=topo_settings.get("interp_points", 500))
            tp.set_data(values)
            return tp
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
                topo = _topo(ax, sel_filters[j, :], (-max_f, max_f), "RdBu")
                ax.set_title(f"Filter {j}")
                if plot_eig and not plot_patterns:
                    ax.set_xlabel(f'Eig: {sel_eigenvalues[j]:.3f}')
                # Colorbar
                if j == sel_filters.shape[0] - 1:
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes('right', size='5%', pad=0.05)
                    figure.colorbar(
                        topo.mappable, cax=cax, orientation='vertical')
            j += 1

        # Plot patterns
        if plot_patterns:
            for i in range(sel_filters.shape[0]):
                ax = figure.add_subplot(n_row, sel_filters.shape[0], j + i + 1)
                topo = _topo(ax, sel_patterns[i, :], (-max_p, max_p), "PiYG")
                ax.set_title(f"Pattern {i}")
                if plot_eig:
                    ax.set_xlabel(f'Eig: {sel_eigenvalues[i]:.3f}')
                # Colorbar
                if i == sel_filters.shape[0] - 1:
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes('right', size='5%', pad=0.05)
                    figure.colorbar(
                        topo.mappable, cax=cax, orientation='vertical')
        plt.suptitle("CSP")

        # Show?
        if show:
            plt.show()
        return figure


class CCA:
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
    ax : {(channels, no_filters) ndarray}
        Activation patterns for the X backward model [1]. Each column
        corresponds to the activation pattern of each spatial filter.
    ay : {(channels, no_filters) ndarray}
        Activation patterns for the Y forward model [1]. Each column
        corresponds to the activation pattern of each spatial filter.

    References
    ----------
    [1] Haufe, S., Meinecke, F., Görgen, K., Dähne, S., Haynes, J. D.,
    Blankertz, B., & Bießmann, F. (2014). On the interpretation of weight
    vectors of linear models in multivariate neuroimaging. Neuroimage, 87,
    96-110.

    Examples
    --------
    >>> import numpy as np
    >>> from medusa.signal.spatial_filtering import CCA
    >>> rng = np.random.default_rng(0)
    >>> signal = rng.standard_normal((500, 4))
    >>> reference = rng.standard_normal((500, 4))
    >>> cca = CCA()
    >>> cca.fit(signal, reference)
    >>> cca.project(signal, filter_idx=0, projection='wx').shape
    (500,)
    """

    def __init__(self) -> None:
        self.wx = None
        self.wy = None
        self.r = None
        self.ax = None
        self.ay = None

    def fit(
        self,
        signal: NDArray,
        reference: NDArray,
        sign_correction: str | None = None,
    ) -> None:
        """Fit the CCA spatial filters from the signal and the reference.

        Parameters
        ----------
        signal :
            ``(n_samples, n_channels)`` data matrix, usually with the
            segments concatenated along the sample axis.
        reference :
            ``(n_samples, n_channels)`` reference matrix. Its number of
            samples must match that of ``signal``; repeat the reference if
            necessary before calling this function.
        sign_correction :
            Sign of spatial filters and activation patterns in CCA is
            ambiguous. If ``None``, no sign correction is applied. If
            ``"max"``, the spatial filter weights and the activation patterns
            are flipped so that their maximum value is positive. If a channel
            index (as a string), they are flipped so that the value for that
            channel is positive.

        Raises
        ------
        ValueError
            Propagated from :meth:`canoncorr` if the inputs do not share the
            same number of samples, contain a single sample, or are rank 0.
        """
        signal = np.asarray(signal)
        reference = np.asarray(reference)
        # CCA
        [self.wx, self.wy, self.r] = self.canoncorr(signal, reference)

        # Sign correction
        if sign_correction is not None:
            if sign_correction == "max":
                if np.abs(np.min(self.wy[:, 0])) > np.abs(
                        np.max(self.wy[:, 0])):
                    self.wy *= -1
                if np.abs(np.min(self.wx[:, 0])) > np.abs(
                        np.max(self.wx[:, 0])):
                    self.wx *= -1
            else:
                target_ch_idx = int(sign_correction)
                self.wy *= np.sign(self.wy[target_ch_idx, 0])
                self.wx *= np.sign(self.wx[target_ch_idx, 0])

        # Activation patterns
        self.ax = self._compute_activation(signal, self.wx)
        self.ay = self._compute_activation(reference, self.wy)

    def project(
        self,
        signal: NDArray,
        filter_idx: int | tuple[int, ...] = 0,
        projection: str = 'wy',
    ) -> NDArray:
        """Project the input signal using the selected spatial filter(s).

        Parameters
        ----------
        signal :
            ``(n_samples, n_channels)`` testing data matrix. The number of
            channels must match the one used to train.
        filter_idx :
            Index (or indices) of the spatial filters to use. Filters are
            sorted by their importance in terms of correlation, so the default
            filter (0) is the one that yields the highest correlation.
        projection :
            Canonical coefficients to use in the projection. By default the
            function uses ``Wy``. Typically, if the signal is averaged ``Wy``
            must be used; otherwise, if it is not averaged but just
            concatenated, ``Wx`` must be used.

        Returns
        -------
        NDArray
            ``(n_samples,)`` signal projected along the selected spatial
            filter.

        Raises
        ------
        ValueError
            If ``projection`` is neither ``'wy'`` nor ``'wx'``.
        """
        signal = np.asarray(signal)
        if projection.lower() == 'wy':
            return np.matmul(signal, self.wy[:, filter_idx])
        elif projection.lower() == 'wx':
            return np.matmul(signal, self.wx[:, filter_idx])
        else:
            raise ValueError(f'[CCA] Unknown projection {projection}')

    def get_activation_patterns(
        self,
        filter_idx: int | tuple[int, ...] = 0,
        projection: str = 'wy',
    ) -> NDArray:
        """Get the activation patterns (in columns) for the given filters.

        Parameters
        ----------
        filter_idx :
            Index (or indices) of the spatial filters to use. Filters are
            sorted by their importance in terms of correlation, so the default
            filter (0) is the one that yields the highest correlation.
        projection :
            Canonical coefficients to use in the projection. By default the
            function uses ``Wy``. Typically, if the signal is averaged ``Wy``
            must be used; otherwise, if it is not averaged but just
            concatenated, ``Wx`` must be used.

        Returns
        -------
        NDArray
            ``(n_channels, n_filters)`` activation patterns for the desired
            filters and projection.

        Raises
        ------
        ValueError
            If ``projection`` is neither ``'wy'`` nor ``'wx'``.
        """
        if projection.lower() == 'wy':
            return self.ay[:, filter_idx]
        elif projection.lower() == 'wx':
            return self.ax[:, filter_idx]
        else:
            raise ValueError(f'[CCA] Unknown projection {projection}')

    @staticmethod
    def canoncorr(
        signal: NDArray,
        reference: NDArray,
    ) -> tuple[NDArray, NDArray, NDArray]:
        """Compute the canonical correlation analysis (CCA) of two matrices.

        Computes the CCA for the data matrices ``signal`` (dimensions
        ``N``-by-``P1``) and ``reference`` (dimensions ``N``-by-``P2``). Both
        must have the same number of observations (rows) but can have a
        different number of variables (cols). The j-th columns of ``A`` and
        ``B`` contain the canonical coefficients, i.e. the linear combination
        of variables making up the j-th canonical variable for ``signal`` and
        ``reference``, respectively. If ``signal`` or ``reference`` are less
        than full rank, ``canoncorr`` warns and returns zeros in the rows of
        ``A`` or ``B`` corresponding to dependent columns. The final dimension
        ``D`` is ``D = min(rank_signal, rank_reference)``.

        Notes
        -----
        This method is adapted from the MATLAB function ``canoncorr``. Check
        that file in case of conflicts, doubts or additional information.

        Parameters
        ----------
        signal :
            ``(N, P1)`` input matrix. Rows are observations and cols are
            variables.
        reference :
            ``(N, P2)`` input matrix. Rows are observations and cols are
            variables.

        Returns
        -------
        A : NDArray
            ``(P1, D)`` sample canonical coefficients for the variables in
            ``signal``. The j-th column of ``A`` contains the linear
            combination of variables that makes up the j-th canonical
            variable for ``signal``. If ``signal`` is less than full rank,
            ``A`` has zeros in the rows corresponding to its dependent cols.
        B : NDArray
            ``(P2, D)`` sample canonical coefficients for the variables in
            ``reference``. The j-th column of ``B`` contains the linear
            combination of variables that makes up the j-th canonical
            variable for ``reference``. If ``reference`` is less than full
            rank, ``B`` has zeros in the rows corresponding to its dependent
            cols.
        r : NDArray
            ``(D,)`` sample canonical correlations. The j-th element of ``r``
            is the correlation between the j-th columns of the canonical
            scores for the variables in ``signal`` and ``reference``.

        Raises
        ------
        ValueError
            If ``signal`` and ``reference`` do not share the same number of
            observations, if there is a single observation, or if either
            matrix is rank 0.

        References
        ----------
        [1] Krzanowski, W.J., Principles of Multivariate Analysis,
        Oxford University Press, Oxford, 1988.
        [2] Seber, G.A.F., Multivariate Observations, Wiley, New York, 1984.

        Examples
        --------
        >>> import numpy as np
        >>> from medusa.signal.spatial_filtering import CCA
        >>> rng = np.random.default_rng(0)
        >>> A, B, r = CCA.canoncorr(rng.random((10, 4)), rng.random((10, 4)))
        >>> r.shape
        (4,)
        """
        signal = np.asarray(signal)
        reference = np.asarray(reference)

        # signal dims
        if len(signal.shape) == 1:
            n = signal.shape[0]
            p1 = 1
            signal = signal.reshape(n, 1)
        else:
            [n, p1] = signal.shape

        # Check the input size
        if reference.shape[0] != n:
            raise ValueError('[canoncorr] Input size mismatch')
        elif n == 1:
            raise ValueError('[canoncorr] Not enough data')

        # reference dims
        if len(reference.shape) == 1:
            p2 = 1
            reference = reference.reshape(n, 1)
        else:
            p2 = reference.shape[1]

        # Center the variables
        signal = signal - np.mean(signal, 0)
        reference = reference - np.mean(reference, 0)

        # Factor the inputs and find a full rank set of columns if necessary
        [Q1, T11, perm1] = slinalg.qr(signal, mode='economic', pivoting=True)
        rank_signal = nlinalg.matrix_rank(T11)
        if rank_signal == 0:
            raise ValueError('[canoncorr] Rank of signal is 0. Invalid Data')
        elif rank_signal < p1:
            warnings.warn('[canoncorr] signal is not full rank')
            Q1 = Q1[:, 0:rank_signal]
            T11 = T11[0:rank_signal, 0:rank_signal]
        [Q2, T22, perm2] = slinalg.qr(reference, mode='economic',
                                      pivoting=True)
        rank_reference = nlinalg.matrix_rank(T22)
        if rank_reference == 0:
            raise ValueError('[canoncorr] Rank of reference is 0. Invalid Data')
        elif rank_reference < p2:
            warnings.warn('[canoncorr] reference is not full rank')
            Q2 = Q2[:, 0:rank_reference]
            T22 = T22[0:rank_reference, 0:rank_reference]

        # Compute canonical coefficients and canonical correlations. For
        # rank_signal > rank_reference, the economy-size version ignores the
        # extra columns in L and rows in D. For rank_signal < rank_reference,
        # need to ignore extra columns in M explicitly. Normalize A and B to
        # give U and V unit variance.
        d = np.min([rank_signal, rank_reference])
        [L, D, M] = nlinalg.svd(np.matmul(Q1.T, Q2))
        M = M.T
        A = nlinalg.lstsq(T11, L[:, 0:d], rcond=None)[0] * np.sqrt(n - 1)
        B = nlinalg.lstsq(T22, M[:, 0:d], rcond=None)[0] * np.sqrt(n - 1)
        r = D[0:d]
        r[r < 0] = 0  # remove roundoff errors
        r[r > 1] = 1

        # Put coefficients back to their full size and their correct order
        A = np.concatenate((A, np.zeros((p1 - rank_signal, d))), axis=0)
        A = A[np.argsort(perm1), :]
        B = np.concatenate((B, np.zeros((p2 - rank_reference, d))), axis=0)
        B = B[np.argsort(perm2), :]

        return A, B, r

    @staticmethod
    def _compute_activation(signal: NDArray, weights: NDArray) -> NDArray:
        """Compute an activation pattern from the signal and trained weights.

        Parameters
        ----------
        signal :
            ``(n_samples, n_channels)`` input matrix, usually with the
            segments concatenated along the sample axis.
        weights :
            ``(n_channels, D)`` sample weight coefficients for the variables
            in ``signal``.

        Returns
        -------
        NDArray
            ``(n_channels, D)`` activation patterns.
        """
        cov = np.cov(signal - np.mean(signal, 0), rowvar=0)
        if signal.shape[1] == 1:
            return cov @ weights
        cov_s = np.cov(signal @ weights, rowvar=0)
        return cov @ weights @ cov_s

    def to_dict(self) -> dict:
        return self.__dict__

    @staticmethod
    def from_dict(dict_data: dict) -> "CCA":
        cca = CCA()
        cca.wx = dict_data['wx']
        cca.wy = dict_data['wy']
        cca.r = dict_data['r']
        return cca

class TRCA:
    """
    The class TRCA performs a Task-related Component Analysis filtering. First,
    function fit() should be called to train the spatial filters. Then, spatial
    filters could be used to project testing data. After fit(), the following
    attributes are computed:

    Attributes
    ----------
    w : {(channels, 1) ndarray}
        Spatial filter obtained.

    Examples
    --------
    >>> import numpy as np
    >>> from medusa.signal.spatial_filtering import TRCA
    >>> rng = np.random.default_rng(0)
    >>> segments = rng.standard_normal((10, 200, 4))
    >>> trca = TRCA()
    >>> trca.fit(segments)
    >>> trca.project(rng.standard_normal((200, 4))).shape
    (200,)
    """
    def __init__(self) -> None:
        self.w = None

    def fit(self, signal: NDArray) -> None:
        """Fit the TRCA spatial filter from the segmented signal.

        Parameters
        ----------
        signal :
            ``(n_segments, n_samples, n_channels)`` input matrix, usually with
            the segments stacked along the first axis.
        """
        self.w = self.trca(signal)

    def project(self, signal: NDArray) -> NDArray:
        """Project the input signal using the fitted spatial filter.

        The spatial filter has dimensions ``(n_channels, 1)``.

        Parameters
        ----------
        signal :
            ``(n_samples, n_channels)`` testing data matrix. The number of
            channels must match the one used to train.

        Returns
        -------
        NDArray
            ``(n_samples,)`` signal projected along the spatial filter.
        """
        signal = np.asarray(signal)
        return np.matmul(signal, self.w)

    @staticmethod
    def trca(signal: NDArray) -> NDArray:
        """Compute the task-related component analysis (TRCA) spatial filter.

        Computes TRCA for the segmented data matrix ``signal`` (dimensions
        ``n_segments`` by ``n_samples`` per segment by ``n_channels``).

        Parameters
        ----------
        signal :
            ``(n_segments, n_samples, n_channels)`` input matrix, with the
            segments stacked along the first axis.

        Returns
        -------
        NDArray
            ``(n_channels,)`` spatial filter (eigenvector associated to the
            largest eigenvalue).

        References
        ----------
        [1] Tanaka, H., Group task-related component analysis (GTRCA): A multivariate
        method for inter-trial reproducibility and inter-subject similarity
        maximization for EEG data analysis, Scientific Reports, 10, 2020.
        [2] Nakanishi, M., Wang, Y., Chen, X., Wang, Y.T., Gao, X., Jung, T.P.,
        Enhancing detection of SSVEPs for a high-speed brain speller using task-related
        component analysis, IEEE Transactions on Biomedical Engineering, 65, 2018
        """
        signal = np.asarray(signal)
        # signal shape (n_segments, n_samples per segment, n_channels)
        n_segments = np.shape(signal)[0]
        n_samples = np.shape(signal)[1]
        n_channels = np.shape(signal)[2]
        S = np.zeros((n_channels, n_samples))

        # Definition of S (cross covariance through segments)
        for k in range(n_segments):
            S = S + np.transpose(signal[k, :, :])
        S = np.matmul(S, np.transpose(S))
        for k in range(n_segments):
            S = S - np.matmul(np.transpose(signal[k, :, :]), signal[k, :, :])

        S = (1 / ((n_segments - 1) * n_segments * n_samples)) * S

        # Definition of Q (Covariance matrix of continuous data)
        signal = signal.reshape((n_segments * n_samples, n_channels))
        n_total_samples = np.shape(signal)[0]
        Q = (1 / n_total_samples) * np.matmul(np.transpose(signal), signal)

        # Eigenvalues and vectors of Q^-1 S
        eig_val, eig_vect = slinalg.eigh(S, Q)

        # Return eigenvector related to greater eigenvalue
        return eig_vect[:, np.where(eig_val == np.max(eig_val))[0][0]]
