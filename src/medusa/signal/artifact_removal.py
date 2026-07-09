"""Artifact removal methods for biomedical signals.

This module provides utilities to detect and remove artifacts from time-series
biosignals (e.g., EEG):

- :func:`reject_noisy_segments`: amplitude-threshold rejection of noisy
  segments.
- :class:`ICA`: Independent Component Analysis for blind source separation and
  artifact rejection.
- :class:`ArtifactRegression`: regression-based removal of a reference artifact
  (e.g., EOG) recorded simultaneously with the signal.
"""

# External imports
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy.signal import welch

# Medusa imports
from medusa.core.legacy.biosignals.eeg.eeg import EEGChannelSet
from medusa.core.serialization import SerializableComponent
from medusa.core.utils import check_data_dims

__all__ = [
    "reject_noisy_segments",
    "ICA",
    "ICAData",
    "ArtifactRegression",
]


def reject_noisy_segments(
    segments: NDArray,
    signal_mean: NDArray,
    signal_std: NDArray,
    k: float = 4,
    n_samp: int = 2,
    n_channels: int = 1,
) -> tuple[float, NDArray, NDArray]:
    """Reject noisy segments by amplitude thresholding.

    Discards segments that contain at least ``n_samp`` samples exceeding
    ``k * signal_std`` (in absolute value, relative to ``signal_mean``) in at
    least ``n_channels`` channels.

    Parameters
    ----------
    segments
        Shape (n_segments, n_samples, n_channels). Segmented signal. A single
        2-D segment ``(n_samples, n_channels)`` is accepted and promoted to the
        canonical 3-D shape.
    signal_mean
        Shape (n_channels,). Per-channel mean of the signal.
    signal_std
        Shape (n_channels,). Per-channel standard deviation of the signal.
    k
        Standard-deviation multiplier used to compute the rejection threshold.
    n_samp
        Minimum number of samples above the threshold (per channel) required to
        flag a segment.
    n_channels
        Minimum number of channels meeting the ``n_samp`` criterion required to
        discard a segment.

    Returns
    -------
    pct_rejected
        Percentage of rejected segments.
    clean_segments
        Shape (n_clean_segments, n_samples, n_channels). Segments that passed
        the rejection criterion (always canonical 3-D).
    rejected_idx
        Shape (n_segments,). Boolean mask, ``True`` for discarded segments.

    Raises
    ------
    ValueError
        If ``signal_mean`` or ``signal_std`` does not have one value per
        channel of ``segments``.

    Examples
    --------
    >>> import numpy as np
    >>> from medusa.signal.artifact_removal import reject_noisy_segments
    >>> segments = np.random.randn(10, 500, 4)
    >>> mean = segments.reshape(-1, 4).mean(axis=0)
    >>> std = segments.reshape(-1, 4).std(axis=0)
    >>> pct, clean, idx = reject_noisy_segments(segments, mean, std)
    """
    # Validate and promote to canonical (n_segments, n_samples, n_channels)
    segments, _ = check_data_dims(segments, rep_type='time_segments')
    n_data_channels = segments.shape[2]

    # Check errors
    if signal_std.shape[0] != n_data_channels:
        raise ValueError('Array signal_std does not match with segments size. '
                         'It must have the same number of channels')
    if signal_mean.shape[0] != n_data_channels:
        raise ValueError('Array signal_mean does not match with segments size. '
                         'It must have the same number of channels')

    segments_abs = np.abs(segments)
    cmp = segments_abs > np.abs(signal_mean) + k * signal_std
    idx = np.sum((np.sum(cmp, axis=1) >= n_samp), axis=1) >= n_channels
    pct_rejected = (np.sum(idx) / segments.shape[0]) * 100
    return pct_rejected, segments[~idx, :, :], idx


class ICA:
    """Independent Component Analysis for blind source separation.

    Implements ICA (via :class:`sklearn.decomposition.FastICA`) on top of a PCA
    whitening stage. It is typically used to decompose EEG into independent
    components, identify those dominated by artifacts (e.g., ocular, muscular),
    and rebuild the signal excluding them.

    Parameters
    ----------
    random_state
        Seed or random state forwarded to ``FastICA`` for reproducibility.

    Examples
    --------
    >>> import numpy as np
    >>> from medusa.signal.artifact_removal import ICA
    >>> signal = np.random.randn(2000, 8)
    >>> ica = ICA(random_state=0)
    >>> ica.fit(signal, l_cha=['Fz', 'Cz', 'Pz', 'Oz',
    ...                        'C3', 'C4', 'P3', 'P4'], fs=250, n_components=8)
    >>> clean = ica.rebuild(signal, exclude=[0])
    """

    def __init__(self, random_state: int | None = None) -> None:

        self.n_components = None
        self.random_state = random_state

        self.pre_whitener = None
        self.unmixing_matrix = None
        self.mixing_matrix = None
        self._ica_n_iter = None
        self.components_excluded = None
        self.ica_labels = None

        # PCA attributes
        self.n_pca_components = None
        self._pca_mean = None
        self._pca_explained_variance = None
        self._pca_explained_variance_ratio = None
        self._pca_components = None

        # Signal attributes
        self.l_cha = None
        self.channel_set = None
        self.fs = None

    def fit(self, signal: NDArray, l_cha: list[str], fs: float,
            n_components: int | float | None) -> None:
        """Fit the ICA decomposition to the signal.

        Parameters
        ----------
        signal
            Shape (n_samples, n_channels) or
            (n_segments, n_samples, n_channels).
            Time-domain signal used to estimate the unmixing matrix.
        l_cha
            Channel labels, used to build the EEG channel set / montage.
        fs
            Sampling frequency in Hz.
        n_components
            Number of ICA components. If ``int``, the exact number; if
            ``float`` in (0, 1), the number of components capturing that
            fraction of variance; if ``None``, estimated automatically.
        """
        from scipy import linalg
        from sklearn.decomposition import FastICA

        # Set arguments
        self.n_components = n_components
        self.l_cha = l_cha
        self.fs = fs

        # Define MEEG Channel Set
        self.channel_set = EEGChannelSet()
        self.channel_set.set_standard_montage(self.l_cha)

        signal, _ = self._check_signal_dimensions(signal.copy())

        self._get_pre_whitener(signal)
        signal_pca = self._whitener(signal.copy())

        ica = FastICA(whiten=False, random_state=self.random_state,
                      max_iter=1000)
        ica.fit(signal_pca[:, :self.n_components])
        self.unmixing_matrix = ica.components_
        self._ica_n_iter = ica.n_iter_

        assert self.unmixing_matrix.shape == (self.n_components,) * 2

        # Whitening unmixing matrix
        norm = np.sqrt(self._pca_explained_variance[:self.n_components])
        norm[norm == 0] = 1.
        self.unmixing_matrix /= norm
        self.mixing_matrix = linalg.pinv(self.unmixing_matrix)

        # Sort ica components from greater to lower explained variance
        self._sort_components(signal)

        self.ica_labels = [f"ICA_{n}" for n in range(self.n_components)]

    def get_sources(self, signal: NDArray) -> NDArray:
        """Project the signal onto the estimated independent components.

        Parameters
        ----------
        signal
            Shape (n_samples, n_channels) or
            (n_segments, n_samples, n_channels).

        Returns
        -------
        sources
            Shape (n_samples, n_components). Independent source activations.
        """
        if self.mixing_matrix is None:
            raise RuntimeError('ICA has not been fitted yet. Please, fit ICA.')

        signal, _ = self._check_signal_dimensions(signal.copy())

        signal = self._pre_whiten(signal)
        if self._pca_mean is not None:
            signal -= self._pca_mean

        # Transform signal to PCA space and then apply unmixing matrix
        pca_transform = np.dot(self._pca_components[:self.n_components],
                               signal.T)
        sources = np.dot(self.unmixing_matrix, pca_transform)

        return sources.T

    def rebuild(self, signal: NDArray,
                exclude: int | list[int] | None = None) -> NDArray:
        """Reconstruct the signal excluding the given ICA components.

        Parameters
        ----------
        signal
            Shape (n_samples, n_channels) or
            (n_segments, n_samples, n_channels).
        exclude
            Index or list of indices of the ICA components to remove from the
            reconstruction (e.g., artifact components). If ``None``, no
            component is excluded.

        Returns
        -------
        signal_rebuilt
            Signal reconstructed without the excluded components, with the same
            dimensionality as the input.
        """
        signal, n_epo_samples = self._check_signal_dimensions(signal.copy())

        signal = self._pre_whiten(signal)

        if exclude is not None:
            if isinstance(exclude, int):
                exclude = [exclude]
            exclude = np.array(list(set(exclude)))
            if len(np.where(exclude > self.n_components)[0]) > 0:
                raise ValueError("One or more ICA component keys that you have"
                                 "marked to exclude from signal rebuild are "
                                 "greater than the total number of "
                                 "ICA components.")
            self.components_excluded = exclude

        # Apply PCA
        if self._pca_mean is not None:
            signal -= self._pca_mean

        # Determine ica components to keep in signal rebuild
        c_to_keep = np.setdiff1d(np.arange(self.n_components), exclude)

        # Define projection matrix
        proj = np.dot(np.dot(self._pca_components[:self.n_components].T,
                             self.mixing_matrix[:, c_to_keep]),
                      np.dot(self.unmixing_matrix[c_to_keep, :],
                             self._pca_components[:self.n_components]))

        # Apply projection to signal
        signal_rebuilt = np.transpose(np.dot(proj, signal.T))

        if self._pca_mean is not None:
            signal_rebuilt += self._pca_mean

        signal_rebuilt *= self.pre_whitener

        # Restore epochs if original signal was divided in epochs
        if n_epo_samples is not None:
            signal_rebuilt = np.reshape(signal_rebuilt,
                                        (int(signal_rebuilt.shape[
                                                 0] / n_epo_samples)
                                         , n_epo_samples,
                                         signal_rebuilt.shape[1]))

        return signal_rebuilt

    def get_components(self) -> NDArray:
        """Return the ICA spatial components (topographies).

        Returns
        -------
        components
            Shape (n_channels, n_components). Spatial pattern (mixing
            topography) of each independent component.
        """
        # Transform
        components = np.dot(self.mixing_matrix[:, :self.n_components].T,
                            self._pca_components[:self.n_components]).T
        return components

    def save(self, path: str) -> None:
        """Persist the fitted ICA model to disk (BSON format).

        Parameters
        ----------
        path
            Destination file path.
        """
        if self.mixing_matrix is None:
            raise RuntimeError('ICA has not been fitted yet. Please, fit ICA.')

        ica_data = ICAData(self.pre_whitener, self.unmixing_matrix,
                           self.mixing_matrix, self.n_components,
                           self._pca_components, self._pca_mean,
                           self.components_excluded,
                           self.random_state)

        ica_data.save(path, 'bson')

    def load(self, path: str) -> None:
        """Load a previously saved ICA model from disk.

        Parameters
        ----------
        path
            Path to the saved BSON file.
        """
        # Load ICAData instance
        ica_data = ICAData().load_from_bson(path)

        # Update ICA arguments
        self.pre_whitener = ica_data.pre_whitener
        self.unmixing_matrix = np.array(ica_data.unmixing_matrix)
        self.mixing_matrix = np.array(ica_data.mixing_matrix)
        self.n_components = ica_data.n_components
        self._pca_components = np.array(ica_data.pca_components)
        self._pca_mean = np.array(ica_data.pca_mean)
        self.components_excluded = np.array(ica_data.components_excluded)
        self.random_state = ica_data.random_state

    # ------------------------- AUXILIARY METHODS ------------------------------
    @staticmethod
    def _check_signal_dimensions(signal: NDArray) -> tuple[NDArray, int | None]:
        n_epo_samples = None
        # Check input dimensions
        if len(signal.shape) == 3:
            # Stack the epochs
            n_epo_samples = signal.shape[1]
            signal = np.vstack(signal)
        elif len(signal.shape) == 1:
            raise ValueError("signal input only has one dimension, but at "
                             "least two dimensions (n_samples, n_channels) "
                             "are needed")
        return signal, n_epo_samples

    def _get_pre_whitener(self, signal: NDArray) -> None:
        self.pre_whitener = np.std(signal)

    def _pre_whiten(self, signal: NDArray) -> NDArray:
        signal /= self.pre_whitener
        return signal

    def _whitener(self, signal: NDArray) -> NDArray:
        from sklearn.decomposition import PCA

        signal_pw = self._pre_whiten(signal)

        pca = PCA(n_components=None, whiten=True)
        signal_pca = pca.fit_transform(signal_pw)

        self._pca_mean = pca.mean_
        self._pca_components = pca.components_
        self._pca_explained_variance = pca.explained_variance_
        self._pca_explained_variance_ratio = pca.explained_variance_ratio_
        self.n_pca_components = pca.n_components_
        del pca

        # Check a correct input of n_components parameter
        if self.n_components is None:
            self.n_components = min(self.n_pca_components,
                                    self._exp_var_ncomp(
                                        self._pca_explained_variance_ratio,
                                        0.99999)[0])
        elif isinstance(self.n_components, float):
            self.n_components = self._exp_var_ncomp(
                self._pca_explained_variance_ratio, self.n_components)[0]
            if self.n_components == 1:
                raise RuntimeError(
                    'One PCA component captures most of the '
                    'explained variance, your threshold '
                    'results in 1 component. You should select '
                    'a higher value.')
        else:
            if not isinstance(self.n_components, int):
                raise ValueError(
                    f'n_components={self.n_components} must be None,'
                    f'float or int value')

        if self.n_components > self.n_pca_components:
            raise ValueError(f'The number of ICA components('
                             f'n_components={self.n_components}) must be '
                             f'lower than the number of PCA components'
                             f'({self.n_pca_components}).')

        return signal_pca

    def _sort_components(self, signal: NDArray) -> None:
        sources = self.get_sources(signal)
        meanvar = np.sum(self.mixing_matrix ** 2, axis=0) * \
                  np.sum(sources ** 2, axis=0) / \
                  (sources.shape[0] * sources.shape[1] - 1)
        c_order = np.argsort(meanvar)[::-1]
        self.unmixing_matrix = self.unmixing_matrix[c_order, :]
        self.mixing_matrix = self.mixing_matrix[:, c_order]

    @staticmethod
    def _exp_var_ncomp(var: NDArray, n: float) -> tuple[int, float]:
        cvar = np.asarray(var, dtype=np.float64)
        cvar = cvar.cumsum()
        cvar /= cvar[-1]
        # We allow 1., which would give us N+1
        n = min((cvar <= n).sum() + 1, len(cvar))
        return n, cvar[n - 1]

    # --------------------------- PLOT METHODS ---------------------------------

    def plot_components(self, cmap: str = 'bwr') -> Figure:
        """Plot the spatial topographies of the ICA components."""
        # Late imports: plotting deps are optional (headless-safe)
        import matplotlib.pyplot as plt

        import medusa.plots as mp
        from medusa.core.data import ChannelSet

        components = self.get_components()
        n_components = components.shape[1]
        channel_set = ChannelSet().add_unipolar_eeg_channels(self.l_cha)

        rows, cols = mp.optimal_grid(n_components)
        fig, axes = plt.subplots(rows, cols, squeeze=False)
        for idx, ax in enumerate(axes.flat):
            if idx < n_components:
                mp.plot_topography(components[:, idx], channel_set, ax,
                                   cmap=cmap)
                ax.set_title(self.ica_labels[idx])
            else:
                ax.set_axis_off()
        return fig

    def plot_sources(self, signal: NDArray, sources_to_show: list[int] | None = None,
                     time_to_show: float | None = None,
                     ch_offset: float | None = None) -> Figure:
        """Plot the independent source activations as stacked time series."""
        # Late imports: plotting deps are optional (headless-safe)
        import matplotlib.pyplot as plt

        from medusa.plots import plot_timeline

        sources = self.get_sources(signal)            # (n_samples, n_sources)
        labels = list(self.ica_labels)
        if sources_to_show is not None:               # select a subset of sources
            sources = sources[:, sources_to_show]
            labels = [labels[i] for i in sources_to_show]
        if time_to_show is not None:                  # limit the time window (s)
            sources = sources[:int(time_to_show * self.fs)]
        fig, ax = plt.subplots(1, 1)
        # offset=None auto-scales the channel spacing to the data.
        plot_timeline(sources, ax, fs=self.fs, cha_labels=labels,
                      offset=ch_offset)
        return fig

    def plot_summary(self, signal: NDArray, component: int | list[int],
                     psd_freq_range: tuple[float, float] = (1, 70),
                     psd_window: str = 'hamming', time_to_show: float = 2,
                     cmap: str = 'bwr') -> "Figure":
        """Plot a per-component summary for the requested ICA component(s).

        For each requested component, the figure stacks four panels: the
        spatial topography, the power spectral density (mean ± std across
        segments), an image of the stacked source segments, and the source
        time course.

        Parameters
        ----------
        signal
            Shape (n_samples, n_channels) for continuous data, or
            (n_segments, n_samples, n_channels) for segmented data. A 2-D
            input is promoted to the canonical 3-D shape and re-segmented
            internally using ``time_to_show``; a 3-D input keeps its segments.
        component
            Index, or list of indices, of the ICA component(s) to summarise.
        psd_freq_range
            Lower and upper frequency limits (Hz) of the PSD panel.
        psd_window
            Window passed to :func:`scipy.signal.welch` (see its docs).
        time_to_show
            Segment length in seconds used to re-segment continuous data.
            Ignored when ``signal`` is already segmented (3-D).
        cmap
            Matplotlib colormap used for the topography and segment image.

        Returns
        -------
        matplotlib.figure.Figure
            The figure of the last plotted component.

        Examples
        --------
        >>> import numpy as np
        >>> from medusa.signal.artifact_removal import ICA
        >>> signal = np.random.randn(2000, 8)
        >>> ica = ICA(random_state=0)
        >>> ica.fit(signal, l_cha=['Fz', 'Cz', 'Pz', 'Oz',
        ...                        'C3', 'C4', 'P3', 'P4'], fs=250,
        ...         n_components=8)
        >>> fig = ica.plot_summary(signal, component=0)  # doctest: +SKIP
        """
        # Late imports: plotting deps are optional (headless-safe)
        import matplotlib.pyplot as plt

        from medusa.core.data import ChannelSet
        from medusa.plots import TopographicPlot, plot_timeline

        # Check error
        if isinstance(component, int):
            component = np.array([component])
        elif isinstance(component, list):
            component = np.array(component)
        else:
            raise ValueError("Component parameter must be a int or list of int"
                             "of the ICA components.")
        if np.any(component > self.n_components):
            raise ValueError("There is a component greater than the total"
                             f"number of ICA components:{self.n_components}.")

        # Validate / promote to canonical (n_segments, n_samples, n_channels).
        # An inserted segments axis means the caller passed continuous (2-D)
        # data; an empty `inserted` means it was already segmented (3-D).
        signal = np.asarray(signal)
        signal, inserted = check_data_dims(signal, rep_type='time_segments')
        is_segmented = len(inserted) == 0

        n_samples_epoch = None
        if is_segmented:
            n_samples_epoch = signal.shape[1]
            n_stacks = signal.shape[0]
            time_to_show = int(n_samples_epoch / self.fs)

        sources = self.get_sources(signal)[:, component]
        components = self.get_components()
        # The new plots package needs the new ChannelSet (positions/labels); rebuild
        # one from the channel labels (as plot_components does), bridging the
        # legacy EEGChannelSet stored on the model.
        channel_set = ChannelSet().add_unipolar_eeg_channels(self.l_cha)

        if n_samples_epoch is None:
            n_samples_epoch = int(time_to_show * self.fs)
            n_stacks = (int(len(sources[:, 0]) / n_samples_epoch))

        for ii in range(len(component)):
            fig = plt.figure()
            ax_1 = fig.add_subplot(3,4,(1,6))
            ax_2 = fig.add_subplot(3, 4, (3,4))
            ax_3 = fig.add_subplot(3, 4, (7,8))
            ax_4 =fig.add_subplot(3, 1, 3)

            # Topoplot
            topo = TopographicPlot(channel_set, ax_1,
                                   interp_resolution=300, line_width=1.5,
                                   cmap=cmap, extra_radius=0)
            topo.set_data(components[:, component[ii]])
            ax_1.set_title(self.ica_labels[component[ii]])

            stacked_source = np.reshape(
                sources[:(n_stacks * n_samples_epoch), ii],
                (n_stacks, n_samples_epoch))

            # PSD
            f, psd = welch(stacked_source, self.fs, window=psd_window, )
            f_range = np.logical_and(f>=psd_freq_range[0],f<=psd_freq_range[1])
            psd_mean = np.mean(10*np.log10(psd),axis=0)
            psd_std = np.std(10*np.log10(psd),axis=0)
            ax_2.fill_between(f[f_range], (psd_mean-psd_std)[f_range],
                              (psd_mean+psd_std)[f_range],
                              color='k',alpha=0.3)
            ax_2.plot(f[f_range],psd_mean[f_range],'k')
            ax_2.set_xlim(f[f_range][0],f[f_range][-1])
            ax_2.set_xlabel('Frequency (Hz)')
            ax_2.set_ylabel('Power/Hz (dB)')
            ax_2.set_title('Power spectral density')

            # Stacked data image
            ax_3.pcolormesh(np.linspace(0,time_to_show,n_samples_epoch),
                               np.arange(n_stacks),stacked_source,cmap=cmap,
                               shading='gouraud')
            ax_3.set_xlabel('Time (s)')
            ax_3.set_ylabel('Segments')
            ax_3.set_title('Stacked source segments')

            # Time plot (windowed to the requested duration)
            src = sources[:int(time_to_show * self.fs), ii]
            plot_timeline(src, ax_4, fs=self.fs,
                          cha_labels=[self.ica_labels[component[ii]]])
            ax_4.set_title('Source time plot')

            fig.tight_layout(pad=1)
        return fig

    def show_exclusion(self, signal: NDArray, exclude: list[int] | None = None,
                       ch_to_show: list[int] | None = None,
                       time_to_show: float | None = None,
                       ch_offset: float | None = None) -> Figure:
        """Overlay the original and the reconstructed signal (with the excluded
        components removed) to visualise the effect of the rejection.

        The two signals are drawn on the same stacked channel baselines via
        :class:`~medusa.plots.TimeLinePlot` (``Pre-ICA`` as the primary trace,
        ``Post-ICA`` as an overlay), so they share scaling and a single legend.
        """
        # Late imports: plotting deps are optional (headless-safe)
        import matplotlib.pyplot as plt

        from medusa.plots import TimeLinePlot

        rebuilt = self.rebuild(signal, exclude)
        labels = list(self.l_cha)
        original = signal
        if ch_to_show is not None:                # select a subset of channels
            original = original[..., ch_to_show]
            rebuilt = rebuilt[..., ch_to_show]
            labels = [labels[i] for i in ch_to_show]
        if time_to_show is not None and original.ndim == 2:   # window (s), 2-D
            n = int(time_to_show * self.fs)
            original, rebuilt = original[:n], rebuilt[:n]

        fig, ax = plt.subplots(1, 1)
        tl = TimeLinePlot(ax, fs=self.fs, cha_labels=labels, offset=ch_offset)
        tl.set_data(original, label='Pre-ICA')    # primary trace
        tl.add_overlay(rebuilt, label='Post-ICA')  # overlaid on the same baselines
        return fig

class ICAData(SerializableComponent):
    """Serializable container holding a fitted :class:`ICA` model.

    Stores the matrices and parameters required to reapply or rebuild an ICA
    decomposition, and implements the :class:`SerializableComponent` interface
    so the model can be persisted to disk.
    """

    def __init__(self, pre_whitener: NDArray | None = None,
                 unmixing_matrix: NDArray | None = None,
                 mixing_matrix: NDArray | None = None,
                 n_components: int | None = None,
                 pca_components: NDArray | None = None,
                 pca_mean: NDArray | None = None,
                 components_excluded: list[int] | None = None,
                 random_state: int | None = None) -> None:

        # General parameters
        self.n_components = n_components
        self.random_state = random_state
        self.pre_whitener = pre_whitener

        # ICA
        self.unmixing_matrix = unmixing_matrix
        self.mixing_matrix = mixing_matrix
        self.components_excluded = components_excluded

        # PCA
        self.pca_components = pca_components
        self.pca_mean = pca_mean

    def to_serializable_obj(self) -> dict:
        # Build a new dict so the live instance is not mutated
        rec_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, np.ndarray):
                rec_dict[key] = value.tolist()
            else:
                rec_dict[key] = value
        return rec_dict

    @classmethod
    def from_serializable_obj(cls, dict_data: dict) -> "ICAData":
        return cls(**dict_data)


class ArtifactRegression:
    """Regression-based artifact removal.

    Removes artifacts (e.g., EOG) from a main signal (e.g., EEG) when the
    artifact is recorded simultaneously with a reference sensor. For each
    signal channel, the artifact contribution is estimated by least-squares
    regression against the reference signal and then subtracted.

    Notes
    -----
    Both ``signal`` and ``reference`` should be preprocessed (e.g., band-pass
    filtered) before use for better performance and accuracy.

    Examples
    --------
    >>> import numpy as np
    >>> from medusa.signal.artifact_removal import ArtifactRegression
    >>> signal = np.random.randn(1000, 8)     # (n_samples, n_channels)
    >>> reference = np.random.randn(1000, 2)  # e.g., 2 EOG channels
    >>> reg = ArtifactRegression()
    >>> clean = reg.fit_transform(signal, reference)
    """

    def __init__(self) -> None:
        # Initialize the coefficient matrix (will be filled after fitting)
        self.coefs = None
        self.is_fit = False

    def fit(self, signal: NDArray, reference: NDArray) -> None:
        """Estimate the regression coefficients for artifact removal.

        Performs a least-squares regression for each signal channel to estimate
        how much of the reference artifact is present in it. The coefficients
        are stored in ``self.coefs``.

        Parameters
        ----------
        signal
            Shape (n_samples, n_channels). Main signal (e.g., EEG) containing
            the artifacts.
        reference
            Shape (n_samples, n_reference_channels). Reference artifact signal
            (e.g., EOG) recorded alongside the main signal.
        """
        # Validate / promote to canonical (n_samples, n_channels)
        signal, _ = check_data_dims(signal, rep_type='time')
        reference, _ = check_data_dims(reference, rep_type='time')
        # Mean-center each channel over time (axis=0 → per-channel temporal mean)
        reference = reference - np.mean(reference, axis=0, keepdims=True)
        # Compute covariance of reference signal
        cov_ref = reference.T @ reference
        # Regression coefficients for each signal channel
        n_channels = signal.shape[1]
        n_reference_channels = reference.shape[1]
        coefs = np.zeros((n_channels, n_reference_channels))
        # Process each signal channel separately to reduce memory load
        for c in range(n_channels):
            # Mean-center the signal channel
            signal_cha = signal[:, c]
            signal_cha = signal_cha - np.mean(signal_cha, -1, keepdims=True)
            signal_cha = signal_cha.reshape(1, -1)
            # Perform the least squares regression to estimate coefficients
            coefs[c] = np.linalg.lstsq(
                cov_ref, reference.T @ signal_cha.T, rcond=None)[0].T
        # Store the regression coefficients
        self.coefs = coefs
        self.is_fit = True

    def transform(self, signal: NDArray, reference: NDArray) -> NDArray:
        """Remove the artifacts from the signal using the fitted coefficients.

        Subtracts the estimated artifact contribution from each signal channel.

        Parameters
        ----------
        signal
            Shape (n_samples, n_channels). Main signal (e.g., EEG) to clean.
        reference
            Shape (n_samples, n_reference_channels). Reference artifact signal
            (e.g., EOG) to regress out.

        Returns
        -------
        signal
            Shape (n_samples, n_channels). Cleaned signal (a new array; the
            input is not modified).

        Raises
        ------
        ValueError
            If :meth:`fit` has not been called before :meth:`transform`.
        """
        # Check errors
        if not self.is_fit:
            raise ValueError('Method fit must be called first!')
        # Validate / promote to canonical (n_samples, n_channels). Keep the
        # inserted axes to restore the caller's ndim at the end.
        signal, inserted = check_data_dims(signal, rep_type='time')
        reference, _ = check_data_dims(reference, rep_type='time')
        # Work on a copy to avoid mutating the caller's array
        signal = signal.copy()
        # Mean-center each channel over time (axis=0 → per-channel temporal mean)
        reference = reference - np.mean(reference, axis=0, keepdims=True)
        n_channels = signal.shape[1]
        # Subtract the artifact contribution from each signal channel
        for c in range(n_channels):
            signal[:, c] -= (self.coefs[c] @ reference.T).reshape(
                signal[:, c].shape)
        # Restore the caller's original dimensionality
        return np.squeeze(signal, axis=inserted) if inserted else signal

    def fit_transform(self, signal: NDArray, reference: NDArray) -> NDArray:
        """Fit the model and remove the artifacts in a single call.

        Parameters
        ----------
        signal
            Shape (n_samples, n_channels). Main signal to clean.
        reference
            Shape (n_samples, n_reference_channels). Reference artifact signal
            to regress out.

        Returns
        -------
        signal
            Shape (n_samples, n_channels). Cleaned signal.
        """
        self.fit(signal, reference)  # Fit the regression model
        return self.transform(signal, reference)  # Apply the transformation
