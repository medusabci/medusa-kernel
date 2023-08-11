# External imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy.signal import welch

# Medusa imports
from medusa.plots.head_plots import plot_head,plot_topography
from medusa.meeg.meeg import EEGChannelSet
from medusa import epoching
from medusa.plots.timeplot import time_plot
from medusa.components import SerializableComponent


def reject_noisy_epochs(epochs, signal_mean, signal_std, k=4, n_samp=2,
                        n_cha=1):
    """Simple thresholding method to reject noisy epochs. It discards epochs
    with n_samp samples greater than k*std in n_cha channels

    Parameters
    ----------
     epochs : list or numpy.ndarray
        Epochs of signal with dimensions [n_epochs x samples x channels]
    signal_mean : float
        Mean of the signal
    signal_std : float
        Standard deviation of the signal
    k : float
        Standard deviation multiplier to calculate threshold
    n_samp : int
        Minimum number of samples that have to be over the threshold in each
        epoch to be discarded
    n_cha : int
        Minimum number of channels that have to have n_samples over the
        threshold in each epoch to be discarded

    Returns
    -------
     float
        Percentage of reject epochs in
    numpy.ndarray
        Clean epochs
    numpy.ndarray
        Indexes for rejected epochs. True for discarded epoch
    """

    # Check errors
    if len(epochs.shape) != 3:
        raise Exception('Malformed epochs array. It must be of dimmensions '
                        '[epochs x samples x channels]')
    if signal_std.shape[0] != epochs.shape[2]:
        raise Exception('Array signal_std does not match with epochs size. '
                        'It must have the same number of channels')
    if signal_mean.shape[0] != epochs.shape[2]:
        raise Exception('Array signal_mean does not match with epochs size. '
                        'It must have the same number of channels')

    epochs_abs = np.abs(epochs)
    cmp = epochs_abs > np.abs(signal_mean) + k * signal_std
    idx = np.sum((np.sum(cmp, axis=1) >= n_samp), axis=1) >= n_cha
    pct_rejected = (np.sum(idx) / epochs.shape[0]) * 100
    return pct_rejected, epochs[~idx, :, :], idx


class ICA:
    def __init__(self, random_state=None):

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


    def fit(self, signal, l_cha, fs, n_components):
        from sklearn.decomposition import FastICA
        from scipy import linalg

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

    def get_sources(self, signal):
        if not hasattr(self, 'mixing_matrix'):
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

    def rebuild(self, signal, exclude=None):

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

    def get_components(self):
        # Transform
        components = np.dot(self.mixing_matrix[:, :self.n_components].T,
                            self._pca_components[:self.n_components]).T
        return components

    def save(self, path):
        if not hasattr(self, 'mixing_matrix'):
            raise RuntimeError('ICA has not been fitted yet. Please, fit ICA.')

        ica_data = ICAData(self.pre_whitener, self.unmixing_matrix,
                           self.mixing_matrix, self.n_components,
                           self._pca_components, self._pca_mean,
                           self.components_excluded,
                           self.random_state)

        ica_data.save(path, 'bson')

    def load(self, path):
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
    def _check_signal_dimensions(signal):
        n_epo_samples = None
        # Check input dimensions
        if len(signal.shape) == 3:
            # Stack the epochs
            n_epo_samples = signal.shape[1]
            signal = np.vstack(signal)
        elif len(signal.shape) == 1:
            raise ValueError("signal input only has one dimension, but two"
                             "dimensions (n_samples, n_channels) arer nedded at"
                             "least")
        return signal, n_epo_samples

    def _get_pre_whitener(self, signal):
        self.pre_whitener = np.std(signal)

    def _pre_whiten(self, signal):
        signal /= self.pre_whitener
        return signal

    def _whitener(self, signal):
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

    def _sort_components(self, signal):
        sources = self.get_sources(signal)
        meanvar = np.sum(self.mixing_matrix ** 2, axis=0) * \
                  np.sum(sources ** 2, axis=0) / \
                  (sources.shape[0] * sources.shape[1] - 1)
        c_order = np.argsort(meanvar)[::-1]
        self.unmixing_matrix = self.unmixing_matrix[c_order, :]
        self.mixing_matrix = self.mixing_matrix[:, c_order]

    @staticmethod
    def _exp_var_ncomp(var, n):
        cvar = np.asarray(var, dtype=np.float64)
        cvar = cvar.cumsum()
        cvar /= cvar[-1]
        # We allow 1., which would give us N+1
        n = min((cvar <= n).sum() + 1, len(cvar))
        return n, cvar[n - 1]

    # --------------------------- PLOT METHODS ---------------------------------

    def plot_components(self, cmap='bwr'):

        # Get ICA components
        components = self.get_components()

        # Define subplot parameters
        n_components = components.shape[1]

        if n_components <= 5:
            cols = n_components
            rows = 1
        else:
            cols = 5
            rows = np.ceil(n_components / 5)

        # Define subplot
        fig, axes = plt.subplots(int(rows), int(cols))
        if len(axes.shape) == 1:
            axes = np.array([axes])

        # Topo plots
        ic_c = 0
        for r in axes:
            for c in r:
                if ic_c < n_components:
                    plot_head(axes=c, channel_set=self.channel_set,
                              interp_points=300,linewidth=1.5)
                    plot_topography(axes=c,channel_set=self.channel_set,
                                    values=components[:, ic_c],
                                    interp_points=300,cmap=cmap,
                                    show_colorbar=False,plot_extra=0)
                    c.set_title(self.ica_labels[ic_c])
                    ic_c += 1
                else:
                    c.set_axis_off()
        fig.show()
        return fig

    def plot_sources(self, signal, sources_to_show=None, time_to_show=None,
                     ch_offset=None):
        sources = self.get_sources(signal)

        if ch_offset is None:
            ch_offset = np.max(np.abs(sources[:, 0]))
        fig, ax = plt.subplots(1,1)
        time_plot(sources, self.fs, self.ica_labels,
                            ch_to_show=sources_to_show,
                            time_to_show=time_to_show,
                            ch_offset=ch_offset,show=False,fig=fig,
                  axes=ax)

    def plot_summary(self, signal, component, psd_freq_range=[1,70],
                     psd_window='hamming', time_to_show=2,cmap='bwr'):
        # Check error
        if isinstance(component,int):
            component = np.array([component])
        elif isinstance(component,list):
            component = np.array(component)
        else:
            raise ValueError("Component parameter must be a int or list of int"
                             "of the ICA components.")
        if np.any(component > self.n_components):
            raise ValueError("There is a component greater than the total"
                             f"number of ICA components:{self.n_components}.")

        # Check if signal is epoched
        n_samples_epoch = None
        if len(signal.shape) == 3:
            n_samples_epoch = signal.shape[1]
            n_stacks = signal.shape[0]
            time_to_show = int(n_samples_epoch/self.fs)

        sources = self.get_sources(signal)[:,component]
        components = self.get_components()

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
            plot_head(axes=ax_1, channel_set=self.channel_set,
                      interp_points=300, linewidth=1.5)
            plot_topography(axes=ax_1, channel_set=self.channel_set,
                            values=components[:, component[ii]],
                            interp_points=300, cmap=cmap,
                            show_colorbar=False, plot_extra=0)
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

            # Time plot
            time_plot(sources[:,ii],self.fs,[self.ica_labels[component[ii]]],
                      time_to_show=time_to_show,fig=fig,axes=ax_4)
            ax_4.set_title('Source time plot')

            fig.tight_layout(pad=1)
        return fig

    def show_exclusion(self, signal, exclude=None, ch_to_show=None,
                       time_to_show=None, ch_offset=None):

        if ch_offset is None:
            ch_offset = np.max(np.abs(signal.copy()))

        # Check if signal is divided in epochs
        if len(signal.shape) == 3:
            n_epochs = signal.shape[0]
        else:
            n_epochs = 1
        fig, ax = plt.subplots(1,1)
        time_plot(signal,self.fs,self.l_cha,time_to_show,
                             ch_to_show,ch_offset,axes=ax,fig=fig)
        signal_rebuilt = self.rebuild(signal,exclude)
        time_plot(signal_rebuilt,self.fs,self.l_cha,time_to_show,
                             ch_to_show,ch_offset,fig=fig,axes=ax,color='b',
                              show_epoch_lines=False)
        #Create legend
        handles = [fig.axes[0].lines[0],fig.axes[0].lines[signal.shape[-1]
                                                             + n_epochs -1]]
        fig.axes[0].legend(handles=handles,labels=['Pre-ICA','Post-ICA'],
                    loc='upper center', bbox_to_anchor=(0.5, 1.15),
                    ncol=3, fancybox=True, shadow=True)

class ICAData(SerializableComponent):
    def __init__(self, pre_whitener=None, unmixing_matrix=None,
                 mixing_matrix=None,
                 n_components=None, pca_components=None, pca_mean=None,
                 components_excluded=None, random_state=None):

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

    def to_serializable_obj(self):
        rec_dict = self.__dict__
        for key in rec_dict.keys():
            if type(rec_dict[key]) == np.ndarray:
                rec_dict[key] = rec_dict[key].tolist()
        return rec_dict

    @classmethod
    def from_serializable_obj(cls, dict_data):
        return cls(**dict_data)