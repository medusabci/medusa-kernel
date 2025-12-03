# External imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy.signal import welch

# Medusa imports
from medusa.plots.head_plots import TopographicPlot
from medusa.meeg.meeg import EEGChannelSet
from medusa import epoching
from medusa.plots.timeplot import time_plot
from medusa.components import SerializableComponent, ProcessingMethod

# Classification imports
import h5py
import torch
import sqlite3

connection = sqlite3.connect("mibase.db")

from medusa.classification_utils import EarlyStopping
from eeg_inception_v1_TFM_pytorch import EEGInceptionV1ICA
from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, \
    cohen_kappa_score
import gc
import os

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Graphic interface imports
import keras
import tkinter as tk
from tkinter import Tk, Button, Label, Entry, messagebox
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from tkinter import messagebox as MessageBox
from scipy.ndimage import gaussian_filter1d


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
                    topo = TopographicPlot(axes=c,channel_set=self.channel_set,
                                    interp_points=300,head_line_width=1.5,
                                    cmap=cmap,extra_radius=0)
                    topo.update(values=components[:, ic_c])
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
            topo = TopographicPlot(axes=ax_1, channel_set=self.channel_set,
                                   interp_points=300, head_line_width=1.5,
                                   cmap=cmap, extra_radius=0)
            topo.update(values=components[:, component[ii]])
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


class ArtifactRegression(ProcessingMethod):

    def __init__(self):
        """
        Initialize the artifact regression method.

        This class implements a method to remove artifacts (e.g., EOG signals)
        from time series data (e.g., EEG signals). The artifacts must be
        recorded simultaneously with the signal, and this method attempts to
        regress out the artifacts from the main signal.

        Attributes:
        - self.coefs: Stores the regression coefficients after fitting the
            model.

        Notes:
        The input signals (sig and art_sig) must be preprocessed (e.g.,
        band-pass filtering) before using them in this class for better
        performance and accuracy.

        Alternative implementation:
        self.coefs = np.linalg.lstsq(
            art_sig.T @ art_sig, art_sig.T, rcond=None)[0] @ sig
        sig = (sig.T - self.coefs.T @ art_sig.T).T
        """
        # Calling the superclass constructor and defining the inputs that
        # are transformed and fit by the method.
        super().__init__(transform=['signal', 'artefacts_signal'],
                         fit_transform=['signal', 'artefacts_signal'])
        # Initialize the coefficient matrix (will be filled after fitting)
        self.coefs = None
        self.is_fit = False

    def fit(self, sig, art_sig):
        """
        Fits the artifact regression model by computing regression coefficients
        for removing the artifacts from the signal.

        This method performs a linear regression for each signal channel to
        estimate how much of the artifact is present in each channel. The
        regression coefficients (self.coefs) are computed using the least
        squares solution for each channel.

        Steps
        1. Remove the mean from the artifact signal for normalization.
        2. Compute the covariance matrix of the artifact signal.
        3. For each signal channel:
           - Remove the mean from the signal.
           - Perform a least squares fit to estimate the regression
                coefficients.

        Parameters
        ----------
        sig : The main signal (e.g., EEG) that contains the artifacts.
        art_sig : The artifact signal (e.g., EOG) recorded alongside the main
            signal.
        """
        # Mean-center the signal
        art_sig = art_sig - np.mean(art_sig, axis=-1, keepdims=True)
        # Compute covariance of artifact signal
        cov_ref = art_sig.T @ art_sig
        # Regression coefficients for each signal channel
        n_sig_cha = sig.shape[1]
        n_art_sig_cha = art_sig.shape[1]
        coefs = np.zeros((n_sig_cha, n_art_sig_cha))
        # Process each signal channel separately to reduce memory load
        for c in range(n_sig_cha):
            # Mean-center the signal channel
            sig_cha = sig[:, c]
            sig_cha = sig_cha - np.mean(sig_cha, -1, keepdims=True)
            sig_cha = sig_cha.reshape(1, -1)
            # Perform the least squares regression to estimate coefficients
            coefs[c] = np.linalg.lstsq(
                cov_ref, art_sig.T @ sig_cha.T, rcond=None)[0].T
        # Store the regression coefficients
        self.coefs = coefs
        self.is_fit = True

    def transform(self, sig, art_sig):
        """
        Removes the artifacts from the signal using the previously computed
        coefficients.

        This method applies the regression coefficients (self.coefs) to remove
        the artifacts from each channel of the signal. It subtracts the artifact
        contribution from each signal channel.

        Steps:
        1. Mean-center the artifact signal.
        2. For each signal channel:
           - Subtract the estimated artifact component using the regression
                coefficients.

        Parameters:
        -----------
        - sig: The main signal (e.g., EEG) to clean.
        - art_sig: The artifact signal (e.g., EOG) to regress out.
        """
        # Check errors
        if not self.is_fit:
            raise ValueError('Function fit_dataset must be called first!')
        # Mean-center the artifact signal
        art_sig = art_sig - np.mean(art_sig, -1,
                                    keepdims=True)
        n_sig_cha = sig.shape[1]

        # Subtract the artifact contribution from each signal channel
        for c in range(n_sig_cha):
            sig_cha = sig[:, c]
            # Remove artifact contribution using pre-computed coefficients
            sig_cha -= (self.coefs[c] @ art_sig.T).reshape(sig_cha.shape)

        return sig  # Return the cleaned signal

    def fit_transform(self, sig, art_sig):
        """
        Combines the fit and transform steps into a single method.

        Parameters:
        - sig: The main signal to clean.
        - art_sig: The artifact signal to regress out.

        This method first fits the regression model to estimate the coefficients,
        then applies those coefficients to remove the artifacts from the signal.

        Returns:
        - The cleaned signal with the artifacts removed.
        """
        self.fit(sig, art_sig)  # Fit the regression model
        return self.transform_signal(sig, art_sig)  # Apply the transformation


def train_meg_model(
    ica_power: bool = True,
    ica_chosen_components: int = 40,
    multi: bool = False,
    trial: int = 6,
    input_path: str = "dataset/data_w_ica.h5",
    n_rows: int = 75680,
    n_columns: int = 60000,
    n_columns_ica: int = 160,
    confidence: float = 0.9,
    model_path: str = "Model/my_model_PyTorch.keras",
    output_path: str = "results"
):
    """
        Run the modified EEG-Inception model train and test to classify MEG signals
        in a binary manner or by discriminating between classes

        Parameters
        ----------
        ica_power : bool
            Whether to apply ICA (Independent Component Analysis) power decomposition.
        ica_chosen_components : int
            Number of ICA components selected for analysis.
        multi : bool
            If True, enables multiclass classification mode. If False, binary classification.
        trial : int
            Test parameter.
        input_path : str
            Path to the input file containing the dataset with ICA applied.
        n_rows : int
            Number of rows in the dataset.
        n_columns : int
            Number of columns in the dataset before ICA reduction.
        n_columns_ica : int
            Number of columns after ICA dimensionality reduction.
        confidence : float
            Confidence threshold for classification.
        model_path : str
            Path to the pre-trained PyTorch model file.
        output_path : str
            Path to the folder where the results (.h5) will be saved.

        Returns
        -------
        Trained model
            The neural network model trained on the dataset.
        Results file : h5
            File containing predictions and related outputs.
        """
    # ......................................... LOAD DATA ..............................................................
    with h5py.File(input_path, 'r') as f: # 'dataset/' + 'data_w_ica' + '.h5'
        for key in ["features", "labels", "subjects", "ica_winv"]:
            data = f[key]
            shape = data.shape
            dtype = data.dtype

            # Create memmap file
            mmap = np.memmap(f"{key}.npy", dtype=dtype, mode="w+", shape=shape)

            # Cpy data in blocks
            for i in range(0, shape[0], 10000):
                mmap[i:i+10000] = data[i:i+10000]
            mmap.flush()

    features = np.memmap("features.npy", dtype="float32", mode="r+", shape=(n_rows, n_columns))
    labels = np.memmap("labels.npy", dtype="int", mode="r+", shape=(n_rows,))
    labels_multi = np.memmap("labels.npy", dtype="int", mode="r+", shape=(n_rows,))
    subjects = np.memmap("subjects.npy", dtype="int", mode="r+", shape=(n_rows,))
    ica = np.memmap("ica_winv.npy", dtype="int", mode="r+", shape=(n_rows,n_columns_ica))
    ica_idx = np.array(
        list(range(ica.shape[0]))) % n_columns_ica  # Extract ica index of component

    # ........................................ CLASSIFICATION ..........................................................
    # ..................... MULTICLASS CLASSIFICATION ...............................
    if multi:
        # Labeled as 0,1,2,3,4 (0 is no artefact, the rest is artefact)
        labels = labels_multi.astype(int)
    else:
        # ..................... BINARY CLASSIFICATION ..............................
        # Since they are labeled with 0, 1, 2, 3, 4, they are converted to binary,
        # where 0 remains as 0 and the rest are converted to 1
        labels = (labels > 0).astype(int)

    # .................................................. (TRAIN)........................................................
    # The number of splits is set to 5 so that the set is divided into 5 parts
    # (each containing 4 training data and 1 test data)
    n_splits = 5
    # KFold: provides train/test indices to divide the data into train sets and test sets
    kf = KFold(n_splits=n_splits)

    # Features is converted to a matrix of size (75680x60000x1x1) because to use EEGInception, it has to be that way
    features = np.expand_dims(features, axis=-1)
    features = np.expand_dims(features, axis=-1)
    # ICA is converted to a matrix of size (160x1x1) because to use EEGInception, it has to be that way
    ica = np.expand_dims(ica, axis=-1)
    ica = np.expand_dims(ica, axis=-1)

    b = list()
    y_probs_elimination = list()
    y_prob_modtot = list()
    sensitivity_acum = list()
    specificity_acum = list()

    #
    for z in range(20):

        y_prob_modfinal_list = list()
        y_pred_list = None
        y_probs_list = None
        y_true_list = None
        test_list = None

        acc_list = list()
        if ica_chosen_components is not None:
            acc_left_list = list()
            kappa_left_list = list()
            sensitivity_left_list = list()
            sensitivity_left_list_1 = list()
            sensitivity_left_list_2 = list()
            sensitivity_left_list_3 = list()
            sensitivity_left_list_4 = list()
            specificity_left_list = list()

        # Choose the first 40 and include all artefacts up to 160
        if ica_chosen_components is not None:
            # Choose first ica_chosen_components from each subject
            indices = list()
            indices_left = list()
            # For each of the 473 subjects
            for sub in np.unique(subjects):
                # The channels indicated in ica_chosen_components are selected
                indices.append(np.where(subjects == sub)[0][:ica_chosen_components])
                # The remaining channels are taken
                if ica_chosen_components < 160:
                    temp_left = np.where(subjects == sub)[0][ica_chosen_components:]
                    indices_left.append(temp_left)
                    indices.append(temp_left[labels[temp_left] == 1])
            indices = np.concatenate(indices)
            if ica_chosen_components < 160:
                indices_left = np.concatenate(indices_left)

        if z == 0:  # for the first iteration, it does not delete anything
            eliminate = False
        else:  # for the rest of the iterations, it eliminates
            eliminate = True
        if eliminate:
            # As b accumulates, lists are created within a global list
            # What you need to do is concatenate them so that they appear in a single list
            doubt_idx = np.concatenate(b)

            indices_orig = indices.copy()
            indices = np.delete(indices,
                                np.where(np.in1d(indices_orig, doubt_idx))[0])
            print('Total indices deleted: {}'.format(
                len(indices_orig) - len(indices)))
            print('Indices removed in this iteration: {}'.format(len(b)))

        # features, labels, subjects, ICA depend on those subjects chosen
        if ica_chosen_components is not None:
            if ica_chosen_components < 160:
                features_ind = features[indices]
                labels_ind = labels[indices]
                labels_multi_ind = labels_multi[indices]
                subjects_ind = subjects[indices]
            else:
                features_ind = features
                labels_ind = labels
                labels_multi_ind = labels_multi
                subjects_ind = subjects

            if ica_power:
                ica_ind = ica[indices]
        else:
            indices = list(range(features.shape[0]))
            features_ind = features
            labels_multi_ind = labels_multi
            labels_ind = labels
            subjects_ind = subjects
            if ica_power:
                ica_ind = ica

        for iteration, (train, test) in enumerate(kf.split(features_ind)):
            torch.cuda.empty_cache()
            gc.collect()
            # ........................................ EEG INCEPTION ................................................................
            model = EEGInceptionV1ICA(
                input_time=features_ind.shape[1] / 200 * 1000,
                fs=200,
                n_cha=1,
                filters_per_branch=8,
                scales_time=(125, 125),
                dropout_rate=0.4,
                activation='elu',
                n_classes=len(np.unique(labels_ind)),
                learning_rate=0.001
            )
            if z == 0 and iteration == 0:
                model.summary()
            # ...................................... EARLY STOPPING .............................................................
            # Early Stopping is configured for when overfitting occurs (when it stops learning and only memorizes)
            # In Early Stopping, PATIENCE is how long you wait after the network stops improving
            early_stopping = [(EarlyStopping(
                # monitor='val_loss',
                min_delta=0.001,
                mode='auto',
                patience=10,
                verbose=1,
                # restore_best_weights=True
            ))]
            # ....................................... CLASS BALANCING ..................................................
            if multi:
                # ............................ MULTICLASS CASE .............................
                class_weight = compute_class_weight(
                    class_weight='balanced',
                    classes=np.unique(labels_ind),
                    y=np.squeeze(labels_ind)
                )
                class_weights = {0: class_weight[0],
                                 1: class_weight[1],
                                 2: class_weight[2],
                                 3: class_weight[3],
                                 4: class_weight[4]}
            else:
                # ............................ BINARY CASE ................................
                class_weight = compute_class_weight(
                    class_weight='balanced',
                    classes=np.unique(labels_ind),
                    y=np.squeeze(labels_ind[train])
                )
                class_weights = {0: class_weight[0], 1: class_weight[1]}

            validation_split = 0.2

            zeros = sum(labels_multi_ind[train] == 0)
            art_1 = sum(labels_multi_ind[train] == 1)
            art_2 = sum(labels_multi_ind[train] == 2)
            art_3 = sum(labels_multi_ind[train] == 3)
            art_4 = sum(labels_multi_ind[train] == 4)

            ind_zeros = (np.where(labels_multi_ind[train] == 0))[0][
                        0:int(round(validation_split * zeros))]
            ind_art_1 = (np.where(labels_multi_ind[train] == 1))[0][
                        0:int(round(validation_split * art_1))]
            ind_art_2 = (np.where(labels_multi_ind[train] == 2))[0][
                        0:int(round(validation_split * art_2))]
            ind_art_3 = (np.where(labels_multi_ind[train] == 3))[0][
                        0:int(round(validation_split * art_3))]
            ind_art_4 = (np.where(labels_multi_ind[train] == 4))[0][
                        0:int(round(validation_split * art_4))]

            # Concatenate in indices_validation
            indices_validation = np.concatenate((train[ind_zeros], train[ind_art_1],
                                                 train[ind_art_2], train[ind_art_3],
                                                 train[ind_art_4]))
            # Whatever is in train but not in validation_indices goes into train_indices
            indices_train = [i for i in train if i not in indices_validation]

            labels_ind = np.eye(len(np.unique(labels_ind)))[labels_ind]

            # ............................................ FIT ....................................................................
            if ica_power:
                fittedModel = model.fit(
                    X=features_ind[indices_train],
                    ica=ica_ind[indices_train],
                    y=labels_ind[indices_train],
                    batch_size=4,
                    epochs=100,
                    shuffle=True,
                    validation_data=(features_ind[indices_validation],
                                     ica_ind[indices_validation],
                                     labels_ind[indices_validation]),
                    callbacks=[early_stopping],
                    class_weight=class_weights,
                    verbose=True
                )
            else:
                fittedModel = model.fit(
                    X=features_ind[indices_train],
                    y=labels_ind[indices_train],
                    batch_size=128,
                    epochs=100,
                    shuffle=True,
                    validation_data=(features_ind[indices_validation],
                                     labels_ind[indices_validation]),
                    callbacks=[early_stopping],
                    class_weight=class_weights
                )

            # .................................... TEST .......................................................................
            if ica_power:
                probs_test = model.predict_proba(features_ind[test], ica_ind[test])
            else:
                probs_test = model.predict_proba(features_ind[test])

            pred_test = probs_test.argmax(axis=-1)
            labels_ind = labels_ind.argmax(axis=-1)

            if multi:
                # MULTICLASS CASE
                pred_test = pred_test.astype(int)
                acc = np.sum((pred_test == labels_multi_ind[test]) / len(
                    labels_multi_ind[test]))

                acc_list.append(acc)

                chance_level = np.max([np.sum(labels_multi_ind[test] == 0),
                                       np.sum(labels_multi_ind[test] > 0)]) / len(
                    labels_multi_ind[test])
                report_test = classification_report(
                    labels_multi_ind[test].astype(int), pred_test.astype(int),
                    output_dict=True)

                # Confusion matrix
                CM_test = confusion_matrix(labels_multi_ind[test].astype(int),
                                           pred_test)
                kappa_test = cohen_kappa_score(labels_multi_ind[test].astype(int),
                                               pred_test)

                # Confusion matrix results
                true_positives_1 = CM_test[0][
                    0]  # how many have been correctly classified as type 1 artefacts
                true_positives_2 = CM_test[1][
                    1]  # how many have been correctly classified as type 2 artefacts
                true_positives_3 = CM_test[2][
                    2]  # how many have been correctly classified as type 3 artefacts
                true_positives_4 = CM_test[3][
                    3]  # how many have been correctly classified as type 4 artefacts

                true_negatives = CM_test[4][
                    4]  # how many have been correctly classified as non-artefacts

                false_negatives_1_2 = CM_test[0][1]  # it was 1 but it predicted 2
                false_negatives_1_3 = CM_test[0][2]  # it was 1 but it predicted 3
                false_negatives_1_4 = CM_test[0][3]  # it was 1 but it predicted 4

                false_negatives_2_1 = CM_test[1][0]  # it was 2 but it predicted 1
                false_negatives_2_3 = CM_test[1][2]  # it was 2 but it predicted 3
                false_negatives_2_4 = CM_test[1][3]  # it was 2 but it predicted 4

                false_negatives_3_1 = CM_test[2][0]  # it was 3 but it predicted 1
                false_negatives_3_2 = CM_test[2][1]  # it was 3 but it predicted 2
                false_negatives_3_4 = CM_test[2][3]  # it was 3 but it predicted 4

                false_negatives_4_1 = CM_test[3][0]  # it was 4 but it predicted 1
                false_negatives_4_2 = CM_test[3][1]  # it was 4 but it predicted 2
                false_negatives_4_3 = CM_test[3][2]  # it was 4 but it predicted 3

                false_positives_1_0 = CM_test[0][4]  # it was 1 but it predicted 0
                false_positives_2_0 = CM_test[1][4]  # it was 2 but it predicted 0
                false_positives_3_0 = CM_test[2][4]  # it was 3 but it predicted 0
                false_positives_4_0 = CM_test[3][4]  # it was 4 but it predicted 0

                # Sensitivity (fraction of true positives)
                sensitivity_test_1 = report_test['1']['recall']
                sensitivity_test_2 = report_test['2']['recall']
                sensitivity_test_3 = report_test['3']['recall']
                sensitivity_test_4 = report_test['4']['recall']
                # Specificity (fraction of true negatives)
                specificity_test = report_test['0']['recall']

                print(CM_test)
                print("Kappa Index is {}".format(kappa_test))
                print("Accuracy of this iteration is {:.2%}".format(np.mean(acc)))
                print("Sensitivity for 1 of this iteration is {}".format(
                    sensitivity_test_1))
                print("Sensitivity for 2 of this iteration is {}".format(
                    sensitivity_test_2))
                print("Sensitivity for 3 of this iteration is {}".format(
                    sensitivity_test_3))
                print("Sensitivity for 4 of this iteration is {}".format(
                    sensitivity_test_4))
                print(
                    "Specificity of this iteration is {}".format(specificity_test))

            else:
                # BINARY CASE
                pred_test = (pred_test > 0).astype(int)
                acc = np.sum(
                    (pred_test == (labels_ind[test] > 0).astype(int))) / len(
                    labels_ind[test])

                acc_list.append(acc)

                chance_level = np.max([np.sum(labels_ind[test] == 0),
                                       np.sum(labels_ind[test] > 0)]) / len(
                    labels_ind[test])

                report_test = classification_report(
                    (labels_ind[test] > 0).astype(int), (pred_test > 0).astype(int),
                    output_dict=True)

                # Confusion matrix
                CM_test = confusion_matrix((labels_ind[test] > 0).astype(int),
                                           (pred_test > 0).astype(int))
                kappa_test = cohen_kappa_score((labels_ind[test] > 0).astype(int),
                                               (pred_test > 0).astype(int))

                # Confusion matrix results
                true_negatives = CM_test[0][
                    0]  # how many correctly classified as non-artefacts (said it was 0)
                true_positives = CM_test[1][
                    1]  # how many correctly classified as artefacts (said it was 1)
                false_negatives = CM_test[1][
                    0]  # how many misclassified as non-artefacts (said it was 0, but was wrong)
                false_positives = CM_test[0][
                    1]  # how many misclassified as artefacts (said it was 1, but was wrong)

                # Sensitivity (fraction of true positives)
                sensitivity_test = report_test['1']['recall']
                # Specificity (fraction of true negatives)
                specificity_test = report_test['0']['recall']

                print(CM_test)
                print("Kappa Index is {}".format(kappa_test))
                print("Accuracy of this iteration is {:.2%}".format(np.mean(acc)))
                print(
                    "Sensitivity of this iteration is {}".format(sensitivity_test))
                print(
                    "Specificity of this iteration is {}".format(specificity_test))

            # .......................................... TOTAL ....................................................................
            y_pred_list = np.concatenate(
                (y_pred_list, pred_test)) if y_pred_list is not None else pred_test
            y_probs_list = np.concatenate((y_probs_list,
                                           probs_test)) if y_probs_list is not None else probs_test
            if z > 0:
                y_prob_modfinal_list.append(
                    model.predict_proba(features[doubt_idx], ica[doubt_idx])[:, 0])
            if multi:
                y_true_list = np.concatenate((y_true_list, labels_multi_ind[
                    test])) if y_true_list is not None else labels_multi_ind[test]
            else:
                y_true_list = np.concatenate(
                    (y_true_list, labels_ind[test])) if y_true_list is not None else \
                labels_ind[test]
            test_list = np.concatenate(
                (test_list, test)) if test_list is not None else test

        if z > 0:
            y_prob_modtot.append(np.average(y_prob_modfinal_list, axis=0))
        # Confusion matrix total
        CM_total = confusion_matrix(y_true_list, y_pred_list)
        report = classification_report(y_true_list.astype(int),
                                       y_pred_list.astype(int), output_dict=True)

        kappa = cohen_kappa_score(y_true_list, y_pred_list)

        if multi:
            # The results are extracted from the confusion matrix
            true_negatives = CM_total[0][
                0]  # how many have been correctly classified as non-artefacts
            true_positives_1 = CM_total[1][
                1]  # how many have been correctly classified as type 1 artefacts
            true_positives_2 = CM_total[2][
                2]  # how many have been correctly classified as type 2 artefacts
            true_positives_3 = CM_total[3][
                3]  # how many have been correctly classified as type 3 artefacts
            true_positives_4 = CM_total[4][
                4]  # how many have been correctly classified as type 4 artefacts

            false_positives_0_1 = CM_total[0][1]  # it was 0 but it predicted 1
            false_positives_0_2 = CM_total[0][2]  # it was 0 but it predicted 2
            false_positives_0_3 = CM_total[0][3]  # it was 0 but it predicted 3
            false_positives_0_4 = CM_total[0][4]  # it was 0 but it predicted 4

            false_negatives_1_0 = CM_total[1][0]  # it was 1 but it predicted 0
            false_negatives_1_2 = CM_total[1][2]  # it was 1 but it predicted 2
            false_negatives_1_3 = CM_total[1][3]  # it was 1 but it predicted 3
            false_negatives_1_4 = CM_total[1][4]  # it was 1 but it predicted 4

            false_negatives_2_0 = CM_total[2][0]  # it was 2 but it predicted 0
            false_negatives_2_1 = CM_total[2][1]  # it was 2 but it predicted 1
            false_negatives_2_3 = CM_total[2][3]  # it was 2 but it predicted 3
            false_negatives_2_4 = CM_total[2][4]  # it was 2 but it predicted 4

            false_negatives_3_0 = CM_total[3][0]  # it was 3 but it predicted 0
            false_negatives_3_1 = CM_total[3][1]  # it was 3 but it predicted 1
            false_negatives_3_2 = CM_total[3][2]  # it was 3 but it predicted 2
            false_negatives_3_4 = CM_total[3][4]  # it was 3 but it predicted 4

            false_negatives_4_0 = CM_total[4][0]  # it was 4 but it predicted 0
            false_negatives_4_1 = CM_total[4][1]  # it was 4 but it predicted 1
            false_negatives_4_2 = CM_total[4][2]  # it was 4 but it predicted 2
            false_negatives_4_3 = CM_total[4][3]  # it was 4 but it predicted 3

            # Sensitivity (fraction of true positives)
            sensitivity_1 = report['1']['recall']
            sensitivity_2 = report['2']['recall']
            sensitivity_3 = report['3']['recall']
            sensitivity_4 = report['4']['recall']
            # Specificity (fraction of true negatives)
            specificity = report['0']['recall']

            print(CM_total)
            print("Kappa Index over all dataset is {}".format(kappa))
            print("Accuracy over all dataset is {:.2%}".format(np.mean(acc_list)))
            print("Sensitivity for 1 over all dataset is {}".format(sensitivity_1))
            print("Sensitivity for 2 over all dataset is {}".format(sensitivity_2))
            print("Sensitivity for 3 over all dataset is {}".format(sensitivity_3))
            print("Sensitivity for 4 over all dataset is {}".format(sensitivity_4))
            print("Specificity over all dataset is {}".format(specificity))

        else:
            # Confusion matrix results
            true_negatives = CM_total[0][0]
            true_positives = CM_total[1][1]
            false_negatives = CM_total[1][0]
            false_positives = CM_total[0][1]

            # Sensitivity (fraction of true positives)
            sensitivity = report['1']['recall']
            # Specificity (fraction of true negatives)
            specificity = report['0']['recall']

            print(CM_total)
            print("Kappa Index over all dataset is {}".format(kappa))
            print("Accuracy over all dataset is {:.2%}".format(np.mean(acc_list)))
            print("Sensitivity over all dataset is {}".format(sensitivity))
            print("Specificity over all dataset is {}".format(specificity))

        # ........................... HISTOGRAM .......................................
        # It wants to display a 1 where no device has been detected when there actually was one
        # Where y_true_list and y_pred_list match, a 1 is placed; where they do not match, a 0 is placed
        false_neg = list()
        wrong = list()
        for i, j in zip(y_true_list, y_pred_list):
            if i == j:
                wrong.append(1)
            else:
                wrong.append(0)

        false_neg_class = list()
        # Wrong inverted in bool mode
        all_misclassification = np.array([x == 0 for x in wrong])
        for label in np.unique(y_true_list):
            # y_true_list in bool mode
            all_true_label = np.array([x == label for x in y_true_list])
            # Multiply and obtain a list with two elements
            false_neg_class.append(all_misclassification * all_true_label)

        # When there is no artefact and an artefact is detected (item 0 in the list)
        false_pos = false_neg_class[0]
        # When there is an artefact and it detects a non-artefact (item 1 on the list)
        false_neg = false_neg_class[1]

        # ...................... CLASSIFICATION TYPE OF artefact .......................
        # False negatives (for each type)
        for selection in [1, 2, 3, 4]:
            print("False negatives of component {}: {} of {}".format(selection,
                                                                     np.sum(
                                                                         labels_multi[
                                                                             indices][
                                                                             test_list] == selection) -
                                                                     np.sum((
                                                                                        y_pred_list == y_true_list) *
                                                                            labels_multi[
                                                                                indices][
                                                                                test_list] == selection),
                                                                     np.sum(
                                                                         labels_multi[
                                                                             indices][
                                                                             test_list] == selection)))

        # Total of false negatives
        total_neg = np.sum(
            ~(y_pred_list == y_true_list) * labels[indices][test_list] == 1)

        # False positives (of each type)
        for selection in [1, 2, 3, 4]:
            print("False positives of component {}: {}".format(selection,
                                                               np.sum(labels_multi[
                                                                          indices][
                                                                          test_list] == selection) -
                                                               np.sum(~(
                                                                           y_pred_list == y_true_list) *
                                                                      labels_multi[
                                                                          indices][
                                                                          test_list] == selection)))

        # Total of false positives
        total_pos = np.sum(
            (y_pred_list == y_true_list) * labels[indices][test_list] == 1)

        # Representation by type of artefact
        false_neg_multi = false_neg * (labels_multi[indices][test_list])
        false_pos_multi = false_pos * (labels_multi[indices][test_list])

        sensitivity_acum.append(sensitivity)
        specificity_acum.append(specificity)

        # If the sensitivity exceeds 0.95, it exits the loop (because it is considered sufficiently good)
        if np.max(sensitivity_acum) > 0.95:
            break
        # If sensitivity worsens compared to the previous two, exit the loop
        if z > 2:
            if sensitivity_acum[-1] < sensitivity_acum[-3]:
                if sensitivity_acum[-2] < sensitivity_acum[-3]:
                    break

        # Accumulate the eliminated in b
        b_orig = np.where((y_probs_list[:, 0] < confidence) & (y_true_list == 0))[0]
        b.append(indices[
                     b_orig])

        b_flat = np.concatenate(b)

        b_new = np.where((y_probs_list[:, 0] < 0.9) & (y_true_list == 0))[
            0]  # Score on the prediction made

        y_probs_elimination.append(y_probs_list[b_orig, 0])

        # CLASSIFICATION OF FALSE POSITIVES FOR POSSIBLE RELABELING
        false_pos_et = ~(y_pred_list == y_true_list)
        # Those that have not been classified correctly are taken
        data_fail = np.squeeze((features[indices][test_list])[false_pos_et, :])
        data_doubt = np.squeeze(features[np.concatenate(b)])
        pred_labels = y_pred_list[false_pos_et]

    # SAVE MODEL
    model.save(model_path)
    # SAVE DATASET
    hf_r = h5py.File(f"{output_path}/trial_{trial}.h5", 'w')
    # Elements
    hf_r.create_dataset("labels", data=y_true_list)
    hf_r.create_dataset("predictions", data=y_pred_list)
    hf_r.create_dataset("probabilities", data=y_probs_list)
    hf_r.create_dataset("indices", data=indices)
    hf_r.create_dataset("test_idx", data=test_list)

    hf_r.create_dataset("false_positives", data=data_fail)
    hf_r.create_dataset("Doubt", data=data_doubt)  # features eliminados
    hf_r.create_dataset("Doubt_idx_ordered", data=b_flat)
    hf_r.create_dataset("Doubt_idx", data=b_new)

    # Close file
    hf_r.close()


def MEG_artifact_recovery_GUI(
    input_path: str,
    block_size: int = 1000,
    total_rows: int = 75680,
    total_cols: int = 60000,
    results_path: str = "results/trial_5.h5",
    confidence: float = 0.9,
    ica_chosen_components: int = 40
):
    """
    Displays the signals classified as doubtful after executing the previous code
    block and removing the artifacts present in them. It also allows the generation
    of synthetic signals and their cleaning to evaluate the model's performance.

    Parameters
    ----------
    input_path : str
        Path to the input file containing the dataset with ICA applied.
    block_size : int
        Size of the memory blocks.
    total_rows : int
        Total number of features.
    total_cols : int
        Total number of samples of each signal.
    results_path : str
        Path to save the results obtained.
    confidence : float
        Confidence threshold for classification.
    ica_chosen_components : int
        Number of ICA components selected for analysis.

    Returns
    -------
    Graphic interface.
    """

    # ORIGINAL FOLDER
    hf = h5py.File(input_path, 'r')

    # Create a memmap file to load data
    memmap_filename = 'large_array.dat'
    features_memmap = np.memmap(memmap_filename, dtype='float32', mode='w+', shape=(total_rows, total_cols))

    # Load data in blocks and write them in the memmap file
    for start_row in range(0, total_rows, block_size):
        end_row = min(start_row + block_size, total_rows)
        block = np.array(hf['features'][start_row:end_row, :])
        features_memmap[start_row:end_row, :] = block

    # Ensure that changes are written to disk
    features_memmap.flush()

    # Features
    features = np.memmap(memmap_filename, dtype='float32', mode='r', shape=(total_rows, total_cols))
    print(features.shape)

    # Labels classified as 0,1,2,3,4 (0 is no artefact, the rest is artefact)
    labels = np.array(hf.get('labels'))

    # Subjects
    subjects = np.array(hf.get('subjects'))

    # ICA components
    ica = np.array(hf.get('ica_winv'))

    # The file from which the data was loaded is closed
    hf.close()

    # RESULTS FOLDER
    hf_dudosas = h5py.File(results_path, 'r')

    labels_dudosas = np.array(hf_dudosas.get('labels'))
    pred_dudosas = np.array(hf_dudosas.get('predictions'))
    prob_dudosas = np.array(hf_dudosas.get('probabilities'))
    index_dudosas = np.array(hf_dudosas.get('indices'))
    test_idx_dudosas = np.array(hf_dudosas.get('test_idx'))

    false_pos_dudosas = np.array(hf_dudosas.get('false_positives'))
    doubt_dudosas = np.array(hf_dudosas.get('Doubt'))

    b = np.where((prob_dudosas[:, 0] < confidence) & (labels_dudosas == 0))[0]  # Score on the prediction made
    doubt_idx = np.squeeze(b)

    hf_dudosas.close()

    total_dudosas = len(doubt_dudosas)

    ica_idx = np.array(list(range(ica.shape[0]))) % 160  # Extract ica index of component
    labels = labels.astype(int)  # Format 0,1,2,3,4

    # Choose first ica_chosen_components of each component
    if ica_chosen_components is not None:
        indices = list()
        for sub in np.unique(subjects):
            # Channels indicated in ica_chosen_components
            indices.append(np.where(subjects == sub)[0][:ica_chosen_components])
        # Rest of channels
        indices = np.concatenate(indices)

    if ica_chosen_components is not None:
        if ica_chosen_components < 160:
            features_ind = features[indices]
            labels_ind = labels[indices]
            subjects_ind = subjects[indices]

    # NEURAL SIGNALS (LABEL = 0)
    label_value_neuro = 0
    indices_with_label_0 = np.where(labels_ind == label_value_neuro)[0]
    total_neuro = len(indices_with_label_0)

    signal_neuro_list = []
    label_neuro_list = []
    ica_index_neuro_list = []

    for i, idx in enumerate(indices_with_label_0):
        # Extract the signal (a row from the features matrix)
        signal_neuro = features_ind[idx, :]
        signal_neuro_list.append(signal_neuro)

        # Extract the corresponding label (in this case, it will always be 0) and the ICA index
        label_neuro = labels_ind[idx]
        label_neuro_list.append(label_neuro)
        ica_index_neuro = ica_idx[idx]
        ica_index_neuro_list.append(ica_index_neuro)

    signal_neuro_total = np.array(signal_neuro_list)
    label_neuro_total = np.array(label_neuro_list)
    ica_index_neuro_total = np.array(ica_index_neuro_list)

    # ARTEFACT SIGNALS, the cardiac ones are loaded at first (LABEL = 1)
    label_value_artefacto = 1
    indices_with_label_1 = np.where(labels_ind == label_value_artefacto)[0]
    total_artefactos = len(indices_with_label_1)

    signal_artefacto_list = []
    label_artefacto_list = []
    ica_index_artefacto_list = []

    for i, idx in enumerate(indices_with_label_1):
        # Extract the signal (a row from the features matrix)
        signal_artefacto = features_ind[idx, :]
        signal_artefacto_list.append(signal_artefacto)

        # Extract the corresponding label (in this case, it will always be 1) and the ICA index
        label_artefacto = labels_ind[idx]
        label_artefacto_list.append(label_artefacto)
        ica_index_artefacto = ica_idx[idx]
        ica_index_artefacto_list.append(ica_index_artefacto)

    signal_artefacto_total = np.array(signal_artefacto_list)
    label_artefacto_total = np.array(label_artefacto_list)
    ica_index_artefacto_total = np.array(ica_index_artefacto_list)

    # MASK
    fs = 200  # 200 Hz (200 samples per second)
    total_time = 5 * 60  # 5 minutes in seconds
    total_points = total_time * fs  # Total points = 5 * 60 * 200 = 60000

    mask = np.zeros(total_points)
    segment1 = fs * 60 * 1  # From minute 0 to 1
    segment2 = fs * 60 * 2  # From minute 1 to 2
    segment3 = fs * 60 * 3  # From minute 2 to 3
    segment4 = fs * 60 * 5  # From minute 3 to 5

    # From minute 0 to 1 (value 0)

    # From minute 1 to 2: 2-second square pulses (1 second = 200 samples, so 2s = 400 samples)
    for i in range(0, segment2 - segment1, 800):  # Iterate each 800 samples (4 seconds)
        mask[segment1 + i: segment1 + i + 400] = 1  # Put to 1 during 400 samples (2 seconds)

    # From minute 2 to 3: 5-second square pulses (1 second = 200 samples, so 5s = 1000 samples)
    for i in range(0, segment3 - segment2, 2000):  # Iterate each 2000 samples (10 seconds)
        mask[segment2 + i: segment2 + i + 1000] = 1  # Put to 1 during 1000 samples (5 seconds)

    # From minute 3 to 5: constant signal of 1
    mask[segment3:segment4] = 1  # From minute 3 until the end of the 5 minutes


    # GRAPHIC INTERFACE
    def onClosing():
        root.destroy()
        print('Window closed')


    root = Tk()
    root.title('Checking functionalities')
    root.geometry('8900x775')
    root.attributes('-fullscreen', True)
    root.protocol('WM_DELETE_WINDOW', onClosing)
    nb = ttk.Notebook(root)
    nb.pack(fill='both', expand='yes')

    # ADD TABS
    p1 = ttk.Frame(nb)  # --> Preprocessing of doubtful signals
    p2 = ttk.Frame(nb)  # --> Preprocessing of synthetic signals

    # ............................................. WINDOW GRAPHICS 1 ..................................................
    fig1 = Figure(figsize=(11, 8), dpi=100)
    ax1 = fig1.add_subplot(111)

    # Integrate the graph into Tkinter using FigureCanvasTkAgg
    canvas1 = FigureCanvasTkAgg(fig1, master=p1)
    canvas1.get_tk_widget().place(x=10, y=10, width=1150, height=365)

    # Create the navigation bar and place it above the graph
    toolbar1 = NavigationToolbar2Tk(canvas1, p1)
    toolbar1.update()
    toolbar1.place(x=845, y=10)  # Navigation bar position

    fig2 = Figure(figsize=(11, 8), dpi=100)
    ax2 = fig2.add_subplot(111)

    # Integrate the graph into Tkinter using FigureCanvasTkAgg
    canvas2 = FigureCanvasTkAgg(fig2, master=p1)
    canvas2.get_tk_widget().place(x=10, y=431, width=1150, height=365)
    # Create the navigation bar and place it above the graph
    toolbar2 = NavigationToolbar2Tk(canvas2, p1)
    toolbar2.update()
    toolbar2.place(x=810, y=428)  # Navigation bar position

    # ............................................... WINDOW GRAPHICS 1 ................................................
    fig3 = Figure(figsize=(11, 8), dpi=100)
    ax3 = fig3.add_subplot(111)

    fig4 = Figure(figsize=(11, 8), dpi=100)
    ax4 = fig4.add_subplot(111)

    # Integrate the graph into Tkinter using FigureCanvasTkAgg
    canvas3 = FigureCanvasTkAgg(fig3, master=p2)
    canvas3.get_tk_widget().place(x=10, y=10, width=1150, height=365)
    # Create the navigation bar and place it above the graph
    toolbar3 = NavigationToolbar2Tk(canvas3, p2)
    toolbar3.update()
    toolbar3.place(x=810, y=10)  # Navigation bar position

    # Integrate the graph into Tkinter using FigureCanvasTkAgg
    canvas4 = FigureCanvasTkAgg(fig4, master=p2)
    canvas4.get_tk_widget().place(x=10, y=431, width=1150, height=365)
    # Create the navigation bar and place it above the graph
    toolbar4 = NavigationToolbar2Tk(canvas4, p2)
    toolbar4.update()
    toolbar4.place(x=810, y=425)  # Navigation bar position

    # .............................................. INFORMATION TABLES ................................................
    # WINDOW 1
    info_frame_1 = tk.Frame(p1, bg="lightgreen", bd=2, relief=tk.SOLID)
    info_frame_1.place(x=1180, y=10, width=350, height=183)  # Position and size of the frame

    info_frame_1a = tk.Frame(p1, bg="gainsboro", bd=2, relief=tk.SOLID)
    info_frame_1a.place(x=1180, y=224, width=350, height=150)  # Position and size of the frame

    info_frame_2 = tk.Frame(p1, bg="lightgreen", bd=2, relief=tk.SOLID)
    info_frame_2.place(x=1180, y=431, width=350, height=183)  # Position and size of the frame

    info_frame_2a = tk.Frame(p1, bg="gainsboro", bd=2, relief=tk.SOLID)
    info_frame_2a.place(x=1180, y=639, width=350, height=150)  # Position and size of the frame

    # WINDOW 2
    info_frame_3 = tk.Frame(p2, bg="lightgreen", bd=2, relief=tk.SOLID)
    info_frame_3.place(x=1180, y=10, width=350, height=183)  # Position and size of the frame

    info_frame_3a = tk.Frame(p2, bg="gainsboro", bd=2, relief=tk.SOLID)
    info_frame_3a.place(x=1180, y=197, width=350, height=80)  # Position and size of the frame

    info_frame_3b = tk.Frame(p2, bg="gainsboro", bd=2, relief=tk.SOLID)
    info_frame_3b.place(x=1180, y=281, width=350, height=80)  # Position and size of the frame

    info_frame_3c = tk.Frame(p2, bg="gainsboro", bd=2, relief=tk.SOLID)
    info_frame_3c.place(x=1180, y=365, width=350, height=80)  # Position and size of the frame

    info_frame_4 = tk.Frame(p2, bg="lightgreen", bd=2, relief=tk.SOLID)
    info_frame_4.place(x=1180, y=479, width=350, height=183)  # Position and size of the frame

    info_frame_4a = tk.Frame(p2, bg="gainsboro", bd=2, relief=tk.SOLID)
    info_frame_4a.place(x=1180, y=666, width=350, height=110)  # Position and size of the frame

    # Add the label “INFO” at the top of the information box
    title_label_1 = tk.Label(info_frame_1, text="INFO DUDOSA", font=("Arial", 14, "bold"), bg="lightgreen", anchor="center")
    title_label_1.grid(row=0, column=0, columnspan=4, pady=5)  # Position in the first row, occupying both columns

    title_label_2 = tk.Label(info_frame_2, text="INFO RECUPERADA", font=("Arial", 14, "bold"), bg="lightgreen",
                             anchor="center")
    title_label_2.grid(row=0, column=0, columnspan=4, pady=5)  # Position in the first row, occupying both columns

    title_label_3 = tk.Label(info_frame_3, text="INFO NEURONAL", font=("Arial", 14, "bold"), bg="lightgreen",
                             anchor="center")
    title_label_3.grid(row=0, column=0, columnspan=4, pady=5)  # Position in the first row, occupying both columns

    title_label_3a = tk.Label(info_frame_3a, text="Selección segmento [pts]", font=("Arial", 11, "bold"), bg="gainsboro",
                             anchor="center")
    title_label_3a.grid(row=0, column=0, columnspan=4, pady=5)  # Position in the first row, occupying both columns

    title_label_4 = tk.Label(info_frame_4, text="INFO ARTEFACTO", font=("Arial", 14, "bold"), bg="lightgreen",
                             anchor="center")
    title_label_4.grid(row=0, column=0, columnspan=4, pady=5)  # Position in the first row, occupying both columns


    # .......................................... SEGMENTATION TAB 1 ....................................................
    # Drop-down menu to select the segmentation of the doubt signal
    segmentado_dudosa = ttk.Combobox(state="readonly", values=["1", "5", "10", "20"], master=p1)
    segmentado_dudosa.place(x=1190, y=264)

    def segmentation_dudosa(event):
        global d
        # Parameters
        fs = 200  # Sample frequency (Hz)
        duration = 300  # Duration in seconds
        total_samples = duration * fs  # Total samples = 60000
        segment_duration_str = segmentado_dudosa.get()  # Segment duration in seconds
        segment_duration = int(segment_duration_str)
        segment_samples = segment_duration * fs  # Samples per segment = 200
        num_segments = total_samples // segment_samples  # 300 segments
        # Signal segmentation in x-seconds clips
        segments_dudosa = doubt_dudosas[d, :].reshape(-1, segment_samples)
        # Create an HDF5 file to store the 300 output signals
        with h5py.File('Segmentacion/' + 'segmentos_dudosos.h5', 'w') as hdf_segments_dudosos:
            # Create a dataset within the HDF5 file to store the signals (300 signals of 60000 samples)
            dset = hdf_segments_dudosos.create_dataset('signals', (num_segments, total_samples), dtype='float64')

            # For each 300 segments
            for j in range(num_segments):
                # Repeat the current segment until reaching 60000 samples
                repeated_signal = np.tile(segments_dudosa[j], total_samples // segment_samples)

                # Store the repeated signal in the HDF5 dataset
                dset[j] = repeated_signal

        hdf_segments_dudosos.close()
        messagebox.showinfo(title="Segmentos dudosos", message=f"Segmentos de: {segment_duration} s")


    Label(p1, text="Segmentar en [s]:", font=("Arial", 10, "bold"), bg='gainsboro').place(x=1190, y=234)
    segmentado_dudosa.bind("<<ComboboxSelected>>", segmentation_dudosa)

    # ......................................... SIGNAL CLASSIFICATION TAB 1 ............................................
    def clasificar_dudosa(event):
        global score_vector_dudosa
        # Open the .h5 file in read mode
        file_classify_dudosa = h5py.File('Segmentacion/' + 'segmentos_dudosos' + '.h5', 'r')
        signals_dudosas = np.array(file_classify_dudosa.get('signals'))
        file_classify_dudosa.close()

        # Signals are converted to a matrix of size (seax60000x1x1) because EEGInception requires this format
        signals_dudosas = np.expand_dims(signals_dudosas, axis=-1)
        signals_dudosas = np.expand_dims(signals_dudosas, axis=-1)

        # Load the model directly using load_model
        new_model_dudosa = keras.models.load_model('Modelo/' + 'my_model.keras')
        # MODEL INPUTS:
        # 1. Name: features, Form: (None, None, 1, 1)
        # 2. Name: ICA, Form: (None, None, 1, 1)

        # Generate predictions for samples
        ica_dummy_dudosa = np.zeros_like(signals_dudosas)  # Adjust the size according to the expected shape of the ICA
        probabilidades_dudosas = new_model_dudosa.predict({'features': signals_dudosas, 'ICA': ica_dummy_dudosa})

        predicciones_dudosas = probabilidades_dudosas.argmax(axis=-1)
        predicciones_dudosas = (predicciones_dudosas > 0).astype(int)

        # RECONSTRUCTION OF THE ORIGINAL SIGNAL
        # Assuming that each segment has 60000 data points
        n_segmentos_dudosas = signals_dudosas.shape[0]
        segmento_length_real_dudosas = int(
            60000 / n_segmentos_dudosas)  # (1 second of original signal at 200 Hz)

        # Extract only the first 200 points of each segment
        señal_reconstruida_dudosa = signals_dudosas[:, :segmento_length_real_dudosas, 0, 0].reshape(
            n_segmentos_dudosas * segmento_length_real_dudosas)

        # REPRESENT
        ax1.clear()
        ax2.clear()
        # Plot signal
        ax1.plot(señal_reconstruida_dudosa.flatten(), label=f'Señal reconstruida')

        mostrado_rojo_dudosa = False
        mostrado_naranja_dudosa = False
        mostrado_amarillo_dudosa = False
        score_vector_dudosa = np.zeros(60000)

        # Go through each segment and plot them
        for i in range(n_segmentos_dudosas):
            inicio_dudosa = i * segmento_length_real_dudosas
            fin_dudosa = inicio_dudosa + segmento_length_real_dudosas

            probabilidad_artefacto_dudosa = probabilidades_dudosas[i][
                1]  # Obtain the probability of the segment being an artefact (class 1)

            if predicciones_dudosas[i] == 1:
                if probabilidad_artefacto_dudosa > 0.9:
                    color = '#D62728'  # Very likely artefact
                    label = 'probabilidad > 0.9' if not mostrado_rojo_dudosa else ""
                    mostrado_rojo_dudosa = True
                elif 0.75 < probabilidad_artefacto_dudosa <= 0.9:
                    color = '#FF7F0E'  # Intermediate probability of artefact
                    label = '0.75 < probabilidad < 0.9' if not mostrado_naranja_dudosa else ""
                    mostrado_naranja_dudosa = True
                elif 0.5 < probabilidad_artefacto_dudosa <= 0.75:
                    color = '#FFD700'  # Low probability of artefact
                    label = '0.5 < probabilidad < 0.75' if not mostrado_amarillo_dudosa else ""
                    mostrado_amarillo_dudosa = True

                ax1.axvspan(inicio_dudosa, fin_dudosa, color=color, alpha=0.3, label=label)

                # PLOT SCORE
                for j in range(inicio_dudosa, fin_dudosa):
                    score_vector_dudosa[j] = probabilidades_dudosas[i][1]

        ax2.plot(score_vector_dudosa, label='Scores')

        ax1.set_title(
            "Clasificación - Label:{} Predicted:{} Score:{:.0%} DB_idx:{} ICA_pos:{}".format(labels_dudosas[doubt_idx[d]],
                                                                                             pred_dudosas[doubt_idx[d]],
                                                                                             np.max(prob_dudosas, axis=-1)[
                                                                                                 doubt_idx[d]],
                                                                                             index_dudosas[doubt_idx[d]],
                                                                                             index_dudosas[
                                                                                                 doubt_idx[d]] % 160 + 1), loc='left')
        ax1.set_xlabel('Tiempo (puntos)')
        ax1.set_ylabel('Amplitud')
        ax1.legend()
        ax1.grid()

        canvas1.draw()

        ax2.set_title(f'Score')
        ax2.set_xlabel('Tiempo (puntos)')
        ax2.set_ylabel('Amplitud')
        ax2.legend()
        ax2.grid()

        canvas2.draw()


    # ......................................... SMOOTHING SCORE  TAB 1 .................................................
    # Drop-down menu to choose score smoothing
    suavizado_dudosa = ttk.Combobox(state="readonly", values=["Gaussiana", "Media móvil"], master=p1)
    suavizado_dudosa.place(x=1190, y=679)


    def suavizar_dudosa(event):
        global score_vector_dudosa, smoothed_signal_dudosa
        metodo_dudosa = suavizado_dudosa.get()
        if metodo_dudosa == 'Gaussiana':
            # Smooth the signal using a Gaussian filter
            sigma_dudoso = 334  # Adjust sigma to control the level of smoothing
            # Sigma controls the amount of smoothing. A higher sigma smooths more, but reduces signal detail
            smoothed_signal_dudosa = gaussian_filter1d(score_vector_dudosa, sigma_dudoso)
        else:
            # Smooth the signal using a moving average filter
            smoothed_signal_dudosa = np.zeros(60000)
            tamano_ventana_dudosa = 1200
            delante_dudosa = round(tamano_ventana_dudosa / 2)
            detras_dudosa = round(tamano_ventana_dudosa / 2)
            ventana_dudosa = np.zeros(delante_dudosa + detras_dudosa + 1)

            for z in range(len(smoothed_signal_dudosa)):
                for j in range(detras_dudosa):
                    if z - j - 1 >= 0:
                        ventana_dudosa[detras_dudosa - j] = score_vector_dudosa[z - j - 1]
                ventana_dudosa[detras_dudosa + 1] = score_vector_dudosa[z]
                for x in range(delante_dudosa):
                    if z + x + 1 < len(score_vector_dudosa):
                        ventana_dudosa[detras_dudosa + x + 1] = score_vector_dudosa[z + x + 1]
                smoothed_signal_dudosa[z] = np.mean(ventana_dudosa)

        ax2.clear()
        ax2.plot(score_vector_dudosa, label='Score original')
        ax2.plot(smoothed_signal_dudosa, label='Score suavizado')
        # Plot signal
        ax2.set_title(f'Score suavizado')
        ax2.set_xlabel('Tiempo (puntos)')
        ax2.set_ylabel('Amplitud')
        ax2.legend()
        ax2.grid()

        canvas2.draw()

        messagebox.showinfo(title="Suavizado", message=f"Suavizado con: {metodo_dudosa}")


    Label(p1, text="Suavizar con:", font=("Arial", 10, "bold"), bg='gainsboro').place(x=1190, y=649)
    suavizado_dudosa.bind("<<ComboboxSelected>>", suavizar_dudosa)


    # .......................................... RECOVER SIGNAL WINDOW 1 ...............................................
    def recuperar_dudosa(event):
        global smoothed_signal_dudosa, sig_dudosa_recuperada
        smoothed_signal_dudosa_neg = 1 - smoothed_signal_dudosa
        sig_dudosa_recuperada = smoothed_signal_dudosa_neg * doubt_dudosas[d, :]

        ax2.clear()
        ax2.plot(doubt_dudosas[d, :], label='Señal original', color='blue')
        ax2.plot(sig_dudosa_recuperada, label='Señal recuperada', color='red')
        # Plot signal
        ax2.set_title(f'Señal dudosa recuperada')
        ax2.set_xlabel('Tiempo (puntos)')
        ax2.set_ylabel('Amplitud')
        ax2.legend()
        ax2.grid()

        canvas2.draw()

        # Extract signal characteristics and update the information box
        characteristics_dudosa_recuperada = extract_characteristics(sig_dudosa_recuperada)[0]
        update_info_dudosa_recuperada(characteristics_dudosa_recuperada)


    # ............................................ SAVE SIGNAL WINDOW 1 .................................................
    def save_signal_recuperada_dudosa(event):
        global d, sig_dudosa_normalizada
        hf_recuperada_dudosa = h5py.File('RECUPERADA_DUDOSAS/' + 'sig_{}'.format(d) + '.h5', 'w')
        # Elements
        hf_recuperada_dudosa.create_dataset("feature", data=sig_dudosa_normalizada)
        hf_recuperada_dudosa.create_dataset("index", data=doubt_idx[d])

        # Close file
        hf_recuperada_dudosa.close()

        MessageBox.showinfo('Status', '¡Señal Guardada!')


    # ......................................... CLEAN GRAPHICS WINDOW 1 ................................................
    def volver_dudosa(event):
        global d
        ax1.clear()
        # Plot signal
        ax1.plot(doubt_dudosas[d, :].flatten(), label=f'Señal {d}')
        ax1.set_title("Label:{} Predicted:{} Score:{:.0%} DB_idx:{} ICA_pos:{}".format(labels_dudosas[doubt_idx[d]],
                                                                                       pred_dudosas[doubt_idx[d]],
                                                                                       np.max(prob_dudosas, axis=-1)[
                                                                                           doubt_idx[d]],
                                                                                       index_dudosas[doubt_idx[d]],
                                                                                       index_dudosas[
                                                                                           doubt_idx[d]] % 160 + 1))
        ax1.set_xlabel('Tiempo (puntos)')
        ax1.set_ylabel('Amplitud')
        ax1.legend()
        ax1.grid()

        canvas1.draw()

        # Extract signal characteristics and update the information box
        characteristics_dudosa = extract_characteristics(doubt_dudosas[d])[0]
        update_info_dudosa(characteristics_dudosa)

        ax2.clear()
        canvas2.draw()
        update_info_dudosa_recuperada({})



    # ...................................... ARTIFACT TYPE SELECTION WINDOW 2 ..........................................
    # Drop-down menu to select the type of device in the second window
    Label(p2, text="Selección artefacto: ", font=("Arial", 10, "bold")).place(x=150, y=390)
    combo = ttk.Combobox(state="readonly", values=["CARDIACA", "RED ELÉCTRICA", "OCULAR", "OTRO"], master=p2)
    combo.place(x=300, y=390)

    def selection_artifact_changed(event):
        global total_artefactos, signal_artefacto_total, label_artefacto_total, ica_index_artefacto_total, k
        k = 0
        selection = combo.get()
        messagebox.showinfo(title="Nuevo artefacto seleccionado", message=selection)
        if selection == "CARDIACA":
            label_value_artefacto = 1
        elif selection == "RED ELÉCTRICA":
            label_value_artefacto = 2
        elif selection == "OCULAR":
            label_value_artefacto = 3
        elif selection == "OTRO":
            label_value_artefacto = 4
        else:
            label_value_artefacto = 1  # Defaut cardiac
        indices_with_label_1 = np.where(labels_ind == label_value_artefacto)[0]
        total_artefactos = len(indices_with_label_1)

        signal_artefacto_list = []
        label_artefacto_list = []
        ica_index_artefacto_list = []

        for i, idx in enumerate(indices_with_label_1):
            # Extract the signal (a row from the features matrix)
            signal_artefacto = features_ind[idx, :]
            signal_artefacto_list.append(signal_artefacto)

            # Extract the corresponding label (in this case, it will always be 1) and the ICA index
            label_artefacto = labels_ind[idx]
            label_artefacto_list.append(label_artefacto)
            ica_index_artefacto = ica_idx[idx]
            ica_index_artefacto_list.append(ica_index_artefacto)

        signal_artefacto_total = np.array(signal_artefacto_list)
        label_artefacto_total = np.array(label_artefacto_list)
        ica_index_artefacto_total = np.array(ica_index_artefacto_list)
        update_signal_artefacto()


    combo.bind("<<ComboboxSelected>>", selection_artifact_changed)

    señal_artefacto_ajustada = None
    signal_with_mask = None
    suma = None


    # ................................ SMOOTHING TAB 2 .......................................
    # Drop-down menu to choose score smoothing
    suavizado = ttk.Combobox(state="readonly", values=["Gaussiana", "Media móvil"], master=p2)
    suavizado.place(x=1190, y=709)


    def suavizar(event):
        global score_vector, smoothed_signal
        metodo = suavizado.get()
        if metodo == 'Gaussiana':
            # Smooth the signal using a Gaussian filter
            sigma = 500  # Adjust sigma to control the level of smoothing
            # Sigma controls the amount of smoothing. A higher sigma smooths more, but reduces signal detail
            smoothed_signal = gaussian_filter1d(score_vector, sigma)
        else:
            # Smooth the signal using a moving average filter
            smoothed_signal = np.zeros(60000)
            tamano_ventana = 1200
            delante = round(tamano_ventana / 2)
            detras = round(tamano_ventana / 2)
            ventana = np.zeros(delante + detras + 1)

            for z in range(len(smoothed_signal)):
                for j in range(detras):
                    if z - j - 1 >= 0:
                        ventana[detras - j] = score_vector[z - j - 1]
                ventana[detras + 1] = score_vector[z]
                for x in range(delante):
                    if z + x + 1 < len(score_vector):
                        ventana[detras + x + 1] = score_vector[z + x + 1]
                smoothed_signal[z] = np.mean(ventana)

        ax4.clear()
        ax4.plot(score_vector, label='Score original')
        ax4.plot(smoothed_signal, label='Score suavizado')
        ax4.axvline(x=12000, color='black', linestyle='--', label='Minuto 1')
        ax4.axvline(x=24000, color='black', linestyle='--', label='Minuto 2')
        ax4.axvline(x=36000, color='black', linestyle='--', label='Minuto 3')
        ax4.axvline(x=48000, color='black', linestyle='--', label='Minuto 4')
        # Plot signal
        ax4.set_title(f'Score suavizado')
        ax4.set_xlabel('Tiempo (puntos)')
        ax4.set_ylabel('Amplitud')
        ax4.legend()
        ax4.grid()

        canvas4.draw()

        messagebox.showinfo(title="Suavizado", message=f"Suavizado con: {metodo}")


    Label(p2, text="Suavizar con:", font=("Arial", 10, "bold"), bg='gainsboro').place(x=1190, y=679)
    suavizado.bind("<<ComboboxSelected>>", suavizar)


    # Noisy segment selection: indicates START and END
    # Label and text box for “Start”
    label_inicio = tk.Label(info_frame_3a, text="Inicio:", bg="gainsboro", anchor="w", font=("Arial", 10, "bold"))
    label_inicio.grid(row=1, column=0, padx=5, pady=5, sticky="w")

    entry_inicio = Entry(info_frame_3a, width=10)
    entry_inicio.grid(row=1, column=1, padx=5, pady=5)

    # Label and text box for “End”
    label_fin = tk.Label(info_frame_3a, text="Fin:", bg="gainsboro", anchor="w", font=("Arial", 10, "bold"))
    label_fin.grid(row=1, column=2, padx=5, pady=5, sticky="w")

    entry_fin = Entry(info_frame_3a, width=10)
    entry_fin.grid(row=1, column=3, padx=5, pady=5)

    i = 0  # Index neuronals
    k = 0  # Index artefacts
    d = 0  # Index doubt

    info_entries_dudosa = {}
    info_entries_dudosa_recuperada = {}
    info_entries_neuro = {}
    info_entries_art = {}


    # Function to display characteristics in the information box
    # WINDOW 1
    def update_info_dudosa(characteristics_dudosa):
        font_size = ("Arial", 10)
        for idx_dudosa, (key, value) in enumerate(characteristics_dudosa.items()):
            # Define row and column according to index
            if idx_dudosa < 5:
                row = idx_dudosa + 1  # First column
                column_label = 0
                column_entry = 1
            else:
                row = idx_dudosa - 4  # Adjust the row for the second column
                column_label = 2
                column_entry = 3
            # Create label for feature if it does not exist
            if key not in info_entries_dudosa:
                # Descriptive label
                etiqueta_dudosa = tk.Label(info_frame_1, text=f"{key.capitalize()}:", bg="lightgreen", anchor="w",
                                           font=font_size)
                etiqueta_dudosa.grid(row=row, column=column_label, sticky="w", padx=5, pady=2)

                # Entry text box to display the value
                entry_dudosa = Entry(info_frame_1, width=10, font=font_size)
                entry_dudosa.grid(row=row, column=column_entry, padx=5, pady=2)
                info_entries_dudosa[key] = entry_dudosa

            # Update the value in the text box
            info_entries_dudosa[key].delete(0, tk.END)  # Delete the previous value
            info_entries_dudosa[key].insert(0, f"{value:.2f}")  # Insert the new value


    def update_info_dudosa_recuperada(characteristics_dudosa_recuperada):
        font_size = ("Arial", 10)
        # If the dictionary is empty, clear all input fields
        if not characteristics_dudosa_recuperada:
            for entry in info_entries_dudosa_recuperada.values():
                entry.delete(0, tk.END)  # Delete the value in each field
            return  # Exit the function after cleaning

        for idx_dudosa_recuperada, (key, value) in enumerate(characteristics_dudosa_recuperada.items()):
            # Define row and column according to index
            if idx_dudosa_recuperada < 5:
                row = idx_dudosa_recuperada + 1  # First column
                column_label = 0
                column_entry = 1
            else:
                row = idx_dudosa_recuperada - 4  # Adjust the row for the second column
                column_label = 2
                column_entry = 3
            # Create label for feature if it does not exist
            if key not in info_entries_dudosa_recuperada:
                # Descriptive label
                etiqueta_dudosa_recuperada = tk.Label(info_frame_2, text=f"{key.capitalize()}:", bg="lightgreen",
                                                      anchor="w", font=font_size)
                etiqueta_dudosa_recuperada.grid(row=row, column=column_label, sticky="w", padx=5, pady=2)

                # Entry text box to display the value
                entry_dudosa_recuperada = Entry(info_frame_2, width=10, font=font_size)
                entry_dudosa_recuperada.grid(row=row, column=column_entry, padx=5, pady=2)
                info_entries_dudosa_recuperada[key] = entry_dudosa_recuperada

            # Update the value in the text box
            info_entries_dudosa_recuperada[key].delete(0, tk.END)  # Delete the previous value
            info_entries_dudosa_recuperada[key].insert(0, f"{value:.2f}")  # Insert the new value


    # IN WINDOW 2  --> neuronal
    def update_info_neuro(characteristics_neuro):
        font_size = ("Arial", 10)
        for idx_neuro, (key, value) in enumerate(characteristics_neuro.items()):
            # Define row and column according to index
            if idx_neuro < 5:
                row = idx_neuro + 1  # First column
                column_label = 0
                column_entry = 1
            else:
                row = idx_neuro - 4  # Adjust the row for the second column
                column_label = 2
                column_entry = 3

            # Create label for feature if it does not exist
            if key not in info_entries_neuro:
                # Descriptive label (eg. "Mean:")
                etiqueta_neuro = tk.Label(info_frame_3, text=f"{key.capitalize()}:", bg="lightgreen", anchor="w",
                                          font=font_size)
                etiqueta_neuro.grid(row=row, column=column_label, sticky="w", padx=5, pady=2)

                # Entry text box to display the value
                entry_3 = Entry(info_frame_3, width=10, font=font_size)
                entry_3.grid(row=row, column=column_entry, padx=5, pady=2)
                info_entries_neuro[key] = entry_3

            # Update the value in the text box
            info_entries_neuro[key].delete(0, tk.END)  # Delete the previous value
            info_entries_neuro[key].insert(0, f"{value:.2f}")  # Insert the new value


    # IN WINDOW 2  --> artefact
    def update_info_art(characteristics_art):
        font_size = ("Arial", 10)
        for idx_art, (key, value) in enumerate(characteristics_art.items()):
            # Define row and column according to index
            if idx_art < 5:
                row = idx_art + 1  # First column
                column_label = 0
                column_entry = 1
            else:
                row = idx_art - 4  # Adjust the row for the second column
                column_label = 2
                column_entry = 3

            # Create label for feature if it does not exist
            if key not in info_entries_art:
                # Descriptive label (eg. "Mean:")
                etiqueta_art = tk.Label(info_frame_4, text=f"{key.capitalize()}:", bg="lightgreen", anchor="w",
                                        font=font_size)
                etiqueta_art.grid(row=row, column=column_label, sticky="w", padx=5, pady=2)

                # Entry text box to display the value
                entry_4 = Entry(info_frame_4, width=10, font=font_size)
                entry_4.grid(row=row, column=column_entry, padx=5, pady=2)
                info_entries_art[key] = entry_4

            # Update the value in the text box
            info_entries_art[key].delete(0, tk.END)  # Delete the previous value
            info_entries_art[key].insert(0, f"{value:.2f}")  # Insert the new value


    # Function to plot a signal given by the index i
    def update_signal_dudosa():
        global d
        ax1.clear()
        # Plot signal
        ax1.plot(doubt_dudosas[d, :].flatten(), label=f'Señal {d}')
        ax1.set_title("Label:{} Predicted:{} Score:{:.0%} DB_idx:{} ICA_pos:{}".format(labels_dudosas[doubt_idx[d]],
                                                                                       pred_dudosas[doubt_idx[d]],
                                                                                       np.max(prob_dudosas, axis=-1)[
                                                                                           doubt_idx[d]],
                                                                                       index_dudosas[doubt_idx[d]],
                                                                                       index_dudosas[
                                                                                           doubt_idx[d]] % 160 + 1))
        ax1.set_xlabel('Tiempo (puntos)')
        ax1.set_ylabel('Amplitud')
        ax1.legend()
        ax1.grid()

        canvas1.draw()

        # Extract signal characteristics and update the information box
        characteristics_dudosa = extract_characteristics(doubt_dudosas[d])[0]
        update_info_dudosa(characteristics_dudosa)


    def update_signal_neuro():
        global i
        ax3.clear()
        # Plot signal
        ax3.plot(signal_neuro_total[i, :].flatten(), label=f'Señal {i} de {total_neuro}')
        label_n = label_neuro_total[i]
        ica_index_n = ica_index_neuro_total[i]
        ax3.set_title(f'Signal {i} - Label: {label_n}, ICA Index: {ica_index_n}')
        ax3.set_xlabel('Tiempo (puntos)')
        ax3.set_ylabel('Amplitud')
        ax3.legend()
        ax3.grid()

        canvas3.draw()

        # Extract signal characteristics and update the information box
        characteristics_neuro = extract_characteristics(signal_neuro_total[i])[0]
        update_info_neuro(characteristics_neuro)


    def update_signal_artefacto():
        global k
        ax4.clear()
        # Plot signal
        ax4.plot(signal_artefacto_total[k, :].flatten(), label=f'Señal {k} de {total_artefactos}')
        label_c = label_artefacto_total[k]
        ica_index_c = ica_index_artefacto_total[k]
        ax4.set_title(f'Signal {k} - Label: {label_c}, ICA Index: {ica_index_c}')
        ax4.set_xlabel('Tiempo (puntos)')
        ax4.set_ylabel('Amplitud')
        ax4.legend()
        ax4.grid()

        canvas4.draw()

        # Extract signal characteristics and update the information box
        characteristics_art = extract_characteristics(signal_artefacto_total[k])[0]
        update_info_art(characteristics_art)


    def normalizar_dudosa(event):
        global sig_dudosa_recuperada, sig_dudosa_normalizada
        potencia_dudosa = np.mean(sig_dudosa_recuperada ** 2)
        sig_dudosa_normalizada = sig_dudosa_recuperada / np.sqrt(potencia_dudosa)

        # Represent
        ax2.clear()
        ax2.plot(doubt_dudosas[d, :], label='Señal original', color='blue')
        ax2.plot(sig_dudosa_normalizada, label='Señal normalizada', color='green')
        ax2.plot(sig_dudosa_recuperada, label='Señal recuperada', color='red')
        # Plot signal
        ax2.set_title(f'Señal dudosa recuperada')
        ax2.set_xlabel('Tiempo (puntos)')
        ax2.set_ylabel('Amplitud')
        ax2.legend()
        ax2.grid()

        canvas2.draw()

        # Extract signal characteristics and update the information box
        characteristics_dudosa_normalizada = extract_characteristics(sig_dudosa_normalizada)[0]
        update_info_dudosa_recuperada(characteristics_dudosa_normalizada)


    # GO TO THE NEXT SIGNAL
    def next_signal_neuro(event):
        global i
        if i < total_neuro:
            i += 1
            update_signal_neuro()
        else:
            MessageBox.showwarning("Alerta", "Ya has llegado a la última señal.")


    def next_signal_artefacto(event):
        global k
        if k < total_artefactos:
            k += 1
            update_signal_artefacto()
        else:
            MessageBox.showwarning("Alerta", "Ya has llegado a la última señal.")


    def next_signal_dudosa(event):
        global d
        if d < total_dudosas:
            d += 1
            update_signal_dudosa()
        else:
            MessageBox.showwarning("Alerta", "Ya has llegado a la última señal.")


    # RETURN TO PREVIOUS SIGN
    def previous_signal_neuro(event):
        global i
        if i > 0:
            i -= 1
            update_signal_neuro()
        else:
            MessageBox.showwarning("Alerta", "Ya has llegado a la primera señal.")


    def previous_signal_artefacto(event):
        global k
        if k > 0:
            k -= 1
            update_signal_artefacto()
        else:
            MessageBox.showwarning("Alerta", "Ya has llegado a la primera señal.")


    def previous_signal_dudosa(event):
        global d
        if d > 0:
            d -= 1
            update_signal_dudosa()
        else:
            MessageBox.showwarning("Alerta", "Ya has llegado a la primera señal.")


    # SAVE SIGNALS IN FOLDERS
    def save_signal_neuro(event):
        global i
        hf_neuro = h5py.File('Cerebral/' + 'sig_{}'.format(i) + '.h5', 'w')
        # Elements
        hf_neuro.create_dataset("feature", data=signal_neuro_total[i, :])
        hf_neuro.create_dataset("label", data=label_neuro_total[i])
        hf_neuro.create_dataset("ica_index", data=ica_index_neuro_total[i])

        # Close file
        hf_neuro.close()

        MessageBox.showinfo('Status', 'Señal Guardada!')


    def save_signal_art(event):
        global k
        hf_artifact = h5py.File('Artefacto/' + 'sig_{}'.format(k) + '.h5', 'w')
        # Elements
        hf_artifact.create_dataset("feature", data=signal_artefacto_total[k, :])
        hf_artifact.create_dataset("label", data=label_artefacto_total[k])
        hf_artifact.create_dataset("ica_index", data=ica_index_artefacto_total[k])

        # Close file
        hf_artifact.close()

        MessageBox.showinfo('Status', 'Señal Guardada!')

    def repetir_segmento(segmento, N):
        segmento = np.array(segmento)
        parte_entera = int(np.floor(N))
        parte_decimal = N - parte_entera

        # Repeat the whole number
        array_resultado = np.tile(segmento, parte_entera)

        # Decimal part
        if parte_decimal > 0:
            n_muestras_extra = int(round(parte_decimal * len(segmento)))
            array_resultado = np.concatenate([array_resultado, segmento[:n_muestras_extra]])

        return array_resultado

    def make_segments(event):
        global k, repeated_signal
        fs = 200  # Sample frequency (Hz)
        duration = 300  # Duration in seconds
        total_samples = duration * fs  # Total samples = 60000

        inicio = int(entry_inicio.get())
        fin = int(entry_fin.get())
        segmento = signal_artefacto_total[k, :][inicio:fin+1]
        repeticiones = total_samples / len(segmento)
        repeated_signal = repetir_segmento(segmento, repeticiones)

        ax4.clear()
        # Plot signal
        ax4.plot(repeated_signal.flatten(), label=f'Señal sintética')
        ax4.set_title(f'Señal sintética')
        ax4.set_xlabel('Tiempo (puntos)')
        ax4.set_ylabel('Amplitud')
        ax4.legend()
        ax4.grid()

        canvas4.draw()

        # Extract signal characteristics and update the information box
        characteristics_repeated_signal = extract_characteristics(repeated_signal)[0]
        update_info_art(characteristics_repeated_signal)

    def mask_to_signal(event):
        global repeated_signal, signal_with_mask
        signal_with_mask = mask * repeated_signal

        ax4.clear()
        # Plot signal
        ax4.plot(signal_with_mask.flatten(), label=f'Señal sintética')
        ax4.set_title(f'Señal sintética con máscara')
        ax4.set_xlabel('Tiempo (puntos)')
        ax4.set_ylabel('Amplitud')
        ax4.legend()
        ax4.grid()

        canvas4.draw()

        # Extract signal characteristics and update the information box
        characteristics_signal_with_mask = extract_characteristics(signal_with_mask)[0]
        update_info_art(characteristics_signal_with_mask)


    # Drop-down menu to select artifact preprocessing --> WINDOW 2
    porcentaje = ttk.Combobox(state="readonly",
                              values=["Sin ajuste", "10 %", "25 %", "50 %", "75 %", "100 %", "125 %", "150 %", "175 %",
                                      "200 %", "250 %", "300 %"],
                              master=p2)
    porcentaje.place(x=1294, y=325)


    def selection_percentage_changed(event):
        global i, k, signal_with_mask, señal_artefacto_ajustada
        potencia_neuronal = np.mean(signal_neuro_total[i, :] ** 2)
        potencia_artefacto = np.mean(signal_with_mask ** 2)
        # Get the selected percentage from the combo box and convert it to a numeric value (e.g., “10%” -> 0.10)
        umbral_str = porcentaje.get()
        # If “No adjustment” is selected, display the original artifact signal
        if umbral_str == "Sin ajuste":
            señal_artefacto_ajustada = signal_with_mask
            messagebox.showinfo(title="Señal sin ajuste", message="Señal artefacto sin cambios")
        else:
            umbral = float(umbral_str.replace(" %", "")) / 100  # Convert to a value between 0 and 1

            # Scale the artifact signal using the adjustment factor
            factor = (potencia_neuronal * umbral) / potencia_artefacto
            señal_artefacto_ajustada = signal_with_mask * factor

            messagebox.showinfo(title="Nuevo porcentaje seleccionado",
                                message=f"Umbral seleccionado: {umbral_str}")

        ax4.clear()
        # Plot signal
        ax4.plot(señal_artefacto_ajustada.flatten(), label=f'Señal {k} ajustada')
        ax4.set_title(f'Señal cardiaca')
        ax4.set_xlabel('Tiempo (puntos)')
        ax4.set_ylabel('Amplitud')
        ax4.legend()
        ax4.grid()

        canvas4.draw()

        characteristics_art_ajustada = extract_characteristics(señal_artefacto_ajustada)[0]
        update_info_art(characteristics_art_ajustada)


    Label(p2, text="Ajuste artefacto:", font=("Arial", 10, "bold"), bg='gainsboro').place(x=1294, y=295)
    porcentaje.bind("<<ComboboxSelected>>", selection_percentage_changed)

    def sum_signals(event):
        global señal_artefacto_ajustada, suma, i
        suma = signal_neuro_total[i, :] + señal_artefacto_ajustada
        ax3.clear()
        # Plot signal
        ax3.plot(suma.flatten(), label=f'Señal suma')
        ax3.set_title(f'Señal suma')
        ax3.set_xlabel('Tiempo (puntos)')
        ax3.set_ylabel('Amplitud')
        ax3.legend()
        ax3.grid()

        canvas3.draw()

        # Extract signal characteristics and update the information box
        characteristics_suma = extract_characteristics(suma)[0]
        update_info_neuro(characteristics_suma)



    # Drop-down menu for selecting signal sum segmentation
    segmentado = ttk.Combobox(state="readonly", values=["1", "5", "10", "20"], master=p2)
    segmentado.place(x=1190, y=400)


    def segmentation(event):
        global suma
        # Parameters
        fs = 200  # Sample frequency (Hz)
        duration = 300  # Duration in seconds
        total_samples = duration * fs  # Total samples = 60000
        segment_duration_str = segmentado.get()  # Duration of segments in seconds
        segment_duration = int(segment_duration_str)
        segment_samples = segment_duration * fs  # Samples per segment = 200
        num_segments = total_samples // segment_samples  # 300 segments
        # Segmentation of the signal into segments of x seconds
        segments = suma.reshape(-1, segment_samples)
        # Create an HDF5 file to store the 300 output signals
        with h5py.File('Segmentacion/' + 'segmentos.h5', 'w') as hdf_segments:
            # Create a dataset within the HDF5 file to store the signals (300 signals of 60000 samples)
            dset = hdf_segments.create_dataset('signals', (num_segments, total_samples), dtype='float64')

            # For each of the 300 segments
            for j in range(num_segments):
                # Repeat the current segment until reaching 60000 samples
                repeated_signal = np.tile(segments[j], total_samples // segment_samples)

                # Store the repeated signal in the HDF5 dataset
                dset[j] = repeated_signal

        hdf_segments.close()
        messagebox.showinfo(title="Segmentos", message=f"Segmentos de: {segment_duration} s")


    Label(p2, text="Segmentar en [s]:", font=("Arial", 10, "bold"), bg='gainsboro').place(x=1190, y=370)
    segmentado.bind("<<ComboboxSelected>>", segmentation)


    def clasificar(event):
        global score_vector
        # Open the .h5 file in read mode
        file_classify = h5py.File('Segmentacion/' + 'segmentos' + '.h5', 'r')
        signals = np.array(file_classify.get('signals'))
        file_classify.close()

        # Signals are converted to a matrix of size (Yx60000x1x1) because EEGInception requires this format
        signals = np.expand_dims(signals, axis=-1)
        signals = np.expand_dims(signals, axis=-1)

        # Load the model directly using load_model
        new_model = keras.models.load_model('Modelo/' + 'my_model.keras')
        # MODEL INPUTS:
        # 1. Name: features, Form: (None, None, 1, 1)
        # 2. Name: ICA, Form: (None, None, 1, 1)

        # Generate predictions for samples
        ica_dummy = np.zeros_like(signals)  # Adjust size according to expected ICA shape
        probabilidades = new_model.predict({'features': signals, 'ICA': ica_dummy})

        predicciones = probabilidades.argmax(axis=-1)
        predicciones = (predicciones > 0).astype(int)

        # RECONSTRUCTION OF THE ORIGINAL SIGNAL
        # Assuming each segment has 60000 data points
        n_segmentos = signals.shape[0]
        segmento_length = signals.shape[1]
        segmento_length_real = int(60000 / n_segmentos)  # (1 second of original signal at 200 Hz)

        # Extract only the first 200 points of each segment
        señal_reconstruida = signals[:, :segmento_length_real, 0, 0].reshape(n_segmentos * segmento_length_real)

        # PLOT
        ax3.clear()
        ax4.clear()
        # Plot signal
        ax3.plot(señal_reconstruida.flatten(), label=f'Señal reconstruida')
        ax3.axvline(x=12000, color='black', linestyle='--', label='Minuto 1')
        ax3.axvline(x=24000, color='black', linestyle='--', label='Minuto 2')
        ax3.axvline(x=36000, color='black', linestyle='--', label='Minuto 3')
        ax3.axvline(x=48000, color='black', linestyle='--', label='Minuto 4')

        mostrado_rojo = False
        mostrado_naranja = False
        mostrado_amarillo = False
        score_vector = np.zeros(60000)
        # Go through the segments and plot each one
        for i in range(n_segmentos):
            inicio = i * segmento_length_real
            fin = inicio + segmento_length_real

            probabilidad_artefacto = probabilidades[i][
                1]  # Obtain the probability of the segment being an artefact (class 1)

            if predicciones[i] == 1:
                if probabilidad_artefacto > 0.9:
                    color = '#D62728'  # Most likely artifact
                    label = 'probabilidad > 0.9' if not mostrado_rojo else ""
                    mostrado_rojo = True
                elif 0.75 < probabilidad_artefacto <= 0.9:
                    color = '#FF7F0E'  # Intermediate probability of artefact
                    label = '0.75 < probabilidad < 0.9' if not mostrado_naranja else ""
                    mostrado_naranja = True
                elif 0.5 < probabilidad_artefacto <= 0.75:
                    color = '#FFD700'  # Low probability of artefact
                    label = '0.5 < probabilidad < 0.75' if not mostrado_amarillo else ""
                    mostrado_amarillo = True

                ax3.axvspan(inicio, fin, color=color, alpha=0.3, label=label)

                # GRAPHIC SCORE
                for j in range(inicio, fin):
                    score_vector[j] = probabilidades[i][1]

        ax4.plot(score_vector, label='Scores')
        ax4.axvline(x=12000, color='black', linestyle='--', label='Minuto 1')
        ax4.axvline(x=24000, color='black', linestyle='--', label='Minuto 2')
        ax4.axvline(x=36000, color='black', linestyle='--', label='Minuto 3')
        ax4.axvline(x=48000, color='black', linestyle='--', label='Minuto 4')

        ax3.set_title(f'Clasificación')
        ax3.set_xlabel('Tiempo (puntos)')
        ax3.set_ylabel('Amplitud')
        ax3.legend()
        ax3.grid()

        canvas3.draw()

        # Plot signal
        ax4.set_title(f'Score')
        ax4.set_xlabel('Tiempo (puntos)')
        ax4.set_ylabel('Amplitud')
        ax4.legend()
        ax4.grid()

        canvas4.draw()


    def recuperar(event):
        global smoothed_signal, suma, sig_recuperada
        smoothed_signal_neg = 1 - smoothed_signal
        sig_recuperada = smoothed_signal_neg * suma

        ax4.clear()
        ax4.plot(suma, label='Señal original', color='blue')
        ax4.plot(sig_recuperada, label='Señal recuperada', color='red')
        # Plot signal
        ax4.set_title(f'Señal recuperada')
        ax4.set_xlabel('Tiempo (puntos)')
        ax4.set_ylabel('Amplitud')
        ax4.legend()
        ax4.grid()

        canvas4.draw()

        # Extract signal characteristics and update the information box
        characteristics_recuperada = extract_characteristics(sig_recuperada)[0]
        update_info_art(characteristics_recuperada)


    # Extract signal characteristics
    def extract_characteristics(signal):
        characteristics = {}
        basic_characteristics = {}

        # Basic statistics
        characteristics['mean'] = np.mean(signal)
        characteristics['std'] = np.std(signal)
        characteristics['min'] = np.min(signal)
        basic_characteristics['min'] = np.min(signal)
        characteristics['max'] = np.max(signal)
        basic_characteristics['max'] = np.max(signal)
        characteristics['median'] = np.median(signal)

        # Time domain
        characteristics['potencia'] = np.mean(signal ** 2)
        basic_characteristics['potencia'] = np.mean(signal ** 2)
        characteristics['rms'] = np.sqrt(np.mean(signal ** 2))
        characteristics['zero_crossings'] = np.sum(np.diff(np.sign(signal)) != 0)

        # Frequency domain
        fft = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal))
        characteristics['dominant_freq'] = freqs[np.argmax(np.abs(fft))]

        return characteristics, basic_characteristics



    # BUTTON FUNCTION
    def bttn(master, x, y, text, bcolor, fcolor, command):
        def on_enter(e):
            mybutton['background'] = bcolor
            mybutton['foreground'] = fcolor

        def on_leave(e):
            mybutton['background'] = fcolor
            mybutton['foreground'] = bcolor

        mybutton = Button(master, height=2, text=text,
                          padx=15,
                          fg=bcolor,
                          bg=fcolor,
                          border=2,
                          font=("Arial", 10, "bold"),
                          activeforeground=fcolor,
                          activebackground=bcolor,
                          command=None)
        mybutton.bind("<Enter>", on_enter)
        mybutton.bind("<Leave>", on_leave)
        mybutton.bind("<Button-1>", command)
        mybutton.place(x=x, y=y)


    bttn(p1, 530, 380, '<--', 'black', 'lightgreen', previous_signal_dudosa)
    bttn(p1, 610, 380, '-->', 'black', 'lightgreen', next_signal_dudosa)
    bttn(p1, 1370, 274, 'CLASIFICAR', 'black', 'coral', clasificar_dudosa)
    bttn(p1, 1370, 659, 'RECUPERAR', 'black', 'coral', recuperar_dudosa)
    bttn(p1, 1370, 715, 'Z-SCORE inv', 'black', 'lightgreen', normalizar_dudosa)
    bttn(p1, 1290, 795, 'GUARDAR', 'black', 'gainsboro', save_signal_recuperada_dudosa)
    bttn(p1, 10, 380, 'VOLVER', 'black', 'gainsboro', volver_dudosa)
    bttn(p2, 530, 380, '<--', 'black', 'lightgreen', previous_signal_neuro)
    bttn(p2, 610, 380, '-->', 'black', 'lightgreen', next_signal_neuro)
    bttn(p2, 10, 380, 'GUARDAR', 'black', 'gainsboro', save_signal_neuro)
    bttn(p2, 1180, 795, '<--', 'black', 'lightgreen', previous_signal_artefacto)
    bttn(p2, 1260, 795, '-->', 'black', 'lightgreen', next_signal_artefacto)
    bttn(p2, 1370, 795, 'GUARDAR', 'black', 'gainsboro', save_signal_art)
    bttn(p2, 1435, 225, 'IR', 'black', 'coral', make_segments)
    bttn(p2, 1185, 300, 'MÁSCARA', 'black', 'lightgreen', mask_to_signal)
    bttn(p2, 1440, 300, 'SUMA', 'black', 'lightgreen', sum_signals)
    bttn(p2, 1380, 380, 'CLASIFICAR', 'black', 'coral', clasificar)
    bttn(p2, 1370, 700, 'RECUPERAR', 'black', 'coral', recuperar)

    nb.add(p1, text='Doubt Signals')
    nb.add(p2, text='Synthetic Signals')

    # Display the first signal when starting
    update_signal_neuro()
    update_signal_artefacto()
    update_signal_dudosa()

    # Start the main Tkinter loop
    root.mainloop()
