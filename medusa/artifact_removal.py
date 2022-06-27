import mne
from mne.preprocessing import ICA
from medusa.meeg import *

class MEDUSICA:
    def __init__(self,n_components=None, max_iter='auto', l_cha=None, fs=None,
          biosignal='eeg', montage=None, method='fastica'):

        # Attributes:
        self.n_components = n_components
        self.max_iter = max_iter
        self.l_cha = l_cha
        self.fs = fs
        self.biosignal = biosignal
        self.montage = montage
        self.method = method

        self.check_attributes()
        self.ica = ICA(n_components=self.n_components, max_iter=self.max_iter,
                       method=self.method)

    def check_attributes(self):

        if self.n_components is None:
            if self.l_cha is not None and isinstance(self.l_cha, list):
                self.n_components = len(self.l_cha)
            else:
                raise ValueError("The list with the channel labels "
                                 "has to be defined. ")
        if self.max_iter != 'auto' and not isinstance(self.max_iter, int):
            raise ValueError("max_iter parameter must be an integer value")

        if self.biosignal != 'eeg' and self.biosignal != 'meeg':
            raise ValueError("Only EEG and MEG signals are supported.")

        if self.montage is None:
            raise ValueError("Montage standard kay must be definded")

        if self.fs is None:
            raise ValueError("Sampling rate (in Hz) must be definded")

    def apply_ica(self, signal_filtered, plot_components=True,
                  plot_overlay=False):

        # ICA Process
        info = mne.create_info(self.l_cha, self.fs, self.biosignal)
        signal_T = signal_filtered.T
        signal_mne = mne.io.RawArray(signal_T, info)
        signal_mne.set_montage(self.montage)
        self.ica.fit(signal_mne)

        # Plots
        if plot_components:
            self.ica.plot_components()
        if plot_overlay:
            self.ica.plot_overlay(signal_mne)
        self.ica.plot_sources(signal_mne, block=True)

        input_ica_components_to_remove = input(
            'Select ICA components to remove (Split by "," please): ').split(
            ',')
        ica_compontents_to_remove = []

        if len(input_ica_components_to_remove) == 1 and \
                input_ica_components_to_remove[0] != '':
            for component in input_ica_components_to_remove:
                ica_compontents_to_remove.append(int(component))

        self.ica.exclude = ica_compontents_to_remove

        reconst_signal = signal_mne.copy()
        self.ica.apply(reconst_signal)
        del signal_mne
        cleaned_signal = reconst_signal.get_data()
        return cleaned_signal.T

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
