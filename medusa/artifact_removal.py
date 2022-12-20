import numpy as np

import epoching
from medusa.plots.timeplot import time_plot
import components

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
        self._max_pca_components = None
        self.n_pca_components = None
        self._pca_mean = None
        self._pca_explained_variance = None
        self._pca_explained_variance_ratio = None
        self._pca_components = None

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

        signal = self._pre_whiten(signal)

        pca = PCA(n_components=self._max_pca_components, whiten=True,
                  random_state=self.random_state)
        signal_pca = pca.fit_transform(signal)

        self._pca_mean = pca.mean_
        self._pca_components = pca.components_
        self._pca_explained_variance = pca.explained_variance_
        self._pca_explained_variance_ratio = pca.explained_variance_ratio_
        self.n_pca_components = pca.n_components
        del pca

        # if n_pca is None:
        #     n_pca = len(self._pca_explained_variance_ratio)

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
                    f'explained variance, your threshold '
                    'results in 1 component. You should select '
                    'a higher value.')
        else:
            if not isinstance(self.n_components, int):
                raise ValueError(
                    f'n_components={self.n_components} must be None,'
                    f'float or int value')

        # TODO Checkear esto
        # if self.n_components > self.n_pca_components:
        #     raise ValueError()

        self.ica_labels = [f"ICA_{n}" for n in range(self.n_components)]

        return signal_pca

    def fit(self, signal, n_components):
        from sklearn.decomposition import FastICA
        from scipy import linalg

        self.n_components = n_components

        signal, _ = self._check_signal_dimensions(signal)

        self._get_pre_whitener(signal)
        signal = self._whitener(signal)

        ica = FastICA(whiten=False, random_state=self.random_state, tol=1e-6)
        ica.fit(signal[:, :self.n_components])
        self.unmixing_matrix = ica.components_
        self._ica_n_iter = ica.n_iter_

        assert self.unmixing_matrix.shape == (self.n_components,) * 2

        # Whitening unmixing matrix
        norm = np.sqrt(self._pca_explained_variance[:self.n_components])
        norm[norm == 0] = 1.
        self.unmixing_matrix /= norm
        self.mixing_matrix = linalg.pinv(self.unmixing_matrix)

    def transform(self, signal):
        if not hasattr(self, 'mixing_matrix'):
            raise RuntimeError('ICA has not been fitted yet. Please, fit ICA.')

        signal, _ = self._check_signal_dimensions(signal)

        signal = self._pre_whiten(signal)
        if self._pca_mean is not None:
            signal -= self._pca_mean

        # Transform signal to PCA space and then apply unmixing matrix
        pca_transform = np.dot(self._pca_components[:self.n_components],
                               signal.T)
        sources = np.dot(self.unmixing_matrix, pca_transform)

        return sources.T

    def rebuild(self, signal, exclude=None):

        signal, n_epo_samples = self._check_signal_dimensions(signal)

        signal = self._pre_whiten(signal)

        if exclude is not None:
            if isinstance(exclude,int):
                exclude = [exclude]
            exclude = np.array(list(set(exclude)))
            if len(np.where(exclude>self.n_components)[0]) > 0:
                raise ValueError("One or more ICA component keys that you have"
                                 "marked to exclude from signal rebuild are "
                                 "greater than the total number of "
                                 "ICA components.")
            self.components_excluded = exclude

        # Apply PCA
        if self._pca_mean is not None:
            signal -= self._pca_mean

        # Determine ica components to keep in signal rebuild
        c_to_keep = np.setdiff1d(np.arange(self.n_components),exclude)

        # Define projection matrix
        proj = np.dot(np.dot(self._pca_components[:self.n_components].T,
                             self.mixing_matrix[:,c_to_keep]),
                      np.dot(self.unmixing_matrix[c_to_keep,:],
                             self._pca_components[:self.n_components]))

        # Apply projection to signal
        signal_rebuilt = np.transpose(np.dot(proj,signal.T))

        if self._pca_mean is not None:
            signal_rebuilt += self._pca_mean

        signal_rebuilt *= self.pre_whitener

        # Restore epochs if original signal was divided in epochs
        if n_epo_samples is not None:
            signal_rebuilt = np.reshape(signal_rebuilt,
                                        (int(signal_rebuilt.shape[0]/n_epo_samples)
                                         ,n_epo_samples,signal_rebuilt.shape[1]))


        return signal_rebuilt

    def get_components(self):
        # Transform
        components = np.dot(self.mixing_matrix[:, :self.n_components].T,
                            self._pca_components[:self.n_components]).T

        return components

    def save(self, path):
        if not hasattr(self, 'mixing_matrix'):
            raise RuntimeError('ICA has not been fitted yet. Please, fit ICA.')

        ica_data = ICAData(self.pre_whitener,self.unmixing_matrix,
                           self.mixing_matrix, self.n_components,
                           self._pca_components,self._pca_mean,
                           self.components_excluded,
                           self.random_state)

        ica_data.save(path,'bson')

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

    @staticmethod
    def _exp_var_ncomp(var, n):
        cvar = np.asarray(var, dtype=np.float64)
        cvar = cvar.cumsum()
        cvar /= cvar[-1]
        # We allow 1., which would give us N+1
        n = min((cvar <= n).sum() + 1, len(cvar))
        return n, cvar[n - 1]

class ICAData(components.SerializableComponent):
    def __init__(self,pre_whitener=None,unmixing_matrix=None,mixing_matrix=None,
                 n_components=None, pca_components=None, pca_mean=None,
                 components_excluded=None,random_state=None):

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


if __name__ == "__main__":
    import scipy.signal as ss
    import medusa

    np.random.seed(10)
    T = 10
    fs = 250
    time = np.linspace(0, T, fs * T)

    s1 = np.sin(2 * time)  # Signal 1 : sinusoidal signal
    s2 = np.sign(np.sin(3 * time))  # Signal 2 : square signal
    s3 = ss.sawtooth(2 * np.pi * time)  # Signal 3: saw tooth signal

    S = np.c_[s1, s2, s3]
    S += 0.2 * np.random.normal(size=S.shape)

    A = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])  # Mixing matrix
    signal = np.dot(S, A.T)

    ica = ICA(n_components=3,random_state=0)
    ica.fit(signal)
    sources = ica.transform(signal)
    time_plot(sources, fs, ica.ica_labels)
    time_plot(S, fs)
    rebuilt = ica.rebuild(signal,0)
    time_plot(rebuilt,fs)
    ica.save('C:/Users\Diego\Downloads\ica')
    ica.load('C:/Users\Diego\Downloads\ica')

    path = 'C:/Users\Diego\Documents\Tesis\Estudios\FM-Theta\Sujetos\S13\Sesiones\s_2/registros/resting_pre.rec.bson'
    # from medusa import components
    # from medusa.meeg import meeg
    #
    # file = components.Recording.load(path)
    # # time_plot(file.eeg.signal)
    # from medusa.frequency_filtering import IIRFilter
    #
    # time_stamps_conditions = np.array([0,3750,3750,7500,7500,11250,11250,15000])
    # time_stamps_events= np.array(
    #     [256, 4500, 5678, 7540, 8975, 11242, 16852, 18900])
    # times_c = file.eeg.times[time_stamps_conditions] - file.eeg.times[time_stamps_conditions][0]
    # times_e = file.eeg.times[time_stamps_events] - \
    #           file.eeg.times[time_stamps_events][0]
    #
    # conditions_dict = {'conditions':{'ce':{'desc_name':'Closed eyes','label':0},
    #                                  'oe':{'desc_name':'Open eyes','label':1}},
    #                    'condition_labels':[0,0,1,1,0,0,1,1],
    #                    'condition_times':times_c}
    #
    # events_dict = {
    #     'events': {'eye': {'desc_name': 'Eye Blink', 'label': 0},
    #                    'noise': {'desc_name': 'Jaw noise', 'label': 1},
    #                'pop':{'desc_name':'Pop','label':2}},
    #     'event_labels': [0, 0, 0, 1, 0, 2, 0, 1],
    #     'event_times': times_e}
    #
    # iir = IIRFilter(2, [0.1, 40], 'bandpass')
    # iir.fit(256, 16)
    # s = iir.transform(file.eeg.signal)
    # ss = epoching.get_epochs(s,512)
    # f,a = time_plot(s, 256,time_to_show=30, ch_labels=file.eeg.channel_set.l_cha,ch_to_show=8,
    #           conditions_dict=conditions_dict,events_dict=events_dict)
    #
    # import scipy.io as sio
    #
    # data = sio.loadmat('C:/Users\Diego\Downloads/Apartado4-1-2_ECG.mat')
    # signal = data['ECG'][0][0][0]
    # s = np.repeat(signal,5,axis=1)
    # s[:,2] *= 5
    # time_plot(s,512,channel_offset=1)



