"""
In this module you will find useful functions and classes to apply on-line
Neurofeedback models. Each model is based on different features to be used
as target to train. Enjoy!

@author: Diego Marcos-Martínez
"""

# Built-in imports
from abc import ABC, abstractmethod
import concurrent

# External imports
import numpy as np
# from scipy import signal
# from tqdm import tqdm

# Medusa imports
import medusa as mds
from medusa import components
from medusa import meeg
from medusa.spatial_filtering import LaplacianFilter, car
from medusa.connectivity.phase_connectivity import phase_connectivity
from medusa.connectivity.amplitude_connectivity import aec
from medusa.graph_theory import degree


class SignalPreprocessing(components.ProcessingMethod):
    """
    Common preprocessing applied in Neurofeedback applications.
    It is composed by a frequency IIR filter followed by a Laplacian spatial
    filter. Functions are adapted to filter the signal in more than one frequency
    range, if necessary.
    """

    def __init__(self, filter_dict=None, montage=None, target_channels=None,
                 laplacian=False, car=False, n_cha_lp=None):
        super().__init__(fit_transform_signal=['signal'],
                         transform_signal=['signal'])

        # Error check
        if not filter_dict:
            raise ValueError('[SignalPreprocessing] Filter bank parameter '
                             '"filter_dict" must be a list containing all '
                             'necessary information to perform the filtering!. '
                             'The information should be: order, cutoff, btype and'
                             'filt_method.')

        for filter in filter_dict:
            if not isinstance(filter, dict):
                raise ValueError('[SignalPreprocessing] Each filter must '
                                 'be a dict()!')
            if 'order' not in filter or \
                    'cutoff' not in filter or \
                    'filt_method' not in filter or \
                    'btype' not in filter:
                raise ValueError('[SignalPreprocessing] Each filter must '
                                 'be a dict() containing the following keys: '
                                 '"order", "cutoff", "filt_method" and "btype"!')

        if not montage:
            raise ValueError('[SignalPreprocessing] Pre-processing parameter'
                             '"montage" must be a dict containing all'
                             'labels of channels and montage standard key')
        if laplacian and target_channels is None:
            raise ValueError('[SignalPreprocessing] Laplacian filter needs to '
                             'define "target_channels" parameter.')

        # if not target_channels:
        #     raise ValueError('[SignalPreprocessing] Pre-processing parameter'
        #                      '"target_channels" must be a list containing all'
        #                      'labels of channels to extract NFT features')
        # Parameters
        self.filter_dict = filter_dict
        self.l_cha = montage.l_cha
        self.target_channels = target_channels
        self.n_cha_lp = n_cha_lp
        self.montage = montage
        self.perform_car = car
        self.perform_laplacian = laplacian

        # Variables
        self.filter_dict_iir_filters = None
        self.offset_removal = None
        self.laplacian_filter = None

    def fit(self, fs):
        """
        Fits the IIR filter and Laplacian spatial filter.

        Parameters
        ----------
        fs: float
            Sampling rate in Hz.
        """

        # Fit Spectral Filters
        self.filter_dict_iir_filters = []

        self.offset_removal = mds.IIRFilter(order=2,
                                            cutoff=[1, 40],
                                            btype='bandpass',
                                            filt_method=self.filter_dict[0]
                                            ['filt_method'])
        self.offset_removal.fit(fs, len(self.l_cha))

        for filter in self.filter_dict:
            iir = mds.IIRFilter(order=filter['order'],
                                cutoff=filter['cutoff'],
                                btype=filter['btype'],
                                filt_method=filter['filt_method'])
            if self.target_channels is None:
                iir.fit(fs, len(self.l_cha))
            else:
                iir.fit(fs, len(self.target_channels))
            self.filter_dict_iir_filters.append(iir)

        # Fit Laplacian Filter
        if self.perform_laplacian:
            self.laplacian_filter = LaplacianFilter(self.montage, mode='auto')
            self.laplacian_filter.fit_lp(n_cha_lp=self.n_cha_lp,
                                         l_cha_to_filter=
                                         self.target_channels)

    def transform_signal(self, signal, parallel_computing=True):
        """
        Transforms an EEG signal applying IIR filter. It also applies CAR and
        Laplacian spatial filter sequentially if desired.

        Parameters
        ----------
        signal: list or numpy.ndarray
            Signal to transform. Shape [n_samples x n_channels]
        parallel_computing: bool
            If true, it filters the signal concurrently

        Returns
        -------
        signals: list
            List containing as many arrays of filtered signals as frequency
            bands set to filter.
        """
        # Initialize signals array
        n_samples = signal.shape[0]
        if self.target_channels is None:
            signals = np.empty(
                (len(self.filter_dict), n_samples, len(self.l_cha)))
        else:
            signals = np.empty((len(self.filter_dict), n_samples,
                                len(self.target_channels)))

        signal_ = self.offset_removal.transform(signal)
        if self.perform_car:
            signal_ = car(signal_)
        if self.perform_laplacian:
            signal_ = self.laplacian_filter.apply_lp(signal_)
        signal__ = signal_.copy()
        if parallel_computing:
            filt_threads = []
            for filter in self.filter_dict_iir_filters:
                t = components.ThreadWithReturnValue(target=filter.transform,
                                                     args=(signal__,))
                filt_threads.append(t)
                t.start()

            for filt_idx, thread in enumerate(filt_threads):
                signals[filt_idx, :, :] = thread.join()
        else:
            for filt_idx, filter in enumerate(self.filter_dict_iir_filters):
                signals[filt_idx, :, :] = filter.transform(signal__)
        return signals

    def fit_transform_signal(self, fs, signal):
        """
        Fits the IIR filter and transforms an EEG signal applying IIR
        filter and Laplacian spatial filter sequentially

        Parameters
        ----------
        fs: float
            Sampling rate in Hz.
        n_cha_lp: int
            Number of nearest channels to compute Laplacian spatial
            filter (Auto mode).
        signal: list or numpy.ndarray
            Signal to transform. Shape [n_samples x n_channels]

        Returns
        -------
        signals: list
            List containing as many arrays of filtered signals as frequency
            bands set to filter.
        """
        self.fit(fs)
        return self.transform_signal(signal)


class ConnectivityExtraction(components.ProcessingMethod):
    """
    Functional Connectivity-based features to extract from user's EEG.
    """

    def __init__(self, l_baseline_t=5, fs=250, update_feature_window=2,
                 fc_measure=None, mode=None, montage=None,
                 target_channels=None):
        """
        Class constructor

        l_baseline_t: int
            Time employed to calculate the number of samples to obtain baseline
            connectivity parameter. In seconds.
        fs: int or float
            Sample rate of the recording.
        update_feature_window: int
            Length in seconds of the temporal window applied to calculate
            the feature.
        fc_measure: str
            "WPLI" or "AECORT". Measure of Functional Connectivity to calculate.
        mode: str
            "Global coupling", "Strength" or "Coupling". Information extracted
            from adjacency matrix.
        montage: EEGChannelSet
        target_channels: list or None
            List containing the labels of the target channels.
        """

        super().__init__(ext_feature=['conn_value'])

        # Check errors
        if not montage:
            raise ValueError('[ConnectivityExtraction] "montage parameter"'
                             ' must be a dict containing all'
                             'labels of channels and montage standard key')
        if fc_measure != "WPLI" and fc_measure != "AECORT":
            raise ValueError('[ConnectivityExtraction] Invalid functional '
                             'connectivity measure. Available measures are '
                             '"WPLI" and "AECORT".')
        if mode != "Global coupling" and mode != "Strength" and mode != \
                "Coupling":
            raise ValueError('[ConnectivityExtraction] Invalid mode. '
                             'Available modes are '
                             '"Global coupling", "Strength" and "Coupling".')
        if target_channels is None:
            if mode == "Strength":
                raise UserWarning(
                    '[ConnectivityExtraction] Using "Strength" mode'
                    'without defining target channels. Average strength'
                    'of all channels will be returned instead.')
            if mode == "Coupling":
                mode = "Global coupling"
                raise UserWarning(
                    '[ConnectivityExtraction] Using "Coupling" mode'
                    'without defining target channels. Global coupling of all '
                    'channels will be returned instead.')

            self.target_channels = target_channels
        else:
            self.target_channels = self.montage.get_cha_idx_from_labels(
                target_channels)

        self.fc_measure = fc_measure
        self.mode = mode
        self.fs = fs
        self.l_baseline_t = l_baseline_t
        self.montage = montage
        self.w_signal_samples = int(update_feature_window * self.fs)
        self.w_signal_samples_calibration = int(self.l_baseline_t * self.fs)
        self.baseline_value = None

    def set_baseline(self, signal):
        # TODO PENSAR EN SI ES MEJOR CALCULAR EL BASELINE DE LOS 15
        #  SEGUNDOS O MEJOR DIVIDIR EN ÉPOCAS

        self.baseline_value = self.calculate_feature(
            signal[-self.w_signal_samples_calibration:, :])
        return self.baseline_value

    def ext_feature(self,signal):

        if self.baseline_value is None:
            raise ValueError('[ConnectivityExtraction] Calibration not performed.')

        return self.calculate_feature(signal[-self.w_signal_samples:, :]) \
               - self.baseline_value

    def calculate_feature(self, signal):

        # First calculate adjacency matrix depending on FC measure chosen
        if self.fc_measure == "WPLI":
            _, _, adj_mat = phase_connectivity(signal.squeeze())
        elif self.fc_measure == "AECORT":
            adj_mat = aec(signal.squeeze())

        # Then calculate the baseline value depending on mode chosen
        if self.mode == "Global coupling":
            tri_l_idx = np.tril_indices(16, -1)
            return np.nanmean(np.asarray(adj_mat)[tri_l_idx])
        elif self.mode == "Strength":
            if self.target_channels is None:
                return np.mean(degree.degree(np.asarray(adj_mat),
                                                            'CPU'))
            else:
                return np.mean(degree.degree(np.asarray(adj_mat),
                                                            'CPU')[
                                                  self.target_channels])
        elif self.mode == "Coupling":
            return self.mean_coupling(adj_mat)

    def mean_coupling(self, adj_mat):
        """This function calculates the connectivity values between all the
           target channels and average it value. """
        c = []
        for ind, ch_ind_1 in enumerate(self.target_channels[:-1]):
            for ch_ind_2 in self.target_channels[ind + 1:]:
                c.append(np.array(adj_mat[ch_ind_1, ch_ind_2]))
        return np.mean(c)


class PowerExtraction(components.ProcessingMethod):
    """
    Power-based features to extract from user's EEG.
    """

    def __init__(self, l_baseline_t=5, fs=250, update_feature_window=2,
                 mode=None):
        """
        Class constructor

        l_baseline_t: int
            Time employed to calculate the number of samples to obtain baseline
            power parameter. In seconds.
        fs: int or float
            Sample rate of the recording.
        update_feature_window = int
            Length in seconds of the temporal window applied to calculate
            the feature.
        """

        super().__init__(band_power=['band power'],
                         ban_bands=['band power'])

        self.mode = mode
        self.fs = fs
        self.l_baseline_t = l_baseline_t
        self.w_signal_samples = int(update_feature_window * self.fs)
        self.w_signal_samples_calibration = int(self.l_baseline_t * self.fs)
        self.baseline_power = None
        self.thresholds = None

    def set_baseline(self, signal):
        """
        This function sets the power baseline, given the already filtered EEG
        containing the calibration phase. Also, takes into account the
        Neurofeedback training mode, so performs different baseline calculations
        depending on the mode set.

        Parameters
        ----------
        signal: list or numpy.ndarray
            EEG already pre-processed. Its shape depends on the Neurofeedback
            modality: ([n_samples x n_channels] if SingleBandNFT mode,
            [2 x n_samples x n_channels] if RatioBandNFT mode and
            [1 + n_bands_to_ban x n_samples x n_channels] if
            RestrictionBandsNFT mode).

        Returns
        ------
        baseline_power: float
        """

        power = self.mean_power(signal[-self.w_signal_samples_calibration:, :])

        if self.mode == 'ban mode':

            self.thresholds = np.var(
                signal[1:][-self.w_signal_samples_calibration:,
                :], axis=1).mean(axis=-1)

            self.baseline_power = power[0]

        elif self.mode == 'ratio mode':
            self.baseline_power = power[0] / power[1]
        elif self.mode == 'single mode':
            self.baseline_power = power[0]

        return self.baseline_power

    def band_power(self, signal):
        """
        This function returns the band power applying the mean power to the
        signal. Its performance depends on the Neurofeedback mode.

        Parameters
        ----------
        signal: list or numpy.ndarray
            EEG already pre-processed. Its shape depends on the Neurofeedback
            modality: ([n_samples x n_channels] if SingleBandNFT mode,
            [2 x n_samples x n_channels] if RatioBandNFT mode and
            [1 + n_bands_to_ban x n_samples x n_channels] if RestrictionBandsNFT
            mode). In the RestrictionBandsNFT mode, the target frequency band is
            passed previously as if the mode were SingleBandNFT.
        Returns
        ------
        b_power: float
        """

        if self.baseline_power is None:
            raise ValueError('[PowerExtraction] Calibration not performed.')

        power = self.mean_power(signal[-self.w_signal_samples:, :])

        if self.mode is 'ban mode':
            b_power_main_band = power[0] - self.baseline_power
            # b_power_ban_bands = power[1:]
            return b_power_main_band
        elif self.mode is 'ratio mode':
            b_power = power[0] / power[1] - self.baseline_power
            return b_power
        elif self.mode is 'single mode':
            b_power = power[0] - self.baseline_power
            return b_power

    def ban_bands(self, signals, tolerance=1):
        """"
        This function computes the  power of the already filtered signal at
        target frequency band to enhance. Also computes the mean channel variance
        at other bands. If the number of bands whose variance it is above a
        threshold (previously defined at calibration stage) is greater than a
        tolerance parameter, the function returns None as a sign that the epoch is
        not valid.

        Parameters:
        __________
        signal: list or numpy.ndarray
            Signal already filtered at frequency band of interest.
            Its shape must be [1 + n_bands_to_ban x n_samples x n_channels].
            First element (e.g., [0,:,:]) is the signal filtered at target
            frequency band to enhance. The rest are employed to decide whether
            the epoch is valid or not.
        tolerance: int
            Parameter to restrict the number of banning-bands which are allowed
            to have a variance above the pre-defined thresholds. When this number
            is above the tolerance, the epoch is set as not valid.
        """
        # Get band powers
        b_power_main_band = self.band_power(signals)

        # Check if power in forbidden bands is over thresholds
        over_var = np.sum(np.var(signals[1:], axis=1).mean(axis=-1) >=
                          1.9 * self.thresholds)
        if over_var < tolerance:
            return b_power_main_band
        else:
            return None
        # if np.sum(b_power_ban_bands > self.thresholds) < tolerance:
        #     return b_power_main_band
        # else:
        #     return 0

    @staticmethod
    def mean_power(signal):
        """
        This function computes the classical NF feature: the mean power across
        channels.

        Parameters
        ----------
        signal: list or numpy.ndarray
            EEG signal already filtered. Shape of [n_samples x n_channels]
        """
        return np.mean(np.log(np.mean(np.power(signal, 2), axis=1)), axis=1)

    # @staticmethod
    # def std_power(signal):
    #     """
    #     This function computes the standard deviation of the power of the signal
    #
    #     Parameters
    #     ----------
    #     signal: list or numpy.ndarray
    #         EEG signal already filtered. Shape of [n_samples x n_channels]
    #     """
    #     return np.mean(np.std(np.log(np.power(signal, 2)), axis=1), axis=1)

class ConnectivityBasedNFTModel(components.Algorithm):
    def __init__(self, fs, filter_dict, l_baseline_t, update_feature_window,
                 montage,target_channels, fc_measure, mode):
        super().__init__(calibration=['baseline_value'],
                         training=['feedback_value'])

        # Settings
        self.fs = fs
        self.filter_dict = filter_dict
        self.l_baseline_t = l_baseline_t
        self.update_feature_window = update_feature_window
        self.montage = montage
        self.target_channels = target_channels
        self.fc_measure = fc_measure
        self.mode = mode

        # Variables
        self.baseline_value = None

        # Check filter dict
        if len(self.filter_dict) > 1:
            raise Exception('The number of frequency bands selected does not '
                            'match the Neurofeedback mode.')

        # Add Pre-processing and Feature Extraction methods
        self.add_method('prep_method',
                        SignalPreprocessing(filter_dict=self.filter_dict,
                                            montage=self.montage,
                                            target_channels=self.target_channels,
                                            car=True))
        self.add_method('feat_ext_method',ConnectivityExtraction(fs=self.fs,
                                               l_baseline_t=self.l_baseline_t,
                                               update_feature_window=update_feature_window,
                                               fc_measure=self.fc_measure,mode=self.mode,
                                               montage=self.montage,
                                               target_channels=self.target_channels))

    def calibration(self, eeg):
        filtered_signal = self.get_inst('prep_method').fit_transform_signal(
            signal=eeg,
            fs=self.fs)
        self.baseline_value = self.get_inst('feat_ext_method').set_baseline(
            signal=filtered_signal)

    def training(self, eeg):
        filtered_signal = self.get_inst('prep_method').transform_signal(
            signal=eeg)
        feedback_value = self.get_inst('feat_ext_method').ext_feature(
            signal=filtered_signal)
        return feedback_value


class PowerBasedNFTModel(components.Algorithm):
    def __init__(self, fs, filter_dict, l_baseline_t, update_feature_window,
                 montage, target_channels, n_cha_lp, **kwargs):
        """
        Skeleton class for power-based Neurofeedback training models. This class
        inherits from components.Algorithm. Therefore, it can be used to create
        standalone algorithms that can be used in compatible apps from
        medusa-platform for online experiments. See components.Algorithm to know
        more about this functionality. Calibration and Training methods,
        as are common to all models, are added in this skeleton class.
        """
        super().__init__(calibration=['baseline_parameters'],
                         training=['feedback_value'])

        """
        Class constructor
        """

        # Settings
        self.fs = fs
        self.filter_dict = filter_dict
        self.l_baseline_t = l_baseline_t
        self.update_feature_window = update_feature_window
        self.montage = montage
        self.target_channels = target_channels
        self.n_cha_lp = n_cha_lp

        # Init variables
        self.baseline_power = None
        self.mode = None

        if not self.check_cutoff_settings():
            raise Exception('The number of frequency bands selected does not '
                            'match the Neurofeedback mode.')

        self.add_method('prep_method',
                        SignalPreprocessing(filter_dict=self.filter_dict,
                                            montage=self.montage,
                                            target_channels=self.target_channels,
                                            n_cha_lp=self.n_cha_lp,laplacian=True))
        self.add_method('feat_ext_method',
                        PowerExtraction(fs=fs, l_baseline_t=l_baseline_t,
                                        update_feature_window=update_feature_window))

    def calibration(self, eeg, **kwargs):
        """

        Function that receives the EEG signal, filters it and extract the
        baseline parameter adapted to the Neurofeedback training mode

        Parameters
        ----------
        eeg: numpy.ndarray
            EEG signal to process and extract baseline parameter

        Returns
        -------
        baseline_power: float
            Value of baseline parameter to display it at Platform
        """
        raise NotImplemented

    def training(self, eeg):
        """
        Function that receives the EEG signal, filters it and extract
        the feature adapted to the Neurofeedback training mode

        Parameters
        ----------
        eeg: numpy.ndarray
            EEG signal to process and extract baseline parameter

        Returns
        -------
        baseline_power: float
            Value of baseline parameter to display it at Platform
        """
        raise NotImplemented

    def check_cutoff_settings(self):
        """
        Function that receives the EEG signal, filters it and extract the
        feature adapted to the Neurofeedback training mode

        Parameters
        ----------
        eeg: numpy.ndarray
            EEG signal to process and extract baseline parameter

        Returns
        -------
        baseline_power: float
            Value of baseline parameter to display it at Platform
        """
        raise NotImplemented


class SingleBandNFT(PowerBasedNFTModel):
    """
    The simplest model of Neurofeedback training. The feedback value consist of
    the power of the band selected as target.
    """

    def __init__(self, fs, filter_dict, l_baseline_t, update_feature_window,
                 montage, target_channels, n_cha_lp):
        super().__init__(fs=fs, filter_dict=filter_dict,
                         l_baseline_t=l_baseline_t,
                         update_feature_window=update_feature_window,
                         montage=montage,
                         target_channels=target_channels, n_cha_lp=n_cha_lp)

        self.get_inst('feat_ext_method').mode = 'single mode'

    def check_cutoff_settings(self):
        if len(self.filter_dict) > 1:
            return False
        else:
            return True

    def calibration(self, eeg, **kwargs):
        filtered_signal = self.get_inst('prep_method').fit_transform_signal(
            signal=eeg,
            fs=self.fs)
        self.baseline_power = self.get_inst('feat_ext_method').set_baseline(
            signal=filtered_signal)
        # return self.baseline_power

    def training(self, eeg, **kwargs):
        filtered_signal = self.get_inst('prep_method').transform_signal(
            signal=eeg)
        feedback_value = self.get_inst('feat_ext_method').band_power(
            signal=filtered_signal)
        return feedback_value


class RatioBandNFT(PowerBasedNFTModel):
    """
    This Neurofeedback model is intended to use the ratio between the power of
    two frequency bands as feedback value.
    Thus, the baseline power parameter is the value of this ratio
    at calibration stage.
    """

    def __init__(self, fs, filter_dict, l_baseline_t, update_feature_window,
                 montage, target_channels, n_cha_lp):
        super().__init__(fs=fs, filter_dict=filter_dict,
                         l_baseline_t=l_baseline_t,
                         update_feature_window=update_feature_window,
                         montage=montage,
                         target_channels=target_channels, n_cha_lp=n_cha_lp)

        self.get_inst('feat_ext_method').mode = 'ratio mode'

    def check_cutoff_settings(self):
        if len(self.filter_dict) != 2:
            return False
        else:
            return True

    def calibration(self, eeg, **kwargs):
        filtered_signals = self.get_inst('prep_method').fit_transform_signal(
            signal=eeg,
            fs=self.fs)
        self.baseline_power = self.get_inst('feat_ext_method').set_baseline(
            signal=filtered_signals)

    def training(self, eeg, **kwargs):
        filtered_signals = self.get_inst('prep_method').transform_signal(
            signal=eeg)
        feedback_value = self.get_inst('feat_ext_method').band_power(
            signal=filtered_signals)
        return feedback_value


class RestrictionBandNFT(PowerBasedNFTModel):
    """
    This Neurofeedback model, as SingleBandNFT, is aimed to enhance the power of
    a target band. However, it also tries to keep down the power of other selected
    frequency bands. Thereby, this training mode ensure that the user is
    up-regulating only the desired band. That is, this model is a more specific
    version of SingleBandNFT. It also can be use to prevent the influence of
    artifacts on the feedback.

    """

    def __init__(self, fs, filter_dict, l_baseline_t, update_feature_window,
                 montage, target_channels, n_cha_lp):
        super().__init__(fs=fs, filter_dict=filter_dict,
                         l_baseline_t=l_baseline_t,
                         update_feature_window=update_feature_window,
                         montage=montage,
                         target_channels=target_channels, n_cha_lp=n_cha_lp)

        self.get_inst('feat_ext_method').mode = 'ban mode'

    def check_cutoff_settings(self):
        if len(self.filter_dict) < 2:
            return False
        else:
            return True

    def calibration(self, eeg, **kwargs):
        filtered_signals = self.get_inst('prep_method').fit_transform_signal(
            signal=eeg,
            fs=self.fs)
        self.baseline_power = self.get_inst('feat_ext_method').set_baseline(
            signal=filtered_signals)

    def training(self, eeg):
        filtered_signals = self.get_inst('prep_method').transform_signal(
            signal=eeg)
        feedback_value = self.get_inst('feat_ext_method').ban_bands(
            signals=filtered_signals)
        return feedback_value


class NeurofeedbackData(components.ExperimentData):
    """Experiment info class for Neurofeedback training experiments. It records
    the important events that take place during a Neurofeedback run,
    allowing offline analysis."""

    def __init__(self, run_onsets, run_durations, run_success, run_pauses,
                 run_restarts, medusa_nft_app_settings):

        self.run_onsets = run_onsets
        self.run_durations = run_durations
        self.run_success = run_success
        self.run_pauses = run_pauses
        self.run_restarts = run_restarts
        self.medusa_nft_app_settings = medusa_nft_app_settings

    def to_serializable_obj(self):
        rec_dict = self.__dict__
        for key in rec_dict.keys():
            if type(rec_dict[key]) == np.ndarray:
                rec_dict[key] = rec_dict[key].tolist()
        return rec_dict

    @classmethod
    def from_serializable_obj(cls, dict_data):
        return cls(**dict_data)
