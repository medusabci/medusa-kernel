"""
In this module you will find useful functions and classes to apply on-line
Neurofeedback models. Each model is based on different features to be used
as target to train

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
from medusa.spatial_filtering import LaplacianFilter


class FilterBankPreprocessing(components.ProcessingMethod):
    """
    Common preprocessing applied in Neurofeedback applications.
    It is composed by a frequency IIR filter followed by a Laplacian spatial filter.
    Functions are adapted to filter the signal in more than one frequency range, if necessary.
    """

    def __init__(self, filter_bank=None, montage=None, target_channels=None, n_cha_lp=None):
        super().__init__(fit_transform_signal=['signal'],
                         transform_signal=['signal'])
        if filter_bank is None:
            filter_bank = [{'order': 4,
                            'cutoff': (8.0, 12.0),
                            'btype': 'bandpass',
                            'filt_method':'sosfiltfilt',
                            'action': 'increase'}]
        if montage is None:
            montage = meeg.EEGChannelSet()
            montage.set_standard_montage(l_cha=['FZ', 'CZ', 'PZ', 'OZ'],
                                         montage='10-20')

        if len(target_channels) == 0:
            target_channels = ['CZ']

        # Error check
        if not filter_bank:
            raise ValueError('[FilterBankPreprocessing] Filter bank parameter '
                             '"filter_bank" must be a list containing all '
                             'necessary information to perform the filtering!')
        for filter in filter_bank:
            if not isinstance(filter, dict):
                raise ValueError('[FilterBankPreprocessing] Each filter must '
                                 'be a dict()!')
            if 'order' not in filter or \
                    'cutoff' not in filter or \
                    'filt_method' not in filter or \
                    'btype' not in filter:
                raise ValueError('[FilterBankPreprocessing] Each filter must '
                                 'be a dict() containing the following keys: '
                                 '"order", "cutoff", "filt_method" and "btype"!')

        if not montage:
            raise ValueError('[FilterBankPreprocessing] Filter bank parameter'
                             '"montage" must be a dict containing all'
                             'labels of channels and montage standard key')

        if not target_channels:
            raise ValueError('[FilterBankPreprocessing] Filter bank parameter'
                             '"target_channels" must be a list containing all'
                             'labels of channels to extract NFT features')
        # Parameters
        self.filter_bank = filter_bank
        self.l_cha = montage.l_cha
        self.target_channels = target_channels
        self.n_cha_lp = n_cha_lp
        self.montage = montage

        # Variables
        self.filter_bank_iir_filters = None
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
        self.filter_bank_iir_filters = []

        self.offset_removal = mds.IIRFilter(order=2,
                                            cutoff=[1, 40],
                                            btype='bandpass',
                                            filt_method=self.filter_bank[0]['filt_method'])
        self.offset_removal.fit(fs, len(self.l_cha))

        for filter in self.filter_bank:
            iir = mds.IIRFilter(order=filter['order'],
                                cutoff=filter['cutoff'],
                                btype=filter['btype'],
                                filt_method=filter['filt_method'])
            iir.fit(fs, len(self.target_channels))
            self.filter_bank_iir_filters.append(iir)

        # Fit Laplacian Filter
        self.laplacian_filter = LaplacianFilter(self.montage, mode='auto')
        self.laplacian_filter.fit_lp(n_cha_lp=self.n_cha_lp, l_cha_to_filter=self.target_channels)

    def transform_signal(self, signal, parallel_computing=True):
        """
        Transforms an EEG signal applying IIR filter and Laplacian spatial filter sequentially

        Parameters
        ----------
        signal: list or numpy.ndarray
            Signal to transform. Shape [n_samples x n_channels]

        Returns
        -------
        signals: list
            List containing as many arrays of filtered signals as frequency bands set to filter.
        """
        # Initialize signals array
        n_samples = signal.shape[0]
        signals = np.empty((len(self.filter_bank), n_samples, len(self.target_channels)))

        signal_ = self.offset_removal.transform(signal)
        signal_ = self.laplacian_filter.apply_lp(signal_)
        signal__ = signal_.copy()
        if parallel_computing:
            filt_threads = []
            for filter in self.filter_bank_iir_filters:
                t = components.ThreadWithReturnValue(target=filter.transform,
                    args=(signal__,))
                filt_threads.append(t)
                t.start()

            for filt_idx, thread in enumerate(filt_threads):
                signals[filt_idx, :, :] = thread.join()
        else:
            for filt_idx, filter in enumerate(self.filter_bank_iir_filters):
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
            Number of nearest channels to compute Laplacian spatial filter (Auto mode).
        signal: list or numpy.ndarray
            Signal to transform. Shape [n_samples x n_channels]

        Returns
        -------
        signals: list
            List containing as many arrays of filtered signals as frequency bands set to filter.
        """
        self.fit(fs)
        return self.transform_signal(signal)


class FeatureExtraction(components.ProcessingMethod):
    """
    Standard features to extract from user's EEG.
    """

    def __init__(self, l_baseline_t=5, fs=250, update_feature_window=2, mode=None):
        """
        Class constructor

        l_baseline_t: int
            Time employed to calculate the number of samples to obtain baseline power parameter. In seconds.
        fs: int or float
            Sample rate of the recording.
        update_feature_window = int
            Length in seconds of the temporal window applied to calculate the feature.
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

    def set_baseline(self, signal, ban_std_parameter=1):
        # TODO REWRITE DOCUMENTATION
        """
        This function sets the power baseline, given the already filtered EEG containing the calibration phase.
        Also, takes into account the Neurofeedback training mode, so performs different baseline calculations
        depending on the mode set.

        Parameters
        ----------
        signal: list or numpy.ndarray
            EEG already pre-processed. Its shape depends on the Neurofeedback modality
            ([n_samples x n_channels] if SingleBandNFT mode, [2 x n_samples x n_channels] if RatioBandNFT mode
            and [1 + n_bands_to_ban + n_samples x n_channels] if EnhanceAndBanBandsNFT mode)
        ratio_mode: bool
            Set to True in RatioBandNFT mode. Calculates the baseline parameter from the ratio between baselines of
            each band.
        ban_mode: bool
            Set to True in EnhanceAndBanBandsNFT mode. Calculates the baseline power of the target band to enhance
            and the thresholds of the bands to ban
        ban_std_parameter = int
            Integer to multiply to the standard deviation of the signal filtered in the ban-frequency ranges. It
            defines the threshold in the EnhanceAndBanBandsNFT mode

        Returns
        ------
        baseline_power: float
        """

        power = mean_power(signal[-self.w_signal_samples_calibration:, :])

        if self.mode == 'ban mode':

            self.thresholds = np.empty(len(signal) - 1)
            std = std_power(signal[1:][-self.w_signal_samples_calibration:, :])

            self.baseline_power = power[0]
            self.thresholds = power[1:] + ban_std_parameter * std

        elif self.mode == 'ratio mode':
            self.baseline_power = power[0] / power[1]
        elif self.mode == 'single mode':
            self.baseline_power = power[0]

        return self.baseline_power

    def band_power(self, signal):
        """
        This function returns the band power applying the mean power to the signal. Its performance depends on
        the Neurofeedback mode.

        Parameters
        ----------
        signal: list or numpy.ndarray
            EEG already pre-processed. Its shape depends on the Neurofeedback modality:
            ([n_samples x n_channels] if SingleBandNFT mode, [2 x n_samples x n_channels] if RatioBandNFT mode
            and [n_bands_to_ban + n_samples x n_channels] if RestrictionBandsNFT mode). In the RestrictionBandsNFT
            mode, the target frequency band is passed previously as if the mode were SingleBandNFT
        Returns
        ------
        b_power: float
        """

        if self.baseline_power is None:
            raise ValueError('[FeatureExtaction] Calibration not performed.')

        power = mean_power(signal[-self.w_signal_samples:, :])

        if self.mode is 'ban mode':
            b_power_main_band = power[0] - self.baseline_power
            b_power_ban_bands = power[1:]
            return b_power_main_band, b_power_ban_bands
        elif self.mode is 'ratio mode':
            b_power = power[0] / power[1] - self.baseline_power
            return b_power
        elif self.mode is 'single mode':
            b_power = power[0] - self.baseline_power
            return b_power

    def ban_bands(self, signals, tolerance=1):
        """"
        This function computes the  power of the already filtered signal at target frequency band to enhance.
        Also computes the power at other bands. If the number of bands whose power it is above a threshold
        (previously defined at calibration stage) is greater than a tolerance parameter, the function returns None
        as a sign that the epoch is not valid.

        Parameters:
        __________
        signal: list or numpy.ndarray
            Signal already filtered at frequency band of interest. Its shape must be [1 + n_bands_to_ban x n_samples x
            n_channels]. First element (e.g., [0,:,:]) is the signal filtered at target frequency band to enhance.
            The rest are employed to decide whether the epoch is valid or not
        tolerance: int
            Parameter to restrict the number of banning-bands which are allowed to have a power above the pre-defined
            thresholds. When this number is above the tolerance, the epoch is set as not valid.
        """
        # Get band powers
        b_power_main_band, b_power_ban_bands = self.band_power(signals)

        # Check if power in forbidden bands is over thresholds
        if np.sum(b_power_ban_bands > self.thresholds) < tolerance:
            return b_power_main_band
        else:
            return 0


def mean_power(signal):
    """
    This function computes the classical NF feature: the mean power across channels.
    It also returns the standard
    deviation of the band power if chose

    Parameters
    ----------
    signal: list or numpy.ndarray
        EEG signal already filtered. Shape of [n_samples x n_channels]
    """
    return np.mean(np.log(np.mean(np.power(signal, 2), axis=1)), axis=1)


def std_power(signal):
    """
    This function computes the standard deviation of the power of the signal

    Parameters
    ----------
    signal: list or numpy.ndarray
        EEG signal already filtered. Shape of [n_samples x n_channels]
    """
    return np.mean(np.std(np.log(np.power(signal, 2)), axis=1), axis=1)


class PowerBasedNFTModel(components.Algorithm):
    def __init__(self, fs, filter_bank, l_baseline_t, update_feature_window, montage,
                 target_channels, n_cha_lp, **kwargs):
        """
        Skeleton class for basic Neurofeedback training models. This class inherits from
        components.Algorithm. Therefore, it can be used to create standalone
        algorithms that can be used in compatible apps from medusa-platform
        for online experiments. See components.Algorithm to know more about this
        functionality. Calibration and Training methods, as are common to all models, are added
        in this skeleton class
        """
        super().__init__(calibration=['baseline_parameters'],
                         training=['feedback_value'])

        """
        Class constructor
        """

        # Settings
        self.fs = fs
        self.filter_bank = filter_bank
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

        self.add_method('prep_method', FilterBankPreprocessing(filter_bank=self.filter_bank, montage=self.montage,
                                                               target_channels=self.target_channels,
                                                               n_cha_lp=self.n_cha_lp))
        self.add_method('feat_ext_method',
                        FeatureExtraction(fs=fs, l_baseline_t=l_baseline_t,
                                          update_feature_window=update_feature_window))

    def calibration(self, eeg, **kwargs):
        """

        Function that receives the EEG signal, filters it and extract the baseline parameter adapted to
        the Neurofeedback training mode

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
        Function that receives the EEG signal, filters it and extract the feature adapted to
        the Neurofeedback training mode

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
        Function that receives the EEG signal, filters it and extract the feature adapted to
        the Neurofeedback training mode

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
    The simplest model of Neurofeedback training. The feedback value consist of the
    power of the band selected as target
    """

    def __init__(self, fs, filter_bank, l_baseline_t, update_feature_window, montage, target_channels, n_cha_lp):
        super().__init__(fs=fs, filter_bank=filter_bank, l_baseline_t=l_baseline_t,
                         update_feature_window=update_feature_window, montage=montage,
                         target_channels=target_channels, n_cha_lp=n_cha_lp)

        self.get_inst('feat_ext_method').mode = 'single mode'

    def check_cutoff_settings(self):
        if len(self.filter_bank) > 1:
            return False
        else:
            return True

    def calibration(self, eeg, **kwargs):
        filtered_signal = self.get_inst('prep_method').fit_transform_signal(signal=eeg,
                                                                            fs=self.fs)
        self.baseline_power = self.get_inst('feat_ext_method').set_baseline(signal=filtered_signal)
        # return self.baseline_power

    def training(self, eeg, **kwargs):
        filtered_signal = self.get_inst('prep_method').transform_signal(signal=eeg)
        feedback_value = self.get_inst('feat_ext_method').band_power(signal=filtered_signal)
        return feedback_value


class RatioBandNFT(PowerBasedNFTModel):
    """
    This Neurofeedback model is intended to use the ratio between the power of two frequency bands as feedback value.
    Thus, the baseline power parameter is the value of this ratio at calibration stage
    """

    def __init__(self, fs, filter_bank, l_baseline_t, update_feature_window, montage, target_channels, n_cha_lp):
        super().__init__(fs=fs, filter_bank=filter_bank, l_baseline_t=l_baseline_t,
                         update_feature_window=update_feature_window, montage=montage,
                         target_channels=target_channels, n_cha_lp=n_cha_lp)

        self.get_inst('feat_ext_method').mode = 'ratio mode'

    def check_cutoff_settings(self):
        if len(self.filter_bank) != 2:
            return False
        else:
            return True

    def calibration(self, eeg, **kwargs):
        filtered_signals = self.get_inst('prep_method').fit_transform_signal(signal=eeg,
                                                                             fs=self.fs)
        self.baseline_power = self.get_inst('feat_ext_method').set_baseline(signal=filtered_signals)

    def training(self, eeg, **kwargs):
        filtered_signals = self.get_inst('prep_method').transform_signal(signal=eeg)
        feedback_value = self.get_inst('feat_ext_method').band_power(signal=filtered_signals)
        return feedback_value


class RestrictionBandNFT(PowerBasedNFTModel):
    """
    This Neurofeedback model, as SingleBandNFT, is aimed to enhance the power of a target band. However, it also tries
    to keep down the power of other selected frequency bands. Thereby, this training mode ensure that the user is
    upregulating only the desired band. That is, this model is a more specific version of SingleBandNFT

    """

    def __init__(self, fs, filter_bank, l_baseline_t, update_feature_window, montage, target_channels, n_cha_lp):
        super().__init__(fs=fs, filter_bank=filter_bank, l_baseline_t=l_baseline_t,
                         update_feature_window=update_feature_window, montage=montage,
                         target_channels=target_channels, n_cha_lp=n_cha_lp)

        self.get_inst('feat_ext_method').mode = 'ban mode'

    def check_cutoff_settings(self):
        if len(self.filter_bank) < 2:
            return False
        else:
            return True

    def calibration(self, eeg, **kwargs):
        filtered_signals = self.get_inst('prep_method').fit_transform_signal(signal=eeg,
                                                                             fs=self.fs)
        self.baseline_power = self.get_inst('feat_ext_method').set_baseline(signal=filtered_signals)

    def training(self, eeg):
        filtered_signals = self.get_inst('prep_method').transform_signal(signal=eeg)
        feedback_value = self.get_inst('feat_ext_method').ban_bands(signals=filtered_signals)
        return feedback_value

class NeurofeedbackData(components.ExperimentData):
    """Experiment info class for Neurofeedback training experiments. It records
    the important events that take place during a Neurofeedback run, allowing offline analysis."""
    def __init__(self,run_onsets,run_durations,run_success,run_pauses,run_restarts, medusa_nft_app_settings):

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
