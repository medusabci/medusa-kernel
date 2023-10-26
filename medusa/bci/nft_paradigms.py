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
import scipy.signal

# Medusa imports
import medusa as mds
from medusa import components
from medusa import meeg
from medusa.spatial_filtering import LaplacianFilter, car
from medusa.connectivity.phase_connectivity import phase_connectivity
from medusa.connectivity.amplitude_connectivity import __aec_ort_cpu as aec
from medusa.graph_theory import degree
from medusa.artifact_removal import reject_noisy_epochs
from medusa.epoching import get_epochs_of_events
from medusa.local_activation.spectral_parameteres import absolute_band_power


class SignalPreprocessing(components.ProcessingMethod):
    """
    Common preprocessing applied in Neurofeedback applications.
    It is composed by a frequency IIR filter followed by a spatial
    filters. Functions are adapted to filter the signal in more than one frequency
    range, if necessary.
    """

    def __init__(self, filter_dict=None, montage=None, target_channels=None,
                 laplacian=False, car=False, n_cha_lp=None):
        super().__init__(prep_fit_transform=['signal'],
                         prep_transform=['signal'],
                         narrow_transform=['signal'])

        # Error check
        if not filter_dict:
            raise ValueError('[SignalPreprocessing] Filter dict parameter '
                             '"filter_dict" must be a list containing all '
                             'necessary information to perform the filtering!. '
                             'The information should be: type and cutoff')
        for filter in filter_dict:
            if not isinstance(filter, dict):
                raise ValueError('[SignalPreprocessing] Each filter must '
                                 'be a dict()!')
            if 'cutoff' not in filter or 'type' not in filter:
                raise ValueError('[SignalPreprocessing] Each filter must '
                                 'be a dict() containing the following keys: '
                                 '"cutoff"and "type"!')
            if filter['type'] != 'training' and filter['type'] != 'artifact':
                raise ValueError(
                    '[SignalPreprocessing] "type" must be "training"'
                    'or "artifact".')
        if not montage:
            raise ValueError('[SignalPreprocessing] Pre-processing parameter'
                             '"montage" must be a dict containing all'
                             'labels of channels and montage standard key')
        if laplacian and target_channels is None:
            raise ValueError('[SignalPreprocessing] Laplacian filter needs to '
                             'define "target_channels" parameter.')

        # Parameters
        self.filter_dict = filter_dict
        self.l_cha = montage.l_cha
        self.target_channels = target_channels
        self.montage = montage
        self.perform_car = car
        self.perform_laplacian = laplacian

        # Variables
        self.artifact_iir_filters = []
        self.target_iir_filters = []
        self.offset_line_removal = None
        self.laplacian_filter = None

    def fit(self, fs):
        """
        Fits the IIR filter and Laplacian spatial filter (if selected) for signal
        preprocessing stage.

        Parameters
        ----------
        fs: float
            Sampling rate in Hz.
        """

        # Fit Spectral Filters (Predefined to be optimal)
        self.offset_line_removal = mds.IIRFilter(order=3,
                                                 cutoff=[0.5, 40],
                                                 btype='bandpass',
                                                 filt_method='sosfiltfilt')
        self.offset_line_removal.fit(fs, len(self.l_cha))

        # Define filters for filtering over epochs
        for filter in self.filter_dict:
            for f in filter['cutoff']:
                if len(f) != 0:
                    iir = mds.IIRFilter(order=3,
                                        cutoff=f,
                                        btype='bandpass',
                                        filt_method='sosfiltfilt')

                    if self.target_channels is None:
                        iir.fit(fs, len(self.l_cha))
                    else:
                        iir.fit(fs, len(self.target_channels))
                    if filter['type'] == 'artifact':
                        self.artifact_iir_filters.append(iir)
                    elif filter['type'] == 'training':
                        self.target_iir_filters.append(iir)

        # Fit Laplacian Filter
        if self.perform_laplacian:
            if len(self.montage.l_cha) >= 5:
                self.laplacian_filter = LaplacianFilter(self.montage,
                                                        mode='auto')
                self.laplacian_filter.fit_lp(
                    l_cha_to_filter=self.target_channels)

    def prep_transform(self, signal, parallel_computing=True):
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
        signal_: numpy.ndarray
            Original signal with power line and offset removed, and spatially
            filtered if chosen. [n_samples, n_channels].
        signal_artifacts: numpy.ndarray
            If it has been chosen to reject artifact sections, this matrix
            contains the filtered signal in the frequency bands associated
            with these artifacts. [n_artifact_bands, n_samples, n_channels].
        """
        # Initialize variable
        n_samples = signal.shape[0]

        if len(self.artifact_iir_filters) == 0:
            signal_artifacts = None
        else:
            if self.target_channels is None:
                signal_artifacts = np.empty(
                    (
                        len(self.artifact_iir_filters), n_samples,
                        len(self.l_cha)))
            else:
                signal_artifacts = np.empty(
                    (len(self.artifact_iir_filters), n_samples,
                     len(self.target_channels)))

        signal_ = self.offset_line_removal.transform(signal)

        # Spatial filtering
        if self.perform_car:
            signal_ = car(signal_)
        if self.perform_laplacian:
            # Check if surface laplacian filter cannot be performed
            if self.laplacian_filter is not None:
                signal_ = self.laplacian_filter.apply_lp(signal_)
        else:
            if self.target_channels is not None:
                signal_ = signal_[:, self.montage.get_cha_idx_from_labels(
                    self.target_channels)]

        signal__ = signal_.copy()

        if signal_artifacts is not None:
            # Frequency filtering on artifact-related bands
            if parallel_computing:
                filt_threads = []
                for filter in self.artifact_iir_filters:
                    t = components.ThreadWithReturnValue(target=
                                                         filter.transform,
                                                         args=(signal__.copy(),))
                    filt_threads.append(t)
                    t.start()

                for filt_idx, thread in enumerate(filt_threads):
                    signal_artifacts[filt_idx, :, :] = thread.join()
            else:
                for filt_idx, filter in enumerate(self.artifact_iir_filters):
                    signal_artifacts[filt_idx, :, :] = filter.transform(
                        signal__[np.newaxis, :, :].copy())
        return signal_, signal_artifacts

    def prep_fit_transform(self, fs, signal):
        """
        Fits the IIR filter and transforms an EEG signal applying IIR
        filter and spatial filters sequentially.

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
        signal: numpy.ndarray
            Original signal with power line and offset removed, and spatially
            filtered if chosen. [n_samples, n_channels].
        signal_artifacts: numpy.ndarray
            If it has been chosen to reject artifact sections, this matrix
            contains the filtered signal in the frequency bands associated
            with these artifacts. [n_artifact_bands, n_samples, n_channels].
        """
        self.fit(fs)
        return self.prep_transform(signal)

    def narrow_transform(self, signal, parallel_computing=True):
        """
        Applies the IIR filter for narrow band filtering.

        Parameters
        ----------
        signal: list or numpy.ndarray
            Signal to transform. Shape [n_samples x n_channels].
        parallel_computing: bool
            If true, it filters the signal concurrently.

        Returns
        -------
        signal: numpy.ndarray
            Signal filtered in the training band [n_samples, n_channels].
        """
        f_signals = np.empty(
            (len(self.target_iir_filters), signal.shape[0], signal.shape[1]))
        if parallel_computing:
            filt_threads = []
            for filter in self.target_iir_filters:
                t = components.ThreadWithReturnValue(target=
                                                     filter.transform,
                                                     args=(signal,))
                filt_threads.append(t)
                t.start()

            for filt_idx, thread in enumerate(filt_threads):
                f_signals[filt_idx, :, :] = thread.join()
        else:
            for filt_idx, filter in enumerate(self.target_iir_filters):
                f_signals[filt_idx, :, :] = filter.transform(signal)

        return np.squeeze(f_signals)


def ignore_noisy_windows(signals, thresholds, pct_tol):
    """
    This function check if a specific signal segment contains noise above the
    pre-established thresholds.
    Parameters
    ----------
    signals: numpy.ndarray
        Array containing the signal filtered in the frequency bands associated to
        artifacts to avoid. [n_artifact_bands, n_samples, n_channels].
    thresholds: numpy.ndarray
        Array containing the variance thresholds related to artifacts to avoid.
    pct_tol: numpy.ndarray
        Array containing variance increase (in percentage) tolerated.

    Returns
    -------

    """
    # Check if power in forbidden bands is over thresholds
    over_var = np.sum(np.std(signals, axis=1).mean(axis=-1) >=
                      (1 + pct_tol) * thresholds)
    if over_var < 1:
        return True
    else:
        return False


def make_windows(signal, fs, update_feature_window, update_rate,
                 n_cha=1, n_samp=2, k=4, reject=True):
    """

    Parameters
    ----------
    signal: numpy.ndarray
        Signal to be converted into epochs. [n_samples, n_channels].
    fs: int or float
        Sampling rate.
    update_feature_window: int or float
        Time window taken for the calculation of the characteristic (in seconds).
    update_rate: int or float
        Feedback update time in online mode.
    n_cha: int
        Threshold number of channels meeting the rejection condition to reject
        the epoch.
    n_samp: int
        Threshold number of samples meeting the rejection condition to reject
        the epoch.
    k: int
        Standard deviation of the signal. Used in the definition of the
        rejection criterion.
    reject: bool
        If true, it returns the epochs that have not been reject. Else, it
        returns the whole windowed signal.

    Returns
    -------
    good_epochs: numpy.ndarray
        Array containing the signal divided into epochs that are not noisy.
        [n_epochs, n_samples, n_channels].
    ind: numpy.ndarray
        Array containing bools. True for epochs that were rejected and False for
         epochs that were not.

    """
    if len(signal.shape) == 1:
        signal = signal[:, np.newaxis]
    # Define necessary parameters
    s_duration = signal.shape[0] / fs
    s_mean = np.mean(signal, axis=0)
    s_std = np.std(signal, axis=0)

    # Set onsets vector
    onsets = np.arange(0, s_duration - update_feature_window, update_rate)
    s_windowed = get_epochs_of_events(np.arange(0, s_duration, 1 / fs), signal,
                                      onsets, fs,
                                      [0, update_feature_window * 1000])
    # Return windows without discarding
    if not reject:
        return s_windowed
    pct_rejected, good_epochs, idx = reject_noisy_epochs(s_windowed, s_mean,
                                                         s_std, k, n_samp,
                                                         n_cha)
    return good_epochs, idx


class ConnectivityExtraction(components.ProcessingMethod):
    """
    Functional Connectivity-based features to extract from user's EEG.
    """

    def __init__(self, l_baseline_t=5, fs=250, update_feature_window=2,
                 update_rate=0.25, fc_measure=None, mode=None, montage=None,
                 target_channels=None, pct_tol=0.9):
        """
        Class constructor

        l_baseline_t: int
            Time employed to calculate the number of samples to obtain baseline
            connectivity parameter. In seconds.
        fs: int or float
            Sample rate of the recording.
        update_feature_window: int or float
            Length in seconds of the temporal window applied to calculate
            the feature.
        update_rate: int or float
            Feedback update time in online mode.
        fc_measure: str
            "WPLI" or "AECORT". Measure of Functional Connectivity to calculate.
        mode: str
            "Global coupling", "Strength" or "Coupling". Information extracted
            from adjacency matrix.
        montage: EEGChannelSet
        target_channels: list or None
            List containing the labels of the target channels.
        pct_tol: numpy.ndarray
        Array containing variance increase (in percentage) tolerated.
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
            self.target_channels = montage.get_cha_idx_from_labels(
                target_channels)

        self.fc_measure = fc_measure
        self.mode = mode
        self.fs = fs
        self.l_baseline_t = l_baseline_t
        self.montage = montage
        self.update_feature_window = update_feature_window
        self.update_rate = update_rate
        self.w_signal_samples = int(update_feature_window * self.fs)
        self.w_signal_samples_calibration = int((self.l_baseline_t) * self.fs)
        self.pct_tol = pct_tol
        self.thresholds = None
        self.baseline_value = None

    def set_baseline(self, filtered_signal,signal_artifacts ):
        """
        This functions establish the baseline value.
        Parameters
        ----------
        filtered_signal: numpy.ndarray
            Signal filtered in the narrow band for training.
            [n_samples, n_channels].
        signal_artifacts: numpy.ndarray
            Array containing the signal filtered in each frequency band associated
            to the artifacts  to avoid. [n_artifact_bands, n_samples, n_channels].
        Returns
        -------
        baseline_value: float
        """
        filtered_epochs,_ = make_windows(
            filtered_signal, self.fs,
            self.update_feature_window, self.update_rate,
            reject=True)

        adj_mat = self.calculate_adj_mat(filtered_epochs)

        # Parallel computing baseline values
        filt_threads = []
        baseline_values = []
        for epoch_mat in adj_mat:
            t = components.ThreadWithReturnValue(target=self.calculate_feature,
                                                 args=(epoch_mat,))
            filt_threads.append(t)
            t.start()

        for filt_idx, thread in enumerate(filt_threads):
            baseline_values.append(thread.join())

        self.baseline_value = np.mean(baseline_values)

        # Define artifact related thresholds
        if signal_artifacts is not None:
            self.thresholds = np.std(
                signal_artifacts[:, -self.w_signal_samples_calibration:,
                :], axis=1).mean(axis=-1)
        return self.baseline_value

    def ext_feature(self, signal, signal_artifacts):
        """
        Function for extracting FC values in online mode.
        Parameters
        ----------
        signal: numpy.ndarray
            Signal filtered in the narrow band for training. [n_samples, n_channels].
        signal_artifacts: numpy.ndarray or None
            Array containing the signal filtered in each frequency band associated
            to the artifacts  to avoid. [n_artifact_bands, n_samples, n_channels].
        Returns
        -------
        c_value: float
        """
        if self.baseline_value is None:
            raise ValueError(
                '[ConnectivityExtraction] Calibration not performed.')
        adj_mat = self.calculate_adj_mat(signal)
        c_value = self.calculate_feature(np.squeeze(adj_mat)) \
                  - self.baseline_value
        # Check if artifact bands are defined
        if signal_artifacts is not None:
            if ignore_noisy_windows(
                    signal_artifacts[:, :, self.target_channels],
                    self.thresholds,
                    self.pct_tol):
                return c_value
            else:
                return None
        return c_value

    def calculate_adj_mat(self, signal):
        """
        This function calculates the adjacency matrix depending on the FC mode.
        Parameters
        ----------
        signal: numpy.ndarray
            Signal filtered in the narrow band for training.
            [n_epochs, n_samples, n_channels] or [n_samples, n_channels].
        Returns
        -------
        adj_mat: numpy.ndarray
            [n_epochs, n_channels, n_channels].
        """
        # Calculate adjacency matrix depending on FC measure chosen
        adj_mat = None
        if self.fc_measure == "WPLI":
            adj_mat = phase_connectivity(signal, 'wpli')
        # This is under development
        elif self.fc_measure == "AECORT":
            adj_mat = aec(signal)
        return adj_mat

    def calculate_feature(self, adj_mat):
        """
        Calculates Graph metric from adjacency matrix.

        Parameters
        ----------
        adj_mat: numpy.ndarray
            [n_channels, n_channels].
        """
        # Calculate the baseline value depending on mode chosen
        if self.mode == "Global coupling":
            tri_l_idx = np.tril_indices(adj_mat.shape[0], -1)
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
        """
        This function calculates the connectivity values between all the
           target channels and average it value.

        Parameters
        ----------
        adj_mat: numpy.ndarray
            [n_channels, n_channels].
        """
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
                 right_ch_idx=None, update_rate=0.25, pct_tol=0.9,
                 f_dict=None, mode=None):
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
        update_rate: int or float
            Feedback update time in online mode.
        f_dict: dict
            Dict containing the frequency bands associated to training band
            and artifacts to avoid.
        right_ch_idx: list (optional)
            List containng the indexes of the channels that will be used
            to predict right motor imagery. The indexes are relative to
            the target channels list. This argument is only necessary when
            using  the neurofeedback-based motor imagery.
        update_rate: float
            Value of the real-time feedback calculation rate.
        pct_tol: numpy.ndarray
            Array containing variance increase (in percentage) tolerated.
        mode: str
            "single" or "ratio"
        """

        super().__init__(band_power=['band power'])

        self.mode = mode
        self.fs = fs
        self.l_baseline_t = l_baseline_t
        self.update_feature_window = update_feature_window
        self.update_rate = update_rate
        self.w_signal_samples = int(update_feature_window * self.fs)
        self.w_signal_samples_calibration = int(self.l_baseline_t * self.fs)
        self.f_dict = f_dict
        self.pct_tol = pct_tol
        self.baseline_power = []
        self.thresholds = None
        self.right_ch_idx = right_ch_idx

    def set_baseline(self, signal, signal_artifacts):
        """
        This function sets the power baseline, given the already filtered EEG
        containing the calibration phase. Also, takes into account the
        Neurofeedback training mode, so performs different baseline calculations
        depending on the mode set.

        Parameters
        ----------
        signal: numpy.ndarray
            EEG already pre-processed. [n_samples, n_channels].
        signal_artifacts: numpy.ndarray
            Signal filtered in the frequency bands associated to the artifacts
            to be avoided. [n_artifact_bands, n_samples, n_channels].
        Returns
        ------
        baseline_power: float
        """
        epochs, _ = make_windows(signal, self.fs, self.update_feature_window,
                                 self.update_rate)
        _, psd = scipy.signal.welch(epochs, self.fs, 'hamming',
                                    self.w_signal_samples,
                                    axis=1, scaling='density')
        b_power = []
        if self.right_ch_idx is not None:
            b_power.append(
                self.power(psd[:, :, np.setdiff1d(np.arange(psd.shape[2]),
                                                  self.right_ch_idx)]))
            b_power.append(self.power(psd[:, :, self.right_ch_idx]))
        else:
            b_power.append(self.power(psd))

        # Define artifact related thresholds
        if signal_artifacts is not None:
            self.thresholds = np.std(
                signal_artifacts[:, -self.w_signal_samples_calibration:,
                :], axis=1).mean(axis=-1)

        if self.mode == 'single':
            self.baseline_power = [b_power[0][0]]
            if self.right_ch_idx is not None:
                self.baseline_power.append(b_power[1][0])

        elif self.mode == 'ratio':
            self.baseline_power = [b_power[0][0] / b_power[0][1]]
            if self.right_ch_idx is not None:
                self.baseline_power.append(b_power[1][0] / b_power[1][1])
        if self.right_ch_idx is None:
            return self.baseline_power[0]
        else:
            return self.baseline_power

    def band_power(self, signal, signal_artifacts):
        """
        This function returns the band power from Power Spectral Density.
        If signal noise is above the pre-established thresholds, this function
        will return None.

        Parameters
        ----------
        signal: numpy.ndarray
            Signal pre-processed. [n_samples, n_channels].
        signal_artifacts: numpy.ndarray
            Signal filtered in the frequency bands associated to the artifacts
            to be avoided. [n_artifact_bands, n_samples, n_channels].
        Returns
        ------
        b_power: float or None
        """

        if self.baseline_power is None:
            raise ValueError('[PowerExtraction] Calibration not performed.')
        _, psd = scipy.signal.welch(signal, self.fs, 'hamming',
                                    self.w_signal_samples,
                                    axis=0, scaling='density')
        b_power_uncorrected = []
        b_power = []
        if self.right_ch_idx is not None:
            b_power_uncorrected.append(self.power(psd[:, np.setdiff1d(np.arange(
                psd.shape[1]),
                self.right_ch_idx)]))
            b_power_uncorrected.append(self.power(psd[:, self.right_ch_idx]))
        else:
            b_power_uncorrected.append(self.power(psd))
        if self.mode == 'single':
            b_power = [b_power_uncorrected[0][0] - self.baseline_power[0]]
            if self.right_ch_idx is not None:
                b_power.append(
                    b_power_uncorrected[1][0] - self.baseline_power[1])

        elif self.mode == 'ratio':
            b_power = [b_power_uncorrected[0][0] / b_power_uncorrected[0][1] - \
                       self.baseline_power[0]]
            if self.right_ch_idx is not None:
                b_power.append(
                    b_power_uncorrected[1][0] / b_power_uncorrected[1][1] - \
                    self.baseline_power[1])

        # Check if artifact bands are defined
        if signal_artifacts is not None:
            if signal_artifacts is not None:
                if ignore_noisy_windows(signal_artifacts, self.thresholds,
                                        self.pct_tol):
                    if self.right_ch_idx is None:
                        b_power = b_power[0]
                    return b_power
                else:
                    return None
        if self.right_ch_idx is None:
            b_power = b_power[0]
        return b_power

    def power(self, psd):
        """
        This function calculates power from Power Spectral Density
        Parameters
        ----------
        psd: numpy.ndarray
            [n_epochs, n_samples, n_channels].

        Returns
        -------
        powers: numpy.ndarray
            [n_training_bands].
        """
        # Check if psd has epochs dimension
        if len(psd.shape) == 1:
            psd = psd[:, None]
        if len(psd.shape) == 2:
            psd = psd[np.newaxis, :, :]
        bands = []
        # Extract training bands limits
        for dict in self.f_dict:
            if dict['type'] == 'training' and dict['cutoff'] != [[]]:
                bands.append(dict['cutoff'])
        powers = np.zeros(len(bands))

        # Calculate band power relative to the whole bandwidth
        for idx, band in enumerate(bands):
            for b in band:
                powers[idx] += np.mean(np.mean(absolute_band_power(psd, self.fs,
                                                                   b),
                                               axis=0))
        return powers


class ConnectivityBasedNFTModel(components.Algorithm):
    def __init__(self, fs, filter_dict, l_baseline_t, update_feature_window,
                 update_rate, montage, target_channels, fc_measure, mode,
                 apply_car, pct_tol_ocular=None, pct_tol_muscular=None):
        super().__init__(calibration=['baseline_value'],
                         training=['feedback_value'])
        """
        Pipeline for Connectivity-based Neurofeedback training. This class
        inherits from components.Algorithm. Therefore, it can be used to create
        standalone algorithms that can be used in compatible apps from
        medusa-platform for online experiments. See components.Algorithm to know
        more about this functionality.
        """

        # Settings
        self.fs = fs
        self.filter_dict = filter_dict
        self.l_baseline_t = l_baseline_t
        self.update_feature_window = update_feature_window
        self.montage = montage
        self.target_channels = target_channels
        self.fc_measure = fc_measure
        self.mode = mode
        self.apply_car = apply_car

        # Variables
        self.baseline_value = None
        self.pct_tol = None

        # Set percentage tolerance to noisy signal
        if pct_tol_ocular is None and pct_tol_muscular is not None:
            self.pct_tol = pct_tol_muscular
        elif pct_tol_ocular is not None and pct_tol_muscular is None:
            self.pct_tol = pct_tol_ocular
        elif pct_tol_ocular is not None and pct_tol_muscular is not None:
            self.pct_tol = np.array([pct_tol_ocular, pct_tol_muscular])

        # Check filter dict
        if not self.check_cutoff_settings():
            raise Exception('The number of frequency bands selected does not '
                            'match the Neurofeedback mode.')

        # Add Pre-processing and Feature Extraction methods
        self.add_method('prep_method',
                        SignalPreprocessing(filter_dict=self.filter_dict,
                                            montage=self.montage,
                                            target_channels=None,
                                            car=self.apply_car))
        self.add_method('feat_ext_method',
                        ConnectivityExtraction(fs=self.fs,
                                               l_baseline_t=self.l_baseline_t,
                                               update_feature_window=update_feature_window,
                                               fc_measure=self.fc_measure,
                                               mode=self.mode,
                                               montage=self.montage,
                                               target_channels=self.target_channels,
                                               pct_tol=self.pct_tol,
                                               update_rate=update_rate))

    def calibration(self, eeg):
        """
        It pre-process eeg, gets signal filtered in artifact-related bands and
        filters the pre-processed eeg in training band. Then, it calculates the
        baseline value.
        Parameters
        ----------
        eeg: numpy.ndarray
            [n_samples, n_channels]
        Returns
        -------
        baseline_value: float
        """
        original_signal, signal_artifacts = self.get_inst('prep_method'). \
            prep_fit_transform(
            signal=eeg,
            fs=self.fs)
        narrow_filtered_signal = self.get_inst('prep_method'). \
            narrow_transform(signal=original_signal)
        self.baseline_value = self.get_inst('feat_ext_method'). \
            set_baseline(signal_artifacts=signal_artifacts,
                         filtered_signal=narrow_filtered_signal)

    def training(self, eeg):
        """
        It pre-process eeg, gets signal filtered in artifact-related bands and
        filters the pre-processed eeg in training band. Then, it calculates the
        feedback value.
        Parameters
        ----------
        eeg: numpy.ndarray
            [n_samples, n_channels]
        Returns
        -------
        feedback_value: float
        """
        original_signal, signal_artifacts = self.get_inst(
            'prep_method').prep_transform(
            signal=eeg)
        narrow_filtered_signal = self.get_inst('prep_method'). \
            narrow_transform(signal=original_signal)
        feedback_value = self.get_inst('feat_ext_method').ext_feature(
            signal=narrow_filtered_signal, signal_artifacts=signal_artifacts)
        return feedback_value

    def check_cutoff_settings(self):
        """
        Function to check the correct definition of training band dictionary.
        """
        target_bands = 0
        for filter in self.filter_dict:
            if filter['type'] == 'training':
                target_bands += 1
        if target_bands == 1:
            return True
        else:
            return False


class PowerBasedNFTModel(components.Algorithm):
    def __init__(self, fs, filter_dict, l_baseline_t, update_feature_window,
                 update_rate, montage, target_channels, mode, apply_car,
                 apply_laplacian, right_ch_idx=None,
                 pct_tol_ocular=None, pct_tol_muscular=None):
        """
        Pipeline for Power-based Neurofeedback training. This class
        inherits from components.Algorithm. Therefore, it can be used to create
        standalone algorithms that can be used in compatible apps from
        medusa-platform for online experiments. See components.Algorithm to know
        more about this functionality.
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
        self.right_ch_idx = right_ch_idx
        self.mode = mode
        self.apply_car = apply_car
        self.apply_laplacian = apply_laplacian

        # Init variables
        self.baseline_value = None
        self.pct_tol = None

        # Set percentage tolerance to noisy signal
        if pct_tol_ocular is None and pct_tol_muscular is not None:
            self.pct_tol = pct_tol_muscular
        elif pct_tol_ocular is not None and pct_tol_muscular is None:
            self.pct_tol = pct_tol_ocular
        elif pct_tol_ocular is not None and pct_tol_muscular is not None:
            self.pct_tol = np.array([pct_tol_ocular, pct_tol_muscular])

        # # Check correct filter dict definition
        if not self.check_cutoff_settings():
            raise Exception('The number of frequency bands selected does not '
                            'match the Neurofeedback mode.')

        self.add_method('prep_method',
                        SignalPreprocessing(filter_dict=self.filter_dict,
                                            montage=self.montage,
                                            target_channels=self.target_channels,
                                            laplacian=self.apply_laplacian,
                                            car=self.apply_car))
        self.add_method('feat_ext_method',
                        PowerExtraction(fs=fs, l_baseline_t=l_baseline_t,
                                        update_feature_window=update_feature_window,
                                        update_rate=update_rate, mode=self.mode,
                                        right_ch_idx=right_ch_idx,
                                        pct_tol=self.pct_tol,
                                        f_dict=filter_dict))

    def calibration(self, eeg, **kwargs):
        """
        It pre-process eeg and gets signal filtered in artifact-related bands.
        Then, it calculates the baseline value.
        Parameters
        ----------
        eeg: numpy.ndarray
            [n_samples, n_channels]
        Returns
        -------
        baseline_value: float
        """
        original_signal, signal_artifacts = self.get_inst('prep_method'). \
            prep_fit_transform(signal=eeg, fs=self.fs)
        self.baseline_value = self.get_inst('feat_ext_method').set_baseline(
            signal=original_signal, signal_artifacts=signal_artifacts)

    def training(self, eeg):
        """
        It pre-process eeg, gets signal filtered in artifact-related bands. Then,
        it calculates the feedback value.
        Parameters
        ----------
        eeg: numpy.ndarray
            [n_samples, n_channels]
        Returns
        -------
        feedback_value: float
        """
        original_signal, signal_artifacts = self.get_inst('prep_method'). \
            prep_transform(signal=eeg)
        feedback_value = self.get_inst('feat_ext_method').band_power(
            signal=original_signal, signal_artifacts=signal_artifacts)
        return feedback_value

    def check_cutoff_settings(self):
        """
        Function to check the correct definition of training band dictionary.
        """
        target_bands = 0
        for filter in self.filter_dict:
            if filter['type'] == 'training' and len(filter['cutoff'][0]) != 0:
                target_bands += 1
        if self.mode == 'single':
            if target_bands == 1:
                return True
            else:
                return False
        elif self.mode == 'ratio':
            if target_bands == 2:
                return True
            else:
                return False


class NeurofeedbackData(components.ExperimentData):
    """Experiment info class for Neurofeedback training experiments. It records
    the important events that take place during a Neurofeedback run,
    allowing offline analysis."""

    def __init__(self, run_onsets, run_durations, run_success, run_pauses,
                 run_restarts, medusa_nft_app_settings, nft_values, nft_times,
                 nft_baseline):

        self.run_onsets = run_onsets
        self.run_durations = run_durations
        self.run_success = run_success
        self.run_pauses = run_pauses
        self.run_restarts = run_restarts
        self.medusa_nft_app_settings = medusa_nft_app_settings
        self.nft_values = nft_values
        self.nft_times = nft_times
        self.nft_baseline = nft_baseline

    def to_serializable_obj(self):
        rec_dict = self.__dict__
        for key in rec_dict.keys():
            if type(rec_dict[key]) == np.ndarray:
                rec_dict[key] = rec_dict[key].tolist()
        return rec_dict

    @classmethod
    def from_serializable_obj(cls, dict_data):
        return cls(**dict_data)
