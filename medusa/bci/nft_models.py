import numpy as np
# MEDUSA MODULES
from medusa import frequency_filtering, spatial_filtering
from medusa.eeg_standards import EEG_1010_81ch_pos
from medusa import components


class NFTModel:

    def __init__(self, freq_band, nft_cha_idx, receiver):
        self.freq_band = freq_band
        self.nft_cha_idx = nft_cha_idx

        self.fs = receiver.fs
        self.lcha = receiver.lcha

        self.b_power = None

    def apply_preprocessing_stage(self, eeg):
        """ This function pre-processes the signal according to the processing pipeline in neurofeedback training, consisting
            on filtering the signal spectrally (IIR, order 5) and spatially (4-order Laplacian).

            :param eeg: numpy ndarray
                EEG raw signal with dimensions [samples x ch]. Note that the TRIGGER channel must be DELETED before calling
                this function. In other words, eeg_signal must not contain the TRIGGER channel, only the EEG channels.
        """
        # Frequency filtering (IIR with order 5)
        filter = frequency_filtering.IIRFilter(order=5,
                                               cutoff=self.freq_band,
                                               btype='bandpass',filt_method='sosfilt')

        eeg = filter.fit_transform(signal=eeg,fs = self.fs)

        # Spatial filtering (Laplace, automatic over the desired electrodes)
        eeg = spatial_filtering.laplacian(s=eeg,
                                          mode='auto',
                                          channel_set=EEG_1010_81ch_pos,
                                          n=4,
                                          l_cha_to_filter=None,
                                          l_cha_laplace=None)
        return eeg

    def set_baseline(self, all_eeg, l_baseline_t):
        """ This function sets the power baseline, given the raw EEG containing the calibration phase.

            :param all_eeg: ndarray
                Raw EEG, without being pre-processed.

            :param l_baseline_t: float
                Number of seconds that last the calibration phase.

            IMPORTANT: It is recommended to give a raw EEG whose duration not only contains the calibration phase, but
            also the starting phase, in order to have enough samples to ignore the initial effects that cause the frequency
            filtering.
        """
        # Pre-process the data
        # IMPORTANT: The "starting..." phase is also included to avoid huge signal variations on the first samples due
        # to IIR filtering (in that case, the mean is corrected, but the variations would affect the std)
        eeg = self.apply_preprocessing_stage(all_eeg)

        # Get baseline
        baseline_s = int(l_baseline_t * self.fs)

        # Compute the log-power of the baseline data
        self.b_power = self.mean_power(eeg[-baseline_s:, self.nft_cha_idx])

    def get_nf_feature(self, all_eeg, l_segment_t):
        """ This function gets the real-time power amplitude relative to the baseline, given the raw EEG.

            :param all_eeg: ndarray
                Raw EEG, without being pre-processed.
            :param l_segment_t: float
                Number of seconds to consider for getting the feature.

            IMPORTANT: It is recommended to give a raw EEG whose duration not only contains the update segment in order
            to have enough samples to ignore the initial effects that cause the frequency filtering.
        """
        if self.b_power is None:
            raise ValueError('Baseline power is not specified!')

        # Pre-process all signal
        eeg = self.apply_preprocessing_stage(all_eeg)

        # Get valid window
        w_signal_s = int(l_segment_t * self.fs)

        # Get power, relative to the baseline
        feature = self.mean_power(eeg[-w_signal_s:, self.nft_cha_idx]) - self.b_power

        return feature

    def mean_power(self, eeg_chunk):
        """ This function computes the classical NF feature: the mean power across channels.

            :param eeg_chunk: ndarray
                Processed EEG. The function will use all channels and samples available.
        """
        return np.mean(np.log(np.mean(np.power(eeg_chunk, 2), axis=0)))
