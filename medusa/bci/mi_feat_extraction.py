"""
Created on Fri Dec 21 19:27:14 2018

Implementation of several methods used in ERP-Based Spellers.

@author: Eduardo Santamaría-Vázquez
"""

import numpy as np
import copy
from medusa import csp
from medusa.storage.medusa_data import MedusaData
from medusa.epoching import get_epochs_of_events


def extract_csp_mi_features(trials, csp, filt_idx=None):
    """ This method computes the standard motor imagery log-variance features given the CSP trained filters.
        :param trials: numpy array
        EEG signal corresponding to SMR trials with dimensions [trials x samples x channels]
        :param csp: CSP class
        CSP trained class.

        :return: numpy array
        Array with the computed features with dimensions [n_filt x 1]
    """
    if len(trials.shape) != 3:
        raise ValueError('[Extracting MI features] Parameter "trials" is not 3-dimensional!')

    if filt_idx is None:
        filt_idx = list(range(0,len(csp.w[0])))

    # Extract the features
    feat = np.zeros((0, len(filt_idx)))
    for i in range(trials.shape[0]):
        # Project the signal
        x = trials[i, :, :]
        p_csp = csp.project(x.T)    # projections x samples
        p_csp = p_csp[filt_idx,:]

        # Compute the log-variance features
        logvar = np.log(np.var(p_csp, axis=1))
        feat = np.vstack((feat, logvar))
    return feat


def get_csp_filter_idxs(n_filt, csp, type='classical', trials_c1=None, trials_c2=None):
    """ This method returns the indexes of the most n_filt discriminative filters according to the given method.
        :param n_filt:  No. of filters (must be even).
        :param csp:     CSP filtered class.
        :param type:    Type of selection (default: 'theoretical')
            - 'classical':      The extremes are taken. I.e., the method selects the first n_filt/2 filters, together
                                with the last n_filt/2 filters, which represents the largest and smallest eigenvalues,
                                respectively [1].
            - 'medians':        The ratio-of-medians method scores each filter according to the distance to the median
                                of the positive class; thus, favoring the robustness against outliers [1].
        :param trials_c1:   SMR trials of positive class with dimensions [trials x samples x channels]
        :param trials_c2:   SMR trials of negative class with dimensions [trials x samples x channels]

        :return Indexes.

        References:
            [1] Blankertz B, Tomioka R, Lemm S, Kawanabe M, Müller K. Optimizing Spatial filters for Robust EEG
            Single-Trial Analysis. Signal Processing Magazine, IEEE. 2008; 25(1):41–56.
            https://doi.org/10.1109/MSP.2008.4408441
            [2] Sannelli, C., Vidaurre, C., Müller, K. R., & Blankertz, B. (2019). A large scale screening study
            with a SMR-based BCI: Categorization of BCI users and differences in their SMR activity. PloS one,
            14(1), e0207351.

    """
    # Error detection
    if csp.w is None:
        raise Exception("CSP must be fitted first")
    if (n_filt % 2) != 0:
        raise ValueError('Parameter n_filt must be even.')
    m = int(n_filt / 2)
    N = len(csp.w[0])

    # Selection method
    if type == 'classical':
        filt_idx = np.concatenate((list(range(m)), list(np.arange(N - m, N))))
    elif type == 'medians':
        if trials_c1 is None or trials_c2 is None:
            raise ValueError('Parameters trials_c1 and trials_c2 data are required if the selection is "medians".')

        # Project the training data for each trial
        var1 = np.zeros((N, trials_c1.shape[0]))
        for i in range(trials_c1.shape[0]):
            var1[:, i] = np.var(csp.project(trials_c1[i, :, :].T), axis=1)
        var2 = np.zeros((N, trials_c2.shape[0]))
        for i in range(trials_c2.shape[0]):
            var2[:, i] = np.var(csp.project(trials_c2[i, :, :].T), axis=1)

        # Median along trials
        med1 = np.median(var1, axis=1)
        med2 = np.median(var2, axis=1)

        # Compute the ratio-of-medians score
        score = med1 / (med1 + med2)

        # Get the 'm' extremes
        sorted_idx = np.argsort(score)
        filt_idx = np.concatenate((sorted_idx[range(0, m)], sorted_idx[range(N - m, N)]))
    elif type == 'roc':
        # TODO: ROC-based selection
        raise ValueError('Not implemented yet (ROC).')
    else:
        raise ValueError('Unknown CSP filter selection method.')
    return filt_idx


def extract_std_mi_features(trials, ch_idx=None):
    """ This method computes the standard motor imagery log-variance features given the CSP trained filters.
        :param trials: numpy array
        EEG signal corresponding to SMR trials with dimensions [trials x samples x channels]
        :param ch_idx: list of numpy array
        Index of channels to consider

        :return: numpy array
        Array with the computed features with dimensions [n_cha x 1]
    """
    if len(trials.shape) != 3:
        raise ValueError('[Extracting MI features] Parameter "trials" is not 3-dimensional!')

    if ch_idx is None:
        ch_idx = list(range(0,len(trials.shape[3])))

    # Extract the features
    feat = np.zeros((0, len(ch_idx)))
    for i in range(trials.shape[0]):
        # Project the signal
        x = trials[i, :, :].T

        # Compute the log-variance features
        logvar = np.log(np.var(x[ch_idx, :], axis=1))
        feat = np.vstack((feat, logvar))
    return feat


def extract_mi_trials_from_midata(mi_data, w_trial_t, use_calibration_baseline=True, norm='z'):
    """

    This function computes the offline features for 2 class motor imagery of a single or several
    medusa.medusa_data.MedusaData classes

    """

    # Make a copy of the original data in order to not modify it
    mi_data = copy.deepcopy(mi_data)
    # Check errors
    if type(mi_data) != list:
        mi_data = [mi_data]
    # Init
    trials = None           # Features [N_epochs x features]
    mi_labels = None        # MI labels [N_epochs x 1]
    onsets = None
    # Compute features and labels
    for d in mi_data:
        # Check errors
        if not isinstance(d, MedusaData):
            raise TypeError('Data must be of type '+ str(MedusaData) +', not ' + str(type(d)))

        # Get epochs
        epochs = get_epochs_of_events(timestamps=d.eeg.times, signal=d.eeg.signal, onsets=d.experiment.onsets,
                                      fs=d.eeg.fs, w_epoch_t=w_trial_t, w_baseline_t=None, norm=norm)

        # Calibration baseline
        if use_calibration_baseline:
            b_mean, b_std = get_calibration_baseline(d)
            print('mean: ' + ' '.join(map(str, b_mean.tolist())))
            print('std: ' + ' '.join(map(str, b_std.tolist())))
            for i in range(epochs.shape[0]):
                epochs[i, :, :] = (epochs[i, :, :] - b_mean) / b_std

        # Stack features
        trials = np.concatenate((trials, epochs), axis=0) if trials is not None else epochs
        # Only if the data is copy mode
        if d.experiment.mode == "Train":
            # Stack MI labels
            mi_labels = np.concatenate((mi_labels, d.experiment.mi_labels), axis=0) if mi_labels is not None else np.array(d.experiment.mi_labels)
            # Stack MI onsets
            onsets = np.concatenate((onsets, d.experiment.onsets), axis=0) if onsets is not None else np.array(d.experiment.onsets)

    # Put all the info info in a dictionary
    trials_info = dict()
    trials_info['mi_labels'] = mi_labels
    trials_info['onsets'] = onsets

    return trials, trials_info


def extract_mi_trials(times, signal, onsets, fs, w_trial_t, norm='z', use_calibration_baseline=True, b_mean=None, b_std=None):
    """

    This function computes the offline features for 2 class motor imagery of a single or several
    medusa.medusa_data.MedusaData classes

    """
    # Make a copy
    signal = copy.deepcopy(signal)
    # Get epochs
    epochs = get_epochs_of_events(timestamps=times, signal=signal, onsets=onsets, fs=fs, w_epoch_t=w_trial_t,
                                  w_baseline_t=None, norm=norm)

    # Calibration baseline
    if use_calibration_baseline:
        if b_mean is not None and b_std is not None:
            for i in range(epochs.shape[0]):
                epochs[i, :, :] = (epochs[i, :, :] - b_mean) / b_std
        else:
            Warning('[extract_mi_trials] Baseline mean and standard deviation are None. Normalization was not applied.')

    return epochs


def get_calibration_baseline(medusa_data, w_calibration_t=None):
    """ This function computes the mean and standard deviation of the baseline.

    :param medusa_data: MedusaData
    Already pre-processed EEG signal of MI
    :param w_calibration_t: ndarray
    Window of the calibration phase in milliseconds with respect of the very first start of the recording.

    :return b_mean, b_std
    """

    # Get the calibration phase window in milliseconds
    if w_calibration_t is None:
        w_calibration_t = np.array([medusa_data.experiment.start,
                                    medusa_data.experiment.start+medusa_data.experiment.calibration])

    # Compute the mean and standard deviation of the calibration
    idx_b = [np.argmin(np.abs(medusa_data.eeg.times - (medusa_data.eeg.times[0] + w_calibration_t[0]/1000))),
             np.argmin(np.abs(medusa_data.eeg.times - (medusa_data.eeg.times[0] + w_calibration_t[1]/1000)))]
    b_mean = np.mean(medusa_data.eeg.signal[idx_b[0]:idx_b[1],:], axis=0)
    b_std = np.std(medusa_data.eeg.signal[idx_b[0]:idx_b[1],:], axis=0)
    print('no.samples calibration: ' + str((idx_b[1]-idx_b[0])))

    return b_mean, b_std