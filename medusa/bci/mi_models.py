# PYTHON MODULES
import copy, warnings
# EXTERNAL MODULES
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
# MEDUSA MODULES
from medusa.storage.medusa_data import MedusaData
from medusa import frequency_filtering, spatial_filtering
from medusa.csp import CSP
from medusa.eeg_standards import EEG_1010_81ch_pos
from medusa.bci import mi_feat_extraction
from medusa.epoching import get_epochs_of_events
from medusa import dataset_splitting


class MIModelSettings:
    # PRE-PROCESSING
    p_filt_method = "IIR"
    p_filt_params = {
        'FIR': {'type': 'bandpass', 'fpass1': 8, 'fpass2': 15, 'order': 1000, 'method': 'filtfilt'},
        'IIR': {'type': 'bandpass', 'fpass1': 8, 'fpass2': 15, 'order': 5, 'method': 'lfilter'},
        'None': {'type': 'bandpass', 'fpass1': 0, 'fpass2': 0, 'order': 0, 'method': 'lfilter'}
    }

    p_notch_method = "None"
    p_notch_params = {
        'FIR': {'freq': 50.0, 'bw': 2.0, 'order': 201, 'method': 'filtfilt'},
        'IIR': {'freq': 50.0, 'bw': 2.0, 'order': 5, 'method': 'lfilter'},
        'None': {'freq': 0, 'bw': 0, 'order': 0, 'method': 'lfilter'}
    }

    p_spatial_method = "laplace"
    p_spatial_params = {
        'CAR': {},
        'laplace': {
            'mode': 'auto',
            'locations': EEG_1010_81ch_pos,
            'n': 4,
            'lcha_to_filter': ['C3', 'C4'],
            'lcha_laplace': [
                ['F3', 'T7', 'Cz', 'Pz'],
                ['F4', 'T8', 'Cz', 'Pz']
            ]
        },
        'None': {}
    }

    # FEATURE EXTRACTION
    f_method = "csp"
    f_params = {
        'std': {
            'w_trial_t': [1000, 4000],
            'use_calibration_baseline': True,
            'normalization': 'z'
        },
        'csp': {
            'w_trial_t': [1000, 4000],
            'use_calibration_baseline': True,
            'normalization': 'z',
            'n_csp_filt': 6,
            'type': 'medians'
        }
    }

    # FEATURE CLASSIFICATION
    c_method = "rLDA"
    c_params = {
        'rLDA': {'shrinkage': 'auto', 'solver': 'eigen', 'verbose': False}
    }


class MIModel:

    """ Standard model for MI classification. Frequency and spatial filtering + epoching + CSP + LDA """

    def __init__(self):
        # Settings
        self.settings = None
        # Data parameters that must remain constant to use the model
        self.fs = None                  # Sample rate of the EEG
        self.lcha = None                # Channels labels in order
        # Parameters to be fit
        self.notch_coeffs = None        # Notch filter coefficients
        self.filt_coeffs = None         # Frequency filter coefficients
        self.included_features = None   # Included features
        self.csp_filt = None            # CSP filter
        self.mi_classifier = None       # MI classifier
        self.filt_idx = None            # Selected CSP filters
        # States
        self.fitted = False
        # Temporal variables
        self.b_mean = None
        self.b_std = None

    def configure(self, settings):
        """
        Model configuration
        :param settings: CSPModelSettings
            Model settings.
        """
        if not isinstance(settings, MIModelSettings):
            raise ValueError("Parameter settings must be type CSPModelSettings")
        self.settings = settings
        self.fitted = False

    def apply_preprocessing_stage(self, eeg):
        # Frequency filtering
        if self.settings.p_filt_method != "None":
            eeg = frequency_filtering.apply_filter_offline(eeg, self.filt_coeffs[0], self.filt_coeffs[1],
                                                           method=self.settings.p_filt_params[self.settings.p_filt_method]['method'])

        # Notch filtering
        if self.settings.p_notch_method != "None":
            eeg = frequency_filtering.apply_filter_offline(eeg, self.notch_coeffs[0], self.notch_coeffs[1],
                                                           method=self.settings.p_notch_params[self.settings.p_n]['method'])

        # Spatial filtering
        if self.settings.p_spatial_method != 'None':
            if self.settings.p_spatial_method == 'laplace':
                eeg = spatial_filtering.apply_laplacian(eeg, self.lcha,
                                                        mode=self.settings.p_spatial_params['laplace']['mode'],
                                                        locations=self.settings.p_spatial_params['laplace']['locations'],
                                                        n=self.settings.p_spatial_params['laplace']['n'],
                                                        lcha_to_filter=self.settings.p_spatial_params['laplace']['lcha_to_filter'],
                                                        lcha_laplace=self.settings.p_spatial_params['laplace']['lcha_laplace'])
            elif self.settings.p_spatial_method == "CAR":
                eeg = spatial_filtering.apply_car(eeg)
        # Return
        return eeg

    def fit(self, mi_data, k_fold=True, k=None):

        # ============================================ CHECK ERRORS ================================================== #
        # Check that the model has not been fitted
        if self.fitted:
            raise Exception("The model has been fitted already")
        # Convert mds_data to list if necessary
        if not isinstance(mi_data, list):
            mi_data = [mi_data]
        # Check erp_data
        for d in mi_data:
            if not isinstance(d, MedusaData):
                raise Exception("Parameter mds_data must be a list of medusa.storage.medusa_data.MedusaData classes")
        # Check that all the data have the same sample rate and that corresponds to runs in copy mode
        self.fs = mi_data[0].eeg.fs
        self.lcha = mi_data[0].eeg.lcha
        for d in mi_data:
            if d.experiment.mode != "Train":
                raise Exception("Field MedusaData.experiment.mode must be 'Train' to fit the model")
            if d.eeg.fs != self.fs:
                raise Exception("Data must have the same sample rate to fit the model")
            if d.eeg.lcha != self.lcha:
                raise Exception("Data must have the same channels and in the same order to fit the model")
            if self.settings.f_params[self.settings.f_method]["w_trial_t"][1] > d.experiment.trial:
                raise Exception("The trial window is larger than the trial time used to train")
        # Make a copy of the data to not change the original data
        mi_data = copy.deepcopy(mi_data)
        # ======================================== PREPROCESSING STAGE =============================================== #
        # Frequecy filtering
        ftype = self.settings.p_filt_method
        if ftype != "None":
            btype = self.settings.p_filt_params[ftype]['type']
            order = self.settings.p_filt_params[ftype]['order']
            if btype == "bandpass":
                fpass = [self.settings.p_filt_params[ftype]['fpass1'],
                         self.settings.p_filt_params[ftype]['fpass2']]
            else:
                fpass = self.settings.p_filt_params[ftype]['fpass1']
            [b, a] = frequency_filtering.filter_designer(fpass, self.fs, order, ftype=ftype, btype=btype)
            self.filt_coeffs = [b, a]
        else:
            warnings.warn('[Frequency Filtering]\t\t No method has been selected.')
        # Notch filtering
        ftype = self.settings.p_notch_method
        if ftype != "None":
            freq = self.settings.p_notch_params[ftype]["freq"]
            bw = self.settings.p_notch_params[ftype]["bw"]
            order = self.settings.p_notch_params[ftype]["order"]
            fpass = [freq - bw / 2, freq + bw / 2]
            [b, a] = frequency_filtering.filter_designer(fpass, self.fs, order, ftype=ftype, btype='bandstop')
            self.notch_coeffs = [b, a]
            warnings.warn('[Notch Filtering] Notch filtering is usually unnecessary in MI applications')
        # Notch filtering
        sfilter = self.settings.p_spatial_method
        if sfilter != "laplace":
            warnings.warn('[Spatial Filtering] Laplace spatial filtering is usually the best option for this model')
        # Apply pre-processing stage
        for d in mi_data:
            d.eeg.signal = self.apply_preprocessing_stage(d.eeg.signal)
        # ====================================== FEATURE EXTRACTION STAGE ============================================ #
        if self.settings.f_method == "std":
            # Apply feature extraction stage
            w_trial_t = self.settings.f_params[self.settings.f_method]['w_trial_t']
            use_calibration_baseline = self.settings.f_params[self.settings.f_method]['use_calibration_baseline']
            norm = self.settings.f_params[self.settings.f_method]['normalization']
            trials, trials_info = mi_feat_extraction.extract_mi_trials_from_midata(mi_data, w_trial_t,
                                                                                   use_calibration_baseline=use_calibration_baseline, norm=norm)
            mi_labels = trials_info["mi_labels"]
            # Separate classes
            classes = np.unique(mi_labels)
            if len(classes) > 2:
                raise Exception("More than 2 motor imagery classes")
            # Classifier features
            x_train_cla = mi_feat_extraction.extract_std_mi_features(trials)
            y_train_cla = mi_labels
        elif self.settings.f_method == "csp":
            # Apply feature extraction stage
            w_trial_t = self.settings.f_params[self.settings.f_method]['w_trial_t']
            use_calibration_baseline = self.settings.f_params[self.settings.f_method]['use_calibration_baseline']
            norm = self.settings.f_params[self.settings.f_method]['normalization']
            n_csp_filt = self.settings.f_params[self.settings.f_method]['n_csp_filt']
            csp_type = self.settings.f_params[self.settings.f_method]['type']
            trials, trials_info = mi_feat_extraction.extract_mi_trials_from_midata(mi_data, w_trial_t,
                                                                                   use_calibration_baseline=use_calibration_baseline, norm=norm)
            mi_labels = trials_info["mi_labels"]
            # Separate classes
            classes = np.unique(mi_labels)
            if len(classes) > 2:
                raise Exception("More than 2 motor imagery classes")
            trials_train_c1 = trials[mi_labels == classes[0], :, :]
            trials_train_c2 = trials[mi_labels == classes[1], :, :]
            # Features to train the CSP filter
            x_train_csp_c1 = trials_train_c1.reshape(trials_train_c1.shape[0] * trials_train_c1.shape[1], trials_train_c1.shape[2])
            x_train_csp_c2 = trials_train_c2.reshape(trials_train_c2.shape[0] * trials_train_c2.shape[1], trials_train_c2.shape[2])
            # Fit CSP
            self.csp_filt = CSP()
            self.csp_filt.fit(x_train_csp_c1.T, x_train_csp_c2.T)
            self.filt_idx = mi_feat_extraction.get_csp_filter_idxs(n_csp_filt, self.csp_filt, type=csp_type, trials_c1=trials_train_c1, trials_c2=trials_train_c2)
            # Classifier features
            x_train_cla = mi_feat_extraction.extract_csp_mi_features(trials, self.csp_filt, self.filt_idx)
            y_train_cla = mi_labels
        else:
            raise Exception("Unknown feature extraction method")
        # ==================================== FEATURE CLASSIFICATION STAGE ========================================== #
        # METHOD 1: rLDA
        performance = None
        if self.settings.c_method == "rLDA":
            shrinkage = self.settings.c_params[self.settings.c_method]['shrinkage']
            solver = self.settings.c_params[self.settings.c_method]['solver']

            shrinkage = None if shrinkage == 'None' else shrinkage
            shrinkage = None if shrinkage == 0.0 else shrinkage
            if shrinkage is not None:
                if shrinkage == 'auto':
                    pass
                else:
                    shrinkage = float(shrinkage)
            if shrinkage is not None and solver == 'svd':
                warnings.warn("Solver 'svd' cannot be used if shrinkage is None.")

            # K-fold cross validation analysis
            if k_fold:
                k = x_train_cla.shape[0] if k is None else k    # If k is None, leave-one-out
                k_fold_sets = dataset_splitting.k_fold_split(x_train_cla, y_train_cla, k)
                cv_performance = list()
                for set in k_fold_sets:
                    cv_classifier = LinearDiscriminantAnalysis()
                    cv_classifier.set_params(shrinkage=shrinkage, solver=solver)
                    cv_classifier.fit(set['x_train'], set['y_train'])

                    y_test_pred = cv_classifier.predict(set['x_test'])
                    cv_performance.append(np.sum(y_test_pred == set['y_test']) / len(y_test_pred))
                print(cv_performance)
                performance = np.mean(cv_performance)

            # Final fit with all data
            mi_classifier = LinearDiscriminantAnalysis()
            mi_classifier.set_params(shrinkage=shrinkage, solver=solver)
            mi_classifier.fit(x_train_cla, y_train_cla)
            self.mi_classifier = mi_classifier
        else:
            raise Exception("Classification method %s is not implemented yet." % self.settings.c_method)
        # ======================================= PERFORMANCE ASSESSMENT ============================================= #
        if performance is None:
            pred_y_train_cla = self.mi_classifier.predict(x_train_cla)
            performance = np.sum(pred_y_train_cla == y_train_cla) / len(pred_y_train_cla)
        self.fitted = True
        return performance

    def predict(self, onsets, times, eeg):
        """ This function predicts a test data with the MI model and returns the probability of belonging to the class 1.

            :return pred_s:     Probability of belonging to the class 1 (if pred_s > 0.5, class1; otherwise, class0).
        """
        if self.fitted is False:
            raise Exception("The model has not been fitted")
        # Make a copy of the eeg signal in order to not modify it for further processing
        eeg = copy.deepcopy(eeg)
        # ------------------------------------------- PRE-PROCCESSING ------------------------------------------------ #
        eeg = self.apply_preprocessing_stage(eeg)
        # ------------------------------------------------------------------------------------------------------------ #
        # ----------------------------------------- FEATURE EXTRACTION ----------------------------------------------- #
        if self.settings.f_method == 'std':
            # Parameters
            w_trial_t = self.settings.f_params[self.settings.f_method]['w_trial_t']
            use_calibration_baseline = self.settings.f_params[self.settings.f_method]['use_calibration_baseline']
            norm = self.settings.f_params[self.settings.f_method]['normalization']
            # Get the trials
            if (self.b_mean is None or self.b_std is None) and use_calibration_baseline:
                raise ValueError('If calibration baseline is going to be used, b_mean and b_std need to be indicated by calling set_baselines().')
            trials = mi_feat_extraction.extract_mi_trials(times, eeg, onsets, self.fs, w_trial_t,
                                                          use_calibration_baseline=use_calibration_baseline,
                                                          b_mean=self.b_mean, b_std=self.b_std,norm=norm)
            # Get the features
            features = mi_feat_extraction.extract_std_mi_features(trials)
        elif self.settings.f_method == 'csp':
            # Parameters
            w_trial_t = self.settings.f_params[self.settings.f_method]['w_trial_t']
            use_calibration_baseline = self.settings.f_params[self.settings.f_method]['use_calibration_baseline']
            norm = self.settings.f_params[self.settings.f_method]['normalization']
            n_csp_filt = self.settings.f_params[self.settings.f_method]['n_csp_filt']
            # Get the trials
            if (self.b_mean is None or self.b_std is None) and use_calibration_baseline:
                raise ValueError('If calibration baseline is going to be used, b_mean and b_std need to be indicated by calling set_baselines().')
            trials = mi_feat_extraction.extract_mi_trials(times, eeg, onsets, self.fs, w_trial_t,
                                                          use_calibration_baseline=use_calibration_baseline,
                                                          b_mean=self.b_mean, b_std=self.b_std,norm=norm)
            # Get the features
            features = mi_feat_extraction.extract_csp_mi_features(trials, self.csp_filt, filt_idx=self.filt_idx)
        else:
            raise Exception("Unknown feature extraction method")
        # ------------------------------------------------------------------------------------------------------------ #
        # --------------------------------------- FEATURE CLASSIFICATION --------------------------------------------- #
        pred_s = self.mi_classifier.predict_proba(features)[:,1]
        # ------------------------------------------------------------------------------------------------------------ #
        return pred_s

    def predict_online(self, eeg, times, l_segment_t):
        """
        This function predicts an EEG segment in an online fashion.

        :param eeg:  ndarray
            Numpy array that contains the EEG samples of this feedback window until the current time.
        :param times:  ndarray
            Numpy array that contains the timestamps of the EEG samples until the current time.
        :param l_segment_t: integer
            Length of the segment to consider in ms. Then, the window to predict is [curr_time(ms)-l_segment, curr_time(ms)].
        """
        # --------------------------------------------- CHECK ERRORS ------------------------------------------------- #
        if self.fitted is False:
            raise Exception("The model has not been fitted")
        # Make a copy of the eeg signal in order to not modify it for further processing
        eeg = copy.deepcopy(eeg)

        # ------------------------------------------- PRE-PROCCESSING ------------------------------------------------ #
        eeg = self.apply_preprocessing_stage(eeg)

        # ----------------------------------------- FEATURE EXTRACTION ----------------------------------------------- #
        # Get the trial
        trial = get_epochs_of_events(timestamps=times, signal=eeg, onsets=[times[-1]], fs=self.fs,
                                     w_epoch_t=[-l_segment_t, 0], w_baseline_t=None)

        # Get the baseline
        use_calibration_baseline = self.settings.f_params[self.settings.f_method]['use_calibration_baseline']
        if use_calibration_baseline:
            if self.b_mean is None or self.b_std is None:
                raise ValueError('If calibration baseline is going to be used, b_mean and b_std need ' +
                                 'to be indicated by calling set_baselines().')
            else:
                if self.settings.f_params[self.settings.f_method]["normalization"] == 'z':  # Z-score
                    trial = (trial - self.b_mean) / self.b_std
                elif self.settings.f_params[self.settings.f_method]["normalization"] == 'dc':  # DC subtraction
                    trial = trial - self.b_mean

        # Feature extraction
        if self.settings.f_method == 'std':
            features = mi_feat_extraction.extract_std_mi_features(trial)
        elif self.settings.f_method == 'csp':
            features = mi_feat_extraction.extract_csp_mi_features(trial, self.csp_filt, self.filt_idx)
        else:
            raise Exception("Unknown feature extraction method")

        # --------------------------------------- FEATURE CLASSIFICATION --------------------------------------------- #
        pred_s = self.mi_classifier.predict_proba(features)[:, 1]

        return pred_s[0]

    def set_baseline(self, b_mean, b_std):
        """ This function sets the baseline of the calibration phase. """
        self.b_mean = b_mean
        self.b_std = b_std