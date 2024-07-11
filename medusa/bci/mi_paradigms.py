"""
In this module you will find useful functions and classes to operate with data
recorded using motor imagery paradigms, which are widely used by the BCI
community. Enjoy!

@author: Sergio Pérez-Velasco & Víctor Martínez-Cagigal
"""
# Built-in imports
import copy, warnings
from abc import abstractmethod

# External imports
import numpy as np
from tqdm import tqdm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report

# Medusa imports
from medusa import IIRFilter, car
from medusa.epoching import normalize_epochs
from medusa import get_epochs_of_events, resample_epochs
from medusa import components
from medusa import meeg
from medusa.spatial_filtering import CSP, LaplacianFilter
from medusa import classification_utils as cu

class MIData(components.ExperimentData):
    # TODO: Check everything

    """Class with the necessary attributes to define motor imagery (MI)
    experiments. It provides compatibility with high-level functions for this
    MI paradigms in BCI module.
    """

    def __init__(self, mode, onsets, w_trial_t,
                 calibration_onset_w=None, w_preparation_t=None, w_rest_t=None,
                 mi_labels=None, mi_labels_info=None, mi_result=None,
                 paradigm_info=None, **kwargs):
        """MIData constructor

        Parameters
        ----------
        mode : str {"train"|"test"|"guided test"}
            Mode of this run.
        onsets : list or numpy.ndarray [n_stim x 1]
            Timestamp of the cue with respect to the EEG signal (just when the
            motor imagery starts).
        w_trial_t: list [start, end]
            Temporal window of the motor imagery with respect to each onset in
            ms. For example, if  w_trial_t = [500, 4000] the subject was
            performing the motor imagery task from 500ms to 4000ms after
            the onset.
        calibration_onset_w: list [start, end]
            Timestamps of the onsets regarding the calibration window,
            if exists, respect to the EEG signal.
        w_preparation_t: list [start, end]
            Temporal window of the preparation time (no motor imagery),
            if exists, with respect to each onset in ms.
        w_rest_t: list [start, end]
            Temporal window of the rest time (no motor imagery),
            if exists, with respect to each onset in ms.
        mi_labels : list or numpy.ndarray [n_mi_labels x 1]
            Only in train mode. Contains the mi labels of each stimulation,
            as many as classes in the experiment.
        mi_labels_info : dict
            Contains the description of the mi labels.
            Example:
                mi_labels_info =
                    {0: "Rest",
                    1: "Left_hand",
                    2: "Right_hand"}
        mi_result : list or numpy.ndarray [n_mi_labels x 1]
            Result of this run. Each position contains the data of the
            selected target at each trial.
        paradigm_info: dict()
            Recommended but not mandatory. Use this variable to keep the
            information regarding the different timings of the paradigm.
        kwargs : kwargs
            Custom arguments that will also be saved in the class
            (e.g., timings, calibration gaps, etc.)
        """

        # Check errors
        mode = mode.lower()
        if mode == 'train':
            if mi_labels is None:
                raise ValueError('Attribute "mi_labels" must be provided in '
                                 'train mode')

        # Standard attributes
        self.mode = mode
        self.onsets = np.array(onsets)
        self.w_trial_t = np.array(w_trial_t)
        self.calibration_onset_w = np.array(calibration_onset_w)
        self.w_preparation_t = np.array(w_preparation_t)
        self.w_rest_t = np.array(w_rest_t)
        self.mi_labels = np.array(mi_labels) if mi_labels is not None else \
            mi_labels
        self.mi_labels_info = mi_labels_info
        self.mi_result = np.array(mi_result) if mi_result is not None else \
            mi_result
        self.paradigm_info = paradigm_info

        # Optional attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_serializable_obj(self):
        rec_dict = self.__dict__
        for key in rec_dict.keys():
            if type(rec_dict[key]) == np.ndarray:
                rec_dict[key] = rec_dict[key].tolist()
        return rec_dict

    @classmethod
    def from_serializable_obj(cls, dict_data):
        return cls(**dict_data)


class MIDataset(components.Dataset):
    """This class inherits from medusa.data_structures.Dataset, increasing
    its functionality for datasets with data from MI experiments. It
    provides common ground for the rest of functions in the module.
    """

    def __init__(self, channel_set, fs=None, biosignal_att_key='eeg',
                 experiment_att_key='midata', experiment_mode=None,
                 track_attributes=None):
        """ Constructor

        Parameters
        ----------
        channel_set : meeg.EEGChannelSet
            EEG channel set. Only these channels will be kept in the dataset,
            the others will be discarded. Also, the signals will be rearranged,
            keeping the same channel order, avoiding errors in future stages of
            the signal processing pipeline
        fs : int, float or None
            Sample rate of the recordings. If there are recordings with
            different sample rates, the consistency of the dataset can be
            still assured using resampling
        biosignal_att_key : str
            Name of the attribute containing the target biosginal that will be
            used to extract the features. It has to be the same in all
            recordings (e.g., 'eeg', 'meg').
        experiment_att_key : str or None
            Name of the attribute containing the target experiment that will be
            used to extract the features. It has to be the same in all
            recordings (e.g., 'mi_left_right', 'rest_mi'). It is
            mandatory when a recording of the dataset contains more than 1
            experiment data
        experiment_mode : str {'train'|'test'|'guided test'|None}
            Mode of the experiment. If this dataset will be used to fit a model,
            set to train to avoid errors
        track_attributes: dict of dicts or None
            This parameter indicates custom attributes that must be tracked in
            feature extraction functions and how. The keys are the name of the
            attributes, whereas the values are dicts indicating the tracking
            mode {'concatenate'|'append'} and parent. Option concatenate is
            only available for attributes of type list or numpy arrays,
            forming a 1 dimensional array with the data from all recordings.
            Option append is used to save all kind of objects for each
            recording, forming a list whose length will be the number of
            recordings in the dataset. A set of default attributes is defined,
            so this parameter will be None in most cases. Example to track 2
            custom attributes (i.e., date and experiment_equipment):
                track_attributes = {
                    'date': {
                        'track_mode': 'append',
                        'parent': None
                    },
                    'experiment_equipment': {
                        'track_mode': 'append',
                        'parent': experiment_att_key
                    }
                }
        """
        # Check errors
        if experiment_mode is not None:
            if experiment_mode not in ('train', 'test', 'guided test'):
                raise ValueError('Parameter experiment_mode must be '
                                 '{train|test|guided test|None}')

        # Default track attributes
        default_track_attributes = {
            'subject_id': {
                'track_mode': 'append',
                'parent': None
            },
            'onsets': {
                'track_mode': 'concatenate',
                'parent': experiment_att_key
            },
            'w_trial_t': {
                'track_mode': 'append',
                'parent': experiment_att_key
            },
            'calibration_onset_w': {
                'track_mode': 'append',
                'parent': experiment_att_key
            },
            'w_preparation_t': {
                'track_mode': 'append',
                'parent': experiment_att_key
            },
            'w_rest_t': {
                'track_mode': 'append',
                'parent': experiment_att_key
            },
            'mi_labels_info': {
                'track_mode': 'append',
                'parent': experiment_att_key
            },
            'paradigm_info': {
                'track_mode': 'append',
                'parent': experiment_att_key
            }
        }

        if experiment_mode in ['train', 'guided test']:
            default_track_attributes_train = {
                'mi_labels': {
                    'track_mode': 'concatenate',
                    'parent': experiment_att_key
                }
            }
            default_track_attributes = {
                **default_track_attributes,
                **default_track_attributes_train
            }
        if experiment_mode in ['test', 'guided test']:
            default_track_attributes_test = {
                'mi_result': {
                    'track_mode': 'concatenate',
                    'parent': experiment_att_key
                }
            }
            default_track_attributes = {
                **default_track_attributes,
                **default_track_attributes_test
            }
        track_attributes = \
            default_track_attributes if track_attributes is None else \
                {**default_track_attributes, **track_attributes}

        # Class attributes
        self.channel_set = channel_set
        self.fs = fs
        self.biosignal_att_key = biosignal_att_key
        self.experiment_att_key = experiment_att_key
        self.experiment_mode = experiment_mode
        self.track_attributes = track_attributes

        # Consistency checker
        checker = self.__get_consistency_checker()
        super().__init__(consistency_checker=checker)

    def __get_consistency_checker(self):
        """Creates a standard consistency checker for MI datasets

        Returns
        -------
        checker : data_structures.ConsistencyChecker
            Standard consistency checker for MI feature extraction
        """
        # Create consistency checker
        checker = components.ConsistencyChecker()
        # Check that the biosignal exists
        checker.add_consistency_rule(
            rule='check-attribute',
            rule_params={'attribute': self.biosignal_att_key}
        )
        checker.add_consistency_rule(
            rule='check-attribute-type',
            rule_params={'attribute': self.biosignal_att_key,
                         'type': meeg.EEG}
        )
        # Check channels
        checker.add_consistency_rule(
            rule='check-values-in-attribute',
            rule_params={'attribute': 'channels',
                         'values': self.channel_set.channels},
            parent=self.biosignal_att_key + '.channel_set'
        )
        # Check sample rate
        if self.fs is not None:
            checker.add_consistency_rule(rule='check-attribute-value',
                                         rule_params={'attribute': 'fs',
                                                      'value': self.fs},
                                         parent=self.biosignal_att_key)
        else:
            warnings.warn('Parameter fs is None. The consistency of the '
                          'dataset cannot be assured. Still, you can use '
                          'target_fs parameter for feature extraction '
                          'and everything should be fine.')

        # Check experiment
        checker.add_consistency_rule(
            rule='check-attribute',
            rule_params={'attribute': self.experiment_att_key}
        )
        checker.add_consistency_rule(
            rule='check-attribute-type',
            rule_params={'attribute': self.experiment_att_key,
                         'type': MIData}
        )

        # Check track_attributes
        if self.track_attributes is not None:
            for key, value in self.track_attributes.items():
                checker.add_consistency_rule(
                    rule='check-attribute',
                    rule_params={'attribute': key},
                    parent=value['parent']
                )
                if value['track_mode'] == 'concatenate':
                    checker.add_consistency_rule(
                        rule='check-attribute-type',
                        rule_params={'attribute': key,
                                     'type': [list, np.ndarray, dict]},
                        parent=value['parent']
                    )

        return checker

    def custom_operations_on_recordings(self, recording):
        # Select channels
        eeg = getattr(recording, self.biosignal_att_key)
        eeg.change_channel_set(self.channel_set)
        return recording


class StandardPreprocessing(components.ProcessingMethod):
    """Just the common preprocessing applied in MI-based BCI. Simple,
    quick and effective: frequency IIR filter followed by common average
    reference (CAR) spatial filter.
    """

    def __init__(self, order=5, cutoff=[8, 30], btype='bandpass',
                 temp_filt_method='sosfiltfilt'):
        super().__init__(fit_transform_signal=['signal'],
                         fit_transform_dataset=['dataset'])
        # Parameters
        self.order = order
        self.cutoff = cutoff
        self.btype = btype
        self.temp_filt_method = temp_filt_method

        # Variables that
        self.iir_filter = None

    def fit(self, fs, n_cha=None):
        """Fits the IIR filter.

        Parameters
        ----------
        fs: float
            Sample rate of the signal.
        n_cha: int
            Number of channels. Used to compute the initial conditions of the
            frequency filter. Only required with sosfilt filtering method
            (online filtering)
        """
        self.iir_filter = IIRFilter(order=self.order,
                                    cutoff=self.cutoff,
                                    btype=self.btype,
                                    filt_method=self.temp_filt_method)
        self.iir_filter.fit(fs, n_cha=n_cha)

    def transform_signal(self, signal):
        """Transforms an EEG signal applying IIR filtering and CAR sequentially

        Parameters
        ----------
        signal: np.array or list
            Signal to transform. Shape [n_samples x n_channels]
        """
        signal = self.iir_filter.transform(signal)
        signal = car(signal)
        return signal

    def transform_dataset(self, dataset: MIDataset, show_progress_bar=True,
                          **kwargs):
        """Transforms an MIDataset applying IIR filtering and CAR sequentially

        Parameters
        ----------
        signal: np.array or list
            Signal to transform. Shape [n_samples x n_channels]
        """
        pbar = None
        if show_progress_bar:
            pbar = tqdm(total=len(dataset.recordings),
                        desc='Preprocessing')
        for rec in dataset.recordings:
            eeg = getattr(rec, dataset.biosignal_att_key)
            eeg.signal = self.transform_signal(eeg.signal)
            setattr(rec, dataset.biosignal_att_key, eeg)
            if show_progress_bar:
                pbar.update(1)
        if show_progress_bar:
            pbar.close()
        return dataset

    def fit_transform_signal(self, signal, fs):
        """Fits the IIR filter and transforms an EEG signal applying IIR
        filtering and CAR sequentially

        Parameters
        ----------
        signal: np.array or list
            Signal to transform. Shape [n_samples x n_channels]
        fs: float
            Sample rate of the signal.
        """
        self.iir_filter = IIRFilter(order=self.order,
                                    cutoff=self.cutoff,
                                    btype=self.btype,
                                    filt_method=self.temp_filt_method)
        signal = self.iir_filter.fit_transform(signal, fs)
        signal = car(signal)
        return signal

    def fit_transform_dataset(self, dataset, show_progress_bar=True):
        """Fits the IIR filter and transforms an EEG signal applying the
        filter and CAR sequentially. Each recording is preprocessed
        independently, taking into account possible differences in sample rate.

        Parameters
        ----------
        dataset: MIDataset
            MIDataset with the recordings to be preprocessed.
        show_progress_bar: bool
            Show progress bar
        """
        pbar = None
        if show_progress_bar:
            pbar = tqdm(total=len(dataset.recordings),
                        desc='Preprocessing')
        for rec in dataset.recordings:
            eeg = getattr(rec, dataset.biosignal_att_key)
            eeg.signal = self.fit_transform_signal(eeg.signal, eeg.fs)
            setattr(rec, dataset.biosignal_att_key, eeg)
            if show_progress_bar:
                pbar.update(1)
        if show_progress_bar:
            pbar.close()
        return dataset


class StandardFeatureExtraction(components.ProcessingMethod):
    """
    Standard feature extraction method for MI-based spellers. Basically,
    it gets the raw epoch for each MI event.
    """

    def __init__(self, safe_copy=True, **kwargs):
        """
        Class constructor. All parameters except "safe_copy" must be specified
        in each method.

        Parameters
        -----------
        safe_copy : bool
            Makes a safe copy of the signal to avoid changing the original
            samples due to references.
        """
        super().__init__(transform_signal=['x'],
                         transform_dataset=['x', 'x_info'])
        self.safe_copy = safe_copy

    def transform_signal(self, times, signal, fs, onsets, w_epoch_t=(0, 3000),
                         baseline_mode="trial", w_baseline_t=(-1500, -500),
                         norm='z', target_fs=128, concatenate_channels=False,
                         **kwargs):
        """
        Method to extract epochs from an individual signal.

        Parameters
        -----------------
        times : ndarray (n_samples,)
            Timestamp array
        signal: ndarray (n_samples x n_channels)
            Signal data.
        fs : int
            Sampling frequency.
        onsets: ndarray (n_trials,)
            Timestamps for the MI event onsets.
        w_epoch_t : list, tuple, or ndarray
            Temporal window in ms for each epoch relative to the event onset
            (e.g., [0, 3000])
        baseline_mode : basestring {'run', 'trial', None}
            If "run", the baseline will be extracted from the very beginning
            of the run (i.e., the first samples of the signal).
            If "trial" (default), the baseline will be extracted for each
            trial relative to the onset.
            If None, no baseline extraction will be performed.
        w_baseline_t : list, tuple, or ndarray
            Temporal window in ms to be used for baseline normalization.
            If baseline_mode = "run", the window is relative to the start of
            the signal.
            If baseline_mode = "trial", the window is relative to each trial
            onset (e.g., [-1500, -500]).
        norm : str {'z'|'dc'}
            Type of baseline normalization. Set to 'z' for Z-score
            normalization (subtract the mean and divide by the std),
            or 'dc' for DC normalization (subtract the mean).
        target_fs : float of None
            Target sample rate of each epoch. If None, no resampling will be
            applied. Please note that, in this case, all the recordings must
            have the same sample rate.
        concatenate_channels : bool
            This parameter controls the shape of the feature array. If True, all
            channels will be concatenated, returning an array of shape [n_events
            x (samples x channels)]. If false, the array will have shape
            [n_events x samples x channels].

        Returns
        -----------
        features : ndarray
            MI epochs extracted with shape [n_events x samples x channels], or
            [n_events x (samples x channels)] if concatenate_channels == True.
        """
        # Avoid changes in the original signal (this may not be necessary)
        if self.safe_copy:
            signal = signal.copy()

        # Baseline options
        if baseline_mode == "sliding":
            baseline_mode = "trial"
        assert baseline_mode in ('run', 'trial', None), \
            ValueError('Parameter baseline_mode must be {"run", "trial", None}')
        if baseline_mode in ('run', None):
            w_baseline_t = None
            norm = None

        # Extract features
        features = get_epochs_of_events(
            timestamps=times, signal=signal, onsets=onsets, fs=fs,
            w_epoch_t=w_epoch_t,
            w_baseline_t=w_baseline_t, norm=norm
        )

        # Common baseline for the entire run
        if baseline_mode == "run":
            # Here baseline is taken relative to the start of the signal
            norm_epoch_s = np.array(w_baseline_t * fs / 1000, dtype=int)
            norm_epoch = np.expand_dims(signal[norm_epoch_s, :], axis=0)
            features = normalize_epochs(features, norm_epochs=norm_epoch,
                                        norm=self.norm)

        # Apply resampling if required
        if target_fs is not None:
            if target_fs > fs:
                raise warnings.warn('Target fs is greater than data fs')
            features = resample_epochs(features, w_epoch_t, target_fs)

        # Channel concatenation if desired
        if concatenate_channels:
            features = np.squeeze(features.reshape(
                (features.shape[0], features.shape[1] * features.shape[2], 1)))

        return features

    def transform_dataset(self, dataset, show_progress_bar=True,
                          w_epoch_t=(0, 3000), baseline_mode="trial",
                          w_baseline_t=(-1500, -500), norm='z',
                          target_fs=128, concatenate_channels=False,
                          sliding_w_lims_t=None, sliding_t_step=None,
                          sliding_win_len=None, **kwargs):
        """
        Method to extract epochs from an entire MIDataset

        High level function to easily extract features from EEG recordings
        and save useful info for later processing. Nevertheless, the provided
        functionality has several limitations and it will not be suitable for
        all cases and processing pipelines. If it does not fit your needs,
        create a custom function iterating the recordings and using
        extract_erp_features, a much more low-level and general function. This
        function does not apply any preprocessing to the signals, this must
        be done before.

        Most parameters are shared with
        StandardFeatureExtraction.transform_signal() method, check it for
        more information. Only the new parameters or those that behave
        differently than the aforementioned method are detailed below. These
        new parameters are related to a continuous extraction of epochs using a
        sliding window between a desired epoch range, instead of extracting only
        one epoch for each onset.

        Parameters
        ----------
        dataset: MIDataset
            MIDataset instance containing the MIData recordings.
        show_progress_bar: bool
            Boolean to show (or not) the progress bar info.
        sliding_w_lims_t: list, tuple, ndarray, or None
            2D window that indicates the range (start, end) for the sliding
            window approach. If None, no sliding window would be applied.
            This parameter delimits the number of windows to be extracted for
            each onset, so none of them can fall outside the range (excluding
            baseline).
        sliding_t_step: int, None
            Step in samples that is used to separate the sliding windows. If
            None, no sliding window would be applied.
        sliding_win_len: int, None
            Length in samples of the sliding windows. Please note that the
            w_epoch_t parameter would not be used if the sliding window approach
            is used. If None, no sliding window would be applied.
        baseline_mode : {'run', 'trial', 'sliding', None}
            The baseline_mode has an additional feature when sliding window is
            used:
                - "run": common baseline extracted from the start of the run.
                - "trial": common baseline for the trial, i.e., all the windows
                that belongs to an onset; extracted relative to the onset.
                - "sliding": baseline is applied to each sliding window,
                and it is relative to the start of each sliding window.
                - None: no baseline.

        Returns
        -------
        features : numpy.ndarray
            Array with the biosignal samples arranged in epochs.
        track_info : dict
            Dictionary with tracked information across all recordings.
        """

        # Continuous settings
        use_continuous = False
        if sliding_w_lims_t is not None or sliding_t_step is not None or \
                sliding_win_len is not None:
            assert sliding_w_lims_t is not None and sliding_t_step is not \
                   None and sliding_win_len is not None, \
                ValueError("All these three parameters {sliding_w_lims_t, "
                           "sliding_t_step, sliding_win_len} must be "
                           "specified to use sliding windows. If not desired, "
                           "then put all three to None. Aborting.")
            use_continuous = True

        # Avoid changes in the original recordings (this may not be necessary)
        if self.safe_copy:
            dataset = copy.deepcopy(dataset)

        # Avoid consistency problems
        if dataset.fs is None and target_fs is None:
            raise ValueError('The consistency of the features is not assured '
                             'since dataset.fs and target_fs are both None. '
                             'Specify one of these parameters')

        # Additional track attributes
        track_attributes = dataset.track_attributes

        # Initialization
        track_info = dict()
        for key, value in track_attributes.items():
            if value['track_mode'] == 'append':
                track_info[key] = list()
            elif value['track_mode'] == 'concatenate':
                track_info[key] = None
            else:
                raise ValueError('Unknown track mode')

        # Init progress bar
        pbar = None
        if show_progress_bar:
            pbar = tqdm(total=len(dataset.recordings),
                        desc='Extracting features')

        # Compute features
        features_list = list()
        for run_counter, rec in enumerate(dataset.recordings):
            # Extract recording experiment and biosignal
            rec_exp = getattr(rec, dataset.experiment_att_key)
            rec_sig = getattr(rec, dataset.biosignal_att_key)

            # If continuous: get different windows relative to the onsets
            if use_continuous:
                # Get windows
                start, stop = sliding_w_lims_t      # respect to each onset
                step_array = np.arange(0, (stop - start - sliding_win_len) //
                                       sliding_t_step + 1)
                win_0 = start + step_array * sliding_t_step
                win_1 = win_0 + sliding_win_len
                list_w_epoch_t = [(w0, w1) for w0, w1 in zip(win_0, win_1)
                                  if w1 <= stop]

                # Get baselines
                if baseline_mode == "sliding":
                    bas_0 = start + w_baseline_t[0] + step_array * sliding_t_step
                    bas_1 = start + w_baseline_t[1] + step_array * sliding_t_step
                    list_w_bas_t = [(w0, w1) for w0, w1 in zip(bas_0, bas_1)]
                elif baseline_mode == "trial":
                    bas_0 = start + w_baseline_t[0] + step_array * 0
                    bas_1 = start + w_baseline_t[1] + step_array * 0
                    list_w_bas_t = [(w0, w1) for w0, w1 in zip(bas_0, bas_1)]
                elif baseline_mode == "run":
                    list_w_bas_t = [w_baseline_t for _ in range(len(
                        list_w_epoch_t))]
                else:
                    list_w_bas_t = [None for _ in range(len(list_w_epoch_t))]

                # Modify the mi_labels
                if hasattr(rec.midata, "mi_labels"):
                    rec.midata.mi_labels = \
                        np.tile(rec.midata.mi_labels,
                                (len(list_w_epoch_t), 1)).flatten()
            else:
                list_w_epoch_t = [w_epoch_t]
                list_w_bas_t = [w_baseline_t]

            # For each window
            for i in range(len(list_w_epoch_t)):

                # Get features
                rec_feat = self.transform_signal(
                    times=rec_sig.times,
                    signal=rec_sig.signal,
                    fs=rec_sig.fs,
                    onsets=rec_exp.onsets,
                    w_epoch_t=list_w_epoch_t[i],
                    w_baseline_t=list_w_bas_t[i],
                    baseline_mode=baseline_mode,
                    norm=norm,
                    target_fs=target_fs,
                    concatenate_channels=concatenate_channels
                )
                features_list.append(rec_feat)
            features = np.concatenate((features_list), axis=0) if (
                features_list) else None
            # Track experiment info
            for key, value in track_attributes.items():
                if value['parent'] is None:
                    parent = rec
                else:
                    parent = rec
                    for p in value['parent'].split('.'):
                        parent = getattr(parent, p)
                att = getattr(parent, key)
                if value['track_mode'] == 'append':
                    track_info[key].append(att)
                elif value['track_mode'] == 'concatenate':
                    track_info[key] = np.concatenate(
                        (track_info[key], att), axis=0
                    ) if track_info[key] is not None else att
                else:
                    raise ValueError('Unknown track mode')
            if show_progress_bar:
                pbar.update(1)
        if show_progress_bar:
            pbar.close()

        return features, track_info


class CSPFeatureExtraction(components.ProcessingMethod):
    """Common Spatial Patterns (CSP) feature extraction method for MI-based
    spellers.

    Processing pipeline:
    - Use of StandardFeatureExtraction to get the raw epoch of each MI event.
    - Extract CSP features of those MI events.
    - Log-var features
    """

    def __init__(self, n_filters=4, safe_copy=True,
                 normalize_log_vars=True, **kwargs):
        """Class constructor.

        n_filters : int or None
            Number of most discriminant CSP filters to decompose the signal
            into (must be less or equal to the number of channels in your
            signal). If None, all filters will be used.
        safe_copy : bool
            Makes a safe copy of the signal to avoid changing the original
            samples due to references
        normalize_log_vars : bool
            If true, log-var features are normalized.
        **kwargs : dict()
            Parameters from StandardFeatureExtraction are not detailed here.
            Please refer to the StandardFeatureExtraction documentation to know
            more, as they are passed through kwargs.

        After using the function self.fit(), the attributes are computed:

        Attributes
        ----------
        CSP : CSP class with attributes filters, eigenvalues, patterns and
            methods fit and project.

            filters : {(…, M, M) numpy.ndarray, (…, M, M) matrix}
                Mixing matrix (spatial filters are stored in columns).
            eigenvalues : (…, M) numpy.ndarray
                Eigenvalues of w.
            patterns : numpy.ndarray
                De-mixing matrix (activation patterns are stored in columns).
        """
        self.normalize_log_vars = normalize_log_vars
        self.feature_extractor = StandardFeatureExtraction(safe_copy=safe_copy)
        self.CSP = CSP(n_filters=n_filters, selection="extremes")

    def fit_signal(self, labels, times, signal, fs, onsets, **kwargs):
        # Standard extraction: just epoching (epochs x samples x channels)
        features = self.feature_extractor.transform_signal(
            times=times, signal=signal, fs=fs, onsets=onsets, **kwargs
        )

        # Fit the CSP filter
        self.CSP.fit(X=features, y=labels)
        return features

    def fit_dataset(self, dataset, show_progress_bar=True, **kwargs):
        # Standard extraction: just epoching (epochs x samples x channels)
        features, track_info = self.feature_extractor.transform_dataset(
            dataset=dataset, show_progress_bar=show_progress_bar, **kwargs
        )

        # Fit the CSP filter
        self.CSP.fit(X=features, y=track_info['mi_labels'])
        return features, track_info

    def transform_signal(self, times, signal, fs, onsets, **kwargs):
        if not self.CSP.is_fitted:
            raise ValueError('CSPFeatureExtraction must be fitted before '
                             'predict!')

        # Standard extraction: just epoching (epochs x samples x channels)
        features = self.feature_extractor.transform_signal(
            times=times, signal=signal, fs=fs, onsets=onsets, **kwargs
        )

        # Project using CSP
        projection = self.CSP.project(X=features) # trials x samples x projects

        # Log-variance features across samples
        log_var = self._get_log_vars(projection, self.normalize_log_vars)
        return log_var

    def transform_dataset(self, dataset, show_progress_bar=True, **kwargs):
        if not self.CSP.is_fitted:
            raise ValueError('CSPFeatureExtraction must be fitted before '
                             'predict!')

        # Standard extraction: just epoching (epochs x samples x channels)
        features, track_info = self.feature_extractor.transform_dataset(
            dataset=dataset, show_progress_bar=show_progress_bar, **kwargs
        )

        # Project using CSP
        projection = self.CSP.project(X=features)

        # Log-variance features
        log_var = self._get_log_vars(projection, self.normalize_log_vars)
        return log_var, track_info

    def fit_transform_signal(self, labels, times, signal, fs, onsets, **kwargs):
        # Fit
        features = self.fit_signal(labels, times, signal, fs, onsets, **kwargs)

        # Project using CSP
        projection = self.CSP.project(X=features)  # trials x samples x projects

        # Log-variance features across samples
        log_var = self._get_log_vars(projection, self.normalize_log_vars)
        return log_var

    def fit_transform_dataset(self, dataset, show_progress_bar=True, **kwargs):
        # Fit
        features, track_info = self.fit_dataset(
            dataset, show_progress_bar=show_progress_bar, **kwargs)

        # Project using CSP
        projection = self.CSP.project(X=features)

        # Log-variance features
        log_var = self._get_log_vars(projection, self.normalize_log_vars)
        return log_var, track_info

    @staticmethod
    def _get_log_vars(projection, normalize):
        if normalize:
            return np.log(
                np.var(projection, axis=1) /
                np.tile(np.sum(np.var(projection, axis=1), axis=1),
                        (projection.shape[2], 1)).T
            )
        else:
            return np.log(np.var(projection, axis=1))


class MIModel(components.Algorithm):
    """Skeleton class for MI-based BCIs models. This class inherits from
    components.Algorithm. Therefore, it can be used to create standalone
    algorithms that can be used in compatible apps from medusa-platform
    for online experiments. See components.Algorithm to know more about this
    functionality.

    Related tutorials:

        - Overview of mi_paradigms module [LINK]
        - Create standalone models for MI-based BCIs compatible with
            Medusa platform [LINK]
    """

    def __init__(self):
        """Class constructor
        """
        super().__init__(fit_dataset=['mi_target', 'mi_result', 'accuracy'],
                         predict=['mi_result'])
        # Settings
        self.settings = None
        self.channel_set = None
        self.configure()
        # Configuration
        self.is_configured = False
        self.is_built = False
        self.is_fit = False

    @abstractmethod
    def configure(self, **kwargs):
        """This function must be used to configure the model before calling
        build method. Class attribute settings attribute must be set with a dict
        """
        # Update state
        self.is_configured = True
        self.is_built = False
        self.is_fit = False

    @abstractmethod
    def build(self, *args, **kwargs):
        """This function builds the model, adding all the processing methods
        to the pipeline. It must be called after configure.
        """
        # Check errors
        if not self.is_configured:
            raise ValueError('Function configure must be called first!')
        # Update state
        self.is_built = True
        self.is_fit = False

    def fit_dataset(self, dataset, **kwargs):
        """Function that receives an MIDataset and uses its data to
        fit the model. By default, executes pipeline 'fit_dataset'. Override
        method for other behaviour.

        Parameters
        ----------
        dataset: MIDataset
            Dataset with recordings from an MI-based BCI experiment
        kwargs: key-value arguments
            Optional parameters depending on the specific implementation of
            the model

        Returns
        -------
        fit_results: dict
            Dict with the information of the fit process. For command
            decoding models, at least it has to contain keys
            mi_target, mi_result and accuracy, which contain the target MI,
            the decoded MI and the decoding accuracy in the analysis.
        """
        # Check errors
        if not self.is_built:
            raise ValueError('Function build must be called first!')
        # Execute pipeline
        output = self.exec_pipeline('fit_dataset', dataset=dataset)
        # Set channels
        self.channel_set = dataset.channel_set
        # Update state
        self.is_fit = True
        return output

    def predict(self, times, signal, fs, l_cha, x_info, **kwargs):
        """Function that receives EEG signal and experiment info from an
        MI-based trial to decode the user's intentions. Used in online
        experiments. By default, executes pipeline 'predict'. Override method
        for other behaviour.

        Parameters
        ---------
        times: list or numpy.ndarray
            Timestamps of the EEG samples
        signal: list or numpy.ndarray
            EEG samples with shape [n_samples x n_channels]
        fs: float
            Sample rate of the EEG signal
        l_cha: list
            List of channel labels
        x_info: dict
            Dict with the needed experiment info to decode the commands. It
            has to contain keys: mode, onsets, w_trial_t.
            See MIData to know how are defined these variables.
        kwargs: key-value arguments
            Optional parameters depending on the specific implementation of
            the model
        """
        # Check errors
        if not self.is_fit:
            raise ValueError('Function fit_dataset must be called first!')
        # Check channels
        self.channel_set.check_channels_labels(l_cha, strict=True)
        # Execute predict pipeline
        return self.exec_pipeline('predict', times=times, signal=signal,
                                  fs=fs, x_info=x_info, **kwargs)


class MIModelCSP(MIModel):
    """Decoding model for MI-based BCI applications based on
    Common Spatial Patterns (CSP).

    It is strongly recommended to update the default settings by passing
    arguments through **kwargs. See the different algorithms to know more
    about the possible parameters: StandardPreprocessing,
    CSPFeatureExtraction and StandardFeatureExtraction.

    Dataset features:
    - Sample rate of the signals > 60 Hz. The model can handle recordings
        with different sample rates.
    - Recommended channels to be present: ['C3', 'C4'].

    Processing pipeline:
    - Preprocessing (medusa.bci.erp_spellers.StandardPreprocessing):
        - IIR Filter (order=5, cutoff=(8, 30) Hz: unlike FIR filters, IIR
            filters are quick and can be applied in small signal chunks. Thus,
            they are the preferred method for frequency filter in online
            systems.
        - Common average reference (CAR): widely used spatial filter that
            increases the signal-to-noise ratio of the MI control signals.
    - Feature extraction (medusa.bci.mi_paradigms.CSPFeatureExtraction):
        - Epochs (window=(0, 2000) ms, resampling to 60 HZ): the epochs of
            signal are extracted for each stimulation. Baseline normalization
            is also applied, taking the window (-1000, 0) ms relative to the
            stimulus onset.
        - CSP projection: Epochs are then projected according to a CSP filter
            previously trained.
        - Log variance: Log variance features are extracted from the CSP
            projection.
    - Feature classification (
    sklearn.discriminant_analysis.LinearDiscriminantAnalysis)
        - Regularized linear discriminant analysis (rLDA): we use the sklearn
            implementation, with eigen solver and auto shrinkage paramers. See
            reference in sklearn doc.
    """

    def __init__(self):
        super().__init__()

    def configure(self, p_filt_cutoff=(8, 30), w_epoch_t=(0, 2000),
                  w_baseline_t=(-1000, 0), target_fs=60, **kwargs):
        """ Configures the default settings. """
        self.settings = {
            # StandardPreprocessing
            'p_filt_cutoff': p_filt_cutoff,
            # CSPFeatureExtraction
            'n_filters': 4,
            'normalize_log_vars': False,
            # StandardFeatureExtraction
            'w_epoch_t': w_epoch_t,
            'baseline_mode': 'trial',
            'w_baseline_t': w_baseline_t,
            'norm': 'z',
            'target_fs': target_fs,
            'concatenate_channels': False,
            'sliding_w_lims_t': None,
            'sliding_t_step': None,
            'sliding_win_len': None
        }
        self.settings = dict(self.settings, **kwargs)
        # Update state
        self.is_configured = True
        self.is_built = False
        self.is_fit = False

    def build(self):
        """ Initializes the different methods that comprise the MIModelCSP
        pipeline.
        """
        # Check errors
        if not self.is_configured:
            raise ValueError('Function configure must be called first!')
        # Preprocessing (default: bandpass IIR filter [8, 30] Hz + CAR)
        self.add_method('prep_method', StandardPreprocessing(
            cutoff=self.settings['p_filt_cutoff']
        ))
        # Feature extraction (default: epochs [500, 4000] ms + resampling to 80
        # Hz)
        self.add_method('ext_method', CSPFeatureExtraction(**self.settings))
        # Feature classification (rLDA)
        clf = components.ProcessingClassWrapper(
            LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto'),
            fit=[], predict_proba=['y_pred']
        )
        self.add_method('clf_method', clf)
        # Update state
        self.is_built = True
        self.is_fit = False

    def fit_dataset(self, dataset, get_training_accuracy=True,
                    k_fold=5, **kwargs):
        """ Function to fit a dataset using MIModelCSP.

        Parameters
        -------------
        dataset : MIDataset
            MI dataset used for training.
        get_training_accuracy : bool
            Whether to perform an estimation on training accuracy after fitting.
        k_fold : int
            Number of k-folds used in the k-fold cross-validation procedure
            to estimate the training accuracy.
        **kwargs : dict
            These parameters will be overwritten over self.settings.

        Returns
        -------------
        assessment : dict
            Dictionary containing the details of the training accuracy
            estimation.
        """
        # Check errors
        if not self.is_built:
            raise ValueError('Function build must be called first!')

        # Merge settings
        settings = dict(self.settings, **kwargs)
        self.channel_set = dataset.channel_set

        # Preprocessing
        dataset = self.get_inst('prep_method').fit_transform_dataset(dataset)

        # Extract CSP features
        x, x_info = self.get_inst('ext_method').fit_transform_dataset(
            dataset, **settings)

        # Training accuracy using a k-fold cross-validation
        assessment = None
        if get_training_accuracy:
            folds_decoding = list()
            k_fold_iter = cu.k_fold_split(x, x_info['mi_labels'], k_fold)
            k_fold_acc = 0
            for iter in k_fold_iter:
                clf = copy.deepcopy(self.get_inst('clf_method'))
                clf.fit(iter["x_train"], iter["y_train"])
                y_test_pred = clf.predict(iter["x_test"])
                y_test_prob = clf.predict_proba(iter["x_test"])
                fold_acc = np.sum(y_test_pred == iter["y_test"]) / \
                           len(iter["y_test"])
                k_fold_acc += fold_acc
                folds_decoding.append({
                    "y_pred": y_test_pred,
                    "y_prob": y_test_prob,
                    "accuracy": fold_acc,
                })
            k_fold_acc /= len(k_fold_iter)

            assessment = {
                'x': x,
                'x_info': x_info,
                'k-fold': folds_decoding,
                'accuracy': k_fold_acc
            }

        # Fit classifier with all the data
        self.get_inst('clf_method').fit(x, x_info['mi_labels'])

        # Save info
        self.channel_set = dataset.channel_set

        # Update state
        self.is_fit = True
        return assessment

    def predict(self, times, signal, fs, channel_set, x_info, **kwargs):
        """ Function to predict an individual signal in MIModelCSP.

        Parameters
        --------------
        times : ndarray (n_samples,)
            Timestamp array
        signal: ndarray (n_samples x n_channels)
            Signal data.
        fs : int
            Sampling frequency.
        channel_set : EEGChannelSet or similar
            Channel montage.
        x_info : dict
            Dictionary containing the trial "onsets" and "mi_labels". If the
            latter is not specified, accuracy is not calculated.
        **kwargs : dict
            These parameters will be overwritten over self.settings.

        Returns
        -------------
        decoding : dict
            Dictionary containing the decoding.
        """
        # Check errors
        if not self.is_fit:
            raise ValueError('Function fit_dataset must be called first!')
        # Merge settings
        settings = dict(self.settings, **kwargs)
        # Check channel set
        if self.channel_set.l_cha != channel_set.l_cha:
            warnings.warn('The channel set is not the same that was used to '
                          'fit the model. Be careful!')
        # Preprocessing
        signal = self.get_inst('prep_method').transform_signal(signal)

        # Extract features
        x = self.get_inst('ext_method').transform_signal(
            times, signal, fs, x_info['onsets'], **settings)

        # Classification
        y_prob = self.get_inst('clf_method').predict_proba(x)
        y_pred = self.get_inst('clf_method').predict(x)

        # Decoding
        accuracy = None
        clf_report = None
        if x_info['mi_labels'] is not None:
            accuracy = np.sum((y_pred == x_info['mi_labels'])) / len(y_pred)
            clf_report = classification_report(x_info['mi_labels'], y_pred,
                                               output_dict=True)
        decoding = {
            'x': x,
            'x_info': x_info,
            'y_prob': y_prob,
            'y_pred': y_pred,
            'accuracy': accuracy,
            'report': clf_report
        }
        return decoding

    def predict_dataset(self, dataset, **kwargs):
        """ Function to predict a dataset using MIModelCSP.

        Parameters
        -------------
        dataset : MIDataset
            Test dataset.
        **kwargs : dict
            These parameters will be overwritten over self.settings.

        Returns
        -------------
        decoding : dict
            Dictionary containing the decoding.
        """
        # Check errors
        if not self.is_fit:
            raise ValueError('Function fit_dataset must be called first!')
        # Check channel set
        if self.channel_set.l_cha != dataset.channel_set.l_cha:
            warnings.warn('The channel set is not the same that was used to '
                          'fit the model. Be careful!')
        # Merge settings
        settings = dict(self.settings, **kwargs)

        # Preprocessing
        dataset = self.get_inst('prep_method').transform_dataset(dataset,
                                                                 **settings)

        # Extract features
        x, x_info = self.get_inst('ext_method').transform_dataset(
            dataset, **settings)

        # Classification
        y_prob = self.get_inst('clf_method').predict_proba(x)
        y_pred = self.get_inst('clf_method').predict(x)

        # Decoding
        accuracy = None
        clf_report = None
        if x_info['mi_labels'] is not None:
            accuracy = np.sum((y_pred == x_info['mi_labels'])) / len(y_pred)
            clf_report = classification_report(x_info['mi_labels'], y_pred,
                                               output_dict=True)
        decoding = {
            'x': x,
            'x_info': x_info,
            'y_prob': y_prob,
            'y_pred': y_pred,
            'accuracy': accuracy,
            'report': clf_report
        }
        return decoding


class MIModelEEGSym(MIModel):
    """Decoding model for MI-based BCI applications based on EEGSym [1], a deep
    convolutional neural network developed for inter-subjects MI classification.

    Dataset features:

    - Sample rate of the signals > 128 Hz. The model can handle recordings
        with different sample rates.
    - Recommended channels: ['F7', 'C3', 'Po3', 'Cz', 'Pz', 'F8', 'C4', 'Po4'].

    Processing pipeline:

    - Preprocessing:

        - IIR Filter (order=4, lowpass=49 Hz: unlike FIR filters, IIR filters
            are quick and can be applied in small signal chunks. Thus,
            they are the preferred method for frequency filter in online systems
        - Common average reference (CAR): widely used spatial filter that
            increases the signal-to-noise ratio.
    - Feature extraction:

        - Epochs (window=(0, 2000) ms, resampling to 128 HZ): the epochs of
            signal are extracted after each onset. Baseline normalization
            is also applied, taking the same epoch window.

    - Feature classification

        - EEGSym: convolutional neural network [1].

    References
    ----------
    [1] Pérez-Velasco, S., Santamaría-Vázquez, E., Martínez-Cagigal, V.,
    Marcos-Mateo, D., & Hornero, R. (2020). EEGSym: Overcoming Intersubject
    Variability in Motor Imagery Based BCIs with Deep Learning. ?.
    """
    def __init__(self):
        super().__init__()

    def configure(self, p_filt_cutoff=(0.1, 45), w_epoch_t=(0, 2000),
                  w_baseline_t=(0, 2000), target_fs=128, cnn_n_cha=8,
                  ch_lateral=3, fine_tuning=False, validation_split=0.4,
                  init_weights_path=None, gpu_acceleration=False,
                  augmentation=False, **kwargs):
        self.settings = {
            # StandardPreprocessing
            'p_filt_cutoff': p_filt_cutoff,
            # StandardFeatureExtraction
            'w_epoch_t': w_epoch_t,
            'baseline_mode': 'sliding',
            'w_baseline_t': w_baseline_t,
            'norm': 'z',
            'target_fs': target_fs,
            'concatenate_channels': False,
            'sliding_w_lims_t': None,
            'sliding_t_step': None,
            'sliding_win_len': None,
            # EEGSym features
            'cnn_n_cha': cnn_n_cha,
            'ch_lateral': ch_lateral,
            'fine_tuning': fine_tuning,
            'augmentation': augmentation,
            'validation_split': validation_split,
            'init_weights_path': init_weights_path,
            'gpu_acceleration': gpu_acceleration
        }
        self.settings = dict(self.settings, **kwargs)
        # Update state
        self.is_configured = True
        self.is_built = False
        self.is_fit = False

    def build(self):
        """ Initializes the different methods that comprise the MIModelEEGSym
        pipeline.
        """
        # Check errors
        if not self.is_configured:
            raise ValueError('Function configure must be called first!')
        # Only import deep learning models if necessary
        from medusa.deep_learning_models import EEGSym
        # Preprocessing (default: bandpass IIR filter [0.5, 45] Hz + CAR)
        self.add_method('prep_method',
                        StandardPreprocessing(cutoff=self.settings['p_filt_cutoff']))
        # Feature extraction (epochs [0, 2000] ms + resampling to 128 Hz)
        self.add_method('ext_method', StandardFeatureExtraction(**self.settings))
        # Feature classification
        clf = EEGSym(
            input_time=int(self.settings['w_epoch_t'][1] -
                        self.settings['w_epoch_t'][0]),
            fs=self.settings['target_fs'],
            n_cha=self.settings['cnn_n_cha'],
            ch_lateral=self.settings['ch_lateral'],
            filters_per_branch=24,
            scales_time=(125, 250, 500),
            dropout_rate=0.4,
            activation='elu', n_classes=2,
            learning_rate=0.001,
            gpu_acceleration=self.settings['gpu_acceleration'])
        self.is_fit = False
        if self.settings['init_weights_path'] is not None:
            clf.load_weights(self.settings['init_weights_path'])
            self.channel_set = meeg.EEGChannelSet()
            standard_lcha = ['F7', 'C3', 'PO3', 'CZ', 'PZ', 'F8', 'C4', 'PO4']
            self.channel_set.set_standard_montage(standard_lcha)
            self.get_inst('prep_method').fit(fs=250, n_cha=8)
            self.is_fit = True
        else:
            self.is_fit = False
        self.add_method('clf_method', clf)
        # Update state
        self.is_built = True

    def fit_dataset(self, dataset, **kwargs):
        """ Function to fit a dataset using MIModelCSP.

        Parameters
        -------------
        dataset : MIDataset
            MI dataset used for training.
        **kwargs : dict
            These parameters will be overwritten over self.settings.

        Returns
        -------------
        assessment : dict
            Dictionary containing the details of the training accuracy
            estimation.
        """
        # Check errors
        if not self.is_built:
            raise ValueError('Function build must be called first!')
        # Merge settings
        settings = dict(self.settings, **kwargs)
        # Preprocessing
        dataset = self.get_inst('prep_method').fit_transform_dataset(dataset)
        # Extract features
        x, x_info = self.get_inst('ext_method').transform_dataset(dataset,
                                                                  **settings)
        # Put channels in symmetric order
        x, _ = self.get_inst('clf_method').symmetric_channels(
            x, dataset.channel_set.l_cha)

        # Classification
        self.get_inst('clf_method').fit(
            x, x_info['mi_labels'],
            fine_tuning=self.settings['fine_tuning'],
            validation_split=self.settings['validation_split'],
            augmentation=self.settings['augmentation'],
            **kwargs)

        y_prob = self.get_inst('clf_method').predict_proba(x)
        y_pred = self.get_inst('clf_method').predict(x)

        # Accuracy
        accuracy = np.sum((y_pred == x_info['mi_labels'])) / len(y_pred)
        clf_report = classification_report(x_info['mi_labels'], y_pred,
                                           output_dict=True)
        assessment = {
            'x': x,
            'x_info': x_info,
            'y_pred': y_pred,
            'y_prob': y_prob,
            'accuracy': accuracy,
            'report': clf_report
        }
        # Save info
        self.channel_set = dataset.channel_set
        # Update state
        self.is_fit = True
        return assessment

    def predict(self, times, signal, fs, channel_set, x_info, **kwargs):
        """ Function to predict an individual signal in MIModelCSP.

        Parameters
        --------------
        times : ndarray (n_samples,)
            Timestamp array
        signal: ndarray (n_samples x n_channels)
            Signal data.
        fs : int
            Sampling frequency.
        channel_set : EEGChannelSet or similar
            Channel montage.
        x_info : dict
            Dictionary containing the trial "onsets" and "mi_labels". If the
            latter is not specified, accuracy is not calculated.
        **kwargs : dict
            These parameters will be overwritten over self.settings.

        Returns
        -------------
        decoding : dict
            Dictionary containing the decoding.
        """
        # Check errors
        if not self.is_fit:
            raise ValueError('Function fit_dataset must be called first!')
        # Merge settings
        settings = dict(self.settings, **kwargs)
        # Check channel set
        if self.channel_set.l_cha != channel_set.l_cha:
            warnings.warn('The channel set is not the same that was used to '
                          'fit the model. Be careful!')
        # Preprocessing
        signal = self.get_inst('prep_method').transform_signal(signal)

        # Extract features
        x = self.get_inst('ext_method').transform_signal(
            times=times, signal=signal, fs=fs, onsets=x_info['onsets'],
            **settings)

        # Put channels in symmetric order
        x, _ = self.get_inst('clf_method').symmetric_channels(
            x, channel_set.l_cha)

        # Classification
        y_prob = self.get_inst('clf_method').predict_proba(x)
        y_pred = y_prob.argmax(axis=-1)

        # Decoding
        accuracy = None
        clf_report = None
        if x_info['mi_labels'] is not None:
            accuracy = np.sum((y_pred == x_info['mi_labels'])) / len(y_pred)
            clf_report = classification_report(x_info['mi_labels'], y_pred,
                                               output_dict=True)
        decoding = {
            'x': x,
            'x_info': x_info,
            'y_pred': y_pred,
            'y_prob': y_prob,
            'accuracy': accuracy,
            'report': clf_report
        }
        return decoding

    def predict_dataset(self, dataset, **kwargs):
        """ Function to predict a dataset using MIModelCSP.

        Parameters
        -------------
        dataset : MIDataset
            Test dataset.
        **kwargs : dict
            These parameters will be overwritten over self.settings.

        Returns
        -------------
        decoding : dict
            Dictionary containing the decoding.
        """
        # Check errors
        if not self.is_fit:
            raise ValueError('Function fit_dataset must be called first!')
        # Check channel set
        if self.channel_set.l_cha != dataset.channel_set.l_cha:
            warnings.warn('The channel set is not the same that was used to '
                          'fit the model. Be careful!')
        # Merge settings
        settings = dict(self.settings, **kwargs)

        # Preprocessing
        dataset = self.get_inst('prep_method').transform_dataset(dataset,
                                                                 **settings)

        # Extract features
        x, x_info = self.get_inst('ext_method').transform_dataset(
            dataset, **settings)

        # Put channels in symmetric order
        x, _ = self.get_inst('clf_method').symmetric_channels(x,
                                                              dataset.channel_set.l_cha)

        # Classification
        y_prob = self.get_inst('clf_method').predict_proba(x)
        y_pred = y_prob.argmax(axis=-1)

        # Decoding
        accuracy = None
        clf_report = None
        if x_info['mi_labels'] is not None:
            accuracy = np.sum((y_pred == x_info['mi_labels'])) / len(y_pred)
            clf_report = classification_report(x_info['mi_labels'], y_pred,
                                               output_dict=True)
        decoding = {
            'x': x,
            'x_info': x_info,
            'y_prob': y_prob,
            'y_pred': y_pred,
            'accuracy': accuracy,
            'report': clf_report
        }
        return decoding
