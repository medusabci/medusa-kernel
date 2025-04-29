from abc import ABC, abstractmethod
import math, copy, warnings
import numpy as np
from matplotlib import pyplot as plt
import medusa as mds
from medusa import meeg
from medusa import components
from tqdm import tqdm


# --------------------------- SSVEP DATA MANAGEMENT -------------------------- #
class SSVEPSpellerData(components.ExperimentData):
    """Experiment info class for SSVEP-based spellers. It supports nested
    multi-matrix multi-level paradigms. This unified class can be used to
    represent a run of every SSVEP stimulation paradigm designed to date,
    and is the expected class for feature extraction and command decoding
    functions of the module medusa.bci.ssvep_spellers. It is complicated,
    but powerful so.. use it well!
    """

    def __init__(self, mode, paradigm_conf, commands_info, onsets,
                 unit_idx, level_idx, matrix_idx, trial_idx,
                 cmd_model, csd_model, spell_result, control_state_result,
                 fps_resolution, stim_time, stim_freq_range, spell_target=None,
                 control_state_target=None, **kwargs):

        # Check errors
        mode = mode.lower()
        if mode not in ('train', 'test'):
            raise ValueError('Unknown mode. Possible values {train, test}')

        # Standard attributes
        self.mode = mode
        self.paradigm_conf = paradigm_conf
        self.commands_info = commands_info
        self.onsets = onsets
        self.unit_idx = unit_idx
        self.level_idx = level_idx
        self.matrix_idx = matrix_idx
        self.trial_idx = trial_idx
        self.cmd_model = cmd_model
        self.csd_model = csd_model
        self.spell_result = spell_result
        self.control_state_result = control_state_result
        self.fps_resolution = fps_resolution
        self.stim_time = stim_time
        self.stim_freq_range = stim_freq_range
        self.spell_target = spell_target
        self.control_state_target = control_state_target

        # Optional attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

    @staticmethod
    def get_paradigm_conf_for_freq_enc(matrix_dims, commands_info=None):
        """Simple frequency encoding paradigm with no levels

        Example of a frequency encoding paradigm with 4 commands:

                paradigm_conf = [
                    # Matrices
                    [
                        # Units
                        [
                            # Groups
                            [
                                # Batches
                                [0, 1, 2, 3, 4]
                            ]
                        ]
                    ]
                ]
        """
        matrix_dims = np.array(matrix_dims)
        paradigm_conf = list()
        commands_info_list = list()

        for m in range(matrix_dims.shape[0]):
            # Commands matrix
            n_rows = matrix_dims[m, 0]
            n_cols = matrix_dims[m, 1]
            commands_ids = np.arange(n_rows * n_cols).tolist()
            # Paradigm conf. Groups and batches are not necessary for SSVEP
            # spellers, only matrices and units for multilevel paradigms.
            paradigm_conf.append(list())                    # Matrix
            paradigm_conf[m].append(list())                 # Unit
            paradigm_conf[m][0].append(commands_ids)        # Group
            # Commands info
            if commands_info is None:
                cmd_info_values = [dict() for i in commands_ids]
            else:
                cmd_info_values = np.array(commands_info[m]).flatten()
                cmd_info_values = cmd_info_values.tolist()
            commands_info_list.append(dict(zip(commands_ids, cmd_info_values)))

        return paradigm_conf, commands_info_list

    def to_serializable_obj(self):
        rec_dict = self.__dict__
        for key in rec_dict.keys():
            if type(rec_dict[key]) == np.ndarray:
                rec_dict[key] = rec_dict[key].tolist()
        return rec_dict

    @classmethod
    def from_serializable_obj(cls, dict_data):
        return cls(**dict_data)


class SSVEPSpellerDataset(components.Dataset):

    """This class inherits from medusa.data_structures.Dataset, increasing
    its functionality for datasets with data from ERP-based spellers. It
    provides common ground for the rest of functions in the module.
    """

    def __init__(self, channel_set, fs=None, stim_time=None,
                 biosignal_att_key='eeg', experiment_att_key='ssvepspellerdata',
                 experiment_mode=None, track_attributes=None):
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
            recordings (e.g., 'rcp_data', 'cake_paradigm_data'). It is
            mandatory when a recording of the dataset contains more than 1
            experiment data
        experiment_mode : str {'train'|'test'|None}
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
            if experiment_mode not in ('train', 'test'):
                raise ValueError('Parameter experiment_mode must be '
                                 '{train|test|None}')

        # Default track attributes
        default_track_attributes = {
            'subject_id': {
                'track_mode': 'append',
                'parent': None
            },
            'paradigm_conf': {
                'track_mode': 'append',
                'parent': experiment_att_key
            },
            'commands_info': {
                'track_mode': 'append',
                'parent': experiment_att_key
            },
            'onsets': {
                'track_mode': 'concatenate',
                'parent': experiment_att_key
            },
            'level_idx': {
                'track_mode': 'concatenate',
                'parent': experiment_att_key
            },
            'matrix_idx': {
                'track_mode': 'concatenate',
                'parent': experiment_att_key
            },
            'trial_idx': {
                'track_mode': 'concatenate',
                'parent': experiment_att_key
            },
            'spell_result': {
                'track_mode': 'append',
                'parent': experiment_att_key
            },
            'control_state_result': {
                'track_mode': 'append',
                'parent': experiment_att_key
            }
        }

        if experiment_mode == 'train':
            default_track_attributes_train = {
                'spell_target': {
                    'track_mode': 'append',
                    'parent': experiment_att_key
                },
                'control_state_target': {
                    'track_mode': 'append',
                    'parent': experiment_att_key
                }
            }
            default_track_attributes = {
                **default_track_attributes,
                **default_track_attributes_train
            }

        track_attributes = \
            default_track_attributes if track_attributes is None else \
            {**default_track_attributes, **track_attributes}

        # Class attributes
        self.channel_set = channel_set
        self.fs = fs
        self.stim_time = stim_time
        self.biosignal_att_key = biosignal_att_key
        self.experiment_att_key = experiment_att_key
        self.experiment_mode = experiment_mode
        self.track_attributes = track_attributes

        # Consistency checker
        checker = self.__get_consistency_checker()
        super().__init__(consistency_checker=checker)

    def __get_consistency_checker(self):
        """Creates a standard consistency checker for ERP datasets

        Returns
        -------
        checker : data_structures.ConsistencyChecker
            Standard consistency checker for ERP feature extraction
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
            parent=self.biosignal_att_key+'.channel_set'
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
        # Check stim time
        if self.stim_time is not None:
            checker.add_consistency_rule(rule='check-attribute-value',
                                         rule_params={'attribute': 'stim_time',
                                                      'value': self.stim_time},
                                         parent=self.experiment_att_key)
        else:
            warnings.warn('Parameter stim_time is None. The consistency of the '
                          'dataset cannot be assured.')

        # Check experiment
        checker.add_consistency_rule(
            rule='check-attribute',
            rule_params={'attribute': self.experiment_att_key}
        )
        checker.add_consistency_rule(
            rule='check-attribute-type',
            rule_params={'attribute': self.experiment_att_key,
                         'type': SSVEPSpellerData}
        )
        # Check mode
        if self.experiment_mode is not None:
            checker.add_consistency_rule(
                rule='check-attribute-value',
                rule_params={'attribute': 'mode',
                             'value': self.experiment_mode},
                parent=self.experiment_att_key
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
                                     'type': [list, np.ndarray]},
                        parent=value['parent']
                    )

        return checker

    def custom_operations_on_recordings(self, recording):
        # Select channels
        eeg = getattr(recording, self.biosignal_att_key)
        eeg.change_channel_set(self.channel_set)
        return recording


def detect_control_state(scores, run_idx, trial_idx):
    """Detects the user's control state for each trial, assigning 0 to
    non-control and 1 to control states.

    Parameters
    ----------
    scores : list or np.ndarray
        Array with the score per stimulation.
    run_idx : list or numpy.ndarray [n_stim x 1]
        Index of the run for each stimulation.
    trial_idx : list or numpy.ndarray [n_stim x 1]
        Index of the trial for each stimulation. A trial represents
        the selection of a final command. Depending on the number of levels,
        the final selection takes N intermediate selections.

    Returns
    -------
    selected_control_state: list
        Selected control state for each trial considering all sequences of
        stimulation. Shape [n_runs x n_trials]
    selected_control_state_per_seq: list
        Selected command for each trial and sequence of stimulation. The
        fourth dimension of the array contains [matrix_idx, command_id]. To
        calculate the command for each sequence, it takes into account the
        scores of all the previous sequences as well. Shape [n_runs x
        n_trials x n_sequences]
    scores: list
        Scores for each command per sequence. Shape [n_runs x n_trials x
        n_levels x n_sequences]. The score of each sequence is calculated
        independently of the previous sequences.
    """

    # Avoid errors
    scores = np.array(scores)
    run_idx = np.array(run_idx)
    trial_idx = np.array(trial_idx)

    # Check errors
    if len(scores.shape) > 1:
        if len(scores.shape) > 2 or scores.shape[-1] != 1:
            raise ValueError('Parameter scores must have shape '
                             '(n_stim,) or (n_stim, 1)')
    n_stim = scores.shape[0]
    if run_idx.shape[0] != n_stim or trial_idx.shape[0] != n_stim:
        raise ValueError('Shape mismatch. Parameters scores, run_idx, '
                         'trial_idx and sequence_idx must have a the same '
                         'dimensions')

    # Control state detection
    state_scores = list()
    selected_control_state = list()
    selected_control_state_per_seq = list()
    idx = np.arange(trial_idx.shape[0])

    # Get selected control state
    for r_cnt, r in enumerate(np.unique(run_idx)):
        idx_r = idx[np.where(run_idx == r)]
        # Increment dimensionality
        state_scores.append(list())
        selected_control_state.append(list())
        selected_control_state_per_seq.append(list())
        for t_cnt, t in enumerate(np.unique(trial_idx[idx_r])):
            idx_t = idx_r[np.where(trial_idx[idx_r] == t)]
            state_scores[r_cnt].append(list())
            selected_control_state_per_seq[r_cnt].append(list())
            # Append selected csd (avoid reference problems with copy)
            # TODO: select an actual control state
            selected_control_state[r_cnt].append(
                np.random.choice([0, 1])
            )

    return selected_control_state, state_scores


def get_selected_commands_info(selected_commands, commands_info):
    """Returns the info dict of the selected commands

    Parameters
    ----------
    selected_commands : list
        Selected command for each trial. Each command is organized in an array
        [matrix_idx, command_id]. Take into account that the command ids are
        unique for each matrix, and therefore only the command of the last
        level should be useful to take action. Shape [n_runs x n_trials x
        n_levels x 2]
    commands_info : list
        List containing the command information per run and matrix. Each
        position must be a dict, whose keys are the commands ids used in
        paradigm_conf. The value must be another dict containing important
        information about each command (e.g., label, text, action, icon
        path, etc). This information may be different for different use
        cases, but must be serializable (i.e., contain primitive types).
        Shape [n_runs x n_matrices x n_commands]

    Returns
    -------
    selected_commands_info : list
        List containing the information dict of the selected commands
    """
    try:
        # Print info
        selected_commands_info = list()
        for r in range(len(selected_commands)):
            for t in range(len(selected_commands[r])):
                [m_d, cmd_d] = selected_commands[r][t][-1]
                selected_commands_info.append(commands_info[m_d][cmd_d])
    except Exception as e:
        raise type(e)(str(e) + '\nCheck that selected_commands has shape '
                               '[n_runs x n_trials x n_levels x 2]')

    return selected_commands_info


class StandardPreprocessing(components.ProcessingMethod):
    """Just the common preprocessing applied in SSVEP-based spellers. Simple,
    quick and effective: frequency IIR filter followed by common average
    reference (CAR) spatial filter.
    """
    def __init__(self, freq_filt={'order':5, 'cutoff':1, 'btype': 'highpass'},
                 notch_filt={'order':5, 'cutoff':(49,51), 'btype':'bandstop'},
                 filt_method='sosfiltfilt'):
        super().__init__(fit_transform_signal=['signal'],
                         fit_transform_dataset=['dataset'])
        # Parameters
        self.freq_filt_params = freq_filt
        self.notch_filt_params = notch_filt
        self.filt_method = filt_method

        # Variables that
        self.filters = list()

    def fit(self, fs):
        """Fits the IIR filter.

        Parameters
        ----------
        fs: float
            Sample rate of the signal.
        """
        if self.freq_filt_params is not None:
            filt = mds.IIRFilter(**self.freq_filt_params,
                                 filt_method=self.filt_method)
            filt.fit(fs)
            self.filters.append(filt)
        if self.notch_filt_params is not None:
            filt = mds.IIRFilter(**self.notch_filt_params,
                                 filt_method=self.filt_method)
            filt.fit(fs)
            self.filters.append(filt)

    def transform_signal(self, signal):
        """Transforms an EEG signal applying IIR filtering and CAR sequentially

        Parameters
        ----------
        signal: np.array or list
            Signal to transform. Shape [n_samples x n_channels]
        """
        for filt in self.filters:
            signal = filt.transform(signal)
        # signal = mds.car(signal)
        return signal

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
        self.fit(fs)
        signal = self.transform_signal(signal)
        return signal

    def fit_transform_dataset(self, dataset, show_progress_bar=True):
        """Fits the IIR filter and transforms an EEG signal applying the
        filter and CAR sequentially. Each recording is preprocessed
        independently, taking into account possible differences in sample rate.

        Parameters
        ----------
        dataset: ERPSpellerDataset
            ERPSpellerDataset with the recordings to be preprocessed.
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
    """Standard feature extraction method for ERP-based spellers. Basically,
    it gets the raw epoch for each stimulation event.
    """
    def __init__(self, norm='z', safe_copy=True):
        """Class constructor

        norm : str {'z'|'dc'}
            Type of baseline normalization. Set to 'z' for Z-score normalization
            or 'dc' for DC normalization
        safe_copy : bool
            Makes a safe copy of the signal to avoid changing the original
            samples due to references
        """
        super().__init__(transform_signal=['x'],
                         transform_dataset=['x', 'x_info'])
        self.norm = norm
        self.safe_copy = safe_copy

    def transform_signal(self, times, signal, fs, onsets, stim_time):
        """Function to extract ERP features from raw signal. It returns a 3D
        feature array with shape [n_events x n_samples x n_channels]. This
        function does not track any other attributes. Use for online processing
        and custom higher level functions.

        Parameters
        ----------
       times : list or numpy.ndarray
                1D numpy array [n_samples]. Timestamps of each sample. If they
                are not available, generate them artificially. Nevertheless,
                all signals and events must have the same temporal origin
        signal : list or numpy.ndarray
            2D numpy array [n_samples x n_channels]. EEG samples (the units
            should be defined using kwargs)
        fs : int or float
            Sample rate of the recording.
        onsets : list or numpy.ndarray [n_events x 1]
                Timestamp of each event


        Returns
        -------
        features : np.ndarray [n_events x n_samples x n_channels]
            Feature array with the epochs of signal
        """
        # Avoid changes in the original signal (this may not be necessary)
        if self.safe_copy:
            signal = signal.copy()
        # Extract features
        features = mds.get_epochs_of_events(timestamps=times, signal=signal,
                                            onsets=onsets, fs=fs,
                                            w_epoch_t=[0, stim_time*1000],
                                            w_baseline_t=[0, stim_time*1000],
                                            norm=self.norm)
        return features

    def transform_dataset(self, dataset, show_progress_bar=True):
        """High level function to easily extract features from EEG recordings
        and save useful info for later processing. Nevertheless, the provided
        functionality has several limitations and it will not be suitable for
        all cases and processing pipelines. If it does not fit your needs,
        create a custom function iterating the recordings and using
        extract_erp_features, a much more low-level and general function. This
        function does not apply any preprocessing to the signals, this must
        be done before

        Parameters
        ----------
        dataset: ERPSpellerDataset
            List of data_structures.Recordings or data_structures.Dataset. If this
            parameter is a list of recordings, the consistency of the dataset will
            be checked. Otherwise, if the parameter is a dataset, this function
            assumes that the consistency is already checked
        show_progress_bar: bool
            Show progress bar

        Returns
        -------
        features : numpy.ndarray
            Array with the biosignal samples arranged in epochs
        track_info : dict
            Dictionary with tracked information across all recordings

        """
        # Avoid changes in the original recordings (this may not be necessary)
        if self.safe_copy:
            dataset = copy.deepcopy(dataset)
        # Avoid consistency problems
        if dataset.fs is None and self.target_fs is None:
            raise ValueError('The consistency of the features is not assured '
                             'since dataset.fs and target_fs are both None. '
                             'Specify one of these parameters')

        # Additional track attributes
        track_attributes = dataset.track_attributes
        track_attributes['run_idx'] = {
            'track_mode': 'concatenate',
            'parent': dataset.experiment_att_key
        }

        # Initialization
        features = None
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
        run_counter = 0
        trial_counter = 0
        for rec in dataset.recordings:
            # Extract recording experiment and biosignal
            rec_exp = getattr(rec, dataset.experiment_att_key)
            rec_sig = getattr(rec, dataset.biosignal_att_key)

            # Get features
            rec_feat = self.transform_signal(
                times=rec_sig.times,
                signal=rec_sig.signal,
                fs=rec_sig.fs,
                onsets=rec_exp.onsets,
                stim_time=dataset.stim_time
            )
            features = np.concatenate((features, rec_feat), axis=0) \
                if features is not None else rec_feat

            # Special attributes that need tracking across runs to assure the
            # consistency of the dataset
            rec_exp.run_idx = run_counter * np.ones_like(rec_exp.trial_idx)
            rec_exp.trial_idx = trial_counter + np.array(rec_exp.trial_idx)
            rec_exp.level_idx = np.array(rec_exp.level_idx)
            rec_exp.matrix_idx = np.array(rec_exp.matrix_idx)

            # Update counters of special attributes
            run_counter += 1
            trial_counter += np.unique(rec_exp.trial_idx).shape[0]

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


class SSVEPSpellerModel(components.Algorithm):
    """Skeleton class for SSVEP-based spellers models. This class inherits from
    components.Algorithm. Therefore, it can be used to create standalone
    algorithms that can be used in compatible apps from medusa-platform
    for online experiments. See components.Algorithm to know more about this
    functionality.

    Related tutorials:

        - Overview of ssvep_spellers module [LINK]
        - Create standalone models for SSVEP-based spellers compatible with
            Medusa platform [LINK]
    """

    def __init__(self):
        """Class constructor
        """
        super().__init__(fit_dataset=['spell_target'],
                         predict=['spell_result',
                                  'spell_result_per_seq'])
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

    @staticmethod
    def check_for_non_control_data(cs_labels, cs_target, throw_warning=True):
        """Checks the data for non-control trials.

        Returns
        -------
        check: bool
            True if there is non-control trials in the data, False otherwise.
        """
        check = False
        if not np.all(np.array(cs_labels) == 1) or \
                not np.all(np.array(cs_target) == 1):
            if np.all(np.unique(cs_labels) == [0, 1]) and \
                    np.all(np.unique(cs_target) == [0, 1]):
                check = True
                if throw_warning:
                    warnings.warn('Non-control trials detected. Only control '
                                  'trials will be used to fit the ERP '
                                  'speller model.')
            else:
                raise ValueError('Incorrect format of control_state_labels '
                                 'or control_state_result. These arrays '
                                 'must be binary {0|1})')
        return check

    @staticmethod
    def get_control_data(x, x_info):
        """Selects only the control trials in the dataset. Useful to fit
        command decoding models.
        """
        # Copy variables to avoid problems with referencing
        x = x.copy()
        x_info = copy.deepcopy(x_info)

        # Useful variables
        cs_labels = np.array(x_info['control_state_labels'])
        runs_idx = np.unique(x_info['run_idx']).astype(int)

        # Get control runs indexes
        control_runs_idx = \
            np.unique(x_info['run_idx'][cs_labels == 1]).astype(int)

        # Get control observations
        x = x[cs_labels == 1]

        # Get control info
        for key, val in x_info.items():

            if len(val) == cs_labels.shape[0]:
                x_info[key] = val[cs_labels == 1]
            elif len(val) == runs_idx.shape[0]:
                x_info[key] = [val[r] for r in control_runs_idx]
            else:
                raise ValueError('Incorrect dimension of x_info[%s]' % key)

        return x, x_info

    def fit_dataset(self, dataset, **kwargs):
        """Function that receives an ERPSpellerDataset and uses its data to
        fit the model. By default, executes pipeline 'fit_dataset'. Override
        method for other behaviour.

        Parameters
        ----------
        dataset: ERPSpellerDataset
            Dataset with recordings from an ERP-based speller experiment
        kwargs: key-value arguments
            Optional parameters depending on the specific implementation of
            the model

        Returns
        -------
        fit_results: dict
            Dict with the information of the fot process. For command
            decoding models, at least it has to contain keys
            spell_result, spell_result_per_seq and spell_acc_per_seq,
            which contain the decoded commands, the decoded commands and the
            command decoding accuracy per sequences of stimulation considered in
            the analysis.
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
        ERP-based speller to decode the user's intentions. Used in online
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
            has to contain keys: paradigm_conf, onsets, batch_idx, group_idx,
            unit_idx, level_idx, matrix_idx, sequence_idx, trial_idx, run_idx.
            See ERPSpellerData to know how are defined these variables.
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


class CMDModelCCA(SSVEPSpellerModel):

    def __init__(self):
        super().__init__()

    def configure(self, p_freq_filt={'order':5, 'cutoff':4,'btype':'highpass'},
                 p_notch_filt={'order':5, 'cutoff':(49,51),
                               'btype':'bandstop'}):
        self.settings = {
            'p_freq_filt': p_freq_filt,
            'p_notch_filt': p_notch_filt,
        }
        # Update state
        self.is_configured = True
        self.is_built = False
        self.is_fit = False

    def build(self):
        # Check errors
        if not self.is_configured:
            raise ValueError('Function configure must be called first!')
        # Preprocessing
        self.add_method('prep_method', StandardPreprocessing(
            freq_filt=self.settings['p_freq_filt'],
            notch_filt=self.settings['p_notch_filt'],
        ))
        # Feature extraction
        self.add_method('ext_method', StandardFeatureExtraction(
            norm='z',
            safe_copy=True
        ))
        # Update state
        self.is_built = True
        self.is_fit = False

    def get_stim_times_to_test(self, stim_time):
        t = 1
        stim_times = list()
        while t <= stim_time:
            stim_times.append(t)
            t += 1
        return stim_times

    def predict_dataset(self, dataset, **kwargs):
        # Check errors
        if not self.is_built:
            raise ValueError('Function build must be called first!')
        # Preprocessing
        dataset = self.get_inst('prep_method').fit_transform_dataset(dataset)
        # Feat extraction
        x, x_info = \
            self.get_inst('ext_method').transform_dataset(dataset)
        stim_times = self.get_stim_times_to_test(dataset.stim_time)
        # Decode commands
        sel_cmds, sel_cmd_per_stim_time, cmd_scores = self.__decode_commands(
            x, x_info, dataset.fs, stim_times)
        # Assessment
        cmd_assessment = {
            'x': x,
            'x_info': x_info,
            'cmd_scores': cmd_scores,
            'spell_result': sel_cmds,
            'spell_result_per_seq': sel_cmd_per_stim_time,
        }
        if dataset.experiment_mode.lower() == 'train':
            # Spell accuracy
            spell_acc = command_decoding_accuracy(
                sel_cmds,
                x_info['spell_target'])
            cmd_assessment['spell_acc'] = spell_acc
            # Spell accuracy per seq
            spell_acc_per_seq = command_decoding_accuracy_per_seq(
                sel_cmd_per_stim_time,
                x_info['spell_target'])
            cmd_assessment['spell_acc_per_stim_time'] = spell_acc_per_seq

        return cmd_assessment

    def predict(self, times, signal, fs, channel_set, exp_data, **kwargs):
        # Check errors
        if not self.is_built:
            raise ValueError('Function build must be called first!')
        if not isinstance(exp_data, SSVEPSpellerData):
            raise ValueError('Parameter exp_data must be of type '
                             'SSVEPSpellerData.')
        # Special kwargs
        if 'trial_idx' in kwargs:
            exp_data_idx = np.array(exp_data.trial_idx) == kwargs['trial_idx']
        else:
            exp_data_idx = np.ones_like(exp_data.trial_idx)
        # Get x_info
        x_info = dict()
        x_info['run_idx'] = np.zeros_like(exp_data.trial_idx)[exp_data_idx]
        x_info['trial_idx'] = np.array(exp_data.trial_idx)[exp_data_idx]
        x_info['level_idx'] = np.array(exp_data.level_idx)[exp_data_idx]
        x_info['matrix_idx'] = np.array(exp_data.matrix_idx)[exp_data_idx]
        x_info['commands_info'] = [exp_data.commands_info]
        onsets = np.array(exp_data.onsets)[exp_data_idx]
        # Preprocessing
        signal = self.get_inst('prep_method').fit_transform_signal(signal, fs)
        # Feat extraction
        x = self.get_inst('ext_method').transform_signal(
            times, signal, fs, onsets, exp_data.stim_time)
        stim_times = [exp_data.stim_time]
        # Decode commands
        sel_cmds, __, cmd_scores = self.__decode_commands(
            x, x_info, fs, stim_times)
        return sel_cmds, cmd_scores

    def __decode_commands(self, x, x_info, fs, stim_times):
        # Decode commands
        idx = np.arange(x_info['trial_idx'].shape[0])
        cmd_scores = list()
        sel_cmds = list()
        sel_cmd_per_stim_time = list()
        r_cnt = 0
        for r in np.unique(x_info['run_idx']):
            idx_r = idx[np.where(x_info['run_idx'] == r)]
            # Increment dimensionality
            cmd_scores.append(list())
            sel_cmd_per_stim_time.append(list())
            sel_cmds.append(list())
            # Reset trial counter
            t_cnt = 0
            for t in np.unique(x_info['trial_idx'][idx_r]):
                idx_t = idx_r[np.where(x_info['trial_idx'][idx_r] == t)]
                cmd_scores[r_cnt].append(list())
                sel_cmd_per_stim_time[r_cnt].append(list())
                sel_cmds[r_cnt].append(list())
                l_cnt = 0
                for l in np.unique(x_info['level_idx'][idx_t]):
                    idx_l = idx_t[np.where(x_info['level_idx'][idx_t] == l)]
                    cmd_scores[r_cnt][t_cnt].append(list())
                    sel_cmd_per_stim_time[r_cnt][t_cnt].append(list())
                    for s_cnt, stim_time in enumerate(stim_times):
                        # Get trial signal
                        trial_test_l = int(stim_time * fs)
                        trial_sig = np.squeeze(x[idx_t, 0:trial_test_l, :])
                        trial_len = trial_sig.shape[0]
                        # Get trial info (todo: check paradigm conf)
                        m = int(np.squeeze(np.unique(
                            x_info['matrix_idx'][idx_t])))
                        trial_unit = 0
                        trial_cmd_info = \
                            x_info['commands_info'][r_cnt][m][trial_unit]
                        # Get correlations with reference signals
                        trial_scores = dict()
                        for k, v in trial_cmd_info.items():
                            cmd_freq = v['stim_freq']
                            n_harm = 2
                            ref_times = np.linspace(
                                0, trial_len / fs - (1 / fs),
                                trial_len)
                            # Create reference signals
                            ref_sig = list()
                            for h in range(1, n_harm+1):
                                ref_sig.append(
                                    np.sin(2*np.pi*h*cmd_freq*ref_times))
                                ref_sig.append(
                                    np.cos(2*np.pi*h*cmd_freq*ref_times))
                            ref_sig = np.column_stack(ref_sig)
                            cca = mds.CCA()
                            cca.fit(trial_sig, ref_sig)
                            r = np.abs(cca.r)
                            trial_scores[v['uid']] = r[0]
                        # Get command for this stim time
                        sel_cmd = [m, max(trial_scores, key=trial_scores.get)]
                        # Save result
                        cmd_scores[r_cnt][t_cnt][l_cnt].append(trial_scores)
                        sel_cmd_per_stim_time[r_cnt][t_cnt][l_cnt].append(
                            sel_cmd)
                    sel_cmds[r_cnt][t_cnt].append(sel_cmd)
                    l_cnt += 1
                t_cnt += 1
            r_cnt += 1

        return sel_cmds, sel_cmd_per_stim_time, cmd_scores


def command_decoding_accuracy(selected_commands, target_commands):
    """Computes the accuracy of the selected sequence of targets given the
    objective

    Parameters
    ----------
    selected_commands: list
        Target commands. Each position contains the matrix index and command
        id per level that identifies the selected command of the trial. Shape
        [n_runs x n_trials x n_levels x 2]
    target_commands: list
        Target commands. Each position contains the matrix index and command
        id per level that identifies the target command of the trial. Shape
        [n_runs x n_trials x n_levels x 2]

    Returns
    -------
    accuracy : float
        Accuracy of the command decoding stage
    """
    # Check errors
    if len(selected_commands) != len(target_commands):
        raise ValueError('Parameters selected_commands and target_commands '
                         'must have the same shape [n_runs x n_trials x '
                         'n_levels x 2]')

    t_correct_cnt = 0
    t_total_cnt = 0
    for r in range(len(selected_commands)):
        for t in range(len(selected_commands[r])):
            if selected_commands[r][t] == target_commands[r][t]:
                t_correct_cnt += 1
            t_total_cnt += 1

    accuracy = t_correct_cnt / t_total_cnt
    return accuracy


def command_decoding_accuracy_per_seq(selected_commands_per_seq,
                                      target_commands):
    """
    Computes the accuracy of the selected sequence of targets given the
    target

    Parameters
    ----------
    selected_commands_per_seq: list
        List with the spell result per sequence as given by function
        decode_commands. Shape [n_runs x n_trials x n_levels x n_sequences x 2]
    target_commands: list
        Target commands. Each position contains the matrix index and command
        id per level that identifies the target command of the trial. Shape
        [n_runs x n_trials x n_levels x 2]

    Returns
    -------
    acc_per_seq : float
        Accuracy of the command decoding stage for each number of sequences
        considered in the analysis. Shape [n_sequences]
    """
    # Check errors
    selected_commands_per_seq = list(selected_commands_per_seq)
    target_commands = list(target_commands)
    if len(selected_commands_per_seq) != len(target_commands):
        raise ValueError('Parameters selected_commands_per_seq and spell_target'
                         'must have the same length.')

    # Compute accuracy per sequence
    bool_result_per_seq = []
    n_seqs = []
    for r in range(len(selected_commands_per_seq)):
        r_sel_cmd_per_seq = selected_commands_per_seq[r]
        r_spell_target = target_commands[r]
        for t in range(len(r_sel_cmd_per_seq)):
            t_sel_cmd_per_seq = r_sel_cmd_per_seq[t]
            t_spell_target = r_spell_target[t]
            t_bool_result_per_seq = []
            t_n_seqs = []
            for l in range(len(t_sel_cmd_per_seq)):
                l_sel_cmd_per_seq = t_sel_cmd_per_seq[l]
                l_spell_target = t_spell_target[l]
                t_bool_result_per_seq.append(list())
                t_n_seqs.append(len(l_sel_cmd_per_seq))
                for s in range(len(l_sel_cmd_per_seq)):
                    s_sel_cmd_per_seq = l_sel_cmd_per_seq[s]
                    t_bool_result_per_seq[l].append(l_spell_target ==
                                                    s_sel_cmd_per_seq)

            # Calculate the trial result per seq (all levels must be correct)
            t_n_levels = len(t_sel_cmd_per_seq)
            t_max_n_seqs = np.max(t_n_seqs)
            t_acc_per_seq = np.empty((t_max_n_seqs, t_n_levels))
            t_acc_per_seq[:] = np.nan
            for t in range(t_n_levels):
                t_acc_per_seq[:t_n_seqs[t], t] = t_bool_result_per_seq[t]
            bool_result_per_seq.append(np.all(t_acc_per_seq, axis=1))
            n_seqs.append(t_max_n_seqs)

    # Calculate the accuracy per number of sequences considered in the analysis
    max_n_seqs = np.max(n_seqs)
    n_trials = len(bool_result_per_seq)
    acc_per_seq = np.empty((max_n_seqs, n_trials))
    acc_per_seq[:] = np.nan
    for t in range(n_trials):
        acc_per_seq[:n_seqs[t], t] = bool_result_per_seq[t]

    return np.nanmean(acc_per_seq, axis=1)


# ---------------------------- SSVEP CODE GENERATORS ------------------------- #
class SSVEPCodeGenerator:

    def __init__(self, stim_time, fps, base):
        self.stim_time = stim_time
        self.fps = fps
        self.base = base
        self.seq_len = stim_time * fps
        self.bins = self.__get_quantification_bins()

    def generate_seq(self, freq):
        if freq > self.fps/2:
            raise ValueError('The SSVEP frequency cannot be higher than fps/2.')
        # Generate analog code
        t = np.arange(0, self.seq_len/self.fps, 1/self.fps)
        analog_code = np.sin(2*np.pi*freq*t)
        # Quantification
        digital_code = np.digitize(analog_code, bins=self.bins, right=False)
        return digital_code

    def __get_quantification_bins(self):
        pointer = -1
        step = 2 / self.base
        bins = list()
        for i in range(self.base-1):
            pointer += step
            bins.append(pointer)
        return bins
























