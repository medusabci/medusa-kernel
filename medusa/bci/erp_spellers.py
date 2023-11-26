"""Created on Monday March 15 19:27:14 2021

In this module you will find useful functions and classes to operate with data
recorded using spellers based on event-related pontentials (ERP), which are
widely used by the BCI community. Enjoy!

@author: Eduardo Santamaría-Vázquez
"""

# Built-in imports
import copy, warnings
from abc import ABC, abstractmethod

# External imports
import numpy as np
from scipy import signal
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from tqdm import tqdm

# Medusa imports
import medusa as mds
from medusa import components
from medusa import meeg


class ERPSpellerData(components.ExperimentData):
    """Experiment info class for ERP-based spellers. It supports nested
    multi-level paradigms. This unified class can be used to represent a run
    of every ERP stimulation paradigm designed to date, and is the expected
    class for feature extraction and command decoding functions of the module
    medusa.bci.erp_paradigms. It is complicated, but powerful so.. use it well!
    """

    def __init__(self, mode, paradigm_conf, commands_info, onsets, batch_idx,
                 group_idx, unit_idx, level_idx, matrix_idx, sequence_idx,
                 trial_idx, spell_result, control_state_result,
                 spell_target=None, control_state_target=None, **kwargs):

        """ERPSpellerData constructor

        Parameters
        ----------
        mode : str
            Mode of this run. Values: {"train"|"test"}
        paradigm_conf :  list
            This parameter describes the paradigm configuration for the
            experiment. The array must have shape [n_matrices x n_units x
            n_groups x n_batches x n_commands/batch]. The matrix is the maximum
            entity of the paradigm and only one can be used in each trial.
            The units are smaller entities that are used in multi-level
            paradigms, such as the Hex-O-spell (HOS) paradigm [1]. In this
            case, each level can use a different unit, affecting the selected
            command for the trial. For instance, in the HOS paradigm,
            you should define 1 matrix with 7 units, one for the initial menu
            and 6 for the second level of each command (letters).
            Importantly, commands must be unequivocally defined in each matrix.
            Therefore, units cannot share command identifiers. Then, the groups
            describe aggregations of commands that are highlighted at the
            same time. For instance, the row-column paradigm (RCP) [2]
            has 2 groups of commands (i.e., rows and columns), whereas the
            HOS has only 1 (i.e., each command is highlighted individually).
            Finally, batches contain the commands IDs defined in each group.
            In an RCP matrix of 6x6, each of the 2 groups has 6 batches,
            corresponding to the rows and columns. This structure supports
            nested multi-level matrices, providing compatibility with all
            paradigms to date and setting a general framework for feature
            extraction and command decoding functions. The relationship between
            the command IDs and the letters or actions should be defined in
            other variable, but it is not necessary for signal processing.

            Example of 2x2 RCP paradigm:

                rcp_conf = [
                    # Matrices
                    [
                        # Units
                        [
                            # Groups
                            [
                                # Batches
                                [0, 1],
                                [2, 3]
                            ],
                            [
                                [0, 2],
                                [1, 3]
                            ]
                        ]
                    ]
                ]

            Example of HOS paradigm:

                hos_conf = [
                    # Matrices
                    [
                        # Units
                        [
                            # Groups
                            [
                                # Batches
                                [0], [1], [2], [3], [4], [5]
                            ],
                        ],
                        [
                            [
                                [6], [7], [8], [9], [10], [11]
                            ],
                        ],
                        [
                            [
                                [12], [13], [14], [15], [16], [17]
                            ],
                        ],
                        [
                            [
                                [18], [19], [20], [21], [22], [23]
                            ],
                        ],
                        [
                            [
                                [24], [25], [26], [27], [28], [29]
                            ],
                        ]
                    ]
                ]
        commands_info : list
            List containing the command information per matrix. Each
            position must be a dict, whose keys are the command ids used in
            paradigm_conf. The value must be another dict containing important
            information about each command (e.g., label, text, action, icon
            path, etc). This information may be different for different use
            cases, but must be serializable (i.e., contain primitive types).
            Shape [n_matrices x n_commands].
        onsets : list or numpy.ndarray
            Timestamp of each stimulation. This timestamps have to be
            synchronized with the EEG (or other biosignal) timestamps in
            order to assure a correct functioning of all medusa functions.
            Shape: [n_stim x 1]
        batch_idx : list or numpy.ndarray
            Index of the highlighted batch for each stimulation. A batch
            represents the highlighted commands in each  stimulation. For
            example in the row-col paradigm (RCP) represents each row and
            column. Shape: [n_stim x 1]
        group_idx : list or numpy.ndarray
            Index of the group that has been highlighted. Groups represent the
            different aggregations of batches. Between batches of different
            groups, 1 command must be common. For example in the RCP there
            are 2 groups: rows and columns. In this paradigm, between each
            pair of batches (e.g., row=2, col=4), there is only one command
            in common. Shape: [n_stim x 1]
        unit_idx: list or numpy.ndarray
            Index of the unit used in each stimulation. Units are low level
            entities used in multi-level paradigms, such as HOS paradigm [1].
            For each level, only 1 unit can be used. As the trial may have
            several layers, several units can be used in 1 trial. For
            instance, in the HOS, the first unit is the main menu. The other
            6 units are each of the lower level entities that are displayed
            in the second level of stimulation. Shape: [n_stim x 1]
        level_idx : list or numpy.ndarray
            Index of the level of each stimulation. Levels represent each
            one of the selections that must be made before a trial is
            finished. For example, in the Hex-O-spell paradigm there are 2
            levels (see [1]). Shape: [n_stim x 1]
        matrix_idx : list or numpy.ndarray
            Index of the matrix used in each stimulation. Each matrix can
            contain several levels. The matrix has to be the same accross the
            entire trial. Shape: [n_stim x 1]
        sequence_idx : list or numpy.ndarray
            Index of the sequence for each stimulation. A sequence
            represents a round of stimulation: all commands have been
            highlighted 1 time. This class support dynamic stopping in
            different levels. Shape: [n_stim x 1]
        trial_idx : list or numpy.ndarray
            Index of the trial for each stimulation. A trial represents
            the selection of a final command. Depending on the number of levels,
            the final selection takes N intermediate selections.
        spell_result : list or numpy.ndarray
            Spell result of the run. Each position contains the matrix and
            command id that identifies the selected command per trial and
            level. Shape [n_trials x n_levels x 2]. Eg., in the RCP:
            [[[matrix_idx, cmd_id], [matrix_idx, cmd_id]]].
        control_state_result : list or numpy.ndarray
            Control state result of the run. Each position contains the
            detected control state of the user per trial (0 -> non-control,
            1-> control). Shape: [n_trials x 1]. Values {0|1}
        spell_target : list or numpy.ndarray or None
            Spell target of the run. Each position contains the matrix and
            command id per level that identifies the target command of the
            trial. Shape [n_trials x n_levels x 2]. Eg., in the RCP:
            [[[matrix_idx, cmd_id], [matrix_idx, cmd_id], etc]].
        control_state_target : list or numpy.ndarray or None
            Control state target of the run. Each position contains the
            target control state of the user per trial (0 -> non-control,
            1-> control). Shape: [n_trials x 1]. Values {0|1}
        kwargs : kwargs
            Custom arguments that will also be saved in the class


        References
        ----------
        [1] Blankertz, B., Dornhege, G., Krauledat, M., Schröder,
        M., Williamson, J., Murray-Smith, R., & Müller, K. R. (2006). The
        Berlin Brain-Computer Interface presents the novel mental typewriter
        Hex-o-Spell.

        [2] Farwell, L. A., & Donchin, E. (1988). Talking off the top of your
        head: toward a mental prosthesis utilizing event-related brain
        potentials. Electroencephalography and clinical Neurophysiology,
        70(6), 510-523.
        """
        # Check errors
        mode = mode.lower()
        if mode not in ('train', 'test'):
            raise ValueError('Unknown mode. Possible values {train, test}')

        # Standard attributes
        self.mode = mode
        self.paradigm_conf = paradigm_conf
        self.commands_info = commands_info
        self.onsets = onsets
        self.batch_idx = batch_idx
        self.group_idx = group_idx
        self.unit_idx = unit_idx
        self.level_idx = level_idx
        self.matrix_idx = matrix_idx
        self.sequence_idx = sequence_idx
        self.trial_idx = trial_idx
        self.spell_result = spell_result
        self.control_state_result = control_state_result
        self.spell_target = spell_target
        self.control_state_target = control_state_target
        self.erp_labels = self.compute_erp_labels() \
            if mode == 'train' else None
        self.control_state_labels = self.compute_control_state_labels() \
            if mode == 'train' else None
        # Optional attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

    def get_batches_associated_to_cmd(self, matrix_idx, command_idx):
        """This function returns the batches associated to the given command
        as defined in the paradigm configuration

        Parameters
        ----------
        matrix_idx: int
            Matrix of the command
        command_idx: int
            Index of the command as defined in attribute commands_info
        """
        tg_ids_batches_list = list()
        for u, unit in enumerate(self.paradigm_conf[matrix_idx]):
            for g, group in enumerate(unit):
                for b, batch in enumerate(group):
                    if command_idx in batch:
                        tg_ids_batches_list.append(
                            [matrix_idx, u, g, b])
        return tg_ids_batches_list

    def compute_erp_labels(self):
        """
        This function computes the erp label vector (0 if the epoch doesn't
        have ERP, 1 if the epoch have ERP).
        """
        # Convert to numpy array
        batch_idx = np.array(self.batch_idx)
        group_idx = np.array(self.group_idx)
        unit_idx = np.array(self.unit_idx)
        level_idx = np.array(self.level_idx)
        matrix_idx = np.array(self.matrix_idx)
        trial_idx = np.array(self.trial_idx)
        # Get batches associated to each target
        target_ids_batches = dict()
        for t, trial_target in enumerate(self.spell_target):
            for l, level_target in enumerate(trial_target):
                tg_matrix_idx = level_target[0]
                tg_id = level_target[1]
                target_ids_batches[(t, l)] = self.get_batches_associated_to_cmd(
                    tg_matrix_idx, tg_id)
        # Compute erp_labels
        erp_labels = np.zeros((len(batch_idx))).astype(int)
        for key, batches in target_ids_batches.items():
            for batch in batches:
                erp_labels_idx = np.ones((len(batch_idx))).astype(int)
                erp_labels_idx = np.logical_and(erp_labels_idx,
                                                trial_idx == key[0])
                erp_labels_idx = np.logical_and(erp_labels_idx,
                                                level_idx == key[1])
                erp_labels_idx = np.logical_and(erp_labels_idx,
                                                matrix_idx == batch[0])
                erp_labels_idx = np.logical_and(erp_labels_idx,
                                                unit_idx == batch[1])
                erp_labels_idx = np.logical_and(erp_labels_idx,
                                                group_idx == batch[2])
                erp_labels_idx = np.logical_and(erp_labels_idx,
                                                batch_idx == batch[3])
                erp_labels[erp_labels_idx] = 1
        return erp_labels

    def compute_control_state_labels(self):
        """
        This function computes the control state label vector (0 -> non-control
        state, 1 -> control state).
        """
        trial_idx = np.array(self.trial_idx)
        labels = np.zeros_like(trial_idx).astype(int)
        for t, trial in enumerate(np.unique(trial_idx)):
            labels[trial_idx == trial] = self.control_state_target[t]
        return labels

    @staticmethod
    def get_paradigm_conf_for_rcp(matrix_dims, commands_info_rcp=None):
        """Returns the paradigm configuration matrix for the row-column paradigm
        (RCP) experiment [1]

        Parameters
        ----------
        matrix_dims : list or np.array
            Array containing the dimensions of the matrices used in the
            experiment. For each matrix, the first position is the number of
            rows and the second the number of columns. Shape [n_matrices x 2]
        commands_info_rcp: list or None
            Array containing the dict info of each command, arranged in 2D
            matrices. Shape [n_matrices x n_rows x n_cols]

        Returns
        -------
        paradigm_conf : list
            Array with the paradigm configuration for an RCP paradigm
        commands_info : list
            Flattened version of commands_info input. It contains the command
            information corresponding to paradigm_conf. If input parameter
            commands_info is None, this output will be a skeleton with an empty
            dict for each command. If input commands_info is provided, it will
            be incorporated in the output

        References
        ----------
        [1] Farwell, L. A., & Donchin, E. (1988). Talking off the top of your
        head: toward a mental prosthesis utilizing event-related brain
        potentials. Electroencephalography and clinical Neurophysiology,
        70(6), 510-523.
        """
        # Paradigm conf
        matrix_dims = np.array(matrix_dims)
        paradigm_conf = list()
        commands_info = list()
        for m in range(matrix_dims.shape[0]):
            # Commands matrix
            n_rows = matrix_dims[m, 0]
            n_cols = matrix_dims[m, 1]
            commands_ids = np.arange(n_rows * n_cols)
            matrix = commands_ids.reshape((n_rows, n_cols))
            # Paradigm conf
            paradigm_conf.append(list())                    # Matrix
            paradigm_conf[m].append(list())                 # Unit
            paradigm_conf[m][0].append(matrix.tolist())     # Append group rows
            paradigm_conf[m][0].append(matrix.T.tolist())   # Append group cols
            # Commands info
            cmd_info_keys = commands_ids.tolist()
            if commands_info_rcp is None:
                cmd_info_values = [dict() for i in cmd_info_keys]
            else:
                cmd_info_values = np.array(commands_info_rcp[m]).flatten()
                cmd_info_values = cmd_info_values.tolist()
            commands_info.append(dict(zip(cmd_info_keys, cmd_info_values)))

        return paradigm_conf, commands_info

    @staticmethod
    def get_paradigm_conf_for_hox(matrix_dims, commands_info_hox=None):
        """Returns the paradigm configuration matrix for the Hex-O-Speller (HOX)
        or cake paradigms from the Berlin BCI Group [1]. This paradigm has 2
        levels of selection with 6 commands in each unit.

        Parameters
        ----------
        matrix_dims : list or np.array
            Array containing the dimensions of the matrices used in the
            experiment. For each matrix, the first position is the number of
            commands of the first level and the second the number of commands
            of the second level (typically both are 6). Shape [n_matrices x 2]
        commands_info_hox: list or None
            Array containing the dict info of each command. The first
            dimension are the matrices, the second dimension represent the
            units, and the third dimension contains the dictionary with the
            info of each command. Typically, this paradigm has 7 units of 6
            commands each. As defined by the Berlin BCI group: 1 menu matrix and
            6 matrix for each group of 6 letters. Therefore, with this
            setup, this array has shape [n_matrices x 7 x 6]

        Returns
        -------
        paradigm_conf : list
            Array with the paradigm configuration for an RCP paradigm
        commands_info : list
            Flattened version of commands_info input. It contains the command
            information corresponding to paradigm_conf. If input parameter
            commands_info is None, this output will be a skeleton with an empty
            dict for each command. If input commands_info is provided, it will
            be incorporated in the output

        References
        ----------
        [1] Blankertz, B., Dornhege, G., Krauledat, M., Schröder,
        M., Williamson, J., Murray-Smith, R., & Müller, K. R. (2006). The
        Berlin Brain-Computer Interface presents the novel mental typewriter
        Hex-o-Spell.
        """
        # Paradigm conf
        matrix_dims = np.array(matrix_dims)
        paradigm_conf = list()
        commands_info = list()
        for m in range(matrix_dims.shape[0]):
            # Useful variables
            n_cmd_l1 = matrix_dims[m, 0]
            n_cmd_l2 = matrix_dims[m, 0]
            units = list()
            commands_ids = list()
            # First unit (level 1)
            cmd_ids_u1 = list(range(n_cmd_l1))
            units.append([[[int(i)] for i in cmd_ids_u1]])
            commands_ids += cmd_ids_u1
            # Rest of units (level 2)
            for u in range(n_cmd_l1):
                cmd_ids_ux = list(range(
                    commands_ids[-1] + 1, commands_ids[-1] + 1 + n_cmd_l2
                ))
                units.append([[[int(i)] for i in cmd_ids_ux]])
                commands_ids += cmd_ids_ux
            paradigm_conf.append(units)
            # Commands info
            if commands_info_hox is None:
                cmd_info_values = [dict() for __ in commands_ids]
            else:
                cmd_info_values = np.array(commands_info_hox[m]).flatten()
                cmd_info_values = cmd_info_values.tolist()
            commands_info.append(dict(zip(commands_ids, cmd_info_values)))

        return paradigm_conf, commands_info

    def to_serializable_obj(self):
        rec_dict = self.__dict__
        for key in rec_dict.keys():
            if type(rec_dict[key]) == np.ndarray:
                rec_dict[key] = rec_dict[key].tolist()
        return rec_dict

    @classmethod
    def from_serializable_obj(cls, dict_data):
        return cls(**dict_data)


class ERPSpellerDataset(components.Dataset):

    """This class inherits from medusa.data_structures.Dataset, increasing
    its functionality for datasets with data from ERP-based spellers. It
    provides common ground for the rest of functions in the module.
    """

    def __init__(self, channel_set, fs=None, biosignal_att_key='eeg',
                 experiment_att_key='erpspellerdata', experiment_mode=None,
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
            'batch_idx': {
                'track_mode': 'concatenate',
                'parent': experiment_att_key
            },
            'group_idx': {
                'track_mode': 'concatenate',
                'parent': experiment_att_key
            },
            'unit_idx': {
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
            'sequence_idx': {
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
                'erp_labels': {
                    'track_mode': 'concatenate',
                    'parent': experiment_att_key
                },
                'control_state_labels': {
                    'track_mode': 'concatenate',
                    'parent': experiment_att_key
                },
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

        # Check experiment
        checker.add_consistency_rule(
            rule='check-attribute',
            rule_params={'attribute': self.experiment_att_key}
        )
        checker.add_consistency_rule(
            rule='check-attribute-type',
            rule_params={'attribute': self.experiment_att_key,
                         'type': ERPSpellerData}
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


class StandardPreprocessing(components.ProcessingMethod):
    """Just the common preprocessing applied in ERP-based spellers. Simple,
    quick and effective: frequency IIR filter followed by common average
    reference (CAR) spatial filter.
    """
    def __init__(self, order=5, cutoff=(0.5, 10), btype='bandpass',
                 filt_method='sosfiltfilt'):
        super().__init__(fit_transform_signal=['signal'],
                         fit_transform_dataset=['dataset'])
        # Parameters
        self.order = order
        self.cutoff = cutoff
        self.btype = btype
        self.filt_method = filt_method

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
        self.iir_filter = mds.IIRFilter(order=self.order,
                                        cutoff=self.cutoff,
                                        btype=self.btype,
                                        filt_method=self.filt_method)
        self.iir_filter.fit(fs, n_cha=n_cha)

    def transform_signal(self, signal):
        """Transforms an EEG signal applying IIR filtering and CAR sequentially

        Parameters
        ----------
        signal: np.array or list
            Signal to transform. Shape [n_samples x n_channels]
        """
        signal = self.iir_filter.transform(signal)
        signal = mds.car(signal)
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
        self.iir_filter = mds.IIRFilter(order=self.order,
                                        cutoff=self.cutoff,
                                        btype=self.btype,
                                        filt_method=self.filt_method)
        signal = self.iir_filter.fit_transform(signal, fs)
        signal = mds.car(signal)
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
    def __init__(self, w_epoch_t=(0, 1000), target_fs=20,
                 w_baseline_t=(-250, 0), norm='z',
                 concatenate_channels=True, safe_copy=True):
        """Class constructor

        w_epoch_t : list
            Temporal window in ms for each epoch relative to the event onset
            (e.g., [0, 1000])
        target_fs : float of None
            Target sample rate of each epoch. If None, all the recordings must
            have the same sample rate, so it is strongly recommended to set this
            parameter to a suitable value to avoid problems and save time
        w_baseline_t : list
            Temporal window in ms to be used for baseline normalization for each
            epoch relative to the event onset (e.g., [-250, 0])
        norm : str {'z'|'dc'}
            Type of baseline normalization. Set to 'z' for Z-score normalization
            or 'dc' for DC normalization
        concatenate_channels : bool
            This parameter controls the shape of the feature array. If True, all
            channels will be concatenated, returning an array of shape [n_events
            x (samples x channels)]. If false, the array will have shape
            [n_events x samples x channels]
        safe_copy : bool
            Makes a safe copy of the signal to avoid changing the original
            samples due to references
        """
        super().__init__(transform_signal=['x'],
                         transform_dataset=['x', 'x_info'])
        self.w_epoch_t = w_epoch_t
        self.target_fs = target_fs
        self.w_baseline_t = w_baseline_t
        self.norm = norm
        self.concatenate_channels = concatenate_channels
        self.safe_copy = safe_copy

    def transform_signal(self, times, signal, fs, onsets):
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
                                            w_epoch_t=self.w_epoch_t,
                                            w_baseline_t=self.w_baseline_t,
                                            norm=self.norm)
        # Resample each epoch to the target frequency
        if self.target_fs is not None:
            if self.target_fs > fs:
                raise warnings.warn('Target fs is greater than data fs')
            features = mds.resample_epochs(features,
                                           self.w_epoch_t,
                                           self.target_fs)
        # Reshape epochs and concatenate the channels
        if self.concatenate_channels:
            features = np.squeeze(features.reshape((features.shape[0],
                                                    features.shape[1] *
                                                    features.shape[2], 1)))
        return features

    def transform_dataset(self, dataset, show_progress_bar=True):
        #TODO: Review this description (EDUARDO)
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
            )
            features = np.concatenate((features, rec_feat), axis=0) \
                if features is not None else rec_feat

            # Special attributes that need tracking across runs to assure the
            # consistency of the dataset
            rec_exp.run_idx = run_counter * np.ones_like(rec_exp.trial_idx)
            rec_exp.trial_idx = trial_counter + np.array(rec_exp.trial_idx)

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


def decode_commands(scores, paradigm_conf, run_idx, trial_idx, matrix_idx,
                    level_idx, unit_idx, sequence_idx, group_idx, batch_idx):

    """Command decoder for ERP-based spellers.

    Parameters
    ----------
    scores : list or np.ndarray
        Array with the score for each stimulation
    paradigm_conf : list or np.ndarray
        Array containing the unified speller matrix structure with shape
        [n_runs x n_matrices x n_units x n_groups x n_batches x
        n_commands/batch]. All ERP-based speller paradigms can be adapted to
        this format and use this function for command decoding. See
        ERPSpellerData class for more info.
    run_idx : list or numpy.ndarray [n_stim x 1]
        Index of the run for each stimulation. This variable is automatically
        retrieved by function extract_erp_features_from_dataset as part of
        the track info dict. The run indexes must be related to
        paradigm_conf, keeping the same order. Therefore
        paradigm_conf[np.unique(run_idx)[0]] must retrieve the paradigm
        configuration of run 0.
    trial_idx : list or numpy.ndarray [n_stim x 1]
        Index of the trial for each stimulation. A trial represents
        the selection of a final command. Depending on the number of levels,
        the final selection takes N intermediate selections.
    matrix_idx : list or numpy.ndarray [n_stim x 1]
        Index of the matrix used in each stimulation. Each matrix can
        contain several levels. The matrix has to be the same accross the
        entire trial.
    level_idx : list or numpy.ndarray [n_stim x 1]
        Index of the level of each stimulation. Levels represent each
        one of the selections that must be made before a trial is
        finished. For example, in the Hex-O-spell paradigm there are 2
        levels (see [1]).
    unit_idx: list or numpy.ndarray [n_stim x 1]
        Index of the unit used in each stimulation. Units are low level
        entities used in multi-level paradigms, such as HOS paradigm [1].
        For each level, only 1 unit can be used. As the trial may have
        several layers, several units can be used in 1 trial. For instance,
        in the HOS, there are 7 units. The first unit is the main  menu. The
        other 6 units are each of the lower level entities that are
        displayed in the second level of stimulation.
    sequence_idx : list or numpy.ndarray [n_stim x 1]
        Index of the sequence for each stimulation. A sequence
        represents a round of stimulation: all commands have been
        highlighted 1 time. This class support dynamic stopping in
        different levels.
    group_idx : list or numpy.ndarray [n_stim x 1]
        Index of the group that has been highlighted. Groups represent the
        different aggregations of batches. Between batches of different
        groups, 1 command must be common. For example in the RCP there
        are 2 groups: rows and columns. In this paradigm, between each
        pair of batches (e.g., row=2, col=4), there is only one command
        in common.
    batch_idx : list or numpy.ndarray [n_stim x 1]
        Index of the code of the highlighted batch for each stimulation.
        A batch represents the highlighted commands in each stimulation.
        For example in the row-col paradigm (RCP) represents each row and
        column.

    Returns
    -------
    selected_commands: list
        Selected command for each trial considering all sequences of
        stimulation. Each command is organized in an array [matrix_idx,
        command_id]. Take into account that the command ids are unique for each
        matrix, and therefore only the command of the last level should be
        useful to take action. Shape [n_runs x n_trials x n_levels x 2]
    selected_commands_per_seq: list
        Selected command for each trial and sequence of stimulation. The
        fourth dimension of the array contains [matrix_idx, command_id]. To
        calculate the command for each sequence, it takes into account the
        scores of all the previous sequences as well. Shape [n_runs x
        n_trials x n_levels x n_sequences x 2]
    scores: list
        Scores for each command per sequence. Shape [n_runs x n_trials x
        n_levels x n_sequences x n_commands x 1]. The score of each sequence
        is calculated independently of the previous sequences.
    """

    # Avoid errors
    scores = np.array(scores)
    run_idx = np.array(run_idx)
    trial_idx = np.array(trial_idx)
    matrix_idx = np.array(matrix_idx)
    level_idx = np.array(level_idx)
    unit_idx = np.array(unit_idx)
    sequence_idx = np.array(sequence_idx)
    group_idx = np.array(group_idx)
    batch_idx = np.array(batch_idx)

    # Check errors
    if len(scores.shape) > 1:
        if len(scores.shape) > 2 or scores.shape[-1] != 1:
            raise ValueError('Parameter scores must have shape '
                             '(n_stim,) or (n_stim, 1)')
    n_stim = scores.shape[0]
    if trial_idx.shape[0] != n_stim or matrix_idx.shape[0] != n_stim or \
            level_idx.shape[0] != n_stim or sequence_idx.shape[0] != n_stim or \
            group_idx.shape[0] != n_stim or  batch_idx.shape[0] != n_stim:
        raise ValueError('Shape mismatch. Parameters scores, trial_idx, '
                         'matrix_idx, level_idx, sequence_idx, group_idx and '
                         'batch_idx must have the same dimensions')

    depth = lambda l: isinstance(l, list) and max(map(depth, l)) + 1
    if len(paradigm_conf) != np.unique(run_idx).shape[0] or \
        depth(paradigm_conf) != 6:
        raise ValueError('Shape mismatch. Parameter paradigm_conf must be a '
                         'list of length n_runs. Take into account that the '
                         'paradigm configuration can change between runs')

    # Command decoding
    cmd_scores = list()
    selected_commands = list()
    selected_commands_per_seq = list()
    idx = np.arange(trial_idx.shape[0])

    # Iterate each trial
    r_cnt = 0  # Run counter
    for r in np.unique(run_idx):
        idx_r = idx[np.where(run_idx == r)]
        # Increment dimensionality
        cmd_scores.append(list())
        selected_commands_per_seq.append(list())
        selected_commands.append(list())
        # Reset trial counter
        t_cnt = 0  # Trial counter
        for t in np.unique(trial_idx[idx_r]):
            try:
                idx_t = idx_r[np.where(trial_idx[idx_r] == t)]
                # Get matrix index
                m = int(np.squeeze(np.unique(matrix_idx[trial_idx == t])))
                # Update lists
                cmd_scores[r_cnt].append(list())
                selected_commands_per_seq[r_cnt].append(list())
                selected_commands[r_cnt].append(list())
                # Reset level counter
                l_cnt = 0  # Level counter
                for l in np.unique(level_idx[idx_t]):
                    idx_l = idx_t[np.where(level_idx[idx_t] == l)]
                    # Append list to cmd_scores
                    cmd_scores[r_cnt][t_cnt].append(list())
                    selected_commands_per_seq[r_cnt][t_cnt].append(list())
                    # selected_commands[r_cnt][t_cnt].append(list())
                    # Get unit index
                    u = int(np.squeeze(np.unique(unit_idx[idx_l])))
                    # Commands in this unit
                    commands_id = list()
                    for x in paradigm_conf[r_cnt][m][u]:
                        for y in x:
                            for z in y:
                                commands_id.append(z)
                    commands_id = np.unique(commands_id)
                    # Reset sequences counter
                    s_cnt = 0
                    for s in np.unique(sequence_idx[idx_l]):
                        idx_s = idx_l[np.where(sequence_idx[idx_l] == s)]
                        # Append one list for each command
                        cmd_scores[r_cnt][t_cnt][l_cnt].append(list())
                        selected_commands_per_seq[r_cnt][t_cnt][l_cnt].\
                            append([m])
                        for __ in commands_id:
                            cmd_scores[r_cnt][t_cnt][l_cnt][s_cnt].append([m])
                        # Iterate over groups
                        for g in np.unique(group_idx[idx_s]):
                            idx_g = idx_s[np.where(group_idx[idx_s] == g)]
                            # Iterate over batches
                            for b in np.unique(batch_idx[idx_g]):
                                idx_b = np.squeeze(
                                    idx_g[np.where(batch_idx[idx_g] == b)]
                                )
                                # Reset command counter
                                for cmd in paradigm_conf[r_cnt][m][u][g][b]:
                                    c = np.squeeze(np.where(commands_id == cmd))
                                    sc = scores[idx_b]
                                    cmd_scores[r_cnt][t_cnt][l_cnt][s_cnt][
                                        int(c)].append(sc)

                        # Append selected command for this sequence and trial
                        s_scores = np.array(cmd_scores[r_cnt][t_cnt][l_cnt])
                        s_scores_mean = np.mean(np.mean(s_scores, axis=2),
                                                axis=0)
                        sel_cmd = commands_id[np.argmax(s_scores_mean)]
                        selected_commands_per_seq[r_cnt][t_cnt][l_cnt][s_cnt].\
                            append(sel_cmd)
                        # Increment sequence counter
                        s_cnt += 1
                    # Append selected cmd (avoid reference problems with copy)
                    selected_commands[r_cnt][t_cnt].append(
                        copy.copy(
                            selected_commands_per_seq[r_cnt][t_cnt][-1][-1]
                        )
                    )
                    # Increment level counter
                    l_cnt += 1
                # Increment trial counter
                t_cnt += 1
            except Exception as e:
                raise type(e)('Error in trial %i: %s' % (t, str(e)))
        # Increment run counter
        r_cnt += 1

    return selected_commands, selected_commands_per_seq, cmd_scores


def detect_control_state(scores, run_idx, trial_idx, sequence_idx):
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
    sequence_idx : list or numpy.ndarray [n_stim x 1]
        Index of the sequence for each stimulation. A sequence
        represents a round of stimulation: all commands have been
        highlighted 1 time. This class support dynamic stopping in
        different levels.

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
    sequence_idx = np.array(sequence_idx)

    # Check errors
    if len(scores.shape) > 1:
        if len(scores.shape) > 2 or scores.shape[-1] != 1:
            raise ValueError('Parameter scores must have shape '
                             '(n_stim,) or (n_stim, 1)')
    n_stim = scores.shape[0]
    if run_idx.shape[0] != n_stim or trial_idx.shape[0] != n_stim or \
            sequence_idx.shape[0] != n_stim:
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
            for s_cnt, s in enumerate(np.unique(sequence_idx[idx_t])):
                idx_s = idx_t[np.where(sequence_idx[idx_t] == s)]
                # Score for this sequence (only this sequence)
                state_scores[r_cnt][t_cnt].append(np.mean(scores[idx_s]))
                # Score for this sequence (all sequences <= s)
                cs_score = np.mean(state_scores[r_cnt][t_cnt])
                cs_pred = int(cs_score > 0.5)
                selected_control_state_per_seq[r_cnt][t_cnt].append(cs_pred)
            selected_control_state[r_cnt].append(
                selected_control_state_per_seq[r_cnt][t_cnt][-1]
            )

    return selected_control_state, selected_control_state_per_seq, state_scores


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
            for t in range(len(t_sel_cmd_per_seq)):
                l_sel_cmd_per_seq = t_sel_cmd_per_seq[t]
                l_spell_target = t_spell_target[t]
                t_bool_result_per_seq.append(list())
                t_n_seqs.append(len(l_sel_cmd_per_seq))
                for s in range(len(l_sel_cmd_per_seq)):
                    s_sel_cmd_per_seq = l_sel_cmd_per_seq[s]
                    t_bool_result_per_seq[t].append(l_spell_target ==
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


def control_state_detection_accuracy(selected_control_state,
                                     target_control_state):
    """Computes the accuracy of the selected control states given the target

    Parameters
    ----------
    selected_control_state: list
        Target commands. Each position contains the matrix index and command
        id per level that identifies the selected command of the trial. Shape
        [n_runs x n_trials]
    target_control_state: list
        Target control state. Each position contains the control state of the
        trial. Shape [n_runs x n_trials]

    Returns
    -------
    accuracy : float
        Accuracy of the command decoding stage
    """
    # Check errors
    if len(selected_control_state) != len(target_control_state):
        raise ValueError('Parameters selected_control_state and '
                         'target_control_state must have the same shape '
                         '[n_runs x n_trials]')

    t_correct_cnt = 0
    t_total_cnt = 0
    for r in range(len(selected_control_state)):
        for t in range(len(selected_control_state[r])):
            if selected_control_state[r][t] == target_control_state[r][t]:
                t_correct_cnt += 1
            t_total_cnt += 1

    accuracy = t_correct_cnt / t_total_cnt
    return accuracy


def control_state_detection_accuracy_per_seq(selected_control_state_per_seq,
                                             target_control_state):
    """
    Returns the accuracy of the selected sequence of predicted control
    states given the target.

    Parameters
    ----------
    selected_control_state_per_seq: list
        List with the control state detection result per sequence as given by
        function detect_control_state. Shape [n_runs x n_trials x n_sequences]
    target_control_state : list
        Numpy array with dimensions [n_runs x n_trials] with the real control
        state of each trial

    Returns
    -------
    acc_per_seq : float
        Accuracy of the control state detection stage
    """

    # Check errors
    selected_control_state_per_seq = list(selected_control_state_per_seq)
    target_control_state = list(target_control_state)
    if len(selected_control_state_per_seq) != len(target_control_state):
        raise ValueError('Parameters selected_control_state_per_seq and '
                         'target_control_state must have the same length.')

    # Compute accuracy per sequence
    bool_result_per_seq = []
    n_seqs = []
    for r in range(len(selected_control_state_per_seq)):
        r_sel_cmd_per_seq = selected_control_state_per_seq[r]
        r_cs_target = target_control_state[r]
        for t in range(len(r_sel_cmd_per_seq)):
            t_sel_cmd_per_seq = r_sel_cmd_per_seq[t]
            t_cs_target = r_cs_target[t]
            t_bool_result_per_seq = []
            for s in range(len(t_sel_cmd_per_seq)):
                s_sel_cmd_per_seq = t_sel_cmd_per_seq[s]
                t_bool_result_per_seq.append(t_cs_target == s_sel_cmd_per_seq)
            # Save for later use
            bool_result_per_seq.append(t_bool_result_per_seq)
            n_seqs.append(len(t_sel_cmd_per_seq))

    # Calculate the accuracy per number of sequences
    max_n_seqs = np.max(n_seqs)
    n_trials = len(bool_result_per_seq)
    acc_per_seq = np.empty((max_n_seqs, n_trials))
    acc_per_seq[:] = np.nan
    for t in range(n_trials):
        acc_per_seq[:n_seqs[t], t] = bool_result_per_seq[t]

    return np.nanmean(acc_per_seq, axis=1)


def split_erp_features(sets_pct, trial_idx_key="trial_idx", **kwargs):
    """
    This function splits randomly an ERP dataset keeping the relation between
    epochs and trials, which enables a later analysis of command prediction.

    Parameters
    ----------
    sets_pct: list
        List containing the percentage of for each set. For example,
        sets_pct=[60, 20, 20] will split the dataset in three sets, one that
        contains 60% of the trials, and two that contains 20% each.

    trial_idx_key: string
        Sets the trial_track_key, which is a vector that keeps the relation
        between trials and stimulus.

    kwargs: key-value arguments
        Variables to split.

    Returns
    -------
    variables: list
        List of the sets containing a dict with the split variables
    """

    # Check that the sets_pct parameter is a list
    if type(sets_pct) != list:
        raise ValueError("Parameter sets_pct must be of type list")
    # Check that the number of output sets is greater than 1
    n_sets = len(sets_pct)
    if n_sets < 2:
        raise ValueError("The number of output sets must be greater than 1")
    # Check that the sum of the percentages is equal to 1
    if sum(sets_pct) != 100:
        raise ValueError("The sum of the percentages for each set must be "
                         "equal to 1")
    # Check that the number of sets and the length of the percentage list
    # are equal
    if n_sets != len(sets_pct):
        raise ValueError("The number of sets and the set pct length must "
                         "coincide")
    # Check the existence of the trial_track array
    if not trial_idx_key in kwargs:
        raise ValueError("Array '" + trial_idx_key +
                         "' not found. This array must relate uniquely each "
                         "observation or epoch with the corresponding trial")
    # Check that all the arrays are numpy arrays
    for key, value in kwargs.items():
        if type(value) != np.ndarray:
            raise ValueError("Matrix '" + key + "' must be a numpy array")
    # Compute the number of epochs and trials and check the dimensions of all
    # the arrays
    n_epochs = kwargs[trial_idx_key].shape[0]
    n_trials = len(np.unique(kwargs[trial_idx_key]))
    for key, value in kwargs.items():
        if value.shape[0] != n_epochs and value.shape[0] != n_trials:
            raise ValueError("Array " + key +
                             " must be either of length n_epochs = " +
                             str(n_epochs) + " or n_trials = " +
                             str(n_trials) + " in axis=0")

    # Compute the number of trials per set
    n_trials_per_set = []
    for s in range(n_sets):
        n_trials_per_set.append(
            np.round(float(sets_pct[s])/100 * n_trials).astype(int)
        )

    # Create the trial index array for each set
    idx = np.unique(kwargs[trial_idx_key])
    np.random.shuffle(idx)  # TODO: Reorganize the indices randomly?
    trials_per_set = []
    last_idx = 0
    for s in range(n_sets):
        # Train set trials
        trials_per_set.append(idx[last_idx:last_idx+n_trials_per_set[s]])
        last_idx = last_idx + n_trials_per_set[s]
    # Split the sets
    sets = list()
    for s in range(n_sets):
        set_dict = dict()
        idx_epochs = np.isin(kwargs[trial_idx_key], trials_per_set[s])
        idx_trials = np.isin(np.unique(kwargs[trial_idx_key]),
                             trials_per_set[s])
        for key, value in kwargs.items():
            if value.shape[0] == n_epochs:
                set_dict[key] = value[idx_epochs]
            else:
                set_dict[key] = value[idx_trials]
        sets.append(set_dict)
    return sets


class ERPSpellerModel(components.Algorithm):
    """Skeleton class for ERP-based spellers models. This class inherits from
    components.Algorithm. Therefore, it can be used to create standalone
    algorithms that can be used in compatible apps from medusa-platform
    for online experiments. See components.Algorithm to know more about this
    functionality.

    Related tutorials:

        - Overview of erp_spellers module [LINK]
        - Create standalone models for ERP-based spellers compatible with
            Medusa platform [LINK]
    """

    def __init__(self):
        """Class constructor
        """
        super().__init__(fit_dataset=['spell_target',
                                      'spell_result_per_seq',
                                      'spell_acc_per_seq'],
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


class CMDModelRLDA(ERPSpellerModel):
    """Command decoding model for ERP-based spellers model based on regularized
    linear discriminant analysis (rLDA) based on the implementation of
    Krusienski et al 2008 [1].

    Dataset features:

    - Sample rate of the signals > 20 Hz. The model can handle recordings
        with different sample rates.
    - Recommended channels: ['Fz', 'Cz', 'Pz', 'P3', 'P4', 'PO7', 'PO8', 'Oz'].

    Processing pipeline:
    - Preprocessing (medusa.bci.erp_spellers.StandardPreprocessing):

        - IIR Filter (order=5, cutoff=(0.5, 10) Hz: unlike FIR filters, IIR
            filters are quick and can be applied in small signal chunks. Thus,
            they are the preferred method for frequency filter in online
            systems.
        - Common average reference (CAR): widely used spatial filter that
            increases the signal-to-noise ratio of the ERPs.
    - Feature extraction (medusa.bci.erp_spellers.StandardFeatureExtraction):

        - Epochs (window=(0, 1000) ms, resampling to 20 HZ): the epochs of
            signal are extracted for each stimulation. Baseline normalization
            is also applied, taking the window (-250, 0) ms relative to the
            stimulus onset.

    - Feature classification (
    sklearn.discriminant_analysis.LinearDiscriminantAnalysis)

        - Regularized linear discriminant analysis (rLDA): we use the sklearn
            implementation, with eigen solver and auto shrinkage paramers. See
            reference in sklearn doc.

    References
    ----------
    [1] Krusienski, D. J., Sellers, E. W., McFarland, D. J., Vaughan, T. M., &
    Wolpaw, J. R. (2008). Toward enhanced P300 speller performance. Journal of
    neuroscience methods, 167(1), 15-21.
    """

    def __init__(self):
        super().__init__()

    def configure(self, p_filt_cutoff=(0.5, 10), f_w_epoch_t=(0, 800),
                  f_target_fs=20):
        self.settings = {
            'p_filt_cutoff': p_filt_cutoff,
            'f_w_epoch_t': f_w_epoch_t,
            'f_target_fs': f_target_fs
        }
        # Update state
        self.is_configured = True
        self.is_built = False
        self.is_fit = False

    def build(self):
        # Check errors
        if not self.is_configured:
            raise ValueError('Function configure must be called first!')
        # Preprocessing (default: bandpass IIR filter [0.5, 10] Hz + CAR)
        self.add_method('prep_method', StandardPreprocessing(
            cutoff=self.settings['p_filt_cutoff']
        ))
        # Feature extraction (default: epochs [0, 800] ms + resampling to 20 Hz)
        self.add_method('ext_method', StandardFeatureExtraction(
            w_epoch_t=self.settings['f_w_epoch_t'],
            target_fs=self.settings['f_target_fs'],
        ))
        # Feature classification (rLDA)
        clf = components.ProcessingClassWrapper(
            LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto'),
            fit=[], predict_proba=['y_pred']
        )
        self.add_method('clf_method', clf)
        # Update state
        self.is_built = True
        self.is_fit = False

    def fit_dataset(self, dataset, **kwargs):
        # Check errors
        if not self.is_built:
            raise ValueError('Function build must be called first!')
        # Preprocessing
        dataset = self.get_inst('prep_method').fit_transform_dataset(dataset)
        # Extract features
        x, x_info = self.get_inst('ext_method').transform_dataset(dataset)
        if self.check_for_non_control_data(x_info['control_state_labels'],
                                           x_info['control_state_target'],
                                           throw_warning=True):
            x, x_info = self.get_control_data(x, x_info)
        # Classification
        self.get_inst('clf_method').fit(x, x_info['erp_labels'])
        y_pred = self.get_inst('clf_method').predict_proba(x)[:, 1]
        # Command decoding
        spell_result, spell_result_per_seq, __ = decode_commands(
            scores=y_pred,
            paradigm_conf=x_info['paradigm_conf'],
            run_idx=x_info['run_idx'],
            trial_idx=x_info['trial_idx'],
            matrix_idx=x_info['matrix_idx'],
            level_idx=x_info['level_idx'],
            unit_idx=x_info['unit_idx'],
            sequence_idx=x_info['sequence_idx'],
            group_idx=x_info['group_idx'],
            batch_idx=x_info['batch_idx']
        )
        # Spell accuracy per seq
        spell_acc_per_seq = command_decoding_accuracy_per_seq(
            spell_result_per_seq,
            x_info['spell_target']
        )
        cmd_assessment = {
            'x': x,
            'x_info': x_info,
            'y_pred': y_pred,
            'spell_result': spell_result,
            'spell_result_per_seq': spell_result_per_seq,
            'spell_acc_per_seq': spell_acc_per_seq
        }
        # Save info
        self.channel_set = dataset.channel_set
        # Update state
        self.is_fit = True
        return cmd_assessment

    def predict(self, times, signal, fs, channel_set, x_info, **kwargs):
        # Check errors
        if not self.is_fit:
            raise ValueError('Function fit_dataset must be called first!')
        # Check channel set
        if self.channel_set != channel_set:
            warnings.warn('The channel set is not the same that was used to '
                          'fit the model. Be careful!')
        # Preprocessing
        signal = self.get_inst('prep_method').fit_transform_signal(signal, fs)

        # Extract features
        x = self.get_inst('ext_method').transform_signal(times, signal, fs,
                                                         x_info['onsets'])
        # Classification
        y_pred = self.get_inst('clf_method').predict_proba(x)[:, 1]

        # Command decoding
        spell_result, spell_result_per_seq, __ = decode_commands(
            scores=y_pred,
            paradigm_conf=x_info['paradigm_conf'],
            run_idx=x_info['run_idx'],
            trial_idx=x_info['trial_idx'],
            matrix_idx=x_info['matrix_idx'],
            level_idx=x_info['level_idx'],
            unit_idx=x_info['unit_idx'],
            sequence_idx=x_info['sequence_idx'],
            group_idx=x_info['group_idx'],
            batch_idx=x_info['batch_idx']
        )
        cmd_decoding = {
            'x': x,
            'x_info': x_info,
            'y_pred': y_pred,
            'spell_result': spell_result,
            'spell_result_per_seq': spell_result_per_seq
        }
        return cmd_decoding


class CMDModelEEGNet(ERPSpellerModel):
    """Command decoding model for ERP-based spellers model based on EEGNet, a
    compact deep convolutional neural network specifically developed for EEG
    applications [1].

    Dataset features:

    - Sample rate of the signals > 128 Hz. The model can handle recordings
        with different sample rates.
    - Recommended channels: ['Fz', 'Cz', 'Pz', 'P3', 'P4', 'PO7', 'PO8', 'Oz'].

    Processing pipeline:
    - Preprocessing:

        - IIR Filter (order=5, cutoff=(0.5, 45) Hz: unlike FIR filters, IIR
            filters are quick and can be applied in small signal chunks. Thus,
            they are the preferred method for frequency filter in online systems
        - Common average reference (CAR): widely used spatial filter that
            increases the signal-to-noise ratio of the ERPs.
    - Feature extraction:

        - Epochs (window=(0, 1000) ms, resampling to 128 HZ): the epochs of
            signal are extract for each stimulation. Baseline normalization
            is also applied, taking the window (-250, 0) ms relative to the
            stimulus onset.

    - Feature classification

        - EEGNet: compact convolutional network.

    References
    ----------
    [1] Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon, S. M., Hung,
    C. P., & Lance, B. J. (2018). EEGNet: a compact convolutional neural network
    for EEG-based brain–computer interfaces. Journal of neural engineering,
    15(5), 056013.
    """
    def __init__(self):
        super().__init__()

    def configure(self, cnn_n_cha=8, fine_tuning=False,
                  shuffle_before_fit=True, validation_split=0.2,
                  init_weights_path=None, gpu_acceleration=True):
        self.settings = {
            'cnn_n_cha': cnn_n_cha,
            'fine_tuning': fine_tuning,
            'shuffle_before_fit': shuffle_before_fit,
            'validation_split': validation_split,
            'init_weights_path': init_weights_path,
            'gpu_acceleration': gpu_acceleration
        }
        # Update state
        self.is_configured = True
        self.is_built = False
        self.is_fit = False

    def build(self):
        # Check errors
        if not self.is_configured:
            raise ValueError('Function configure must be called first!')
        # Only import deep learning models if necessary
        from medusa.deep_learning_models import EEGNet
        # Preprocessing (bandpass IIR filter [0, 10] Hz + CAR)
        self.add_method('prep_method', StandardPreprocessing(cutoff=(0.5, 45)))
        # Feature extraction (epochs [0, 1000] ms + resampling to 128 Hz)
        ext_method = StandardFeatureExtraction(
            target_fs=128, concatenate_channels=False)
        self.add_method('ext_method', ext_method)
        # Feature classification
        clf = EEGNet(nb_classes=2, n_cha=self.settings['cnn_n_cha'],
                     samples=128, dropout_rate=0.25, kern_length=64, F1=8,
                     D=2, F2=16, dropout_type='Dropout', norm_rate=0.25,
                     gpu_acceleration=self.settings['gpu_acceleration'])
        if self.settings['init_weights_path'] is not None:
            clf.load_weights(self.settings['init_weights_path'])
        self.add_method('clf_method', clf)
        # Update state
        self.is_built = True
        self.is_fit = False

    def fit_dataset(self, dataset, **kwargs):
        # Check errors
        if not self.is_built:
            raise ValueError('Function build must be called first!')
        if dataset.channel_set.n_cha != self.settings['cnn_n_cha']:
            raise ValueError('The number of channels of the model and the '
                             'dataset does not match!')
        # Preprocessing
        dataset = self.get_inst('prep_method').fit_transform_dataset(dataset)
        # Extract features
        x, x_info = self.get_inst('ext_method').transform_dataset(dataset)
        if self.check_for_non_control_data(x_info['control_state_labels'],
                                           x_info['control_state_target'],
                                           throw_warning=True):
            x, x_info = self.get_control_data(x, x_info)
        # Classification
        self.get_inst('clf_method').fit(
            x, x_info['erp_labels'],
            fine_tuning=self.settings['fine_tuning'],
            shuffle_before_fit=self.settings['shuffle_before_fit'],
            validation_split=self.settings['validation_split'],
            **kwargs)
        y_pred = self.get_inst('clf_method').predict_proba(x)[:, 1]
        # Command decoding
        spell_result, spell_result_per_seq, __ = decode_commands(
            scores=y_pred,
            paradigm_conf=x_info['paradigm_conf'],
            run_idx=x_info['run_idx'],
            trial_idx=x_info['trial_idx'],
            matrix_idx=x_info['matrix_idx'],
            level_idx=x_info['level_idx'],
            unit_idx=x_info['unit_idx'],
            sequence_idx=x_info['sequence_idx'],
            group_idx=x_info['group_idx'],
            batch_idx=x_info['batch_idx']
        )
        # Spell accuracy per seq
        spell_acc_per_seq = command_decoding_accuracy_per_seq(
            spell_result_per_seq,
            x_info['spell_target']
        )
        cmd_assessment = {
            'x': x,
            'x_info': x_info,
            'y_pred': y_pred,
            'spell_result': spell_result,
            'spell_result_per_seq': spell_result_per_seq,
            'spell_acc_per_seq': spell_acc_per_seq
        }
        # Save info
        self.channel_set = dataset.channel_set
        # Update state
        self.is_fit = True
        return cmd_assessment

    def predict(self, times, signal, fs, channel_set, x_info, **kwargs):
        # Check channel set
        if self.channel_set != channel_set:
            warnings.warn('The channel set is not the same that was used to '
                          'fit the model. Be careful!')
        # Preprocessing
        signal = self.get_inst('prep_method').fit_transform_signal(signal, fs)
        # Extract features
        x = self.get_inst('ext_method').transform_signal(times, signal, fs,
                                                         x_info['onsets'])
        # Classification
        y_pred = self.get_inst('clf_method').predict_proba(x)[:, 1]
        # Command decoding
        spell_result, spell_result_per_seq, __ = decode_commands(
            scores=y_pred,
            paradigm_conf=x_info['paradigm_conf'],
            run_idx=x_info['run_idx'],
            trial_idx=x_info['trial_idx'],
            matrix_idx=x_info['matrix_idx'],
            level_idx=x_info['level_idx'],
            unit_idx=x_info['unit_idx'],
            sequence_idx=x_info['sequence_idx'],
            group_idx=x_info['group_idx'],
            batch_idx=x_info['batch_idx']
        )
        cmd_decoding = {
            'x': x,
            'x_info': x_info,
            'y_pred': y_pred,
            'spell_result': spell_result,
            'spell_result_per_seq': spell_result_per_seq
        }
        return cmd_decoding


class CMDModelEEGInception(ERPSpellerModel):
    """Command decoding model for ERP-based spellers model based on
    EEG-Inception, a powerful deep convolutional neural network specifically
    developed for BCI applications [1].

    Dataset features:

    - Sample rate of the signals > 128 Hz. The model can handle recordings
        with different sample rates.
    - Recommended channels: ['Fz', 'Cz', 'Pz', 'P3', 'P4', 'PO7', 'PO8', 'Oz'].

    Processing pipeline:

    - Preprocessing:

        - IIR Filter (order=5, cutoff=(0.5, 45) Hz: unlike FIR filters, IIR
            filters are quick and can be applied in small signal chunks. Thus,
            they are the preferred method for frequency filter in online systems
        - Common average reference (CAR): widely used spatial filter that
            increases the signal-to-noise ratio of the ERPs.
    - Feature extraction:

        - Epochs (window=(0, 1000) ms, resampling to 128 HZ): the epochs of
            signal are extract for each stimulation. Baseline normalization
            is also applied, taking the window (-250, 0) ms relative to the
            stimulus onset.

    - Feature classification

        - EEG-Inception: convolutional neural network [1].

    References
    ----------
    [1] Santamaría-Vázquez, E., Martínez-Cagigal, V., Vaquerizo-Villar, F., &
    Hornero, R. (2020). EEG-Inception: A Novel Deep Convolutional Neural Network
    for Assistive ERP-based Brain-Computer Interfaces. IEEE Transactions on
    Neural Systems and Rehabilitation Engineering.
    """
    def __init__(self):
        super().__init__()

    def configure(self, cnn_n_cha=8, fine_tuning=False,
                  shuffle_before_fit=True, validation_split=0.2,
                  init_weights_path=None, gpu_acceleration=True):
        self.settings = {
            'cnn_n_cha': cnn_n_cha,
            'fine_tuning': fine_tuning,
            'shuffle_before_fit': shuffle_before_fit,
            'validation_split': validation_split,
            'init_weights_path': init_weights_path,
            'gpu_acceleration': gpu_acceleration
        }
        # Update state
        self.is_configured = True
        self.is_built = False
        self.is_fit = False

    def build(self):
        # Check errors
        if not self.is_configured:
            raise ValueError('Function configure must be called first!')
        # Only import deep learning models if necessary
        from medusa.deep_learning_models import EEGInceptionv1
        # Preprocessing (bandpass IIR filter [0.5, 45] Hz + CAR)
        self.add_method('prep_method',
                        StandardPreprocessing(cutoff=(0.5, 45)))
        # Feature extraction (epochs [0, 1000] ms + resampling to 128 Hz)
        self.add_method('ext_method',
                        StandardFeatureExtraction(
                            target_fs=128, concatenate_channels=False)
        )
        # Feature classification
        clf = EEGInceptionv1(
            input_time=1000,
            fs=128,
            n_cha=self.settings['cnn_n_cha'],
            filters_per_branch=8,
            scales_time=(500, 250, 125),
            dropout_rate=0.25,
            activation='elu', n_classes=2,
            learning_rate=0.001,
            gpu_acceleration=self.settings['gpu_acceleration'])
        if self.settings['init_weights_path'] is not None:
            clf.load_weights(self.settings['init_weights_path'])
        self.add_method('clf_method', clf)
        # Update state
        self.is_built = True
        self.is_fit = False

    def fit_dataset(self, dataset, **kwargs):
        # Check errors
        if not self.is_built:
            raise ValueError('Function build must be called first!')
        if dataset.channel_set.n_cha != self.settings['cnn_n_cha']:
            raise ValueError('The number of channels of the model and the '
                             'dataset does not match!')
        # Preprocessing
        dataset = self.get_inst('prep_method').fit_transform_dataset(dataset)
        # Extract features
        x, x_info = self.get_inst('ext_method').transform_dataset(dataset)
        if self.check_for_non_control_data(x_info['control_state_labels'],
                                           x_info['control_state_target'],
                                           throw_warning=True):
            x, x_info = self.get_control_data(x, x_info)
        # Classification
        self.get_inst('clf_method').fit(
            x, x_info['erp_labels'],
            fine_tuning=self.settings['fine_tuning'],
            shuffle_before_fit=self.settings['shuffle_before_fit'],
            validation_split=self.settings['validation_split'],
            **kwargs)
        y_pred = self.get_inst('clf_method').predict_proba(x)[:, 1]
        # Command decoding
        spell_result, spell_result_per_seq, __ = decode_commands(
            scores=y_pred,
            paradigm_conf=x_info['paradigm_conf'],
            run_idx=x_info['run_idx'],
            trial_idx=x_info['trial_idx'],
            matrix_idx=x_info['matrix_idx'],
            level_idx=x_info['level_idx'],
            unit_idx=x_info['unit_idx'],
            sequence_idx=x_info['sequence_idx'],
            group_idx=x_info['group_idx'],
            batch_idx=x_info['batch_idx']
        )
        # Spell accuracy per seq
        spell_acc_per_seq = command_decoding_accuracy_per_seq(
            spell_result_per_seq,
            x_info['spell_target']
        )
        cmd_assessment = {
            'x': x,
            'x_info': x_info,
            'y_pred': y_pred,
            'spell_result': spell_result,
            'spell_result_per_seq': spell_result_per_seq,
            'spell_acc_per_seq': spell_acc_per_seq
        }
        # Save variables
        self.channel_set = dataset.channel_set
        # Update state
        self.is_fit = True
        return cmd_assessment

    def predict(self, times, signal, fs, channel_set, exp_info, **kwargs):
        # Check errors
        if not self.is_fit:
            raise ValueError('Function fit_dataset must be called first!')
        # Check channel set
        if self.channel_set != channel_set:
            warnings.warn('The channel set is not the same that was used to '
                          'fit the model. Be careful!')
        # Preprocessing
        signal = self.get_inst('prep_method').fit_transform_signal(signal, fs)
        # Extract features
        x = self.get_inst('ext_method').transform_signal(times, signal, fs,
                                                         exp_info['onsets'])
        # Classification
        y_pred = self.get_inst('clf_method').predict_proba(x)[:, 1]
        # Command decoding
        spell_result, spell_result_per_seq, __ = decode_commands(
            scores=y_pred,
            paradigm_conf=exp_info['paradigm_conf'],
            run_idx=exp_info['run_idx'],
            trial_idx=exp_info['trial_idx'],
            matrix_idx=exp_info['matrix_idx'],
            level_idx=exp_info['level_idx'],
            unit_idx=exp_info['unit_idx'],
            sequence_idx=exp_info['sequence_idx'],
            group_idx=exp_info['group_idx'],
            batch_idx=exp_info['batch_idx']
        )
        cmd_decoding = {
            'x': x,
            'x_info': exp_info,
            'y_pred': y_pred,
            'spell_result': spell_result,
            'spell_result_per_seq': spell_result_per_seq
        }
        return cmd_decoding


class CSDModelEEGInception(ERPSpellerModel):
    """Control state detection model for ERP-based spellers model based on
    EEG-Inception, a powerful deep convolutional neural network specifically
    developed for BCI applications [1]. More information about this control
    state detection method can be found in [2].

    Dataset features:

    - Sample rate of the signals > 128 Hz. The model can handle recordings
        with different sample rates.
    - Recommended channels: ['Fz', 'Cz', 'Pz', 'P3', 'P4', 'PO7', 'PO8', 'Oz'].

    Processing pipeline:

    - Preprocessing:

        - IIR Filter (order=5, cutoff=(0.5, 45) Hz: unlike FIR filters, IIR
            filters are quick and can be applied in small signal chunks. Thus,
            they are the preferred method for frequency filter in online systems
        - Common average reference (CAR): widely used spatial filter that
            increases the signal-to-noise ratio of the ERPs.
    - Feature extraction:

        - Epochs (window=(0, 1000) ms, resampling to 128 HZ): the epochs of
            signal are extract for each stimulation. Baseline normalization
            is also applied, taking the window (-250, 0) ms relative to the
            stimulus onset.

    - Feature classification

        - EEG-Inception: convolutional neural network

    References
    ----------
    [1] Santamaría-Vázquez, E., Martínez-Cagigal, V., Vaquerizo-Villar, F., &
    Hornero, R. (2020). EEG-Inception: A Novel Deep Convolutional Neural Network
    for Assistive ERP-based Brain-Computer Interfaces. IEEE Transactions on
    Neural Systems and Rehabilitation Engineering.

    [2] Eduardo Santamaría-Vázquez, Víctor Martínez-Cagigal, Sergio
    Pérez-Velasco, Diego Marcos-Martínez, Roberto Hornero, Robust Asynchronous
    Control of ERP-Based Brain-Computer Interfaces using Deep Learning,
    Computer Methods and Programs in Biomedicine, vol. 215, Marzo, 2022
    """
    def __init__(self):
        super().__init__()

    def configure(self, cnn_n_cha=8, fine_tuning=False,
                  shuffle_before_fit=True, validation_split=0.2,
                  init_weights_path=None, gpu_acceleration=True):
        self.settings = {
            'cnn_n_cha': cnn_n_cha,
            'fine_tuning': fine_tuning,
            'shuffle_before_fit': shuffle_before_fit,
            'validation_split': validation_split,
            'init_weights_path': init_weights_path,
            'gpu_acceleration': gpu_acceleration
        }
        # Update state
        self.is_configured = True
        self.is_built = False
        self.is_fit = False

    def build(self):
        # Check errors
        if not self.is_configured:
            raise ValueError('Function configure must be called first!')
        # Only import deep learning models if necessary
        from medusa.deep_learning_models import EEGInceptionv1
        # Preprocessing (bandpass IIR filter [0, 10] Hz + CAR)
        self.add_method('prep_method',
                        StandardPreprocessing(cutoff=(0.5, 45)))
        # Feature extraction (epochs [0, 1000] ms + resampling to 128 Hz)
        self.add_method(
            'ext_method',
            StandardFeatureExtraction(target_fs=128,
                                      concatenate_channels=False)
        )
        # Feature classification
        clf = EEGInceptionv1(
            input_time=1000,
            fs=128,
            n_cha=self.settings['cnn_n_cha'],
            filters_per_branch=8,
            scales_time=(500, 250, 125),
            dropout_rate=0.25,
            activation='elu', n_classes=2,
            learning_rate=0.001,
            gpu_acceleration=self.settings['gpu_acceleration'])
        if self.settings['init_weights_path'] is not None:
            clf.load_weights(self.settings['init_weights_path'])
        self.add_method('clf_method', clf)
        # Update state
        self.is_built = True
        self.is_fit = False

    def fit_dataset(self, dataset, **kwargs):
        # Check errors
        if not self.is_built:
            raise ValueError('Function build must be called first!')
        if dataset.channel_set.n_cha != self.settings['cnn_n_cha']:
            raise ValueError('The number of channels of the model and the '
                             'dataset does not match!')
        # Preprocessing
        dataset = self.get_inst('prep_method').fit_transform_dataset(dataset)

        # Feature extraction
        x, x_info = self.get_inst('ext_method').transform_dataset(dataset)

        # Check errors
        if np.all(np.array(x_info['control_state_labels']) == 0) or \
            np.all(np.array(x_info['control_state_labels']) == 1):
            raise ValueError('The dataset does not contain examples of '
                             'different control states')
        # Classification
        self.get_inst('clf_method').fit(
            x, x_info['control_state_labels'],
            fine_tuning=self.settings['fine_tuning'],
            shuffle_before_fit=self.settings['shuffle_before_fit'],
            validation_split=self.settings['validation_split'],
            **kwargs)
        y_pred = self.get_inst('clf_method').predict_proba(x)[:, 1]

        # Control state detection
        csd_result, csd_result_per_seq, scores = detect_control_state(
            y_pred, x_info['run_idx'],
            x_info['trial_idx'],
            x_info['sequence_idx']
        )

        # Control state accuracy
        csd_acc_per_seq = control_state_detection_accuracy_per_seq(
            csd_result_per_seq,
            x_info['control_state_target']
        )

        csd_assessment = {
            'x': x,
            'x_info': x_info,
            'y_pred': y_pred,
            'control_state_result': csd_result,
            'control_state_result_per_seq': csd_result_per_seq,
            'control_state_acc_per_seq': csd_acc_per_seq,
        }
        # Save info
        self.channel_set = dataset.channel_set
        # Update state
        self.is_fit = True
        return csd_assessment

    def predict(self, times, signal, fs, channel_set, x_info, **kwargs):
        # Check errors
        if not self.is_fit:
            raise ValueError('Function fit_dataset must be called first!')
        # Check channel set
        if self.channel_set != channel_set:
            warnings.warn('The channel set is not the same that was used to '
                          'fit the model. Be careful!')
        # Preprocessing
        signal = self.get_inst('prep_method').fit_transform_signal(signal, fs)
        # Extract features
        x = self.get_inst('ext_method').transform_signal(times, signal, fs,
                                                         x_info['onsets'])
        # Classification
        y_pred = self.get_inst('clf_method').predict_proba(x)[:, 1]
        # Control state detection
        csd_result, csd_result_per_seq, __ = detect_control_state(
            y_pred, x_info['run_idx'],
            x_info['trial_idx'],
            x_info['sequence_idx']
        )

        cs_detection = {
            'x': x,
            'x_info': x_info,
            'y_pred': y_pred,
            'control_state_result': csd_result,
            'control_state_result_per_seq': csd_result_per_seq,
        }
        return cs_detection