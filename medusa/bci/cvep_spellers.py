"""
In this module you will find useful functions and classes to operate with data
recorded using spellers based on code-modulated visual evoked potentials
(c-VEP), which are widely used by the BCI community. Enjoy!

@author: Víctor Martínez-Cagigal
"""
import medusa as mds
from medusa import components
from medusa import meeg
from medusa import spatial_filtering as sf
from medusa import epoching as ep
from medusa import classification_utils as clf_utils
import copy, warnings
import itertools

from abc import ABC, abstractmethod
import numpy as np
from tqdm import tqdm

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

LFSR_PRIMITIVE_POLYNOMIALS = \
    {
        'base': {
            2: {
                'order': {
                    2: [1, 1],
                    3: [1, 0, 1],
                    4: [1, 0, 0, 1],
                    5: [0, 1, 0, 0, 1],
                    6: [0, 0, 0, 0, 1, 1],
                    7: [0, 0, 0, 0, 0, 1, 1],
                    8: [1, 1, 0, 0, 0, 0, 1, 1],
                    9: [0, 0, 0, 1, 0, 0, 0, 0, 1],
                    10: [0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
                    11: [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                    12: [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1],
                    13: [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1],
                    14: [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1],
                    15: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                    16: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                    17: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                    18: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                    19: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1,
                         1],
                    20: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                         0, 0],
                    21: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 1, 0],
                    22: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 1],
                    23: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         1, 0, 0, 0, 0],
                    24: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                         0, 0, 0, 0, 1, 1],
                    25: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 1, 0, 0],
                    26: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 1, 0, 0, 0, 1, 1],
                    27: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 1, 0, 0, 1, 1],
                    28: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                    29: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                    30: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
                }
            },
            3: {
                'order': {
                    2: [2, 1],
                    3: [0, 1, 2],
                    4: [0, 0, 2, 1],
                    5: [0, 0, 0, 1, 2],
                    6: [0, 0, 0, 0, 2, 1],
                    7: [0, 0, 0, 0, 2, 1, 2],
                }
            },
            5: {
                'order': {
                    2: [4, 3],
                    3: [0, 2, 3],
                    4: [0, 4, 3, 3],
                }
            },
            7: {
                'order': {
                    2: [1, 4]
                }
            },
            11: {
                'order': {
                    2: [1, 3]
                }
            },
            13: {
                'order': {
                    2: [1, 11]
                }
            }
        }
    }


# --------------------------- c-VEP DATA MANAGEMENT -------------------------- #
class CVEPSpellerData(components.ExperimentData):
    """Experiment info class for c-VEP-based spellers. It supports nested
    multi-level paradigms. This unified class can be used to represent a run
    of every c-VEP stimulation paradigm designed to date, and is the expected
    class for feature extraction and command decoding functions of the module
    medusa.bci.cvep_paradigms. It is complicated, but powerful so.. use it well!
    """

    def __init__(self, mode, paradigm_conf, commands_info, onsets, command_idx,
                 unit_idx, level_idx, matrix_idx, cycle_idx, trial_idx,
                 spell_result, fps_resolution, spell_target=None,
                 raster_events=None, **kwargs):

        # Check errors
        if mode not in ('train', 'test'):
            raise ValueError('Unknown mode. Possible values {train, test}')

        # Standard attributes
        self.mode = mode
        self.paradigm_conf = paradigm_conf
        self.commands_info = commands_info
        self.onsets = onsets
        self.command_idx = command_idx
        self.unit_idx = unit_idx
        self.level_idx = level_idx
        self.matrix_idx = matrix_idx
        self.cycle_idx = cycle_idx
        self.trial_idx = trial_idx
        self.spell_result = spell_result
        self.fps_resolution = fps_resolution
        self.spell_target = spell_target
        self.raster_events = raster_events

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


class CVEPSpellerDataset(components.Dataset):
    """ This class inherits from medusa.data_structures.Dataset, increasing
    its functionality for datasets with data from c-VEP-based spellers. It
    provides common ground for the rest of functions in the module.
    """

    def __init__(self, channel_set, fs=None, biosignal_att_key='eeg',
                 experiment_att_key='cvepspellerdata', experiment_mode=None,
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
            Name of the attribute containing the target biosignal that will be
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
            'command_idx': {
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
            'cycle_idx': {
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
            }
        }

        if experiment_mode == 'train':
            default_track_attributes_train = {
                'spell_target': {
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
        """Creates a standard consistency checker for c-VEP datasets

        Returns
        -------
        checker : data_structures.ConsistencyChecker
            Standard consistency checker for c-VEP feature extraction
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
                         'type': CVEPSpellerData}
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


def decode_commands_from_events(event_scores, commands_info, event_run_idx,
                                event_trial_idx, event_cycle_idx):
    """Command decoder for c-VEP-based spellers based on the bitwise
    reconstruction paradigm (BWR), i.e., models that predict the command
    sequence stimulus by stimulus.

    ToDo: allow multi-matrix paradigms with different number of levels. See
        module erp_based_spellers for reference.

    Parameters
    ----------
    event_scores : list or np.ndarray
        Array with the score for each stimulation
    commands_info : list or np.ndarray
        Array containing the unified speller matrix structure with shape
        [n_runs x n_matrices x n_units x n_groups x n_batches x
        n_commands/batch]. All ERP-based speller paradigms can be adapted to
        this format and use this function for command decoding. See
        ERPSpellerData class for more info.
    event_run_idx : list or numpy.ndarray [n_stim x 1]
        Index of the run for each stimulation. This variable is automatically
        retrieved by function extract_erp_features_from_dataset as part of
        the track info dict. The run indexes must be related to
        paradigm_conf, keeping the same order. Therefore,
        paradigm_conf[np.unique(run_idx)[0]] must retrieve the paradigm
        configuration of run 0.
    event_trial_idx : list or numpy.ndarray [n_stim x 1]
        Index of the trial for each stimulation. A trial represents
        the selection of a final command. Depending on the number of levels,
        the final selection takes N intermediate selections.
    event_cycle_idx : list or numpy.ndarray [n_stim x 1]
        Index of the sequence for each stimulation. A sequence
        represents a round of stimulation: all commands have been
        highlighted 1 time. This class support dynamic stopping in
        different levels.

    Returns
    -------
    selected_commands: list
        Selected command for each trial considering all sequences of
        stimulation. Each command is organized in an array [matrix_idx,
        command_id]. Take into account that the command ids are unique for each
        matrix, and therefore only the command of the last level should be
        useful to take action. Shape [n_runs x n_trials x n_levels x 2]
    selected_commands_per_cycle: list
        Selected command for each trial and sequence of stimulation. The
        fourth dimension of the array contains [matrix_idx, command_id]. To
        calculate the command for each sequence, it takes into account the
        scores of all the previous sequences as well. Shape [n_runs x
        n_trials x n_levels x n_cycles x 2]
    cmd_scores_per_cycle:
        Scores for each command per cycle. Shape [n_runs x n_trials x
        n_levels x n_cycles x n_commands x 1]. The score of each cycle
        is calculated using all the previous cycles as well.
    """
    # Decode commands
    selected_commands = list()
    selected_commands_per_cycle = list()
    scores = list()
    for r, run in enumerate(np.unique(event_run_idx)):
        # Get run data
        run_event_scores = event_scores[event_run_idx == run]
        run_event_cycle_idx = event_cycle_idx[event_run_idx == run]
        run_event_trial_idx = event_trial_idx[event_run_idx == run]
        # Initialize
        run_selected_commands = list()
        run_selected_commands_per_cycle = list()
        run_cmd_scores = list()
        # Iterate trials
        for t, trial in enumerate(np.unique(run_event_trial_idx)):
            # Get trial data
            trial_event_scores = run_event_scores[
                run_event_trial_idx == trial]
            trial_event_cycle_idx = run_event_cycle_idx[
                run_event_trial_idx == trial]
            # Initialize
            trial_cmd_scores_per_cycle = list()
            trial_selected_commands_per_cycle = list()
            # Iterate cycles
            for c, cycle in enumerate(np.unique(trial_event_cycle_idx)):
                cycle_event_scores = trial_event_scores[
                    trial_event_cycle_idx <= cycle]
                # Get target sequences
                cmd_ids = list()
                cmd_seqs = list()
                for cmd_id, cmd_info in commands_info[r][0].items():
                    cmd_ids.append(cmd_id)
                    cmd_seqs.append(cmd_info['sequence'] * (c + 1))
                # Calculate correlations to all commands
                corr_scores = np.abs(
                    np.corrcoef(cycle_event_scores, cmd_seqs)[0, 1:])
                # Save trial data
                cmd_id = cmd_ids[np.argmax(corr_scores)]
                trial_cmd_scores_per_cycle.append(corr_scores)
                trial_selected_commands_per_cycle.append([0, cmd_id])
            # Save run data
            # ToDo: add another loop for levels
            run_selected_commands.append(
                [trial_selected_commands_per_cycle[-1]])
            run_selected_commands_per_cycle.append(
                [trial_selected_commands_per_cycle])
            run_cmd_scores.append(
                [trial_cmd_scores_per_cycle])
        # Save run data
        selected_commands.append(run_selected_commands)
        selected_commands_per_cycle.append(run_selected_commands_per_cycle)
        scores.append(run_cmd_scores)

    return selected_commands, selected_commands_per_cycle, scores


def command_decoding_accuracy_per_cycle(selected_commands_per_cycle,
                                        target_commands):
    """
    Computes the accuracy of the selected sequence of targets given the
    target

    Parameters
    ----------
    selected_commands_per_cycle: list
        List with the spell result per sequence as given by function
        decode_commands. Shape [n_runs x n_trials x n_levels x n_cycles x 2]
    target_commands: list
        Target commands. Each position contains the matrix index and command
        id per level that identifies the target command of the trial. Shape
        [n_runs x n_trials x n_levels x 2]

    Returns
    -------
    acc_per_cycle : float
        Accuracy of the command decoding stage for each number of cycles
        considered in the analysis. Shape [n_sequences]
    """
    # Check errors
    selected_commands_per_cycle = list(selected_commands_per_cycle)
    target_commands = list(target_commands)
    if len(selected_commands_per_cycle) != len(target_commands):
        raise ValueError('Parameters selected_commands_per_seq and spell_target'
                         'must have the same length.')

    # Compute accuracy per sequence
    bool_result_per_seq = []
    n_seqs = []
    for r in range(len(selected_commands_per_cycle)):
        r_sel_cmd_per_seq = selected_commands_per_cycle[r]
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


# ---------------------------------- MODELS ---------------------------------- #
class CVEPSpellerModel(components.Algorithm):

    def __init__(self):
        """Class constructor
        """
        super().__init__(fit_dataset=['spell_target',
                                      'spell_result_per_cycles',
                                      'spell_acc_per_cycles'],
                         predict=['spell_result',
                                  'spell_result_per_cycles'])
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

    @abstractmethod
    def fit_dataset(self, dataset, **kwargs):
        pass

    @abstractmethod
    def predict_dataset(self, dataset, **kwargs):
        pass

    @abstractmethod
    def predict(self, times, signal, fs, channel_set, x_info, **kwargs):
        pass


class CMDModelBWRLDA(CVEPSpellerModel):
    """Class that uses the bitwise reconstruction (BWR) paradigm with an LDA
    classifier"""

    def __int__(self):
        super().__init__()

    def configure(self, bpf=(7, (1.0, 60.0)), notch=(7, (49.0, 51.0)),
                  w_epoch_t=(0, 500), target_fs=None):
        self.settings = {
            'bpf': bpf,
            'notch': notch,
            'w_epoch_t': w_epoch_t,
            'target_fs': target_fs
        }
        # Update state
        self.is_configured = True
        self.is_built = False
        self.is_fit = False

    def build(self):
        # Preprocessing
        bpf = self.settings['bpf']
        notch = self.settings['notch']
        if notch is not None:
            self.add_method('prep_method', StandardPreprocessing(
                bpf_order=bpf[0], bpf_cutoff=bpf[1],
                notch_order=notch[0], notch_cutoff=notch[1]))
        else:
            self.add_method('prep_method', StandardPreprocessing(
                bpf_order=bpf[0], bpf_cutoff=bpf[1],
                notch_order=None, notch_cutoff=None))

        # Feature extraction
        self.add_method('ext_method', BWRFeatureExtraction(
            w_epoch_t=self.settings['w_epoch_t'],
            target_fs=self.settings['target_fs'],
            w_baseline_t=(-250, 0), norm='z',
            concatenate_channels=True, safe_copy=True))

        # Feature classification
        clf = components.ProcessingClassWrapper(
            LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto'),
            fit=[], predict_proba=['y_pred']
        )
        self.add_method('clf_method', clf)
        # Update state
        self.is_built = True
        self.is_fit = False

    def check_predict_feasibility_signal(self, times, cycle_onsets, fps,
                                         code_len, fs):
        return self.get_inst('ext_method').check_predict_feasibility_signal(
            times, cycle_onsets, fps, code_len, fs)

    def fit_dataset(self, dataset, show_progress_bar=False):
        # Check errors
        if not self.is_built:
            raise ValueError('The model must be built first!')
        # Preprocessing
        dataset = self.get_inst('prep_method').fit_transform_dataset(
            dataset, show_progress_bar=show_progress_bar)
        # Extract features
        x, x_info = self.get_inst('ext_method').transform_dataset(
            dataset, show_progress_bar=show_progress_bar)
        # Classification
        self.get_inst('clf_method').fit(x, x_info['event_cvep_labels'])
        # Save info
        self.channel_set = dataset.channel_set
        # Update state
        self.is_fit = True

    def predict_dataset(self, dataset, show_progress_bar=False):
        # Check errors
        if not self.is_fit:
            raise ValueError('The model must be fitted first!')
        # Preprocessing
        dataset = self.get_inst('prep_method').fit_transform_dataset(
            dataset, show_progress_bar=show_progress_bar)
        # Extract features
        x, x_info = self.get_inst('ext_method').transform_dataset(
            dataset, show_progress_bar=show_progress_bar)
        # Predict
        y_pred = self.get_inst('clf_method').predict_proba(x)[:, 1]
        # Command decoding
        sel_cmd, sel_cmd_per_cycle, scores = decode_commands_from_events(
            event_scores=y_pred,
            commands_info=x_info['commands_info'],
            event_run_idx=x_info['event_run_idx'],
            event_trial_idx=x_info['event_trial_idx'],
            event_cycle_idx=x_info['event_cycle_idx']
        )
        cmd_assessment = {
            'x': x,
            'x_info': x_info,
            'y_pred': y_pred,
            'spell_result': sel_cmd,
            'spell_result_per_cycle': sel_cmd_per_cycle,
        }
        # Spell accuracy
        if dataset.experiment_mode == 'train':
            # Spell accuracy per seq
            spell_acc_per_cycle = command_decoding_accuracy_per_cycle(
                sel_cmd_per_cycle,
                x_info['spell_target']
            )
            cmd_assessment = {
                'spell_acc_per_cycle': spell_acc_per_cycle
            }
        return sel_cmd, cmd_assessment

    def predict(self, times, signal, fs, channel_set, x_info, **kwargs):
        # Check errors
        if not self.is_fit:
            raise ValueError('The model must be fitted first!')
        # Check channel set
        if self.channel_set != channel_set:
            warnings.warn(
                'The channel set is not the same that was used to fit the '
                'model. Be careful!')
        # Pre-processing
        signal = self.get_inst('prep_method').transform_signal(signal=signal)
        # Extract features
        x = self.get_inst('ext_method').transform_signal(
            times, signal, fs, x_info['cycle_onsets'],
            x_info['fps'], x_info['code_len'])
        # Predict
        y_pred = self.get_inst('clf_method').predict_proba(x)[:, 1]
        # Get run_idx, trial_idx and cycle_idx per stimulation event
        event_run_idx = np.repeat(x_info['run_idx'], x_info['code_len'])
        event_trial_idx = np.repeat(x_info['trial_idx'], x_info['code_len'])
        event_cycle_idx = np.repeat(x_info['cycle_idx'], x_info['code_len'])
        # Command decoding
        sel_cmd, sel_cmd_per_cycle, scores = decode_commands_from_events(
            event_scores=y_pred,
            commands_info=x_info['commands_info'],
            event_run_idx=event_run_idx,
            event_trial_idx=event_trial_idx,
            event_cycle_idx=event_cycle_idx
        )
        return sel_cmd, sel_cmd_per_cycle, scores


class CMDModelBWREEGInception(CVEPSpellerModel):
    """Class that uses the bitwise reconstruction (BWR) paradigm with an
    EEG-Inception model """

    def __int__(self):
        super().__init__()

    def configure(self, bpf=(7, (1.0, 60.0)), notch=(7, (49.0, 51.0)),
                  w_epoch_t=(0, 500), target_fs=200, n_cha=16,
                  filters_per_branch=12, scales_time=(250, 125, 62.5),
                  dropout_rate=0.15, activation='elu', n_classes=2,
                  learning_rate=0.001, batch_size=256,
                  max_training_epochs=500, validation_split=0.1,
                  shuffle_before_fit=True):
        self.settings = {
            'bpf': bpf,
            'notch': notch,
            'w_epoch_t': w_epoch_t,
            'target_fs': target_fs,
            'n_cha': n_cha,
            'filters_per_branch': filters_per_branch,
            'scales_time': scales_time,
            'dropout_rate': dropout_rate,
            'activation': activation,
            'n_classes': n_classes,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'max_training_epochs': max_training_epochs,
            'validation_split': validation_split,
            'shuffle_before_fit': shuffle_before_fit
        }
        # Update state
        self.is_configured = True
        self.is_built = False
        self.is_fit = False

    def build(self):
        # Preprocessing
        bpf = self.settings['bpf']
        notch = self.settings['notch']
        if notch is not None:
            self.add_method('prep_method', StandardPreprocessing(
                bpf_order=bpf[0], bpf_cutoff=bpf[1],
                notch_order=notch[0], notch_cutoff=notch[1]))
        else:
            self.add_method('prep_method', StandardPreprocessing(
                bpf_order=bpf[0], bpf_cutoff=bpf[1],
                notch_order=None, notch_cutoff=None))

        # Feature extraction
        self.add_method('ext_method', BWRFeatureExtraction(
            w_epoch_t=self.settings['w_epoch_t'],
            target_fs=self.settings['target_fs'],
            w_baseline_t=(-250, 0), norm='z',
            concatenate_channels=False, safe_copy=True))

        # Feature classification
        from medusa.deep_learning_models import EEGInceptionv1
        input_time = \
            self.settings['w_epoch_t'][1] - self.settings['w_epoch_t'][0]
        clf = EEGInceptionv1(
            input_time=input_time,
            fs=self.settings['target_fs'],
            n_cha=self.settings['n_cha'],
            filters_per_branch=self.settings['filters_per_branch'],
            scales_time=self.settings['scales_time'],
            dropout_rate=self.settings['dropout_rate'],
            activation=self.settings['activation'],
            n_classes=self.settings['n_classes'],
            learning_rate=self.settings['learning_rate'])
        self.add_method('clf_method', clf)

        # Update state
        self.is_built = True
        self.is_fit = False

    def check_predict_feasibility_signal(self, times, cycle_onsets, fps,
                                         code_len, fs):
        return self.get_inst('ext_method').check_predict_feasibility_signal(
            times, cycle_onsets, fps, code_len, fs)

    def fit_dataset(self, dataset, show_progress_bar=False):
        # Check errors
        if not self.is_built:
            raise ValueError('The model must be built first!')
        # Preprocessing
        dataset = self.get_inst('prep_method').fit_transform_dataset(
            dataset, show_progress_bar=show_progress_bar)
        # Extract features
        x, x_info = self.get_inst('ext_method').transform_dataset(
            dataset, show_progress_bar=show_progress_bar)
        # Classification
        self.get_inst('clf_method').fit(
            x, x_info['event_cvep_labels'],
            shuffle_before_fit=self.settings['shuffle_before_fit'],
            epochs=self.settings['max_training_epochs'],
            validation_split=self.settings['validation_split'],
            batch_size=self.settings['batch_size'])
        # Save info
        self.channel_set = dataset.channel_set
        # Update state
        self.is_fit = True

    def predict_dataset(self, dataset, show_progress_bar=False):
        # Check errors
        if not self.is_fit:
            raise ValueError('The model must be fitted first!')
        # Preprocessing
        dataset = self.get_inst('prep_method').fit_transform_dataset(
            dataset, show_progress_bar=show_progress_bar)
        # Extract features
        x, x_info = self.get_inst('ext_method').transform_dataset(
            dataset, show_progress_bar=show_progress_bar)
        # Predict
        y_pred = self.get_inst('clf_method').predict_proba(x)
        y_pred = clf_utils.categorical_labels(y_pred)
        # Command decoding
        sel_cmd, sel_cmd_per_cycle, scores = decode_commands_from_events(
            event_scores=y_pred,
            commands_info=x_info['commands_info'],
            event_run_idx=x_info['event_run_idx'],
            event_trial_idx=x_info['event_trial_idx'],
            event_cycle_idx=x_info['event_cycle_idx']
        )
        cmd_assessment = {
            'x': x,
            'x_info': x_info,
            'y_pred': y_pred,
            'spell_result': sel_cmd,
            'spell_result_per_cycle': sel_cmd_per_cycle,
        }
        # Spell accuracy
        if dataset.experiment_mode == 'train':
            # Spell accuracy per seq
            spell_acc_per_cycle = command_decoding_accuracy_per_cycle(
                sel_cmd_per_cycle,
                x_info['spell_target']
            )
            cmd_assessment = {
                'spell_acc_per_cycle': spell_acc_per_cycle
            }
        return sel_cmd, cmd_assessment

    def predict(self, times, signal, fs, channel_set, x_info, **kwargs):
        # Check errors
        if not self.is_fit:
            raise ValueError('The model must be fitted first!')
        # Check channel set
        if self.channel_set != channel_set:
            warnings.warn(
                'The channel set is not the same that was used to fit the '
                'model. Be careful!')
        # Pre-processing
        signal = self.get_inst('prep_method').transform_signal(signal=signal)
        # Extract features
        x = self.get_inst('ext_method').transform_signal(
            times, signal, fs, x_info['cycle_onsets'],
            x_info['fps'], x_info['code_len'])
        # Predict
        y_pred = self.get_inst('clf_method').predict_proba(x)
        y_pred = clf_utils.categorical_labels(y_pred)
        # Get run_idx, trial_idx and cycle_idx per stimulation event
        event_run_idx = np.repeat(x_info['run_idx'], x_info['code_len'])
        event_trial_idx = np.repeat(x_info['trial_idx'], x_info['code_len'])
        event_cycle_idx = np.repeat(x_info['cycle_idx'], x_info['code_len'])
        # Command decoding
        sel_cmd, sel_cmd_per_cycle, scores = decode_commands_from_events(
            event_scores=y_pred,
            commands_info=x_info['commands_info'],
            event_run_idx=event_run_idx,
            event_trial_idx=event_trial_idx,
            event_cycle_idx=event_cycle_idx
        )
        return sel_cmd, sel_cmd_per_cycle, scores


class CVEPModelCircularShifting(components.Algorithm):

    def __init__(self, bpf=[[7, (1.0, 30.0)]], notch=[7, (49.0, 51.0)],
                 art_rej=None, correct_raster_latencies=False, *args, **kwargs):
        super().__init__()

        if len(bpf) == 1:
            if notch is not None:
                self.add_method('prep_method', StandardPreprocessing(
                    bpf_order=bpf[0][0], bpf_cutoff=bpf[0][1],
                    notch_order=notch[0], notch_cutoff=notch[1]))
                max_order = max(bpf[0][0], notch[0])
            else:
                self.add_method('prep_method', StandardPreprocessing(
                    bpf_order=bpf[0][0], bpf_cutoff=bpf[0][1],
                    notch_order=None, notch_cutoff=None))
                max_order = bpf[0][0]
        else:
            filter_bank = []
            max_order = 0
            for i in range(len(bpf)):
                filter_bank.append({
                    'order': bpf[i][0],
                    'cutoff': bpf[i][1],
                    'btype': 'bandpass'
                })
                max_order = bpf[i][0] if bpf[i][0] > max_order else max_order
            if notch is not None:
                self.add_method('prep_method', FilterBankPreprocessing(
                    filter_bank=filter_bank, notch_order=notch[0],
                    notch_cutoff=notch[1]))
                max_order = max(max_order, notch[0])
            else:
                self.add_method('prep_method', FilterBankPreprocessing(
                    filter_bank=filter_bank, notch_order=None,
                    notch_cutoff=None))

        # Feature extraction and classification (circular shifting)
        self.add_method('clf_method', CircularShiftingClassifier(
            art_rej=art_rej,
            correct_raster_latencies=correct_raster_latencies,
            extra_epoch_samples=3 * max_order
        ))

        # Early stopping
        self.add_method('es_method', CircularShiftingAsyncESExtension(
            predict_by_cycles_callback=self.get_inst(
                'clf_method').predict_cycles
        ))

    def check_predict_feasibility(self, dataset):
        return self.get_inst('clf_method')._is_predict_feasible(dataset)

    def check_predict_feasibility_signal(self, times, onsets, fs):
        return self.get_inst('clf_method')._is_predict_feasible_signal(
            times, onsets, fs)

    def fit_dataset(self, dataset, roll_targets=False, **kwargs):
        # Safe copy
        data = copy.deepcopy(dataset)

        # Preprocessing
        data = self.get_inst('prep_method').fit_transform_dataset(
            dataset=data,
            show_progress_bar=True
        )

        # Feature extraction and classification
        fitted_info_clf = self.get_inst('clf_method').fit_dataset(
            dataset=data,
            show_progress_bar=True,
            roll_targets=roll_targets
        )

        return fitted_info_clf

    def predict_dataset(self, dataset):
        # Safe copy
        data = copy.deepcopy(dataset)

        # Preprocessing
        data = self.get_inst('prep_method').transform_dataset(
            dataset=data,
            show_progress_bar=True
        )

        # Feature extraction and classification
        pred_items = self.get_inst('clf_method').predict_dataset(
            dataset=data,
            show_progress_bar=True
        )

        # Extract the selected items using the maximum number of cycles
        selected_seq = 0  # todo: several sequences in the same matrix
        spell_result = []
        for item in pred_items:
            spell_result.append(
                item[-1][selected_seq]['sorted_cmds'][0]['label'])

        # Extract the selected items depending on the number of cycles
        spell_result_per_cycle = []
        for item in pred_items:
            trial_result_per_cycle = {}
            for nc, pred in enumerate(item):
                trial_result_per_cycle[nc] = pred[selected_seq][
                    'sorted_cmds'][0]['label']
            spell_result_per_cycle.append(trial_result_per_cycle)

        # Create the decoding dictionary
        cmd_decoding = {
            'spell_result': spell_result,
            'spell_result_per_cycle': spell_result_per_cycle,
            'items_by_no_cycle': pred_items
        }
        return cmd_decoding

    def predict(self, times, signal, trial_idx, exp_data, sig_data,
                return_all_cycles=True):
        # Safe copy
        times_ = copy.deepcopy(times)
        signal_ = copy.deepcopy(signal)
        trial_idx_ = copy.deepcopy(trial_idx)
        exp_data_ = copy.deepcopy(exp_data)
        sig_data_ = copy.deepcopy(sig_data)

        # Preprocessing
        signal_ = self.get_inst('prep_method').transform_signal(
            signal=signal_
        )

        # Feature extraction and classification
        pred_item_by_no_cycles = self.get_inst('clf_method').predict(
            times_, signal_, trial_idx_, exp_data_, sig_data_,
            return_all_cycles=return_all_cycles
        )

        # Extract the selected label using the maximum number of cycles
        selected_seq = 0  # todo: several sequences in the same matrix
        spell_result = pred_item_by_no_cycles[-1][selected_seq][
            'sorted_cmds'][0]['label']

        # Extract the selected label depending on the number of cycles
        spell_result_per_cycle = {}
        for nc, pred in enumerate(pred_item_by_no_cycles):
            spell_result_per_cycle[nc] = pred[selected_seq]['sorted_cmds'][
                0]['label']

        # Create the decoding dictionary
        cmd_decoding = {
            'spell_result': spell_result,
            'spell_result_per_cycle': spell_result_per_cycle,
            'items_by_no_cycle': pred_item_by_no_cycles
        }

        return cmd_decoding

    def predict_earlystopping(self, times, signal, trial_idx, exp_data,
                              sig_data, return_all_cycles=True,
                              multi_window=None, ewma_beta=0.95,
                              std=3.75):
        # Safe copy
        times_ = copy.deepcopy(times)
        signal_ = copy.deepcopy(signal)
        trial_idx_ = copy.deepcopy(trial_idx)
        exp_data_ = copy.deepcopy(exp_data)
        sig_data_ = copy.deepcopy(sig_data)

        # Preprocessing
        signal_ = self.get_inst('prep_method').transform_signal(
            signal=signal_
        )

        # Feature extraction and classification
        pred_item_by_no_cycles = self.get_inst('es_method').predict(
            times_, signal_, trial_idx_, exp_data_, sig_data_,
            return_all_cycles=return_all_cycles,
            multi_window=multi_window, ewma_beta=ewma_beta,
            std=std
        )

        # Extract the selected label using the maximum number of cycles
        selected_seq = 0  # todo: several sequences in the same matrix
        spell_result = pred_item_by_no_cycles[-1][selected_seq][
            'sorted_cmds'][0]['label']

        # Extract the selected label depending on the number of cycles
        spell_result_per_cycle = {}
        for nc, pred in enumerate(pred_item_by_no_cycles):
            spell_result_per_cycle[nc] = pred[selected_seq]['sorted_cmds'][
                0]['label']

        # Create the decoding dictionary
        cmd_decoding = {
            'spell_result': spell_result,
            'spell_result_per_cycle': spell_result_per_cycle,
            'items_by_no_cycle': pred_item_by_no_cycles
        }

        return cmd_decoding

    def predict_dataset_earlystopping(self, dataset, multi_window=None,
                                      ewma_beta=0.95, std=3.75):
        # Safe copy
        data = copy.deepcopy(dataset)

        # Preprocessing
        data = self.get_inst('prep_method').transform_dataset(
            dataset=data,
            show_progress_bar=True
        )

        # Feature extraction and classification
        pred_items = self.get_inst('es_method').predict_dataset(
            dataset=data,
            show_progress_bar=True
        )

        # TODO: all of this is incorrect

        # Extract the selected items using the maximum number of cycles and
        # the early stopping
        selected_seq = 0  # todo: several sequences in the same matrix
        spell_result = []
        for item in pred_items:
            spell_result.append(
                item[-1][selected_seq]['sorted_cmds'][0]['label'])

        # Extract the selected items depending on the number of cycles
        spell_result_per_cycle = []
        for item in pred_items:
            trial_result_per_cycle = {}
            for nc, pred in enumerate(item):
                trial_result_per_cycle[nc] = pred[selected_seq][
                    'sorted_cmds'][0]['label']
            spell_result_per_cycle.append(trial_result_per_cycle)

        # Create the decoding dictionary
        cmd_decoding = {
            'spell_result': spell_result,
            'spell_result_per_cycle': spell_result_per_cycle,
            'items_by_no_cycle': pred_items
        }
        return cmd_decoding

    # TODO: delete
    # def must_stop(self, corr_vector, std=3.0):
    #     # Safe copy
    #     corr_vector_ = copy.deepcopy(corr_vector)
    #     return self.get_inst('es_method').check_early_stop(
    #         corr_vector=corr_vector_,
    #         std=std
    #     )


class CVEPModelTRCAGoldCodes(components.Algorithm):

    def __init__(self, bpf=[[7, (1.0, 30.0)]], notch=[7, (49.0, 51.0)],
                 art_rej=None, correct_raster_latencies=False,
                 *args, **kwargs):
        super().__init__()

        if len(bpf) == 1:
            if notch is not None:
                self.add_method('prep_method', StandardPreprocessing(
                    bpf_order=bpf[0][0], bpf_cutoff=bpf[0][1],
                    notch_order=notch[0], notch_cutoff=notch[1]))
            else:
                self.add_method('prep_method', StandardPreprocessing(
                    bpf_order=bpf[0][0], bpf_cutoff=bpf[0][1],
                    notch_order=None, notch_cutoff=None))
        else:
            filter_bank = []
            for i in range(len(bpf)):
                filter_bank.append({
                    'order': bpf[i][0],
                    'cutoff': bpf[i][1],
                    'btype': 'bandpass'
                })
            if notch is not None:
                self.add_method('prep_method', FilterBankPreprocessing(
                    filter_bank=filter_bank, notch_order=notch[0],
                    notch_cutoff=notch[1]))
            else:
                self.add_method('prep_method', FilterBankPreprocessing(
                    filter_bank=filter_bank, notch_order=None,
                    notch_cutoff=None))

        # Feature extraction and classification (circular shifting)
        self.add_method('clf_method', TRCAGoldCodesClassifier(
            art_rej=art_rej,
            correct_raster_latencies=correct_raster_latencies
        ))

        # Early stopping
        self.add_method('es_method', CircularShiftingEarlyStopping())

    def check_predict_feasibility(self, dataset):
        return self.get_inst('clf_method')._is_predict_feasible(dataset)

    def check_predict_feasibility_signal(self, times, onsets, fs):
        return self.get_inst('clf_method')._is_predict_feasible_signal(
            times, onsets, fs)

    def fit_dataset(self, dataset, **kwargs):
        # Safe copy
        data = copy.deepcopy(dataset)

        # Preprocessing
        data = self.get_inst('prep_method').fit_transform_dataset(
            dataset=data,
            show_progress_bar=True
        )

        # Feature extraction and classification
        fitted_info = self.get_inst('clf_method').fit_dataset(
            dataset=data,
            std_epoch_rejection=3.0,
            show_progress_bar=True
        )

        return fitted_info

    def predict_dataset(self, dataset):
        # Safe copy
        data = copy.deepcopy(dataset)

        # Preprocessing
        data = self.get_inst('prep_method').transform_dataset(
            dataset=data,
            show_progress_bar=True
        )

        # Feature extraction and classification
        pred_items = self.get_inst('clf_method').predict_dataset(
            dataset=data,
            show_progress_bar=True
        )

        # Extract the selected items using the maximum number of cycles
        selected_seq = 0  # todo: several sequences in the same matrix
        spell_result = []
        for item in pred_items:
            spell_result.append(
                item[-1][selected_seq]['sorted_cmds'][0]['label'])

        # Extract the selected items depending on the number of cycles
        spell_result_per_cycle = []
        for item in pred_items:
            trial_result_per_cycle = {}
            for nc, pred in enumerate(item):
                trial_result_per_cycle[nc] = pred[selected_seq][
                    'sorted_cmds'][0]['label']
            spell_result_per_cycle.append(trial_result_per_cycle)

        # Create the decoding dictionary
        cmd_decoding = {
            'spell_result': spell_result,
            'spell_result_per_cycle': spell_result_per_cycle,
            'items_by_no_cycle': pred_items
        }
        return cmd_decoding

    def predict(self, times, signal, trial_idx, exp_data, sig_data):
        # Safe copy
        times_ = copy.deepcopy(times)
        signal_ = copy.deepcopy(signal)
        trial_idx_ = copy.deepcopy(trial_idx)
        exp_data_ = copy.deepcopy(exp_data)
        sig_data_ = copy.deepcopy(sig_data)

        # Preprocessing
        signal_ = self.get_inst('prep_method').transform_signal(
            signal=signal_
        )

        # Feature extraction and classification
        pred_item_by_no_cycles = self.get_inst('clf_method').predict(
            times_, signal_, trial_idx_, exp_data_, sig_data_
        )

        # Extract the selected label using the maximum number of cycles
        selected_seq = 0  # todo: several sequences in the same matrix
        spell_result = pred_item_by_no_cycles[-1][selected_seq][
            'sorted_cmds'][0]['label']

        # Extract the selected label depending on the number of cycles
        spell_result_per_cycle = {}
        for nc, pred in enumerate(pred_item_by_no_cycles):
            spell_result_per_cycle[nc] = pred[selected_seq]['sorted_cmds'][
                0]['label']

        # Create the decoding dictionary
        cmd_decoding = {
            'spell_result': spell_result,
            'spell_result_per_cycle': spell_result_per_cycle,
            'items_by_no_cycle': pred_item_by_no_cycles
        }

        return cmd_decoding

    def must_stop(self, corr_vector, std=3.0):
        # Safe copy
        corr_vector_ = copy.deepcopy(corr_vector)
        return self.get_inst('es_method').check_early_stop(
            corr_vector=corr_vector,
            std=std
        )


# ------------------------------- ALGORITHMS -------------------------------- #
class StandardPreprocessing(components.ProcessingMethod):
    """Just the common preprocessing applied in c-VEP-based spellers. Simple,
    quick and effective: frequency IIR band-pass and notch filters
    """

    def __init__(self, bpf_order=7, bpf_cutoff=(0.5, 60.0), notch_order=7,
                 notch_cutoff=(49.0, 51.0)):
        super().__init__(fit_transform_signal=['signal'],
                         fit_transform_dataset=['dataset'])
        # Parameters
        self.bpf_order = bpf_order
        self.bpf_cutoff = bpf_cutoff
        self.notch_order = notch_order
        self.notch_cutoff = notch_cutoff
        self.filt_method = 'sosfiltfilt'

        # Variables
        self.bpf_iir_filter = None
        self.notch_iir_filter = None

    def fit(self, fs):
        """Fits the IIR filters.

        Parameters
        ----------
        fs: float
            Sample rate of the signal.
        """
        # Bandpass
        self.bpf_iir_filter = mds.IIRFilter(order=self.bpf_order,
                                            cutoff=self.bpf_cutoff,
                                            btype='bandpass',
                                            filt_method=self.filt_method)
        self.bpf_iir_filter.fit(fs)
        # Notch
        if self.notch_cutoff is not None:
            self.notch_iir_filter = mds.IIRFilter(order=self.notch_order,
                                                  cutoff=self.notch_cutoff,
                                                  btype='bandstop',
                                                  filt_method=self.filt_method)
            self.notch_iir_filter.fit(fs)

    def transform_signal(self, signal):
        """Transforms an EEG signal applying IIR filterings

        Parameters
        ----------
        signal: np.array or list
            Signal to transform. Shape [n_samples x n_channels]
        """
        signal = self.bpf_iir_filter.transform(signal)
        if self.notch_iir_filter is not None:
            signal = self.notch_iir_filter.transform(signal)
        return signal

    def fit_transform_signal(self, signal, fs):
        """Fits the IIR filters and transforms an EEG signal applying them
        sequentially

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

    def fit_transform_dataset(self, dataset: CVEPSpellerDataset,
                              show_progress_bar=True):
        """Fits the IIR filters and transforms an EEG dataset applying the
        filters sequentially. Each recording is preprocessed independently,
        taking into account possible differences in sample rate.

        Parameters
        ----------
        dataset: CVEPSpellerDataset
            CVEPSpellerDataset with the recordings to be preprocessed.
        show_progress_bar: bool
            Show progress bar
        """
        pbar = None
        if show_progress_bar:
            pbar = tqdm(total=len(dataset.recordings),
                        desc='Fitting preprocessing')
        for rec in dataset.recordings:
            eeg = getattr(rec, dataset.biosignal_att_key)
            eeg.signal = self.fit_transform_signal(eeg.signal, eeg.fs)
            setattr(rec, dataset.biosignal_att_key, eeg)
            if show_progress_bar:
                pbar.update(1)
        if show_progress_bar:
            pbar.close()
        return dataset

    def transform_dataset(self, dataset: CVEPSpellerDataset,
                          show_progress_bar=True):
        pbar = None
        if show_progress_bar:
            pbar = tqdm(total=len(dataset.recordings),
                        desc='Applying preprocessing')
        for rec in dataset.recordings:
            eeg = getattr(rec, dataset.biosignal_att_key)
            eeg.signal = self.transform_signal(eeg.signal)
            setattr(rec, dataset.biosignal_att_key, eeg)
            if show_progress_bar:
                pbar.update(1)
        if show_progress_bar:
            pbar.close()
        return dataset


class BWRFeatureExtraction(components.ProcessingMethod):
    """Feature extraction method designed to extract event-wise epochs from
    c-VEP stimulation paradigms to perform bitwise reconstruction (BWR)
    """

    def __init__(self, w_epoch_t=(0, 500), target_fs=20,
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

    @staticmethod
    def generate_bit_wise_onsets(cycle_onsets, frames_per_second, code_len):
        # Generate bit-wise onsets
        onsets = []
        for o in cycle_onsets:
            onsets += np.linspace(
                o, o + (code_len - 1) / frames_per_second, code_len).astype(
                float).tolist()
        return onsets

    def check_predict_feasibility_signal(self, times, cycle_onsets, fps,
                                         code_len, fs):
        # Generate bit-wise onsets, because, for BWR methods, we need w_epoch_t
        # ms after the last stimulus of the sequence
        bit_wise_onsets = self.generate_bit_wise_onsets(
            cycle_onsets, fps, code_len)
        check = ep.check_epochs_feasibility(
            times, bit_wise_onsets, fs, self.w_epoch_t)
        return True if check == 'ok' else False

    def transform_signal(self, times, signal, fs, cycle_onsets, fps, code_len):
        """Function to extract VEP features from raw signal. It returns a 3D
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
        cycle_onsets : list or numpy.ndarray [n_cycles x 1]
            Timestamps indicating the start of each stimulation cycle
        fps: int
            Frames per second of the screen that presents the stimulation
        code_len: int
            Length of the c-VEP codes

        Returns
        -------
        features : np.ndarray [n_events x n_samples x n_channels]
            Feature array with the epochs of signal
        """
        # Avoid changes in the original signal (this may not be necessary)
        if self.safe_copy:
            signal = signal.copy()
        # Get event-wise onsets
        onsets = self.generate_bit_wise_onsets(cycle_onsets, fps, code_len)
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
        """High level function to easily extract features from EEG recordings
        and save useful info for later processing. Nevertheless, the provided
        functionality has several limitations, and it will not be suitable for
        all cases and processing pipelines. If it does not fit your needs,
        create a custom function iterating the recordings and using
        extract_erp_features, a much more low-level and general function. This
        function does not apply any preprocessing to the signals, this must
        be done before.

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
        track_attributes['event_run_idx'] = {
            'track_mode': 'concatenate',
            'parent': dataset.experiment_att_key
        }
        track_attributes['event_trial_idx'] = {
            'track_mode': 'concatenate',
            'parent': dataset.experiment_att_key
        }
        track_attributes['event_cycle_idx'] = {
            'track_mode': 'concatenate',
            'parent': dataset.experiment_att_key
        }
        if dataset.experiment_mode == 'train':
            track_attributes['event_cvep_labels'] = {
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
                cycle_onsets=rec_exp.onsets,
                fps=rec_exp.fps_resolution,
                code_len=len(rec_exp.commands_info[0]['0']['sequence'])
            )
            features = np.concatenate((features, rec_feat), axis=0) \
                if features is not None else rec_feat

            # Special attributes that need tracking across runs to assure the
            # consistency of the dataset
            rec_exp.run_idx = run_counter * np.ones_like(rec_exp.trial_idx)
            rec_exp.trial_idx = trial_counter + np.array(rec_exp.trial_idx)

            # Event tracking attributes
            seq_len = len(list(rec_exp.commands_info[0].values())[0][
                              'sequence'])
            rec_exp.event_run_idx = np.repeat(rec_exp.run_idx, seq_len)
            rec_exp.event_trial_idx = np.repeat(rec_exp.trial_idx, seq_len)
            rec_exp.event_cycle_idx = np.repeat(rec_exp.cycle_idx, seq_len)

            # Get labels of the individual events as required in BWR method
            if dataset.experiment_mode == 'train':
                rec_exp.event_cvep_labels = np.array([])
                for i, t in enumerate(np.unique(rec_exp.trial_idx)):
                    # ToDo: add another loop for levels
                    target = rec_exp.spell_target[i][0]
                    cmd_mtx = target[0]
                    cmd_id = target[1]
                    cmd_seq = rec_exp.commands_info[cmd_mtx][cmd_id]['sequence']
                    n_cycles = np.array(rec_exp.cycle_idx)[
                                   rec_exp.trial_idx == t][-1] + 1
                    rec_exp.event_cvep_labels = np.concatenate(
                        (rec_exp.event_cvep_labels, cmd_seq * int(n_cycles)),
                        axis=0)

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


class FilterBankPreprocessing(components.ProcessingMethod):
    """Just the common preprocessing applied in c-VEP-based spellers. Simple,
    quick and effective: frequency IIR band-pass and notch filters
    """

    def __init__(self, filter_bank=None, notch_order=7,
                 notch_cutoff=(49.0, 51.0)):
        super().__init__(fit_transform_signal=['signal'],
                         fit_transform_dataset=['dataset'])
        if filter_bank is None:
            filter_bank = [{'order': 7,
                            'cutoff': (8.0, 60.0),
                            'btype': 'bandpass'},
                           {'order': 7,
                            'cutoff': (12.0, 60.0),
                            'btype': 'bandpass'},
                           {'order': 7,
                            'cutoff': (30.0, 60.0),
                            'btype': 'bandpass'},
                           ]

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
                    'btype' not in filter:
                raise ValueError('[FilterBankPreprocessing] Each filter must '
                                 'be a dict() containing the following keys: '
                                 '"order", "cutoff" and "btype"!')

        # Parameters
        self.filter_bank = filter_bank
        self.notch_order = notch_order
        self.notch_cutoff = notch_cutoff
        self.filt_method = 'sosfiltfilt'

        # Variables
        self.filter_bank_iir_filters = None
        self.notch_iir_filter = None

    def fit(self, fs):
        """Fits the IIR filters.

        Parameters
        ----------
        fs: float
            Sample rate of the signal.
        """
        # Filter bank
        self.filter_bank_iir_filters = []
        for filter in self.filter_bank:
            iir = mds.IIRFilter(order=filter['order'],
                                cutoff=filter['cutoff'],
                                btype=filter['btype'],
                                filt_method=self.filt_method)
            iir.fit(fs)
            self.filter_bank_iir_filters.append(iir)

        # Notch
        if self.notch_cutoff is not None:
            self.notch_iir_filter = mds.IIRFilter(order=self.notch_order,
                                                  cutoff=self.notch_cutoff,
                                                  btype='bandstop',
                                                  filt_method=self.filt_method)
            self.notch_iir_filter.fit(fs)

    def transform_signal(self, signal):
        """Transforms an EEG signal applying the filter bank

        Parameters
        ----------
        signal: np.array or list
            Signal to transform. Shape [n_samples x n_channels]
        """
        if self.notch_iir_filter is not None:
            signal = self.notch_iir_filter.transform(signal)
        signals = []
        for filter in self.filter_bank_iir_filters:
            signal_ = signal.copy()
            signals.append(filter.transform(signal_))
        return signals

    def fit_transform_signal(self, signal, fs):
        """Fits the IIR filters and transforms an EEG signal applying them
        sequentially

        Parameters
        ----------
        signal: np.array or list
            Signal to transform. Shape [n_samples x n_channels]
        fs: float
            Sample rate of the signal.
        """
        self.fit(fs)
        signals = self.transform_signal(signal)
        return signals

    def fit_transform_dataset(self, dataset: CVEPSpellerDataset,
                              show_progress_bar=True):
        """Fits the IIR filters and transforms an EEG dataset applying the
        filters sequentially. Each recording is preprocessed independently,
        taking into account possible differences in sample rate.

        Parameters
        ----------
        dataset: CVEPSpellerDataset
            CVEPSpellerDataset with the recordings to be preprocessed.
        show_progress_bar: bool
            Show progress bar
        """
        pbar = None
        if show_progress_bar:
            pbar = tqdm(total=len(dataset.recordings),
                        desc='Fitting preprocessing')
        for rec in dataset.recordings:
            eeg = getattr(rec, dataset.biosignal_att_key)
            eeg.signal = self.fit_transform_signal(eeg.signal, eeg.fs)
            setattr(rec, dataset.biosignal_att_key, eeg)
            if show_progress_bar:
                pbar.update(1)
        if show_progress_bar:
            pbar.close()
        return dataset

    def transform_dataset(self, dataset: CVEPSpellerDataset,
                          show_progress_bar=True):
        pbar = None
        if show_progress_bar:
            pbar = tqdm(total=len(dataset.recordings),
                        desc='Applying preprocessing')
        for rec in dataset.recordings:
            eeg = getattr(rec, dataset.biosignal_att_key)
            eeg.signal = self.transform_signal(eeg.signal)
            setattr(rec, dataset.biosignal_att_key, eeg)
            if show_progress_bar:
                pbar.update(1)
        if show_progress_bar:
            pbar.close()
        return dataset


class CircularShiftingClassifier(components.ProcessingMethod):
    """Standard feature classification method for c-VEP-based spellers.
    Basically, it computes a template for each sequence.
    """

    def __init__(self, correct_raster_latencies=False, art_rej=None,
                 extra_epoch_samples=21, **kwargs):
        """ Class constructor """
        super().__init__(fit_dataset=['templates',
                                      'cca_by_seq'])
        self.fitted = dict()

        self.art_rej = art_rej
        self.correct_raster_latencies = correct_raster_latencies
        self.extra_epoch_samples = extra_epoch_samples

    def _assert_consistency(self, dataset: CVEPSpellerDataset):
        # TODO: this function is not necessary. Use CVEPSpellerDataset
        #  consistency checker instead!
        len_seqs = set()
        fs = set()
        fps_resolution = set()
        unique_seqs_by_run = []
        unique_all_seqs = set()
        is_filter_bank = []
        for rec in dataset.recordings:
            rec_sig = getattr(rec, dataset.biosignal_att_key)
            rec_exp = getattr(rec, dataset.experiment_att_key)
            if rec_exp.mode != 'train':
                raise ValueError('There is at least one CVEPSpellerData '
                                 'instance that was not recording under train '
                                 'mode. Aborting feature extraction...')
            if not hasattr(rec_exp, 'spell_target'):
                raise ValueError('There is at least one CVEPSpellerData '
                                 'instance that has not "spell_target" data. '
                                 'Aborting feature extraction...')
            fps_resolution.add(rec_exp.fps_resolution)
            fs.add(rec_sig.fs)
            unique_seqs = get_unique_sequences_from_targets(rec_exp)
            for seq_ in unique_seqs:
                len_seqs.add(len(seq_))
                unique_all_seqs.add(seq_)
            unique_seqs_by_run.append(unique_seqs)

            if isinstance(rec_sig.signal, list):
                is_filter_bank.append(True)
            else:
                is_filter_bank.append(False)
        if len(len_seqs) > 1:
            raise ValueError('There are sequences with different lengths in '
                             'the CVEPSpellerDataset instance! Aborting feature'
                             ' extraction...')
        if len(fs) > 1:
            raise ValueError('There are CVEPSpellerData instances with '
                             'different sampling rates! Aborting feature '
                             'extraction...')
        if len(fps_resolution) > 1:
            raise ValueError('There are CVEPSpellerData instances with '
                             'different refresh rates! Aborting feature '
                             'extraction...')
        if len(unique_all_seqs) > 1:
            # Check if some sequences are shifted versions of another
            unique_ = list(unique_all_seqs)
            all_combos = np.array(list(itertools.combinations(
                np.arange(0, len(unique_)), 2)))
            for i_comb in range(all_combos.shape[0]):
                s1 = unique_[all_combos[i_comb][0]]
                s2 = unique_[all_combos[i_comb][1]]
                if check_if_shifted(s1, s2):
                    raise ValueError('There are targets that share shifted '
                                     'versions of the same sequence. Solve that'
                                     'before extracting features!')

        if len(np.unique(is_filter_bank)):
            is_filter_bank = is_filter_bank[0]
        else:
            raise ValueError('There are recordings that have filter banks and '
                             'other do not.')
        len_seq = len_seqs.pop()
        fs = fs.pop()
        fps_resolution = fps_resolution.pop()

        return fs, fps_resolution, len_seq, unique_seqs_by_run, is_filter_bank

    def fit_dataset(self, dataset: CVEPSpellerDataset,
                    roll_targets=False, show_progress_bar=True):

        # Error checking
        fs, fps_resolution, len_seq, unique_seqs_by_run, is_filter_bank = \
            self._assert_consistency(dataset)

        # Init progress bar for sequences
        if show_progress_bar:
            pbar = tqdm(total=len(dataset.recordings),
                        desc='Extracting unique sequences')

        # Compute sequence length in milliseconds
        len_epoch_ms = len_seq / fps_resolution * 1000
        len_epoch_sam = int(len_seq / fps_resolution * fs)

        # Get the epochs of each sequence
        epochs_by_seq = {}
        for rec_idx, rec in enumerate(dataset.recordings):
            # Extract recording experiment and biosignal
            rec_exp = getattr(rec, dataset.experiment_att_key)
            rec_sig = getattr(rec, dataset.biosignal_att_key)

            # Filter bank init
            if not is_filter_bank:
                rec_sig.signal = [rec_sig.signal]

            # Get unique sequences for this run
            unique_seqs = unique_seqs_by_run[rec_idx]
            if roll_targets:
                new_unique_seqs = dict()
                for key, value in unique_seqs.items():
                    c_ = rec_exp.command_idx[value[0]]
                    c_lag_ = rec_exp.commands_info[0][str(int(c_))]['lag']
                    c_seq_ = rec_exp.commands_info[0][str(int(c_))]['sequence']
                    new_key = np.roll(c_seq_, c_lag_)
                    new_unique_seqs[tuple(new_key)] = value
                unique_seqs = new_unique_seqs

            # For each filter bank
            for filter_idx, signal in enumerate(rec_sig.signal):

                # Extract epochs
                epochs = mds.get_epochs_of_events(timestamps=rec_sig.times,
                                                  signal=signal,
                                                  onsets=rec_exp.onsets,
                                                  fs=fs,
                                                  w_epoch_t=[0, len_epoch_ms],
                                                  w_baseline_t=None,
                                                  norm=None)

                # Roll targets if training was not made with the 0 lag command
                if roll_targets:
                    for idx_, c_ in enumerate(rec_exp.command_idx):
                        # TODO: nested matrices
                        c_lag_ = rec_exp.commands_info[0][str(int(c_))]['lag']
                        lag_samples = int(
                            np.round(c_lag_ / fps_resolution * fs))
                        # Revert the lag in the epoch
                        epochs[idx_, :, :] = np.roll(
                            epochs[idx_, :, :], lag_samples, axis=0)

                # Organize epochs by sequence
                for seq_, ep_idxs_ in unique_seqs.items():
                    if tuple(seq_) not in epochs_by_seq:
                        epochs_by_seq[tuple(seq_)] = \
                            [None for i in range(len(rec_sig.signal))]
                        epochs_by_seq[tuple(seq_)][filter_idx] = \
                            epochs[ep_idxs_, :, :]
                    else:
                        if epochs_by_seq[tuple(seq_)][filter_idx] is None:
                            epochs_by_seq[tuple(seq_)][filter_idx] = \
                                epochs[ep_idxs_, :, :]
                        else:
                            epochs_by_seq[tuple(seq_)][
                                filter_idx] = np.concatenate((
                                epochs_by_seq[tuple(seq_)][filter_idx],
                                epochs[ep_idxs_, :, :]
                            ), axis=0)

            if show_progress_bar:
                pbar.update(1)

        # Precompute nearest channels for online artifact rejection
        sorted_dist_ch = None
        if self.art_rej is not None:
            sorted_dist_ch = rec_sig.channel_set.sort_nearest_channels()

        # New bar
        if show_progress_bar:
            pbar.close()
            pbar = tqdm(total=len(epochs_by_seq) * len_seq,
                        desc='Creating templates')

        # For each sequence
        seq_dict = dict()
        discarded_epochs = 0
        total_epochs = 0
        for seq_ in epochs_by_seq:

            # For each filter of the bank
            for filter_idx in range(len(epochs_by_seq[seq_])):

                # Offline artifact rejection
                if self.art_rej is not None:
                    epochs_std = np.std(epochs_by_seq[seq_][filter_idx],
                                        axis=1)  # STD per samples
                    ch_std = np.std(epochs_std, axis=0)  # Variation of epochs
                    # For each channel, check if the variation is adequate
                    epoch_to_keep = np.zeros(epochs_std.shape)
                    for i in range(len(ch_std)):
                        epoch_to_keep[:, i] = (
                                (epochs_std[:, i] < (
                                        np.median(epochs_std[:, i]) +
                                        self.art_rej * ch_std[
                                            i])) &
                                (epochs_std[:, i] > (
                                        np.median(epochs_std[:, i]) -
                                        self.art_rej * ch_std[
                                            i]))
                        )
                    # Keep only epochs that are suitable for all channels
                    idx_to_keep = (
                            np.sum(epoch_to_keep, axis=1) == epochs_std.shape[1]
                    )
                    epochs_by_seq[seq_][filter_idx] = \
                        epochs_by_seq[seq_][filter_idx][idx_to_keep, :, :]
                    discarded_epochs += np.sum(idx_to_keep == False)
                    total_epochs += len(idx_to_keep)

                # Canonical Correlation Analysis
                cca = sf.CCA()

                # Reference (main template repeated 'no_cycles' times)
                main_template = np.mean(epochs_by_seq[seq_][filter_idx], axis=0)
                R = np.tile(main_template.T,
                            epochs_by_seq[seq_][filter_idx].shape[0]).T

                # Input data (concatenated epochs)
                X = epochs_by_seq[seq_][filter_idx]
                X = X.reshape((X.shape[0] * X.shape[1], X.shape[2]))

                # Fit CCA and project the main template
                cca.fit(X, R)
                main_template = cca.project(main_template, filter_idx=0,
                                            projection='Wy')
                # Create all possible template shifts
                templates = dict()
                for lag in range(len(seq_)):
                    lag_samples = int(np.round(lag / fps_resolution * fs))
                    lagged_seq = np.roll(seq_, -lag, axis=0)
                    lagged_template = np.roll(main_template, -lag_samples,
                                              axis=0)
                    templates[tuple(lagged_seq)] = lagged_template

                    if show_progress_bar:
                        pbar.update(1)

                # STD by channel (useful for online artifact rejection)
                std_by_channel = np.std(X, axis=0)

                # Store data of each trained sequence
                if tuple(seq_) not in seq_dict:
                    seq_dict[tuple(seq_)] = []
                seq_dict[tuple(seq_)].append(
                    {'cca': cca,
                     'templates': templates,
                     'std_by_channel': std_by_channel,
                     }
                )

        # Store fitted params
        self.fitted = {'sequences': seq_dict,
                       'fs': fs,
                       'fps_resolution': fps_resolution,
                       'len_epoch_ms': len_epoch_ms,
                       'len_epoch_sam': len_epoch_sam,
                       'std_epoch_rejection': self.art_rej,
                       'no_discarded_epochs': discarded_epochs,
                       'no_total_epochs': total_epochs,
                       'sorted_dist_ch': sorted_dist_ch
                       }
        if show_progress_bar:
            pbar.close()

        return self.fitted

    def _is_predict_feasible(self, dataset):
        l_ms = self.fitted['len_epoch_ms'] + \
               np.ceil(self.extra_epoch_samples / dataset.fs)
        for rec in dataset.recordings:
            rec_sig = getattr(rec, dataset.biosignal_att_key)
            rec_exp = getattr(rec, dataset.experiment_att_key)
            feasible = ep.check_epochs_feasibility(timestamps=rec_sig.times,
                                                   onsets=rec_exp.onsets,
                                                   fs=rec_sig.fs,
                                                   t_window=[0, l_ms])
            if feasible != 'ok':
                return False
        return True

    def _is_predict_feasible_signal(self, times, onsets, fs):
        l_ms = self.fitted['len_epoch_ms'] + \
               np.ceil(self.extra_epoch_samples / fs)
        feasible = ep.check_epochs_feasibility(timestamps=times,
                                               onsets=onsets,
                                               fs=fs,
                                               t_window=[0, l_ms])
        if feasible != 'ok':
            return False
        return True

    def _interpolate_epoch(self, epoch, channel_set, bad_channels_idx,
                           no_neighbors=3):
        interp_epoch = epoch.copy()
        # Are all channels bad?
        if len(bad_channels_idx) == len(channel_set.channels):
            print('> Artifact rejection: Cannot interpolate because all '
                  'channels are bad.')
            return interp_epoch

        # For each bad channel
        bad_labels = []
        for i in bad_channels_idx:
            bad_labels.append(channel_set.channels[i]['label'])
        for i, bad_label in enumerate(bad_labels):
            if bad_label not in self.fitted['sorted_dist_ch']:
                raise Exception('Label %s is not present in the EEGChannelSet'
                                ' in which the model was fitted for!'
                                % bad_label)
            # Find the K labels of the nearest neighbors
            sorted_ch = self.fitted['sorted_dist_ch'][bad_label]
            interp_labels = []
            for ch in sorted_ch:
                interp_labels.append(ch["channel"]["label"])
                if len(interp_labels) == no_neighbors:
                    break

            # Interpolate using average
            interp_idxs = channel_set.get_cha_idx_from_labels(interp_labels)
            interp_epoch[:, bad_channels_idx[i]] = \
                np.mean(interp_epoch[:, np.array(interp_idxs)], axis=1)
        print('> Artifact rejection: interpolated %i channels' %
              len(bad_channels_idx))
        return interp_epoch

    def predict(self, times, signal, trial_idx, exp_data, sig_data,
                return_all_cycles=True):
        # For each number of cycles
        pred_item_by_no_cycles = []
        _exp_cycle_idx = np.array(exp_data.cycle_idx)
        no_cycles = np.max(_exp_cycle_idx[np.array(exp_data.trial_idx) ==
                                          trial_idx]).astype(int) + 1

        # Cycles to return
        if return_all_cycles:
            cycles_to_return = np.arange(no_cycles)
        else:
            cycles_to_return = [no_cycles]

        # For each cycle to return
        for nc in cycles_to_return:
            pred_item = self.predict_cycles(
                times, signal, trial_idx, exp_data, sig_data,
                cycle_idxs_to_consider=np.arange(0, nc + 1)
            )
            pred_item_by_no_cycles.append(pred_item)

        return pred_item_by_no_cycles

    def predict_cycles(self, times, signal, trial_idx, exp_data, sig_data,
                       cycle_idxs_to_consider):
        # Parameters
        len_epoch_ms = self.fitted['len_epoch_ms']
        len_epoch_sam = self.fitted['len_epoch_sam']
        fs = self.fitted['fs']
        exp_data.onsets = np.array(exp_data.onsets)

        # Assert filter bank
        if not isinstance(signal, list):
            signal = [signal]
        for seq_, seq_data_ in self.fitted['sequences'].items():
            if len(seq_data_) != len(signal):
                raise ValueError('[CircularShiftingClassifier] Cannot predict '
                                 'if the signal do not have the same number of '
                                 'filter banks than the fitted one!')

        # Identify what are the epochs that must be processed
        idx = np.where(
            np.isin(np.array(exp_data.cycle_idx), cycle_idxs_to_consider) &
            (np.array(exp_data.trial_idx) == trial_idx)
        )[0]

        # Raster latencies?
        raster_dict = None
        if self.correct_raster_latencies:
            possible_onsets_idx = np.where(
                exp_data.raster_events['onset'] <
                exp_data.onsets[idx][-1]
            )[0]
            if possible_onsets_idx.size > 0:
                raster_dict = exp_data.raster_events['event'][
                    possible_onsets_idx[-1]]

        # For each fitted sequence
        pred_items = []
        for seq_, seq_data_ in self.fitted['sequences'].items():

            # For each possible filter bank
            f_corrs = []
            for filter_idx, filter_signal in enumerate(signal):

                # Extract the epochs for that signal, trial and no. cycles
                epochs = mds.get_epochs_of_events(
                    timestamps=times,
                    signal=filter_signal,
                    onsets=exp_data.onsets[idx],
                    fs=fs,
                    w_epoch_t=[0, len_epoch_ms],
                    w_baseline_t=None,
                    norm=None)
                if len(epochs.shape) == 2:
                    # Create a dummy dimension if we have only one epoch
                    epochs = np.expand_dims(epochs, 0)

                # Artifact rejection
                if self.art_rej is not None:
                    epoch_std_by_channel = \
                        np.std(epochs[:, :len_epoch_sam, :], axis=1)
                    for i in range(epoch_std_by_channel.shape[0]):
                        discard_epoch = epoch_std_by_channel[i, :] > \
                                        seq_data_[filter_idx][
                                            'std_by_channel'] \
                                        * self.art_rej
                        if np.any(discard_epoch):
                            # TODO: precompute distance matrix before
                            epochs[i, :len_epoch_sam, :] = \
                                self._interpolate_epoch(
                                    epoch=epochs[i, :len_epoch_sam, :],
                                    channel_set=sig_data.channel_set,
                                    bad_channels_idx=
                                    np.where(discard_epoch)[0],
                                    no_neighbors=3
                                )

                # Average the epochs
                avg = np.mean(epochs[:, :len_epoch_sam, :], axis=0)

                # CCA projection
                x_ = seq_data_[filter_idx]['cca'].project(
                    avg, filter_idx=0, projection='Wy')

                # Correlation coefficients between x_ and the templates
                corrs = []
                seqs = []
                for shift_seq_, template_ in \
                        seq_data_[filter_idx]['templates'].items():
                    # Correct template using raster latencies
                    lat_s = 0
                    if raster_dict is not None:
                        if shift_seq_ in raster_dict:
                            lat_s = int(raster_dict[shift_seq_] * fs)
                    tem_ = np.roll(template_, lat_s)
                    temp_p = np.dot(tem_, x_) / np.sqrt(np.dot(np.dot(
                        tem_, tem_), np.dot(x_, x_)))
                    corrs.append(temp_p)
                    seqs.append(shift_seq_)
                f_corrs.append(corrs)

            # Average the correlations between different filter banks
            corrs = np.mean(np.array(f_corrs), axis=0)
            seqs = np.array(seqs)

            # Sort the sequences by corrs' descending order
            sorted_idx = np.argsort(-corrs)
            sorted_corrs = corrs[sorted_idx]
            sorted_seqs = seqs[sorted_idx, :]

            # Identify the selected command
            sorted_cmds = get_items_by_sorted_sequences(
                experiment=exp_data,
                trial_idx=trial_idx,
                sorted_seqs=sorted_seqs,
                sorted_corrs=sorted_corrs
            )
            pred_item = {
                'sorted_cmds': sorted_cmds,
                'fitted_sequence': seq_,
                'full_corrs': corrs.tolist(),
                'seqs': seqs.tolist()
            }

            # Store the predicted item
            pred_items.append(pred_item)

        return pred_items

    def predict_dataset(self, dataset: CVEPSpellerDataset,
                        show_progress_bar=True):
        # Error detection
        if not self.fitted:
            raise Exception(
                'Cannot predict if circular shifting templates and '
                'CCA projections are not fitted before! Aborting...')
        for rec in dataset.recordings:
            rec_sig = getattr(rec, dataset.biosignal_att_key)
            rec_exp = getattr(rec, dataset.experiment_att_key)
            if rec_sig.fs != self.fitted['fs']:
                raise ValueError('The sampling rate of this test recording '
                                 '(%.2f Hz) is not the same as for the fitted '
                                 'recordings! (%.2f Hz)' %
                                 (rec_sig.fs, self.fitted['fs']))
            if rec_exp.fps_resolution != self.fitted['fps_resolution']:
                raise ValueError('The refresh rate of this test recording '
                                 '(%.2f Hz) is not the same as for the fitted '
                                 'recordings! (%.2f Hz)' %
                                 (rec_exp.fps_resolution,
                                  self.fitted['fps_resolution']))

        if show_progress_bar:
            pbar = tqdm(total=len(dataset.recordings),
                        desc='Predicting dataset')

        # For each recording
        pred_items_by_no_cycles = []
        for rec in dataset.recordings:
            rec_sig = getattr(rec, dataset.biosignal_att_key)
            rec_exp = getattr(rec, dataset.experiment_att_key)

            # For each trial
            for t_idx in np.unique(rec_exp.trial_idx):
                decoding_by_no_cycles = \
                    self.predict(rec_sig.times, rec_sig.signal, t_idx,
                                 rec_exp, rec_sig)
                pred_items_by_no_cycles.append(decoding_by_no_cycles)
            if show_progress_bar:
                pbar.update(1)
        if show_progress_bar:
            pbar.close()
        return pred_items_by_no_cycles


class CircularShiftingAsyncESExtension(components.ProcessingMethod):

    def __init__(self, predict_by_cycles_callback, **kwargs):
        self.predict_by_cycles_callback = predict_by_cycles_callback

    @staticmethod
    def _must_stop(rhos, sel_idx, std=3.75):
        """
            Early stopping method to detect if the selected correlation must
            be sent. The selected item will be sent if its correlation is an
            outlier of the distribution formed by the rest of correlations,
            assumming normality.

            Parameters
            --------------------
            rhos: list() or numpy 1D array
                List of correlations
            sel_idx: int
                Index of the selected correlation
            std: float
                Standard deviation that indicates the threshold that must be
                overcome to send a selection

            Returns
            ---------------------
            tuple(must_stop, probs):
                must_stop is a boolean that determines if the item must be sent.
                probs is a numpy 1D array with the same dimension as rhos
                that indicates, for each correlation, the probability of
                being sent between 0 and 1.
        """
        rhos = np.array(rhos)
        sel_rho = rhos[sel_idx]
        spu_rho = np.delete(rhos, sel_idx)

        # Stop if the selected correlation is an outlier of the spurious dist
        threshold = np.mean(spu_rho) + std * np.std(spu_rho)
        must_stop = sel_rho > threshold

        # Update probabilities of being selected for each correlation value
        probs = threshold - rhos
        probs = 1 - (probs / np.max(probs))
        return must_stop, probs

    def predict(self, times, signal, trial_idx, exp_data, sig_data,
                return_all_cycles=True, multi_window=None, ewma_beta=0.95,
                std=3.75):
        # Get the maximum cycle
        _exp_cycle_idx = np.array(exp_data.cycle_idx)
        mc = np.max(_exp_cycle_idx[np.array(exp_data.trial_idx) ==
                                   trial_idx]).astype(int)
        pred_item_by_no_cycles = list()

        # Return all cycles or just the last one (online) ?
        if return_all_cycles:
            cycles_to_return = np.arange(mc + 1)
        else:
            cycles_to_return = [mc]

        # For each number of cycles
        for nc in cycles_to_return:
            # Get the cycle windows to extract correlations
            if multi_window is None:
                # Consider all cycles in our window
                ind_wins = [list(range(nc, i, -1)) for i in
                            range(nc - 1, -2, -1)]
            else:
                # Consider the multi-window length
                ind_wins = [list(range(nc, i, -1)) for i in
                            range(nc - 1, nc - multi_window - 1, -1)]
                ind_wins = [lst for lst in ind_wins if all(x >= 0 for x in lst)]

            # Predict all windows
            items = list()
            for win in ind_wins:
                pred_item = self.predict_by_cycles_callback(
                    times, signal, trial_idx, exp_data, sig_data,
                    cycle_idxs_to_consider=win)
                items.append(pred_item)

            # Weight using EWMA for each fitted sequence
            n_seq = len(items[0])
            decoding = list()
            for k in range(n_seq):
                ewma_decoding = {
                    'sorted_cmds': None,
                    'fitted_sequence': items[0][k]['fitted_sequence'],
                    'full_corrs': None,
                    'seqs': np.array(items[0][k]['seqs']),
                    'must_stop': None
                }

                # Modify the full correlations
                ewma_corr = np.array(items[0][k]['full_corrs'])
                for i in range(1, len(items)):
                    ewma_corr += ewma_beta * ewma_corr + \
                                 (1 - ewma_beta) * \
                                 np.array(items[i][k]['full_corrs'])
                ewma_decoding['full_corrs'] = ewma_corr

                # Sort the correlations and sequences
                sorted_idx = np.argsort(-ewma_corr)
                sorted_corrs = ewma_corr[sorted_idx]
                sorted_seqs = ewma_decoding['seqs'][sorted_idx, :]

                # Update item correlations and probs
                sorted_cmds = get_items_by_sorted_sequences(
                    experiment=exp_data,
                    trial_idx=trial_idx,
                    sorted_seqs=sorted_seqs,
                    sorted_corrs=sorted_corrs
                )

                # Early stopping
                sel_idx = np.where(sorted_corrs == sorted_cmds[0][
                    'correlation'])[0][0]
                must_stop, probs = self._must_stop(
                    rhos=sorted_corrs,
                    sel_idx=sel_idx,
                    std=std)
                ewma_decoding['must_stop'] = True if must_stop else False

                # Update probs for each item
                for si, seq_ in enumerate(sorted_seqs):
                    for item in sorted_cmds:
                        if np.all(np.array(item['item']['sequence']) == seq_):
                            item['prob'] = probs[si]
                            break
                ewma_decoding['sorted_cmds'] = sorted_cmds

                # Add to the decoding
                decoding.append(ewma_decoding)

            # Add to the big decoding list
            pred_item_by_no_cycles.append(decoding)
        return pred_item_by_no_cycles

    def predict_dataset(self, dataset: CVEPSpellerDataset,
                        show_progress_bar=True, multi_window=None,
                        ewma_beta=0.95, std=3.75):
        if show_progress_bar:
            pbar = tqdm(total=len(dataset.recordings),
                        desc='Predicting dataset')

        # For each recording
        pred_items_by_no_cycles = []
        for rec in dataset.recordings:
            rec_sig = getattr(rec, dataset.biosignal_att_key)
            rec_exp = getattr(rec, dataset.experiment_att_key)

            # For each trial
            for t_idx in np.unique(rec_exp.trial_idx):
                decoding_by_no_cycles = \
                    self.predict(rec_sig.times, rec_sig.signal, t_idx,
                                 rec_exp, rec_sig, multi_window=multi_window,
                                 ewma_beta=ewma_beta, std=std)
                pred_items_by_no_cycles.append(decoding_by_no_cycles)
            if show_progress_bar:
                pbar.update(1)
        if show_progress_bar:
            pbar.close()
        return pred_items_by_no_cycles


class TRCAGoldCodesClassifier(components.ProcessingMethod):
    """Standard feature extraction method for c-VEP-based spellers. Basically,
    it computes a template for each sequence, using TRCA. ATTENTION: Only if
    works if the test matrix is 4x4.
    """

    def __init__(self, correct_raster_latencies=False, art_rej=None, **kwargs):
        """ Class constructor """
        super().__init__(fit_dataset=['templates',
                                      'cca_by_seq'])
        self.fitted = dict()

        self.art_rej = art_rej
        self.correct_raster_latencies = correct_raster_latencies

    def _assert_consistency(self, dataset: CVEPSpellerDataset):
        len_seqs = set()
        fs = set()
        fps_resolution = set()
        unique_seqs_by_run = []
        unique_all_seqs = set()
        is_filter_bank = []
        for rec in dataset.recordings:
            rec_sig = getattr(rec, dataset.biosignal_att_key)
            rec_exp = getattr(rec, dataset.experiment_att_key)
            if rec_exp.mode != 'train':
                raise ValueError('There is at least one CVEPSpellerData '
                                 'instance that was not recording under train '
                                 'mode. Aborting feature extraction...')
            if not hasattr(rec_exp, 'spell_target'):
                raise ValueError('There is at least one CVEPSpellerData '
                                 'instance that has not "spell_target" data. '
                                 'Aborting feature extraction...')
            fps_resolution.add(rec_exp.fps_resolution)
            fs.add(rec_sig.fs)
            unique_seqs = get_unique_sequences_from_targets(rec_exp)
            for seq_ in unique_seqs:
                len_seqs.add(len(seq_))
                unique_all_seqs.add(seq_)
            unique_seqs_by_run.append(unique_seqs)

            if isinstance(rec_sig.signal, list):
                is_filter_bank.append(True)
            else:
                is_filter_bank.append(False)
        if len(len_seqs) > 1:
            raise ValueError('There are sequences with different lengths in '
                             'the CVEPSpellerDataset instance! Aborting feature'
                             ' extraction...')
        if len(fs) > 1:
            raise ValueError('There are CVEPSpellerData instances with '
                             'different sampling rates! Aborting feature '
                             'extraction...')
        if len(fps_resolution) > 1:
            raise ValueError('There are CVEPSpellerData instances with '
                             'different refresh rates! Aborting feature '
                             'extraction...')
        if len(unique_all_seqs) > 1:
            # Check if some sequences are shifted versions of another
            unique_ = list(unique_all_seqs)
            all_combos = np.array(list(itertools.combinations(
                np.arange(0, len(unique_)), 2)))
            for i_comb in range(all_combos.shape[0]):
                s1 = unique_[all_combos[i_comb][0]]
                s2 = unique_[all_combos[i_comb][1]]
                if check_if_shifted(s1, s2):
                    raise ValueError('There are targets that share shifted '
                                     'versions of the same sequence. Solve that'
                                     'before extracting features!')

        if len(np.unique(is_filter_bank)):
            is_filter_bank = is_filter_bank[0]
        else:
            raise ValueError('There are recordings that have filter banks and '
                             'other do not.')
        len_seq = len_seqs.pop()
        fs = fs.pop()
        fps_resolution = fps_resolution.pop()

        return fs, fps_resolution, len_seq, unique_seqs_by_run, is_filter_bank

    def fit_dataset(self, dataset: CVEPSpellerDataset, std_epoch_rejection=3.0,
                    show_progress_bar=True):

        # Error checking
        fs, fps_resolution, len_seq, unique_seqs_by_run, is_filter_bank = \
            self._assert_consistency(dataset)

        # Init progress bar for sequences
        if show_progress_bar:
            pbar = tqdm(total=len(dataset.recordings),
                        desc='Extracting unique sequences')

        # Compute sequence length in milliseconds
        len_epoch_ms = len_seq / fps_resolution * 1000
        len_epoch_sam = int(len_seq / fps_resolution * fs)

        # Get the epochs of each sequence
        epochs_by_seq = {}
        for rec_idx, rec in enumerate(dataset.recordings):
            # Extract recording experiment and biosignal
            rec_exp = getattr(rec, dataset.experiment_att_key)
            rec_sig = getattr(rec, dataset.biosignal_att_key)

            # Filter bank init
            if not is_filter_bank:
                rec_sig.signal = [rec_sig.signal]

            # Get unique sequences for this run
            unique_seqs = unique_seqs_by_run[rec_idx]

            # For each filter bank
            for filter_idx, signal in enumerate(rec_sig.signal):

                # Extract epochs
                epochs = mds.get_epochs_of_events(timestamps=rec_sig.times,
                                                  signal=signal,
                                                  onsets=rec_exp.onsets,
                                                  fs=fs,
                                                  w_epoch_t=[0, len_epoch_ms],
                                                  w_baseline_t=None,
                                                  norm=None)

                # Organize epochs by sequence
                for seq_, ep_idxs_ in unique_seqs.items():
                    if tuple(seq_) not in epochs_by_seq:
                        epochs_by_seq[tuple(seq_)] = \
                            [None for i in range(len(rec_sig.signal))]
                        epochs_by_seq[tuple(seq_)][filter_idx] = \
                            epochs[ep_idxs_, :, :]
                    else:
                        if epochs_by_seq[tuple(seq_)][filter_idx] is None:
                            epochs_by_seq[tuple(seq_)][filter_idx] = \
                                epochs[ep_idxs_, :, :]
                        else:
                            epochs_by_seq[tuple(seq_)][
                                filter_idx] = np.concatenate((
                                epochs_by_seq[tuple(seq_)][filter_idx],
                                epochs[ep_idxs_, :, :]
                            ), axis=0)

            if show_progress_bar:
                pbar.update(1)

        # Precompute nearest channels for online artifact rejection
        sorted_dist_ch = rec_sig.channel_set.sort_nearest_channels()

        # New bar
        if show_progress_bar:
            pbar.close()
            pbar = tqdm(total=len(epochs_by_seq) * len_seq,
                        desc='Creating templates')

        # For each sequence
        seq_dict = dict()
        discarded_epochs = 0
        total_epochs = 0
        for seq_ in epochs_by_seq:

            # For each filter of the bank
            for filter_idx in range(len(epochs_by_seq[seq_])):

                # Offline artifact rejection
                if std_epoch_rejection is not None:
                    epochs_std = np.std(epochs_by_seq[seq_][filter_idx],
                                        axis=1)  # STD per samples
                    ch_std = np.std(epochs_std, axis=0)  # Variation of epochs
                    # For each channel, check if the variation is adequate
                    epoch_to_keep = np.zeros(epochs_std.shape)
                    for i in range(len(ch_std)):
                        epoch_to_keep[:, i] = (
                                (epochs_std[:, i] < (
                                        np.median(epochs_std[:, i]) +
                                        std_epoch_rejection * ch_std[
                                            i])) &
                                (epochs_std[:, i] > (
                                        np.median(epochs_std[:, i]) -
                                        std_epoch_rejection * ch_std[
                                            i]))
                        )
                    # Keep only epochs that are suitable for all channels
                    idx_to_keep = (
                            np.sum(epoch_to_keep, axis=1) == epochs_std.shape[1]
                    )
                    epochs_by_seq[seq_][filter_idx] = \
                        epochs_by_seq[seq_][filter_idx][idx_to_keep, :, :]
                    discarded_epochs += np.sum(idx_to_keep == False)
                    total_epochs += len(idx_to_keep)

                # Task related component analysis
                Trca = sf.TRCA()

                # Reference (main template repeated 'no_cycles' times)
                main_template = np.mean(epochs_by_seq[seq_][filter_idx], axis=0)

                # Input data
                X = epochs_by_seq[seq_][filter_idx]

                # Fit TRCA and project the main template
                Trca.fit(X)
                main_template = Trca.project(main_template)
                # Create all possible template shifts
                templates = dict()
                for lag in range(0, 4 * int(len_seq / 4), int(len_seq / 4)):
                    lag_samples = int(np.round(lag / fps_resolution * fs))
                    lagged_seq = np.roll(seq_, -lag, axis=0)
                    lagged_template = np.roll(main_template, -lag_samples,
                                              axis=0)
                    templates[tuple(lagged_seq)] = lagged_template

                    if show_progress_bar:
                        pbar.update(1)

                # STD by channel (useful for online artifact rejection)
                X = X.reshape((X.shape[0] * X.shape[1], X.shape[2]))
                std_by_channel = np.std(X, axis=0)

                # Store data of each trained sequence
                if tuple(seq_) not in seq_dict:
                    seq_dict[tuple(seq_)] = []
                seq_dict[tuple(seq_)].append(
                    {'trca': Trca,
                     'templates': templates,
                     'std_by_channel': std_by_channel,
                     }
                )

        # Store fitted params
        self.fitted = {'sequences': seq_dict,
                       'fs': fs,
                       'fps_resolution': fps_resolution,
                       'len_epoch_ms': len_epoch_ms,
                       'len_epoch_sam': len_epoch_sam,
                       'std_epoch_rejection': std_epoch_rejection,
                       'no_discarded_epochs': discarded_epochs,
                       'no_total_epochs': total_epochs,
                       'sorted_dist_ch': sorted_dist_ch
                       }
        if show_progress_bar:
            pbar.close()

        return self.fitted

    def _is_predict_feasible(self, dataset):
        l_ms = self.fitted['len_epoch_ms']
        for rec in dataset.recordings:
            rec_sig = getattr(rec, dataset.biosignal_att_key)
            rec_exp = getattr(rec, dataset.experiment_att_key)
            feasible = ep.check_epochs_feasibility(timestamps=rec_sig.times,
                                                   onsets=rec_exp.onsets,
                                                   fs=rec_sig.fs,
                                                   t_window=[0, l_ms])
            if feasible != 'ok':
                return False
        return True

    def _is_predict_feasible_signal(self, times, onsets, fs):
        l_ms = self.fitted['len_epoch_ms']
        feasible = ep.check_epochs_feasibility(timestamps=times,
                                               onsets=onsets,
                                               fs=fs,
                                               t_window=[0, l_ms])
        if feasible != 'ok':
            return False
        return True

    def _interpolate_epoch(self, epoch, channel_set, bad_channels_idx,
                           no_neighbors=3):
        interp_epoch = epoch.copy()
        # Are all channels bad?
        if len(bad_channels_idx) == len(channel_set.channels):
            print('> Artifact rejection: Cannot interpolate because all '
                  'channels are bad.')
            return interp_epoch

        # For each bad channel
        bad_labels = []
        for i in bad_channels_idx:
            bad_labels.append(channel_set.channels[i]['label'])
        for i, bad_label in enumerate(bad_labels):
            if bad_label not in self.fitted['sorted_dist_ch']:
                raise Exception('Label %s is not present in the EEGChannelSet'
                                ' in which the model was fitted for!'
                                % bad_label)
            # Find the K labels of the nearest neighbors
            sorted_ch = self.fitted['sorted_dist_ch'][bad_label]
            interp_labels = []
            for ch in sorted_ch:
                interp_labels.append(ch["channel"]["label"])
                if len(interp_labels) == no_neighbors:
                    break

            # Interpolate using average
            interp_idxs = channel_set.get_cha_idx_from_labels(interp_labels)
            interp_epoch[:, bad_channels_idx[i]] = \
                np.mean(interp_epoch[:, np.array(interp_idxs)], axis=1)
        print('> Artifact rejection: interpolated %i channels' %
              len(bad_channels_idx))
        return interp_epoch

    def predict(self, times, signal, trial_idx, exp_data, sig_data):
        # Parameters
        len_epoch_ms = self.fitted['len_epoch_ms']
        len_epoch_sam = self.fitted['len_epoch_sam']
        fs = self.fitted['fs']
        exp_data.onsets = np.array(exp_data.onsets)

        # Assert filter bank
        if not isinstance(signal, list):
            signal = [signal]
        for seq_, seq_data_ in self.fitted['sequences'].items():
            if len(seq_data_) != len(signal):
                raise ValueError('[TRCAGoldCodesClassifier] Cannot predict '
                                 'if the signal do not have the same number of '
                                 'filter banks than the fitted one!')

        # For each number of cycles
        pred_item_by_no_cycles = []
        no_cycles = np.max(exp_data.cycle_idx).astype(int) + 1
        for nc in range(no_cycles):
            # Identify what are the epochs that must be processed
            idx = (np.array(exp_data.trial_idx) == trial_idx) & \
                  (np.array(exp_data.cycle_idx) <= nc)

            # Raster latencies?
            raster_dict = None
            if self.correct_raster_latencies:
                possible_onsets_idx = np.where(
                    exp_data.raster_events['onset'] <
                    exp_data.onsets[idx][-1]
                )[0]
                if possible_onsets_idx.size > 0:
                    raster_dict = exp_data.raster_events['event'][
                        possible_onsets_idx[-1]]

            # For each fitted sequence
            pred_item = []
            final_seq = []
            final_corr = []
            for seq_, seq_data_ in self.fitted['sequences'].items():

                # For each possible filter bank
                f_corrs = []
                for filter_idx, filter_signal in enumerate(signal):

                    # Extract the epochs for that signal, trial and no. cycles
                    epochs = mds.get_epochs_of_events(
                        timestamps=times,
                        signal=filter_signal,
                        onsets=exp_data.onsets[idx],
                        fs=fs,
                        w_epoch_t=[0, len_epoch_ms],
                        w_baseline_t=None,
                        norm=None)
                    if len(epochs.shape) == 2:
                        # Create a dummy dimension if we have only one epoch
                        epochs = np.expand_dims(epochs, 0)

                    # Artifact rejection
                    if self.art_rej is not None:
                        epoch_std_by_channel = \
                            np.std(epochs[:, :len_epoch_sam, :], axis=1)
                        for i in range(epoch_std_by_channel.shape[0]):
                            discard_epoch = epoch_std_by_channel[i, :] > \
                                            seq_data_[filter_idx][
                                                'std_by_channel'] \
                                            * self.art_rej
                            if np.any(discard_epoch):
                                # TODO: precompute distance matrix before
                                epochs[i, :len_epoch_sam, :] = \
                                    self._interpolate_epoch(
                                        epoch=epochs[i, :len_epoch_sam, :],
                                        channel_set=sig_data.channel_set,
                                        bad_channels_idx=
                                        np.where(discard_epoch)[0],
                                        no_neighbors=3
                                    )

                    # Average the epochs
                    avg = np.mean(epochs[:, :len_epoch_sam, :], axis=0)

                    # TRCA projection
                    x_ = seq_data_[filter_idx]['trca'].project(
                        avg)

                    # Correlation coefficients between x_ and the templates
                    corrs = []
                    seqs = []
                    for shift_seq_, template_ in \
                            seq_data_[filter_idx]['templates'].items():
                        # Correct template using raster latencies
                        lat_s = 0
                        if raster_dict is not None:
                            if shift_seq_ in raster_dict:
                                lat_s = int(raster_dict[shift_seq_] * fs)
                        tem_ = np.roll(template_, lat_s)
                        temp_p = np.dot(tem_, x_) / np.sqrt(np.dot(np.dot(
                            tem_, tem_), np.dot(x_, x_)))
                        corrs.append(temp_p)
                        # seqs.append(shift_seq_)
                        if filter_idx == 0:
                            final_seq.append(shift_seq_)
                    f_corrs.append(corrs)

                # Average the correlations between different filter banks
                corrs = list(np.mean(np.array(f_corrs), axis=0))
                final_corr = final_corr + corrs
                # seqs = np.array(seqs)
            final_corr = np.array(final_corr)
            final_seq = np.array(final_seq)

            # Sort the sequences by corrs' descending order
            sorted_idx = np.argsort(-final_corr)
            sorted_corrs = final_corr[sorted_idx]
            sorted_seqs = final_seq[sorted_idx, :]

            # Identify the selected command
            sorted_cmds = get_items_by_sorted_sequences(
                experiment=exp_data,
                trial_idx=trial_idx,
                sorted_seqs=sorted_seqs,
                sorted_corrs=sorted_corrs
            )
            pred_item.append({
                'sorted_cmds': sorted_cmds,
                'fitted_sequence': seq_
            })

            # Store the predicted item
            pred_item_by_no_cycles.append(pred_item)

        return pred_item_by_no_cycles

    def predict_dataset(self, dataset: CVEPSpellerDataset,
                        show_progress_bar=True):
        # Error detection
        if not self.fitted:
            raise Exception(
                'Cannot predict if templates and '
                'TRCA projections are not fitted before! Aborting...')
        for rec in dataset.recordings:
            rec_sig = getattr(rec, dataset.biosignal_att_key)
            rec_exp = getattr(rec, dataset.experiment_att_key)
            if rec_sig.fs != self.fitted['fs']:
                raise ValueError('The sampling rate of this test recording '
                                 '(%.2f Hz) is not the same as for the fitted '
                                 'recordings! (%.2f Hz)' %
                                 (rec_sig.fs, self.fitted['fs']))
            if rec_exp.fps_resolution != self.fitted['fps_resolution']:
                raise ValueError('The refresh rate of this test recording '
                                 '(%.2f Hz) is not the same as for the fitted '
                                 'recordings! (%.2f Hz)' %
                                 (rec_exp.fps_resolution,
                                  self.fitted['fps_resolution']))

        if show_progress_bar:
            pbar = tqdm(total=len(dataset.recordings),
                        desc='Predicting dataset')

        # For each recording
        pred_items_by_no_cycles = []
        for rec in dataset.recordings:
            rec_sig = getattr(rec, dataset.biosignal_att_key)
            rec_exp = getattr(rec, dataset.experiment_att_key)

            # For each trial
            for t_idx in np.unique(rec_exp.trial_idx):
                decoding_by_no_cycles = \
                    self.predict(rec_sig.times, rec_sig.signal, t_idx,
                                 rec_exp, rec_sig)
                pred_items_by_no_cycles.append(decoding_by_no_cycles)
            if show_progress_bar:
                pbar.update(1)
        if show_progress_bar:
            pbar.close()
        return pred_items_by_no_cycles


class CircularShiftingEarlyStopping(components.ProcessingMethod):
    def __init__(self, **kwargs):
        """ Class constructor """
        super().__init__()

    def check_early_stop(self, corr_vector, std=3.0):
        """ Early stopping method based on normal distributions.

        Parameters
        --------------
        corr_vector: list() or 1D ndarray
            Vector that represents the sorted correlations for each of the
            possible commands, where corr_vector[0] must point to the most
            probable selected command.
        std: int
            Multiplier that determines if the selected command is an outlier
            of the normal distribution made up from the rest of correlations.
            Typical values are: 1 (outside 68% of data), 2 (outside 95% of
            data), and 3 (default, outside 99.7% of data).

        Returns
        --------------
        must_stop: bool
            True if it is possible to stop now, false otherwise.
        probs: 1D ndarray
            Current estimated probabilities of being selected (sorted).
        """
        corr_vector = np.array(corr_vector)
        threshold = np.mean(corr_vector[1:]) + std * np.std(corr_vector[1:])
        must_stop = corr_vector[0] > threshold
        probs = threshold - corr_vector
        probs = 1 - (probs / np.max(probs))
        return must_stop, probs


# ------------------------------- UTILS -------------------------------------- #
def get_unique_sequences_from_targets(experiment: CVEPSpellerData):
    """ Function that returns the unique sequences of all targets.
            return
        """

    def is_shifted_version(stored_seqs, seq_to_check):
        for s in stored_seqs:
            if len(s) != len(seq_to_check):
                continue
            for j in range(len(seq_to_check)):
                if np.all(np.array(s) == np.roll(seq_to_check, -j)):
                    return s
        return None

    sequences = dict()
    try:
        # todo: command_idx, unit_idx y demas lo tiene que hacer medusa y no unity

        for idx in range(len(experiment.command_idx)):
            # todo: revisar lo de los levels
            # Get the sequence used for the current command
            # l_ = int(experiment.level_idx[idx])
            # u_ = int(experiment.unit_idx[idx])
            # curr_command = experiment.paradigm_conf[m_][l_][u_][c_]
            m_ = int(experiment.matrix_idx[idx])
            c_ = int(experiment.command_idx[idx])
            curr_seq_ = experiment.commands_info[m_][str(c_)]['sequence']
            # Note: str(c_) is used because the previous json serialization
            #       interprets all dictionary keys as strings.

            # Add the command index to its associated sequence
            if len(sequences) == 0:
                sequences[tuple(curr_seq_)] = [idx]
            elif tuple(curr_seq_) in sequences:
                # Already there, add the cycle idx
                sequences[tuple(curr_seq_)].append(idx)
            else:
                # If not there, first check that it is not a shifted version
                # of a present sequences
                orig_seq = is_shifted_version(list(sequences.keys()), curr_seq_)
                if orig_seq is not None:
                    sequences[tuple(orig_seq)].append(idx)
                else:
                    sequences[tuple(curr_seq_)] = [idx]
    except Exception as e:
        print(e)
    return sequences


def check_if_shifted(seq1, seq2):
    max_corr = np.max(np.correlate(seq1, seq1, 'same'))
    cross_corr = np.correlate(seq1, seq2, 'same')
    if np.max(cross_corr) == max_corr:
        print('WARNING: Two sequences are shifted versions of themselves!')
        return True
    else:
        return False


def get_items_by_sorted_sequences(experiment: CVEPSpellerData,
                                  trial_idx, sorted_seqs, sorted_corrs=None):
    # Find the first index of the trial to access matrix_idx, etc
    try:
        idx = list(experiment.trial_idx == trial_idx).index(True)
    except ValueError as e:
        raise ValueError('[get_items_by_sorted_sequences] Trial with idx %i not'
                         ' found in the experiment data! ' + str(e) % trial_idx)

    # Get the possible commands
    m_ = int(experiment.matrix_idx[idx])
    l_ = int(experiment.level_idx[idx])
    u_ = int(experiment.unit_idx[idx])
    possible_cmd = experiment.paradigm_conf[m_][l_][u_]

    # For each sequence in descending order of probability of being selected
    sorted_commands = list()
    for i in range(sorted_seqs.shape[0]):
        # For each possible command
        curr_comm_dict = dict()
        for cmd_id in possible_cmd:
            cmd_seq = experiment.commands_info[m_][str(cmd_id)]['sequence']
            if np.all(np.array(cmd_seq) == sorted_seqs[i, :]):
                # Found!
                curr_comm_dict['item'] = experiment.commands_info[m_][str(
                    cmd_id)]
                curr_comm_dict['label'] = curr_comm_dict['item']['label']
                curr_comm_dict['coords'] = [m_, l_, u_, int(cmd_id)]
                curr_comm_dict['correlation'] = None
                if sorted_corrs is not None:
                    curr_comm_dict['correlation'] = sorted_corrs[i]
                sorted_commands.append(curr_comm_dict)
                break
    return sorted_commands


def autocorr_zeropad(x):
    """ With zero padding, equivalent to np.correlate() """
    N = len(x)
    rxx = []
    x_lagged = np.concatenate((np.zeros((N - 1,)), x, np.zeros((N,))))
    for i in range(2 * N - 1):
        rxx.append(np.sum(x * x_lagged[i:i + N]))
    rxx = np.array(rxx)
    return rxx


def autocorr_circular(x):
    """ With circular shifts (periodic correlation) """
    N = len(x)
    rxx = []
    for i in range(-(N - 1), N):
        rxx.append(np.sum(x * np.roll(x, i)))
    rxx = np.array(rxx)
    return rxx


# ----------------------------- CODE GENERATORS ----------------------------- #
class LFSR:
    """ Computes a Linear-Feedback Shift Register (LFSR) sequence. """

    def __init__(self, polynomial, base=2, seed=None, center=False):
        """ Constructor of LFSR """
        self.polynomial = polynomial
        self.base = base
        self.seed = seed
        self.order = len(polynomial)
        self.N = base ** self.order - 1
        self.sequence = self.lfsr(polynomial, base, seed, center)

    @staticmethod
    def lfsr(polynomial, base=2, seed=None, center=False):
        """ This method implements a Linear-Feedback Shift Register (LFSR).

        IMPORTANT: maximal length sequences (m-sequences) can be only generated
        if the polynomial (taps) is primitive. I.e.:
            - the number of taps is even.
            - the set of taps is setwise co-prime (there must be no divisor
            other than 1 common to all taps).
        A list of primitive polynomials in function of the order m can be
        found here:
        https://en.wikipedia.org/wiki/Linear-feedback_shift_register

        NOTE: if the seed is composed by all zeros, the output sequence will be
        zeros.

        Parameters
        ----------
        polynomial: list
            Generator polynomial. E.g. (bias is specified for math convention
            but not used):
            "1 + x^5 + x^6" would be [0, 0, 0, 0, 1, 1].
            "1 + 2x + x^4" would be [2, 0, 0, 1]
        base : int
            (Optional, default: base = 2) Base of the sequence events that
            belongs to the Galois Field of the same base. By default, base=2,
            so only events of type {0,1} (or {-1,1} are part of the returned
            sequence.
        seed : list
            (Optional) Initial state. If not provided, the default state is a
            one-array with length equal to the order of the polynomial.
        center : bool
            (Optional, default = False) Determines if a centering over zero
            must be performed in the returned sequence (e.g., {0,1} -> {-1,1})

        Returns
        -------
        sequence : list
            LFSR m-sequence (maximum length is base^order-1), where order is the
            order of the polynomial.
        """
        # Defaults and error detection
        order = len(polynomial)
        if seed is None:
            seed = [1 for i in range(order)]
        if order > len(seed):
            raise Exception('[LSFR] The order of the polynom (%i) is higher '
                            'than the initial state length (%i)!' %
                            (order, len(seed)))

        # LFSR
        sequence = seed.copy()
        polynom = np.array(polynomial)
        while len(sequence) < base ** order - 1:
            new_bit = (np.matmul(polynom, np.array(sequence[:order]))) % base
            sequence.insert(0, new_bit)

        # Map the values to center around zero
        if center:
            if base == 2:
                sequence = np.array(sequence) * 2 - 1
            else:
                sequence = np.array(sequence) - np.floor(base / 2).astype(int)
        return sequence


class GOLD_CODES:
    """ Computes a set of 2^N-1 gold codes """

    def __init__(self, polynomial_1, polynomial_2, base=2, center=False):
        """ Constructor of GOLD_CODES """
        self.polynomial_1 = polynomial_1
        self.polynomial_2 = polynomial_2
        self.base = base
        self.order = len(polynomial_1)
        self.N = base ** self.order - 1
        self.sequences = self.gold_codes(polynomial_1, polynomial_2, base,
                                         self.order)

    @staticmethod
    def gold_codes(pol_1, pol_2, base, N):
        """ This method implements a generator of Gold Codes.

        IMPORTANT: Gold Codes with optimized correlation properties
        are obtained from two PREFERRED m-sequences.

        Parameters
        ----------
        pol_1: list
            First Generator polynomial. E.g. (bias is specified for math convention
            but not used):
            "1 + x^5 + x^6" would be [0, 0, 0, 0, 1, 1].
            "1 + 2x + x^4" would be [2, 0, 0, 1]
        pol_2: list
            Second Generator polynomial. E.g. (bias is specified for math convention
            but not used):
        base : int
            Base of the sequence events that
            belongs to the Galois Field of the same base.
        N : int
            Length of the sequence.

        Returns
        -------
        sequence : list
            LFSR m-sequence (maximum length is base^order-1), where order is the
            order of the polynomial.
        """
        if (pol_1[-1] != pol_2[-1]):
            raise Exception("Pol_1 and Pol_2 must have the same degree")
        else:
            sequences = []
            seq_1 = LFSR.lfsr(pol_1)  # seed [0,...,0,1]
            seq_2 = LFSR.lfsr(pol_2)
            seq_1.reverse()
            seq_2.reverse()
            for i in range(0, base ** N - 1):
                seq_2_rolled = list(np.roll(np.array(seq_2), i))
                seq = [int(bool(seq_1[j]) ^ bool(seq_2_rolled[j])) for j in
                       range(len(seq_1))]
                sequences.append(seq)
            return sequences
