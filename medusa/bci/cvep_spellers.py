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
import copy, warnings
import itertools

from abc import ABC, abstractmethod
import numpy as np
from tqdm import tqdm

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
                 cvep_model, spell_result, fps_resolution, spell_target=None,
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
        self.cvep_model = cvep_model
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


# ---------------------------------- MODELS ---------------------------------- #
class CVEPModelCircularShifting(components.Algorithm):

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
        self.add_method('clf_method', CircularShiftingClassifier(
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
            std_epoch_rejection=None,
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

    def __init__(self, correct_raster_latencies=False, art_rej=None, **kwargs):
        """ Class constructor """
        super().__init__(fit_dataset=['templates',
                                      'cca_by_seq'])
        self.fitted = dict()

        self.art_rej = art_rej
        self.correct_raster_latencies = correct_raster_latencies

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

        sorted_dist_ch = None
        if std_epoch_rejection is not None:
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
                raise ValueError('[CircularShiftingClassifier] Cannot predict '
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
            if tuple(curr_seq_) not in sequences:
                sequences[tuple(curr_seq_)] = [idx]
            else:
                sequences[tuple(curr_seq_)].append(idx)

        # todo: check that sequences are not shifted versions of themselves??
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
