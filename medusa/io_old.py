"""Created on Monday March 15 19:27:14 2021

In this module, you will find all the data structures used in medusa, as well as
functionality to save and load files in supported formats. Enjoy!

@author: Eduardo Santamaría-Vázquez
"""

import numpy as np
import json, bson, h5py
import scipy.io
import warnings
from abc import ABC, abstractmethod
import copy
from medusa import meeg


class Recording:
    """
    Skeleton class to save the data from one recording. It implements all
    necessary methods to save and load from several formats. It is composed
    by 2 different parts, the biosignals recordings (e.g., EEG, MEG), and the
    experiment data, which saves all the information about the experiment or
    events. All biosignals and events must be synchronized with the same
    temporal origin. This class must be serializable.
    """
    def __init__(self, id_recording=None, subject=None, description=None,
                 source=None, date=None, **kwargs):
        """Recording dataset constructor. Custom useful parameters can be
        provided to save in the class.

        Parameters
        ----------
        id_recording : str or None
            Identifier of the recording for automated processing or easy
            identification
        subject : str or None
            Subject identifier
        description : str or None
            Description of this recording. Useful to write comments (e.g., the
            subject moved a lot, the experiment
            was interrupted, etc)
        source : str or None
            Source of the data, such as software, equipment, experiment, etc
        kwargs : custom key-value parameters
            Other useful parameters (e.g., software version, research team,
            laboratory, etc)
        """

        # Standard attributes
        self.id_recording = id_recording
        self.subject = subject
        self.description = description
        self.source = source
        self.date = date

        # Useful variables
        self.biosignals = dict()
        self.experiments = dict()

        # Set the specified arguments
        for key, value in kwargs.items():
            setattr(self, key, value)

    def add_biosignal(self, biosignal, key=None):
        """Adds a biosignal recording. Each biosignal has predefined classes
        that must be instantiated before (e.g., EEG, MEG)

        Parameters
        ----------
        biosignal : biosignal class
            Instance of the biosignal class. This class must be serializable.
            Current available: EEG, MEG.
        key: str
            Custom name for this biosignal. If not provided, the biosignal will
            be saved in an attribute according to its type in lowercase
            (e.g., eeg, meg, etc). This parameter is  useful if several
            biosignals of the same type are added to this recording
        """
        # Check errors
        if not issubclass(type(biosignal), Biosignal):
            raise TypeError('Parameter biosignal must subclass '
                            'medusa.io.Biosignal')
        # Check type
        biosignal_class_name = type(biosignal).__name__
        att = biosignal_class_name.lower() if key is None else key
        if biosignal_class_name not in globals():
            warnings.warn('Custom biosignal %s. Some high-level functions may '
                          'not work' % type(biosignal))
        if isinstance(biosignal, CustomBiosignal):
            warnings.warn('Unspecific biosignal %s. Some high-level functions '
                          'may not work' % type(biosignal))
        # Check key
        if hasattr(self, att):
            raise ValueError('This recording already contains an attribute '
                             'with key %s' % att)
        # Add biosignal
        setattr(self, att, biosignal)
        self.biosignals[att] = biosignal_class_name

    def add_experiment_data(self, experiment_data, key=None):
        """Adds the experiment data of this recording. Each experiment should
        have a predefined class that must be instantiated before. Several
        classes are defined within medusa core, but it also can be a custom
        experiment.

        Parameters
        ----------
        experiment_data : experiment class
            Instance of an experiment class. This class can be custom if it is
            serializable, but it is recommended to use the classes provided
            by medusa core in different modules (e.g., bci.erp_paradigms.rcp)
        key: str
            Custom name for this experiment. If not provided, the experiment
            will be saved in an attribute according to its type (e.g., rcp,
            cake paradigm, etc). This parameter is useful if several experiments
            of the same type are added to this recording
        """
        # Check errors
        if not issubclass(type(experiment_data), ExperimentData):
            raise TypeError('Parameter experiment_data must subclass '
                            'medusa.io.ExperimentData')
        # Check type
        experiment_class_name = type(experiment_data).__name__
        att = experiment_class_name.lower() if key is None else key
        if experiment_class_name not in globals():
            warnings.warn('Custom experiment data %s. Some high-level functions'
                          ' may not work' % type(experiment_data))
        if isinstance(experiment_data, CustomExperimentData):
            warnings.warn('Unspecific experiment data %s. Some high-level '
                          'functions may not work' % type(experiment_data))
        # Check key
        if hasattr(self, att):
            raise ValueError('This recording already has an attribute with key '
                             '%s' % att)
        # Add experiment
        setattr(self, att, experiment_data)
        self.experiments[att] = experiment_class_name

    def cast_biosignal(self, key, biosignal_class):
        """This function casts a biosignal of this recording to the class passed
        in biosignal_class
        """
        # Check errors
        if not issubclass(biosignal_class, Biosignal):
            raise TypeError('Class %s must subclass medusa.io.Biosignal' %
                            biosignal_class.__name__)
        biosignal = getattr(self, key)
        biosignal_dict = biosignal.to_serializable_obj()
        setattr(self, key, biosignal_class.from_dict(biosignal_dict))

    def cast_experiment(self, key, experiment_class):
        """This function casts an experiment of recording run to the class
        passed in experiment_class
        """
        # Check errors
        if not issubclass(experiment_class, ExperimentData):
            raise TypeError('Class %s must subclass medusa.io.ExperimentData' %
                            experiment_class.__name__)
        experiment_data = getattr(self, key)
        experiment_data_dict = experiment_data.to_serializable_obj()
        setattr(self, key, experiment_class.from_dict(experiment_data_dict))

    def rename_attribute(self, old_key, new_key):
        """Rename an attribute. Useful to unify attribute names on fly while
        creating a dataset because it's very  cheap

        Parameters
        ----------
        old_key : str
            Old attribute key
        new_key : str
            New attribute key
        """
        self.__dict__[new_key] = self.__dict__.pop(old_key)

    def to_dict(self):
        """Function that converts the class in a python dictionary
        """
        rec_dict = self.__dict__
        # Process biosginals
        for key in self.biosignals:
            biosignal = getattr(self, key)
            rec_dict[key] = biosignal.to_serializable_obj()
        # Process experiments
        for key in self.experiments:
            experiments = getattr(self, key)
            rec_dict[key] = experiments.to_serializable_obj()
        return rec_dict

    @staticmethod
    def from_dict(rec_dict):
        """Function that loads the class from a python dictionary
        """
        # Handle biosignals
        if 'biosignals' in rec_dict:
            for biosignal_key, biosignal_class_name in rec_dict['biosignals'].\
                    items():
                try:
                    rec_dict[biosignal_key] = globals()[biosignal_class_name].\
                        from_serializable_obj(rec_dict[biosignal_key])
                except KeyError:
                    warnings.warn('Biosignal class %s not found. Saving as '
                                  'CustomBiosignal. Use method cast_biosignal '
                                  'to transform it to a custom type' %
                                  type(biosignal_class_name))
                    rec_dict[biosignal_key] = CustomBiosignal.\
                        from_dict(rec_dict[biosignal_key])
        # Handle experiments
        if 'experiments' in rec_dict:
            for exp_key, exp_class_name in rec_dict['experiments'].items():
                try:
                    rec_dict[exp_key] = globals()[exp_class_name].\
                        from_serializable_obj(rec_dict[exp_key])
                except KeyError:
                    warnings.warn('Experiment class %s not found. Saving as '
                                  'CustomExperimentData. Use method '
                                  'cast_experiment to transform it to other '
                                  'type' % type(exp_class_name))
                    rec_dict[exp_key] = CustomExperimentData.\
                        from_dict(rec_dict[exp_key])
        # Instantiate class
        return Recording(**rec_dict)

    def save(self, path):
        """Saves the class to the format specified in the extension of the path.

        Parameters
        ----------
        path: str
            File path. The extension will be automatically interpreted. Current
            supported formats:
                - BSON: This format is efficient, easy to use and multiplatform.
                    Thus, it comes with  advantages in comparison to other
                    formats. BSON format requires serializable classes.
                - JSON: This format is human readable and multiplatform, widely
                    used for web applications. Nevertheless, files are
                    encoded in utf-8 and thus occupy more space. JSON format
                    requires serializable classes.
                - MAT: This is a binary format widely used in research for its
                    compatibility with Matlab. Very powerful, but lacks of
                    wide multiplatform compatibility. MAT format requires
                    serializable classes.
                - HDF5: Binary format for large files (>2GB). Use this format
                    for very large recordings.
        """
        ext = path.split('.')[-1]
        if ext == 'bson':
            return self.save_to_bson(path)
        elif ext == 'json':
            return self.save_to_json(path)
        elif ext == 'mat':
            return self.save_to_mat(path)
        elif ext == 'hdf5':
            return self.save_to_hdf5(path)
        else:
            raise ValueError('Format %s is not available yet' % ext)

    def save_to_bson(self, path):
        """ Saves the class attributes in BSON format"""
        with open(path, 'wb') as f:
            f.write(bson.dumps(self.to_dict()))

    def save_to_json(self, path, encoding='utf-8', indent=4):
        """ Saves the class attributes in JSON format"""
        with open(path, 'w', encoding=encoding) as f:
            json.dump(self.to_dict(), f, indent=indent)

    def save_to_mat(self, path):
        """ Save the class in a MATLAB .mat file using scipy. Attributes set to
        None will be changed to False. since mat data do not support None
        type"""
        rec_dict = self.to_dict()
        for key, value in rec_dict.items():
            if value is None:
                rec_dict[key] = 'None'
        scipy.io.savemat(path, mdict=rec_dict)

    def save_to_hdf5(self, path):
        """ Saves the class using pickle"""
        raise Exception('Not supported yet!')


class Biosignal(ABC):
    """Skeleton class for biosignals data
    """

    @abstractmethod
    def to_dict(self):
        """This function must return a serializable dict (primitive types)
        containing the relevant attributes of the class
        """
        pass

    @staticmethod
    @abstractmethod
    def from_dict(dict_data):
        """This function must return an instance of the class from a
        serializable dict (primitive types)"""
        pass


class CustomBiosignal(Biosignal):
    """Custom experiment data class. This class does not check the arguments and
    provides less functionality that the proper experiment class. It should
    only be used for custom experiments that do not fit in other experiment
    data classes
    """

    def __init__(self, **kwargs):
        """CustomBiosginal constructor

        Parameters
        ----------
        kwargs: kwargs
            Key-value arguments to be saved in the class. This general class
            does not check anything
        """
        # Set the specified arguments
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self):
        rec_dict = self.__dict__
        for key in rec_dict.keys():
            if type(rec_dict[key]) == np.ndarray:
                rec_dict[key] = rec_dict[key].tolist()
        return rec_dict

    @staticmethod
    def from_dict(dict_data):
        return CustomBiosignal(**dict_data)


class EEG(Biosignal):
    """Electroencephalography (EEG) biosignal
    """

    def __init__(self, times, signal, fs, channel_set, **kwargs):
        """EEG constructor

        Parameters
        ----------
        times : list or numpy.ndarray
            1D numpy array [n_samples]. Timestamps of each sample. If they are
            not available, generate them
            artificially. Nevertheless, all signals and events must have the
            same temporal origin
        signal : list or numpy.ndarray
            2D numpy array [n_samples x n_channels]. EEG samples (the units
            should be defined using kwargs)
        fs : int or float
            Sample rate of the recording.
        channel_set : meeg_standards.EEGChannelSet
            EEG channel set
        kwargs: kwargs
            Any other parameter provided will be saved in the class (e.g.,
            equipment description)
        """
        # To numpy arrays
        times = np.array(times)
        signal = np.array(signal)
        # Check errors
        if signal.shape[1] != channel_set.n_cha:
            raise Exception("Signal with shape [samples x channels] does not "
                            "match with the number of channels")
        if times.shape[0] != signal.shape[0]:
            raise Exception("Parameters times and signal must have the same "
                            "length")

        # Standard attributes
        self.times = times
        self.signal = signal
        self.fs = fs
        self.channel_set = channel_set

        # Optional attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

    def change_channel_set(self, channel_set):
        """Smart change of channel set, updating the signal and all related
        attributes

        Parameters
        ----------
        channel_set : meeg_standards.EEGChannelSet
            EEG channel set
        """
        # Get the index of the channels
        cha_idx = self.channel_set.get_cha_idx_from_labels(channel_set.l_cha)
        # Select and reorganize channels channels
        self.channel_set.subset(cha_idx)
        # Reorganize signal
        self.signal = self.signal[:, cha_idx]

    def to_dict(self):
        rec_dict = self.__dict__
        for key in rec_dict.keys():
            if type(rec_dict[key]) == np.ndarray:
                rec_dict[key] = rec_dict[key].tolist()
        return rec_dict

    @staticmethod
    def from_dict(dict_data):
        return EEG(**dict_data)


class MEG(Biosignal):

    # TODO check everything

    """Magnetoencephalography (MEG) biosignal
    """

    def __init__(self, times, signal, fs, channel_set, **kwargs):
        """MEG constructor

        Parameters
        ----------
        times : list or numpy.ndarray
            1D numpy array [n_samples]. Timestamps of each sample. If they are
            not available, generate them
            artificially. Nevertheless, all signals and events must have the
            same temporal origin
        signal : list or numpy.ndarray
            2D numpy array [n_samples x n_channels]. MEG samples (the units
            should be defined using kwargs)
        fs : int or float
            Sample rate of the recording.
        channel_set : list or meeg_standards.MEGChannelSet
            MEG channel set.
        kwargs: kwargs
            Any other parameter provided will be saved in the class (e.g.,
            equipment description)
        """
        # To numpy arrays
        times = np.array(times)
        signal = np.array(signal)
        # Check errors
        if signal.shape[1] != channel_set.n_cha:
            raise Exception("Signal with shape [samples x channels] does not "
                            "match with the number of channels")
        if times.shape[0] != signal.shape[0]:
            raise Exception("Parameters times and signal must have the same "
                            "length")

        # Standard attributes
        self.times = times
        self.signal = signal
        self.fs = fs
        self.channel_set = channel_set

        # Optional attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

    def change_channel_set(self, channel_set):
        """Smart change of channel set, updating the signal and all related
        attributes

        Parameters
        ----------
        channel_set : meeg_standards.MEGChannelSet
            MEG channel set
        """
        raise NotImplementedError

    def to_dict(self):
        rec_dict = self.__dict__
        for key in rec_dict.keys():
            if type(rec_dict[key]) == np.ndarray:
                rec_dict[key] = rec_dict[key].tolist()
        return rec_dict

    @staticmethod
    def from_dict(dict_data):
        return MEG(**dict_data)
            

def load_meg_signal_from_spm_file(path, fs):

    # TODO check everything

    """Function to load a MEG recording from a spm file

    Parameters
    ----------
    path : str
        Path of the file
    fs : int or float
        Sample rate of the recording
    """
    data = scipy.io.loadmat(path)
    info = data['D'][0, 0]

    datatype = info['datatype'][0]
    num_chan = np.size(info['channels'])
    num_samples = info['Nsamples'][0, 0]

    raw = np.fromfile(path[0:len(path)-3] + 'dat', datatype[0:len(datatype)-3])
    signal = np.reshape(raw, [num_chan, num_samples], order='F')
    times = np.linspace(0, signal[0]/fs, signal[0])
    channels = None
    return MEG(times, signal, fs, channels)


class ExperimentData(ABC):

    """Skeleton class for experiment data
    """

    @abstractmethod
    def to_dict(self):
        """This function must return a serializable dict (primitive types)
        containing the relevant attributes of the class
        """
        pass

    @staticmethod
    @abstractmethod
    def from_dict(dict_data):
        """This function must return an instance of the class from a
        serializable dict (primitive types)
        """
        pass


class CustomExperimentData(ExperimentData):
    """Custom experiment data class. This class does not check the arguments and
    provides less functionality that the proper experiment class. It should
    only be used for custom experiments that do not fit in other experiment
    data classes
    """
    def __init__(self, **kwargs):
        """CustomExperimentData constructor

        Parameters
        ----------
        kwargs: kwargs
            Key-value arguments to be saved in the class. This general class
            does not check anything
        """
        # Set the specified arguments
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self):
        rec_dict = self.__dict__
        for key in rec_dict.keys():
            if type(rec_dict[key]) == np.ndarray:
                rec_dict[key] = rec_dict[key].tolist()
        return rec_dict

    @staticmethod
    def from_dict(dict_data):
        return CustomExperimentData(**dict_data)


class RCPData(ExperimentData):
    """Class with the necessary attributes and functionality to represent an
    row-column paradigm (RCP) run, a widely used implementation of ERP-based
    speller. It provides compatibility with high level functions for this
    paradigm in BCI module.
    """

    def __init__(self, mode, onsets, codes, sequences, trials, matrix_indexes,
                 matrix_dims, matrix_labels, n_seqs, spell_result,
                 control_state_result, erp_labels=None,
                 control_state_labels=None, spell_target=None,
                 control_state_target=None, **kwargs):

        """RCPData constructor

        Parameters
        ----------
        mode : str {"train"|"test"}
            Mode of this run.
        onsets : list or numpy.ndarray [n_stim x 1]
            Timestamp of each stimulation
        codes : list or numpy.ndarray [n_stim x 1]
            Code of the highlighted row or column for each stimulation
        sequences : list or numpy.ndarray [n_stim x 1]
            Sequence index for each stimulation
        trials : list or numpy.ndarray [n_stim x 1]
            Trial index for each stimulation
        matrix_indexes : list or numpy.ndarray [n_stim x 1]
            Matrix index for each stimulation
        matrix_dims : list [n_matrices x 2]
            Dimensions of each matrix. Each position of the list contains the
            tuple (n_rows, n_cols)
        matrix_labels :  list [n_matrices x n_rows x n_cols]
            Labels of each RCP matrix
        n_seqs : int
            Sequences of stimulation used in the run
        spell_result : list or numpy.ndarray [n_trials x 3]
            Spell result of this run. Each position contains the data of the
            selected target (matrix_idx, row, col)
        control_state_result : list or numpy.ndarray
            Control state result of this run. Each position contains the
            detected control state of the user (0 -> non-control, 1-> control)
        erp_labels : list or numpy.ndarray [n_stim x 1] {0|1}
            Only in train mode. Contains the erp labels of each stimulation
            (0 -> non-target, 1-> target)
        control_state_labels : list or numpy.ndarray [n_stim x 1] {0|1}
            Only in train mode. Contains the erp labels of each stimulation
            (0 -> non-control, 1-> control)
        spell_target : list or numpy.ndarray [n_trials x 3]
            Only in train mode. List containing the target for each trial.
            Each position represents (matrix_idx, row, col)
        control_state_target : list or numpy.ndarray [n_trials x 1]
            Only in train mode. List containing the control state target for
            each trial.
        kwargs : kwargs
            Custom arguments that will also be saved in the class
        """
        # Check errors
        if mode == 'train':
            if erp_labels is None or control_state_labels is None or \
                    spell_target is None or control_state_target is None:
                raise ValueError('Attributes erp_labels, control_state_labels, '
                                 'spell_target, control_state_target must '
                                 'be provided in train mode')

        # Standard attributes
        self.mode = mode
        self.onsets = np.array(onsets)
        self.codes = np.array(codes)
        self.sequences = np.array(sequences)
        self.trials = np.array(trials)
        self.matrix_indexes = np.array(matrix_indexes)
        self.matrix_dims = np.array(matrix_dims)
        self.matrix_labels = np.array(matrix_labels)
        self.n_seqs = n_seqs
        self.spell_result = np.array(spell_result)
        self.control_state_result = np.array(control_state_result)
        self.erp_labels = np.array(erp_labels) if erp_labels is not None else \
            erp_labels
        self.control_state_labels = np.array(control_state_labels) \
            if control_state_labels is not None else control_state_labels
        self.spell_target = np.array(spell_target)
        self.control_state_target = np.array(control_state_target)

        # Optional attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self):
        rec_dict = self.__dict__
        for key in rec_dict.keys():
            if type(rec_dict[key]) == np.ndarray:
                rec_dict[key] = rec_dict[key].tolist()
        return rec_dict

    @staticmethod
    def from_dict(dict_data):
        return RCPData(**dict_data)


class ERPSpellerData:
    """Experiment info class for ERP-based spellers. It supports nested
    multi-level paradigms. This unified class can be used to represent a run
    of every ERP stimulation paradigm designed to date, and is the expected
    class for feature extraction and command decoding functions of the module
    medusa.bci.erp_paradigms. It is complicated, but powerful so.. use it well!
    """

    def __init__(self, mode, paradigm_conf, onsets, batch_idx, group_idx,
                 unit_idx, level_idx, matrix_idx, sequence_idx, trial_idx,
                 spell_result, control_state_result, erp_labels=None,
                 control_state_labels=None, spell_target=None,
                 control_state_target=None, **kwargs):

        """ERPSpellerData constructor

        Parameters
        ----------
        mode : str {"train"|"test"}
            Mode of this run.
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
            and 6 for the second level of each command (letters). Then,
            the groups describe aggregations of commands that are highlighted
            at the same time. For instance, the row-column paradigm (RCP) [2]
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
        onsets : list or numpy.ndarray [n_stim x 1]
            Timestamp of each stimulation. This timestamps have to be
            synchronized with the EEG (or other biosignal) timestamps in
            order to assure a correct functioning of all medusa functions.
        batch_idx : list or numpy.ndarray [n_stim x 1]
            Index of the code of the highlighted batch for each stimulation.
            A batch represents the highlighted commands in each stimulation.
            For example in the row-col paradigm (RCP) represents each row and
            column.
        group_idx : list or numpy.ndarray [n_stim x 1]
            Index of the group that has been highlighted. Groups represent the
            different aggregations of batches. Between batches of different
            groups, 1 command must be common. For example in the RCP there
            are 2 groups: rows and columns. In this paradigm, between each
            pair of batches (e.g., row=2, col=4), there is only one command
            in common.
        unit_idx: list or numpy.ndarray [n_stim x 1]
            Index of the unit used in each stimulation. Units are low level
            entities used in multi-level paradigms, such as HOS paradigm [1].
            For each level, only 1 unit can be used. As the trial may have
            several layers, several units can be used in 1 trial. For
            instance, in the HOS, the first unit is the main menu. The other
            6 units are each of the lower level entities that are displayed
            in the second level of stimulation.
        level_idx : list or numpy.ndarray [n_stim x 1]
            Index of the level of each stimulation. Levels represent each
            one of the selections that must be made before a trial is
            finished. For example, in the Hex-O-spell paradigm there are 2
            levels (see [1]).
        matrix_idx : list or numpy.ndarray [n_stim x 1]
            Index of the matrix used in each stimulation. Each matrix can
            contain several levels. The matrix has to be the same accross the
            entire trial.
        sequence_idx : list or numpy.ndarray [n_stim x 1]
            Index of the sequence for each stimulation. A sequence
            represents a round of stimulation: all commands have been
            highlighted 1 time. This class support dynamic stopping in
            different levels.
        trial_idx : list or numpy.ndarray [n_stim x 1]
            Index of the trial for each stimulation. A trial represents
            the selection of a final command. Depending on the number of levels,
            the final selection takes N intermediate selections.
        spell_result : list or numpy.ndarray [n_trials x n_levels x 3]
            Spell result of this run. Each position contains the data of the
            selected target (matrix_idx, row, col)
        control_state_result : list or numpy.ndarray
            Control state result of this run. Each position contains the
            detected control state of the user (0 -> non-control, 1-> control)
        erp_labels : list or numpy.ndarray [n_stim x 1] {0|1}
            Only in train mode. Contains the erp labels of each stimulation
            (0 -> non-target, 1-> target)
        control_state_labels : list or numpy.ndarray [n_stim x 1] {0|1}
            Only in train mode. Contains the erp labels of each stimulation
            (0 -> non-control, 1-> control)
        spell_target : list or numpy.ndarray [n_trials x 3]
            Only in train mode. List containing the target for each trial.
            Each position represents (matrix_idx, row, col)
        control_state_target : list or numpy.ndarray [n_trials x 1]
            Only in train mode. List containing the control state target for
            each trial.
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

        # Standard attributes
        self.mode = mode
        self.onsets = onsets
        self.batch_idx = batch_idx
        self.group_idx = group_idx
        self.unit_idx = unit_idx
        self.level_idx = level_idx
        self.matrix_idx = matrix_idx
        self.sequence_idx = sequence_idx
        self.trial_idx = trial_idx
        self.paradigm_conf = paradigm_conf
        self.spell_result = spell_result
        self.control_state_result = control_state_result
        self.erp_labels = erp_labels
        self.control_state_labels = control_state_labels
        self.spell_target = spell_target
        self.control_state_target = control_state_target

        # Optional attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self):
        rec_dict = self.__dict__
        for key in rec_dict.keys():
            if type(rec_dict[key]) == np.ndarray:
                rec_dict[key] = rec_dict[key].tolist()
        return rec_dict

    @staticmethod
    def from_dict(dict_data):
        return ERPSpellerData(**dict_data)


class MIData(ExperimentData):

    # TODO: Check everything

    """Class with the necessary attributes to define motor imagery (MI)
    experiments. It provides compatibility with high-level functions for this
    MI paradigms in BCI module.
    """
    def __init__(self, mode, onsets, w_trial_t, mi_result, control_state_result,
                 mi_labels=None, control_state_labels=None, w_rest_t=None,
                 **kwargs):
        """MIData constructor

        Parameters
        ----------
        mode : str {"train"|"test"}
            Mode of this run.
        onsets : list or numpy.ndarray [n_stim x 1]
            Timestamp of each stimulation
        w_trial_t: list [start, end]
            Temporal window of the motor imagery with respect to each onset in
            ms. For example, if  w_trial_t = [500, 4000] the subject was
            performing the motor imagery task from 500ms after to 4000ms after
            the onset.
        mi_result : list or numpy.ndarray [n_trials x 3]
            Spell result of this run. Each position contains the data of the
            selected target (matrix_idx, row, col)
        control_state_result : list or numpy.ndarray
            Control state result of this run. Each position contains the
            detected control state of the user (0 -> non-control, 1-> control)
        mi_labels : list or numpy.ndarray [n_stim x 1] {0|1}
            Only in train mode. Contains the erp labels of each stimulation
            (0 -> non-target, 1-> target)
        control_state_labels : list or numpy.ndarray [n_stim x 1] {0|1}
            Only in train mode. Contains the erp labels of each stimulation
            (0 -> non-control, 1-> control)
         w_trial_t: list [start, end]
            Temporal window of the rest with respect to each onset in ms. For
            example, if w_rest_t = [-1000, 0] the subject was resting from
            1000ms before to the onset.
        kwargs : kwargs
            Custom arguments that will also be saved in the class
            (e.g., timings, calibration gaps, etc.)
        """

        # Check errors
        if mode == 'train':
            if mi_labels is None or control_state_labels is None:
                raise ValueError('Attributes mi_labels, control_state_labels '
                                 'be provided in train mode')

        # Standard attributes
        self.mode = mode
        self.onsets = np.array(onsets)
        self.w_trial_t = np.array(w_trial_t)
        self.mi_result = np.array(mi_result)
        self.control_state_result = np.array(control_state_result)
        self.mi_labels = np.array(mi_labels) if mi_labels is not None else \
            mi_labels
        self.control_state_labels = np.array(control_state_labels) \
            if control_state_labels is not None else control_state_labels

        # Optional attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self):
        rec_dict = self.__dict__
        for key in rec_dict.keys():
            if type(rec_dict[key]) == np.ndarray:
                rec_dict[key] = rec_dict[key].tolist()
        return rec_dict

    @staticmethod
    def from_dict(dict_data):
        return MIData(**dict_data)


def load_recording(path, data_format=None):
    """Loads the file with the correct data structures

    Parameters
    ----------
    path : str
        File path
    data_format : None or str
        File format. If None, the format will be given by the file extension

    Returns
    -------
    Recording
        Recording class with the correct data structures
    """
    # Check extension
    if data_format is None:
        ext = path.split('.')[-1]
    else:
        ext = data_format
    # Load file
    if ext == 'bson':
        return __load_recording_from_bson(path)
    elif ext == 'json':
        return __load_recording_from_json(path)
    elif ext == 'mat':
        return __load_recording_from_mat(path)
    elif ext == 'hdf5':
        return __load_recording_from_hdf5(path)
    else:
        raise TypeError('Unknown file format %s' % ext)


def __load_recording_from_bson(path):
    with open(path, 'rb') as f:
        rec_dict = bson.loads(f.read())
    return Recording.from_dict(rec_dict)


def __load_recording_from_json(path, encoding='utf-8'):
    with open(path, 'r', encoding=encoding) as f:
        rec_dict = json.load(f)
    return Recording.from_dict(rec_dict)


def __load_recording_from_mat(path):
    rec_dict = scipy.io.loadmat(path, squeeze_me=True, simplify_cells=True)
    return Recording.from_dict(rec_dict)


def __load_recording_from_hdf5(path):
    raise NotImplementedError


class ConsistencyChecker:
    """Class that provides functionality to check consistency across recordings
    to build a dataset
    """

    def __init__(self):
        self.__rules = list()

    def add_consistency_rule(self, rule, rule_params, parent=None):
        """Adds a consistency check for the specified attribute It provides 2
        levels of consistency using parameter key, enough to check attributes
        inside biosignal and experiments classes

        Parameters
        ----------
        rule : str {'check-attribute-type'|'check-attribute-value'|
        'check-values-in-attribute'|'check-if-attribute-exists'|
        'check-if-type-exists'}
            Check mode of this attribute. Modes:
                - check-attribute-type: checks if the attribute has the type
                    specified in parameter check_value
                - check-attribute-value: checks if the attribute has the value
                    specified in parameter check_value
                - check-values-in-attribute: checks if the attribute contains
                    the values (the attribute must support in operation). It can
                    check keys in dicts or values in lists or sets.
                - check-attribute: checks if the attribute exists
                - check-type: checks if the class contains attributes with the
                    specified type.
        rule_params : dict
            Specifies the rule params. Depending on the rule, it must contain
            the following key-value pairs:
                - check-attribute-type: {attribute: str, type: class}
                - check-attribute-value: {attribute: str, value: obj}
                - check-values-in-attribute: {attribute: str, values: list}
                - check-attribute: {attribute: str}
                - check-type: {type: class, limit: int, operator: str {'<'|'>'|'
                    <='|'>='|'=='|'!='}
        parent : str or None
            Checks the rule inside specified parent. If None, the parent is the
            recording itself. Therefore, the parent must be a class. This
            parameter designed to allow check rules inside biosignals or
            experiments class. If the parent is in deeper levels, use points to
            define the parent. For example, you can check the labels of the
            channels in an EEG recording setting this parameter as
            eeg.channel_set
        """
        # Check to avoid errors
        if rule == 'check-attribute-type':
            if not all(k in rule_params for k in ['attribute', 'type']):
                raise ValueError('Rule params must contain keys (attribute, '
                                 'type) for rule %s' % rule)
        elif rule == 'check-attribute-value':
            if not all(k in rule_params for k in ['attribute', 'value']):
                raise ValueError('Rule params must contain keys (attribute, '
                                 'value) for rule %s' % rule)
        elif rule == 'check-values-in-attribute':
            if not all(k in rule_params for k in ['attribute', 'values']):
                raise ValueError('Rule params must contain keys (attribute, '
                                 'values) for rule %s' % rule)
        elif rule == 'check-attribute':
            if not all(k in rule_params for k in ['attribute']):
                raise ValueError('Rule params must contain keys (attribute) '
                                 'for rule %s' % rule)
        elif rule == 'check-type':
            if not all(k in rule_params for k in ['type', 'limit', 'operator']):
                raise ValueError('Rule params must contain keys (type) for '
                                 'rule %s' % rule)
            if rule_params['operator'] not in {'<', '>', '<=', '>=', '==', '!='}:
                raise ValueError("Unknown operator %s. Possible operators: "
                                 "{'<'|'>'|'<='|'>='|'=='|'!='}" %
                                 rule_params['operator'])
        else:
            raise ValueError("Unknown rule. Possible rules: "
                             "{'check-attribute-type'|'check-attribute-value'|"
                             "'check-values-in-attribute'|"
                             "'check-if-attribute-exists'|"
                             "'check-if-type-exists'}")
        # Save rule
        self.__rules.append({'rule': rule, 'rule_params': rule_params,
                             'parent': parent})

    def check_consistency(self, recording):
        """Checks the consistency of a recording according to the current rules

        Parameters
        ----------
        recording : Recording
            Recording to be checked
        """
        # Check general attributes
        for r in self.__rules:
            rule = r['rule']
            rule_params = r['rule_params']
            if r['parent'] is None:
                parent = recording
            else:
                parent = recording
                for p in r['parent'].split('.'):
                    parent = getattr(parent, p)
            if rule == 'check-attribute-type':
                attribute = getattr(parent, rule_params['attribute'])
                if not isinstance(attribute, rule_params['type']):
                    raise TypeError('Type of attribute %s must be %s' %
                                    (rule_params['attribute'],
                                     str(rule_params['type'])))
            elif rule == 'check-attribute-value':
                attribute = getattr(parent, rule_params['attribute'])
                if attribute != rule_params['value']:
                    raise ValueError('Value of attribute %s must be %s' %
                                     (rule_params['attribute'],
                                      str(rule_params['value'])))
            elif rule == 'check-values-in-attribute':
                attribute = getattr(parent, rule_params['attribute'])
                for val in rule_params['values']:
                    if val not in attribute:
                        raise ValueError('Parameter %s must contain value %s' %
                                         (rule_params['attribute'],
                                          str(rule_params['values'])))
            elif rule == 'check-attribute':
                if not hasattr(parent, rule_params['attribute']):
                    raise ValueError('Attribute %s does not exist' %
                                     rule_params['attribute'])
            elif rule == 'check-type':
                # Get number of attributes with type
                n = 0
                for key, val in parent.__dict__.items():
                    if isinstance(val, rule_params['type']):
                        n += 1
                # Check
                if not self.__numeric_check(n, rule_params['limit'],
                                            rule_params['operator']):
                    raise ValueError('Number of attributes with type %s does '
                                     'not meet the rule (%i %s %i)' %
                                     (rule_params['type'], n,
                                      rule_params['operator'],
                                      rule_params['limit']))

    @staticmethod
    def __numeric_check(number, limit, operator):
        result = True
        if operator == '<':
            if number >= limit:
                result = False
        elif operator == '>':
            if number <= limit:
                result = False
        elif operator == '<=':
            if number > limit:
                result = False
        elif operator == '>=':
            if number < limit:
                result = False
        elif operator == '==':
            if number != limit:
                result = False
        elif operator == '!=':
            if number == limit:
                result = False
        else:
            raise ValueError("Unknown operator %s. Possible operators: "
                             "{'<'|'>'|'<='|'>='|'=='|'!='}" % operator)

        return result


class Dataset(ABC):
    """Class to handle multiple recordings maintaining consistency"""

    def __init__(self, consistency_checker=None):
        """Class constructor

        Parameters
        ----------
        consistency_checker : ConsistencyChecker
            Consistency checker for this dataset.
        """
        self.consistency_checker = consistency_checker
        self.recordings = list()

    def add_recordings(self, recordings):
        """Adds one or more recordings to the dataset, checking the consistency

        Parameters
        ----------
        recordings : list or medusa.data_structures.Recording
            List containing the paths to recording files or instances of
            Recording class
        """
        # Avoid errors
        recordings = [recordings] if type(recordings) != list else recordings
        # Add recordings
        for r in recordings:
            # Check if recording is instance of Recording of path
            if type(r) == str:
                recording = load_recording(r)
            elif type(r) == Recording:
                recording = r
            else:
                raise TypeError('Error at index %i: type has to be %s or %s' %
                                (recordings.index(r),
                                 type(str),
                                 type(Recording)))
            # Check consistency
            if self.consistency_checker is not None:
                self.consistency_checker.check_consistency(recording)
            # Append recording
            self.recordings.append(
                self.custom_operations_on_recordings(recording)
            )

    def custom_operations_on_recordings(self, recording):
        """Function add_recordings calls this function before adding each
        recording to the dataset. Implement this method in custom classes to
        have personalized behaviour (e.g., change the channel set)

        Parameters
        ----------
        recording : subclass of Recording
            Recording that will be changed. It can also be a subclass of
            Recording

        Returns
        -------
        recording : Recording
            Modified recording
        """
        return recording


class Connecitivity:

    # TODO: check everything

    """Customizable class with connectivity info from EEG/MEG recordings
    """

    def __init__(self, data, trial_len, parameter, filt_mode, **kwargs):
        """Class constructor

        Parameters
        ----------
        data : bla bla
            Bla bla
        trial_len : bla bla
            Bla bla
        parameter : bla bla
            Bla bla
        filt_mode : bla bla
            Bla bla
        kwargs
            Optional information of the EEG recording (e.g. subject, amplifier,
            etc)
        """
        # Params
        data = np.array(data)

        if not (filt_mode == 'all' or
                filt_mode == 'bands' or
                filt_mode == 'win'):
            raise ValueError("Unknown filtering mode")

        self.data = data
        self.trial_len = trial_len
        self.parameter = parameter
        self.filt_mode = filt_mode

        # Optional attributes
        for key, value in kwargs.items():
            setattr(self, key, value)