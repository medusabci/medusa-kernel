import pickle, os, struct, json
import scipy.io
import numpy as np


def load_from_file(path):
    ext = path.split('.')[-1]
    if ext == 'mds':
        with open(path, 'rb') as f:
            binary_data = f.read()
        return pickle.loads(binary_data)
    elif ext == 'mat':
        print('*.mat data still not supported.')
    else:
        raise Exception("File extension not recognized")


class RCPData:
    """
    ERP speller row-column paradigm (RCP) info class

    """

    def __init__(self, mode, onsets, codes, sequences, trials, matrix_indexes, matrix_dims, matrix_labels, nseqs,
                 spell_result, control_state_result, erp_labels=None, control_state_labels=None, spell_target=None,
                 control_state_target=None, **kwargs):

        # Standard attributes
        self.mode = mode
        self.onsets = np.array(onsets)
        self.codes = np.array(codes)
        self.sequences = np.array(sequences)
        self.trials = np.array(trials)
        self.matrix_indexes = np.array(matrix_indexes)
        self.matrix_dims = np.array(matrix_dims)
        self.matrix_labels = np.array(matrix_labels)
        self.nseqs = nseqs
        self.spell_result = np.array(spell_result)
        self.control_state_result = np.array(control_state_result)
        self.erp_labels = np.array(erp_labels) if erp_labels is not None else erp_labels
        self.control_state_labels = np.array(control_state_labels) if control_state_labels is not None else control_state_labels
        self.spell_target = np.array(spell_target)
        self.control_state_target = np.array(control_state_target)

        # Optional attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

        DeprecationWarning('This class is deprecated and will be removed in '
                           'following versions of Medusa. Use the classes '
                           'provided in module data_structures, which use '
                           'multiplatform formats, have increased '
                           'compatibility and smarter design')


class CakeParadigmData:
    """
    Cake speller paradigm (CSP) from Berlin Brain-Computer Interface (BBCI) info class for ERP-based spellers.
    It is also valid for the center speller paradigm from the same research group.
    """

    def __init__(self, mode, onsets, codes, sequences, levels, trials, matrix_indexes, matrix_dims, matrix_labels, nseqs,
                 nlevels, spell_result, control_state_result, erp_labels=None, control_state_labels=None,
                 spell_target=None, control_state_target=None, **kwargs):

        # Standard attributes
        self.mode = mode
        self.onsets = onsets
        self.codes = codes
        self.sequences = sequences
        self.levels = levels
        self.trials = trials
        self.matrix_indexes = matrix_indexes
        self.matrix_dims = matrix_dims
        self.matrix_labels = matrix_labels
        self.nseqs = nseqs
        self.nlevels = nlevels
        self.spell_result = spell_result
        self.control_state_result = control_state_result
        self.erp_labels = erp_labels
        self.control_state_labels = control_state_labels
        self.spell_target = spell_target
        self.control_state_target = control_state_target

        # Optional attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

        DeprecationWarning('This class is deprecated and will be removed in '
                           'following versions of Medusa. Use the classes '
                           'provided in module data_structures, which use '
                           'multiplatform formats, have increased '
                           'compatibility and smarter design')


class ERPSpellerRun:

    def __init__(self, **kwargs):

        # Set the specified arguments
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.eeg = None
        self.paradigm = None
        self.paradigm_data = None

        DeprecationWarning('This class is deprecated and will be removed in '
                           'following versions of Medusa. Use the classes '
                           'provided in module data_structures, which use '
                           'multiplatform formats, have increased '
                           'compatibility and smarter design')

    def set_eeg_data(self, times, signal, fs, ncha, lcha, **kwargs):
        """
        Sets the EEG info

        :param times: timestamps of EEG samples
        :type times: list or numpy array

        :param signal: EEG signal with shape [samples x channels]
        :type signal: list or numpy array

        :param fs: sample rate
        :type fs: float

        :param ncha: number of channels
        :type ncha: int

        :param lcha: channel labels
        :type lcha: list

        """
        self.eeg = MEEG(signal, fs, ncha, lcha, times=times, **kwargs)

    def set_paradigm_data(self, paradigm, **kwargs):
        """
        Sets the paradigm info

        :param paradigm: paradigm used in this run
        :type paradigm: str
        """
        if paradigm == 'RCP':
            self.paradigm = paradigm
            self.paradigm_data = RCPData(**kwargs)
        elif paradigm == 'Cake paradigm' or paradigm == 'Center paradigm':
            self.paradigm = paradigm
            self.paradigm_data = CakeParadigmData(**kwargs)
        else:
            raise ValueError('Parameter paradigm must be one of {"RCP"}')

    def save_to_file(self, path):
        """ Saves the class to the specified extension in variable path """
        ext = path.split('.')[-1]
        if ext == 'mds':
            return self.save_to_mds_file(path)
        elif ext == 'mat':
            return self.save_to_mat_file(path)
        elif ext == 'json':
            return self.save_to_json_file(path)
        elif ext == 'bson':
            return self.save_to_bson_file(path)
        else:
            raise ValueError('Format %s is not available yet' % ext)

    def save_to_mds_file(self, path):
        """ Save the class in a .mds binary file using pickle"""
        data = pickle.dumps(self)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            f.write(data)

    def save_to_mat_file(self, path):
        """ Save the class in a MATLAB .mat file using scipy"""
        ext = path.split('.')[-1]
        path = path[:-len(ext)] + 'mat'
        scipy.io.savemat(path, mdict={'data': self})

    def save_to_json_file(self, path):
        raise ValueError('Format json is not available yet')

    def save_to_bson_file(self, path):
        raise ValueError('Format bson is not available yet')

    @staticmethod
    def load_from_file(path, data_format=None):
        # Check extension
        if data_format is None:
            ext = path.split('.')[-1]
        else:
            ext = data_format
        # Load file
        if ext == 'mds':
            return ERPSpellerRun.load_from_mds_file(path)
        elif ext == 'mat':
            return ERPSpellerRun.load_from_mat_file(path)
        elif ext == 'json':
            return ERPSpellerRun.load_from_json_file(path)
        elif ext == 'bson':
            return ERPSpellerRun.load_from_bson_file(path)
        else:
            raise Exception("File extension not recognized")

    @staticmethod
    def load_from_mds_file(path):
        with open(path, 'rb') as f:
            binary_data = f.read()
        return pickle.loads(binary_data)

    @staticmethod
    def load_from_mat_file(path):
        raise ValueError('Format mat is not available yet')

    @staticmethod
    def load_from_json_file(path, encoding='utf-8'):
        # Get JSON data
        with open(path, 'r', encoding=encoding) as f:
            json_data = json.load(f)

        # Pop EEG data and convert times to numpy array
        json_data_eeg = json_data.pop('eeg')
        json_data_eeg['times'] = np.array(json_data_eeg['times'])

        # Pop paradigm data and convert to numpy arrays
        json_data_paradigm = json_data.pop('paradigm')
        json_data_paradigm_data = json_data.pop('paradigm_data')

        # Parse JSON data to ERPSpellerRun
        data = ERPSpellerRun(**json_data)
        data.set_eeg_data(**json_data_eeg)
        data.set_paradigm_data(json_data_paradigm, **json_data_paradigm_data)
        return data

    @staticmethod
    def load_from_bson_file(path):
        raise ValueError('Format bson is not available yet')


class FPCData:
    """
    SSVEP frequency-phase coding (FPC) paradigm info class

    """

    def __init__(self, mode, onsets, matrix_indexes, matrix_dims, matrix_labels, matrix_coding, spell_result,
                 control_state_result, spell_target=None, control_state_target=None, **kwargs):

        # Standard attributes
        self.mode = mode
        self.onsets = onsets
        self.matrix_indexes = matrix_indexes
        self.matrix_dims = matrix_dims
        self.matrix_labels = matrix_labels
        self.matrix_coding = matrix_coding
        self.spell_result = spell_result
        self.control_state_result = control_state_result
        self.spell_target = spell_target
        self.control_state_target = control_state_target

        # Optional attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

        DeprecationWarning('This class is deprecated and will be removed in '
                           'following versions of Medusa. Use the classes '
                           'provided in module data_structures, which use '
                           'multiplatform formats, have increased '
                           'compatibility and smarter design')


class SSVEPSpellerRun:

    def __init__(self, **kwargs):

        # Set the specified arguments
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.eeg = None
        self.paradigm = None
        self.paradigm_data = None

    def set_eeg_data(self, times, signal, fs, ncha, lcha, **kwargs):
        """
        Sets the EEG info

        :param times: timestamps of EEG samples
        :type times: list or numpy array

        :param signal: EEG signal with shape [samples x channels]
        :type signal: list or numpy array

        :param fs: sample rate
        :type fs: float

        :param ncha: number of channels
        :type ncha: int

        :param lcha: channel labels
        :type lcha: list

        """
        self.eeg = MEEG(signal, fs, ncha, lcha, times=times, **kwargs)

    def set_paradigm_data(self, paradigm, **kwargs):
        """
        Sets the paradigm info

        :param paradigm: paradigm used in this run
        :type paradigm: str
        """
        if paradigm == 'FPC':
            self.paradigm = paradigm
            self.paradigm_data = FPCData(**kwargs)
        else:
            raise ValueError('Parameter paradigm must be one of {"FPC"}')

    def pack_run(self):
        """
        Packs the run in a simple binary string. This method allows to send the run through the network efficiently, but
        it is not recommended for storage. For that purpose, use
        """
        pass

    def save(self, path):
        """ Saves the class to the specified extension in variable path """
        ext = path.split('.')[-1]
        if ext == 'mds':
            return self.save_to_mds_file(path)
        elif ext == 'mat':
            return self.save_to_mat_file(path)
        else:
            raise ValueError("File extension not recognized")

    def save_to_mat_file(self, path):
        """ Save the class in a MATLAB .mat file using scipy"""
        ext = path.split('.')[-1]
        path = path[:-len(ext)] + 'mat'
        scipy.io.savemat(path, mdict={'data': self})

    def save_to_mds_file(self, path):
        """ Save the class in a .mds binary file using pickle"""
        try:
            data = pickle.dumps(self)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'wb') as f:
                f.write(data)
        except Exception as e:
            print(str(e))

    @staticmethod
    def load_from_file(path):
        ext = path.split('.')[-1]
        if ext == 'mds':
            with open(path, 'rb') as f:
                binary_data = f.read()
            return pickle.loads(binary_data)
        elif ext == 'mat':
            print('*.mat data still not supported.')
        else:
            raise Exception("File extension not recognized")


class CVEPData:
    """
    Coded Visual Evoked Potentials (CVEP) paradigm info class

    :param  mode: string
        Paradigm mode (e.g., "Train" or "Test")
    :param tau: int
        Lag difference between commands in samples.
    :param n_cycles: int
        Number of cycles for each trial.
    :param sequences:  numpy array [N_seq, L]
        LSFR sequences in a matrix with dims [N_seq, L], where L is the length of the m-sequence (e.g. L = 63)
    :param onsets: numpy array (N_cycles*N_targets,)
        Timestamps of the lag=0 onsets with dims (N_cycles,), where N_cycles is the number of cycles.
    :param cycles: numpy array (N_cycles*N_targets,)
        Cycles of each onset (e.g. [0,1,2,3,4,0,1,2,3,4...]).
    :param trials: numpy array (N_cycles*N_targets,)
        Trial identification of each cycle (e.g. [0,0,0,0,0,1,1,1,1,1...]).
    :param fps_resolution: float
        Target clock frequency in Hz (default = 60 Hz).

    :param target_seq: numpy array (N_cycles*N_targets,)
        Target sequence that user is attending to in training mode (e.g. [0,0,0,0,0,1,1,1,1,1...]).

    :param matrix_indexes: numpy array [N_cycles*N_targets,]
        Indexes of the current selected matrix for each onset.
    :param matrix_dims: numpy array of ints[N_matrices, N_rows, N_cols]
        Matrices' dimensions.
    :param matrix_labels: list of chars, dims [N_matrices, N_rows, N_cols, label]
        Label of each target.
    :param matrix_lags: numpy array of ints, dims [N_matrices, N_rows, N_cols, lag]
        Lag of each target.
    :param matrix_sequences_idx: numpy array of ints, dims [N_matrices, N_rows, N_cols, sequence_idx]
        Index of the sequence associated for each target.
    :param spell_result: numpy array of ints [N_targets, 3], where 3 are the matrix_idx, n_row and n_col of the target.
        Target coordinates of the spelling result word.
    :param spell_target: numpy array of ints [N_targets, 3], where 3 are the matrix_idx, n_row and n_col of the target.
        Target coordinates of the spelling target word.
    :param control_state_result: numpy array of ints [N_targets,]
        Result of the asynchronous detection of each result.
    :param control_state_target: numpy array of ints [N_targets,]
        Label of the asynchronous step for each target (1=control, 0=non-control).
    """

    def __init__(self, mode, tau, n_cycles, sequences, onsets, cycles, trials, fps_resolution=60,
                 target_sequence=None,
                 matrix_indexes=None, matrix_dims=None, matrix_labels=None, matrix_lags=None, matrix_sequences_idx=None, spell_result=None,
                 control_state_result=None, spell_target=None, control_state_target=None, neighbors_cell=None,
                 matrix_ignored_items=None, **kwargs):

        # Common attributes
        self.mode = mode
        self.tau = tau
        self.n_cycles = n_cycles
        self.sequences = sequences
        self.onsets = onsets
        self.cycles = cycles
        self.trials = trials
        self.fps_resolution = fps_resolution
        self.neighbors_cell = neighbors_cell

        # Train
        if mode == "Train":
            self.target_sequence = target_sequence

        # Test
        if mode == "Online":
            self.matrix_indexes = matrix_indexes
            self.matrix_dims = matrix_dims
            self.matrix_labels = matrix_labels
            self.matrix_lags = matrix_lags
            self.matrix_sequences_idx = matrix_sequences_idx
            self.matrix_ignored_items = matrix_ignored_items
            self.spell_result = spell_result
            self.control_state_result = control_state_result
            self.spell_target = spell_target
            self.control_state_target = control_state_target

        # Optional attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Compute the sequences signature to allow quick comparisons
        self.sequences_sign = []
        for i in range(sequences.shape[0]):
            str_ = ""
            str_ = str_.join([str(x) for x in sequences[i, :]])
            self.sequences_sign.append(str_)

    def add_attribute(self, key, value):
        setattr(self, key, value)


class CVEPSpellerRun:

    def __init__(self, **kwargs):

        # Set the specified arguments
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.eeg = None
        self.paradigm = None
        self.paradigm_data = None

    def set_eeg_data(self, times, signal, fs, ncha, lcha, **kwargs):
        """
        Sets the EEG info

        :param times: timestamps of EEG samples
        :type times: list or numpy array

        :param signal: EEG signal with shape [samples x channels]
        :type signal: list or numpy array

        :param fs: sample rate
        :type fs: float

        :param ncha: number of channels
        :type ncha: int

        :param lcha: channel labels
        :type lcha: list

        """
        self.eeg = EEG(signal, fs, ncha, lcha, times=times, **kwargs)

    def set_paradigm_data(self, paradigm, **kwargs):
        """
        Sets the paradigm info

        :param paradigm: paradigm used in this run
        :type paradigm: str
        """
        if paradigm == 'c-VEP Speller' or paradigm == 'c-VEP Connect Four':
            self.paradigm = paradigm
            self.paradigm_data = CVEPData(**kwargs)
        else:
            raise ValueError('Parameter paradigm must be one of {"CVEP"}')

    def pack_run(self):
        """
        Packs the run in a simple binary string. This method allows to send the run through the network efficiently, but
        it is not recommended for storage. For that purpose, use
        """
        pass

    def save(self, path):
        """ Saves the class to the specified extension in variable path """
        ext = path.split('.')[-1]
        if ext == 'mds':
            return self.save_to_mds_file(path)
        elif ext == 'mat':
            return self.save_to_mat_file(path)
        else:
            raise ValueError("File extension not recognized")

    def save_to_mat_file(self, path):
        """ Save the class in a MATLAB .mat file using scipy"""
        ext = path.split('.')[-1]
        path = path[:-len(ext)] + 'mat'
        scipy.io.savemat(path, mdict={'data': self})

    def save_to_mds_file(self, path):
        """ Save the class in a .mds binary file using pickle"""
        try:
            data = pickle.dumps(self)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'wb') as f:
                f.write(data)
        except Exception as e:
            print(str(e))

    @staticmethod
    def load_from_file(path):
        ext = path.split('.')[-1]
        if ext == 'mds':
            with open(path, 'rb') as f:
                binary_data = f.read()
            return pickle.loads(binary_data)
        elif ext == 'mat':
            print('*.mat data still not supported.')
        else:
            raise Exception("File extension not recognized")
