import pickle, os
import scipy.io
import warnings


class MedusaData:

    def __init__(self, **kwargs):

        # Set the specified arguments
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.eeg = None
        self.experiment = None

        DeprecationWarning('This class is deprecated and will be removed in '
                           'following versions of Medusa. Use the classes '
                           'provided in module data_structures, which use '
                           'multiplatform formats, have increased '
                           'compatibility and smarter design')

    def set_eeg_info(self, times, signal, fs, ncha, lcha, **kwargs):
        """ Sets the EEG timestamps, signal and info """
        self.eeg = self._EEG(times, signal, fs, ncha, lcha, **kwargs)

    def set_experiment_info(self, name, **kwargs):
        """ Sets the application events and information """
        self.experiment = self._Experiment(name, **kwargs)

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

    class _EEG:
        """
        Customizable class with EEG info

        :param times: ndarray
            Array with the timestamp of each EEG sample. Shape (N,). N is the number of EEG samples.
        :param signal: ndarray:
            Array with the timestamp of each EEG sample. Shape (N, ncha). N is the number of EEG samples.
        :param fs:
            Float: sample rate (Hz).
        :param lcha: list:
            List with the label of the channels (same order that signal columns)
        :param ncha: int:
            Number of channels
        :param **kwargs:
            Optional information of the EEG recording (e.g. amplifier)
        """
        def __init__(self, times, signal, fs, ncha, lcha, **kwargs):
            # Standard attributes
            self.times = times
            self.signal = signal
            self.fs = fs
            self.ncha = ncha
            self.lcha = lcha
            # Optional attributes
            for key, value in kwargs.items():
                setattr(self, key, value)

    class _Experiment:
        """
        Customizable class with EEG info

        :param name: ndarray
            Name of the experiment (e.g. ERP-based speller, Neurofeedback Training, etc)
        :param **kwargs:
            Information of the experiment. onsets, target, events, settings, etc..
        """
        def __init__(self, name, **kwargs):
            # Standard attributes
            self.name = name
            # Optional attributes
            for key, value in kwargs.items():
                setattr(self, key, value)

