import numpy as np
from medusa import components


class EMG(components.BiosignalData):
    """Electromiography (EMG) biosignal data class.
    """

    def __init__(self, times, signal, fs, channel_set, location=None, **kwargs):
        """EMG constructor

        Parameters
        ----------
        times : list or numpy.ndarray
            1D numpy array [n_samples]. Timestamps of each sample. If they are
            not available, generate them
            artificially. Nevertheless, all signals and events must have the
            same temporal origin
        signal : list or numpy.ndarray
            2D numpy array [n_samples x n_channels]. EMG samples (the units
            should be defined using kwargs)
        fs : int or float
            Sample rate of the recording.
        channel_set : list or Object
            Channel information
        location : string
            Location of the recording (e.g., quadriceps)
        kwargs: kwargs
            Key-value arguments to be saved in the class. This general class
            does not check anything
        """

        # Standard attributes
        self.times = times
        self.signal = signal
        self.fs = fs
        self.channel_set = channel_set
        self.location = location

        # Set the specified arguments
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