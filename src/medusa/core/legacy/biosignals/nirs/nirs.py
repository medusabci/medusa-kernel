"""Near-infrared spectroscopy (NIRS) biosignal."""
from medusa.core.legacy.signal import Signal
class NIRS(Signal):
    """Near-infrared spectroscopy (NIRS) biosignal.
    Parameters
    ----------
    times : array_like
        1D array `[n_samples]` with the timestamp of each sample.
    signal : array_like
        2D array `[n_samples x n_channels]` with the NIRS samples (units
        defined via kwargs).
    fs : int or float
        Sampling rate (Hz).
    channel_set : list or object
        Channel / optode information.
    **kwargs
        Extra metadata stored as instance attributes.
    """
    def __init__(self, times, signal, fs, channel_set, **kwargs):
        super().__init__(
            times=times,
            signal=signal,
            fs=fs,
            channel_set=channel_set,
            **kwargs,
        )
    @classmethod
    def from_serializable_obj(cls, dict_data):
        return cls(**dict_data)