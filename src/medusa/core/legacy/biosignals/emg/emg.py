"""Electromyography (EMG) biosignal."""
from medusa.core.legacy.signal import Signal
class EMG(Signal):
    """Electromyography (EMG) biosignal.
    Parameters
    ----------
    times : array_like
        1D array `[n_samples]` with the timestamp of each sample.
    signal : array_like
        2D array `[n_samples x n_channels]` with the EMG samples (units
        defined via kwargs).
    fs : int or float
        Sampling rate (Hz).
    channel_set : list or object
        Channel information.
    location : str, optional
        Location of the recording (e.g. quadriceps).
    **kwargs
        Extra metadata stored as instance attributes.
    """
    def __init__(self, times, signal, fs, channel_set, location=None, **kwargs):
        super().__init__(
            times=times,
            signal=signal,
            fs=fs,
            channel_set=channel_set,
            location=location,
            **kwargs,
        )
    @classmethod
    def from_serializable_obj(cls, dict_data):
        return cls(**dict_data)