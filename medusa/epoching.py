import numpy as np
from scipy import signal


def get_epochs(signal, epochs_length, stride=None, norm=None):
    """This function returns the signal divided in epochs following a sliding
    window approach

   Parameters
   ----------
   signal : list or numpy.ndarray
       Array to extract epochs with shape [samples, channels]
   epochs_length : int
       Epochs length in samples
   stride : int, optional
       Separation between consecutive epochs in samples. If None, stride is set
       to epochs_length.
   norm : str, optional
       Set to 'z' for Z-score normalization or 'dc' for DC normalization.
       Statistical parameters are computed using the whole epoch.

   Returns
   -------
   numpy.ndarray
       Structured data with dimensions [n_epochs x length x n_channels]
    """
    # Check errors
    if epochs_length <= 0:
        ValueError('Parameter epochs_length must be greater than 0')
    if stride is None:
        stride = epochs_length
    if stride <= 0:
        raise ValueError('Parameter stride must be None or greater than 0')
    # Assure ints
    n_cha = signal.shape[1]
    epochs_length = int(epochs_length)
    stride = int(stride)
    # Get epochs
    epochs = np.lib.stride_tricks.sliding_window_view(
        signal, (epochs_length, n_cha))[::stride].squeeze()
    # Normalize
    if norm is not None:
        epochs = normalize_epochs(epochs, norm)
    return epochs


def get_epochs_of_events(timestamps, signal, onsets, fs, w_epoch_t,
                         w_baseline_t=None, norm=None):

    """This function calculates the epochs of signal given the onsets of events.

    Parameters
    ----------
    timestamps : list or numpy.ndarray
        Timestamps of each biosignal sample
    signal : list or numpy.ndarray
        Biosignal samples
    onsets : list or numpy.ndarray
        Events timestamps
    fs : float
        Sample rate
    w_epoch_t : list or numpy.ndarray
        Temporal window in ms of the epoch. For example, t_baseline = [0, 1000]
        takes the epoch form 0 ms to 1000 ms after each onset (0 ms represents
        the onset).
    w_baseline_t : list or numpy.ndarray, optional
        Temporal window in ms of the baseline. For example, t_baseline =
        [-500, 100] takes the baseline form -500 ms before each onset to 100
        ms after each onset (0 ms represents the onset). This chunk of signal is
        used to normalize the epoch, if aplicable
    norm : str, optional
       Set to 'z' for Z-score normalization or 'dc' for DC normalization.
       Statistical parameters are computed using the whole epoch.
    Returns
    -------
    numpy.ndarray
        Structured data with dimensions [events x samples x channels]
    """
    # Error prevention
    epoch_feasibility = check_epochs_feasibility(timestamps, onsets, fs,
                                                 w_epoch_t)
    if epoch_feasibility == 'first':
        raise ValueError("Not enough EEG samples to get the first epoch")
    elif epoch_feasibility == 'last':
        raise ValueError("Not enough EEG samples to get the last epoch")
    if w_baseline_t is not None:
        baseline_feasibility = check_epochs_feasibility(timestamps, onsets, fs,
                                                        w_baseline_t)
        if baseline_feasibility == 'first':
            raise ValueError("Not enough EEG samples to get the first baseline")
        elif baseline_feasibility == 'last':
            raise ValueError("Not enough EEG samples to get the last baseline")
        if norm is None:
            raise ValueError('If parameter w_baseline_t is not None, please '
                             'specify the normalization type with parameter '
                             'norm')
    if norm is not None and w_baseline_t is None:
        raise ValueError('If parameter norm is not None, please specify the '
                         'baseline window with parameter w_baseline_t')

    # Useful parameters
    w_epoch_t = np.array(w_epoch_t)
    w_epoch_s = np.array(w_epoch_t * fs / 1000, dtype=int)
    l_epoch = w_epoch_s[1] - w_epoch_s[0]
    # For each onset
    n_cha = signal.shape[1]
    epochs_idx = np.searchsorted(timestamps, onsets + w_epoch_t[0] / 1000,
                                 side='left')
    epochs = np.lib.stride_tricks.sliding_window_view(
        signal, (l_epoch, n_cha))[epochs_idx].squeeze(axis=1)
    # Baseline normalization
    if w_baseline_t is not None and norm is not None:
        # Baseline start-end (samples)
        w_baseline_t = np.array(w_baseline_t)
        w_baseline_s = np.array(w_baseline_t * fs / 1000, dtype=int)
        l_baseline = w_baseline_s[1] - w_baseline_s[0]
        # Extract baselines
        baseline_idx = np.searchsorted(
            timestamps, onsets + w_baseline_t[0] / 1000, side='left')
        baselines = np.lib.stride_tricks.sliding_window_view(
            signal, (l_baseline, n_cha))[baseline_idx].squeeze(axis=1)
        epochs = normalize_epochs(epochs, norm_epochs=baselines, norm=norm)
    return epochs


def normalize_epochs(epochs, norm_epochs=None, norm='z'):
    """
    Normalizes epochs

    Parameters
    ----------
    epochs: list or numpy.ndarray
        Epochs of signal with dimensions [n_epochs x n_samples x n_channels]
    norm_epochs: list or numpy.ndarray, optional
        Epochs of signal with dimensions [n_epochs x n_samples x n_channels]
        that are used to compute the statistical parameters for normalization.
        If None, norm_epochs=epochs.
    norm: str
        Set to 'z' for Z-score normalization or 'dc' for DC normalization.
        Statistical parameters are computed using the whole epoch.
    """
    # Check errors
    if norm_epochs is None:
        norm_epochs = epochs
    if norm not in ['z', 'dc']:
        raise ValueError("Parameter norm must be 'z' or 'dc'")
    # Normalization
    if norm == 'z':
        # z-score
        mean = np.mean(norm_epochs, axis=1, keepdims=True)
        std = np.std(norm_epochs, axis=1, keepdims=True)
        epochs = (epochs - mean) / std
    elif norm == 'dc':
        # DC subtraction
        mean = np.mean(norm_epochs, axis=1, keepdims=True)
        epochs = epochs - mean
    else:
        raise ValueError("Parameter norm must be 'z' or 'dc'")
    return epochs


def resample_epochs(epochs, t_window, target_fs):
    """Resample epochs to the target_fs.

    IMPORTANT: No antialising filter is applied

    Parameters
    ----------
    epochs : list or numpy.ndarray
        Epochs of signal with dimensions [n_epochs x samples x channels]
    t_window : list or numpy.ndarray
        Temporal window in ms of the epoch. For example, t_window = [0, 1000]
        takes the epoch form 0 ms to 1000 ms after each onset (0 ms
        represents the onset).
    target_fs : float
        Target sample rate

    Returns
    -------
    numpy.ndarray
        Final epochs with dimensions [events x target_samples x channels]
    """
    # Compute the desired window and baseline in samples
    l_t_window = t_window[1] - t_window[0]  # Window length (ms)
    target_samples = np.floor((target_fs * l_t_window) / 1000).astype(int)
    # Resample epochs
    epochs = signal.resample(epochs, target_samples, axis=1)
    return epochs


def check_epochs_feasibility(timestamps, onsets, fs, t_window):
    """Checks if the extraction of the desired window is feasible with the
    available samples. Sometimes, the first/last stimulus sample is so close
    to the beginning/end of the signal data chunk that there are not enough
    samples to compute this window. In this case, the function will return
    "first" or "last" to identify which onset can not be extracted with
    the current t_window. It will return 'ok' if there is no conflict.

    Parameters
    ----------
     timestamps : list or numpy.ndarray
        Timestamps of each biosginal sample
    onsets : list or numpy.ndarray
        Events timestamps
    fs : float
        Sample rate
    t_window : list or numpy.ndarray
        Temporal window in ms of the epoch. For example, t_window = [0, 1000]
        takes the epoch form 0 ms to 1000 ms after each onset (0 ms represents
        the onset).

    Returns
    -------
    feasibility : string
        "ok" If window extraction is feasible.
        "first" If window extraction is not feasible for the first onset.
        "last" If window extraction is not feasible for the last onset.
    """
    first_sti_sam_onset = np.argmin(np.abs(timestamps - onsets[0]))
    last_sti_sam_onset = np.argmin(np.abs(timestamps - onsets[-1]))
    s_window = np.array(np.array(t_window) * fs / 1000, dtype=int)
    last_sti_sam_end = last_sti_sam_onset + s_window[1]
    first_sti_sam_beginning = first_sti_sam_onset + s_window[0]

    if first_sti_sam_beginning < 0:
        feasibility = 'first'
    elif last_sti_sam_end >= len(timestamps):
        feasibility = 'last'
    else:
        feasibility = 'ok'
    return feasibility


def reject_noisy_epochs(epochs, signal_mean, signal_std, k=4, n_samp=2,
                        n_cha=1):
    """Simple thresholding method to reject noisy epochs. It discards epochs
    with n_samp samples greater than k*std in n_cha channels

    Parameters
    ----------
     epochs : list or numpy.ndarray
        Epochs of signal with dimensions [n_epochs x samples x channels]
    signal_mean : float
        Mean of the signal
    signal_std : float
        Standard deviation of the signal
    k : float
        Standard deviation multiplier to calculate threshold
    n_samp : int
        Minimum number of samples that have to be over the threshold in each
        epoch to be discarded
    n_cha : int
        Minimum number of channels that have to have n_samples over the
        threshold in each epoch to be discarded

    Returns
    -------
     float
        Percentage of reject epochs in
    numpy.ndarray
        Clean epochs
    numpy.ndarray
        Indexes for rejected epochs. True for discarded epoch
    """

    # Check errors
    if len(epochs.shape) != 3:
        raise Exception('Malformed epochs array. It must be of dimmensions '
                        '[epochs x samples x channels]')
    if signal_std.shape[0] != epochs.shape[2]:
        raise Exception('Array signal_std does not match with epochs size. '
                        'It must have the same number of channels')
    if signal_mean.shape[0] != epochs.shape[2]:
        raise Exception('Array signal_mean does not match with epochs size. '
                        'It must have the same number of channels')

    epochs_abs = np.abs(epochs)
    cmp = epochs_abs > np.abs(signal_mean) + k * signal_std
    idx = np.sum((np.sum(cmp, axis=1) >= n_samp), axis=1) >= n_cha
    pct_rejected = (np.sum(idx) / epochs.shape[0]) * 100
    return pct_rejected, epochs[~idx, :, :], idx


def time_to_sample_index_events(times, onsets):
    """Converts an array of time onsets to an array that indicates the sample
    index of the event

        Parameters
        ----------
        times : list or numpy.ndarray
            Array of shape [n_samples]. Timestamps of the biosignal
        onsets : list or numpy.ndarray
            Array of shape [n_events]. Onsets in time of the events

        Returns
        -------
        numpy.ndarray
            Array of shape [n_samples]. The array is 1 only in the nearest
            timestamp to each onset, otherwise 0
        """

    # Check errors
    times = np.squeeze(times)
    onsets = np.squeeze(onsets)

    if len(times.shape) != 1:
        raise ValueError('Parameter times must be a 1D array')
    if len(onsets.shape) != 1:
        raise ValueError('Parameter onsets must be a 1D array')

    rep_times = np.matlib.repmat(times, onsets.shape[0], 1).T
    rep_onsets = np.matlib.repmat(onsets, times.shape[0], 1)

    return np.argmin(np.abs(rep_times - rep_onsets), axis=0)
