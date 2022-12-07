import numpy as np


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
