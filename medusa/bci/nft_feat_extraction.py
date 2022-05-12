# EXTERNAL LIBRARIES
import numpy as np
# MEDUSA MODULES
from medusa import frequency_filtering


def extract_nft_feature(trial, fs, cha_idx, bw):
    """
    This function computes the NFT features for online testing. It returns the power of the trial signal on the given
    band an electrodes
    """
    # Select the electrodes of interest
    trial = trial[:, cha_idx]
    # Filter the signal in the specified band
    [b, a] = frequency_filtering.filter_designer(bw, fs, 1000, ftype='FIR', btype='bandpass')
    trial = frequency_filtering.apply_filter_offline(trial, b, a, axis=0, method='filtfilt')
    # Compute the signal log-power
    power = np.log(np.sum(np.mean(np.power(trial, 2), axis=0))/trial.shape[0])

    return power

