import numpy as np


def itr(accuracy, n_commands, selections_per_min):
    """
    Calculates the information transfer rate (ITR) in bits/min. This metric is
    widely used to assess the performance of a BCI. However, it has serious
    limitations for online systems: (i) it assumes that the probability of
    selecting all symbols is the same; (ii) the system is memoryless; (iii) a
    synchronous paradigm is used; (iv) users cannot correct mistakes.

    Parameters
    ---------------
    accuracy: float or list of floats
        Accuracy of the system (between 0 and 1).
    n_commands: int
        Number of possible commands to be selected.
    selections_per_min: float
        Number of selections that have been performed in a minute.

    Returns
    ----------------
    itr: float or list of floats
        ITR corresponding to each accuracy in bits per min (bpm).
    """
    # Special cases
    if accuracy == 0:
        # If accuracy is 0%, then ITR is 0 bpm
        itr = 0
    elif accuracy == 1:
        # If accuracy is 100%, then we take the mathematical limit
        itr = np.log2(n_commands) * selections_per_min
    else:
        # Otherwise: common ITR formula
        itr = (np.log2(n_commands) + accuracy * np.log2(accuracy) +
               (1 - accuracy) * np.log2((1 - accuracy) / (n_commands - 1))) \
              * selections_per_min

    return itr