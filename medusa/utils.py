import numpy as np


def check_dimensions(data):
    """This function checks if data dimensions are [n_epochs x n_samples x
    n_channels] and transforms to it if not"""

    # If dimension is samples
    if len(data.shape) == 1:
        return data[np.newaxis, :, np.newaxis]
    # If dimensions are samples and channels
    elif len(data.shape) == 2:
        return data[np.newaxis, :, :]
    else:
        return data

