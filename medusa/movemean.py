import numpy as np


def movemean(s, points):
    """
    Simple implementation of moving average.

    :param s: numpy array
        Signal [n_samples x n_channels]
    :param points: int
        Number of points considered to compute each window mean
    """
    s_filt = list()
    for x in range(len(s)):
        if x < int(points/2):
            val = np.mean(s[0:x+int(points/2), :], axis=0)
        elif x > len(s)-int(points/2):
            val = np.mean(s[x-int(points/2):, :], axis=0)
        else:
            val = np.mean(s[x-int(points/2):x+int(points/2), :], axis=0)
        s_filt.append(val)
    return np.array(s_filt)
