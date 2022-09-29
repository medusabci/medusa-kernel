"""
Created on Tue Sep 27 10:45:15 2022

@author: Víctor Martínez-Cagigal
"""
import matplotlib.pyplot as plt
import numpy as np


def shaded_plot(data, error='std', times=None, axes=None):
    """ This function plots the mean and the shaded error of a time series.

    Parameters
    ----------------
    data: 2D numpy.ndarray
        Data being plotted, with dimensions [observations x signal]
    error: basestring
        Type of error being plotted (mean+error, mean-error), which can be:
        - 'std':    standard deviation
        - 'sem':    standard error mean
        - 'var':    variance
        - 'c95':    95% confidence interval (assuming normality)
    times: numpy.ndarray
        Timestamps of the time series.
    axes: matplotlib.pyplot.axes
        If a matplotlib axes are specified, the plot is displayed inside it.
        Otherwise, the plot will generate a new figure.

    Returns
    ----------------
    axes: matplotlib.pyplot.axes
    """
    # Error detection
    must_show = False
    if axes is None:
        axes = plt.subplot(111)
        must_show = True
    if not isinstance(data, (np.ndarray, np.generic)):
        data = np.array(data)
    if len(data.shape) != 2:
        raise ValueError("Parameter 'data' must have 2 dimensions!")
    if times is None:
        times = np.linspace(0, data.shape[1], data.shape[1])

    # Mean and std
    mean_ = np.mean(data, axis=0)
    std_ = np.std(data, axis=0)

    # Error
    if error == 'std':
        error_ = std_
    elif error == 'sem':
        error_ = std_ / np.sqrt(data.shape[0])
    elif error == 'var':
        error_ = std_ ** 2
    elif error == 'c95':
        # assuming normality
        error_ = (std_ / np.sqrt(data.shape[0])) * 1.96
    else:
        raise ValueError("Error %s not recognized! Please, use std, sem, "
                         "var or c95" % error)

    # Plot
    axes.fill_between(times, mean_ + error_, mean_ - error_, alpha=0.3)
    axes.plot(times, mean_, linewidth=2)
    if must_show:
        plt.show()

    return axes
