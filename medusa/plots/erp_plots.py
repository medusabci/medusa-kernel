"""Created on Friday October 01 10:09:11 2021

In this module you will find useful functions and classes to plot event-related
potentials (ERPs). This module is not finished, it has numerous improvement
points but can be useful for a quick plot. Enjoy!

@author: Eduardo Santamaría-Vázquez
"""
from medusa.bci import erp_spellers
import numpy as np
import copy


def plot_erp_from_erp_speller_dataset(erp_speller_dataset, channel, axes,
                                      window=(0, 1000)):
    """
    Plots Event-Related Potentials (ERPs) from an ERP Speller dataset.

    Parameters
    ----------
    erp_speller_dataset : ERPDataset
        The ERP Speller dataset containing EEG data and relevant information.

    channel : str
        The name of the EEG channel to be plotted.

    axes : matplotlib.axes.Axes
        The Matplotlib axes on which the ERP plot will be displayed.

    window : tuple, optional
        The time window for which the ERPs will be plotted, specified as a tuple
        (start_time, end_time). Default is (0, 1000) milliseconds.

    Returns
    -------
    matplotlib.axes.Axes
        The Matplotlib axes on which the ERP plot is displayed.

    Raises
    ------
    ValueError
        If the dataset is missing essential information or if the dataset mode
        is not set to 'train'.

    Notes
    -----
    This function performs standard preprocessing and feature extraction on the
    input ERP Speller dataset and then plots the ERPs based on the specified
    channel and time window.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from your_module import ERPDataset, plot_erp_from_erp_speller_dataset

    >>> # Assuming erp_speller_dataset is an instance of ERPDataset
    >>> fig, ax = plt.subplots()
    >>> plot_erp_from_erp_speller_dataset(erp_speller_dataset, channel='Fz', axes=ax)

    """
    # Check errors
    if erp_speller_dataset.fs is None:
        raise ValueError('Define the fs of the dataset')
    if erp_speller_dataset.channel_set is None:
        raise ValueError('Define the channel set of the dataset')
    if erp_speller_dataset.experiment_mode != 'train':
        raise ValueError('The dataset mode must be train')
    # Create copy of the dataset
    dataset = copy.deepcopy(erp_speller_dataset)
    # Standard preprocessing
    preprocessing_pipeline = erp_spellers.StandardPreprocessing(cutoff=(1, 30))
    dataset = preprocessing_pipeline.fit_transform_dataset(dataset)
    # Standard feature extraction
    feat_extraction_pipeline = erp_spellers.StandardFeatureExtraction(
        concatenate_channels=False, target_fs=None)
    x, x_info = feat_extraction_pipeline.transform_dataset(dataset)
    erp_labels = np.array(x_info['erp_labels'])
    # Call plot ERP
    return plot_erp(axes=axes,
                    erp_epochs=x[erp_labels==1, :, :],
                    noerp_epochs=x[erp_labels==0, :, :],
                    channel=channel,
                    window=window)


def plot_erp(axes, erp_epochs, noerp_epochs, channel, window=(0, 1000),
             error_measure="C95"):
    """Function designed to quickly plot an ERP with 95% confidence interval.
    It does offer limited functions that will be improved in the future.

    TODO: a lot of things, very basic functionality


    Parameters
    ----------
    axes : matplotlib.Axes.axes
        Matplotlib axes in which the ERP will be displayed into.
    erp_epochs: numpy.ndarray
        Epochs that contain ERPs (go epochs)
    noerp_epochs: numpy.ndarray
        Epochs that do not contain ERPs (nogo epochs)
    channel: int
        Channel index to plot
    window: list
        List with the lower and upper window time in milliseconds
    error_measure: str
        Error measure (default: "C95" or 95% confidence interval). Check
        parameters of function compute_dev_epochs() for further information.
    plot: bool
        Set to True to plot the ERP

    Returns
    -------
    erp_mean: numpy.ndarray
        ERP activity (mean of the go epochs)
    erp_dev: numpy.ndarray
        Error measure across observations for ERP activity
    noerp_mean: numpy.ndarray
        Non-ERP activity (mean of the nogo epochs)
    noerp_dev: numpy.ndarray
        Error measure across observations for non-ERP activity
    """
    # Select channel
    erp_epochs = erp_epochs[:, :, channel]
    noerp_epochs = noerp_epochs[:, :, channel]

    # Calculate mean and dev measures
    trials_erp_mean = np.mean(erp_epochs, 0)
    trials_erp_dev_pos, trials_erp_dev_neg = \
        compute_dev_epochs(erp_epochs, measure=error_measure)
    trials_noerp_mean = np.mean(noerp_epochs, 0)
    trials_noerp_dev_pos, trials_noerp_dev_neg = \
        compute_dev_epochs(noerp_epochs, measure=error_measure)

    # Plot the data
    t = np.linspace(window[0], window[1], trials_erp_mean.shape[0])
    axes.plot(t, trials_erp_mean)
    axes.fill_between(t, trials_erp_dev_neg, trials_erp_dev_pos, alpha=0.3)
    axes.plot(t, trials_noerp_mean)
    axes.fill_between(t, trials_noerp_dev_neg, trials_noerp_dev_pos,
                      alpha=0.3)

    # Return data
    plot_data = dict()
    plot_data["trials_erp_mean"] = trials_erp_mean
    plot_data["trials_erp_dev"] = (trials_erp_dev_pos, trials_erp_dev_neg)
    plot_data["trials_noerp_mean"] = trials_noerp_mean
    plot_data["trials_noerp_dev"] = (trials_noerp_dev_pos, trials_noerp_dev_neg)

    return plot_data



def compute_dev_epochs(epochs, measure="C95"):
    """ Computes the error of a 2D data.

    Parameters
    -------------
    epochs: ndarray
        Data being plotted, with dimensions [observations x signal]
    error: basestring
        Type of error being plotted (mean+error, mean-error), which can be:
        - 'std':    standard deviation
        - 'sem':    standard error mean
        - 'var':    variance
        - Confidence interval:  For this error, the measure parameter must be
        constituted by 'c' and the desired percentile. E.g. 'c95' for the
        95% confidence interval, 'c90' for the 90%, 'c99' for the 99%, and
        so on.

    Returns
    ----------------
    pos_deviation: ndarray
        1D vector containing the positive deviation measure [1 x signal].
    neg_deviation: ndarray
        1D vector containing the negative deviation measure [1 x signal].
    """
    # Error detection
    measure = measure.upper()
    percentile = 95
    if measure.startswith('C'):
        percentile = int(measure.split('C')[-1])
        if percentile >= 100 or percentile <= 0:
            raise ValueError("[compute_dev_epochs] The confidence interval "
                             "percentile (%i) must be in the range (0, 100)" %
                             percentile)

    # Compute deviation measure
    if measure.startswith('C'):
        pos = np.percentile(epochs, percentile, axis=0)
        neg = np.percentile(epochs, 100 - percentile, axis=0)
        return pos, neg
    elif measure == "STD":
        pos = np.mean(epochs, axis=0) + np.std(epochs, axis=0)
        neg = np.mean(epochs, axis=0) - np.std(epochs, axis=0)
        return pos, neg
    elif measure == "VAR":
        pos = np.mean(epochs, axis=0) + np.var(epochs, axis=0)
        neg = np.mean(epochs, axis=0) - np.var(epochs, axis=0)
        return pos, neg
    else:
        raise ValueError("[compute_dev_epochs] Unknown deviation measure %s!"
                         % measure)
