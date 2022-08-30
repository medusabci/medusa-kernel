"""Created on Friday October 01 10:09:11 2021

In this module you will find useful functions and classes to plot event-related
potentials (ERPs). This module is not finished, it has numerous improvement
points but can be useful for a quick plot. Enjoy!

@author: Eduardo Santamaría-Vázquez
"""
from medusa import frequency_filtering, spatial_filtering
from medusa import epoching
from medusa import components
import matplotlib.pyplot as plt
import numpy as np
import copy


def plot_erp_from_erp_speller_runs(erp_speller_runs, channel,
                                   window=(0, 1000), plot=True):
    data = copy.copy(erp_speller_runs)
    # Error handling. Data can be a list of ERPData instances or an ERPData instance
    if not isinstance(data, list):
        data = [data]
    # Load data
    trials_erp_epochs = list()
    trials_noerp_epochs = list()
    for d in data:
        if not isinstance(d, components.Recording):
            raise ValueError("")
        # Preprocessing
        filter = frequency_filtering.FIRFilter(
            order=500, cutoff=[0.5, 30], btype="bandpass", width=None,
            window='hamming', scale=True, filt_method='filtfilt', axis=0)
        d.eeg.signal = filter.fit_transform(d.eeg.signal, d.eeg.fs)
        d.eeg.signal = spatial_filtering.car(d.eeg.signal)
        # Extract epochs
        epochs = epoching.get_epochs_of_events(timestamps=d.eeg.times,
                                               signal=d.eeg.signal,
                                               onsets=d.erpspellerdata.onsets,
                                               fs=d.eeg.fs,
                                               w_epoch_t=window,
                                               w_baseline_t=[-200, 0],
                                               norm='z')
        # Epochs
        erp_epochs_idx = np.array(d.erpspellerdata.erp_labels) == 1
        noerp_epochs_idx = np.array(d.erpspellerdata.erp_labels) == 0
        erp_epochs = epochs[erp_epochs_idx, :, :]
        noerp_epochs = epochs[noerp_epochs_idx, :, :]

        # Save
        trials_erp_epochs.append(erp_epochs)
        trials_noerp_epochs.append(noerp_epochs)

    # To numpy array
    trials_erp_epochs = np.array(trials_erp_epochs)
    trials_noerp_epochs = np.array(trials_noerp_epochs)

    # Call plot ERP
    return plot_erp(erp_epochs=trials_erp_epochs,
                    noerp_epochs=trials_noerp_epochs,
                    channel=channel,
                    window=window,
                    plot=plot)


def plot_erp(erp_epochs, noerp_epochs, channel, window=(0, 1000), plot=True):
    """Function designed to quickly plot an ERP with 95% confidence interval.
    It does offer limited functions that will be improved in the future.

    TODO: a lot of things, very basic functionality


    Parameters
    ----------
    erp_epochs: numpy.ndarray
        Epochs that contain ERPs (go epochs)
    noerp_epochs: numpy.ndarray
        Epochs that do not contain ERPs (nogo epochs)
    channel: int
        Channel index to plot
    window: list
        List with the lower and upper window time in milliseconds
    plot: bool
        Set to True to plot the ERP

    Returns
    -------
    erp_mean: numpy.ndarray
        ERP activity (mean of the go epochs)
    erp_dev: numpy.ndarray
        95% confidence interval
    noerp_mean: numpy.ndarray
        No ERP activity (mean of the nogo epochs)
    noerp_dev: numpy.ndarray
        95% confidence interval
    """
    # Select channel
    erp_epochs = erp_epochs[:, :, channel]
    noerp_epochs = noerp_epochs[:, :, channel]

    # Calculate mean and dev measures
    trials_erp_mean = np.mean(erp_epochs, 0)
    trials_erp_dev = compute_dev_epochs(erp_epochs)
    trials_noerp_mean = np.mean(noerp_epochs, 0)
    trials_noerp_dev = compute_dev_epochs(noerp_epochs)

    if plot:
        # Plot the data
        t = np.linspace(window[0], window[1], trials_erp_mean.shape[0])
        plt.plot(t, trials_erp_mean)
        plt.fill_between(t, trials_erp_mean + trials_erp_dev,
                         trials_erp_mean - trials_erp_dev, alpha=0.3)
        plt.plot(t, trials_noerp_mean)
        plt.fill_between(t, trials_noerp_mean + trials_noerp_dev,
                         trials_noerp_mean - trials_noerp_dev, alpha=0.3)
        plt.show()

    # Return data
    plot_data = dict()
    plot_data["trials_erp_mean"] = trials_erp_mean
    plot_data["trials_erp_dev"] = trials_erp_dev
    plot_data["trials_noerp_mean"] = trials_noerp_mean
    plot_data["trials_noerp_dev"] = trials_noerp_dev

    return plot_data


def compute_dev_epochs(epochs, measure="C95"):
    # Compute mean and std
    std = np.std(epochs, 0)
    # Compute deviation measure
    dev = np.zeros([1, epochs.shape[1]])
    if measure == "C95":
        dev = (std / np.sqrt(epochs.shape[1])) * 1.96
    elif measure == "STD":
        dev = std
    elif measure == "VAR":
        dev = np.square(std)
    return dev
