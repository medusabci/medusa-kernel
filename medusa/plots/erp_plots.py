from medusa import frequency_filtering, spatial_filtering
from medusa import epoching
from medusa import components
import matplotlib.pyplot as plt
import numpy as np
import copy


def plot_erp(erp_speller_runs, channel, window=(0, 1000), plot=True):
    data = copy.copy(erp_speller_runs)
    # Error handling. Data can be a list of ERPData instances or an ERPData instance
    if not isinstance(data, list):
        data = [data]
    # Load data
    trials_erp_mean = list()
    trials_erp_dev = list()
    trials_noerp_mean = list()
    trials_noerp_dev = list()
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
        erp_epochs_cha = epochs[erp_epochs_idx, :, channel]
        noerp_epochs_cha = epochs[noerp_epochs_idx, :, channel]

        # Compute mean and dev measure
        erp_mean = np.mean(erp_epochs_cha, 0)
        erp_dev = compute_dev_epochs(erp_epochs_cha)
        noerp_mean = np.mean(noerp_epochs_cha, 0)
        noerp_dev = compute_dev_epochs(noerp_epochs_cha)

        # Save
        trials_erp_mean.append(erp_mean)
        trials_erp_dev.append(erp_dev)
        trials_noerp_mean.append(noerp_mean)
        trials_noerp_dev.append(noerp_dev)

    trials_erp_mean = np.mean(np.array(trials_erp_mean), 0)
    trials_erp_dev = np.mean(np.array(trials_erp_dev), 0)
    trials_noerp_mean = np.mean(np.array(trials_noerp_mean), 0)
    trials_noerp_dev = np.mean(np.array(trials_noerp_dev), 0)

    if plot:
        # Plot the data
        t = np.linspace(window[0], window[1], trials_erp_mean.shape[0])
        plt.plot(t, trials_erp_mean)
        plt.fill_between(t, trials_erp_mean + trials_erp_dev, trials_erp_mean - trials_erp_dev, alpha=0.3)
        plt.plot(t, trials_noerp_mean)
        plt.fill_between(t, trials_noerp_mean + trials_noerp_dev, trials_noerp_mean - trials_noerp_dev, alpha=0.3)
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
