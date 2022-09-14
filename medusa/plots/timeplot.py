"""
Created on Thu Aug 25 17:09:18 2022

@author: Diego Marcos-Martínez
"""
import numpy as np
import matplotlib.pyplot as plt

def time_plot(epoch, fs=1.0, ch_labels=None):
    """
    Parameters
    ---------
    epoch: numpy array
        Signal with shape of [n_epochs,n_samples, n_channels] or
        [n_samples, n_channels]
    fs: float
        Sampling rate. Value 1 as default
    ch_labels: list of strings or None
        List containing the channel labels
    """

    try:
        # Check if signals is divided in epochs
        if len(epoch.shape) == 2:
            epoch = epoch[np.newaxis,:,:]
        blocks, samples_per_block, channels = epoch.shape
        if ch_labels is None:
            ch_labels = [f"Channel {idx}" for idx in range(channels)]


        epoch_c = __reshape_signal(epoch)

        channel_offset = np.zeros(epoch_c.shape[1])
        if len(channel_offset) > 1:
            channel_offset[1:] = np.max(np.max(epoch_c[:, 1:], axis=0) -
                                        np.min(epoch_c[:, :-1], axis=0))
            channel_offset = np.cumsum(channel_offset)
        epoch_c = epoch_c - channel_offset
        max_val, min_val = epoch_c.max(), epoch_c.min()
        display_times = np.linspace(0, int(epoch_c.shape[0] / fs),
                                    epoch_c.shape[0])

        # Plot
        plt.plot(display_times, epoch_c, 'k', linewidth=0.5)
        plt.yticks(-channel_offset, labels=ch_labels)
        vertical_lines = np.empty((blocks - 1, 2, 50))
        for block in range(blocks - 1):
            vertical_lines[block, :, :] = np.asarray(
                [np.ones(50) * (block + 1) * int(samples_per_block / fs),
                 np.linspace(min_val, max_val, 50)])
            plt.plot(vertical_lines[block, 0, :], vertical_lines[block, 1, :],
                     '--', color='red', linewidth=1.5)
        plt.show(block=True)

    except Exception as e:
        print(e)

def __reshape_signal(epochs):
    """This is an auxiliary function than reshapes the signal it is divided in
    epochs in order to plot it in a row"""
    try:
        epoch_c = epochs.copy()
        blocks, samples_per_block, channels = epoch_c.shape
        epoch_c = np.reshape(epoch_c, (int(blocks * samples_per_block), channels))
        return epoch_c
    except Exception as e:
        print(e)
