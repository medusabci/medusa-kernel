# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 15:34:18 2019
Edited on Mon Jun 13 10:00:00 2022

@author: VICTOR
@editor: Sergio Pérez-Velasco
"""
from medusa import frequency_filtering as ff
from medusa import spatial_filtering as sf
# from medusa.storage.medusa_data import MedusaData
from medusa.components import Recording
# from medusa.bci.mi_feat_extraction import extract_mi_trials_from_midata
from medusa.local_activation import statistics
# from medusa.bci.mi_models import MIModelSettings
from medusa.plots import topographic_plots
from medusa.bci.mi_paradigms import StandardPreprocessing, \
    StandardFeatureExtraction, MIDataset

import numpy as np
from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt
import scipy.signal as scisig


def plot_erd_ers_time(files, ch_to_plot, order=5,
                      cutoff=[5, 35], btype='bandpass',
                      temp_filt_method='sosfiltfilt',
                      w_epoch_t=(-1000, 6000), target_fs=128,
                      baseline_mode='trial', w_baseline_t=(-1000, 0),
                      norm='z', mov_mean_ms=1000, show=True):
    """Plotting function of ERD/ERS from motor imagery runs of MEDUSA.
    Parameters
    ----------
    files: list
        List of paths pointing to MI files.
    ch_to_plot: list
        List with the labels of the channels to plot
    """
    # Common processing
    rec = Recording.load(files[0])
    channel_set = rec.eeg.channel_set
    dataset = MIDataset(channel_set=channel_set, fs=rec.eeg.fs,
                        biosignal_att_key='eeg', experiment_mode='train')
    for file in files:
        dataset.add_recordings(Recording.load(file))
    fs = rec.eeg.fs
    preprocessing = StandardPreprocessing(order=order, cutoff=cutoff,
                                          btype=btype,
                                          temp_filt_method=temp_filt_method)
    preprocessing.fit(fs=fs)
    dataset = preprocessing.fit_transform_dataset(dataset=dataset)
    feature_extractor = StandardFeatureExtraction(w_epoch_t=w_epoch_t,
                                                  target_fs=target_fs,
                                                  baseline_mode=baseline_mode,
                                                  w_baseline_t=w_baseline_t,
                                                  norm=norm)

    features, track_info = feature_extractor.transform_dataset(dataset=dataset)

    lcha = dataset.channel_set.l_cha
    labels = track_info["mi_labels"]

    # # Baseline parameters
    t_baseline = [w_baseline_t[0] - w_epoch_t[0], w_baseline_t[1] - w_epoch_t[0]]
    idx_baseline = np.round(np.array(t_baseline) * target_fs / 1000).astype(int)

    # Separate the classes
    trials_c1 = features[labels == 0, :, :]
    trials_c2 = features[labels == 1, :, :]

    # Compute the average power
    p_c1 = np.power(trials_c1, 2)
    p_c2 = np.power(trials_c2, 2)
    p_c1_avg = np.mean(p_c1, axis=0)
    p_c2_avg = np.mean(p_c2, axis=0)

    # Compute the reference power for each channel
    r_c1_mean = np.mean(p_c1_avg[idx_baseline[0]:idx_baseline[1], :], axis=0)
    r_c2_mean = np.mean(p_c2_avg[idx_baseline[0]:idx_baseline[1], :], axis=0)

    # Compute ERD/ERS
    ERDERS_c1 = 100 * (p_c1_avg - r_c1_mean) / r_c1_mean
    ERDERS_c2 = 100 * (p_c2_avg - r_c2_mean) / r_c2_mean
    #TODO: Cambiar por https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.uniform_filter1d.html

    ERDERS_c1_smooth = uniform_filter1d(ERDERS_c1,
                                         int(np.floor(mov_mean_ms * fs / 1000)),
                                        axis=0, mode='mirror')
    ERDERS_c2_smooth = uniform_filter1d(ERDERS_c2,
                                        int(np.floor(mov_mean_ms * fs / 1000)),
                                        axis=0, mode='mirror')

    # Signed r2 for the power
    p_c1_marg = 100 * (p_c1 - r_c1_mean / p_c1.shape[0]) / r_c1_mean
    p_c2_marg = 100 * (p_c2 - r_c2_mean / p_c2.shape[0]) / r_c2_mean
    trials_r2 = statistics.signed_r2(p_c1_marg, p_c2_marg, signed=False, axis=0)
    if mov_mean_ms != 0:
        trials_r2 = uniform_filter1d(trials_r2,
                                     int(np.floor(mov_mean_ms * fs / 1000)),
                                     axis=0, mode='mirror')

    # Plotting
    times = np.linspace(w_epoch_t[0], w_epoch_t[1], ERDERS_c1_smooth.shape[0])

    # Plot
    left = 0.1
    bottom = 0
    width = 0.8
    height_psd = 0.6
    height_r2 = 0.06
    height_cbar = 0.06
    gap = 0.12

    figs = list()
    for n in range(len(ch_to_plot)):
        fig = plt.figure()

        ax1 = fig.add_axes([left, bottom + height_r2 + height_cbar + gap, width, height_psd], xticklabels=[])
        ax2 = fig.add_axes([left, bottom + height_cbar + gap, width, height_r2], yticklabels=[])

        if ch_to_plot[n] not in ch_to_plot:
            raise ValueError('Channel ' + ch_to_plot[n] + ' is missing!')
        i = lcha.index(ch_to_plot[n])

        # ERD/ERS(%)
        ax1.minorticks_on()
        ax1.grid(visible=True, which='minor', color='#ededed', linestyle='--')
        ax1.grid(visible=True, which='major')
        ax1.axvline(x=0, color='k', linestyle='--', label='_nolegend_')
        ax1.axvspan(w_baseline_t[0], w_baseline_t[1], alpha=0.1,
                    facecolor='gray', label='_nolegend_')
        ax1.plot(times, ERDERS_c1_smooth[:, i], linewidth=2, color=[255 / 255, 174 / 255, 0 / 255])
        ax1.plot(times, ERDERS_c2_smooth[:, i], linewidth=2, color=[24 / 255, 255 / 255, 73 / 255])
        ax1.set_xlim(w_epoch_t)
        ax1.title.set_text(ch_to_plot[n])
        ax1.set_ylabel(r'ERD/ERS (%)')
        ax1.legend(['Left', 'Right'])

        # Signed-r2
        ax2.pcolormesh(times, range(2), np.tile(trials_r2[:, i], reps=[2, 1]),
                       cmap='YlOrRd')
        ax2.axvline(x=0, color='k', linestyle='--', label='_nolegend_')
        ax2.set_ylabel('$r^2$')
        ax2.set_xlabel('Time (ms)')
        figs.append(fig)

        if show is True:
            plt.show()

    return figs


def plot_erd_ers_freq(files, ch_to_plot, order=5,
                      cutoff=[5, 35], btype='bandpass',
                      temp_filt_method='sosfiltfilt',
                      w_epoch_t=(-1000, 6000), target_fs=128,
                      baseline_mode='trial', w_baseline_t=(-1000, 0),
                      norm='z', mov_mean_hz=0, welch_seg_len_pct=50,
                      welch_overlap_pct=75, show=True):
    # TODO: More options!...
    # TODO: Left and right classes labels are hardcoded!
    """ This function depicts the ERD/ERS events of MI BCIs over the frequency
    spectrum.
    Parameters
    ----------
    files: list
        List of paths pointing to MI files.
    ch_to_plot: list
        List with the labels of the channels to plot
    """

    # Common processing
    rec = Recording.load(files[0])
    channel_set = rec.eeg.channel_set
    dataset = MIDataset(channel_set=channel_set, fs=rec.eeg.fs,
                        biosignal_att_key='eeg', experiment_mode='train')
    for file in files:
        dataset.add_recordings(Recording.load(file))
    fs = rec.eeg.fs
    preprocessing = StandardPreprocessing(order=order, cutoff=cutoff,
                                          btype=btype,
                                          temp_filt_method=temp_filt_method)
    preprocessing.fit(fs=fs)
    dataset = preprocessing.fit_transform_dataset(dataset=dataset)
    feature_extractor = StandardFeatureExtraction(w_epoch_t=w_epoch_t,
                                                  target_fs=target_fs,
                                                  baseline_mode=baseline_mode,
                                                  w_baseline_t=w_baseline_t,
                                                  norm=norm)

    features, track_info = feature_extractor.transform_dataset(dataset=dataset)

    lcha = dataset.channel_set.l_cha
    labels = track_info["mi_labels"]

    # Compute the PSD
    trials_psd = None
    for t in features:
        # Compute PSD of the trial
        welch_seg_len = np.round(welch_seg_len_pct / 100 * t.shape[0]).astype(int)
        welch_overlap = np.round(welch_overlap_pct / 100 * welch_seg_len).astype(int)
        welch_ndft = welch_seg_len
        t_freqs, t_psd = scisig.welch(t, fs=target_fs, nperseg=welch_seg_len,
                                      noverlap=welch_overlap,
                                      nfft=welch_ndft, axis=0)

        # Concatenate
        t_psd = t_psd.reshape(1, t_psd.shape[0], t_psd.shape[1])
        trials_psd = np.concatenate((trials_psd, t_psd), axis=0) if trials_psd is not None else t_psd

    # Separate the classes
    trials_psd_c1 = trials_psd[labels == 0, :, :]
    trials_psd_c2 = trials_psd[labels == 1, :, :]

    # Signed r2
    trials_r2 = statistics.signed_r2(trials_psd_c1, trials_psd_c2, signed=False, axis=0)
    if mov_mean_hz != 0:
        size = int(trials_psd_c1.shape[1] / (cutoff[1]-cutoff[0]) * mov_mean_hz)
        trials_r2 = uniform_filter1d(trials_r2, size,
                                     axis=0, mode='nearest')

    # Mean PSD
    m_psd_c1 = np.mean(trials_psd_c1, axis=0)
    m_psd_c2 = np.mean(trials_psd_c2, axis=0)

    # Plotting
    freqs = np.linspace(0, fs / 2, len(m_psd_c1))

    # Range to plot
    lims = [0, target_fs / 2]
    if btype == 'bandpass':
        lims = [cutoff[0],
                cutoff[1]]
    elif btype == 'highpass':
        lims = [cutoff[0], target_fs / 2]
    elif btype == 'lowpass':
        lims = [0, cutoff[1]]

    # Plot
    left = 0.1
    bottom = 0
    width = 0.8
    height_psd = 0.6
    height_r2 = 0.06
    height_cbar = 0.06
    gap = 0.12

    figs = list()
    for n in range(len(ch_to_plot)):
        fig = plt.figure()

        ax1 = fig.add_axes([left, bottom + height_r2 + height_cbar + gap, width, height_psd], xticklabels=[])
        ax2 = fig.add_axes([left, bottom + height_cbar + gap, width, height_r2], yticklabels=[])

        if ch_to_plot[n] not in ch_to_plot:
            raise ValueError('Channel ' + ch_to_plot[n] + ' is missing!')
        i = lcha.index(ch_to_plot[n])

        # Individual curves
        # for j in range(trials_psd_c1.shape[0]):
        #    plt.plot(freqs, trials_psd_c1[j,:,i], linewidth=0.5,
        #             color=[255/255, 174/255, 0/255], alpha=0.5)
        # for j in range(trials_psd_c2.shape[0]):
        #    plt.plot(freqs, trials_psd_c2[j,:,i], linewidth=0.5,
        #             color=[24/255, 255/255, 73/255], alpha=0.5)

        # Averaged curves
        ax1.minorticks_on()
        ax1.grid(b=True, which='minor', color='#ededed', linestyle='--')
        ax1.grid(b=True, which='major')
        ax1.plot(freqs, m_psd_c1[:, i], linewidth=2, color=[255 / 255, 174 / 255, 0 / 255])
        ax1.plot(freqs, m_psd_c2[:, i], linewidth=2, color=[24 / 255, 255 / 255, 73 / 255])
        ax1.set_xlim(lims)
        ax1.title.set_text(ch_to_plot[n])
        ax1.set_ylabel(r'PSD ($uV^2/Hz$)')
        ax1.legend(['Left', 'Right'])

        # Signed-r2
        ax2.pcolormesh(freqs, range(2), np.tile(trials_r2[:, i], reps=[2, 1]), cmap='YlOrRd', vmin=0)
        ax2.set_xlim(lims)
        ax2.set_ylabel('$r^2$')
        ax2.set_xlabel('Frequency (Hz)')
        figs.append(fig)
        if show is True:
            plt.show()
    return figs


def plot_r2_topoplot(files,  order=5, cutoff=[5, 35], btype='bandpass',
                     temp_filt_method='sosfiltfilt', w_epoch_t=(-1000, 6000),
                     target_fs=128, baseline_mode='trial',
                     w_baseline_t=(-1000, 0), norm='z', welch_seg_len_pct=50,
                     welch_overlap_pct=75, show=True):
    # Common processing
    rec = Recording.load(files[0])
    channel_set = rec.eeg.channel_set
    dataset = MIDataset(channel_set=channel_set, fs=rec.eeg.fs,
                        biosignal_att_key='eeg', experiment_mode='train')
    for file in files:
        dataset.add_recordings(Recording.load(file))
    fs = rec.eeg.fs
    preprocessing = StandardPreprocessing(order=order, cutoff=cutoff,
                                          btype=btype,
                                          temp_filt_method=temp_filt_method)
    preprocessing.fit(fs=fs)
    dataset = preprocessing.fit_transform_dataset(dataset=dataset)
    feature_extractor = StandardFeatureExtraction(w_epoch_t=w_epoch_t,
                                                  target_fs=target_fs,
                                                  baseline_mode=baseline_mode,
                                                  w_baseline_t=w_baseline_t,
                                                  norm=norm)

    features, track_info = feature_extractor.transform_dataset(dataset=dataset)

    lcha = dataset.channel_set.l_cha
    labels = track_info["mi_labels"]

    # Compute the PSD
    trials_psd = None
    for t in features:
        # Compute PSD of the trial
        welch_seg_len = np.round(welch_seg_len_pct / 100 * t.shape[0]).astype(int)
        welch_overlap = np.round(welch_overlap_pct / 100 * welch_seg_len).astype(int)
        welch_ndft = welch_seg_len
        t_freqs, t_psd = scisig.welch(t, fs=fs, nperseg=welch_seg_len,
                                      noverlap=welch_overlap,
                                      nfft=welch_ndft, axis=0)

        # Concatenate
        t_psd = t_psd.reshape(1, t_psd.shape[0], t_psd.shape[1])
        trials_psd = np.concatenate((trials_psd, t_psd), axis=0) if trials_psd is not None else t_psd

    # Separate the classes
    trials_psd_c1 = trials_psd[labels == 0, :, :]
    trials_psd_c2 = trials_psd[labels == 1, :, :]

    # Signed r2
    trials_r2 = statistics.signed_r2(trials_psd_c1, trials_psd_c2, signed=True, axis=0)
    trials_r2 = np.mean(trials_r2, axis=0)

    # Topoplot
    values = trials_r2.reshape(1, len(lcha))
    fig, _ = topographic_plots.plot_topography(dataset.channel_set,
                                               values, cmap='RdBu',
                                               show=show)

    return fig
