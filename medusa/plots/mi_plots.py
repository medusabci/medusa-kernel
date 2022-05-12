# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 15:34:18 2019

@author: VICTOR
"""
from medusa import frequency_filtering as ff
from medusa import spatial_filtering as sf
from medusa.storage.medusa_data import MedusaData
from medusa.bci.mi_feat_extraction import extract_mi_trials_from_midata
from medusa import eeg_standards, signed_r2, movemean
from medusa.bci.mi_models import MIModelSettings
from medusa.plots import topography_plots

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import scipy.signal as scisig
import pickle


def plot_erd_ers_time(files, mi_model_settings, ch_to_plot,
                      t_window=[-1000, 6000], t_baseline=[-1000, 0],
                      t_cue=0, mov_mean_ms=1000, fig_id=None, showing=True):
    # Common processing
    raw_trials, train_info, fs, lcha = common_mi_preprocessing(files, mi_model_settings)
    labels = train_info["mi_labels"]

    # Baseline parameters
    l_window = np.floor(np.dot(t_window, fs / 1000))
    l_baseline = np.floor(np.dot(t_baseline, fs / 1000))
    idx_baseline = l_baseline - l_window[0]
    idx_cue = np.floor(t_cue * fs / 1000) - l_window[0]

    # Extract trials
    trials = None
    refs = None
    for trial in raw_trials:
        # Get the entire trial (including baseline)
        t = trial[:int(l_window[1] - l_window[0]), :]

        # Z-score
        b = trial[np.arange(int(idx_baseline[0]), int(idx_baseline[1])), :]
        b_mean = np.mean(b, axis=0)
        b_std = np.std(b, axis=0)
        t = (t - b_mean) / b_std

        # Append to the trial array
        t = t.reshape(1, t.shape[0], t.shape[1])
        trials = np.concatenate((trials, t), axis=0) if trials is not None else t

    # Separate the classes
    trials_c1 = trials[labels == 0, :, :]
    trials_c2 = trials[labels == 1, :, :]

    # Compute the average power
    p_c1 = np.power(trials_c1, 2)
    p_c2 = np.power(trials_c2, 2)
    p_c1_avg = np.mean(p_c1, axis=0)
    p_c2_avg = np.mean(p_c2, axis=0)

    # Compute the reference power for each channel
    r_c1_mean = np.mean(p_c1_avg[np.arange(int(idx_baseline[0]), int(idx_baseline[1])), :], axis=0)
    r_c2_mean = np.mean(p_c2_avg[np.arange(int(idx_baseline[0]), int(idx_baseline[1])), :], axis=0)

    # Compute ERD/ERS
    ERDERS_c1 = 100 * (p_c1_avg - r_c1_mean) / r_c1_mean
    ERDERS_c2 = 100 * (p_c2_avg - r_c2_mean) / r_c2_mean
    ERDERS_c1_smooth = movemean.movemean(ERDERS_c1, np.floor(mov_mean_ms * fs / 1000))
    ERDERS_c2_smooth = movemean.movemean(ERDERS_c2, np.floor(mov_mean_ms * fs / 1000))

    # Signed r2 for the power
    p_c1_marg = 100 * (p_c1 - r_c1_mean / p_c1.shape[0]) / r_c1_mean
    p_c2_marg = 100 * (p_c2 - r_c2_mean / p_c2.shape[0]) / r_c2_mean
    trials_r2 = signed_r2.r2(p_c1_marg, p_c2_marg, signed=False, dim=0)
    if mov_mean_ms != 0:
        trials_r2 = movemean.movemean(trials_r2, np.floor(mov_mean_ms * fs / 1000))

    # Plotting
    times = np.linspace(t_window[0], t_window[1], ERDERS_c1_smooth.shape[0])

    # Plot
    left = 0.1
    bottom = 0
    width = 0.8
    height_psd = 0.6
    height_r2 = 0.06
    height_cbar = 0.06
    gap = 0.12

    for n in range(len(ch_to_plot)):

        if fig_id is None:
            fig = plt.figure()
        else:
            fig = plt.figure(fig_id[n])

        ax1 = fig.add_axes([left, bottom + height_r2 + height_cbar + gap, width, height_psd], xticklabels=[])
        ax2 = fig.add_axes([left, bottom + height_cbar + gap, width, height_r2], yticklabels=[])
        # axc = fig.add_axes([left, bottom, width, height_cbar])

        if ch_to_plot[n] not in ch_to_plot:
            raise ValueError('Channel ' + ch_to_plot[n] + ' is missing!')
        i = lcha.index(ch_to_plot[n])

        # ERD/ERS(%)
        ax1.minorticks_on()
        ax1.grid(b=True, which='minor', color='#ededed', linestyle='--')
        ax1.grid(b=True, which='major')
        ax1.axvline(x=t_cue, color='k', linestyle='--', label='_nolegend_')
        ax1.axvspan(t_baseline[0], t_baseline[1], alpha=0.1, facecolor='gray')
        ax1.plot(times, ERDERS_c1_smooth[:, i], linewidth=2, color=[255 / 255, 174 / 255, 0 / 255])
        ax1.plot(times, ERDERS_c2_smooth[:, i], linewidth=2, color=[24 / 255, 255 / 255, 73 / 255])
        ax1.set_xlim(t_window)
        ax1.recording_id.set_text(ch_to_plot[n])
        ax1.set_ylabel(r'ERD/ERS (%)')
        ax1.legend(['Left', 'Right'])

        # Signed-r2
        im2 = ax2.pcolormesh(times, range(2), [trials_r2[:, i]], cmap='YlOrRd')
        # ax2.plot(times, trials_r2[:,i])
        # ax2.set_xlim(lims)
        ax2.axvline(x=t_cue, color='k', linestyle='--', label='_nolegend_')
        ax2.set_ylabel('$r^2$')
        ax2.set_xlabel('Time (ms)')
        # plt.colorbar(im2, cax=axc, orientation='horizontal')
        # axc.set_ylabel('max [r2]',rotation=0)
        # axc2 = axc.twinx()
        # axc2.set_ylabel('max [r2]',rotation=0)

        if showing is True:
            plt.show()


def plot_erd_ers_freq(files, mi_model_settings, ch_to_plot,
                      mov_mean_hz=0, welch_seg_len_pct=50,
                      welch_overlap_pct=75, fig_id=None, showing=True):
    # TODO: More options!...
    # TODO: Left and right classes labels are hardcoded!
    """ This function depicts the ERD/ERS events of MI BCIs over the frequency
    spectrum.

        :param files:    List of paths pointing to MedusaData files.
        :param mi_model_settings:   MIModelSettings class.
        :param ch_to_plot:          List with the labels of the chn. to plot
    """

    # Common processing
    raw_trials, train_info, fs, lcha = common_mi_preprocessing(files, mi_model_settings)
    labels = train_info["mi_labels"]

    # Compute the PSD
    trials_psd = None
    for t in raw_trials:
        # Compute PSD of the trial
        welch_seg_len = np.round(welch_seg_len_pct / 100 * t.shape[0]).astype(int)
        welch_overlap = np.round(welch_overlap_pct / 100 * welch_seg_len).astype(int)
        welch_ndft = welch_seg_len
        t_freqs, t_psd = scisig.welch(t.T, fs=fs, nperseg=welch_seg_len, noverlap=welch_overlap, nfft=welch_ndft)
        t_psd = t_psd.T

        # Concatenate
        t_psd = t_psd.reshape(1, t_psd.shape[0], t_psd.shape[1])
        trials_psd = np.concatenate((trials_psd, t_psd), axis=0) if trials_psd is not None else t_psd

    # Separate the classes
    trials_psd_c1 = trials_psd[labels == 0, :, :]
    trials_psd_c2 = trials_psd[labels == 1, :, :]

    # Signed r2
    trials_r2 = signed_r2.r2(trials_psd_c1, trials_psd_c2, signed=False, dim=0)
    if mov_mean_hz != 0:
        trials_r2 = movemean.movemean(trials_r2, np.floor(fs / mov_mean_hz))

    # Mean PSD
    m_psd_c1 = np.mean(trials_psd_c1, axis=0)
    m_psd_c2 = np.mean(trials_psd_c2, axis=0)

    # Plotting
    freqs = np.linspace(0, fs / 2, len(m_psd_c1))

    # Range to plot
    lims = [0, fs / 2]
    ftype = mi_model_settings.p_filt_method
    if mi_model_settings.p_filt_params[ftype]["type"] == 'bandpass':
        lims = [mi_model_settings.p_filt_params[ftype]["fpass1"],
                mi_model_settings.p_filt_params[ftype]["fpass2"]]
    elif mi_model_settings.p_filt_params[ftype]["type"] == 'highpass':
        lims = [mi_model_settings.p_filt_params[ftype]["fpass1"], fs / 2]
    elif mi_model_settings.p_filt_params[ftype]["type"] == 'lowpass':
        lims = [0, mi_model_settings.p_filt_params[ftype]["fpass1"]]

    # Plot
    left = 0.1
    bottom = 0
    width = 0.8
    height_psd = 0.6
    height_r2 = 0.06
    height_cbar = 0.06
    gap = 0.12

    for n in range(len(ch_to_plot)):
        if fig_id is None:
            fig = plt.figure()
        else:
            fig = plt.figure(fig_id[n])

        ax1 = fig.add_axes([left, bottom + height_r2 + height_cbar + gap, width, height_psd], xticklabels=[])
        ax2 = fig.add_axes([left, bottom + height_cbar + gap, width, height_r2], yticklabels=[])
        # axc = fig.add_axes([left, bottom, width, height_cbar])

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
        ax1.recording_id.set_text(ch_to_plot[n])
        ax1.set_ylabel(r'PSD ($uV^2/Hz$)')
        ax1.legend(['Left', 'Right'])

        # Signed-r2
        im2 = ax2.pcolormesh(freqs, range(2), [trials_r2[:, i]], cmap='YlOrRd', vmin=0)
        # ax2.plot(freqs, trials_r2[:,i])
        ax2.set_xlim(lims)
        ax2.set_ylabel('$r^2$')
        ax2.set_xlabel('Frequency (Hz)')
        # plt.colorbar(im2, cax=axc, orientation='horizontal')
        # axc.set_ylabel('max [r2]',rotation=0)
        # axc2 = axc.twinx()
        # axc2.set_ylabel('max [r2]',rotation=0)
        if showing is True:
            plt.show()


def plot_r2_topoplot(files, mi_model_settings,
                     sel_window_hz=[8, 15], sel_window_ms=[0, 4000],
                     welch_seg_len_pct=50, welch_overlap_pct=75,
                     t_baseline=[-1000, 0], t_window=[-1000, 6000], fig=None, showing=True):
    # Pre-processing
    mi_model_settings.p_filt_method = 'FIR'
    mi_model_settings.p_filt_params['FIR']['fpass1'] = sel_window_hz[0]
    mi_model_settings.p_filt_params['FIR']['fpass2'] = sel_window_hz[1]
    raw_trials, train_info, fs, lcha = common_mi_preprocessing(files, mi_model_settings)
    labels = train_info["mi_labels"]

    # Extract trials
    l_window = np.floor(np.dot(t_window, fs / 1000))
    l_sel_window = np.floor(np.dot(sel_window_ms, fs / 1000))
    l_baseline = np.floor(np.dot(t_baseline, fs / 1000))
    idx_baseline = l_baseline - l_window[0]
    idx_sel = l_sel_window - l_window[0]
    trials = None
    for trial in raw_trials:
        # Get the entire trial (including baseline)
        t = trial[:int(l_window[1] - l_window[0]), :]

        # Z-score
        b = trial[np.arange(int(idx_baseline[0]), int(idx_baseline[1])), :]
        b_mean = np.mean(b, axis=0)
        b_std = np.std(b, axis=0)
        t = (t - b_mean) / b_std

        # Get the selected window
        t_sel = t[int(idx_sel[0]):int(idx_sel[1]), :]

        # Append to the trial array
        t_sel = t_sel.reshape(1, t_sel.shape[0], t_sel.shape[1])
        trials = np.concatenate((trials, t_sel), axis=0) if trials is not None else t_sel

    # Compute the PSD
    trials_psd = None
    for t in trials:
        # Compute PSD of the trial
        welch_seg_len = np.round(welch_seg_len_pct / 100 * t.shape[0]).astype(int)
        welch_overlap = np.round(welch_overlap_pct / 100 * welch_seg_len).astype(int)
        welch_ndft = welch_seg_len
        t_freqs, t_psd = scisig.welch(t.T, fs=fs, nperseg=welch_seg_len, noverlap=welch_overlap, nfft=welch_ndft)
        t_psd = t_psd.T

        # Concatenate
        t_psd = t_psd.reshape(1, t_psd.shape[0], t_psd.shape[1])
        trials_psd = np.concatenate((trials_psd, t_psd), axis=0) if trials_psd is not None else t_psd

    # Separate the classes
    trials_psd_c1 = trials_psd[labels == 0, :, :]
    trials_psd_c2 = trials_psd[labels == 1, :, :]

    # Signed r2
    trials_r2 = signed_r2.r2(trials_psd_c1, trials_psd_c2, signed=True, dim=0)
    trials_r2 = np.mean(trials_r2, axis=0)

    # Topoplot
    values = trials_r2.reshape(1, len(lcha))
    if fig is None:
        fig = plt.figure()
    topography_plots.plot_topography(lcha, values, cmap='RdBu', fig=fig, show=showing)


def common_mi_preprocessing(files, mi_model_settings):
    # Get params
    ftype = mi_model_settings.p_filt_method
    fband = [mi_model_settings.p_filt_params[ftype]["fpass1"], mi_model_settings.p_filt_params[ftype]["fpass2"]]
    forder = mi_model_settings.p_filt_params[ftype]["order"]
    fmethod = mi_model_settings.p_filt_params[ftype]["method"]
    fbtype = mi_model_settings.p_filt_params[ftype]["type"]
    smethod = mi_model_settings.p_spatial_method
    emethod = mi_model_settings.f_method

    # Load data
    data = list()
    last_fs = None
    last_lcha = None
    for file in files:
        temp = MedusaData.load_from_file(file)
        temp.eeg.lcha = [cha.upper() for cha in temp.eeg.lcha]
        # Error check
        if last_fs is not None:
            if last_fs != temp.eeg.fs:
                raise ValueError("The sampling rate is not the same in every file!")
            if last_lcha != temp.eeg.lcha:
                raise ValueError("The channels are not the same in every file!")
        last_fs = temp.eeg.fs
        last_lcha = temp.eeg.lcha

        # Frequency filtering (TODO: Method filtfilt is correct also for IIR?)
        [b, a] = ff.filter_designer(fband, temp.eeg.fs, forder, ftype, fbtype)
        temp.eeg.signal = ff.apply_filter_offline(temp.eeg.signal, b, a, axis=0, method=fmethod)

        # Spatial filtering
        if smethod == 'CAR':
            temp.eeg.signal = sf.apply_car(temp.eeg.signal)
        elif smethod == 'laplace':
            temp.eeg.signal = sf.apply_laplacian(s=temp.eeg.signal,
                                                 lcha=temp.eeg.lcha,
                                                 mode=mi_model_settings.p_spatial_params[smethod]["mode"],
                                                 locations=mi_model_settings.p_spatial_params[smethod]["locations"],
                                                 n=mi_model_settings.p_spatial_params[smethod]["n"],
                                                 lcha_to_filter=mi_model_settings.p_spatial_params[smethod][
                                                     "lcha_to_filter"],
                                                 lcha_laplace=mi_model_settings.p_spatial_params[smethod][
                                                     "lcha_laplace"])
        else:
            raise Exception("Unknown spatial filter.")

        # Store it
        data.append(temp)

    # Extract the trials
    # TODO: CSP
    w_trial_t = mi_model_settings.f_params[emethod]["w_trial_t"]
    use_calibration_baseline = mi_model_settings.f_params[emethod]["use_calibration_baseline"]
    normalization = mi_model_settings.f_params[emethod]["normalization"]
    raw_trials, train_info = extract_mi_trials_from_midata(mi_data=data,
                                                           w_trial_t=w_trial_t,
                                                           use_calibration_baseline=use_calibration_baseline,
                                                           norm=normalization)
    return raw_trials, train_info, last_fs, last_lcha
