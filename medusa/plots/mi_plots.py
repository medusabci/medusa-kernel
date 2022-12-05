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


class MIPlots():
    def __init__(self, fir_order=1000, fir_cutoff=(5, 35),
                 fir_btype="bandpass", fir_method="filtfilt",
                 w_baseline_t=(-1000, 0), w_epoch_t=(-1000, 6000), norm="z",
                 baseline_mode="trial", target_fs=None):
        self.fir_order = fir_order
        self.fir_cutoff = fir_cutoff
        self.fir_btype = fir_btype
        self.fir_method = fir_method
        self.w_baseline_t = w_baseline_t
        self.w_epoch_t = w_epoch_t
        self.norm = norm
        self.target_fs = target_fs
        self.baseline_mode = baseline_mode

        # generated
        self.dataset = None
        self.fs = None
        self.channel_set = None
        self.fir = None
        self.features = None
        self.track_info = None

        self.set_sizes()

    def set_sizes(self, label_size=6, axes_size=5, line_width=1):
        self.label_size = label_size
        self.axes_size = axes_size
        self.line_width = line_width

    def extract_features(self, files):
        # Load files
        rec = Recording.load(files[0])
        self.fs = rec.eeg.fs
        self.channel_set = rec.eeg.channel_set
        self.dataset = MIDataset(channel_set=self.channel_set, fs=self.fs,
                                 experiment_att_key='midata',
                                 biosignal_att_key='eeg',
                                 experiment_mode='train')
        for file in files:
            self.dataset.add_recordings(Recording.load(file))

        # Pre-processing
        self.fir = ff.FIRFilter(order=self.fir_order, cutoff=self.fir_cutoff,
                                btype=self.fir_btype,
                                filt_method=self.fir_method)
        self.fir.fit(fs=self.fs)
        for rec in self.dataset.recordings:
            eeg = getattr(rec, self.dataset.biosignal_att_key)
            eeg.signal = self.fir.transform(signal=eeg.signal)
            eeg.signal = sf.car(signal=eeg.signal)
            setattr(rec, self.dataset.biosignal_att_key, eeg)

        # Feature extraction
        feature_extractor = StandardFeatureExtraction(
            w_epoch_t=self.w_epoch_t, target_fs=self.target_fs,
            baseline_mode=self.baseline_mode, w_baseline_t=self.w_baseline_t,
            norm=self.norm)
        self.features, self.track_info = feature_extractor.transform_dataset(
            dataset=self.dataset)

    def plot_spectrogram(self, ch_to_plot, axs_to_plot=None,
                         welch_seg_len_pct=50,
                         welch_overlap_pct=75,
                         mov_mean_hz=0):
        if self.dataset is None:
            raise Exception("Call MiPlots._extract_features() before plotting!")
        if axs_to_plot is None:
            axs_to_plot = list()
            for c in ch_to_plot:
                fig = plt.figure(figsize=(5, 5), dpi=300)
                gs = fig.add_gridspec(2, 1, wspace=0.2, hspace=0.2,
                                      height_ratios=[1, 0.2])
                axs_to_plot.append({'freq': fig.add_subplot(gs[0, 0]),
                                    'r2': fig.add_subplot(gs[1, 0])})

        labels = self.track_info["mi_labels"]
        labels_info = self.track_info["mi_labels_info"][0]

        # Compute the spectrogram
        trials_specgram = None
        new_fs = self.fs if self.target_fs is None else self.target_fs
        for t in self.features:
            welch_seg_len = np.round(
                welch_seg_len_pct / 100 * t.shape[0]).astype(int)
            welch_overlap = np.round(
                welch_overlap_pct / 100 * welch_seg_len).astype(int)
            t_freqs, t_times, t_sxx = scisig.spectrogram(
                t, fs=new_fs, axis=0, nperseg=int(256 / 2)
            )
            t_sxx = np.expand_dims(t_sxx, axis=0)
            trials_specgram = np.concatenate((trials_specgram, t_sxx), axis=0) \
                if trials_specgram is not None else t_sxx

        # Separate the classes
        trials_specgram_c1 = trials_specgram[labels == 0, :, :, :]
        trials_specgram_c2 = trials_specgram[labels == 1, :, :, :]

        # Plot ranges
        lims = [0, new_fs / 2]
        if self.fir_btype == 'bandpass':
            lims = [self.fir_cutoff[0],
                    self.fir_cutoff[1]]
        elif self.fir_btype == 'highpass':
            lims[0] = self.fir_cutoff[0]
        elif self.fir_btype == 'lowpass':
            lims[1] = self.fir_cutoff[1]

        # Plot
        lcha = self.dataset.channel_set.l_cha
        for n in range(len(ch_to_plot)):
            if ch_to_plot[n] not in lcha:
                raise ValueError('Channel ' + ch_to_plot[n] + ' is missing!')
            i = lcha.index(ch_to_plot[n])

            # Signed r2
            temp_c1 = np.squeeze(trials_specgram_c1[:, :, i, :])  # obs, t, f
            temp_c2 = np.squeeze(trials_specgram_c2[:, :, i, :])  # obs, t, f
            temp_r2 = statistics.signed_r2(temp_c1, temp_c2, signed=True,
                                           axis=0)
            with plt.style.context('seaborn'):
                # Averaged curves
                ax1 = axs_to_plot[n]
                ax1.minorticks_on()
                ax1.pcolormesh(t_times, t_freqs, temp_r2, cmap='RdBu_r')
                # ax1.grid(b=True, which='minor', color='#ededed', linestyle='--')
                # ax1.grid(b=True, which='major')
                # ax1.plot(freqs, m_psd_c1[:, i], linewidth=self.line_width,
                #          color=[255 / 255, 174 / 255, 0 / 255])
                # ax1.plot(freqs, m_psd_c2[:, i], linewidth=self.line_width,
                #          color=[24 / 255, 255 / 255, 73 / 255])
                ax1.set_ylim(lims)
                ax1.set_xlim((self.w_baseline_t[1] / 1000,
                              self.w_epoch_t[1] / 1000))
                ax1.set_title(ch_to_plot[n], fontsize=self.label_size)
                ax1.set_ylabel(r'Frequency (Hz)', fontsize=self.label_size)
                ax1.set_xlabel(r'Time (s)', fontsize=self.label_size)

                ax1.tick_params(axis='x', labelsize=self.axes_size)
                ax1.tick_params(axis='y', labelsize=self.axes_size)
        return axs_to_plot

    def plot_erd_ers_freq(self, ch_to_plot, axs_to_plot=None,
                          welch_seg_len_pct=50,
                          welch_overlap_pct=75,
                          mov_mean_hz=0):
        if self.dataset is None:
            raise Exception("Call MiPlots._extract_features() before plotting!")
        if axs_to_plot is None:
            axs_to_plot = list()
            for c in ch_to_plot:
                fig = plt.figure(figsize=(5, 5), dpi=300)
                gs = fig.add_gridspec(2, 1, wspace=0.2, hspace=0.2,
                                      height_ratios=[1, 0.2])
                axs_to_plot.append({'freq': fig.add_subplot(gs[0, 0]),
                                    'r2': fig.add_subplot(gs[1, 0])})

        labels = self.track_info["mi_labels"]
        labels_info = self.track_info["mi_labels_info"][0]

        # Compute the PSD
        trials_psd = None
        new_fs = self.fs if self.target_fs is None else self.target_fs
        for t in self.features:
            # Compute PSD of the trial
            welch_seg_len = np.round(
                welch_seg_len_pct / 100 * t.shape[0]).astype(int)
            welch_overlap = np.round(
                welch_overlap_pct / 100 * welch_seg_len).astype(int)
            t_freqs, t_psd = scisig.welch(t, fs=new_fs, nperseg=welch_seg_len,
                                          noverlap=welch_overlap,
                                          nfft=welch_seg_len, axis=0)

            # Concatenate
            t_psd = t_psd.reshape(1, t_psd.shape[0], t_psd.shape[1])
            trials_psd = np.concatenate((trials_psd, t_psd), axis=0) if \
                trials_psd is not None else t_psd

        # Separate the classes
        trials_psd_c1 = trials_psd[labels == 0, :, :]
        trials_psd_c2 = trials_psd[labels == 1, :, :]

        # Signed r2
        trials_r2 = statistics.signed_r2(trials_psd_c1, trials_psd_c2,
                                         signed=False, axis=0)
        if mov_mean_hz != 0:
            size = int(trials_psd_c1.shape[1] /
                       (self.fir_cutoff[1] - self.fir_cutoff[0]) * mov_mean_hz)
            trials_r2 = uniform_filter1d(trials_r2, size, axis=0,
                                         mode='nearest')

        # Mean PSD
        m_psd_c1 = np.mean(trials_psd_c1, axis=0)
        m_psd_c2 = np.mean(trials_psd_c2, axis=0)

        # Plot ranges
        freqs = np.linspace(0, new_fs / 2, len(m_psd_c1))
        lims = [0, new_fs / 2]
        if self.fir_btype == 'bandpass':
            lims = [self.fir_cutoff[0],
                    self.fir_cutoff[1]]
        elif self.fir_btype == 'highpass':
            lims[0] = self.fir_cutoff[0]
        elif self.fir_btype == 'lowpass':
            lims[1] = self.fir_cutoff[1]

        # Plot
        lcha = self.dataset.channel_set.l_cha
        for n in range(len(ch_to_plot)):
            if ch_to_plot[n] not in lcha:
                raise ValueError('Channel ' + ch_to_plot[n] + ' is missing!')
            i = lcha.index(ch_to_plot[n])

            with plt.style.context('seaborn'):
                # Averaged curves
                ax1 = axs_to_plot[n]['freq']
                ax1.minorticks_on()
                ax1.grid(b=True, which='minor', color='#ededed', linestyle='--')
                ax1.grid(b=True, which='major')
                ax1.plot(freqs, m_psd_c1[:, i], linewidth=self.line_width,
                         color=[255 / 255, 174 / 255, 0 / 255])
                ax1.plot(freqs, m_psd_c2[:, i], linewidth=self.line_width,
                         color=[24 / 255, 255 / 255, 73 / 255])
                ax1.set_xlim(lims)
                ax1.set_title(ch_to_plot[n], fontsize=self.label_size)
                ax1.set_ylabel(r'PSD ($uV^2/Hz$)', fontsize=self.label_size)
                ax1.legend([labels_info[str(0)], labels_info[str(1)]],
                           fontsize=self.label_size)
                ax1.tick_params(axis='x', labelsize=self.axes_size)
                ax1.tick_params(axis='y', labelsize=self.axes_size)

                # Signed-r2
                ax2 = axs_to_plot[n]['r2']
                ax2.pcolormesh(freqs, range(2),
                               np.tile(trials_r2[:, i], reps=[2, 1]),
                               cmap='YlOrRd',
                               vmin=0)
                ax2.set_xlim(lims)
                ax2.set_ylabel('$r^2$', fontsize=self.label_size)
                ax2.set_xlabel('Frequency (Hz)', fontsize=self.label_size)
                ax2.tick_params(axis='x', labelsize=self.axes_size)
                ax2.tick_params(axis='y', labelsize=self.axes_size)
        return axs_to_plot

    def plot_erd_ers_time(self):
        pass

    def plot_erd_ers_r2_topo(self, ch_to_plot, ax_to_plot=None,
                             welch_seg_len_pct=50,
                             welch_overlap_pct=75):
        if self.dataset is None:
            raise Exception("Call MiPlots._extract_features() before plotting!")
        if len(ch_to_plot) != 2:
            raise Exception("We need exactly two channels to compute r2 topo!")
        if ax_to_plot is None:
            ax_to_plot = list()
            for c in ch_to_plot:
                fig = plt.figure(figsize=(5, 5), dpi=300)
                ax_to_plot = fig.add_subplot(111)
        lcha = self.dataset.channel_set.l_cha
        labels = self.track_info["mi_labels"]
        labels_info = self.track_info["mi_labels_info"][0]

        # Compute the PSD
        trials_psd = None
        new_fs = self.fs if self.target_fs is None else self.target_fs
        for t in self.features:
            # Compute PSD of the trial
            welch_seg_len = np.round(
                welch_seg_len_pct / 100 * t.shape[0]).astype(int)
            welch_overlap = np.round(
                welch_overlap_pct / 100 * welch_seg_len).astype(int)
            t_freqs, t_psd = scisig.welch(t, fs=new_fs, nperseg=welch_seg_len,
                                          noverlap=welch_overlap,
                                          nfft=welch_seg_len, axis=0)

            # Concatenate
            t_psd = t_psd.reshape(1, t_psd.shape[0], t_psd.shape[1])
            trials_psd = np.concatenate((trials_psd, t_psd), axis=0) if \
                trials_psd is not None else t_psd

        # Separate the classes
        trials_psd_c1 = trials_psd[labels == 0, :, :]
        trials_psd_c2 = trials_psd[labels == 1, :, :]

        # Signed r2
        trials_r2 = statistics.signed_r2(trials_psd_c1, trials_psd_c2,
                                         signed=True, axis=0)
        trials_r2 = np.mean(trials_r2, axis=0)
        max_r2 = np.abs(np.max(trials_r2.flatten()))

        # Topoplot
        values = trials_r2.reshape(1, len(lcha))
        _, ax_to_plot, p_interp = topographic_plots.plot_topography(
            self.dataset.channel_set, values, clim=(-max_r2, max_r2),
            cmap='RdBu_r', linewidth=self.line_width * 2,
            head_radius=1.0, axes=ax_to_plot, show_colorbar=False,
            show=False, plot_skin_in_color=True)
        ax_to_plot.set_title("Signed $r^2$ (%s)" % ' vs. '.join(ch_to_plot),
                             fontsize=self.label_size)
        return ax_to_plot, p_interp


def _extract_erd_ers_features(files, ch_to_plot, order=1000, cutoff=[5, 35],
                              btype='bandpass', temp_filt_method='filtfilt',
                              w_epoch_t=(-1000, 6000), target_fs=None,
                              baseline_mode='trial', w_baseline_t=(-1000, 0),
                              norm='z'):
    saved_args = locals()
    del saved_args['files']
    del saved_args['ch_to_plot']

    # Load files
    rec = Recording.load(files[0])
    fs = rec.eeg.fs
    channel_set = rec.eeg.channel_set
    dataset = MIDataset(channel_set=channel_set, fs=rec.eeg.fs,
                        experiment_att_key='midataold',
                        biosignal_att_key='eeg', experiment_mode='train')
    for file in files:
        dataset.add_recordings(Recording.load(file))

    # Pre-processing
    fir = ff.FIRFilter(order=order, cutoff=cutoff, btype=btype,
                       filt_method=temp_filt_method)
    fir.fit(fs=fs)
    for rec in dataset.recordings:
        eeg = getattr(rec, dataset.biosignal_att_key)
        eeg.signal = fir.transform(signal=eeg.signal)
        eeg.signal = sf.car(signal=eeg.signal)
        setattr(rec, dataset.biosignal_att_key, eeg)

    # Feature extraction
    feature_extractor = StandardFeatureExtraction(w_epoch_t=w_epoch_t,
                                                  target_fs=target_fs,
                                                  baseline_mode=baseline_mode,
                                                  w_baseline_t=w_baseline_t,
                                                  norm=norm)
    features, track_info = feature_extractor.transform_dataset(dataset=dataset)
    lcha = dataset.channel_set.l_cha
    return features, track_info, fs, lcha, channel_set, saved_args


def plot_erd_ers_time(files, ch_to_plot, features=None, track_info=None,
                      fs=None, lcha=None, channel_set=None, mov_mean_ms=1000,
                      **kwargs):
    """Plotting function of ERD/ERS from motor imagery runs of MEDUSA.
    Parameters
    ----------
    files: list
        List of paths pointing to MI files.
    ch_to_plot: list
        List with the labels of the channels to plot
    """
    for key, value in kwargs.items():
        globals()[key] = value

    # Extract only if required
    if features is None:
        features, track_info, fs, lcha, channel_set, saved_args = \
            _extract_erd_ers_features(
                files, ch_to_plot, **kwargs
            )
        for key, value in saved_args.items():
            globals()[key] = value

    labels = track_info["mi_labels"]
    # todo: hardcoded
    labels_info = track_info["mi_labels_info"][0]
    new_fs = fs if target_fs is None else target_fs

    # # Baseline parameters
    t_baseline = [w_baseline_t[0] - w_epoch_t[0],
                  w_baseline_t[1] - w_epoch_t[0]]
    idx_baseline = np.round(np.array(t_baseline) * new_fs / 1000).astype(int)

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
    # TODO: Cambiar por https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.uniform_filter1d.html

    ERDERS_c1_smooth = uniform_filter1d(ERDERS_c1,
                                        int(np.floor(
                                            mov_mean_ms * new_fs / 1000)),
                                        axis=0, mode='mirror')
    ERDERS_c2_smooth = uniform_filter1d(ERDERS_c2,
                                        int(np.floor(
                                            mov_mean_ms * new_fs / 1000)),
                                        axis=0, mode='mirror')

    # Signed r2 for the power
    p_c1_marg = 100 * (p_c1 - r_c1_mean / p_c1.shape[0]) / r_c1_mean
    p_c2_marg = 100 * (p_c2 - r_c2_mean / p_c2.shape[0]) / r_c2_mean
    trials_r2 = statistics.signed_r2(p_c1_marg, p_c2_marg, signed=False, axis=0)
    if mov_mean_ms != 0:
        trials_r2 = uniform_filter1d(trials_r2,
                                     int(np.floor(mov_mean_ms * new_fs / 1000)),
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

        ax1 = fig.add_axes(
            [left, bottom + height_r2 + height_cbar + gap, width, height_psd],
            xticklabels=[])
        ax2 = fig.add_axes([left, bottom + height_cbar + gap, width, height_r2],
                           yticklabels=[])

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
        ax1.plot(times, ERDERS_c1_smooth[:, i], linewidth=2,
                 color=[255 / 255, 174 / 255, 0 / 255])
        ax1.plot(times, ERDERS_c2_smooth[:, i], linewidth=2,
                 color=[24 / 255, 255 / 255, 73 / 255])
        ax1.set_xlim(w_epoch_t)
        ax1.title.set_text(ch_to_plot[n])
        ax1.set_ylabel(r'ERD/ERS (%)')
        ax1.legend([labels_info[str(0)], labels_info[str(1)]])

        # Signed-r2
        ax2.pcolormesh(times, range(2), np.tile(trials_r2[:, i], reps=[2, 1]),
                       cmap='YlOrRd')
        ax2.axvline(x=0, color='k', linestyle='--', label='_nolegend_')
        ax2.set_ylabel('$r^2$')
        ax2.set_xlabel('Time (ms)')
        figs.append(fig)

    return figs


def plot_erd_ers_freq(files, ch_to_plot, features=None, track_info=None,
                      fs=None, lcha=None, channel_set=None,
                      welch_seg_len_pct=50,
                      welch_overlap_pct=75, mov_mean_hz=0,
                      **kwargs):
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
    for key, value in kwargs.items():
        globals()[key] = value

    # Extract only if required
    if features is None:
        features, track_info, fs, lcha, channel_set, saved_args = \
            _extract_erd_ers_features(
                files, ch_to_plot, **kwargs
            )
        for key, value in saved_args.items():
            globals()[key] = value

    labels = track_info["mi_labels"]
    # todo: hardcoded
    labels_info = track_info["mi_labels_info"][0]

    # Compute the PSD
    trials_psd = None
    new_fs = fs if target_fs is None else target_fs
    for t in features:
        # Compute PSD of the trial
        welch_seg_len = np.round(welch_seg_len_pct / 100 * t.shape[0]).astype(
            int)
        welch_overlap = np.round(
            welch_overlap_pct / 100 * welch_seg_len).astype(int)
        welch_ndft = welch_seg_len
        t_freqs, t_psd = scisig.welch(t, fs=new_fs, nperseg=welch_seg_len,
                                      noverlap=welch_overlap,
                                      nfft=welch_ndft, axis=0)

        # Concatenate
        t_psd = t_psd.reshape(1, t_psd.shape[0], t_psd.shape[1])
        trials_psd = np.concatenate((trials_psd, t_psd),
                                    axis=0) if trials_psd is not None else t_psd

    # Separate the classes
    trials_psd_c1 = trials_psd[labels == 0, :, :]
    trials_psd_c2 = trials_psd[labels == 1, :, :]

    # Signed r2
    trials_r2 = statistics.signed_r2(trials_psd_c1, trials_psd_c2, signed=False,
                                     axis=0)
    if mov_mean_hz != 0:
        size = int(
            trials_psd_c1.shape[1] / (cutoff[1] - cutoff[0]) * mov_mean_hz)
        trials_r2 = uniform_filter1d(trials_r2, size,
                                     axis=0, mode='nearest')

    # Mean PSD
    m_psd_c1 = np.mean(trials_psd_c1, axis=0)
    m_psd_c2 = np.mean(trials_psd_c2, axis=0)

    # Plot ranges
    freqs = np.linspace(0, new_fs / 2, len(m_psd_c1))
    lims = [0, new_fs / 2]
    if btype == 'bandpass':
        lims = [cutoff[0],
                cutoff[1]]
    elif btype == 'highpass':
        lims[0] = cutoff[0]
    elif btype == 'lowpass':
        lims[1] = cutoff[1]

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

        ax1 = fig.add_axes(
            [left, bottom + height_r2 + height_cbar + gap, width, height_psd],
            xticklabels=[])
        ax2 = fig.add_axes([left, bottom + height_cbar + gap, width, height_r2],
                           yticklabels=[])

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
        ax1.plot(freqs, m_psd_c1[:, i], linewidth=2,
                 color=[255 / 255, 174 / 255, 0 / 255])
        ax1.plot(freqs, m_psd_c2[:, i], linewidth=2,
                 color=[24 / 255, 255 / 255, 73 / 255])
        ax1.set_xlim(lims)
        ax1.title.set_text(ch_to_plot[n])
        ax1.set_ylabel(r'PSD ($uV^2/Hz$)')
        ax1.legend([labels_info[str(0)], labels_info[str(1)]])

        # Signed-r2
        ax2.pcolormesh(freqs, range(2), np.tile(trials_r2[:, i], reps=[2, 1]),
                       cmap='YlOrRd', vmin=0)
        ax2.set_xlim(lims)
        ax2.set_ylabel('$r^2$')
        ax2.set_xlabel('Frequency (Hz)')
        figs.append(fig)
    return figs


def plot_r2_topoplot(files, ch_to_plot, features=None, track_info=None,
                     fs=None, lcha=None, channel_set=None,
                     welch_seg_len_pct=50,
                     welch_overlap_pct=75, background=False, **kwargs):
    for key, value in kwargs.items():
        globals()[key] = value

    # Extract only if required
    if features is None:
        features, track_info, fs, lcha, channel_set, saved_args = \
            _extract_erd_ers_features(
                files, ch_to_plot, **kwargs
            )
        for key, value in saved_args.items():
            globals()[key] = value

    labels = track_info["mi_labels"]
    # todo: hardcoded
    labels_info = track_info["mi_labels_info"][0]
    new_fs = fs if target_fs is None else target_fs

    # Compute the PSD
    trials_psd = None
    for t in features:
        # Compute PSD of the trial
        welch_seg_len = np.round(welch_seg_len_pct / 100 * t.shape[0]).astype(
            int)
        welch_overlap = np.round(
            welch_overlap_pct / 100 * welch_seg_len).astype(int)
        welch_ndft = welch_seg_len
        t_freqs, t_psd = scisig.welch(t, fs=new_fs, nperseg=welch_seg_len,
                                      noverlap=welch_overlap,
                                      nfft=welch_ndft, axis=0)

        # Concatenate
        t_psd = t_psd.reshape(1, t_psd.shape[0], t_psd.shape[1])
        trials_psd = np.concatenate((trials_psd, t_psd),
                                    axis=0) if trials_psd is not None else t_psd

    # Separate the classes
    trials_psd_c1 = trials_psd[labels == 0, :, :]
    trials_psd_c2 = trials_psd[labels == 1, :, :]

    # Signed r2
    trials_r2 = statistics.signed_r2(trials_psd_c1, trials_psd_c2, signed=True,
                                     axis=0)
    trials_r2 = np.mean(trials_r2, axis=0)

    # Topoplot
    values = trials_r2.reshape(1, len(lcha))
    fig, _, _ = topographic_plots.plot_topography(channel_set,
                                                  values, cmap='RdBu',
                                                  background=background,
                                                  show=False)

    return fig
