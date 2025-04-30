from medusa import frequency_filtering as ff
from medusa import spatial_filtering as sf
# from medusa.storage.medusa_data import MedusaData
from medusa.components import Recording
# from medusa.bci.mi_feat_extraction import extract_mi_trials_from_midata
from medusa.local_activation import statistics
# from medusa.bci.mi_models import MIModelSettings
from medusa.plots import head_plots
from medusa.bci.mi_paradigms import StandardPreprocessing, \
    StandardFeatureExtraction, MIDataset

import numpy as np
from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt
import scipy.signal as scisig
import copy
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import fdrcorrection


class MIPlots:
    # TODO: Currently only 2 classes are supported and hardcoded, extend!
    def __init__(self):
        self.filter_type = "FIR"
        self.filter_order = 1000
        self.filter_btype = "bandpass"
        self.filter_filt_method = "filtfilt"
        self.apply_car = True

        # Generated
        self.raw_dataset = None
        self.fs = None
        self.channel_set = None

        self.set_sizes()

    def set_sizes(self, label_size=6, axes_size=5, line_width=1):
        self.label_size = label_size
        self.axes_size = axes_size
        self.line_width = line_width

    def set_dataset(self, files):
        """ Call this method to configure a dataset. It must be called before
        plotting anything.

        Parameters
        ----------
        files: list()
            List of mi.bson files.
        """
        rec = Recording.load(files[0])
        self.fs = rec.eeg.fs
        self.channel_set = rec.eeg.channel_set
        self.raw_dataset = MIDataset(channel_set=self.channel_set, fs=self.fs,
                                     experiment_att_key='midata',
                                     biosignal_att_key='eeg',
                                     experiment_mode='train')
        for file in files:
            self.raw_dataset.add_recordings(Recording.load(file))

    def plot_spectrogram(self, ch_to_plot, axs_to_plot=None,
                         f_lims=(8, 30),
                         t_trial_window=(1000, 6000),
                         t_reference_window=(-1000, 0),
                         welch_seg_len_pct=30,
                         welch_overlap_pct=96):
        # TODO: in process
        def _extract_features(full_window):
            if self.filter_type == "IIR":
                filter = ff.IIRFilter(order=self.filter_order,
                                      cutoff=f_lims,
                                      btype=self.filter_btype,
                                      filt_method=self.filter_filt_method)
            else:
                filter = ff.FIRFilter(order=self.filter_order,
                                      cutoff=f_lims,
                                      btype=self.filter_btype,
                                      filt_method=self.filter_filt_method)
            filter.fit(fs=self.fs)
            dataset = copy.deepcopy(self.raw_dataset)
            for rec in dataset.recordings:
                eeg = getattr(rec, dataset.biosignal_att_key)
                eeg.signal = filter.transform(signal=eeg.signal)
                if self.apply_car:
                    eeg.signal = sf.car(signal=eeg.signal)
                setattr(rec, dataset.biosignal_att_key, eeg)
            feature_extractor = StandardFeatureExtraction()
            trials, track_info = feature_extractor.transform_dataset(
                dataset=dataset,
                w_epoch_t=full_window,
                target_fs=None, baseline_mode=None,
                safe_copy=False, w_baseline_t=None, norm=None
            )
        return trials, track_info

        if self.raw_dataset is None:
            raise Exception("Call MiPlots._extract_features() before plotting!")
        if axs_to_plot is None:
            axs_to_plot = list()
            fig = plt.figure(figsize=(7.5, 3), dpi=300)
            l_ = len(ch_to_plot)
            for c in range(l_):
                gs = fig.add_gridspec(3, l_, wspace=0.2, hspace=0.2,
                                      height_ratios=[1, 1])
                axs_to_plot.append({'spec_c1': fig.add_subplot(gs[0, c]),
                                    'spec_c2': fig.add_subplot(gs[1, c])})

        # Get features
        t1 = np.min([t_trial_window + t_reference_window])
        t2 = np.max([t_trial_window + t_reference_window])
        full_trials, track_info = _extract_features(full_window=(t1, t2))
        labels = track_info["mi_labels"]
        labels_info = track_info["mi_labels_info"][0]

        # Separate the classes
        trials_c1 = full_trials[labels == 0, :, :]
        trials_c2 = full_trials[labels == 1, :, :]

        # Compute the spectrogram
        welch_seg_len = np.round(welch_seg_len_pct / 100 *
                                 full_trials.shape[1]).astype(int)
        welch_overlap = np.round(welch_overlap_pct / 100 *
                                 welch_seg_len).astype(int)
        freqs, times, spec_c1 = scisig.spectrogram(
            trials_c1, axis=1, fs=self.fs, nperseg=welch_seg_len,
            noverlap=welch_overlap, nfft=welch_seg_len)
        freqs, times, spec_c2 = scisig.spectrogram(
            trials_c2, axis=1, fs=self.fs, nperseg=welch_seg_len,
            noverlap=welch_overlap, nfft=welch_seg_len)

        s_win = ((np.array(t_trial_window) - t1) * times.shape[0] /
                 (t2 - t1)).astype(int)
        r_win = ((np.array(t_reference_window) - t1) * times.shape[0] /
                 (t2 - t1)).astype(int)

        # ERD/ERS (%) [n_trials x time x n_cha x frequency]
        power_trial_c1 = np.power(spec_c1, 2)
        R_c1 = np.mean(np.power(spec_c1[:, r_win[0]:r_win[1], :, :], 2), axis=1)
        R_c1 = np.expand_dims(R_c1, axis=1)
        erders_c1 = 100 * (power_trial_c1 - R_c1) / R_c1
        power_trial_c2 = np.power(spec_c2, 2)
        R_c2 = np.mean(np.power(spec_c2[:, r_win[0]:r_win[1], :, :], 2), axis=1)
        R_c2 = np.expand_dims(R_c2, axis=1)
        erders_c2 = 100 * (power_trial_c2 - R_c2) / R_c2

        # Average
        erders_c1_avg = np.mean(erders_c1, axis=0)
        erders_c2_avg = np.mean(erders_c2, axis=0)

        # Plot
        c_lims = [10, -10]
        for n in range(len(ch_to_plot)):
            if ch_to_plot[n] not in self.raw_dataset.channel_set.l_cha:
                raise ValueError('Channel ' + ch_to_plot[n] + ' is missing!')
            i = self.raw_dataset.channel_set.l_cha.index(ch_to_plot[n])
            if np.max(erders_c1_avg[:, i, :]) > c_lims[1]:
                c_lims[1] = np.max(erders_c1_avg[:, i, :])
            if np.max(erders_c2_avg[:, i, :]) > c_lims[1]:
                c_lims[1] = np.max(erders_c2_avg[:, i, :])
            if np.min(erders_c1_avg[:, i, :]) < c_lims[0]:
                c_lims[0] = np.min(erders_c1_avg[:, i, :])
            if np.min(erders_c2_avg[:, i, :]) < c_lims[0]:
                c_lims[0] = np.min(erders_c2_avg[:, i, :])

        for n in range(len(ch_to_plot)):
            if ch_to_plot[n] not in self.raw_dataset.channel_set.l_cha:
                raise ValueError('Channel ' + ch_to_plot[n] + ' is missing!')
            i = self.raw_dataset.channel_set.l_cha.index(ch_to_plot[n])

            with plt.style.context('seaborn-v0_8'):
                # Averaged curves
                if "spec_c1" in axs_to_plot[n]:
                    ax1 = axs_to_plot[n]['spec_c1']
                    ax1.minorticks_on()
                    im = ax1.pcolormesh(times, freqs,
                                   np.squeeze(erders_c1_avg[:, i, :]),
                                   cmap='RdBu_r', vmin=c_lims[0],
                                   vmax=c_lims[1])
                    ax1.set_ylabel('Frequency (Hz)', fontsize=self.label_size)
                    ax1.set_xlabel('Time (ms)', fontsize=self.label_size)
                    ax1.tick_params(axis='x', labelsize=self.axes_size)
                    ax1.tick_params(axis='y', labelsize=self.axes_size)
                    ax1.set_ylim(f_lims)
                    # TODO: add colorbar
                if "spec_c2" in axs_to_plot[n]:
                    ax2 = axs_to_plot[n]['spec_c2']
                    ax2.minorticks_on()
                    im = ax2.pcolormesh(times, freqs,
                                   np.squeeze(erders_c2_avg[:, i, :]),
                                   cmap='RdBu_r', vmin=c_lims[0],
                                   vmax=c_lims[1])
                    ax2.set_ylabel('Frequency (Hz)', fontsize=self.label_size)
                    ax2.set_xlabel('Time (ms)', fontsize=self.label_size)
                    ax2.tick_params(axis='x', labelsize=self.axes_size)
                    ax2.tick_params(axis='y', labelsize=self.axes_size)
                    ax2.set_ylim(f_lims)
                    # TODO: add colorbar
        return axs_to_plot

    def plot_erd_ers_freq(self, ch_to_plot, axs_to_plot=None,
                          f_lims=(5, 40),
                          f_sel=(8, 13),
                          t_trial_window=(1000, 6000),
                          welch_seg_len_pct=50,
                          welch_overlap_pct=75,
                          mov_mean_hz=0):
        """ Plots the ERD/ERS in the frequency domain.

        Parameters
        -----------
        ch_to_plot: list(basestring)
            List of channels to be plotted (commonly: ["C3", "C4"])
        axs_to_plot: list(dict())
            List of dictionaries. Each dictionary belongs to each channel to
            be plotted, thus the number of dictionaries must be equal to
            len(ch_to_plot). Dictionary must have the following items:
                - "freq" (matplotib.axes): axes for the PSD plot.
                - "r2" (matplotlib.axes): axes for the statistic r2 plot.
                - "pval" (matplotlib.axes): axes for the p-values plot (
                p-values are computed using a Wilcoxon signed-rank test and
                correcting FDR using Benjamini-Hochberg).
        f_lims: tuple()
            Tuple containing the frequency limits of the PSD (i.e.,
            the filtering cutoff)
        f_sel: tuple()
            Tuple containing the desired frequency band for visualization
            purposes (it will be indicated with a shaded area)
        t_trial_window: tuple()
            Trial window to be considered in ms.
        welch_seg_len_pct: int
            Percentage of the trial window used to extract segments for the
            PSD Welch estimation
        welch_overlap_pct: int
            Percentage of the segment length used to overlap segments.
        mov_mean_hz : int
            Resolution (Hz/bin) desired after a moving average filter (use 0
            to avoid the smooth filtering).

        Returns
        -----------
        matplotlib.axes:
            Modified axes
        """
        def _extract_features():
            dataset = copy.deepcopy(self.raw_dataset)
            if self.filter_type == "IIR":
                filter = ff.IIRFilter(order=self.filter_order,
                                      cutoff=f_lims,
                                      btype=self.filter_btype,
                                      filt_method=self.filter_filt_method)
            else:
                filter = ff.FIRFilter(order=self.filter_order,
                                      cutoff=f_lims,
                                      btype=self.filter_btype,
                                      filt_method=self.filter_filt_method)
            filter.fit(fs=self.fs)
            for rec in dataset.recordings:
                eeg = getattr(rec, dataset.biosignal_att_key)
                eeg.signal = filter.transform(signal=eeg.signal)
                if self.apply_car:
                    eeg.signal = sf.car(signal=eeg.signal)
                setattr(rec, dataset.biosignal_att_key, eeg)
            feature_extractor = StandardFeatureExtraction()
            features, track_info = feature_extractor.transform_dataset(
                dataset=dataset,
                w_epoch_t=t_trial_window,
                target_fs=None, baseline_mode=None,
                safe_copy=False, w_baseline_t=None, norm=None
            )
            return features, track_info

        if self.raw_dataset is None:
            raise Exception("Call MiPlots.set_dataset() before plotting!")
        if axs_to_plot is None:
            axs_to_plot = list()
            fig = plt.figure(figsize=(7.5, 3), dpi=300)
            l_ = len(ch_to_plot)
            for c in range(l_):
                gs = fig.add_gridspec(3, l_, wspace=0.2, hspace=0.2,
                                      height_ratios=[1, 0.1, 0.1])
                axs_to_plot.append({'freq': fig.add_subplot(gs[0, c]),
                                    'r2': fig.add_subplot(gs[1, c]),
                                    'pval': fig.add_subplot(gs[2, c])})

        # Get features
        features, track_info = _extract_features()
        labels = track_info["mi_labels"]
        labels_info = track_info["mi_labels_info"][0]

        # Compute the PSD
        welch_seg_len = np.round(welch_seg_len_pct / 100 *
                                 features.shape[1]).astype(int)
        welch_overlap = np.round(welch_overlap_pct / 100 *
                                 welch_seg_len).astype(int)
        trials_psd = None
        for t in features:
            # Compute PSD of the trial
            t_freqs, t_psd = scisig.welch(t, fs=self.fs, nperseg=welch_seg_len,
                                          noverlap=welch_overlap,
                                          nfft=welch_seg_len, axis=0)
            # Concatenate
            t_psd = np.expand_dims(t_psd, axis=0)
            trials_psd = np.concatenate((trials_psd, t_psd), axis=0) if \
                trials_psd is not None else t_psd

        # Smoothing?
        if mov_mean_hz != 0:
            size = int(mov_mean_hz * trials_psd.shape[1] / (0.5 * self.fs))
            trials_psd = uniform_filter1d(trials_psd, size, axis=0)

        # Separate the classes
        trials_psd_c1 = trials_psd[labels == 0, :, :]
        trials_psd_c2 = trials_psd[labels == 1, :, :]

        # Signed r2
        trials_r2 = statistics.signed_r2(trials_psd_c1, trials_psd_c2,
                                         signed=False, axis=0)
        # Wilcoxon signed-rank test
        trials_p = wilcoxon(trials_psd_c1, trials_psd_c2, axis=0)
        trials_p = trials_p.pvalue
        trials_p_fdr = np.zeros(trials_p.shape)
        for j in range(self.raw_dataset.channel_set.n_cha):
            # Correct FDR (Benjamini-Hochberg)
            _, p_ = fdrcorrection(trials_p[:, j], alpha=0.05, is_sorted=False)
            trials_p_fdr[:, j] = p_

        # Mean PSD
        m_psd_c1 = np.mean(trials_psd_c1, axis=0)
        m_psd_c2 = np.mean(trials_psd_c2, axis=0)

        # Plot
        freqs = np.linspace(0, self.fs / 2, len(m_psd_c1))
        for n in range(len(ch_to_plot)):
            if ch_to_plot[n] not in self.channel_set.l_cha:
                raise ValueError('Channel ' + ch_to_plot[n] + ' is missing!')
            i = self.channel_set.l_cha.index(ch_to_plot[n])

            with plt.style.context('seaborn-v0_8'):
                # Averaged curves
                if "freq" in axs_to_plot[n]:
                    ax1 = axs_to_plot[n]['freq']
                    ax1.minorticks_on()
                    ax1.grid(visible=True, which='minor', color='#ededed',
                             linestyle='--')
                    ax1.grid(visible=True, which='major')

                    # Selected band box
                    mi_ = np.min([np.min(m_psd_c1[:, i]),
                                  np.min(m_psd_c2[:, i])])
                    ma_ = np.max([np.max(m_psd_c1[:, i]),
                                  np.max(m_psd_c2[:, i])])
                    off_ = 0.1 * (ma_ - mi_)
                    tx = (f_sel[0], f_sel[1], f_sel[1], f_sel[0])
                    ty = (mi_ - off_, mi_ - off_, ma_ + off_, ma_ + off_)
                    ax1.fill(tx, ty, edgecolor=None, facecolor="#D1F0FF55",
                             label='_nolegend_')

                    # Lines
                    ax1.plot(freqs, m_psd_c1[:, i], linewidth=self.line_width,
                             color=[255 / 255, 174 / 255, 0 / 255])
                    ax1.plot(freqs, m_psd_c2[:, i], linewidth=self.line_width,
                             color=[24 / 255, 255 / 255, 73 / 255])
                    ax1.set_xlim(f_lims)
                    ax1.set_ylim([mi_ - off_, ma_ + off_])
                    ax1.set_title(ch_to_plot[n], fontsize=self.label_size)
                    ax1.set_ylabel(r'PSD ($uV^2/Hz$)', fontsize=self.label_size)
                    ax1.legend([labels_info[str(0)], labels_info[str(1)]],
                               fontsize=self.label_size)
                    ax1.set_xlabel('Frequency (Hz)', fontsize=self.label_size)
                    ax1.tick_params(axis='x', labelsize=self.axes_size)
                    ax1.tick_params(axis='y', labelsize=self.axes_size)

                # Signed-r2
                if "r2" in axs_to_plot[n]:
                    ax2 = axs_to_plot[n]['r2']
                    ax2.pcolormesh(freqs, range(2),
                                   np.tile(trials_r2[:, i], reps=[2, 1]),
                                   cmap='YlOrRd',
                                   vmin=0)
                    ax2.set_xlim(f_lims)
                    ax2.set_ylabel('$r^2$', fontsize=self.label_size)
                    ax2.set_xlabel('Frequency (Hz)', fontsize=self.label_size)
                    ax2.tick_params(axis='x', labelsize=self.axes_size)
                    ax2.tick_params(axis='y', labelsize=self.axes_size)
                    ax2.get_yaxis().set_ticks([])

                # P-value < 0.05
                if "pval" in axs_to_plot[n]:
                    ax3 = axs_to_plot[n]['pval']
                    ax3.minorticks_on()
                    ax3.pcolormesh(freqs, range(2),
                                   np.tile(trials_p_fdr[:, i] <= 0.05,
                                           reps=[2, 1]),
                                   cmap='binary',
                                   vmin=0, vmax=0.05)
                    ax3.set_xlim(f_lims)
                    ax3.set_ylabel('$p$-val', fontsize=self.label_size)
                    ax3.set_xlabel('Frequency (Hz)', fontsize=self.label_size)
                    ax3.tick_params(axis='x', labelsize=self.axes_size)
                    ax3.tick_params(axis='y', labelsize=self.axes_size)
                    ax3.get_yaxis().set_ticks([])
        return axs_to_plot

    def plot_erd_ers_time(self, ch_to_plot, axs_to_plot=None,
                          t_trial_window=(1000, 6000),
                          t_reference_window=(-1000, 0),
                          f_cutoff=(8, 13),
                          mov_mean_ms=1000):
        """ Plots the ERD/ERS in the temporal domain.

        Parameters
        -----------
        ch_to_plot: list(basestring)
            List of channels to be plotted (commonly: ["C3", "C4"]).
        axs_to_plot: list(dict())
            List of dictionaries. Each dictionary belongs to each channel to
            be plotted, thus the number of dictionaries must be equal to
            len(ch_to_plot). Dictionary must have the following items:
                - "freq" (matplotib.axes): axes for the PSD plot.
                - "r2" (matplotlib.axes): axes for the statistic r2 plot.
                - "pval" (matplotlib.axes): axes for the p-values plot (
                p-values are computed using a Wilcoxon signed-rank test and
                correcting FDR using Benjamini-Hochberg).
        t_trial_window: tuple()
            Trial window to be considered in ms (relative to the onset).
        t_reference_window: tuple()
            Reference window to be considered in ms (relative to the onset).
        f_cutoff: tuple()
            Filter cutoff to select a desired band
        mov_mean_ms : int
            Resolution (ms/bin) desired after a moving average filter (use 0
            to avoid the smooth filtering).

        Returns
        -----------
        matplotlib.axes:
            Modified axes
        """
        def _extract_features(full_window):
            # all_trials : from reference to end of trial window
            # references : only reference window
            dataset = copy.deepcopy(self.raw_dataset)
            if self.filter_type == "IIR":
                filter = ff.IIRFilter(order=self.filter_order,
                                      cutoff=f_cutoff,
                                      btype=self.filter_btype,
                                      filt_method=self.filter_filt_method)
            else:
                filter = ff.FIRFilter(order=self.filter_order,
                                      cutoff=f_cutoff,
                                      btype=self.filter_btype,
                                      filt_method=self.filter_filt_method)
            filter.fit(fs=self.fs)
            for rec in dataset.recordings:
                eeg = getattr(rec, dataset.biosignal_att_key)
                eeg.signal = filter.transform(signal=eeg.signal)
                if self.apply_car:
                    eeg.signal = sf.car(signal=eeg.signal)
                setattr(rec, dataset.biosignal_att_key, eeg)
            feature_extractor = StandardFeatureExtraction()
            all_trials, track_info = feature_extractor.transform_dataset(
                dataset=dataset,
                w_epoch_t=full_window,
                target_fs=None, baseline_mode=None,
                safe_copy=False, w_baseline_t=None, norm=None
            )

            return all_trials, track_info

        if self.raw_dataset is None:
            raise Exception("Call MiPlots._extract_features() before plotting!")
        if axs_to_plot is None:
            axs_to_plot = list()
            fig = plt.figure(figsize=(7.5, 3), dpi=300)
            l_ = len(ch_to_plot)
            for c in range(ch_to_plot):
                gs = fig.add_gridspec(3, l_, wspace=0.2, hspace=0.2,
                                      height_ratios=[1, 0.1, 0.1])
                axs_to_plot.append({'time': fig.add_subplot(gs[0, c]),
                                    'r2': fig.add_subplot(gs[1, c]),
                                    'pval': fig.add_subplot(gs[2, c])})

        # Get features
        t1 = np.min([t_trial_window + t_reference_window])
        t2 = np.max([t_trial_window + t_reference_window])
        fw = (t1, t2)
        full_trials, track_info = _extract_features(full_window=(t1, t2))
        labels = track_info["mi_labels"]
        labels_info = track_info["mi_labels_info"][0]

        # Separate the classes
        s_win = ((np.array(t_trial_window) - t1) * full_trials.shape[1] /
                 (t2 - t1)).astype(int)
        r_win = ((np.array(t_reference_window) - t1) * full_trials.shape[1] /
                 (t2 - t1)).astype(int)
        trials_c1 = full_trials[labels == 0, :, :]
        trials_c2 = full_trials[labels == 1, :, :]

        # Compute the ERD/ERS
        power_trial_c1 = np.power(trials_c1, 2)
        power_ref_avg_c1 = np.mean(np.mean(
            np.power(trials_c1[:, r_win[0]:r_win[1], :], 2), axis=0), axis=0)
        erders_c1 = 100 * (power_trial_c1 - power_ref_avg_c1) / power_ref_avg_c1
        power_trial_c2 = np.power(trials_c2, 2)
        power_ref_avg_c2 = np.mean(np.mean(
            np.power(trials_c2[:, r_win[0]:r_win[1], :], 2), axis=0), axis=0)
        erders_c2 = 100 * (power_trial_c2 - power_ref_avg_c2) / power_ref_avg_c2

        # Smoothing?
        if mov_mean_ms != 0:
            size = int(mov_mean_ms * self.fs / 1000)
            erders_c1 = uniform_filter1d(erders_c1, size, axis=1)
            erders_c2 = uniform_filter1d(erders_c2, size, axis=1)

        # Signed r2
        trials_r2 = statistics.signed_r2(erders_c1, erders_c2, signed=False,
                                         axis=0)

        # Wilcoxon signed-rank test
        trials_p = wilcoxon(erders_c1, erders_c2, axis=0)
        trials_p = trials_p.pvalue
        trials_p_fdr = np.zeros(trials_p.shape)
        for j in range(self.raw_dataset.channel_set.n_cha):
            # Correct FDR (Benjamini-Hochberg)
            _, p_ = fdrcorrection(trials_p[:, j], alpha=0.05, is_sorted=False)
            trials_p_fdr[:, j] = p_

        # Plot
        erders_c1_avg = np.mean(erders_c1, axis=0)
        erders_c2_avg = np.mean(erders_c2, axis=0)
        lcha = self.raw_dataset.channel_set.l_cha
        times = np.linspace(t1, t2, erders_c1.shape[1])
        for n in range(len(ch_to_plot)):
            if ch_to_plot[n] not in lcha:
                raise ValueError('Channel ' + ch_to_plot[n] + ' is missing!')
            i = lcha.index(ch_to_plot[n])

            with plt.style.context('seaborn-v0_8'):
                # Averaged curves
                if "time" in axs_to_plot[n]:
                    ax1 = axs_to_plot[n]['time']
                    ax1.minorticks_on()
                    ax1.grid(visible=True, which='minor', color='#ededed',
                             linestyle='--')
                    ax1.grid(visible=True, which='major')

                    # Reference box
                    m_ = np.max([
                        np.max(np.abs(erders_c1_avg[:, i])),
                        np.max(np.abs(erders_c2_avg[:, i]))
                    ])
                    rx = (t_reference_window[0], t_reference_window[1],
                          t_reference_window[1], t_reference_window[0])
                    ry = (-0.2 * m_, -0.2 * m_, 0.2 * m_, 0.2 * m_)
                    ax1.fill(rx, ry, edgecolor=None, facecolor="#FFBEF055",
                             label='_nolegend_')

                    # Trial box
                    mi_ = np.min([np.min(erders_c1_avg[:, i]),
                                  np.min(erders_c2_avg[:, i])])
                    ma_ = np.max([np.max(erders_c1_avg[:, i]),
                                  np.max(erders_c2_avg[:, i])])
                    off_ = 0.1 * (ma_ - mi_)
                    tx = (t_trial_window[0], t_trial_window[1],
                          t_trial_window[1], t_trial_window[0])
                    ty = (mi_ - off_, mi_ - off_, ma_ + off_, ma_ + off_)
                    ax1.fill(tx, ty, edgecolor=None, facecolor="#D1F0FF55",
                             label='_nolegend_')

                    # Onset line
                    ax1.plot((0, 0), (mi_ - off_, ma_ + off_), '--k',
                             linewidth=self.line_width/2,
                             label='_nolegend_')

                    # ERD/ERS lines
                    l1 = ax1.plot(times, erders_c1_avg[:, i],
                                  linewidth=self.line_width,
                                  color=[255 / 255, 174 / 255, 0 / 255])
                    l2 = ax1.plot(times, erders_c2_avg[:, i],
                                  linewidth=self.line_width,
                                  color=[24 / 255, 255 / 255, 73 / 255])
                    ax1.set_xlim([t1, t2])
                    ax1.set_ylim([mi_ - off_, ma_ + off_])
                    ax1.set_title(ch_to_plot[n], fontsize=self.label_size)
                    ax1.set_ylabel(r'ERD/ERS (%)', fontsize=self.label_size)
                    ax1.legend([labels_info[str(0)], labels_info[str(1)]],
                               fontsize=self.label_size, loc='upper left')
                    ax1.set_xlabel('Time (ms)', fontsize=self.label_size)
                    ax1.tick_params(axis='x', labelsize=self.axes_size)
                    ax1.tick_params(axis='y', labelsize=self.axes_size)

                # Signed-r2
                if "r2" in axs_to_plot[n]:
                    ax2 = axs_to_plot[n]['r2']
                    ax2.minorticks_on()
                    ax2.pcolormesh(times, range(2),
                                   np.tile(trials_r2[:, i], reps=[2, 1]),
                                   cmap='YlOrRd',
                                   vmin=0)
                    ax2.set_xlim([t1, t2])
                    ax2.set_ylabel('$r^2$', fontsize=self.label_size)
                    ax2.set_xlabel('Time (ms)', fontsize=self.label_size)
                    ax2.tick_params(axis='x', labelsize=self.axes_size)
                    ax2.tick_params(axis='y', labelsize=self.axes_size)
                    ax2.get_yaxis().set_ticks([])

                # P-value < 0.05
                if "pval" in axs_to_plot[n]:
                    ax3 = axs_to_plot[n]['pval']
                    ax3.minorticks_on()
                    ax3.pcolormesh(times, range(2),
                                   np.tile(trials_p_fdr[:, i] <= 0.05,
                                           reps=[2, 1]),
                                   cmap='binary',
                                   vmin=0, vmax=0.05)
                    ax3.set_xlim([t1, t2])
                    ax3.set_ylabel('$p$-val', fontsize=self.label_size)
                    ax3.set_xlabel('Time (ms)', fontsize=self.label_size)
                    ax3.tick_params(axis='x', labelsize=self.axes_size)
                    ax3.tick_params(axis='y', labelsize=self.axes_size)
                    ax3.get_yaxis().set_ticks([])
        return axs_to_plot

    def plot_erd_ers_r2_topo(self, ch_to_plot, ax_to_plot=None,
                             welch_seg_len_pct=50,
                             welch_overlap_pct=75):
        if self.raw_dataset is None:
            raise Exception("Call MiPlots._extract_features() before plotting!")
        if len(ch_to_plot) != 2:
            raise Exception("We need exactly two channels to compute r2 topo!")
        if ax_to_plot is None:
            ax_to_plot = list()
            for c in ch_to_plot:
                fig = plt.figure(figsize=(5, 5), dpi=300)
                ax_to_plot = fig.add_subplot(111)
        lcha = self.raw_dataset.channel_set.l_cha
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
        topo_settings = {
            "head_radius": 1.0,
            "head_line_width": self.line_width * 2,
            "interp_contour_width": self.line_width,
            "interp_points": 500,
            "cmap": "RdBu_r",
            "clim": (-max_r2, max_r2)
        }
        topo = head_plots.TopographicPlot(
            axes=ax_to_plot, channel_set=self.raw_dataset.channel_set,
            **topo_settings
        )
        topo.update(values=values)
        ax_to_plot.set_title("Signed $r^2$ (%s)" % ' vs. '.join(ch_to_plot),
                             fontsize=self.label_size)

        # return ax_to_plot, handles["color-mesh"]
        return ax_to_plot, topo.plot_handles["color-mesh"]


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
                        experiment_att_key='midata',
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
