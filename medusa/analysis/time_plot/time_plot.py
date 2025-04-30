# External imports
import os
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QIcon, QShortcut, QCursor
from PySide6.QtCore import Qt
from PySide6.QtUiTools import loadUiType
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib as mpl
from matplotlib.patches import Rectangle
from matplotlib.offsetbox import AuxTransformBox, HPacker, TextArea
from matplotlib.offsetbox import AnchoredOffsetbox
from matplotlib.transforms import BboxBase

# MEDUSA imports
from medusa.utils import check_dimensions

# Constants
HOVER_LEFT_PREVIEW_LIMIT = 0
HOVER_RIGHT_PREVIEW_LIMIT = 1
HOVER_CENTER_PREVIEW = 2
PLOT_PARAMS = {
    'lines.linewidth': 0.7,
    'font.size': 8,
    'axes.labelsize': 6,
    'axes.labelweight': 'bold',
    'axes.grid': True,
    'axes.grid.which': 'both',
    'grid.linewidth': 0.7,
    'xtick.labelsize': 6,
    'xtick.minor.width': 0.5,
    'xtick.major.width': 0.7,
    'ytick.labelsize': 6,
    'ytick.major.width': 0.7,
    'legend.fontsize': 6
}

# UI file
curr_dir = os.path.dirname(os.path.abspath(__file__))
ui_file = loadUiType(curr_dir + "/time_plot.ui")[0]


class VerticalScaleBar:

    def __init__(self, axes, bar_height, bar_label='uV',
                 label_size=6):
        # Create the rectangle and the label
        bars = AuxTransformBox(axes.transData)
        bars.add_artist(
            Rectangle((0, 0), 0, bar_height, ec='k',
                      lw=0.7, fc="none", )
        )
        self.packer = HPacker(
            children=[bars, TextArea(bar_label, textprops={
                "ha": "center",
                "va": "center",
                "size": label_size
            })],
            align="center", pad=0, sep=4
        )

        # Create the anchored box
        self.bar = AnchoredOffsetbox(
            loc='upper left', pad=0, borderpad=0, child=self.packer,
            prop=None, frameon=False, bbox_to_anchor=(1.01, 1),
            bbox_transform=axes.transAxes
        )


class PlotCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5.0, height=4.0, dpi=100):
        """ This class is used to ease the embedding of matplotlib plots into the widget. """
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        self.parent = parent
        super(PlotCanvas, self).__init__(self.fig)

    def wheelEvent(self, event):
        # event.angleDelta().y() gives the scroll direction
        delta = event.angleDelta().y()
        if delta > 0:
            self.parent.scrolled_up()
        else:
            self.parent.scrolled_down()


class TimePlot(QtWidgets.QDialog, ui_file):

    def __init__(self, ch_to_show=None, zoom_multiplier=1.2, vis_step_s=2,
                 units="μV", initial_zoom=1, initial_window_s=10):
        """
            Initializes the TimePlot window with the specified parameters for
            visualizing multi-channel time-series data.

            Shortcuts
            -----------
            Left & right arrows: rewinds or forwards the visualizing window.
            Up & down arrows: increases or decreases the visible channels.
            Mouse scroll: modifies the zoom.

            Parameters:
            -----------
            ch_to_show : int or None, optional
                Number of channels to display in the plot at once. This
                value must be less than or equal to the total number of
                channels available in the data. If set to None, the number of
                channels to show is initialized to the total number of available
                channels.
            zoom_multiplier : float, optional
                Factor by which the plot zooms in or out when the zoom
                buttons are used. The default value is 1.2, meaning each zoom
                action will increase or decrease the visible range by 20%.
            vis_step_s : int, optional
                Time step (in seconds) used when navigating the plot, such as
                fast forwarding or rewinding. The default value is 2 seconds,
                meaning that each step moves the visible window forward or
                backward by 2 seconds.
            units : str, optional
                Units of the plot. The default value is "μV".
            initial_zoom : int, optional
                Initial zoom. The default value is 1.
            initial_window_s : float, optional
                Initial visualization window span in seconds. The default
                value is 10.0 s.
            """
        # Load the custom layout from UIC
        super().__init__()
        self.setupUi(self)
        self.dir = os.path.dirname(__file__)
        self.setWindowTitle("Time Plot")
        self.setWindowIcon(QIcon(
            os.path.join(self.dir, 'icons/medusa_task_icon.png')))
        self.setWindowFlags(
            Qt.Window |
            Qt.WindowMinimizeButtonHint |
            Qt.WindowMaximizeButtonHint |
            Qt.WindowCloseButtonHint
        )
        self.setSizeGripEnabled(True)

        # Style
        plt.style.use('seaborn-v0_8')
        for key, value in PLOT_PARAMS.items():
            mpl.rcParams[key] = value
        self.default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        # Parameters
        self.ch_to_show = ch_to_show
        self.zoom_multiplier = zoom_multiplier
        self.vis_step_s = vis_step_s
        self.units = units
        self.initial_window_s = initial_window_s

        # Buttons
        icon_ff = QIcon(os.path.join(self.dir, 'icons/ff.png'))
        icon_rr = QIcon(os.path.join(self.dir, 'icons/rr.png'))
        icon_zoomin = QIcon(os.path.join(self.dir, 'icons/zoom_in.png'))
        icon_zoomout = QIcon(os.path.join(self.dir, 'icons/zoom_out.png'))
        icon_download = QIcon(os.path.join(self.dir, 'icons/download.png'))
        icon_chup = QIcon(os.path.join(self.dir, 'icons/increase_channels.png'))
        icon_chdown = QIcon(os.path.join(self.dir, 'icons/decrease_channels.png'))
        self.btn_ff.setIcon(icon_ff)
        self.btn_ff.clicked.connect(self.fast_forward)
        self.btn_rr.setIcon(icon_rr)
        self.btn_rr.clicked.connect(self.rewind)
        self.btn_zoomin.setIcon(icon_zoomin)
        self.btn_zoomin.clicked.connect(self.zoom_in)
        self.btn_zoomout.setIcon(icon_zoomout)
        self.btn_zoomout.clicked.connect(self.zoom_out)
        self.btn_download.setIcon(icon_download)
        self.btn_download.clicked.connect(self.download)
        self.btn_ch_up.setIcon(icon_chup)
        self.btn_ch_up.clicked.connect(self.increase_channels)
        self.btn_ch_down.setIcon(icon_chdown)
        self.btn_ch_down.clicked.connect(self.decrease_channels)

        # Keypress events
        QShortcut(Qt.Key_Left, self, self.rewind)
        QShortcut(Qt.Key_Right, self, self.fast_forward)
        QShortcut(Qt.Key_Plus, self, self.zoom_in)
        QShortcut(Qt.Key_Minus, self, self.zoom_out)
        QShortcut(Qt.Key_Up, self, self.increase_channels)
        QShortcut(Qt.Key_Down, self, self.decrease_channels)

        # Initialize the canvas
        self.canvas = PlotCanvas(self, width=8.0, height=4.0, dpi=150)
        self.plot_layout.addWidget(self.canvas)
        self.canvas_preview = PlotCanvas(self, width=8.0, height=0.5, dpi=72)
        self.preview_layout.addWidget(self.canvas_preview)

        # Plot handles
        self.plotted_data = list()
        self.preview_handles = None

        # Initialization
        self.time_to_show = None
        self.zoom = initial_zoom
        self.current_offset = None
        self.current_vis_window = np.array([0, None])
        self.current_vis_ch = np.array([0, ch_to_show])
        self._preview_curr_focused = None
        self.preview_epsilon = 2      # Hardcoded precision to click the window
        self.plot_count = 0
        self.legend_items = {
            "signals": dict(),
            "events": dict(),
            "conditions": dict()
        }
        self.scale_bar = None

    ## ----------------------------- CONTROLS ----------------------------- ##
    def fast_forward(self):
        """ This method makes the time plot go forward. """
        self.current_vis_window = self._make_window_feasible(
            self.current_vis_window + self.vis_step_s
        )
        self.update_xlim()
        self.update_preview_window()

    def rewind(self):
        """ This method makes the time plot go backward. """
        self.current_vis_window = self._make_window_feasible(
            self.current_vis_window - self.vis_step_s
        )
        self.update_xlim()
        self.update_preview_window()

    def zoom_in(self):
        """ This method makes the time plot to zoom in. """
        self.zoom *= self.zoom_multiplier
        self.update_zoom_in_plots()

    def zoom_out(self):
        """ This method makes the time plot to zoom out. """
        self.zoom /= self.zoom_multiplier
        self.update_zoom_in_plots()

    def download(self):
        fdialog = QtWidgets.QFileDialog()
        fname = fdialog.getSaveFileName(
            fdialog, 'Export Figure', '../data/',
            'PNG file (*.png)')
        if fname[0]:
            try:
                dpi = int(self.edit_dpi.text())
            except Exception:
                dpi = 150
                print('Unknown DPIs, using %i...' % dpi)
            self.canvas.fig.savefig(fname=fname[0], dpi=dpi)

            # Save the snapshot
            print('Snapshot saved as %s with %i dpi' % (fname[0], dpi))

    def scrolled_up(self):
        """ Called by the PlotCanvas when the wheel scrolls up. """
        self.zoom_in()

    def scrolled_down(self):
        """ Called by the PlotCanvas when the wheel scrolls down. """
        self.zoom_out()

    def increase_channels(self):
        max_ch = self._get_max_channels()
        self.current_vis_ch += self.ch_to_show
        if self.current_vis_ch[1] >= max_ch:
            self.current_vis_ch[0] = max_ch - self.ch_to_show
            self.current_vis_ch[1] = max_ch
        self.update_ylim()

    def decrease_channels(self):
        self.current_vis_ch -= self.ch_to_show
        if self.current_vis_ch[0] < 0:
            self.current_vis_ch = np.array([0, self.ch_to_show])
        self.update_ylim()

    ## --------------------------- PLOT UPDATES --------------------------- ##
    def update_xlim(self):
        """ This method updates the X limits. """
        self.canvas.axes.set_xlim(
            left=self.current_vis_window[0],
            right=self.current_vis_window[1]
        )
        self.canvas.draw_idle()

    def update_ylim(self):
        self.canvas.axes.set_ylim(
            bottom=self.current_offset * (self.current_vis_ch[0] - 0.5),
            top=self.current_offset * (self.current_vis_ch[1] - 0.5),
        )
        self.canvas.draw_idle()

    def update_zoom_in_plots(self):
        """ This function updates the zoom without plotting again. """
        for plt_data in self.plotted_data:
            offset = np.arange(plt_data["n_cha"]) * self.current_offset
            for i, line in enumerate(plt_data["handle"]):
                line.set_ydata(self.zoom * plt_data["signal"][:, i] + offset[i])
        if self.scale_bar is not None:
            self.update_scale_bar()
        self.canvas.draw_idle()

    @staticmethod
    def __get_number_scientific(number):
        return "%.1e" % number if number > 100 else "%.2f" % number

    def update_scale_bar(self):
        if self.scale_bar is not None:
            self.scale_bar.remove()
        b_height = round(self.current_offset, 2)
        bl = "%s %s" % (self.__get_number_scientific(b_height / self.zoom),
                        self.units)
        v_bar = VerticalScaleBar(
                axes=self.canvas.axes,
                bar_height=b_height,
                bar_label=bl,
                label_size=PLOT_PARAMS['axes.labelsize']
        )
        self.scale_bar = self.canvas.axes.add_artist(v_bar.bar)

    def _get_max_timestamp(self):
        """ Returns the current max length of the signals in seconds. """
        max_times = 0
        for plt_data in self.plotted_data:
            if plt_data["times"][-1] > max_times:
                max_times = plt_data["times"][-1]
        return max_times

    def _get_min_timestamp(self):
        """ Returns the current max length of the signals in seconds. """
        min_times = np.inf
        for plt_data in self.plotted_data:
            if plt_data["times"][0] < min_times:
                min_times = plt_data["times"][0]
        return min_times

    def _get_max_channels(self):
        max_ch = 0
        for plt_data in self.plotted_data:
            if plt_data["n_cha"] > max_ch:
                max_ch = plt_data["n_cha"]
        return max_ch

    def _make_window_feasible(self, new_window):
        max_length_s = self._get_max_timestamp()
        l = self.current_vis_window[1] - self.current_vis_window[0]
        if new_window[1] > max_length_s:
            # Stick to the end
            new_window[0] = max_length_s - l
            new_window[1] = max_length_s
        if new_window[0] < 0:
            new_window = np.array([0, l])
        return new_window

    ## --------------------------- PLOT FUNCTIONS --------------------------- ##
    def check_consistency(self, n_cha):
        if self.plotted_data is not None:
            for plt_data in self.plotted_data:
                if plt_data["n_cha"] != n_cha:
                    raise ValueError("The number of channels is not the same "
                                     "as previous plots! Aborting...")

    @staticmethod
    def __check_conditions(c_dict, times):
        # Check errors
        if not isinstance(c_dict, dict):
            raise ValueError("'c_dict' must be a dict."
                             "Please, read carefully the time_plot documentation"
                             "to know how to define 'conditions_dict' properly. ")
        if not 'conditions' in c_dict.keys():
            raise ValueError("'c_dict' must have 'conditions' key."
                             "Please, read carefully the time_plot documentation"
                             "to know how to define 'conditions_dict' properly.")
        if not 'conditions_labels' in c_dict.keys():
            raise ValueError(
                "'c_dict' must have 'condition_labels' key."
                "Please, read carefully the time_plot documentation"
                "to know how to define 'conditions_dict' properly.")
        if not 'conditions_times' in c_dict.keys():
            raise ValueError(
                "'c_dict' must have 'condition_times' key."
                "Please, read carefully the time_plot documentation"
                "to know how to define 'conditions_dict' properly.")
        cond_times = np.array(c_dict["conditions_times"])
        if cond_times[0] < times[0]:
            raise ValueError("Incorrect format of condition_timestamps. The "
                             "values must be referenced to the beginning of "
                             "the record, so that the first timestamp has "
                             "value 0.")

    @staticmethod
    def __find_condition_name_by_label(c_dict, label):
        for condition in c_dict['conditions'].values():
            if condition['label'] == label:
                return condition['desc-name']

    @staticmethod
    def __check_events(e_dict, times):
        # Check errors
        if not isinstance(e_dict, dict):
            raise ValueError("'e_dict' must be a dict."
                             "Please, read carefully the time_plot documentation"
                             "to know how to define 'events_dict' properly. ")
        if not 'events' in e_dict.keys():
            raise ValueError("'e_dict' must have 'events' key."
                             "Please, read carefully the time_plot documentation"
                             "to know how to define 'events_dict' properly.")
        if not 'events_labels' in e_dict.keys():
            raise ValueError("'e_dict' must have 'event_labels' key."
                             "Please, read carefully the time_plot documentation"
                             "to know how to define 'events_dict' properly.")
        if not 'events_times' in e_dict.keys():
            raise ValueError("'e_dict' must have 'event_times' key."
                             "Please, read carefully the time_plot documentation"
                             "to know how to define 'events_dict' properly.")
        events_timestamps = np.array(e_dict['events_times'])
        if events_timestamps[0] < times[0]:
            raise ValueError("Incorrect format of events_timestamps. "
                             "The values must be referenced to the beginning "
                             "of the record, so that the first timestamp has value 0.")

    @staticmethod
    def __find_event_name_by_label(e_dict, label):
        for event in e_dict['events'].values():
            if event['label'] == label:
                return event['desc-name']

    def add_plot(self, signal, times, ch_labels=None, color=None,
                 conditions_dict=None, events_dict=None, reverse_channels=True,
                 start_from_zero=False, signal_label=None):
        """
            Adds a new plot to the current figure with the provided signal
            and optional annotations for conditions and events.

            Parameters:
            -----------
            signal : numpy.ndarray
                A 3D array representing the signal data to plot, with shape
                (blocks, samples_per_block, n_cha).
            times : array-like
                A 1D array containing the time points corresponding to the
                signal samples. The length of `times` should match the total
                number of samples (blocks * samples_per_block).
            ch_labels : list of str, optional
                A list of strings representing the labels for each channel.
                If not provided, channel labels will be automatically
                assigned as integers starting from 1.
            color : str or None, optional
                The color to use for plotting the signal. If None, a default
                color will be assigned based on the plot count.
            conditions_dict : dict, optional
                A dictionary defining the conditions to display on the plot. The
                dictionary must contain:
                - 'conditions_labels': a list of labels for each condition.
                - 'conditions_times': a list or array of start and end times for
                  each condition. These times should align with the `times`
                  array.
                Note that labels and times must have an even number of items,
                as both the start and end of each condition must be marked!
                - 'conditions': a dictionary containing the condition as
                keys. Each condition must have the keys 'desc_name', 'label'
                and 'shortcut'. An example of a 'conditions' dict would be:
                  {
                      'eyes-closed': {
                          'desc-name': 'Eyes closed',
                          'label': 1,
                          'shortcut': 'C'
                      }
                  }
            events_dict : dict, optional
                A dictionary defining the events to display on the plot. The
                dictionary must contain:
                - 'events_labels': a list of labels for each event.
                - 'events_times': a list or array of times when each event
                occurred. These times should align with the `times` array.
                - 'events': a dictionary containing the event types as keys.
                Each event type must have the keys 'desc_name', 'label',
                and 'shortcut'. An example of an `events` dictionary would be:
                  {
                      'blink': {
                          'desc_name': 'Blink',
                          'label': 0,
                          'marker': 'B'
                      }
                  }
            reverse_channels : bool, optional
                If True, channels are shown from top to bottom. If False,
                channels are shown from bottom to top.
            start_from_zero : bool, optional
                If True, adjusts the time axis such that it starts from zero.
                This will also modify the times in `conditions_dict` and
                `events_dict` so that the conditions and events remain
                correctly aligned with the signal data. By default, the time
                axis starts from the first value in `times`.
            signal_label : str, optional
                Signal name so it appears in the legend. If None, then the
                signal name will be the current signal count.
        """

        # Transform signal dimensions
        signal = check_dimensions(signal)
        blocks, samples_per_block, n_cha = signal.shape
        signal = signal.reshape(blocks * samples_per_block, n_cha)
        times = np.array(times)

        # Check if everything is correct respect other previous plots
        self.check_consistency(n_cha)

        # If start from zero is activated, all timestamps must be relative to
        # the init of the data
        if start_from_zero:
            init_t = times[0]
            times -= init_t
            if conditions_dict is not None:
                conditions_dict['conditions_times'] = \
                    np.array(conditions_dict['conditions_times']) - init_t
            if events_dict is not None:
                events_dict['events_times'] = \
                    np.array(events_dict['events_times']) - init_t

        # If there is no offset, set it one for the first signal
        if self.current_offset is None:
            self.current_offset = 4 * np.var(np.abs(signal.ravel()))

        # If the number of channels to show is None, initialize it
        if self.ch_to_show is None:
            self.ch_to_show = n_cha
            self.current_vis_ch = np.array([0, n_cha])

        # If there is no window length, set it one for the first signal
        if self.time_to_show is None:
            self.time_to_show = min(self.initial_window_s, times[-1] - times[0])
            self.current_vis_window = (np.array([0, self.time_to_show]) +
                                       times[0])
        offset = np.arange(n_cha) * self.current_offset

        # Plot conditions
        if conditions_dict is not None:
            self.__check_conditions(conditions_dict, times)
            c_labels = np.array(conditions_dict['conditions_labels'])
            c_labels_ = c_labels.reshape(int(len(c_labels) / 2), 2)
            c_times = np.array(conditions_dict['conditions_times'])
            c_times_ = c_times.reshape(int(len(c_labels) / 2), 2)
            cmap = plt.get_cmap('jet')(np.linspace(0, 1, c_times_.shape[0]))
            for i in range(c_times_.shape[0]):
                leg_label = self.__find_condition_name_by_label(
                    conditions_dict, c_labels_[i,0])
                patch_ = self.canvas.axes.fill_betweenx(
                    y=(-0.5 * self.current_offset,
                       self.current_offset * (n_cha - 0.5)),
                    x1=c_times_[i, 0], x2=c_times_[i, 1],
                    color=cmap[i], alpha=0.3, label=leg_label, zorder=1
                )
                if leg_label not in self.legend_items["conditions"]:
                    self.legend_items["conditions"][leg_label] = patch_

        # Plot signal
        if color is None:
            color = self.default_colors[self.plot_count]
        if reverse_channels:
            signal = signal[:, ::-1]
        plot_handle = self.canvas.axes.plot(
            times, self.zoom * signal + offset, color, zorder=2
        )
        if signal_label is None:
            signal_label = "Signal %i" % self.plot_count
        if signal_label not in self.legend_items["signals"]:
            self.legend_items["signals"][signal_label] = plot_handle[0]

        # Plot events
        if events_dict is not None:
            self.__check_events(events_dict, times)
            u_ev = np.unique(events_dict['events_labels'])
            cmap = plt.get_cmap('rainbow')(np.linspace(0, 1, len(u_ev)))
            colors = dict()
            for j, l in enumerate(u_ev):
                colors[l] = cmap[j]
            for i, et in enumerate(events_dict['events_times']):
                leg_label = self.__find_event_name_by_label(
                    events_dict, events_dict['events_labels'][i])
                line_ = self.canvas.axes.vlines(
                    x=et,
                    ymin=-0.5 * self.current_offset,
                    ymax=self.current_offset * (n_cha - 0.5),
                    linestyles="dashed",
                    label=leg_label,
                    colors=colors[events_dict['events_labels'][i]],
                    zorder=10
                )
                if leg_label not in self.legend_items["events"]:
                    self.legend_items["events"][leg_label] = line_

        # Legend
        l_handles_ = list()
        l_labels_ = list()
        for key, value in self.legend_items["signals"].items():
            l_handles_.append(value)
            l_labels_.append(key)
        for key, value in self.legend_items["conditions"].items():
            l_handles_.append(value)
            l_labels_.append(key)
        for key, value in self.legend_items["events"].items():
            l_handles_.append(value)
            l_labels_.append(key)
        self.canvas.axes.legend(
            handles=l_handles_, labels=l_labels_,
            loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.15),
            fancybox=True, shadow=True
        )

        # Channel labels
        if ch_labels is not None:
            if not isinstance(ch_labels, list):
                raise ValueError("Channel labels ('ch_labels') must be entered"
                                 "as a list.")
        else:
            ch_labels = ['%i' % i for i in range(1, n_cha + 1)]
        if reverse_channels:
            ch_labels = ch_labels[::-1]

        # Scale bar
        if self.scale_bar is None:
            self.update_scale_bar()

        # Ticks, labels, limits
        self.canvas.axes.set_xlabel('time (s)')
        self.canvas.axes.set_ylabel('channels')
        self.canvas.axes.set_yticks(ticks=offset)
        self.canvas.axes.set_yticklabels(labels=ch_labels)
        self.canvas.axes.set_ylim(
            bottom=self.current_offset * (self.current_vis_ch[0] - 0.5),
            top=self.current_offset * (self.current_vis_ch[1] - 0.5),
        )
        self.canvas.axes.set_xlim(
            left=self.current_vis_window[0],
            right=self.current_vis_window[1]
        )
        # self.canvas.figure.tight_layout()

        # Add data
        current_data = {
            "handle": plot_handle,
            "times": times,
            "signal": signal,
            "n_cha": n_cha,
            "color": color,
            "conditions_dict": conditions_dict
        }
        self.plotted_data.append(current_data)

        # Draw it
        self.canvas.draw_idle()

        # Update the preview canvas
        self.update_preview_plot(current_data)

        # Update counter
        self.plot_count += 1

    ## --------------------------- PREVIEW CANVAS --------------------------- ##

    def _initialize_visualizing_window(self, maxy):
        """ This function initializes the visualizing window by creating it
        and connecting the control bindings."""
        # Parameters
        cx, cy = self.current_vis_window
        barm = 0.2  # Percentage of empty limits (without bar)
        PREVIEW_COLOR_LIMS = (.7, .7, .7)
        PREVIEW_COLOR_LIMIN = 'k'
        PREVIEW_COLOR_FILL = (.7, .7, .7, .5)
        PREVIEW_COLOR_SIGNAL = '#4c72b0'
        PREVIEW_COLOR_BG = '#eaeaf2'

        # Create the visualizing window
        Z_TOP = 10
        self.preview_handles = dict()
        self.preview_handles["maxy"] = maxy
        self.preview_handles["winfill"] = self.canvas_preview.axes.fill(
            [cx, cx, cy, cy],
            [maxy, -maxy, -maxy, maxy],
            facecolor=PREVIEW_COLOR_FILL, edgecolor=PREVIEW_COLOR_LIMS,
            linewidth=2, capstyle='round'
        )
        self.preview_handles["lim_l"] = self.canvas_preview.axes.plot(
            [cx, cx],
            [maxy * (barm * 2 - 1), maxy * (1 - barm * 2)],
            color=PREVIEW_COLOR_LIMS, linewidth=7, solid_capstyle='round',
            zorder=Z_TOP
        )
        self.preview_handles["lim_li"] = self.canvas_preview.axes.plot(
            [cx, cx],
            [maxy * (barm * 2 - 1), maxy * (1 - barm * 2)],
            color=PREVIEW_COLOR_LIMIN, linewidth=1, solid_capstyle='round',
            zorder=Z_TOP
        )
        self.preview_handles["lim_r"] = self.canvas_preview.axes.plot(
            [cy, cy],
            [maxy * (barm * 2 - 1), maxy * (1 - barm * 2)],
            color=PREVIEW_COLOR_LIMS, linewidth=7, solid_capstyle='round',
            zorder=Z_TOP
        )
        self.preview_handles["lim_ri"] = self.canvas_preview.axes.plot(
            [cy, cy],
            [maxy * (barm * 2 - 1), maxy * (1 - barm * 2)],
            color=PREVIEW_COLOR_LIMIN, linewidth=1, solid_capstyle='round',
            zorder=Z_TOP
        )

        # Bindings
        self.canvas_preview.mpl_connect(
            'button_press_event', self.on_preview_mouse_press)
        self.canvas_preview.mpl_connect(
            'button_release_event', self.on_preview_mouse_release)
        self.canvas_preview.mpl_connect(
            'motion_notify_event', self.on_preview_motion)

    def update_preview_plot(self, current_data):
        """ This function updates the preview plot to show new signals. """
        # Plot a mean of the signal
        median_ = np.median(current_data["signal"], axis=1)
        self.canvas_preview.axes.plot(
            current_data["times"], median_,
            current_data["color"], zorder=2, alpha=0.5
        )

        # Initialize only if preview handles are still not set
        if self.preview_handles is None:
            maxy_ = np.max(np.abs(median_)) * 2.05
            self._initialize_visualizing_window(maxy_)

        # Conditions
        if current_data["conditions_dict"] is not None:
            c_times = np.array(current_data["conditions_dict"][
                                   "conditions_times"])
            c_times_ = c_times.reshape(int(len(c_times) / 2), 2)
            cmap = plt.get_cmap('jet')(np.linspace(0, 1, c_times_.shape[0]))
            for i in range(c_times_.shape[0]):
                self.canvas_preview.axes.fill_betweenx(
                    y=(-self.preview_handles["maxy"] * 1.05,
                       self.preview_handles["maxy"] * 1.05),
                    x1=c_times_[i, 0], x2=c_times_[i, 1],
                    color=cmap[i], alpha=0.3, zorder=1
                )

        # Other options
        max_length_s = self._get_max_timestamp()
        min_length_s = self._get_min_timestamp()
        self.canvas_preview.figure.patch.set_alpha(0)
        self.canvas_preview.figure.subplots_adjust(left=0.125, right=0.901,
                                                   top=0.99, bottom=0.07)
        self.canvas_preview.axes.grid(False)
        self.canvas_preview.axes.set_xlim(left=min_length_s, right=max_length_s)
        self.canvas_preview.axes.set_ylim(
            top=self.preview_handles["maxy"] * 1.05,
            bottom=-self.preview_handles["maxy"] * 1.05
        )
        self.canvas_preview.axes.patch.set_alpha(0)
        self.canvas_preview.axes.set_axis_off()

        # Draw it
        self.canvas_preview.draw_idle()

    def update_preview_window(self):
        """ This function updates the limits of the visualization window. """
        cx, cy = self.current_vis_window
        maxy = self.preview_handles["maxy"]
        self.preview_handles["lim_l"][0].set_xdata((cx, cx))
        self.preview_handles["lim_li"][0].set_xdata((cx, cx))
        self.preview_handles["lim_r"][0].set_xdata((cy, cy))
        self.preview_handles["lim_ri"][0].set_xdata((cy, cy))
        self.preview_handles["winfill"][0].set_xy(
            np.array([[cx, cx, cy, cy], [maxy, -maxy, -maxy, maxy]]).T
        )
        self.canvas_preview.draw_idle()

    def on_preview_mouse_press(self, event):
        """ This function detects a mouse press inside the preview plot.
        If it is over the preview window, it also calculates what is the
        element that has been clicked. """
        if event.inaxes is None:
            # Ignore clicks outside axes
            return event
        if event.button != 1:
            # Ignore clicks of buttons that are not the left one
            return event
        self._preview_curr_focused = self._detect_preview_focus_item(event)

    def on_preview_mouse_release(self, event):
        """ This function detects a mouse release inside the preview plot. """
        if event.button != 1:
            # Ignore clicks of buttons that are not the left one
            return event
        if self._preview_curr_focused is not None:
            self._preview_curr_focused = None
            self.update_xlim()
            QApplication.setOverrideCursor(QCursor(Qt.ArrowCursor))

    def on_preview_motion(self, event):
        """ This function detects motion inside the preview plot. """
        # todo: si abandonas muy rápido la gráfica no cambia bien el cursor al default
        if event.inaxes is None:
            # Ignore motion events outside axes
            QApplication.setOverrideCursor(QCursor(Qt.ArrowCursor))
            return event
        # Hovering cursor
        hover = self._detect_preview_focus_item(event)
        if (hover == HOVER_LEFT_PREVIEW_LIMIT or
                hover == HOVER_RIGHT_PREVIEW_LIMIT):
            QApplication.setOverrideCursor(QCursor(Qt.SizeHorCursor))
        elif hover == HOVER_CENTER_PREVIEW:
            QApplication.setOverrideCursor(QCursor(Qt.OpenHandCursor))
        else:
            QApplication.setOverrideCursor(QCursor(Qt.ArrowCursor))

        # Moving things
        if self._preview_curr_focused is None:
            # Ignore if the event is not placed after mouse_press
            return event
        if event.button != 1:
            # Ignore clicks of buttons that are not the left one
            return event
        if self._preview_curr_focused == HOVER_LEFT_PREVIEW_LIMIT:
            # Moving the left limit
            if event.xdata < self.current_vis_window[1] - self.preview_epsilon:
                self.current_vis_window[0] = event.xdata
            QApplication.setOverrideCursor(QCursor(Qt.SizeHorCursor))
        elif self._preview_curr_focused == HOVER_RIGHT_PREVIEW_LIMIT:
            # Moving the right limit
            if event.xdata > self.current_vis_window[0] + self.preview_epsilon:
                self.current_vis_window[1] = event.xdata
            QApplication.setOverrideCursor(QCursor(Qt.SizeHorCursor))
        elif self._preview_curr_focused == HOVER_CENTER_PREVIEW:
            # Moving all the window
            len_ = self.current_vis_window[1] - self.current_vis_window[0]
            self.current_vis_window = self._make_window_feasible(
                [event.xdata - len_/2, event.xdata + len_/2]
            )
            QApplication.setOverrideCursor(QCursor(Qt.ClosedHandCursor))
        self.update_preview_window()

    def _detect_preview_focus_item(self, event):
        """ This function detects if the click event is over something clickable. """
        # Convert the event position into coordinates inside graph
        t = self.canvas_preview.axes.transData.inverted()
        xy = t.transform([event.x, event.y])
        
        # Determine the focused item
        if np.abs(xy[1]) > self.preview_handles["maxy"] + self.preview_epsilon:
            return None
        if np.abs(xy[0] - self.current_vis_window[0]) < self.preview_epsilon:
            return HOVER_LEFT_PREVIEW_LIMIT
        elif np.abs(xy[0] - self.current_vis_window[1]) < self.preview_epsilon:
            return HOVER_RIGHT_PREVIEW_LIMIT
        elif ((xy[0] > self.current_vis_window[0]) & 
              (xy[0] < self.current_vis_window[1])):
            return HOVER_CENTER_PREVIEW
        return None


class TimePlotManager:

    def __init__(self):
        self.time_plot = None
        if not QApplication.instance():
            self.app = QApplication()
        else:
            self.app = QtWidgets.QApplication.instance()

    def set_time_plot(self, time_plot_window):
        self.time_plot = time_plot_window

    def show(self):
        self.time_plot.show()
        self.app.exec()


if __name__ == '__main__':
    import numpy as np

    signal_1 = 1.5 * np.random.randn(10000, 16) - 0.5
    signal_2 = np.random.randn(10000, 16) - 0.5
    times = np.linspace(0, 240, 10000)
    ch_labels = [
        'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4',
        'P3', 'P4', 'O1', 'O2', 'F7', 'F8',
        'T3', 'T4', 'T5', 'T6'
    ]
    conditions = {
        'eyes-closed':
            {'desc-name': 'Eyes closed', 'label': 1,'shortcut': 'C'},
        'eyes-open':
            {'desc-name': 'Eyes open', 'label': 0,'shortcut': 'O'}
    }
    conditions_labels = [0, 0, 1, 1]
    conditions_times = [10, 60, 70, 200]
    events = {
        'blink': {'desc-name': 'Blink', 'label': 0, 'shortcut': 'B'}
    }
    events_labels = [0, 0, 0, 0, 0]
    events_times = [12, 62, 72, 202]
    c_dict = {'conditions': conditions,
              'conditions_labels': conditions_labels,
              'conditions_times': conditions_times}
    e_dict = {'events': events,
              'events_labels': events_labels,
              'events_times': events_times}

    # Create time plot manager for visualization
    time_plot_manager = TimePlotManager()

    # Create time plot
    time_plot = TimePlot(ch_to_show=10, units="μV")
    time_plot_manager.set_time_plot(time_plot)

    # Add plot
    time_plot.add_plot(
        signal=signal_1,
        times=times,
        conditions_dict=c_dict,
        events_dict=e_dict,
        start_from_zero=True,
        ch_labels=ch_labels,
        signal_label='First EEG'
    )
    time_plot_manager.show()

    # Add another plot
    time_plot.add_plot(
        signal=signal_2,
        times=times,
        start_from_zero=True,
        ch_labels=ch_labels,
        signal_label='Second EEG'
    )
    time_plot_manager.show()
