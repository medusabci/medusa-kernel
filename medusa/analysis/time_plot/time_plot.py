# External imports
import os
from abc import ABC, abstractmethod
from sys import orig_argv

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
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FuncFormatter

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

    def __init__(self, n_cha, cha_labels=None, cha_to_show=None,
                 reverse_channels=True, vis_step_s=2, initial_window_s=10,
                 zoom_multiplier=1.2, initial_zoom=1, style_params=None):
        """
        Initializes the TimePlot window with the specified parameters for
        visualizing multi-channel time-series data.

        Shortcuts
        ----------
        - Left / Right arrows: Move the visible time window backward or forward.
        - Up / Down arrows: Increase or decrease the number of visible channels.
        - Mouse scroll: Zoom in or out.

        Parameters
        ----------
        n_cha : int
            Total number of channels in the data. This determines how many
            signal traces can be displayed in the plot and is used to validate
            display limits and layout.

        cha_labels : list of str, optional
            List of channel labels (e.g., ["Fz", "Cz", "Pz", ...]) to display
            on the y-axis or alongside each trace. If None, default channel
            indexes will be used as labels. The length of this list must
            match `n_cha`.

        cha_to_show : int or None, optional
            Number of channels to display in the plot at once. This
            value must be less than or equal to the total number of
            channels available in the data. If set to None, the number of
            channels to show is initialized to the total number of available
            channels.

        reverse_channels : bool, optional
            If True, channels are shown from top to bottom. If False,
            channels are shown from bottom to top.

        vis_step_s : int, optional
            Time step (in seconds) used when navigating the plot, such as
            fast forwarding or rewinding. The default value is 2 seconds,
            meaning that each step moves the visible window forward or
            backward by 2 seconds.

        initial_window_s : float, optional
            Initial visualization window span in seconds. The default
            value is 10.0 s.

        zoom_multiplier : float, optional
            Factor to apply when zooming in or out using scroll or shortcuts.

        initial_zoom : float, optional
            Initial zoom level. 1.0 means no zoom.

        style_params : dict, optional
            Dictionary of optional style customization parameters.

            **General Styling**
            -------------------
            - 'background_color' : str
                Background color of the plot canvas. Default is 'white'.
            - 'font_size' : int
                Base font size for labels and ticks. Default is 10.
            - 'tick_fontsize' : int
                Font size for tick labels. Default is 9.
            - 'label_fontsize' : int
                Font size for axis labels. Default is 10.

            **Legend Styling**
            ------------------
            - 'legend_fontsize' : int
                Font size used for legend entries. Default is 8.
            - 'legend_ncol' : int
                Number of columns in the legend. Default is 5.
            - 'legend_loc' : str
                Location of the legend box (e.g., 'upper center', 'lower left').
                Default is 'upper center'.
            - 'legend_fancybox' : bool
                Whether the legend box has rounded corners. Default is True.
            - 'legend_shadow' : bool
                Whether to draw a drop shadow under the legend. Default is True.

            These style parameters affect both the main plot canvas and the
            preview canvas, and can be used to personalize the appearance of
            the visualization.
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

        # Default channel labels
        if cha_labels is None:
            cha_labels = [str(i) for i in range(n_cha)]

        # General style params
        style_params = style_params or {}
        style_params.setdefault("background_color", None)
        style_params.setdefault("font_size", 6)
        style_params.setdefault("legend_fontsize", 6)
        style_params.setdefault("tick_fontsize", 6)
        style_params.setdefault("label_fontsize", 6)
        style_params.setdefault("legend_fontsize", 8)
        style_params.setdefault("legend_ncol", 5)
        style_params.setdefault("legend_loc", "upper center")
        style_params.setdefault("legend_fancybox", True)
        style_params.setdefault("legend_shadow", True)

        # Default colors
        plt.style.use('seaborn-v0_8')
        for key, value in PLOT_PARAMS.items():
            mpl.rcParams[key] = value
        self.default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        # Checks
        assert n_cha > 0, "Number of channels must be greater than 0."
        assert len(cha_labels) == n_cha, \
            "Number of channel labels must be equal to the number of channels"

        # Parameters
        self.n_cha = n_cha
        self.cha_labels = cha_labels
        self.cha_to_show = cha_to_show
        self.reverse_channels = reverse_channels
        self.vis_step_s = vis_step_s
        self.initial_window_s = initial_window_s
        self.zoom_multiplier = zoom_multiplier
        self.initial_zoom = initial_zoom
        self.zoom = initial_zoom
        self.style_params = style_params

        # Buttons and icons
        icon_map = {
            self.btn_ff: 'ff.png',
            self.btn_rr: 'rr.png',
            self.btn_zoomin: 'zoom_in.png',
            self.btn_zoomout: 'zoom_out.png',
            self.btn_download: 'download.png',
            self.btn_ch_up: 'increase_channels.png',
            self.btn_ch_down: 'decrease_channels.png',
        }
        for btn, icon in icon_map.items():
            btn.setIcon(QIcon(os.path.join(self.dir, f'icons/{icon}')))
        self.btn_ff.clicked.connect(self.fast_forward)
        self.btn_rr.clicked.connect(self.rewind)
        self.btn_zoomin.clicked.connect(self.zoom_in)
        self.btn_zoomout.clicked.connect(self.zoom_out)
        self.btn_download.clicked.connect(self.download)
        self.btn_ch_up.clicked.connect(self.increase_channels)
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

        # Main canvas style)
        self.canvas.axes.set_yticks([])
        self.canvas.axes.set_yticklabels([])

        # Preview canvas style
        self.canvas_preview.axes.set_xlabel(
            "time (s)", fontsize=self.style_params["label_fontsize"])
        self.canvas_preview.axes.set_ylabel(
            "channels", fontsize=self.style_params["label_fontsize"])

        # Plot handles
        self.plotted_data = list()
        self.plotted_conditions = list()
        self.plotted_events = list()
        self.preview_handles = None

        # Initialization
        self.time_to_show = None
        self.current_offset = None
        self.current_vis_window = np.array([0, None])
        self.current_vis_ch = np.array([0, cha_to_show]) if not (
            reverse_channels) else np.array([n_cha - cha_to_show, n_cha])
        self._preview_curr_focused = None
        self.preview_epsilon = 2      # Hardcoded precision to click the window
        self.plot_count = 0
        self.legend_items = {
            "data": dict(),
            "events": dict(),
            "conditions": dict()
        }

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

    @abstractmethod
    def zoom_in(self):
        """ This method makes the time plot to zoom in. """
        pass

    @abstractmethod
    def zoom_out(self):
        """ This method makes the time plot to zoom out. """
        pass

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
        self.current_vis_ch += self.cha_to_show
        if self.current_vis_ch[1] >= self.n_cha:
            self.current_vis_ch[0] = self.n_cha - self.cha_to_show
            self.current_vis_ch[1] = self.n_cha
        self.update_ylim()
        self.canvas.draw_idle()

    def decrease_channels(self):
        self.current_vis_ch -= self.cha_to_show
        if self.current_vis_ch[0] < 0:
            self.current_vis_ch = np.array([0, self.cha_to_show])
        self.update_ylim()
        self.canvas.draw_idle()

    ## --------------------------- PLOT UPDATES --------------------------- ##
    def update_xlim(self):
        """ This method updates the X limits. """
        self.canvas.axes.set_xlim(
            left=self.current_vis_window[0],
            right=self.current_vis_window[1]
        )
        self.canvas.draw_idle()

    @abstractmethod
    def update_ylim(self):
        pass

    @staticmethod
    def get_number_scientific(number):
        return "%.1e" % number if number > 100 else "%.2f" % number

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

    def update_legend(self):
        """
        Adds a separate legend for each category (data, conditions, events),
        arranged horizontally with one column per category.
        """
        categories = ["data", "conditions", "events"]

        # Number of legends what will be displayed
        active_entries = [(cat, list(self.legend_items[cat].items()))
                          for cat in categories if self.legend_items[cat]]
        n_visible = len(active_entries)
        if n_visible == 0:
            return

        # Safely remove any previously added legends
        for artist in self.canvas.axes.artists:
            if isinstance(artist, mpl.legend.Legend):
                artist.remove()

        for i, category in enumerate(categories):

            items = self.legend_items.get(category, {})

            if not items:
                continue

            labels, handles = zip(*items.items())

            # Position: evenly spaced in normalized figure coordinates
            position = (i + 0.5) / n_visible

            leg = self.canvas.axes.legend(
                handles=handles,
                labels=labels,
                loc=self.style_params.get("legend_loc", "upper center"),
                bbox_to_anchor=(position, 1.1),
                ncol=1,
                fontsize=self.style_params.get("legend_fontsize", 6),
                fancybox=self.style_params.get("legend_fancybox", False),
                shadow=self.style_params.get("legend_shadow", False),
                frameon=self.style_params.get("legend_frameon", False),
            )

            # Important: add the legend manually
            self.canvas.axes.add_artist(leg)

        self.canvas.draw_idle()

    ## --------------------------- FORMAT CHECK --------------------------- #
    def validate_conditions_events_dict(self, signal_times,
                                        conditions_dict: dict = None,
                                        events_dict: dict = None) -> None:
        """
        Validates the structure, types, and temporal feasibility of
        `conditions_dict` and `events_dict` with respect to the signal time
        vector.

        Parameters
        ----------
        signal_times : array-like
            1D array or list of time values representing the time axis of the
            signal. All condition and event times must lie within the range
            defined by [min(signal_times), max(signal_times)].

        conditions_dict : dict, optional
            Must contain:
                - 'conditions': dict
                    Each key is a condition name, with value being a dict
                    containing:
                        - 'desc-name': str
                        - 'label': int
                - 'conditions_labels': list of int
                - 'conditions_times': list of float or int (same length as
                    labels, even number of entries)

        events_dict : dict, optional
            Must contain:
                - 'events': dict
                    Each key is an event name, with value being a dict
                    containing:
                        - 'desc-name': str
                        - 'label': int
                - 'events_labels': list of int
                - 'events_times': list of float or int (same length as labels)

        Raises
        ------
        ValueError
            If any structure, type, or time range check fails.
        """
        def check_structure(name, data_dict, expected_keys):
            if not isinstance(data_dict, dict):
                raise ValueError(f"'{name}' must be a dictionary.")
            for key in expected_keys:
                if key not in data_dict:
                    raise ValueError(f"'{name}' is missing required key: "
                                     f"'{key}'.")

        def check_entry_dict(name, entries):
            if not isinstance(entries, dict):
                raise ValueError(f"'{name}' must be a dictionary.")
            for key, val in entries.items():
                if not isinstance(val, dict):
                    raise ValueError(f"Entry '{key}' in '{name}' must be a "
                                     f"dictionary.")
                for required_key in ['desc-name', 'label']:
                    if required_key not in val:
                        raise ValueError(f"Entry '{key}' in '{name}' is "
                                         f"missing key: '{required_key}'.")

        def check_times_are_increasing(times, name):
            for i in range(len(times) - 1):
                if times[i] >= times[i + 1]:
                    raise ValueError(
                        f"'{name}' timestamps must be strictly increasing. "
                        f"Problem at index {i}: {times[i]} >= {times[i + 1]}.")

        def check_conditions(labels, times, signal_min, signal_max):
            if (not isinstance(labels, (list, np.ndarray)) or
                    not np.issubdtype(np.array(labels).dtype, np.integer)):
                raise ValueError("'conditions_labels' must be a list of "
                                 "integers.")
            if (not isinstance(times, (list, np.ndarray)) or
                    not np.issubdtype(np.array(times).dtype, np.number)):
                raise ValueError("'conditions_times' must be a list of numbers "
                                 "(int or float).")
            if len(labels) != len(times):
                raise ValueError("'conditions_labels' and 'conditions_times' "
                                 "must have the same length.")
            if len(times) % 2 != 0:
                raise ValueError("The length of 'conditions_times' must be "
                                 "even (start/end pairs).")
            for i in range(0, len(labels), 2):
                if labels[i] != labels[i + 1]:
                    raise ValueError(f"Condition labels at index {i} and {i+1} "
                                     f"must be equal (start/end pair).")
            for t in times:
                if not (signal_min <= t <= signal_max):
                    raise ValueError(f"Condition time {t} is out of signal "
                                     f"bounds ({signal_min}, {signal_max}).")
            check_times_are_increasing(times, 'conditions_times')

        def check_events(labels, times, signal_min, signal_max):
            if (not isinstance(labels, (list, np.ndarray)) or
                    not np.issubdtype(np.array(labels).dtype, np.integer)):
                raise ValueError("'events_labels' must be a list of integers.")
            if (not isinstance(times, (list, np.ndarray)) or
                    not np.issubdtype(np.array(times).dtype, np.number)):
                raise ValueError("'events_times' must be a list of numbers "
                                 "(int or float).")
            if len(labels) != len(times):
                raise ValueError("'events_labels' and 'events_times' must have "
                                 "the same length.")
            for t in times:
                if not (signal_min <= t <= signal_max):
                    raise ValueError(f"Event time {t} is out of signal bounds "
                                     f"({signal_min}, {signal_max}).")
            check_times_are_increasing(times, 'events_times')

        # Convert signal timestamps to array for range checks
        signal_timestamps = np.asarray(signal_times)
        if signal_timestamps.ndim != 1:
            raise ValueError("signal_timestamps must be a 1D array or list.")
        signal_min, signal_max = signal_timestamps.min(), signal_timestamps.max()

        if conditions_dict is not None:
            check_structure(
                "conditions_dict", conditions_dict,
                ['conditions', 'conditions_labels', 'conditions_times'])
            check_entry_dict("conditions", conditions_dict['conditions'])
            check_conditions(conditions_dict['conditions_labels'],
                             conditions_dict['conditions_times'],
                             signal_min, signal_max)

        if events_dict is not None:
            check_structure(
                "events_dict", events_dict,
                ['events', 'events_labels', 'events_times'])
            check_entry_dict("events", events_dict['events'])
            check_events(events_dict['events_labels'],
                         events_dict['events_times'],
                         signal_min, signal_max)

    def normalize_conditions_labels_times(self, signal_times, conditions_dict):
        """
        Normalize mixed-format condition labels and times into explicit
        start/end pairs.

        Supports:
            - Minimal format: [0, 1, 2], times: [0.5, 1.5, 3.0]
            - Full format:    [0, 0, 1, 1], times: [0.5, 1.5, 2, 3]
            - Mixed format:   [0, 1, 1, 2], times: [0.5, 1.5, 2, 3]

        Parameters
        ----------
        signal_times : array-like
            1D array or list of time values representing the time axis of the
            signal. All condition times must lie within the range defined by
            [min(signal_times), max(signal_times)].
        labels : list of int
            Condition labels (possibly unpaired or mixed).
        times : list of float
            Time values corresponding to the labels.

        Returns
        -------
        norm_labels : list of int
            Normalized list of labels with explicit start/end pairs.
        norm_times : list of float
            Normalized list of times with alternating start/end points.
        """
        # Get data form dict
        labels = conditions_dict['conditions_labels']
        times = conditions_dict['conditions_times']
        # Checks
        if len(labels) != len(times):
            raise ValueError("labels and times must have the same length.")
        if len(labels) < 1:
            raise ValueError(
                "labels and times must contain at least one entry.")
        # Init
        norm_labels = []
        norm_times = []
        # Normalize
        i = 0
        while i < len(labels):
            curr_label = labels[i]
            curr_time = times[i]

            # Case 1: explicit start-end pair
            if (i + 1 < len(labels)) and (labels[i] == labels[i + 1]):
                next_time = times[i + 1]
                norm_labels.extend([curr_label, curr_label])
                norm_times.extend([curr_time, next_time])
                i += 2
            # Case 2: transition or single label
            elif (i + 1 < len(labels)):
                next_time = times[i + 1]
                norm_labels.extend([curr_label, curr_label])
                norm_times.extend([curr_time, next_time])
                i += 1
            # Case 3: last label, no end — close with signal_end
            else:
                norm_labels.extend([curr_label, curr_label])
                norm_times.extend([curr_time, signal_times[-1]])
                i += 1
        # Update dict
        conditions_dict['conditions_labels'] = np.asarray(norm_labels).astype(int)
        conditions_dict['conditions_times'] = np.asarray(norm_times)
        return conditions_dict

    ## --------------------------- PLOT FUNCTIONS --------------------------- ##
    def __find_cond_event_name_by_label(self, d, label, dict_type):
        """
        Find the descriptive name associated with a given label from either a
        conditions or events dictionary.

        Parameters
        ----------
        d : dict
            The dictionary containing either 'conditions' or 'events' as a key.
        label : int
            The label to search for.
        dict_type : str
            Either 'conditions' or 'events'.

        Returns
        -------
        str or None
            The corresponding 'desc-name' if found, otherwise None.
        """
        if dict_type not in d:
            raise ValueError(
                f"Expected key '{dict_type}' not found in dictionary.")

        for item in d[dict_type].values():
            if item['label'] == label:
                return item.get('desc-name', None)
        return None

    def add_conditions(self, signal_times, time_ref, conditions_dict):
        """
        Plots shaded regions on the time axis to indicate condition intervals
        (e.g., tasks, states, or experimental conditions) over the signal.

        Parameters
        ----------
        time_ref : float
            Reference time (in seconds) used to align condition intervals.
            All condition timestamps will be shifted by subtracting `time_ref`.
            This is useful when comparing signals with different starting times
            or when aligning multiple plots to a shared reference (e.g.,
            stimulus onset). In most cases , times are assumed to be relative
            to the signal start.

        conditions_dict : dict, optional
            Dictionary specifying the conditions to display on the plot.
            Must include the following keys:

            - 'conditions_labels' : list of int
                A list of integer labels identifying each condition interval.
                Must have the same length as `conditions_times`, and this
                length must be even, since each condition must be defined by
                a start and end time.

            - 'conditions_times' : list or array-like of float
                A list of timestamps (in seconds) marking the start and end of
                each condition interval. Timestamps are paired: [start1,
                end1, start2, end2, ...].

            - 'conditions' : dict
                A dictionary mapping condition names (e.g., 'eyes-closed') to
                their metadata. Each entry must contain:
                    - 'desc-name' : str — human-readable name of the condition
                    - 'label' : int — numeric identifier used in `
                        conditions_labels`

        Example
        -------
        conditions_dict = {
            'conditions': {
                'eyes-closed': {
                    'desc-name': 'Eyes closed',
                    'label': 0,
                },
                'eyes-open': {
                    'desc-name': 'Eyes open',
                    'label': 1,
                }
            },
            'conditions_labels': [0, 0, 1, 1],
            'conditions_times': [0.0, 60.0, 60.0, 120.0]
        }
        """
        # Copy dict
        conditions_dict = conditions_dict.copy()
        # Normalize conditions
        conditions_dict = self.normalize_conditions_labels_times(
            signal_times,
            conditions_dict)
        # Check format
        self.validate_conditions_events_dict(
            signal_times=signal_times,
            conditions_dict=conditions_dict)
        # Apply time ref
        if time_ref is not None:
            conditions_dict['conditions_times'] = (
                    conditions_dict['conditions_times'] - time_ref)
        # Get data
        c_labels = conditions_dict['conditions_labels']
        c_times = conditions_dict['conditions_times']
        # Reshape labels and times
        c_labels_ = c_labels.reshape(int(len(c_labels) / 2), 2)
        c_times_ = c_times.reshape(int(len(c_labels) / 2), 2)
        # Build consistent color mapping for condition labels
        unique_condition_labels = np.unique(c_labels)
        label_colors = dict()
        cmap = (plt.get_cmap('jet')(np.linspace(
            0, 1, len(unique_condition_labels))))
        for j, lbl in enumerate(unique_condition_labels):
            label_colors[lbl] = cmap[j]
        # Plot shaded areas
        for i in range(c_times_.shape[0]):
            # Get condition
            cond_label_id = c_labels_[i, 0]
            desc_name = self.__find_cond_event_name_by_label(
                conditions_dict, cond_label_id, 'conditions')
            # Get line properties if this event has already been added
            if desc_name in self.legend_items["conditions"]:
                patch_ = self.legend_items["conditions"][desc_name]
                cond_color = patch_.get_facecolor()[0]
            else:
                cond_color = label_colors[cond_label_id]
            # Draw condition
            patch_ = self.canvas.axes.fill_betweenx(
                y=(-self.current_offset,
                   self.current_offset * self.n_cha),
                x1=c_times_[i, 0], x2=c_times_[i, 1],
                color=cond_color,
                alpha=0.3,
                label=desc_name,
                zorder=2
            )
            if desc_name not in self.legend_items["conditions"]:
                self.legend_items["conditions"][desc_name] = patch_
        self.plotted_conditions.append(conditions_dict)
        # Update the preview canvas
        self.update_preview_plot()
        self.update_legend()

    def add_events(self, signal_times, time_ref, events_dict):
        """
        Plots vertical dashed lines on the time axis to mark discrete events
        (e.g., stimuli, user actions, triggers) across a multi-channel
        time-series plot.

        Parameters
        ----------
        time_ref : float
            Reference time (in seconds) used to align condition intervals.
            All condition timestamps will be shifted by subtracting `time_ref`.
            This is useful when comparing signals with different starting times
            or when aligning multiple plots to a shared reference (e.g.,
            stimulus onset). In most cases , times are assumed to be relative
            to the signal start.

        events_dict : dict, optional
            Dictionary specifying the events to display on the plot.
            Must include the following keys:

            - 'events_labels' : list of int
                A list of integer labels corresponding to each event instance.
                Must have the same length as `events_times`.

            - 'events_times' : list or array-like of float
                A list of timestamps (in seconds) indicating when each event
                occurred. Should be in strictly increasing order and fall
                within the signal time range.

            - 'events' : dict
                A dictionary mapping event names (e.g., 'blink') to their
                metadata. Each entry must contain:
                    - 'desc-name' : str — human-readable name of the event
                    - 'label' : int — numeric identifier used in `events_labels`

        Example
        -------
        events_dict = {
            'events': {
                'blink': {
                    'desc-name': 'Blink',
                    'label': 0,
                    'shortcut': 'B'
                },
                'keypress': {
                    'desc-name': 'Key Press',
                    'label': 1,
                    'shortcut': 'K'
                }
            },
            'events_labels': [0, 1, 0, 0],
            'events_times': [1.2, 2.0, 3.5, 4.0]
        }
        """
        # Copy dict
        events_dict = events_dict.copy()
        # Checks
        self.validate_conditions_events_dict(
            signal_times=signal_times,
            events_dict=events_dict)
        # Apply time ref
        if time_ref is not None:
            events_dict['events_times'] = events_dict['events_times'] - time_ref
        # Get params
        e_labels = events_dict['events_labels']
        e_times = events_dict['events_times']
        # Build consistent color mapping for events labels
        unique_events_labels = np.unique(e_labels)
        event_colors = dict()
        cmap = plt.get_cmap('rainbow')(np.linspace(
            0, 1, len(unique_events_labels)))
        for j, l in enumerate(unique_events_labels):
            event_colors[l] = cmap[j]
        # Plot event lines
        for i, event_time in enumerate(e_times):
            # Get event
            event_label_id = e_labels[i]
            desc_name = self.__find_cond_event_name_by_label(
                events_dict, event_label_id, 'events')
            # Get line properties if this event has already been added
            if desc_name in self.legend_items["events"]:
                line = self.legend_items["events"][desc_name]
                # Extract line style from existing line
                event_color = line.get_color()
            else:
                event_color = event_colors[events_dict['events_labels'][i]]
            # Draw new event
            line_ = self.canvas.axes.vlines(
                x=event_time,
                ymin=-self.current_offset,
                ymax= self.n_cha * self.current_offset,
                linestyles="dashed",
                label=desc_name,
                colors=event_color,
                zorder=3
            )
            if desc_name not in self.legend_items["events"]:
                self.legend_items["events"][desc_name] = line_
        self.plotted_events.append(events_dict)
        # Update the preview canvas
        self.update_preview_plot()
        self.update_legend()

    @abstractmethod
    def add_data(self, times, data, cha_idx=None, data_label=None,
                 time_ref=None, conditions_dict=None, events_dict=None,
                 style_params=None):
        """
        Adds a new time-series to the current plot,  with optional
        customization for color, labels, and time alignment.

        Parameters
        ----------
        times : array-like
            A 1D array containing the time points (in seconds) corresponding
            to the signal samples. The length of `times` should match the total
            number of samples.
        data : numpy.ndarray
            A 2D or 3D array representing the data to plot, with shape
            (segments, samples_per_segment, n_cha).
        cha_idx: array-like, optional
            A 1D array containing the indices of the channels of the plot
            that correspond to the channels in the signal. If None, signal
            channels must match the channels of the plot.
        data_label : str, optional
            Signal name so it appears in the legend. If None, then the
            signal name will be the current signal count.
        time_ref : float, None
            Reference time (in seconds) used to align condition intervals.
            All condition timestamps will be shifted by subtracting `time_ref`.
            This is useful when comparing signals with different starting times
            or when aligning multiple plots to a shared reference (e.g.,
            stimulus onset). By default, times are assumed to be relative
            to last timestamp in the plot.
        style_params : dict, optional
            Dictionary containing custom style settings for the plotted data.
        """
        pass

    ## --------------------------- PREVIEW CANVAS --------------------------- ##
    @abstractmethod
    def _initialize_visualizing_window(self, miny, maxy):
        """ This function initializes the visualizing window by creating it
        and connecting the control bindings."""
        # Parameters
        cx, cy = self.current_vis_window

        # barm = 0.2  # Percentage of empty limits (without bar)

        center = (miny + maxy) / 2
        bar_height = (maxy - miny) * 0.6
        half_height = bar_height / 2

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
            [miny, maxy, maxy, miny],
            facecolor=PREVIEW_COLOR_FILL, edgecolor=PREVIEW_COLOR_LIMS,
            linewidth=2, capstyle='round'
        )
        self.preview_handles["lim_l"] = self.canvas_preview.axes.plot(
            [cx, cx],
            [center - half_height, center + half_height],
            color=PREVIEW_COLOR_LIMS, linewidth=7, solid_capstyle='round',
            zorder=Z_TOP
        )
        self.preview_handles["lim_li"] = self.canvas_preview.axes.plot(
            [cx, cx],
            [center - half_height, center + half_height],
            color=PREVIEW_COLOR_LIMIN, linewidth=1, solid_capstyle='round',
            zorder=Z_TOP
        )
        self.preview_handles["lim_r"] = self.canvas_preview.axes.plot(
            [cy, cy],
            [center - half_height, center + half_height],
            color=PREVIEW_COLOR_LIMS, linewidth=7, solid_capstyle='round',
            zorder=Z_TOP
        )
        self.preview_handles["lim_ri"] = self.canvas_preview.axes.plot(
            [cy, cy],
            [center - half_height, center + half_height],
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

    @abstractmethod
    def update_preview_plot(self):
        """
        Updates the overview (preview) plot shown above the main time plot.

        It uses the data stored in self.plotted_data to generate a compact
        representation of all currently plotted signals, including both time-series
        signals and heatmap-like images (e.g., spectrograms). It also overlays
        a shaded rectangle to indicate the currently zoomed-in region of the
        main plot.
        """
        pass

    def update_preview_conds_and_events(self):
        # CONDITIONS
        for cond_dict in self.plotted_conditions:
            c_times = np.array(cond_dict["conditions_times"])
            c_times_ = c_times.reshape(int(len(c_times) / 2), 2)
            c_labels = np.array(cond_dict["conditions_labels"])
            c_labels_ = c_labels.reshape(int(len(c_labels) / 2), 2)
            for i in range(len(c_labels_)):
                # Get condition
                cond_label_id = c_labels_[i][0]
                desc_name = self.__find_cond_event_name_by_label(
                    cond_dict, cond_label_id, 'conditions')
                # Get patch properties if this condition has already been added
                patch = self.legend_items["conditions"][desc_name]
                cond_color = patch.get_facecolor()[0]
                self.canvas_preview.axes.fill_betweenx(
                    y=(-self.preview_handles["maxy"] * 1.05,
                       self.preview_handles["maxy"] * 1.05),
                    x1=c_times_[i, 0],
                    x2=c_times_[i, 1],
                    color=cond_color,
                    alpha=0.3,
                    zorder=1)
        # EVENTS
        for event_dict in self.plotted_events:
            e_times = np.array(event_dict["events_times"])
            e_labels = np.array(event_dict["events_labels"])
            for i in range(e_times.shape[0]):
                # Get event
                event_label_id = e_labels[i]
                desc_name = self.__find_cond_event_name_by_label(
                    event_dict, event_label_id, 'events')
                # Get line properties if this event has already been added
                line = self.legend_items["events"][desc_name]
                event_color = line.get_color()
                # Draw line
                self.canvas_preview.axes.vlines(
                    x=e_times[i],
                    ymin=-self.preview_handles["maxy"] * 1.05,
                    ymax=self.preview_handles["maxy"] * 1.05,
                    linewidth=1,
                    linestyles="dashed",
                    color=event_color,
                    zorder=3)

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
        # todo: si abandonas muy rápido la gráfica no cambia bien el cursor al
        #  default
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


class TimeSeriesPlot(TimePlot):

    def __init__(self, n_cha, cha_labels=None, cha_to_show=None,
                 reverse_channels=True, vis_step_s=2, initial_window_s=10,
                 zoom_multiplier=1.2, initial_zoom=1, style_params=None):
        """
        Initializes the TimePlot window with the specified parameters for
        visualizing multi-channel time-series data.

        Shortcuts
        ----------
        - Left / Right arrows: Move the visible time window backward or forward.
        - Up / Down arrows: Increase or decrease the number of visible channels.
        - Mouse scroll: Zoom in or out.

        Parameters:
        -----------
        n_cha : int
            Total number of channels in the data. This determines how many
            signal traces can be displayed in the plot and is used to validate
            display limits and layout.
        ch_labels : list of str, optional
            List of channel labels (e.g., ["Fz", "Cz", "Pz", ...]) to display
            on the y-axis or alongside each trace. If None, default channel
            indexes will be used as labels. The length of this list must
            match `n_cha`.
        ch_to_show : int or None, optional
            Number of channels to display in the plot at once. This
            value must be less than or equal to the total number of
            channels available in the data. If set to None, the number of
            channels to show is initialized to the total number of available
            channels.
        reverse_channels : bool, optional
            If True, channels are shown from top to bottom. If False,
            channels are shown from bottom to top.
        zoom_multiplier : float, optional
            Factor by which the plot zooms in or out when the zoom
            buttons are used. The default value is 1.2, meaning each zoom
            action will increase or decrease the visible range by 20%.
        vis_step_s : int, optional
            Time step (in seconds) used when navigating the plot, such as
            fast forwarding or rewinding. The default value is 2 seconds,
            meaning that each step moves the visible window forward or
            backward by 2 seconds.
        initial_zoom : int, optional
            Initial zoom. The default value is 1.
        initial_window_s : float, optional
            Initial visualization window span in seconds. The default
            value is 10.0 s.
        """
        # Load the custom layout from UIC
        super().__init__(
            n_cha, cha_labels=cha_labels,
            cha_to_show=cha_to_show,
            reverse_channels=reverse_channels,
            vis_step_s=vis_step_s,
            initial_window_s=initial_window_s,
            zoom_multiplier=zoom_multiplier,
            initial_zoom=initial_zoom,
            style_params=style_params)

        # Change default window settings
        self.setWindowTitle("Time Series Plot")

        # Parameters
        self.scale_bar = None
        self.style_params.setdefault("units", 'uV')

    ## ----------------------------- CONTROLS ----------------------------- ##
    def zoom_in(self):
        """ This method makes the time plot to zoom in. """
        self.zoom *= self.zoom_multiplier
        self.update_zoom_in_plots()

    def zoom_out(self):
        """ This method makes the time plot to zoom out. """
        self.zoom /= self.zoom_multiplier
        self.update_zoom_in_plots()

    ## --------------------------- PLOT UPDATES --------------------------- ##
    def set_y_axis_ticks(self):
        cha_labels = self.cha_labels
        if self.reverse_channels:
            cha_labels = cha_labels[::-1]

        # Ticks, labels, limits
        offset = np.arange(self.n_cha) * self.current_offset
        self.canvas.axes.set_yticks(ticks=offset)
        self.canvas.axes.set_yticklabels(labels=cha_labels)

    def update_ylim(self):
        y_lim = (self.current_offset * (self.current_vis_ch[0] - 0.5),
                 self.current_offset * (self.current_vis_ch[1] - 0.5))
        self.canvas.axes.set_ylim(y_lim)

    def update_zoom_in_plots(self):
        """ This function updates the zoom without plotting again. """
        for plt_data in self.plotted_data:
            offset = np.array([c * self.current_offset for c in plt_data[
                "cha_idx"]])
            for i, line in enumerate(plt_data["handle"]):
                line.set_ydata(self.zoom * plt_data["data"][:, i] + offset[i])
        if self.scale_bar is not None:
            self.update_scale_bar()
        self.canvas.draw_idle()

    def update_scale_bar(self):
        if self.scale_bar is not None:
            self.scale_bar.remove()
        b_height = round(self.current_offset, 2)
        bl = "%s %s" % (self.get_number_scientific(b_height / self.zoom),
                        self.style_params["units"])
        v_bar = VerticalScaleBar(
            axes=self.canvas.axes,
            bar_height=b_height,
            bar_label=bl,
            label_size=PLOT_PARAMS['axes.labelsize']
        )
        self.scale_bar = self.canvas.axes.add_artist(v_bar.bar)

    def add_data(self, times, data, cha_idx=None, data_label=None,
                 time_ref=None, conditions_dict=None, events_dict=None,
                 style_params=None):
        """
        Adds a new time-series to the current plot,  with optional
        customization for color, labels, and time alignment.

        Parameters
        ----------
        times : array-like
            A 1D array containing the time points (in seconds) corresponding
            to the signal samples. The length of `times` should match the total
            number of samples.
        data : numpy.ndarray
            A 2D or 3D array representing the data to plot, with shape
            (segments, samples_per_segment, n_cha).
        cha_idx: array-like, optional
            A 1D array containing the indices of the channels of the plot
            that correspond to the channels in the signal. If None, signal
            channels must match the channels of the plot.
        data_label : str, optional
            Signal name so it appears in the legend. If None, then the
            signal name will be the current signal count.
        time_ref : float, None
            Reference time (in seconds) used to align condition intervals.
            All condition timestamps will be shifted by subtracting `time_ref`.
            This is useful when comparing signals with different starting times
            or when aligning multiple plots to a shared reference (e.g.,
            stimulus onset). By default, times are assumed to be relative
            to last timestamp in the plot.
        style_params : dict, optional
            Dictionary of style options for customizing the appearance of the
            signal and its legend entry. Supported keys include:

            Signal Line Style:
            - 'line_color': str, line color (default: from color cycle)
            - 'line_width': float, line thickness (default: 1.0)
            - 'line_alpha': float, transparency (0.0 transparent to 1.0 opaque, default: 1.0)
            - 'line_style': str, line style (e.g., '-', '--', ':', default: '-')
        """

        # Transform signal dimensions
        data = check_dimensions(data, mode='time-series')
        blocks, samples_per_block, n_cha = data.shape
        data = data.reshape(blocks * samples_per_block, n_cha)
        times = np.array(times)

        # Checks
        assert data.shape[0] == times.shape[0]

        # Channels
        if cha_idx is None:
            assert n_cha == self.n_cha, "Number of channels must match."
            cha_idx = np.arange(self.n_cha)
        else:
            assert len(cha_idx) == n_cha, "Number of channels must match."

        # Set time reference
        if time_ref is None:
            max_time = self._get_max_timestamp()
            time_ref = times[0] - max_time
        ref_times = times - time_ref

        # Reverse
        if self.reverse_channels:
            data = data[:, ::-1]

        # If there is no offset, set it one for the first signal
        if self.current_offset is None:
            self.current_offset = 4 * np.var(np.abs(data.ravel()))

        # If the number of channels to show is None, initialize it
        if self.cha_to_show is None:
            self.cha_to_show = n_cha
            self.current_vis_ch = np.array([0, n_cha])

        # If there is no window length, set it one for the first signal
        if self.time_to_show is None:
            self.time_to_show = min(self.initial_window_s,
                                    ref_times[-1] - ref_times[0])
            self.current_vis_window = (np.array([0, self.time_to_show]) +
                                       ref_times[0])

        # Ensure defaults are set in the style_params dictionary
        style_params = style_params or {}
        style_params.setdefault(
            "line_color", self.default_colors[self.plot_count])
        style_params.setdefault("line_width", 1.0)
        style_params.setdefault("line_alpha", 1.0)
        style_params.setdefault("line_style", "-")

        # Plot signal
        offset = np.array([c * self.current_offset for c in cha_idx])
        plot_handle = self.canvas.axes.plot(
            ref_times, self.zoom * data + offset,
            color=style_params["line_color"],
            alpha=style_params["line_alpha"],
            linestyle=style_params["line_style"],
            linewidth=style_params["line_width"],
            zorder=1)

        # Set ticks
        self.set_y_axis_ticks()

        if data_label is None:
            data_label = "Signal %i" % self.plot_count
        if data_label not in self.legend_items["data"]:
            self.legend_items["data"][data_label] = plot_handle[0]
        self.update_legend()

        # Limits
        self.update_ylim()
        self.canvas.axes.set_xlim(self.current_vis_window)

        # === Scale bar and draw ===
        if self.scale_bar is None:
            self.update_scale_bar()

        # === Store and update ===
        current_data = {
            "handle": plot_handle,
            "times": ref_times,
            "original_times": times,
            "data": data,
            "n_cha": n_cha,
            "cha_idx": cha_idx,
            "color": style_params["line_color"],
        }
        self.plotted_data.append(current_data)
        self.update_preview_plot()
        self.plot_count += 1

        # Conditions and events
        if conditions_dict is not None:
            self.add_conditions(times, time_ref, conditions_dict)
        if events_dict is not None:
            self.add_events(times, time_ref, events_dict)

        self.canvas.draw_idle()

    ## --------------------------- PREVIEW CANVAS --------------------------- ##
    def update_preview_plot(self):
        """
        Updates the overview (preview) plot shown above the main time plot.

        It uses the data stored in self.plotted_data to generate a compact
        representation of all currently plotted signals, including both time-series
        signals and heatmap-like images (e.g., spectrograms). It also overlays
        a shaded rectangle to indicate the currently zoomed-in region of the
        main plot.
        """
        if not hasattr(self, 'plotted_data') or not self.plotted_data:
            return

        self.canvas_preview.axes.clear()
        self.preview_handles = None

        preview_signal_data = list()

        # DATA
        for plot_dict in self.plotted_data:
            # Get data
            times = plot_dict["times"]
            data = plot_dict["data"]
            n_cha = plot_dict["n_cha"]
            color = plot_dict["color"]

            # Append preview signal
            median_ = np.median(data, axis=1)
            # Plot data
            self.canvas_preview.axes.plot(
                times, median_,
                color, zorder=2, alpha=0.5)
            # Append data
            preview_signal_data += median_.tolist()

        # Initialize only if preview handles are still not set
        maxy_ = np.max(np.abs(preview_signal_data)) * 2
        self._initialize_visualizing_window(-maxy_, maxy_)

        # Add conditions and events to preview
        self.update_preview_conds_and_events()

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


class TimeHeatmapPlot(TimePlot):

    def __init__(self, n_cha, cha_labels=None, cha_to_show=None,
                 reverse_channels=True, vis_step_s=2, initial_window_s=10,
                 style_params=None):
        """
       Initializes the TimePlot window with the specified parameters for
        visualizing multi-channel time-frequency (heatmap) data.

        Shortcuts
        ----------
        - Left / Right arrows: Move the visible time window backward or forward.
        - Up / Down arrows: Increase or decrease the number of visible channels.
        - Mouse scroll: Zoom in or out.

        Parameters:
        -----------
        n_cha : int
            Total number of channels in the data. This determines how many
            signal traces can be displayed in the plot and is used to validate
            display limits and layout.
        ch_labels : list of str, optional
            List of channel labels (e.g., ["Fz", "Cz", "Pz", ...]) to display
            on the y-axis or alongside each trace. If None, default channel
            indexes will be used as labels. The length of this list must
            match `n_cha`.
        ch_to_show : int or None, optional
            Number of channels to display in the plot at once. This
            value must be less than or equal to the total number of
            channels available in the data. If set to None, the number of
            channels to show is initialized to the total number of available
            channels.
        reverse_channels : bool, optional
            If True, channels are shown from top to bottom. If False,
            channels are shown from bottom to top.
        zoom_multiplier : float, optional
            Factor by which the plot zooms in or out when the zoom
            buttons are used. The default value is 1.2, meaning each zoom
            action will increase or decrease the visible range by 20%.
        vis_step_s : int, optional
            Time step (in seconds) used when navigating the plot, such as
            fast forwarding or rewinding. The default value is 2 seconds,
            meaning that each step moves the visible window forward or
            backward by 2 seconds.
        initial_window_s : float, optional
            Initial visualization window span in seconds. The default
            value is 10.0 s.
        style_params : dict, optional
            Dictionary of plot and visual style parameters shared with TimePlot.
            Additional parameters for TimeHeatmapPlot:

            **Channel axis**
            ------------------------
            - 'gap_ratio' : float
                Fraction of vertical space to leave as a gap between channels
                when `channel_overlap` is False. Value should be between 0
                and 1. Default is 0.05 (i.e., 5% of the offset per channel).

            **Dimension Axis**
            ----------------------------
            - 'show_dims_axis' : bool
                Whether to show an additional axis with values (e.g.,
                frequencies) for each channel heatmap. Default is True.
            - 'dims_axis_n_ticks' : int
                Number of evenly spaced ticks shown per channel in the
                dimension axis. Default is 2.
            - 'dims_axis_range' : array-like
                Tuple containing the range of the dimension axis.
        """
        # Load the custom layout from UIC
        super().__init__(
            n_cha, cha_labels=cha_labels,
            cha_to_show=cha_to_show,
            reverse_channels=reverse_channels,
            vis_step_s=vis_step_s,
            initial_window_s=initial_window_s,
            style_params=style_params)

        # Change default window settings
        self.setWindowTitle("Time Series Plot")

        # Defaults
        self.current_offset = 1
        self.colorbars = []
        self.colorbar_divider = None
        self.colorbar_axes = []
        self.style_params.setdefault("gap_ratio", 0.05)
        self.style_params.setdefault("show_dims_axis", False)
        self.style_params.setdefault("dims_axis_range", None)
        self.style_params.setdefault("dims_axis_n_ticks", 2)
        self.style_params.setdefault("dims_axis_left_pos", 0.005)
        self.style_params.setdefault("cha_axis_left_pos", -0.03)

        # Checks
        if self.style_params['show_dims_axis']:
            assert self.style_params['dims_axis_n_ticks'] >= 2, \
                'The minimum number of ticks in the dimension axis is 2'
            assert self.style_params['dims_axis_range'] is not None, \
                'You have to set dims_axis_range if show_dims_axis is True'
            assert len(self.style_params['dims_axis_range']) == 2, \
                ('Parameter dims_axis_range is expected to be a tuple of '
                 'length 2 with inferior and superior values of the range')

        # Add dims axis
        self.show_dims_axis = self.style_params['show_dims_axis']
        self.dims_axes = None
        if self.show_dims_axis:
            # Create dims axes
            self.dims_axes = self.canvas.axes.twinx()
            self.dims_axes.set_frame_on(False)
            self.dims_axes.yaxis.set_ticks_position('left')
            self.dims_axes.yaxis.set_label_position('left')
            self.dims_axes.set_yticks([])
            self.dims_axes.set_yticklabels([])
            self.dims_axes.spines['left'].set_position(
                ('axes', self.style_params['dims_axis_left_pos']))
            self.canvas.axes.spines['left'].set_position(
                ('axes', self.style_params['cha_axis_left_pos']))

    ## ----------------------------- CONTROLS ----------------------------- ##
    def zoom_in(self):
        """ This method makes the time plot to zoom in. """
        self.zoom *= self.zoom_multiplier
        self.update_zoom_in_plots()

    def zoom_out(self):
        """ This method makes the time plot to zoom out. """
        self.zoom /= self.zoom_multiplier
        self.update_zoom_in_plots()

    ## --------------------------- PLOT UPDATES --------------------------- ##
    def set_y_axis_ticks(self):
        # Reverse
        if self.reverse_channels:
            cha_labels = self.cha_labels[::-1]

        y_dims = self.style_params["dims_axis_range"]
        gap_size = self.current_offset * self.style_params['gap_ratio']
        dims_axis_n_ticks = self.style_params["dims_axis_n_ticks"]
        plot_height = self.current_offset - gap_size

        ytick_cha_positions, ytick_cha_labels = [], []
        ytick_dims_positions, ytick_dims_labels = [], []
        for i in range(self.n_cha):
            bottom = i * self.current_offset
            top = bottom + plot_height
            center = bottom + plot_height / 2
            # Channel axis ticks
            ytick_cha_positions.append(center)
            ytick_cha_labels.append(cha_labels[i])
            # Dims axis ticks
            if self.show_dims_axis:
                # ytick_dims_positions.extend([bottom, top])
                # ytick_dims_labels.extend([
                #     f"{y_dims[0]:.1f}",
                #     f"{y_dims[1]:.1f}"])
                # Generate equally spaced tick positions and labels between bottom and top
                tick_positions = np.linspace(bottom, top, dims_axis_n_ticks)
                tick_labels = np.linspace(y_dims[0], y_dims[1],
                                          dims_axis_n_ticks)

                ytick_dims_positions.extend(tick_positions)
                ytick_dims_labels.extend([f"{v:.1f}" for v in tick_labels])

        # Channel ticks
        self.canvas.axes.set_yticks(ytick_cha_positions)
        self.canvas.axes.set_yticklabels(ytick_cha_labels)

        # Dims axis
        if self.show_dims_axis:
            self.dims_axes.set_yticks(ytick_dims_positions)
            self.dims_axes.set_yticklabels(ytick_dims_labels)

    def update_ylim(self):
        gap_size = self.current_offset * self.style_params['gap_ratio']
        y_lim = (self.current_offset * self.current_vis_ch[0],
                 self.current_offset * self.current_vis_ch[1] - gap_size)
        self.canvas.axes.set_ylim(y_lim)
        if self.show_dims_axis:
            self.dims_axes.set_ylim(y_lim)

    def update_zoom_in_plots(self):
        """Updates signal amplitude and heatmap color range after zoom with global vmin/vmax."""
        if not self.plotted_data:
            return

        # Compute global vmin and vmax from all data
        all_data = [d["data"] for d in self.plotted_data]
        global_min = min(np.min(data) for data in all_data)
        global_max = max(np.max(data) for data in all_data)

        vmin = global_min * self.zoom
        vmax = global_max * self.zoom

        # Apply uniform clim to all imshow handles
        for plt_data in self.plotted_data:
            for im in plt_data["handle"]:
                im.set_clim(vmin=vmin, vmax=vmax)

        self.canvas.draw_idle()

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

    def add_data(self, times, data, cha_idx=None, data_label=None,
                 time_ref=None, conditions_dict=None, events_dict=None,
                 style_params=None):
        """
        Adds a multi-channel heatmap (e.g., time-frequency or power envelope)
        to the plot using color intensity instead of waveform traces.

        Parameters
        ----------
        times : array-like
            1D array of time values (seconds).
        data : numpy.ndarray
            3D or 4D array of shape (features, samples, n_cha) or
            (segments, features, samples, n_cha).
        cha_idx : list or np.ndarray, optional
            Channel indices that the signal corresponds to.
        data_label : str, optional
            Name of the signal for the legend.
        time_ref : float, optional
            Reference time to align timestamps.
        style_params : dict, optional
            Dictionary of style options. Supported keys:
                - 'colormap': str, Matplotlib colormap name (default: 'viridis')
                - 'alpha': float, transparency for heatmap [0.0 - 1.0] (default: 1.0)
        """
        data = check_dimensions(data, mode='time-heatmap')
        blocks, features_per_block, samples_per_block, n_cha = data.shape
        data = data.reshape(features_per_block, blocks * samples_per_block,
                            n_cha)
        times = np.array(times)

        assert data.shape[1] == len(times), "Mismatch between data and times."

        if cha_idx is None:
            assert n_cha == self.n_cha
            cha_idx = np.arange(self.n_cha)
        else:
            assert n_cha == len(cha_idx), "Mismatch between cha_idx and n_cha"

        if time_ref is None:
            max_time = self._get_max_timestamp()
            time_ref = times[0] - max_time
        ref_times = times - time_ref

        if self.reverse_channels:
            data = data[..., ::-1]

        # Set style defaults
        style_params = style_params or {}
        style_params.setdefault("colormap", "viridis")
        style_params.setdefault("alpha", 1.0)

        cmap = plt.get_cmap(style_params["colormap"])
        alpha = style_params["alpha"]
        norm = mcolors.Normalize(vmin=np.min(data), vmax=np.max(data))

        if self.cha_to_show is None:
            self.cha_to_show = n_cha
            self.current_vis_ch = np.array([0, n_cha])

        if self.time_to_show is None:
            self.time_to_show = min(self.initial_window_s,
                                    ref_times[-1] - ref_times[0])
            self.current_vis_window = (
                    np.array([0, self.time_to_show]) + ref_times[0])

        handles = []
        gap_size = self.current_offset * self.style_params["gap_ratio"]
        plot_height = self.current_offset - gap_size
        for i in range(n_cha):
            # Plot heatmap
            bottom = cha_idx[i] * self.current_offset
            top = bottom + plot_height
            extent = [ref_times[0], ref_times[-1], bottom, top]
            im = self.canvas.axes.imshow(
                data[:, :, i],
                extent=extent,
                aspect='auto',
                origin='lower',
                cmap=cmap,
                norm=norm,
                alpha=alpha,
                zorder=1
            )
            handles.append(im)

        # Channel ticks
        self.set_y_axis_ticks()

        # Add colorbar
        if self.colorbar_divider is None:
            self.colorbar_divider = make_axes_locatable(self.canvas.axes)

        colormap = style_params["colormap"]
        if colormap not in self.colorbars:
            # Add new colorbar
            position = "right"
            size = "2.5%"
            pad = 0.05
            cax = self.colorbar_divider.append_axes(position=position,
                                                    size=size, pad=pad)
            colorbar = self.canvas.fig.colorbar(im, cax=cax,
                                                orientation='vertical')
            self.colorbars.append(colormap)
            self.colorbar_axes.append(colorbar)

        # Hide ticks on all but the last
        for i, cb in enumerate(self.colorbar_axes):
            if i != len(self.colorbar_axes) - 1:
                cb.formatter = FuncFormatter(lambda x, pos: '')
                cb.update_ticks()

        # Legend
        if data_label is None:
            data_label = f"Heatmap {self.plot_count}"
        if data_label not in self.legend_items["data"]:
            patch = plt.Rectangle((0, 0), 1, 1, color=cmap(0.6))
            self.legend_items["data"][data_label] = patch
        self.update_legend()

        # Limits
        self.update_ylim()
        self.canvas.axes.set_xlim(left=self.current_vis_window[0],
                                  right=self.current_vis_window[1])

        # Save data
        self.plotted_data.append({
            "handle": handles,
            "times": ref_times,
            "original_times": times,
            "data": data,
            "n_cha": n_cha,
            "color": cmap,
        })

        # Conditions and events
        if conditions_dict is not None:
            self.add_conditions(times, time_ref, conditions_dict)
        if events_dict is not None:
            self.add_events(times, time_ref, events_dict)

        self.update_preview_plot()
        self.update_zoom_in_plots()
        self.plot_count += 1
        self.canvas.draw_idle()

    ## --------------------------- PREVIEW CANVAS --------------------------- ##
    def update_preview_plot(self):
        """
        Updates the overview (preview) plot shown above the main time plot.

        It uses the data stored in self.plotted_data to generate a compact
        representation of all currently plotted signals, including both time-series
        signals and heatmap-like images (e.g., spectrograms). It also overlays
        a shaded rectangle to indicate the currently zoomed-in region of the
        main plot.
        """
        if not hasattr(self, 'plotted_data') or not self.plotted_data:
            return

        self.canvas_preview.axes.clear()

        preview_signal_data = list()
        preview_im_handles = list()

        # DATA
        for plot_dict in self.plotted_data:
            # Get data
            times = plot_dict["times"]
            data = plot_dict["data"]
            n_cha = plot_dict["n_cha"]
            color = plot_dict["color"]

            median_ = np.median(data, axis=2)
            extent = [times[0],
                      times[-1],
                      0.2 * self.current_offset,
                      0.8 * self.current_offset]

            # Colormap
            im = self.canvas_preview.axes.imshow(
                median_,
                extent=extent,
                aspect='auto',
                origin='lower',
                cmap=color,
                alpha=0.7,
                zorder=1
            )

            # Append data
            preview_signal_data += median_.tolist()

        # Initialize only if preview handles are still not set
        self._initialize_visualizing_window(0, self.current_offset)
        self.preview_handles["im_handles"] = preview_im_handles

        # Add conditions and events to preview
        self.update_preview_conds_and_events()

        # Other options
        max_length_s = self._get_max_timestamp()
        min_length_s = self._get_min_timestamp()
        self.canvas_preview.figure.patch.set_alpha(0)
        self.canvas_preview.figure.subplots_adjust(left=0.125, right=0.901,
                                                   top=0.99, bottom=0.07)
        self.canvas_preview.axes.grid(False)
        self.canvas_preview.axes.set_xlim(left=min_length_s, right=max_length_s)
        self.canvas_preview.axes.set_ylim((0, self.current_offset))
        self.canvas_preview.axes.patch.set_alpha(0)
        self.canvas_preview.axes.set_axis_off()

        # Draw it
        self.canvas_preview.draw_idle()


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

    # Simulated multi-channel signal
    fs = 100
    n_channels = 16
    t_range = [0, 250]
    n_samples =  (t_range[1] - t_range[0]) * fs
    t = np.linspace(t_range[0], t_range[1], n_samples)

    # Generate noisy sine signals
    times1 = t
    signal_1 = np.random.randn(n_samples, n_channels)

    # Generate signal 2
    times2 = t + 1000
    sine_wave = np.sin(2 * np.pi * 1 * t)[:, np.newaxis]
    signal_2 = sine_wave + 0.25 * np.random.randn(n_samples, n_channels)

    # Channel labels
    ch_labels = [
        'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4',
        'P3', 'P4', 'O1', 'O2', 'F7', 'F8',
        'T3', 'T4', 'T5', 'T6'
    ]

    # Conditions and events
    conditions = {
        'eyes-closed': {'desc-name': 'Eyes closed', 'label': 1,
                        'shortcut': 'C'},
        'eyes-open': {'desc-name': 'Eyes open', 'label': 0}
    }
    conditions_labels = [0, 0, 1, 1]
    conditions_times = [10, 60, 70, 200]
    c_dict = {
        'conditions': conditions,
        'conditions_labels': conditions_labels,
        'conditions_times': conditions_times
    }

    c_dict_2 = c_dict.copy()
    c_dict_2['conditions_times'] = \
        [t + 1000 for t in c_dict_2['conditions_times']]

    events = {
        'blink': {'desc-name': 'Blink', 'label': 0, 'shortcut': 'B'},
        'movement': {'desc-name': 'Movement', 'label': 1}
    }
    events_labels = [0, 1, 0, 1, 0]
    events_times = [15, 62, 72, 150, 202]
    e_dict = {
        'events': events,
        'events_labels': events_labels,
        'events_times': events_times
    }

    e_dict_2 = e_dict.copy()
    e_dict_2['events_times'] = [t + 1000 for t in e_dict_2['events_times']]

    # ================= TIME SERIES PLOT ================= #

    # Create time plot manager and window
    time_plot_manager = TimePlotManager()
    time_plot = TimeSeriesPlot(
        n_cha=n_channels,
        cha_labels=ch_labels,
        cha_to_show=4,
        reverse_channels=True)
    time_plot_manager.set_time_plot(time_plot)

    # Add time-series EEG signal
    time_plot.add_data(
        times=times1,
        data=signal_1[:, 0:8],
        data_label="S1",
        cha_idx=np.arange(0, 8),
        time_ref=None,
        conditions_dict=c_dict,
        events_dict=e_dict,
        style_params=None
    )

    # Add time-series EEG signal
    time_plot.add_data(
        times=times2,
        data=signal_2[:, 8:16],
        data_label="S2",
        cha_idx=np.arange(8, 16),
        time_ref=times2[0],
        conditions_dict=c_dict_2,
        events_dict=e_dict_2,
        style_params=None

    )

    # Launch plot
    time_plot_manager.show()

    # ================= TIME HEATMAP PLOT ================= #

    # Spectrogram
    from medusa.transforms import fourier_spectrogram
    spec_params = {
        'fs': fs,
        'time_window': 10.0,
        'overlap_pct': 90,
        'smooth': False,
        'smooth_sigma': 1,
        'apply_detrend': True,
        'apply_normalization': True,
        'scale_to': None,
    }
    # Spectrogram 1
    spec1_cha = []
    for i in range(n_channels):
        spec_i, t_spec, f_spec = fourier_spectrogram(signal_1[:, i],
                                                     **spec_params)
        spec_i = 10 * np.log10(spec_i + 1e-12)
        spec1_cha.append(spec_i[..., np.newaxis])
    f_spec_1 = f_spec
    t_spec_1 = t_spec
    spec1 = np.concatenate(spec1_cha, axis=-1)
    # Spectrogram 2
    spec2_cha = []
    for i in range(n_channels):
        spec_i, t_spec, f_spec = fourier_spectrogram(signal_2[:, i],
                                                     **spec_params)
        spec_i = 10 * np.log10(spec_i + 1e-12)
        spec2_cha.append(spec_i[..., np.newaxis])
    f_spec_2 = f_spec
    t_spec_2 = t_spec
    spec2 = np.concatenate(spec2_cha, axis=-1)

    # Create time plot manager and window
    time_plot_manager = TimePlotManager()
    time_plot = TimeHeatmapPlot(
        n_cha=n_channels,
        cha_labels=ch_labels,
        cha_to_show=4,
        reverse_channels=True,
        style_params={
            'gap_ratio': 0.05,
            'show_dims_axis': True,
            'dims_axis_range': (0, 3),
            'dims_axis_n_ticks': 4,
            'dims_axis_left_pos': 0.005,
            'cha_axis_left_pos': -0.02
        })
    time_plot_manager.set_time_plot(time_plot)

    freq_mask = (f_spec_1 >= 0) & (f_spec_1 <= 5)
    time_plot.add_data(
        times=t_spec_1,
        data=spec1[freq_mask, :, 0:8],
        data_label="S1",
        cha_idx=np.arange(0, 8),
        time_ref=None,
        conditions_dict = c_dict,
        events_dict = e_dict,
    )

    freq_mask = (f_spec_1 >= 0) & (f_spec_1 <= 3)
    time_plot.add_data(
        times=t_spec_2,
        data=spec2[freq_mask, :, 8:16],
        data_label="S2",
        cha_idx=np.arange(8, 16),
        time_ref=t_spec_2[0],
        conditions_dict=c_dict,
        events_dict=e_dict,
        style_params={
            'colormap': 'magma'
        }
    )

    # Launch plot
    time_plot_manager.show()

