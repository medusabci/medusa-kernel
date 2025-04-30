"""
Created on Thu Aug 25 17:09:18 2022
Edited on Mon Jan 09 14:00:00 2023
@author: Diego Marcos-Martínez
"""
import warnings

# External imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from matplotlib.widgets import Slider
from matplotlib.widgets import Button
# Medusa imports
from medusa.utils import check_dimensions


def __plot_epochs_lines(ax, blocks, samples_per_block, fs, min_val, max_val):
    """Aux function to plot vertical lines in case of signal is divided in two
        or more epochs"""
    t_ = np.arange(1,blocks) * int(samples_per_block / fs)
    ax.vlines(t_, min_val, max_val, colors='k',
                  linewidth=2, linestyles='solid')


def __plot_events_lines(ax, events_dict, min_val, max_val, display_times):
    """Aux function to plot vertical lines corresponding  to marked events"""
    # Check errors
    if not isinstance(events_dict, dict):
        raise ValueError("'events_dict' must be a dict."
                         "Please, read carefully the time_plot documentation"
                         "to know how to define 'events_dict' properly. ")
    if not 'events' in events_dict.keys():
        raise ValueError("'events_dict' must have 'events' key."
                         "Please, read carefully the time_plot documentation"
                         "to know how to define 'events_dict' properly.")
    if not 'events_labels' in events_dict.keys():
        raise ValueError("'events_dict' must have 'event_labels' key."
                         "Please, read carefully the time_plot documentation"
                         "to know how to define 'events_dict' properly.")
    if not 'events_times' in events_dict.keys():
        raise ValueError("'events_dict' must have 'event_times' key."
                         "Please, read carefully the time_plot documentation"
                         "to know how to define 'events_dict' properly.")

    events_names = []
    for key_event in list(events_dict['events'].keys()):
        events_names.append(events_dict['events'][key_event][
                                'desc-name'])
    legend_lines = {}
    previous_conditions = None
    cmap = plt.get_cmap('rainbow')(np.linspace(0,1,len(events_names)))
    events_order = np.array(events_dict['events_labels'])

    if ax.legend_ is not None:
        handles, labels = ax.get_legend_handles_labels()
        previous_conditions = list(np.unique(labels))


    events_timestamps = np.array(events_dict['events_times'])

    # Check if events_timestamps are referenced to recording start
    if np.any(events_timestamps > display_times[-1]):
        raise ValueError("Incorrect format of events_timestamps. "
                         "The values must be referenced to the beginning "
                         "of the record, so that the first timestamp has value 0.")
    for event_idx, event_type in enumerate(set(events_order)):
        t_ = events_timestamps[events_order == event_type]
        l = ax.vlines(t_, min_val, max_val, colors=cmap[event_idx],
                      linewidth=2, linestyles='dashed',
                      label=events_names[event_idx])
        if event_type not in legend_lines.keys():
            legend_lines[event_type] = l

    # Create legend above the plot
    if previous_conditions is not None:
        previous_handles = ax.legend_.legend_handles
        for legend_line in list(legend_lines.values()):
            previous_handles.append(legend_line)
            previous_conditions.append(legend_line._label)
        ax.legend(handles=previous_handles, labels=previous_conditions,
                  loc='upper center', bbox_to_anchor=(0.5, 1.15),
                  ncol=3, fancybox=True, shadow=True)
    else:
        ax.legend(handles=list(legend_lines.values()), loc='upper center',
                  bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True,
                  shadow=True)


def __plot_condition_shades(ax, conditions_dict, display_times, min_val, max_val):
    """Aux function to plot background shades to corresponding to different
       conditions during the signal recording. """
    # Check errors
    if not isinstance(conditions_dict, dict):
        raise ValueError("'conditions_dict' must be a dict."
                         "Please, read carefully the time_plot documentation"
                         "to know how to define 'conditions_dict' properly. ")
    if not 'conditions' in conditions_dict.keys():
        raise ValueError("'conditions_dict' must have 'conditions' key."
                         "Please, read carefully the time_plot documentation"
                         "to know how to define 'conditions_dict' properly.")
    if not 'conditions_labels' in conditions_dict.keys():
        raise ValueError("'conditions_dict' must have 'condition_labels' key."
                         "Please, read carefully the time_plot documentation"
                         "to know how to define 'conditions_dict' properly.")
    if not 'conditions_times' in conditions_dict.keys():
        raise ValueError("'conditions_dict' must have 'condition_times' key."
                         "Please, read carefully the time_plot documentation"
                         "to know how to define 'conditions_dict' properly.")

    conditions_names = []
    legend_patches = {}

    for key_condition in list(conditions_dict['conditions'].keys()):
        conditions_names.append(conditions_dict['conditions'][key_condition][
                                    'desc-name'])
    condition_timestamps = conditions_dict['conditions_times']

    # Check if timestamps are an iterable object
    if not isinstance(condition_timestamps,np.ndarray) and \
        not isinstance(condition_timestamps,list):
        condition_timestamps = np.asarray([condition_timestamps])
        labels_order = np.array([conditions_dict['conditions_labels']])
    else:
        labels_order = np.array(conditions_dict['conditions_labels'])

    # Check if timestamps are referenced to recording start
    if np.any(condition_timestamps > display_times[-1]):
        raise ValueError("Incorrect format of condition_timestamps. "
                         "The values must be referenced to the beginning "
                         "of the record, so that the first timestamp has value 0.")

    # Check if all conditions have a start and an end and fix it if not
    c_idx = 0
    corrected = False
    while c_idx < len(labels_order)-1:
        if (labels_order[c_idx] != labels_order[c_idx+1]):
            if c_idx != 0:
                if labels_order[c_idx] != labels_order[c_idx-1]:
                    labels_order = np.insert(labels_order, c_idx + 1,
                                             labels_order[c_idx])
                    condition_timestamps = np.insert(condition_timestamps,
                                                     c_idx + 1,
                                                     condition_timestamps[
                                                         c_idx + 1])
                    corrected = True
            else:
                labels_order = np.insert(labels_order,c_idx+1,labels_order[c_idx])
                condition_timestamps = np.insert(condition_timestamps,
                                                 c_idx+1,
                                                 condition_timestamps[c_idx+1])
                corrected = True
        c_idx += 1
    if labels_order[-1] != labels_order[-2]:
        labels_order = np.append(labels_order,labels_order[-1])
        condition_timestamps = np.append(condition_timestamps,
                                                 display_times[-1])
        corrected = True

    if corrected:
        warnings.warn("The dictionary of conditions does not follow the "
                "correct format ([Start condition X, End condition X,"
                " Start condition Y, End condition Y ...]). "
                "The labels and timestamps vector has been "
                "automatically corrected. Check that the OK "
                "is correct. ")

    cmap = plt.get_cmap('jet')(np.linspace(0,1,len(conditions_names)))

    for condition_margin_idx in np.arange(0,len(condition_timestamps),2):
        l = ax.fill_betweenx([min_val,max_val],
                             condition_timestamps[condition_margin_idx],
                             condition_timestamps[condition_margin_idx+1],
                             color= cmap[labels_order[condition_margin_idx]],
                             alpha=0.3,
                             label=np.array(conditions_names)[np.array(
                                 labels_order[condition_margin_idx])])
        if np.array(conditions_names)[
            np.array(labels_order[
                         condition_margin_idx])] not in legend_patches.keys():
            legend_patches.update(
                {np.array(conditions_names)[np.array(
                                 labels_order[condition_margin_idx])]: l})

    # Create legend above the plot
    ax.legend(handles=list(legend_patches.values()),
              loc='upper center', bbox_to_anchor=(0.5, 1.15),
              ncol=3, fancybox=True, shadow=True)


def __reshape_signal(epochs):
    """Aux function than reshapes the signal if it is divided in
    epochs in order to plot it in a row"""
    epoch_c = epochs.copy()
    blocks, samples_per_block, channels = epoch_c.shape
    epoch_c = np.reshape(epoch_c,
                         (int(blocks * samples_per_block), channels))
    return epoch_c


def time_plot(signal, times=None, fs=1.0, ch_labels=None, time_to_show=None,
              ch_to_show=None, ch_offset=None, color='k',
              conditions_dict=None, events_dict=None, show_epoch_lines=True,
              fig=None, axes=None):
    """
    Parameters
    ---------
    signal: numpy ndarray
        Signal with shape of [n_epochs,n_samples, n_channels] or
        [n_samples, n_channels]
    times: numpy ndarray
        Timestamps of each sample of the signal with shape [n_samples]
    fs: float
        Sampling rate. Value 1 as default
    ch_labels: list of strings or None
        List containing the channel labels
    time_to_show: float or None
        Width of the time window displayed. If time_to_show value is greater than
        the entire signal duration, this will be set as new time_to_show value.
        If None, time_to_show value will be chosen between the minimum value of
        the following windows: five seconds or the entire duration of the signal.
    ch_to_show: int or None
        Number of channels depicted in the plot. This parameter must be less or
        equal to the number of channels available in the recording. If None,
        this parameter is set as the total number of channels.
    ch_offset: flot or None
        Amplitude value to compute the offset of each channel. If None, the value
        is automatically calculated from signal values.
    color: string or tuple
        Color of the signal line. It is plotted in black by default.
    conditions_dict: dict
        Dictionary with the following structure:
        {'conditions':{'con_1':{'desc-name':'Condition 1','label':0},
                       'con_2':{'desc-name':'Condition 2','label':1}},
         'condition_labels': [0,0,1,1,0,0],
         'condition_times': [0,14,14,28,28,35]}
         In this example, the sub-dictionary 'conditions' include each condition
         with a descriptor name ('desc-name') which will be show in the time-plot
         legend, and the label to identify the condition. For its part,
         'condition_labels' must be a list containing the order of start and end
         of each condition. Finally, 'condition_times' value must be a list
         with the same length as 'condition_labels' containing the time stamps
         (in seconds) related with the start and the end of each condition. Note
         that these time stamps must be referenced to the start of the signal
         recording (the value 14 in the example means the 14th second after the
         start of recording). Note that as the end of a conditions coincides
         with the start of the following condition, the same time stamps must be
         included twice (see 14 and 28 values in 'condition_times' in the
         example).
    events_dict:
        Dictionary with the following structure:
        {'events':{'event_1':{'desc-name':'Event 1','label':0},
                   'event_2':{'desc-name':'Event 2','label':1}},
         'event_labels': [0,1,1,1,0,1],
         'event_times': [0,14,15.4,28,2,35]}
         In this example, the sub-dictionary 'events' include each event with a
         descriptor name ('desc-name') which will be show in the time-plot
         legend, and the label to identify the event. For its part,
         'event_labels' must be a list containing the order in which each event
         ocurred. Finally, 'condition_times' value must be a list
         with the same length as 'event_times' containing the time stamps
         (in seconds) related with each event. Note that, as in 'conditions_dict'
         argument, these time stamps must be referenced to the start of the
         signal recording.
    show_epoch_lines: bool
        If signal is divided in epochs and the parameter value is True, vertical
        dotted red lines will be plotted, splitting the epochs. Otherwise, they
        will not be plotted. True is the default value.
    fig: matplotlib.pyplot.figure or None
        If a matplotlib figure is specified, the plot is displayed inside it.
        Otherwise, the plot will generate a new figure.
    axes: matplotlib.pyplot.axes or None
        If a matplotlib axes are specified, the plot is displayed inside it.
        Otherwise, the plot will generate a new figure.
    Notes
    ---------
    If time_to_show or ch_to_show parameters are defined and the signal is
    partially represented, vertical and horizontal sliders will be added to
    control the represented channels and time window, respectively. Vertical
    slider can be controlled by pressing up and dow arrows, and by dragging the
    marker. For its part, horizontal slider van be controlled by pressing
    right or left arrow, and by dragging the marker.
    """

    # Deprecated warning
    warnings.warn("This function is deprecated. Use "
                  "medusa.analysis.time_plot instead")

    # Check signal dimensions
    signal = check_dimensions(signal)

    # Get signal dimensions
    blocks, samples_per_block, channels = signal.shape

    # Check if there are channel labels
    if ch_labels is None:
        ch_labels = [f"Channel {idx}" for idx in range(channels)]
    else:
        if not isinstance(ch_labels, list):
            raise ValueError("Channel labels ('ch_labels') must be entered"
                             "as a list.")

    epoch_c = __reshape_signal(signal)
    del signal

    # Set maximum length of x-axis to be displayed
    if time_to_show is None:
        # The default time window is 5 seconds
        time_to_show = min(5, epoch_c.shape[0] / fs)

    # Set offset between channels
    if ch_offset is None:
        ch_offset = time_to_show * 2 * np.std(np.abs(
            epoch_c.copy().ravel()))
    offset_values = np.arange(channels) * ch_offset
    epoch_c = epoch_c - offset_values
    ch_off = offset_values
    del ch_offset, offset_values

    max_val, min_val = epoch_c.max(), epoch_c.min()

    # Define times vector
    if times is None:
        display_times = np.linspace(0, (epoch_c.shape[0] - 1) / fs,
                                    epoch_c.shape[0])
    else:
        display_times = times

    # Initialize plot
    if fig is None:
        fig = plt.figure()
    if axes is None:
        axes = fig.add_subplot(111)

    fig.patch.set_alpha(0)
    axes.set_alpha(0)
    fig.subplots_adjust(left=0.1, bottom=0.1)
    ch_slider, time_slider = None, None

    # Set maximum length of x-axis to be displayed
    if ch_to_show is None:
        # The default time window is 5 seconds
        ch_to_show = len(ch_labels)
    else:
        if not isinstance(ch_to_show, int):
            raise ValueError("Channel to show ('ch_to_show') must be an "
                             "integer value.")
        if ch_to_show > len(ch_labels):
            ch_to_show = len(ch_labels)
            raise Warning("Entered a channel to show ('ch_to_show') value "
                          "greater than the number of channels in recording."
                          "This parameter will be set equal to the number of "
                          "channels.")

    # Create a time slider only if the time window to show is less than the
    # signal duration
    if time_to_show < epoch_c.shape[0] / fs:
        # Get the bounding box of the x-axis label
        left, bottom, width, height = axes.get_position().bounds
        # Add a new axis just below the axes
        # ax_time = fig.add_axes([0.15, 0.02, 0.70, 0.03])
        ax_time = fig.add_axes([left, bottom-0.005, width, 0.02])
        # Adjust the main plot to make room for the sliders
        # Define the max value of slider
        max_x = int(time_to_show * fs)
        max_val_time_slider = display_times[-1] - max_x / fs
        # Define slider
        time_slider = Slider(ax=ax_time, label='', valmin=0,
                             valmax=max_val_time_slider, valinit=0,
                             color='k')
        time_slider.valtext.set_visible(False)

        # Function to be call everytime the slider is moved
        def __update_time(val):
            # Update x-axis
            axes.set_xlim(val, max_x / fs + val)
            # Update canvas
            fig.canvas.draw()

        # Assign the update function to the slider
        time_slider.on_changed(__update_time)

    else:
        # The time window to show is the whole signal
        max_x = epoch_c.shape[0]

    # Create a channel slider only if the channel window to show is less
    # than the total number of channels
    if ch_to_show < epoch_c.shape[1]:
        # Get the bounding box of the x-axis label
        left, bottom, width, height = axes.get_position().bounds
        # Add a new axis just below the axes
        # ax_ch = fig.add_axes([0.86, 0.15, 0.02, 0.73])
        ax_ch = fig.add_axes([left+width, bottom, 0.01, height])
        # Define the max value of slider
        max_y = ch_to_show
        max_val_ch_slider = epoch_c.shape[1] - ch_to_show
        # Define slider
        ch_slider = Slider(ax=ax_ch, label='', valmin=-max_val_ch_slider,
                           valmax=0, valinit=0, valstep=1,
                           color='k', orientation='vertical')
        ch_slider.valtext.set_visible(False)

        # Function to be call everytime the slider is moved
        def __update_ch(val):
            # Update y-axis
            axes.set_ylim(
                -ch_off[max_y - val - 1] - 0.5 * ch_off[1],
                -ch_off[-val] + 0.5 * ch_off[1])
            # Update canvas
            fig.canvas.draw()

        # Assign the update function to the slider
        ch_slider.on_changed(__update_ch)

    else:
        # The time window to show is the whole signal
        max_y = epoch_c.shape[1]

    # Allow sliders to be controlled by arrow keys
    def on_key(event):
        if event.key == 'up':
            if ch_slider is not None:
                if ch_slider.val != 0:
                    ch_slider.set_val(ch_slider.val + 1)
        elif event.key == 'down':
            if ch_slider is not None:
                if ch_slider.val != -max_val_ch_slider:
                    ch_slider.set_val(ch_slider.val - 1)
        elif event.key == 'right':
            if time_slider is not None:
                if time_slider.val < max_val_time_slider:
                    if time_slider.val + 1 > max_val_time_slider:
                        time_slider.set_val(max_val_time_slider)
                    else:
                        time_slider.set_val(time_slider.val + 1)
        elif event.key == 'left':
            if time_slider is not None:
                if time_slider.val > 0:
                    if time_slider.val - 1 < 0:
                        time_slider.set_val(0)
                    else:
                        time_slider.set_val(time_slider.val - 1)

    fig.canvas.mpl_connect('key_press_event', on_key)

    #  Call the aux function to plot conditions
    if conditions_dict is not None:
        __plot_condition_shades(axes, conditions_dict, display_times,
                                min_val, max_val)

    #  Call the aux function to plot vertical lines to mark the events
    if events_dict is not None:
        __plot_events_lines(axes, events_dict, min_val, max_val, display_times)

    # Plot the signal
    axes.plot(display_times, epoch_c, color, linewidth=0.5)
    axes.set_yticks(-ch_off, labels=ch_labels)
    if len(ch_off) > 1:
        axes.set_ylim(-ch_off[max_y - 1] - 0.5 * ch_off[1],
                      -ch_off[0] + 0.5 * ch_off[1])
    axes.set_xlim(0, display_times[max_x])
    axes.set_xlabel('Time (s)')

    # Call the aux function to plot vertical lines to split signal in epochs
    if show_epoch_lines:
        __plot_epochs_lines(axes, blocks, samples_per_block, fs,
                            min_val, max_val)

    warnings.warn("In order to enjoy all the available options of the "
                            "time_plot function, it is necessary to have an "
                            "interactive backend compatible with matplotlib "
                            "enabled.")
    return fig, axes


if __name__ == "__main__":
    """ Example of use: """
    from medusa.components import Recording
    from medusa.meeg import meeg
    import medusa.frequency_filtering as ff

    # Using an interactive backend
    matplotlib.use('TkAgg')

    # Defining some signal parameters 
    fs = 256
    T = 60
    t = np.arange(0, T, 1 / fs)
    l_cha = ['F7', 'F3', 'FZ', 'F4', 'F8', 'FCz', 'C3', 'CZ', 'C4', 'CPz', 'P3',
             'PZ', 'P4','PO7', 'POZ', 'PO8']
    A = 1  # noise amplitude
    sigma = 0.5  # Gaussian noise variance
    f = 5  # frequency of sinusoids (Hz)
    ps = np.linspace(0, -np.pi / 2, len(l_cha))  # Phase differences
    np.random.seed(0)

    # Define signal
    signal = np.empty((len(t), len(l_cha)))
    for c in range(len(l_cha)):
        signal[:, c] = np.sin(2 * np.pi * f * t - ps[c]) + A * np.random.normal(
            0, sigma, size=t.shape)

    signal = signal.reshape((10, int(signal.shape[0] / 10), signal.shape[1]))

    # Define events and conditions dicts
    e_dict = {'events': {'event_1': {'desc-name': 'Event 1', 'label': 0},
                         'event_2': {'desc-name': 'Event 2', 'label': 1},
                         'event_3': {'desc-name': 'Event 3', 'label': 2},
                         'event_4': {'desc-name': 'Event 4', 'label': 3},
                         'event_5': {'desc-name': 'Event 5', 'label': 4}},

              'events_labels': [0, 1, 1, 2, 0, 1, 3, 0, 1, 4],
              'events_times': [5, 14, 15.4, 28, 2, 35, 43, 49, 53, 58.5]}

    c_dict = {'conditions': {'con_1': {'desc-name': 'Condition 1', 'label': 0},
                             'con_2': {'desc-name': 'Condition 2', 'label': 1}},
              'conditions_labels': [0, 0, 1, 1,  0, 0,  1, 1, 0, 0 ],
              'conditions_times': [0, 14, 14, 28, 28, 35, 35, 50, 50, 59.9]}

    # Initialize TimePlot instance
    figure = plt.figure()
    time_plot(signal=signal,times=None,fs=fs,ch_labels=l_cha,time_to_show=None,
              ch_to_show=5,ch_offset=None,conditions_dict=c_dict,
              events_dict=e_dict,show_epoch_lines=True,fig=figure)