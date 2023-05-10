"""
Created on Thu Aug 25 17:09:18 2022
Edited on Mon Jan 09 14:00:00 2023
@author: Diego Marcos-Martínez
"""
# External imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Qt5Agg')
from matplotlib.widgets import Slider
from matplotlib.widgets import Button

# Medusa imports
from medusa.utils import check_dimensions


def __plot_epochs_lines(ax, blocks, samples_per_block, fs, min_val, max_val,
                        color='red'):
    """Aux function to plot vertical lines in case of signal is divided in two
        or more epochs"""
    vertical_lines = np.empty((blocks - 1, 2, 50))
    for block in range(blocks - 1):
        vertical_lines[block, :, :] = np.asarray(
            [np.ones(50) * (block + 1) * int(samples_per_block / fs),
             np.linspace(min_val, 1.5 * max_val, 50)])
        ax.plot(vertical_lines[block, 0, :], vertical_lines[block, 1, :],
                '--', color=color, linewidth=1.5)


def __plot_events_lines(ax, events_dict, min_val, max_val):
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
    if not 'event_labels' in events_dict.keys():
        raise ValueError("'events_dict' must have 'event_labels' key."
                         "Please, read carefully the time_plot documentation"
                         "to know how to define 'events_dict' properly.")
    if not 'event_times' in events_dict.keys():
        raise ValueError("'events_dict' must have 'event_times' key."
                         "Please, read carefully the time_plot documentation"
                         "to know how to define 'events_dict' properly.")

    events_names = []
    for key_event in list(events_dict['events'].keys()):
        events_names.append(events_dict['events'][key_event][
                                'desc_name'])

    events_order = events_dict['event_labels']
    events_timestamps = events_dict['event_times']
    legend_lines = {}
    previous_conditions = None
    cmap = matplotlib.cm.get_cmap('tab10')
    if len(list(set(events_order))) > 10:
        raise Warning("Attention! The maximum number of different events"
                      "is 10. If you have entered more than 10 different "
                      "events, there will be events whose color "
                      "matches. ")

    if ax.legend_ is not None:
        handles, labels = ax.get_legend_handles_labels()
        previous_conditions = list(set(labels))

    for event_idx in range(len(events_timestamps)):
        l = ax.plot(np.ones(50) * events_timestamps[event_idx],
                    np.linspace(min_val, 2 * max_val, 50),
                    '--', color=cmap.colors[events_order[event_idx]],
                    linewidth=1.5, label=np.array(events_names)[np.array(
                events_order[event_idx])])
        if str(events_order[event_idx]) not in legend_lines.keys():
            legend_lines.update({str(events_order[event_idx]): l[0]})

    # Create legend above the plot
    if previous_conditions is not None:
        previous_handles = ax.legend_.legendHandles
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


def __plot_condition_shades(ax, conditions_dict, min_val, max_val):
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
    if not 'condition_labels' in conditions_dict.keys():
        raise ValueError("'conditions_dict' must have 'condition_labels' key."
                         "Please, read carefully the time_plot documentation"
                         "to know how to define 'conditions_dict' properly.")
    if not 'condition_times' in conditions_dict.keys():
        raise ValueError("'conditions_dict' must have 'condition_times' key."
                         "Please, read carefully the time_plot documentation"
                         "to know how to define 'conditions_dict' properly.")

    conditions_names = []
    for key_condition in list(conditions_dict['conditions'].keys()):
        conditions_names.append(conditions_dict['conditions'][key_condition][
                                    'desc_name'])

    labels_order = conditions_dict['condition_labels']
    condition_timestamps = conditions_dict['condition_times']
    legend_patches = {}

    if len(list(set(labels_order))) > 8:
        cmap = matplotlib.cm.get_cmap('Set2')
    else:
        cmap = matplotlib.cm.get_cmap('Set3')
        if len(list(set(labels_order))) > 12:
            raise Warning(
                "Attention! The maximum number of different conditions"
                "is 12. If you have entered more than 12 different "
                "conditions, there will be conditions whose color "
                "matches. ")

    for condition_margin_idx in range(1, len(condition_timestamps)):
        if condition_timestamps[condition_margin_idx - 1] != \
                condition_timestamps[condition_margin_idx]:
            l = ax.fill_betweenx(np.linspace(min_val, 2 * max_val, 50),
                                 np.ones(50) * condition_timestamps
                                 [condition_margin_idx - 1],
                                 np.ones(50) * condition_timestamps
                                 [condition_margin_idx], color=
                                 cmap.colors[
                                     labels_order[condition_margin_idx]],
                                 alpha=0.3,
                                 label=np.array(conditions_names)[np.array(
                                     labels_order[condition_margin_idx])])
            if str(labels_order[
                       condition_margin_idx]) not in legend_patches.keys():
                legend_patches.update(
                    {str(labels_order[condition_margin_idx]): l})

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


def time_plot(signal, fs=1.0, ch_labels=None, time_to_show=None,
              ch_to_show=None, channel_offset=None, color='k',
              conditions_dict=None, events_dict=None, show_epoch_lines=True,
              show=False, fig=None, axes=None):
    """
    Parameters
    ---------
    signal: numpy ndarray
        Signal with shape of [n_epochs,n_samples, n_channels] or
        [n_samples, n_channels]
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
    channel_offset: flot or None
        Amplitude value to compute the offset of each channel. If None, the value
        is automatically calculated from signal values.
    color: string or tuple
        Color of the signal line. It is plotted in black by default.
    conditions_dict: dict
        Dictionary with the following structure:
        {'conditions':{'con_1':{'desc_name':'Condition 1','label':0},
                       'con_2':{'desc_name':'Condition 2','label':1}},
         'condition_labels': [0,0,1,1,0,0],
         'condition_times': [0,14,14,28,28,35]}
         In this example, the sub-dictionary 'conditions' include each condition
         with a descriptor name ('desc_name') which will be show in the time-plot
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
        {'events':{'event_1':{'desc_name':'Event 1','label':0},
                   'event_2':{'desc_name':'Event 2','label':1}},
         'event_labels': [0,1,1,1,0,1],
         'event_times': [0,14,15.4,28,2,35]}
         In this example, the sub-dictionary 'events' include each event with a
         descriptor name ('desc_name') which will be show in the time-plot
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
    show: bool
        Show matplotlib figure
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
    if channel_offset is None:
        channel_offset = time_to_show * 2 * np.std(np.abs(
            epoch_c.copy().ravel()))
    offset_values = np.arange(channels) * channel_offset
    if len(offset_values)>1:
        max_val = -offset_values[0] + offset_values[1]
        min_val = -offset_values[-1] - offset_values[1]
    else:
        max_val = 2 * np.max(epoch_c)
        min_val = 2 * np.min(epoch_c)

    epoch_c = epoch_c - offset_values
    ch_off = offset_values
    del channel_offset, offset_values

    max_val, min_val = epoch_c.max(), epoch_c.min()

    # Define times vector
    display_times = np.linspace(0, int(epoch_c.shape[0] / fs),
                                epoch_c.shape[0])

    # Initialize plot
    if fig is None:
        fig = plt.figure()
    if axes is None:
        axes = fig.add_subplot(111)

    fig.patch.set_alpha(0)
    axes.set_alpha(0)
    fig.subplots_adjust(left=0.15, bottom=0.15, right=0.85)
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
        max_x = int(time_to_show * fs)
        # Adjust the main plot to make room for the sliders
        ax_time = fig.add_axes([0.15, 0.02, 0.70, 0.03])
        # Define the max value of slider
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
        max_y = ch_to_show
        # Adjust the main plot to make room for the sliders
        ax_ch = fig.add_axes([0.86, 0.15, 0.02, 0.73])
        # Define the max value of slider
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
        __plot_condition_shades(axes, conditions_dict, min_val, max_val)

    #  Call the aux function to plot vertical lines to mark the events
    if events_dict is not None:
        __plot_events_lines(axes, events_dict, min_val, max_val)

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

    if show:
        plt.show()
    return fig, axes

if __name__ == "__main__":
    """ Example of use: """
    fs = 256
    T = 60
    t = np.arange(0, T, 1 / fs)
    l_cha = ['F7', 'F3', 'FZ', 'F4', 'F8', 'FCz', 'C3', 'CZ', 'C4', 'CPz', 'P3',
             'PZ', 'P4',
             'PO7', 'POZ', 'PO8']
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
    e_dict = {'events': {'event_1': {'desc_name': 'Event 1', 'label': 0},
                         'event_2': {'desc_name': 'Event 2', 'label': 1}},
              'event_labels': [0, 1, 1, 1, 0, 1, 0, 0, 1, 0],
              'event_times': [0, 14, 15.4, 28, 2, 35, 42, 49, 53, 58.5]}

    c_dict = {'conditions': {'con_1': {'desc_name': 'Condition 1', 'label': 0},
                             'con_2': {'desc_name': 'Condition 2', 'label': 1}},
              'condition_labels': [0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
              'condition_times': [0, 14, 14, 28, 28, 35, 35, 50, 50, 60]}

    # Initialize TimePlot instance
    time_plot(signal=signal,fs=fs,ch_labels=l_cha,time_to_show=None,
              ch_to_show=None,channel_offset=None,conditions_dict=None,
              events_dict=None,show_epoch_lines=True,show=True)
