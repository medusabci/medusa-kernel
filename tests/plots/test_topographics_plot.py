from medusa.plots import head_plots
from medusa.meeg import EEGChannelSet
import numpy as np
from matplotlib import pyplot as plt


def test_plot_channel_set():
    # TOPOGRAPHIC PLOT DEMO
    cha_set = EEGChannelSet()
    cha_set.set_standard_montage(l_reference=None, montage='10-20')
    values = np.random.rand(len(cha_set.channels))

    # Create figure and axes
    fig = plt.figure()
    fig.add_subplot(1, 1, 1)

    # Create topographic plot
    topo = head_plots.TopographicPlot(
        axes=fig.axes[0],
        channel_set=cha_set,
        interpolate=True,
        plot_channel_points=True,
        plot_channel_labels=True,
        interp_contour_width=0.8
    )
    topo.update(values=values)
    plt.close(fig)
