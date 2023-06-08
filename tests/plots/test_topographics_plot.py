from medusa.plots import head_plots
from medusa.meeg import EEGChannelSet
import numpy as np


def test_plot_channel_set():
    # TOPOGRAPHIC PLOT DEMO
    cha_set = EEGChannelSet()
    cha_set.set_standard_montage(l_reference=None, montage='10-20')
    values = np.random.rand(len(cha_set.channels))
    topographic_plots.plot_topography(cha_set, values, plot_clabels=True,
                                      plot_contour_ch=True)