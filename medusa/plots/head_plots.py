"""

In this module, you will find some functions to represent connectivity graphs
and topographic plots over a 2D head model. Enjoy!

"""

# External imports
import warnings
import scipy.interpolate as sp
import matplotlib
from matplotlib.collections import LineCollection
from matplotlib import cm, colors
import numpy as np

# Medusa imports
from medusa.meeg import UnlocatedChannel


class TopographicPlot:
    """ Helper function to use a Topographic plot.

    Parameters
    ------------
    axes : matplotlib.Axes.axes
        Matplotlib axes in which the head will be displayed into.
    channel_set : eeg_standards.EEGChannelSet
        EEG channel set.
    **kwargs : **dict() (Optional)
        Settings for the topoplot. Refer to plot_topography and plot_head to
        check the different parameters.
    """
    def __init__(self, axes, channel_set, **kwargs):
        self.axes = axes
        self.channel_set = channel_set
        self.kwargs = kwargs

        # Init
        self.head_handles = plot_head(
            axes=self.axes, channel_set=self.channel_set, **self.kwargs
        )
        self.plot_handles = None

    def update(self, values):
        """ Use this function to update the topographic plot in real-time.

        Parameters
        ------------
        values: list or numpy.ndarray
            Numpy array with the channel values. It must be of the same size as
            channels.
        """
        if self.plot_handles is not None:
            _remove_handles(self.plot_handles)
        self.plot_handles = plot_topography(
            values=values, axes=self.axes, channel_set=self.channel_set,
            **self.kwargs
        )

    def clear(self):
        """ This method clears all the handles. """
        if self.plot_handles is not None:
            _remove_handles(self.plot_handles)
        if self.head_handles is not None:
            _remove_handles(self.head_handles)


class ConnectivityPlot:
    """ Helper function to use a Connectivity topographic plot.

    Parameters
    ------------
    axes : matplotlib.Axes.axes
        Matplotlib axes in which the head will be displayed into.
    channel_set : eeg_standards.EEGChannelSet
        EEG channel set.
    **kwargs : **dict() (Optional)
        Settings for the topoplot. Refer to plot_topography and plot_head to
        check the different parameters.
    """
    def __init__(self, axes, channel_set, **kwargs):
        self.axes = axes
        self.channel_set = channel_set
        self.kwargs = kwargs

        # Init
        self.head_handles = plot_head(
            axes=self.axes, channel_set=self.channel_set, **self.kwargs
        )
        self.plot_handles = None

    def update(self, adj_mat):
        """ Use this function to update the topographic plot in real-time.

         Parameters
        ------------
        adj_mat: numpy.ndarray
            Numpy array with the connectivity values. It must be one of the
            following dimensions [n_channels, n_channels]
        """
        if self.plot_handles is not None:
            _remove_handles(self.plot_handles)
        self.plot_handles = plot_connectivity(
            adj_mat=adj_mat, axes=self.axes, channel_set=self.channel_set,
            **self.kwargs
        )

    def clear(self):
        """ This method clears all the handles. """
        if self.plot_handles is not None:
            _remove_handles(self.plot_handles)
        if self.head_handles is not None:
            _remove_handles(self.head_handles)


def plot_connectivity(adj_mat, axes, channel_set, cmap="bwr", clim=None,
                      **kwargs):

    """This function depicts a connectivity map over the desired channel
    locations.

    Parameters
    ----------
    adj_mat: numpy.ndarray
        Numpy array with the connectivity values. It must be one of the
        following dimensions [n_channels, n_channels]
    axes : matplotlib.Axes.axes
        Matplotlib axes in which the head will be displayed into.
    channel_set : eeg_standards.EEGChannelSet
        EEG channel set.
    cmap : str (Optional)
        Matplotlib colormap.
    clim : list or None (Optional)
        Color bar limits. Index 0 contain the lower limit, whereas index 1 must
        contain the upper limit. if None, min and max values are used.

    Returns
    -------
    handles : dict
        Dict with all the handles that have been added to the axes.
    """

    # Check adjacency matrix  dimensions
    if adj_mat.shape[0] != len(channel_set.channels):
        raise Exception('Adjacency matrix must have the shape '
                        '[n_channels, n_channels]')

    # Init handles
    handles = dict()

    # Get connectivity values
    values_indx = np.triu_indices(adj_mat.shape[0],1)
    conn_values = adj_mat[values_indx]

    # Map connectivity values to colors
    if clim is None:
        clim = [conn_values.min(), conn_values.max()]
    norm = colors.Normalize(vmin=clim[0], vmax=clim[1], clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    conn_colors = mapper.to_rgba(conn_values)

    # Connectivity line widths
    widths = 3 * (np.ones(len(conn_values)) * np.abs(conn_values) - clim[0])/\
             (clim[1] - clim[0])

    ch_x, ch_y = __get_cartesian_coordinates(channel_set)

    edges = []
    for indx, chx in enumerate(ch_x):
        for second_indx in range(indx + 1, len(ch_x)):
            edges.append(
                [[chx, ch_y[indx]], [ch_x[second_indx], ch_y[second_indx]]])

    edges_collection = LineCollection(edges, colors=conn_colors,
                                      linewidths=widths)
    handles['lines'] = axes.add_collection(edges_collection)

    return handles


def plot_topography(values, axes, channel_set, extra_radius=0.29,
                    interp_neighbors=3, interp_points=500,
                    interp_contour_width=0.8, cmap="YlGnBu_r", clim=None,
                    **kwargs):

    """ This function depicts a topographic map of the scalp
    over the desired channel locations.

    Parameters
    ----------
    values: list or numpy.ndarray
        Numpy array with the channel values. It must be of the same size as
        channels.
    axes : matplotlib.Axes.axes
        Matplotlib axes in which the head will be displayed into.
    channel_set : eeg_standards.EEGChannelSet
        EEG channel set
    extra_radius : float (Optional)
        Extra radius of the plot surface.
    interp_neighbors : int (Optional)
        Number of nearest neighbors for interpolation.
    interp_points: int (Optional)
        No. interpolation points. The lower N, the lower resolution and faster
        computation (default: 500)
    interp_contour_width: float or None (Optional)
        Line width of the contour lines. If None, no contour lines will be
        plotted (default: 0.8).
    cmap : str (Optional)
        Matplotlib colormap.
    clim : tuple or None (Optional)
        Color bar limits. Index 0 contain the lower limit, whereas index 1 must
        contain the upper limit. if None, min and max values are used.

    Returns
    -------
    handles : dict
        Dict with all the handles that have been added to the axes.
    """
    # Check values dimensions
    values = np.array(values)
    if values.size != len(channel_set.channels):
        raise Exception('Parameters ch_list and values must have the same '
                        'size')
    if len(values.shape) == 1:
        # Reshape to the correct dimensions [1 x len(ch_list)]
        values = values.reshape(1, -1)
    elif len(values.shape) == 2:
        # Reshape to the correct dimensions [1 x len(ch_list)]
        values = np.squeeze(values).reshape(1, -1)
    else:
        raise Exception('The dimensions of the parameter are not correct')

    # Init handles
    handles = dict()

    # Create points out of the head to get a natural interpolation
    r_ext_points = 1.5  # Radius of the virtual electrodes
    no_ve = 16  # No. virtual electrodes
    add_x, add_y = __pol2cart(r_ext_points * np.ones((1, no_ve)),
                            np.arange(0, 2 * np.pi, 2 * np.pi / no_ve))
    linear_grid = np.linspace(-r_ext_points, r_ext_points, interp_points)
    interp_x, interp_y = np.meshgrid(linear_grid, linear_grid)

    # Get cartesian coordinates
    ch_x, ch_y = __get_cartesian_coordinates(channel_set)

    # Create the mask
    mask_radius = np.max(np.sqrt(ch_x**2+ch_y**2)) + extra_radius
    mask = (np.sqrt(np.power(interp_x, 2) +
                    np.power(interp_y, 2)) < mask_radius)

    # Interpolate the data
    ch_x = ch_x.reshape(ch_x.shape[0], 1)
    ch_y = ch_y.reshape(ch_y.shape[0], 1)
    add_values = __compute_nearest_values(np.hstack((add_x.T, add_y.T)),
                                        np.hstack((ch_x, ch_y)), values,
                                        interp_neighbors)
    grid_points = np.hstack((np.vstack((ch_x, add_x.T)),
                             np.vstack((ch_y, add_y.T))))
    grid_values = np.vstack((values.T, add_values))
    interp_values = np.vstack((interp_x.ravel(), interp_y.ravel())).T
    interp_z = sp.griddata(grid_points, grid_values, interp_values, 'cubic')

    # Mask the data
    interp_z = np.reshape(interp_z, (interp_points, interp_points))
    interp_z[~mask] = float('nan')

    # Plotting the final interpolation
    color_mesh = axes.pcolormesh(interp_x, interp_y, interp_z, cmap=cmap)
    handles['color-mesh'] = color_mesh
    if clim is not None:
        color_mesh.set_clim(clim[0], clim[1])

    # Plotting the contour
    if interp_contour_width is not None:
        contour = axes.contour(interp_x, interp_y, interp_z, alpha=1,
                               colors='0.2', linewidths=interp_contour_width)
        handles['contour'] = contour

    return handles


def plot_head(axes, channel_set, head_radius=0.76266, head_line_width=4.0,
              head_skin_color="#E8BEAC", plot_channel_labels=False,
              plot_channel_points=True, channel_radius_size=None, **kwargs):
    """This function depicts a two-dimensional head diagram.

    Parameters
    ----------
    axes : matplotlib.Axes.axes
        Matplotlib axes in which the head will be displayed into.
    channel_set : eeg_standards.EEGChannelSet
        EEG channel set.
    head_radius : float (Optional)
        Head radius. Default is 0.7266, coinciding with FPz. The nasion and
        inion are located at r=1.0
    head_line_width : float (Optional)
        Line width for the head, ears and nose.
    head_skin_color : basestring or None (Optional)
        If None, skin will be transparent. Otherwise, skin will be colored.
    plot_channel_labels : bool (Optional)
        Boolean that controls if the channel labels should be plotted (default:
        False)
    plot_channel_points : bool (Optional)
       Boolean that controls if the channel points should be plotted (default:
        True)
    channel_radius_size : float (Optional)
        Channels can be surrounded by a circunference to ease their
        visualization. Use this parameter to control the radius of the
        circle. If 0, no circle will be used; if None, an automatic value
        will be computed considering the minimum distance between channels
        (default: None)

    Returns
    -------
    handles : dict
        Dict with all the handles that have been added to the axes
    """

    # Check channels errors
    if channel_set.dim != '2D':
        raise ValueError('The channel set must have 2 dimensions')

    # Init handles
    handles = dict()

    # Compute the cartesian coordinates of each channel
    ch_x, ch_y = __get_cartesian_coordinates(channel_set)

    # Plotting the nose
    head_rho = head_radius
    nt = 0.15  # Half-nose width (in percentage of pi/2)
    nr = 0.22  # Nose length (in radius units)
    nose_rho = [head_rho, head_rho + head_rho * nr, head_rho]
    nose_theta = [(np.pi / 2) + (nt * np.pi / 2), np.pi / 2,
                  (np.pi / 2) - (nt * np.pi / 2)]
    nose_x = nose_rho * np.cos(nose_theta)
    nose_y = nose_rho * np.sin(nose_theta)
    handle = axes.plot(nose_x, nose_y, 'k', linewidth=head_line_width)
    handles['nose-line'] = handle[0]
    if head_skin_color is not None:
        handle = axes.fill(nose_x, nose_y, 'k', facecolor=head_skin_color,
                           edgecolor='k', linewidth=head_line_width)
        handles['nose-fill'] = handle[0]

    # Plotting the ears as ellipses
    interp_points = 500
    ellipse_a = 0.08  # Horizontal eccentricity
    ellipse_b = 0.16  # Vertical eccentricity
    ear_angle = 0.9 * np.pi / 8  # Mask angle
    offset = 0.058 * head_radius  # Ear offset
    ear_theta_right = np.linspace(-np.pi / 2 - ear_angle,
                                  np.pi / 2 + ear_angle, interp_points)
    ear_theta_left = np.linspace(np.pi / 2 - ear_angle,
                                 3 * np.pi / 2 + ear_angle, interp_points)
    ear_x_right = __ear_rho(ear_theta_right, ellipse_a, ellipse_b) * \
                  np.cos(ear_theta_right)
    ear_y_right = __ear_rho(ear_theta_right, ellipse_a, ellipse_b) * \
                  np.sin(ear_theta_right)
    ear_x_left = __ear_rho(ear_theta_left, ellipse_a, ellipse_b) * \
                 np.cos(ear_theta_left)
    ear_y_left = __ear_rho(ear_theta_left, ellipse_a, ellipse_b) * \
                 np.sin(ear_theta_left)
    handle = axes.plot(ear_x_right + head_rho + offset,
                       ear_y_right, 'k', linewidth=head_line_width)
    handles['right-ear-line'] = handle[0]
    handle = axes.plot(ear_x_left - head_rho - offset,
                       ear_y_left, 'k', linewidth=head_line_width)
    handles['left-ear-line'] = handle[0]

    # Plotting the head limits as a circle
    head_theta = np.linspace(0, 2 * np.pi, interp_points)
    head_x = head_rho * np.cos(head_theta)
    head_y = head_rho * np.sin(head_theta)
    handle = axes.plot(head_x, head_y, 'k', linewidth=head_line_width)
    handles['head-line'] = handle[0]
    if head_skin_color is not None:
        handle = axes.fill(head_x, head_y, facecolor=head_skin_color,
                           edgecolor='k', linewidth=head_line_width)
        handles['head-fill'] = handle[0]

    if head_skin_color is not None:
        handle = axes.fill(ear_x_right + head_rho + offset, ear_y_right,
                           facecolor=head_skin_color, edgecolor='k',
                           linewidth=head_line_width)
        handles['right-ear-fill'] = handle[0]
        handle = axes.fill(ear_x_left - head_rho - offset, ear_y_left,
                           facecolor=head_skin_color, edgecolor='k',
                           linewidth=head_line_width)
        handles['left-ear-fill'] = handle[0]

    # Compute optimal minimum distance between channels
    if channel_radius_size is None:
        dist_matrix = channel_set.compute_dist_matrix()
        dist_matrix.sort()
        min_dist = dist_matrix[:, 1].min()

        #  Adjust radius
        if isinstance(channel_set.montage, str):
            if channel_set.montage == '10-05':
                M = 345
            elif channel_set.montage == '10-10':
                M = 71
            elif channel_set.montage == '10-20':
                M = 21
        elif isinstance(channel_set.montage, dict) or channel_set.montage\
                is None:
            M = channel_set.n_cha
        percentage = len(channel_set.channels) * (0.25 / (M - 2)) + \
                     0.25 * ((M - 4) / (M - 2))
        channel_radius_size = min_dist * percentage

    # Plot channels as circunferences
    if channel_radius_size != 0:
        handles['ch-contours'] = list()
        for ch_idx in range(len(channel_set.channels)):
            patch = matplotlib.patches.Circle(
                (ch_x[ch_idx], ch_y[ch_idx]), radius=channel_radius_size,
                facecolor='#ffffff', edgecolor=None, alpha=0.4, zorder=10)
            handle = axes.add_patch(patch)
            handles['ch-contours'].append(handle)

    # Plot channels points
    if plot_channel_points:
        handle = axes.scatter(ch_x, ch_y, head_line_width*3.5, facecolors='w',
                              edgecolors='k', zorder=10)
        handles['ch-points'] = handle

    # Plot channels labels
    if plot_channel_labels:
        handles['ch-labels'] = list()
        for t in range(len(channel_set.channels)):
            handle = axes.text(ch_x[t] + 0.01, ch_y[t] - 0.85 * channel_radius_size,
                               channel_set.channels[t]['label'],
                               fontsize=head_line_width*2, color='w', zorder=11)
            handles['ch-labels'].append(handle)

    # Last considerations
    plot_lim = max(head_radius + 0.2, np.max(np.sqrt(ch_x**2 + ch_y**2)) + 0.2)
    axes.set_xlim([-plot_lim, plot_lim])
    axes.set_ylim([-plot_lim, plot_lim])
    axes.set_aspect('equal', 'box')
    axes.set_yticklabels([])
    axes.set_xticklabels([])
    for side in ['top', 'right', 'bottom', 'left']:
        axes.spines[side].set_visible(False)
    axes.tick_params(axis='both', which='both', bottom=False, left=False)
    axes.set_facecolor('#00000000')         # Transparent

    return handles


def __ear_rho(ear_theta, ellipse_a, ellipse_b):
    """  This function computes the ear coordinates according to an ellipse.
    """
    d1 = np.power(np.cos(ear_theta), 2) / np.power(ellipse_a, 2)
    d2 = np.power(np.sin(ear_theta), 2) / np.power(ellipse_b, 2)
    return 1 / np.sqrt(d1 + d2)


def __pol2cart(rho, phi):
    """This function converts polar coordinates to cartesian coordinates.

    Parameters
    ----------
    rho:     Array of radii
    phi:     Array of angles
    """
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


def __compute_nearest_values(coor_add, coor_neigh, val_neigh, k):
    """ This function computes the mean values of the k-nearest neighbors.

    Parameters
    ----------
    coor_add:    XY coordinates of the virtual electrodes.
    coor_neigh:  XY coordinates of the real electrodes.
    val_neigh:   Values of the real electrodes.
    k:           Number of neighbors to consider.
    """
    add_val = np.empty((len(coor_add), 1))
    L = len(coor_add)

    for i in range(L):
        # Distances between the added electrode and the original ones
        target = coor_add[i, :] * np.ones((len(coor_neigh), 2))
        d = np.sqrt(np.sum(np.power(target - coor_neigh, 2), axis=1))

        # K-nearest neighbors
        idx = np.argsort(d)
        sel_idx = idx[1:1 + k]

        # Final value as the mean value of the k-nearest neighbors
        add_val[i] = np.mean(val_neigh[0, sel_idx])
    return add_val


def __get_cartesian_coordinates(channel_set):
    # Restructure the channels list to treat it more easily
    if channel_set.coord_system == 'spherical':
        radius = list()
        theta = list()
        for c in channel_set.channels:
            try:
                radius.append(c['r'])
                theta.append(c['theta'])
            except KeyError as e:
                raise UnlocatedChannel(c)
        radius, theta = np.array(radius),np.array(theta)
        ch_x, ch_y = __pol2cart(radius, theta)
    else:
        ch_x, ch_y = list(),list()
        for c in channel_set.channels:
            try:
                ch_x.append(c['x'])
                ch_y.append(c['y'])
            except KeyError as e:
                raise UnlocatedChannel(c)
        ch_x,ch_y = np.array(ch_x),np.array(ch_y)
    return ch_x, ch_y


def _remove_handles(handles):
    """ Utility function to remove all matplotlib handles. """
    for h in handles.values():
        if isinstance(h, list):
            for h2 in h:
                h2.remove()
        else:
            if isinstance(h, matplotlib.contour.QuadContourSet):
                for h2 in h.collections:
                    h2.remove()
            else:
                h.remove()


if __name__ == "__main__":
    """ Example of use: """
    from matplotlib import pyplot as plt
    from medusa.meeg.meeg import EEGChannelSet
    from medusa.plots.head_plots import *
    import numpy as np

    # Set channel set
    channel_set = EEGChannelSet()
    channel_set.set_standard_montage(standard='10-20')

    # Initialize figure
    fig = plt.figure()
    fig.add_subplot(1, 1, 1)

    # # Plot topography
    # values = np.random.random(channel_set.n_cha)
    # topo = TopographicPlot(axes=fig.axes[0], channel_set=channel_set)
    # topo.update(values=values)

    # Plot connectivity
    adj_mat = np.random.randn(channel_set.n_cha, channel_set.n_cha)
    conn = ConnectivityPlot(axes=fig.axes[0], channel_set=channel_set)
    conn.update(adj_mat=adj_mat)

    # Show figure
    fig.tight_layout()
    fig.set_alpha(0)
    fig.show()
