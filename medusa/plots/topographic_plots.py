"""

In this module, you will find some functions to represent connectivity graphs
and topographic plots over a 2D head model. Enjoy!

@authors: Víctor Martínez-Cagigal and Diego Marcos-Martínez
"""

# External imports
import warnings
import scipy.interpolate as sp
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import cm, colors
import numpy as np

# Medusa imports
from medusa.meeg import UnlocatedChannel


def plot_connectivity(channel_set, adj_mat, head_radius=0.7266,
                      plot_channels=True, plot_skin_in_color=True,
                      plot_clabels=True,plot_contour_ch=False,
                      chcontour_radius=None,interp_points=500,
                      cmap='seismic', show=True, clim=None):

    """ This function depicts a connectivity map over the
        desired channel locations.

        Parameters
        ----------
        channel_set : eeg_standards.EEGChannelSet
            EEG channel set according of class eeg_standards.EEGChannelSet
        adj_mat: numpy.ndarray
            Numpy array with the connectivity values. It must be of the must
            have the following dimensions [n_channels, n_channels]
        head_radius : float
            Head radius. Default is 0.7266, coinciding with FPz. The nasion and
            inion are located at r=1.0
        plot_channels: bool
            Boolean that controls if the channel points should be plotted (default:
            True)
        plot_skin_in_color: bool
            Boolean that controls if the skin of the head should be coloured (default:
            False)
        plot_clabels: bool (Optional)
            Boolean that controls if the channel labels should be plotted (default:
            False)
        plot_contour_ch: bool (Optional)
            Boolean that controls if a contour around each channel should be plotted
            (default: False)
        chcontour_radius: float or None
            Radius of the channel contour if plot_contour_ch is set True. If None
            value, an automatic value is computed, considering the minimum distance
            between channels (default: None)
        interp_points: int (Optional)
            No. interpolation points. The lower N, the lower resolution and faster
            computation (default: 500)
        cmap : str
            Matplotlib colormap
        show : bool
            Show matplotlib figure
        clim : list or None
            Color bar limits. Index 0 contain the lower limit, whereas index 1 must
            contain the upper limit. if none, min and max values are used

        Returns
        -------
        figure : plt.figure
            Figure with the connectivity plot
        """

    # Check adjacency matrix  dimensions
    if adj_mat.shape[0] != len(channel_set.channels):
        raise Exception('Adjacency matrix must have the shape '
                        '[n_channels, n_channels]')

    # Get connectivity values
    values_indx = np.tril_indices(adj_mat.shape[0],1)
    conn_values = adj_mat[values_indx]

    # Map connectivity values to colors
    if clim is None:
        clim = [conn_values.min(),conn_values.max()]
    norm = colors.Normalize(vmin=clim[0],vmax=clim[1],clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    conn_colors = mapper.to_rgba(conn_values)

    # Connectivity line widths
    widths = 3 * (np.ones(len(conn_values)) * np.abs(conn_values) - clim[0])/\
             (clim[1] - clim[0])

    ch_x, ch_y = get_cartesian_coordinates(channel_set)

    fig, axes = plot_head(channel_set=channel_set, head_radius=head_radius,
                          plot_channels=plot_channels,
                          plot_skin_in_color=plot_skin_in_color,
                          plot_clabels=plot_clabels,
                          plot_contour_ch=plot_contour_ch,
                          chcontour_radius=chcontour_radius,
                          interp_points=interp_points,
                          show=False)

    edges = []
    for indx, chx in enumerate(ch_x):
        for second_indx in range(indx + 1, len(ch_x)):
            edges.append(
                [[chx, ch_y[indx]], [ch_x[second_indx], ch_y[second_indx]]])

    edges_collection = LineCollection(edges,colors=conn_colors,linewidths=widths)
    axes.add_collection(edges_collection)

    # Depict color bar
    cbar = plt.colorbar(mapper)

    fig.patch.set_alpha(0.0)
    if show is True:
        plt.show(dpi=400)
    return fig, axes


def plot_topography(channel_set, values, head_radius=0.7266,
                    plot_extra=0.29, k=3, make_contour=True, plot_channels=True,
                    plot_skin_in_color=False, plot_clabels=False,
                    plot_contour_ch=False, chcontour_radius=None,
                    interp_points=500, cmap='YlGnBu_r', show=True, clim=None,
                    axes=None, fig=None, show_colorbar=True, linewidth=4.0,
                    background=False):

    """ This function depicts a topographic map of the scalp
    over the desired channel locations.

    Parameters
    ----------
    channel_set : eeg_standards.EEGChannelSet
        EEG channel set according of class eeg_standards.EEGChannelSet
    values: list or numpy.ndarray
        Numpy array with the channel values. It must be of the same size as
        channels.
    head_radius : float
        Head radius. Default is 0.7266, coinciding with FPz. The nasion and
        inion are located at r=1.0
    plot_extra : float
        Extra radius of the plot surface
    k : int
        Number of nearest neighbors for interpolation
    make_contour: bool
        Boolean that controls if the contour lines should be plotted (default:
        True)
    plot_channels: bool
        Boolean that controls if the channel points should be plotted (default:
        True)
    plot_skin_in_color: bool (Optional)
        Boolean that controls if the skin of the head should be coloured (default:
        False)
    plot_clabels: bool (Optional)
        Boolean that controls if the channel labels should be plotted (default:
        False)
    plot_contour_ch: bool (Optional)
        Boolean that controls if a contour around each channel should be plotted
        (default: False)
    chcontour_radius: float or None
        Radius of the channel contour if plot_contour_ch is set True. If None
        value, an automatic value is computed, considering the minimum distance
        between channels (default: None)
    interp_points: int (Optional)
        No. interpolation points. The lower N, the lower resolution and faster
        computation (default: 500)
    cmap : str
        Matplotlib colormap
    background: bool (Optional)
        Set background
    show : bool
        Show matplotlib figure
    axes : matplotlib.pyplot.axes
        If a matplotlib axes are specified, the plot is displayed inside it.
        Otherwise, the plot will generate a new axes.
    clim : list or None
        Color bar limits. Index 0 contain the lower limit, whereas index 1 must
        contain the upper limit. if none, min and max values are used

    Returns
    -------
    figure : plt.figure
        Figure with the topography plot
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

    # Plot head
    fig, axes = plot_head(channel_set=channel_set,head_radius=head_radius,
                          plot_channels=plot_channels,
                          plot_skin_in_color=plot_skin_in_color,
                          plot_clabels=plot_clabels,
                          plot_contour_ch=plot_contour_ch,
                          chcontour_radius=chcontour_radius,
                          interp_points=interp_points,
                          show=False, axes=axes, fig=fig, linewidth=linewidth,
                          background=background)

    # Create points out of the head to get a natural interpolation
    r_ext_points = 1.5  # Radius of the virtual electrodes
    no_ve = 16  # No. virtual electrodes
    add_x, add_y = pol2cart(r_ext_points * np.ones((1, no_ve)),
                            np.arange(0, 2 * np.pi, 2 * np.pi / no_ve))
    linear_grid = np.linspace(-r_ext_points, r_ext_points, interp_points)
    interp_x, interp_y = np.meshgrid(linear_grid, linear_grid)

    # Get cartesian coordinates
    ch_x, ch_y = get_cartesian_coordinates(channel_set)

    # Create the mask
    mask_radius = np.max(np.sqrt(ch_x**2+ch_y**2)) + plot_extra
    mask = (np.sqrt(np.power(interp_x, 2) +
                    np.power(interp_y, 2)) < mask_radius)

    # Interpolate the data
    ch_x = ch_x.reshape(ch_x.shape[0], 1)
    ch_y = ch_y.reshape(ch_y.shape[0], 1)
    add_values = compute_nearest_values(np.hstack((add_x.T, add_y.T)),
                                        np.hstack((ch_x, ch_y)), values, k)
    grid_points = np.hstack((np.vstack((ch_x, add_x.T)),
                             np.vstack((ch_y, add_y.T))))
    grid_values = np.vstack((values.T, add_values))
    interp_values = np.vstack((interp_x.ravel(), interp_y.ravel())).T
    interp_z = sp.griddata(grid_points, grid_values, interp_values, 'cubic')

    # Mask the data
    interp_z = np.reshape(interp_z, (interp_points, interp_points))
    interp_z[~mask] = float('nan')

    # Plotting the final interpolation
    p_interp = axes.pcolor(interp_x, interp_y, interp_z, cmap=cmap)
    if clim is not None:
        p_interp.set_clim(clim[0], clim[1])
    if show_colorbar:
        cbar = plt.colorbar(p_interp)

    # Plotting the contour
    if make_contour:
        axes.contour(interp_x, interp_y, interp_z, alpha=1, colors='0.2',
                     linewidths=linewidth/4.2)

    if show is True:
        plt.show(dpi=400)
    return fig, axes, p_interp


def plot_head(channel_set, head_radius=0.7266, plot_channels=True,
              plot_skin_in_color=False, plot_clabels=False,
              plot_contour_ch=False, chcontour_radius=None,
              interp_points=500, show=True, axes=None, fig=None,
              linewidth=4.0, background=False):
    """This function depicts a two-dimensional head diagram.

    Parameters
    ----------
    channel_set : eeg_standards.EEGChannelSet
        EEG channel set according of class eeg_standards.EEGChannelSet
    head_radius : float
        Head radius. Default is 0.7266, coinciding with FPz. The nasion and
        inion are located at r=1.0
    plot_channels: bool
        Boolean that controls if the channel points should be plotted (default:
        True)
    plot_skin_in_color: bool (Optional)
        Boolean that controls if the skin of the head should be coloured (default:
        False)
    plot_clabels: bool (Optional)
        Boolean that controls if the channel labels should be plotted (default:
        False)
    plot_contour_ch: bool (Optional)
        Boolean that controls if a contour around each channel should be plotted
        (default: False)
    chcontour_radius: float or None
        Radius of the channel contour if plot_contour_ch is set True. If None
        value, an automatic value is computed, considering the minimum distance
        between channels (default: None)
    interp_points: int (Optional)
        No. interpolation points. The lower N, the lower resolution and faster
        computation (default: 500)
    background: bool (Optional)
        Set background
    show : bool
        Show matplotlib figure
    axes : matplotlib.pyplot.axes
        If a matplotlib axes are specified, the plot is displayed inside it.
        Otherwise, the plot will generate a new figure.
    Returns
    -------
    figure : plt.figure
        Figure with the head plot
    """
    # Check channels errors
    if channel_set.dim != '2D':
        raise ValueError('The channel set must have 2 dimensions')

    # Initialize figure and axis
    if fig is None:
        fig = plt.figure()
    if axes is None:
        axes = fig.add_subplot(111)

    # Compute the cartesian coordinates of each channel
    ch_x, ch_y = get_cartesian_coordinates(channel_set)

    # Plotting the nose
    head_rho = head_radius
    nt = 0.15  # Half-nose width (in percentage of pi/2)
    nr = 0.22  # Nose length (in radius units)
    nose_rho = [head_rho, head_rho + head_rho * nr, head_rho]
    nose_theta = [(np.pi / 2) + (nt * np.pi / 2), np.pi / 2,
                  (np.pi / 2) - (nt * np.pi / 2)]
    nose_x = nose_rho * np.cos(nose_theta)
    nose_y = nose_rho * np.sin(nose_theta)
    axes.plot(nose_x, nose_y, 'k', linewidth=linewidth)
    if plot_skin_in_color:
        axes.fill(nose_x, nose_y, 'k',
                  facecolor='#E8BEAC', edgecolor='k', linewidth=linewidth)

    # Plotting the ears as ellipses
    ellipse_a = 0.08  # Horizontal eccentricity
    ellipse_b = 0.16  # Vertical eccentricity
    ear_angle = 0.9 * np.pi / 8  # Mask angle
    offset = 0.058 * head_radius  # Ear offset
    ear_theta_right = np.linspace(-np.pi / 2 - ear_angle,
                                  np.pi / 2 + ear_angle, interp_points)
    ear_theta_left = np.linspace(np.pi / 2 - ear_angle,
                                 3 * np.pi / 2 + ear_angle, interp_points)
    ear_x_right = ear_rho(ear_theta_right, ellipse_a, ellipse_b) * \
                  np.cos(ear_theta_right)
    ear_y_right = ear_rho(ear_theta_right, ellipse_a, ellipse_b) * \
                  np.sin(ear_theta_right)
    ear_x_left = ear_rho(ear_theta_left, ellipse_a, ellipse_b) * \
                 np.cos(ear_theta_left)
    ear_y_left = ear_rho(ear_theta_left, ellipse_a, ellipse_b) * \
                 np.sin(ear_theta_left)
    axes.plot(ear_x_right + head_rho + offset, ear_y_right, 'k', linewidth=linewidth)
    axes.plot(ear_x_left - head_rho - offset, ear_y_left, 'k', linewidth=linewidth)

    # Plotting the head limits as a circle
    head_theta = np.linspace(0, 2 * np.pi, interp_points)
    head_x = head_rho * np.cos(head_theta)
    head_y = head_rho * np.sin(head_theta)
    axes.plot(head_x, head_y, 'k', linewidth=linewidth)
    if plot_skin_in_color:
        axes.fill(head_x, head_y, facecolor='#E8BEAC',
                  edgecolor='k', linewidth=4)

    if plot_skin_in_color:
        axes.fill(ear_x_right + head_rho + offset, ear_y_right,
                  facecolor='#E8BEAC', edgecolor='k', linewidth=linewidth)
        axes.fill(ear_x_left - head_rho - offset, ear_y_left,
                  facecolor='#E8BEAC', edgecolor='k', linewidth=linewidth)

    # Compute optimal minimum distance between channels
    if chcontour_radius is None:
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
        min_dist = min_dist * percentage

    else:
        min_dist = chcontour_radius

    # Plot a contour around electrodes
    if plot_contour_ch:
        for ch_idx in range(len(channel_set.channels)):
            axes.add_patch(plt.Circle(
                (ch_x[ch_idx], ch_y[ch_idx]), radius=min_dist,
                facecolor='#ffffff', edgecolor=None, alpha=0.4, zorder=10))

    # Plotting the electrodes
    if plot_channels:
        axes.scatter(ch_x, ch_y, linewidth*3.5, facecolors='w', edgecolors='k',
                     zorder=10)

    if plot_clabels:
        for t in range(len(channel_set.channels)):
            axes.text(ch_x[t] + 0.01, ch_y[t] - 0.85 * min_dist,
                      channel_set.channels[t]['label'], fontsize=linewidth*2,
                      color='w',
                      zorder=11)

    # Last considerations
    plot_lim = max(head_radius + 0.2, np.max(np.sqrt(ch_x**2 + ch_y**2)) + 0.2)
    axes.set_xlim([-plot_lim, plot_lim])
    axes.set_ylim([-plot_lim, plot_lim])
    axes.set_aspect('equal', 'box')
    axes.axis('off')
    if fig is not None:
        # fig = plt.gcf()
        fig.patch.set_alpha(background)  # Set transparent background
        fig.tight_layout()
    if show is True:
        plt.show(dpi=400)

    return fig, axes


def ear_rho(ear_theta, ellipse_a, ellipse_b):
    """  This function computes the ear coordinates according to an ellipse.
    """
    d1 = np.power(np.cos(ear_theta), 2) / np.power(ellipse_a, 2)
    d2 = np.power(np.sin(ear_theta), 2) / np.power(ellipse_b, 2)
    return 1 / np.sqrt(d1 + d2)


def pol2cart(rho, phi):
    """This function converts polar coordinates to cartesian coordinates.

    Parameters
    ----------
    rho:     Array of radii
    phi:     Array of angles
    """
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


def compute_nearest_values(coor_add, coor_neigh, val_neigh, k):
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

def get_cartesian_coordinates(channel_set):
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
        ch_x, ch_y = pol2cart(radius, theta)
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

if __name__ == "__main__":
    """ Example of use: """
    from medusa.meeg.meeg import EEGChannelSet

    # Set channel set
    channel_set = EEGChannelSet()
    channel_set.set_standard_montage(
        l_cha=['F3','F7','FZ', 'F4','F8', 'FCZ','C3', 'CZ', 'C4','CPZ', 'P3',
               'PZ', 'P4','PO7','POZ','PO8'],
        standard='10-10')

    # Plot topographic plot
    plt.figure()
    dummy_values_topo = np.arange(len(channel_set.channels))
    plot_topography(channel_set, dummy_values_topo, plot_clabels=True,
                    plot_contour_ch=True, plot_extra=0.1,
                    plot_skin_in_color=True,cmap='plasma')

    # Plot connectivity plot
    plt.figure()
    dummy_values_conn = np.random.randn(16,16)
    plot_connectivity(channel_set,dummy_values_conn)
