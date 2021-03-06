import numpy as np
import scipy.interpolate as sp
import matplotlib.pyplot as plt
from medusa.spatial_filtering import LaplacianFilter


def plot_topography(channel_set, values=None, head_radius=0.7266, plot_extra=0.29,
                    k=3, make_contour=True, plot_channels=True, plot_skin_in_color=False,
                    plot_clabels=False, plot_contour_ch=False, chcontour_radius=None,
                    interp_points=500, cmap='YlGnBu_r', show=True, clim=None):

    """The function 'plot_topography' depicts a topographical map of the scalp
    over the desired channel locations.

    Parameters
    ----------
    channel_set : eeg_standards.EEGChannelSet
        EEG channel set according of class eeg_standards.EEGChannelSet
    values: list or numpy.ndarray or None
        Numpy array with the channel values. It must be of the same size as
        channels. If None value, the function only returns a plot of the head
        and the channels
    head_radius : float
        Head radius. Default is 0.7266, coinciding with FPz. The nasion and
        inion are located at r=1.0
    plot_extra : float
        Extra radius of the plot surface
    k : int
        Number of nearest neighbors for interpolation
    make_contour: bool (Optional)
        Boolean that controls if the contour lines should be plotted (default:
        True)
    plot_channels: bool (Optional)
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
        Radius of the channel contour if plot_contour_ch is set True. If None value,
        an automatic value is computed, considering the minimum distance between
        channels (default: None)
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
        Figure with the topography plot
    """
    # Check channels errors
    if channel_set.dim != '2D':
        raise ValueError('The channel set must have 2 dimensions')
    if channel_set.coord_system != 'spherical':
        raise ValueError('The channel set must have polar coordinates')
    # Check values dimensions
    channels = channel_set.channels
    if values is not None:
        values = np.array(values)
        if values.size != len(channels):
            raise Exception('Parameters ch_list and values must have the same size')
        if len(values.shape) == 1:
            # Reshape to the correct dimensions [1 x len(ch_list)]
            values = values.reshape(1, -1)
        elif len(values.shape) == 2:
            # Reshape to the correct dimensions [1 x len(ch_list)]
            values = np.squeeze(values).reshape(1, -1)
        else:
            raise Exception('The dimensions of the parameter are not correct')

    # Initialize figure and axis
    fig = plt.figure()
    axes = fig.add_subplot(111)

    # Restructure the channels list to treat it more easily
    radius = np.array([c['r'] for c in channels])
    theta = np.array([c['theta'] for c in channels])
    # labels = [c['label'] for c in channels]

    # Compute the cartesian coordinates of each channel
    ch_x, ch_y = pol2cart(radius, theta)

    if values is not None:
        # Create points out of the head to get a natural interpolation
        r_ext_points = 1.5  # Radius of the virtual electrodes
        no_ve = 16  # No. virtual electrodes
        add_x, add_y = pol2cart(r_ext_points * np.ones((1, no_ve)),
                                np.arange(0, 2 * np.pi, 2 * np.pi / no_ve))
        linear_grid = np.linspace(-r_ext_points, r_ext_points, interp_points)
        interp_x, interp_y = np.meshgrid(linear_grid, linear_grid)

        # Create the mask
        # outer_rho = np.max(radius)
        # mask_radius = outer_rho + head_extra if outer_rho > head_radious else
        # head_radious
        mask_radius = np.max(radius) + plot_extra
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
        p_interp = plt.pcolor(interp_x, interp_y, interp_z, cmap=cmap)
        if clim is not None:
            plt.clim(clim[0], clim[1])
        cbar = plt.colorbar(p_interp)

        # Plotting the contour
        if make_contour:
            axes.contour(interp_x, interp_y, interp_z, alpha=1, colors='0.2',
                         linewidths=0.75)

    # Plotting the nose
    head_rho = head_radius
    nt = 0.15  # Half-nose width (in percentage of pi/2)
    nr = 0.22  # Nose length (in radius units)
    nose_rho = [head_rho, head_rho + head_rho * nr, head_rho]
    nose_theta = [(np.pi / 2) + (nt * np.pi / 2), np.pi / 2, (np.pi / 2) - (nt * np.pi / 2)]
    nose_x = nose_rho * np.cos(nose_theta)
    nose_y = nose_rho * np.sin(nose_theta)
    axes.plot(nose_x, nose_y, 'k', linewidth=4)
    if plot_skin_in_color:
        axes.fill(nose_x, nose_y, 'k', facecolor='#E8BEAC', edgecolor='k', linewidth=4)

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
    axes.plot(ear_x_right + head_rho + offset, ear_y_right, 'k', linewidth=4)
    axes.plot(ear_x_left - head_rho - offset, ear_y_left, 'k', linewidth=4)

    # Plotting the head limits as a circle
    head_theta = np.linspace(0, 2 * np.pi, interp_points)
    head_x = head_rho * np.cos(head_theta)
    head_y = head_rho * np.sin(head_theta)
    axes.plot(head_x, head_y, 'k', linewidth=4)
    if plot_skin_in_color:
        axes.fill(head_x, head_y, facecolor='#E8BEAC', edgecolor='k', linewidth=4)

    if plot_skin_in_color:
        axes.fill(ear_x_right + head_rho + offset, ear_y_right, facecolor='#E8BEAC', edgecolor='k', linewidth=4)
        axes.fill(ear_x_left - head_rho - offset, ear_y_left, facecolor='#E8BEAC', edgecolor='k', linewidth=4)

    # Plot a contour around electrodes
    if plot_contour_ch:
        if chcontour_radius is None:
            dist_matrix = channel_set.compute_dist_matrix()
            dist_matrix.sort()
            min_dist = dist_matrix[:, 1].min()
        else:
            min_dist = chcontour_radius
        for ch_idx in range(len(channels)):
            axes.add_patch(plt.Circle((ch_x[ch_idx], ch_y[ch_idx]), radius=min_dist, facecolor='#ffffff',
                                      edgecolor=None, alpha=0.4, zorder=10))

    # Plotting the electrodes
    if plot_channels:
        axes.scatter(ch_x, ch_y, 15, facecolors='w', edgecolors='k', zorder=10)

    if plot_clabels:
        for t in range(len(channels)):
            axes.text(ch_x[t] + 0.01, ch_y[t] - 0.85 * min_dist, channels[t]['label'], fontsize=9,color = 'w',
                      zorder = 11)

    # Last considerations
    plot_lim = max(head_radius + 0.2, np.max(radius) + 0.2)
    axes.set_xlim([-plot_lim, plot_lim])
    axes.set_ylim([-plot_lim, plot_lim])
    axes.set_aspect('equal', 'box')
    plt.axis('off')
    # fig = plt.gcf()
    fig.patch.set_alpha(0.0)  # Set transparent background
    fig.tight_layout()
    if show is True:
        plt.show()

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
