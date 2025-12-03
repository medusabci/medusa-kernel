import numpy as np


def check_dimensions(data, mode='time-series'):
    """
    Ensures that input `data` conforms to the expected dimensionality
    for time-domain visualization modes and reshapes it if needed.

    Parameters
    ----------
    data : np.ndarray
        Input data array. Can be 1D, 2D, 3D, or 4D depending on format.

        **For mode='time-series'** (standard multi-channel signals):
        - (n_samples,) → (1, n_samples, 1)
          Single-channel signal, single trial.
        - (n_samples, n_channels) → (1, n_samples, n_channels)
          Multi-channel signal, single trial.
        - (n_epochs, n_samples, n_channels) → unchanged
          Proper format for multiple trials and channels.

        **For mode='time-heatmap'** (e.g., time-frequency representations):
        Expects output shape (n_epochs, n_features, n_samples, n_channels)

        - (n_samples,) → (1, 1, n_samples, 1)
          Single-channel, single-feature heatmap.
        - (n_samples, n_channels) → (1, 1, n_samples, n_channels)
          One feature (e.g., power) over time and channels.
        - (n_features, n_samples, n_channels) → (1, n_features, n_samples, n_channels)
          One trial with multi-feature heatmap.
        - (n_epochs, n_features, n_samples, n_channels) → unchanged
          Full 4D heatmap data.

    mode : str, optional
        Determines the expected format. Must be one of:
        - 'time-series' : returns array of shape (n_epochs, n_samples, n_channels)
        - 'time-heatmap': returns array of shape (n_epochs, n_features, n_samples, n_channels)

    Returns
    -------
    np.ndarray
        Reshaped array with standardized dimensions based on the selected mode.

    Raises
    ------
    ValueError
        If the input data does not conform to expected dimensions or
        if the mode is not recognized.

    Examples
    --------
    >>> check_dimensions(np.random.randn(1000), mode='time-series').shape
    (1, 1000, 1)

    >>> check_dimensions(np.random.randn(1000, 16), mode='time-series').shape
    (1, 1000, 16)

    >>> check_dimensions(np.random.randn(5, 1000, 16), mode='time-series').shape
    (5, 1000, 16)

    >>> check_dimensions(np.random.randn(1000), mode='time-heatmap').shape
    (1, 1, 1000, 1)

    >>> check_dimensions(np.random.randn(32, 1000), mode='time-heatmap').shape
    (1, 1, 32, 1000)  # interpreted as single-feature multichannel

    >>> check_dimensions(np.random.randn(40, 1000, 16), mode='time-heatmap').shape
    (1, 40, 1000, 16)
    """
    data = np.asarray(data)
    if mode == 'time-series':
        if len(data.shape) == 1:
            return data[np.newaxis, :, np.newaxis]  # (1, n_samples, 1)
        elif len(data.shape) == 2:
            return data[np.newaxis, :, :]  # (1, n_samples, n_channels)
        elif len(data.shape) == 3:
            return data  # (n_epochs, n_samples, n_channels)
        else:
            raise ValueError('Incorrect number of dimensions for time-series')

    elif mode == 'time-heatmap':
        if len(data.shape) == 1:
            return data[np.newaxis, np.newaxis, :,
                   np.newaxis]  # (1, 1, n_samples, 1)
        elif len(data.shape) == 2:
            return data[np.newaxis, np.newaxis, :,
                   :]  # (1, 1, n_samples, n_channels)
        elif len(data.shape) == 3:
            return data[np.newaxis, :, :,
                   :]  # (1, n_features, n_samples, n_channels)
        elif len(data.shape) == 4:
            return data  # (n_epochs, n_features, n_samples, n_channels)
        else:
            raise ValueError('Incorrect number of dimensions for time-heatmap')

    else:
        raise ValueError(f"Mode {mode} not recognized")


