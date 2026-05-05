# External imports
import numpy as np
from scipy.io import loadmat

# Medusa imports
from scipy.signal import hilbert
from scipy.linalg import pinv
from scipy.stats import zscore
from joblib import Parallel, delayed


def __iac_ort(data, n_jobs=-1):
    """
    Parallel computation of Instantaneous Amplitude Correlation (IAC)
    WITH pairwise orthogonalisation (leakage correction).

    Parameters
    ----------
    data : np.ndarray (n_channels, n_samples)
    n_jobs : int
        Number of parallel workers (-1 = all CPUs)

    Returns
    -------
    dyn_IAC : np.ndarray (n_channels, n_channels, n_samples)
    mean_IAC : np.ndarray (n_channels, n_channels)
    """

    data = np.asarray(data, dtype=float)
    n_chan, n_samp = data.shape

    # Z-score across time
    data = zscore(data, axis=1, ddof=1)

    # Precompute analytic signal envelopes of original signals
    envelopes = np.abs(hilbert(data, axis=1))

    def compute_row(r1):
        """Compute dyn_IAC for one seed channel r1."""
        x = data[r1]
        env_x = envelopes[r1]

        # Pseudoinverse of x (matches MATLAB pinv(x))
        inv_x = pinv(x.reshape(1, -1))

        row = np.zeros((n_chan, n_samp))

        for r2 in range(n_chan):
            if r1 == r2:
                continue

            y = data[r2]

            # Projection coefficient
            coef = float(inv_x.T @ y.reshape(-1, 1))

            # Orthogonalised signal
            y_cor = y - x * coef

            # Envelope of corrected signal
            env_y = np.abs(hilbert(y_cor))

            # Envelope product
            row[r2, :] = env_x * env_y

        return row

    # Parallel over seed channels
    rows = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(compute_row)(r1) for r1 in range(n_chan)
    )

    dyn_IAC = np.stack(rows, axis=0)

    return dyn_IAC


def __iac(data, n_jobs=-1):
    """
    Parallel computation of Instantaneous Amplitude Correlation (IAC)
    WITHOUT leakage correction.

    Parameters
    ----------
    data : np.ndarray (n_channels, n_samples)
    n_jobs : int
        Number of parallel workers (-1 = all CPUs)

    Returns
    -------
    dyn_IAC : np.ndarray (n_channels, n_channels, n_samples)
    mean_IAC : np.ndarray (n_channels, n_channels)
    """

    data = np.asarray(data, dtype=float)
    n_chan, n_samp = data.shape

    # Z-score across time
    data = zscore(data, axis=1, ddof=1)

    # Precompute envelopes once (big speedup)
    envelopes = np.abs(hilbert(data, axis=1))   # shape: (n_chan, n_samp)

    def compute_row(r1):
        """Compute dyn_IAC for one seed channel r1."""
        env_x = envelopes[r1]
        row = np.zeros((n_chan, n_samp))

        for r2 in range(n_chan):
            if r1 == r2:
                continue
            row[r2, :] = env_x * envelopes[r2]

        return row

    # Parallel over rows
    rows = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(compute_row)(r1) for r1 in range(n_chan)
    )

    dyn_IAC = np.stack(rows, axis=0)

    return dyn_IAC


def iac(data, ort=True):
    """
   Computes the Instantaneous Amplitude Correlation (IAC) from M/EEG signals.
   Offers the option to orthogonalize the signals before estimating amplitude
   envelope correlations. CPU is used for computation.

   REFERENCES:
   Tewarie, P., Liuzzi, L., O'Neill, G. C., Quinn, A. J., Griffa,
   A., Woolrich, M. W., ... & Brookes, M. J. (2019). Tracking dynamic brain
   networks using high temporal resolution MEG measures of functional
   connectivity_metrics. Neuroimage, 200, 38-50.

   O’Neill, G. C., Barratt, E. L., Hunt, B. A., Tewarie, P. K., & Brookes, M.
   J. (2015). Measuring electrophysiological connectivity_metrics by power envelope
   correlation: a technical review on MEG methods. Physics in Medicine &
   Biology, 60(21), R271.

   Parameters
   ----------
   data : numpy.ndarray
       M/EEG signal array. Accepted shapes:
       - [n_epochs, n_samples, n_channels]
       - [n_samples, n_channels] (interpreted as one epoch)

   ort : bool, optional
       If True (default), signals are orthogonalized before computing the IAC.

   Returns
   -------
   iac : numpy.ndarray
       Functional connectivity_metrics matrix based on IAC.
       Shape: [n_epochs, n_channels, n_channels, n_samples].

   Examples
   --------
   >>> data = np.random.randn(1000, 64)  # One epoch, 1000 samples, 64 channels
   >>> conn = iac(data, ort=False)
   >>> print(conn.shape)
   (1, 64, 64, 1000)

   >>> data_multi = np.random.randn(3, 2000, 64)  # Three epochs
   >>> conn_ort = iac(data_multi, ort=True)
   >>> print(conn_ort.shape)
   (3, 64, 64, 2000)
   """
    #  Error check
    if not np.issubdtype(data.dtype, np.number):
        raise ValueError('data matrix contains non-numeric values')

    if not ort:
        iac = __iac(data)

    else:
        iac = __iac_ort(data)

        # If single epoch, add epoch dimension
    if iac.ndim == 3:
        iac = iac[np.newaxis, :, :, :]
    for i in range(iac.shape[0]):
        for t in range(iac.shape[3]):
            iac[i, :, :, t] = (iac[i, :, :, t] + iac[i, :, :, t].T) / 2
            np.fill_diagonal(iac[i, :, :, t], 0)
    return iac
