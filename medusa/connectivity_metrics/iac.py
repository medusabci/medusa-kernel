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


# ----------------------------
# Load MATLAB results
# ----------------------------
mat = loadmat(r"C:\Users\1993_\Desktop\test_dyn_env_data.mat")

data_mat       = mat["data"]
dyn_IAC_mat    = mat["dyn_IAC_mat"]
mean_IAC_mat   = mat["mean_IAC_mat"]

print("Loaded MATLAB data")

# ----------------------------
# Run Python version
# ----------------------------
dyn_IAC_py = iac(data_mat, ort=True)

# ----------------------------
# Numerical comparison
# ----------------------------
err_dyn  = np.max(np.abs(dyn_IAC_py - dyn_IAC_mat))

print("\nMax absolute error:")
print("dyn_IAC  =", err_dyn)

# Tolerance check
tol = 1e-10

if err_dyn < tol:
    print("\n✅ SUCCESS: MATLAB and Python results match within tolerance.")
else:
    print("\n⚠️ WARNING: Results differ more than expected.")


# def __iac(data):
#     """
#     Computes the Instantaneous Amplitude Correlation (IAC) using the standard method
#     without signal orthogonalization. This version runs on the CPU.
#
#     NOTE: The original research recommends using the orthogonalized version
#     to mitigate spurious zero-lag correlations due to common sources.
#     See `__iac_ort`.
#
#     Parameters
#     ----------
#     data : numpy.ndarray
#         M/EEG signal data of shape [n_epochs, n_samples, n_channels].
#
#     Returns
#     -------
#     iac : numpy.ndarray
#         IAC-based functional connectivity_metrics matrix.
#         Shape: [n_epochs, n_channels, n_channels, n_samples].
#
#     Examples
#     --------
#     >>> from medusa.connectivity_metrics.iac import iac
#     >>> data = np.random.randn(10, 1000, 64)  # 10 epochs, 1000 time points, 64 channels
#     >>> iac_matrix = __iac(data)
#     >>> print(iac_matrix.shape)
#     (10, 64, 64, 1000)
#     """
#     # Error check
#     if type(data) != np.ndarray:
#         raise ValueError("Parameter data must be of type numpy.ndarray")
#
#     # Set to correct dimensions
#     data = check_dimensions(data)
#
#     #  Variable initialization
#     n_epo = data.shape[0]
#     n_samp = data.shape[1]
#     n_cha = data.shape[2]
#
#     # IAC computation
#     data = sp_stats.zscore(data, axis=1)
#
#     hilb = hilbert(data)
#     envelope = abs(hilb)
#     iac = np.multiply(np.reshape(
#         np.tile(envelope, (1, n_cha, 1)), (n_epo, n_samp, n_cha*n_cha), order='F'),
#         np.tile(envelope, (1, 1, n_cha)))
#     iac = np.reshape(np.transpose(iac,(0,2,1)), (n_epo, n_cha, n_cha, n_samp))
#
#     # Set diagonal to 0
#     diag_mask = np.ones((n_cha, n_cha))
#     np.fill_diagonal(diag_mask, 0)
#     iac = iac * np.repeat(np.repeat(diag_mask[None,:, :, None], n_samp, axis=-1),
#                           n_epo,axis=0)
#
#     return iac
#
#
# def __iac_ort(data):
#     """
#     Computes the orthogonalized Instantaneous Amplitude Correlation (IAC) using CPU.
#     Orthogonalization reduces spurious connectivity_metrics caused by signal leakage or
#     volume conduction, preserving only the genuine amplitude correlations.
#
#     Parameters
#     ----------
#     data : numpy.ndarray
#         M/EEG signal data of shape [n_epochs, n_samples, n_channels].
#
#     Returns
#     -------
#     iac_ort : numpy.ndarray
#         Symmetrized IAC-based functional connectivity_metrics matrix after orthogonalization.
#         Shape: [n_epochs, n_channels, n_channels, n_samples].
#
#     Examples
#     --------
#     >>> data = np.random.randn(5, 1500, 32)  # 5 epochs, 1500 time points, 32 channels
#     >>> iac_ort_matrix = __iac_ort(data)
#     >>> print(iac_ort_matrix.shape)
#     (5, 32, 32, 1500)
#     """
#
#     # Error check
#     if type(data) != np.ndarray:
#         raise ValueError("Parameter data must be of type numpy.ndarray")
#
#     # Set to correct dimensions
#     data = check_dimensions(data)
#
#     # Variable initialization
#     n_epo = data.shape[0]
#     n_samp = data.shape[1]
#     n_cha = data.shape[2]
#
#     # IAC Ort Calculation
#     data = sp_stats.zscore(data, axis=1)
#
#     # IAC Ort Calculation - Orthogonalized data has one additional dimension (the channel dimension is duplicated), as
#     # each channel (1st channel dimension) is orthogonalized regarding every other channel (2nd channel dimension)
#     signal_ort = orthogonalizate.signal_orthogonalization(data, data)
#     # The two channel dimensions are merged to paralelize the computation of the AEC
#     # epochs*chann*chann*samples -> epochs*chann^2*samples
#     signal_ort = np.transpose(
#         np.reshape(np.transpose(signal_ort, (0, 3, 2, 1)),
#                    (n_epo, n_cha * n_cha, n_samp)), (0, 2, 1))
#
#     hilb_1 = hilbert(signal_ort)
#     envelope_1 = np.abs(hilb_1)
#
#     # Comnputing IAC for each (duplicated) channel with every other one, obatining a chann^2*chann^2 matrix
#     iac = np.multiply(np.reshape(np.tile(
#         envelope_1, (1, n_cha**2, 1)), (n_epo, n_samp, n_cha**2*n_cha**2),
#         order='F'), np.tile(envelope_1, (1, 1, n_cha**2)))
#     # Reshape the data and take only the indices of interest, resulting in a chann*chann matrix
#     iac = np.reshape(np.transpose(iac,[0,2,1]), (n_epo,n_cha**2, n_cha**2, n_samp))
#     iac_tmp2 = np.transpose(
#         np.reshape(
#             np.transpose(iac, (0,2, 1, 3)),
#             (n_epo,int(iac.shape[1] * iac.shape[1] / n_cha), -1, n_samp)
#         ), (0,2, 1, 3)
#     )
#     idx = np.linspace(0, iac_tmp2.shape[2]-1, n_cha).astype(np.int32)
#     iac = iac_tmp2[:,:, idx, :]
#
#     # Orthogonalize A regarding B is not the same as orthogonalize B regarding
#     # A, so we average lower and upper triangular matrices to construct the
#     # symmetric matrix required for Orthogonalized IAC
#     iac_upper = np.triu(np.transpose(iac, (0,3, 1, 2)), k=1)
#     iac_lower = np.transpose(np.tril(np.transpose(iac, (0,3, 1, 2)), k=-1), (0,1, 3,2))
#     iac_ort = (iac_upper + iac_lower) / 2
#     iac_ort = abs(np.triu(iac_ort, k=1) + np.transpose(iac_ort, (0,1, 3, 2)))
#
#     return np.transpose(iac_ort, (0,2, 3, 1))


# def __iac_fixed_no_ort(data):
#     """
#     Compute Instantaneous Amplitude Correlation (IAC)
#     WITHOUT orthogonalisation.
#
#     Parameters
#     ----------
#     data : np.ndarray
#         Array with shape (n_channels, n_samples)
#
#     Returns
#     -------
#     dyn_IAC : np.ndarray
#         (n_channels, n_channels, n_samples)
#     """
#
#     data = np.asarray(data, dtype=float)
#     n_chan, n_samp = data.shape
#
#     # Z-score across samples for each channel
#     data = zscore(data, axis=1, ddof=1)
#
#     dyn_IAC = np.zeros((n_chan, n_chan, n_samp))
#
#     for r1 in range(n_chan):
#         x = data[r1, :]
#         env_x = np.abs(hilbert(x))
#
#         for r2 in range(n_chan):
#             if r1 == r2:
#                 continue
#
#             y = data[r2, :]
#             env_y = np.abs(hilbert(y))
#
#             # No orthogonalisation: direct envelope product
#             dyn_IAC[r1, r2, :] = env_x * env_y
#
#     return dyn_IAC


# def __iac_fixed(data):
#     """
#     Compute Instantaneous Amplitude Correlation (IAC) with pairwise leakage correction
#     (pairwise orthogonalisation) following the logic in the provided MATLAB code.
#     Reference: Tewarie et al., 2019 (as in the original code comment).
#
#     Parameters
#     ----------
#     data : np.ndarray
#         2D array with shape (n_channels, n_samples). Each row is a channel/time series.
#
#     Returns
#     -------
#     dyn_IAC : np.ndarray
#         3D array with shape (n_channels, n_channels, n_time_used) containing the
#         instantaneous amplitude product (env_x * env_y) for each pair (r1,r2) across time.
#     """
#
#     # Ensure input is a 2D numpy array: channels x samples
#     data = np.asarray(data, dtype=float)
#     if data.ndim != 2:
#         raise ValueError("data must be a 2D array with shape (n_channels, n_samples)")
#
#     n_chan, n_samp = data.shape
#
#     # Z-score across samples for each channel (axis=1).
#     data = zscore(data, axis=1, ddof=1)
#
#     # Pre-allocate dynamic IAC tensor (channels x channels x time)
#     dyn_IAC = np.zeros((n_chan, n_chan, n_samp), dtype=float)
#
#     # Main double loop over channel pairs
#     for r1 in range(n_chan):
#         x = data[r1, :]  # 1D array (n_samples,)
#         # compute pseudoinverse of the row-vector x (shape -> (n_samples,1))
#         inv_x = pinv(x.reshape(1, -1))  # shape (n_samples, 1)
#
#         # envelope of x (instantaneous amplitude)
#         env_x = np.abs(hilbert(x))
#
#         for r2 in range(n_chan):
#             if r1 == r2:
#                 continue
#
#             y = data[r2, :]
#
#             # pairwise leakage correction / orthogonalisation:
#             # y_cor = y - x * (inv_x' * y')
#             # inv_x has shape (n_samples, 1). inv_x.T @ y.reshape(-1,1) gives a (1,1) scalar.
#             coef = float(inv_x.T.dot(y.reshape(-1, 1)))  # scalar
#             projection = x * coef  # 1D vector (n_samples,)
#             y_cor = y - projection
#
#             # envelope of the corrected y
#             env_y = np.abs(hilbert(y_cor))
#
#             # store pointwise product of envelopes
#             dyn_IAC[r1, r2, :] = env_x * env_y
#
#     return dyn_IAC