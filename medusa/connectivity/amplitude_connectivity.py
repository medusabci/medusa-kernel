# Built-in imports
import warnings, os

# External imports
import numpy as np
from scipy import stats as sp_stats

# Medusa imports
import medusa.components
from medusa import signal_orthogonalization as orthogonalizate
from medusa.transforms import hilbert
from medusa.utils import check_dimensions
from medusa import tensorflow_integration


if os.environ.get("MEDUSA_EXTRAS_GPU_TF") == "1":
    import tensorflow as tf
    import tensorflow_probability as tfp


def __aec_gpu(data):
    """ This method implements the amplitude envelope correlation using GPU.

    NOTE: See the orthogonalized version.

    Parameters
    ----------
    data : numpy.ndarray
        MEEG Signal. [n_epochs, n_samples, n_channels].

    Returns
    -------
    aec : numpy.ndarray
        aec-based connectivity matrix. [n_epochs, n_channels, n_channels].
    """
    # Error check
    if type(data) != np.ndarray:
        raise ValueError("Parameter data must be of type numpy.ndarray")
    # Set to correct dimensions
    data = check_dimensions(data)

    # AEC computation
    hilb = hilbert(data)
    envelope = tf.math.abs(hilb) 
    env = tf.math.log(tf.math.square(envelope))

    aec = tfp.stats.correlation(env,sample_axis=1)

    return aec.numpy()


def __aec_cpu(data):
    """ This method implements the amplitude envelope correlation using CPU.

    NOTE: See the orthogonalized version.

    Parameters
    ----------
    data : numpy.ndarray
        MEEG Signal. [n_epochs, n_samples, n_channels].

    Returns
    -------
    aec : numpy.ndarray
        aec-based connectivity matrix. [n_epochs, n_channels, n_channels].

    """
    # Error check
    if type(data) != np.ndarray:
        raise ValueError("Parameter data must be of type numpy.ndarray")
    # Set to correct dimensions
    data = check_dimensions(data)

    #  Variable initialization
    n_epo = data.shape[0]
    n_cha = data.shape[2]
    aec = np.empty((n_epo, n_cha, n_cha))
    aec[:] = np.nan

    # AEC computation
    hilb = hilbert(data)
    envelope = abs(hilb)
    env = np.log(envelope ** 2)

    # Concurrent calculation for more than one epoch
    w_threads = []
    for epoch in env:
        t = medusa.components.ThreadWithReturnValue(target= np.corrcoef,
                                                    args=(epoch,None,False,))
        w_threads.append(t)
        t.start()

    for epoch_idx, thread in enumerate(w_threads):
        aec[epoch_idx, :, :] = thread.join()

    return aec


def __aec_ort_gpu(data):
    """ This method implements the orthogonalized version of the amplitude
    envelope correlation using GPU. This orthogonalized version minimizes the
    spurious connectivity caused by common sources (zero-lag correlations).

    Parameters
    ----------
    data : numpy.ndarray
        MEEG Signal. [n_epochs, n_samples, n_channels].

    Returns
    -------
    aec : numpy.ndarray
        aec-based connectivity matrix. [n_epochs, n_channels, n_channels].

    """
    # Error check
    if type(data) != np.ndarray:
        raise ValueError("Parameter data must be of type numpy.ndarray")
    # Set to correct dimensions
    data = check_dimensions(data)

    # Variable initialization
    n_epo = data.shape[0]
    n_samp = data.shape[1]
    n_cha = data.shape[2]
    aec_ort = np.empty((n_epo,n_cha, n_cha))
    aec_ort[:] = np.nan
    
    # AEC Ort Calculation (CPU orthogonalization is much faster than GPU one)
    signal_ort = orthogonalizate.signal_orthogonalization_cpu(data, data)
    signal_ort_2 = tf.transpose(tf.reshape(tf.transpose(signal_ort,perm=[0,3,2,1]),
                                           (n_epo,n_cha*n_cha,
                                            n_samp)),perm=[0,2,1])

    hilb_1 = hilbert(signal_ort_2)
    envelope_1 = tf.math.abs(hilb_1) 
    env = tf.math.log(tf.math.square(envelope_1))

    # Concurrent calculation for more than one epoch
    w_threads = []
    for epoch in env:
        t = medusa.components.ThreadWithReturnValue(target=__aec_ort_comp_aux,
                                                    args=(epoch, n_cha, 'gpu',))
        w_threads.append(t)
        t.start()

    for epoch_idx, thread in enumerate(w_threads):
        aec_ort[epoch_idx, :, :] = thread.join()
        
    return aec_ort


def __aec_ort_cpu(data):
    """ This method implements the orthogonalized version of the amplitude
    envelope correlation using CPU. This orthogonalized version minimizes the
    spurious connectivity caused by common sources (zero-lag correlations).

    Parameters
    ----------
    data : numpy.ndarray
        MEEG Signal. [n_epochs, n_samples, n_channels].

    Returns
    -------
    aec : numpy.ndarray
        aec-based connectivity matrix. [n_epochs, n_channels, n_channels].

    """
    # Error check
    if type(data) != np.ndarray:
        raise ValueError("Parameter data must be of type numpy.ndarray")

    # Set to correct dimensions
    data = check_dimensions(data)

    # Variable initialization
    n_epo = data.shape[0]
    n_samp = data.shape[1]
    n_cha = data.shape[2]
    aec_ort = np.empty((n_epo, n_cha, n_cha))
    aec_ort[:] = np.nan

    # AEC Ort Calculation
    signal_ort = orthogonalizate.signal_orthogonalization_cpu(data, data)
    signal_ort_2 = np.transpose(np.reshape(np.transpose(signal_ort, (0, 3, 2, 1)),
                            (n_epo, n_cha * n_cha, n_samp)),(0,2,1))

    hilb_1 = hilbert(signal_ort_2)
    envelope_1 = np.abs(hilb_1)
    env = np.log(np.square(envelope_1))

    # Concurrent calculation for more than one epoch
    w_threads = []
    for epoch in env:
        t = medusa.components.ThreadWithReturnValue(target=__aec_ort_comp_aux,
                                                    args=(epoch,n_cha,'cpu',))
        w_threads.append(t)
        t.start()

    for epoch_idx, thread in enumerate(w_threads):
        aec_ort[epoch_idx,:,:] = thread.join()

    return aec_ort


def __aec_ort_comp_aux(env, n_cha, ctype='cpu'):
    """
    Auxiliary method that implements a function to compute the orthogonalized AEC.
    Parameters
    ----------
    env: numpy.ndarray
        Array with signal envelope. [n_epochs, n_samples, n_channels x n_channels].
    type: str
        Calculation type: 'cpu' or 'gpu'.
    Returns
    -------
    aec_ort: numpy.ndarray
        AEC orthogonalized connectivity matrix. [n_channels, n_channels].
    """
    # Note: Orthogonalize A regarding B is not the same as orthogonalize B regarding
    # A, so we average lower and upper triangular matrices to construct the
    # symmetric matrix required for Orthogonalized AEC

    if ctype == 'cpu':
        aec_tmp = np.corrcoef(env, rowvar=False)
        aec_tmp2 = np.transpose(
            np.reshape(
                np.transpose(aec_tmp),
                (int(aec_tmp.shape[0] * aec_tmp.shape[0] / n_cha), -1)
            )
        )
        idx = np.linspace(0, aec_tmp2.shape[1] - 1, n_cha).astype(np.int32)
        aec = aec_tmp2[:, idx]
        aec_upper = np.triu(np.squeeze(aec))
        aec_lower = np.transpose(np.tril(np.squeeze(aec)))
        aec_ort = (aec_upper + aec_lower) / 2
        aec_ort = abs(np.triu(aec_ort, 1) + np.transpose(aec_ort))
        return aec_ort

    elif ctype == 'gpu':
        aec_tmp = tfp.stats.correlation(env)
        aec_tmp2 = tf.transpose(
            tf.reshape(
                tf.transpose(aec_tmp),
                (tf.cast(aec_tmp.shape[0] * aec_tmp.shape[0] / n_cha,
                         tf.int32), -1)
            )
        )
        idx = tf.cast(tf.linspace(0, len(aec_tmp2[0]) - 1, n_cha), tf.int32)
        aec = tf.gather(aec_tmp2, idx, axis=1).numpy()
        aec_upper = tf.linalg.band_part(aec, 0, -1)
        aec_lower = tf.transpose(tf.linalg.band_part(aec, -1, 0))
        aec_ort = tf.math.divide(tf.math.add(aec_upper, aec_lower), 2)
        aux = tf.linalg.band_part(aec_ort, 0, -1) - tf.linalg.band_part(
            aec_ort, 0, 0)
        aec_ort = tf.math.abs(tf.math.add(aux, tf.transpose(aec_ort)))
        return aec_ort


def aec(data, ort=True):
    """ This method implements the amplitude envelope correlation (using GPU if
    available). Based on the "ort" param, the signals could be orthogonalized
    before the computation of the amplitude envelope correlation.

    REFERENCES:
    Liu, Z., Fukunaga, M., de Zwart, J. A., & Duyn, J. H. (2010).
    Large-scale spontaneous fluctuations and correlations in brain electrical
    activity observed with magnetoencephalography. Neuroimage, 51(1), 102-111.

    Hipp, J. F., Hawellek, D. J., Corbetta, M., Siegel, M., & Engel,
    A. K. (2012). Large-scale cortical correlation structure of spontaneous
    oscillatory activity. Nature neuroscience, 15(6), 884-890.

    O’Neill, G. C., Barratt, E. L., Hunt, B. A., Tewarie, P. K., & Brookes, M.
    J. (2015). Measuring electrophysiological connectivity by power envelope
    correlation: a technical review on MEG methods. Physics in Medicine &
    Biology, 60(21), R271.

    Parameters
    ----------
    data : numpy.ndarray
        MEEG Signal. Allowed dimensions: [n_epochs, n_samples, n_channels] and
        [n_samples, n_channels].
    ort : bool
        If True, the signals on "data" will be orthogonalized before the
        computation of the amplitude envelope correlation.

    Returns
    -------
    aec : numpy.ndarray
        aec-based connectivity matrix. [n_epochs, n_channels, n_channels].

    """
    #  Error check
    if not np.issubdtype(data.dtype, np.number):
        raise ValueError('data matrix contains non-numeric values') 

    if not ort:
        if os.environ.get("MEDUSA_EXTRAS_GPU_TF") == "1" and \
                tensorflow_integration.check_tf_config(autoconfig=True):
            aec = __aec_gpu(data)
        else:
            aec = __aec_cpu(data)

    else:
        if os.environ.get("MEDUSA_EXTRAS_GPU_TF") == "1" and \
                tensorflow_integration.check_tf_config(autoconfig=True):
            aec = __aec_ort_gpu(data)
        else:
            aec = __aec_ort_cpu(data)

    return aec


def __iac_gpu(data):
    """ This method implements the instantaneous amplitude correlation using
    GPU.

    NOTE: See the orthogonalized version. In the original paper, the
    orthogonalized version was used

    Parameters
    ----------
    data : numpy.ndarray
        MEEG Signal. [n_epochs, n_samples, n_channels].

    Returns
    -------
    iac : numpy.ndarray
        iac-based connectivity matrix.
        [n_epochs, n_channels, n_channels, n_samples].


    """
    # Error check
    if type(data) != np.ndarray:
        raise ValueError("Parameter data must be of type numpy.ndarray")

    # Set to correct dimensions
    data = check_dimensions(data)

    #  Variable initialization
    n_epo = data.shape[0]
    n_samp = data.shape[1]
    n_cha = data.shape[2]

    # Z-score
    data = tf.math.divide(
        tf.math.subtract(data,
                         tf.math.reduce_mean(data, axis=1, keepdims=True)),
        tf.math.reduce_std(data, axis=1, keepdims=True))

    #  IAC computation
    hilb = hilbert(data)
    envelope = tf.math.abs(hilb)

    iac = tf.multiply(tf.tile(envelope, (1, 1, n_cha)),
                      tf.transpose(tf.reshape(
                          tf.transpose(tf.tile(envelope, (1, n_cha,1)),perm=[0,2,1]),
                          (n_epo, n_cha*n_cha, n_samp)), perm=[0,2,1])).numpy()
    iac = tf.reshape(tf.transpose(iac,perm=[0,2,1]),
                     (n_epo, n_cha, n_cha, n_samp)).numpy()

    # Set diagonal to 0
    diag_mask = tf.ones((n_cha, n_cha))
    diag_mask = tf.linalg.set_diag(diag_mask, tf.zeros(diag_mask.shape[0:-1]),
                                   name=None)
    iac = tf.multiply(iac, tf.repeat(tf.repeat(diag_mask[None,:, :, None], n_samp,
                                     axis=-1), n_epo,axis=0))

    return iac.numpy()


def __iac_cpu(data):
    """ This method implements the instantaneous amplitude correlation using
    CPU.

    NOTE: See the orthogonalized version. In the original paper, the
    orthogonalized version was used

    Parameters
    ----------
    data : numpy.ndarray
        MEEG Signal. [n_epochs, n_samples, n_channels].

    Returns
    -------
    iac : numpy.ndarray
        iac-based connectivity matrix.
        [n_epochs, n_channels, n_channels, n_samples].

    """
    # Error check
    if type(data) != np.ndarray:
        raise ValueError("Parameter data must be of type numpy.ndarray")

    # Set to correct dimensions
    data = check_dimensions(data)

    #  Variable initialization
    n_epo = data.shape[0]
    n_samp = data.shape[1]
    n_cha = data.shape[2]

    # IAC computation
    data = sp_stats.zscore(data, axis=1)

    hilb = hilbert(data)
    envelope = abs(hilb)
    iac = np.multiply(np.reshape(
        np.tile(envelope, (1, n_cha, 1)), (n_epo, n_samp, n_cha*n_cha), order='F'),
        np.tile(envelope, (1, 1, n_cha)))
    iac = np.reshape(np.transpose(iac,(0,2,1)), (n_epo, n_cha, n_cha, n_samp))

    # Set diagonal to 0
    diag_mask = np.ones((n_cha, n_cha))
    np.fill_diagonal(diag_mask, 0)
    iac = iac * np.repeat(np.repeat(diag_mask[None,:, :, None], n_samp, axis=-1),
                          n_epo,axis=0)

    return iac


def __iac_ort_gpu(data):
    """ This method implements the orthogonalized version of the instantaneous
    amplitude correlation using GPU. This orthogonalized version minimizes the
    spurious connectivity caused by common sources (zero-lag correlations).

    data : numpy.ndarray
        MEEG Signal. [n_epochs, n_samples, n_channels].

    Returns
    -------
    iac_ort : numpy.ndarray
        iac-based connectivity matrix.
        [n_epochs, n_channels, n_channels, n_samples].
    """
    # Error check
    if type(data) != np.ndarray:
        raise ValueError("Parameter data must be of type numpy.ndarray")

    # Set to correct dimensions
    data = check_dimensions(data)

    #  Variable initialization
    n_epo = data.shape[0]
    n_samp = data.shape[1]
    n_cha = data.shape[2]

    # IAC Ort Calculation (CPU orthogonalization is much faster than GPU one)
    # Z-score
    data = tf.math.divide(
        tf.math.subtract(data,
                         tf.math.reduce_mean(data, axis=1, keepdims=True)),
        tf.math.reduce_std(data, axis=1, keepdims=True))

    signal_ort = orthogonalizate.signal_orthogonalization_cpu(data.numpy(),
                                                              data.numpy())

    signal_ort_2 = tf.transpose(tf.reshape(tf.transpose(signal_ort, perm=
                                                        [0,3,2,1]),
                                           (n_epo, n_cha * n_cha,n_samp)),
                                perm = [0,2,1])

    hilb_1 = hilbert(signal_ort_2)
    envelope = tf.math.abs(hilb_1)

    iac = tf.multiply(tf.tile(envelope, (1, 1, n_cha**2)),
                      tf.transpose(tf.reshape(
                          tf.transpose(tf.tile(envelope, (1, n_cha**2, 1)), perm=
                                       [0,2,1]),
                          (n_epo,(n_cha*n_cha)**2, n_samp)),perm=[0,2,1]))
    iac = tf.reshape(tf.transpose(iac,perm=[0,2,1]), (n_epo, n_cha**2, n_cha**2,
                                                      n_samp))

    iac_tmp2 = tf.transpose(
        tf.reshape(
            tf.transpose(iac, (0,2, 1, 3)),
            (n_epo,int(iac.shape[1] * iac.shape[1] / n_cha), -1, iac.shape[3])
        ), (0,2, 1, 3)
    )
    idx = tf.cast(tf.linspace(0, iac_tmp2.shape[2]-1, n_cha), dtype=tf.int32)
    iac = tf.gather(iac_tmp2, idx, axis=2)

    # Orthogonalize A regarding B is not the same as orthogonalize B regarding
    # A, so we average lower and upper triangular matrices to construct the
    # symmetric matrix required for Orthogonalized iac

    iac_upper = tf.linalg.band_part(tf.transpose(iac, (0,3, 1, 2)), 0, -1)
    iac_lower = tf.transpose(tf.linalg.band_part(tf.transpose(iac, (0,3, 1, 2)),
                                                 -1, 0), (0,1, 3, 2))
    iac_ort = tf.math.divide(tf.math.add(iac_upper, iac_lower), 2)
    aux = tf.linalg.band_part(iac_ort, 0, -1) - tf.linalg.band_part(iac_ort, 0,
                                                                    0)
    iac_ort = tf.math.abs(tf.math.add(aux, tf.transpose(aux, (0,1, 3, 2))))

    return tf.transpose(iac_ort, (0, 2, 3, 1)).numpy()


def __iac_ort_cpu(data):
    """ This method implements the orthogonalized version of the instantaneous
    amplitude correlation using CPU. This orthogonalized version minimizes the
    spurious connectivity caused by common sources (zero-lag correlations).

    Parameters
    ----------
    data : numpy.ndarray
        MEEG Signal. [n_epochs, n_samples, n_channels].

    Returns
    -------
    iac_ort : numpy.ndarray
        iac-based connectivity matrix.
        [n_epochs, n_channels, n_channels, n_samples].
    """
    # Error check
    if type(data) != np.ndarray:
        raise ValueError("Parameter data must be of type numpy.ndarray")

    # Set to correct dimensions
    data = check_dimensions(data)

    #  Variable initialization
    n_epo = data.shape[0]
    n_samp = data.shape[1]
    n_cha = data.shape[2]

    # AEC Ort Calculation
    data = sp_stats.zscore(data, axis=1)

    signal_ort = orthogonalizate.signal_orthogonalization_cpu(data, data)
    signal_ort_2 = np.transpose(
        np.reshape(np.transpose(signal_ort, (0, 3, 2, 1)),
                   (n_epo, n_cha * n_cha, n_samp)), (0, 2, 1))

    hilb_1 = hilbert(signal_ort_2)
    envelope_1 = np.abs(hilb_1)

    iac = np.multiply(np.reshape(np.tile(
        envelope_1, (1, n_cha**2, 1)), (n_epo, n_samp, n_cha**2*n_cha**2),
        order='F'), np.tile(envelope_1, (1, 1, n_cha**2)))
    iac = np.reshape(np.transpose(iac,[0,2,1]), (n_epo,n_cha**2, n_cha**2, n_samp))
    iac_tmp2 = np.transpose(
        np.reshape(
            np.transpose(iac, (0,2, 1, 3)),
            (n_epo,int(iac.shape[1] * iac.shape[1] / n_cha), -1, n_samp)
        ), (0,2, 1, 3)
    )
    idx = np.linspace(0, iac_tmp2.shape[2]-1, n_cha).astype(np.int32)
    iac = iac_tmp2[:,:, idx, :]

    # Orthogonalize A regarding B is not the same as orthogonalize B regarding
    # A, so we average lower and upper triangular matrices to construct the
    # symmetric matrix required for Orthogonalized AEC

    iac_upper = np.triu(np.transpose(iac, (0,3, 1, 2)), k=1)
    iac_lower = np.transpose(np.tril(np.transpose(iac, (0,3, 1, 2)), k=-1), (0,1, 3,
                                                                           2))
    iac_ort = (iac_upper + iac_lower) / 2
    iac_ort = abs(np.triu(iac_ort, k=1) + np.transpose(iac_ort, (0,1, 3, 2)))

    return np.transpose(iac_ort, (0,2, 3, 1))


def iac(data, ort=True):
    """ This method implements the instantaneous amplitude correlation (using
    GPU if available). Based on the "ort" param, the signals could be
    orthogonalized before the computation of the amplitude envelope correlation.

    REFERENCES:
    Tewarie, P., Liuzzi, L., O'Neill, G. C., Quinn, A. J., Griffa,
    A., Woolrich, M. W., ... & Brookes, M. J. (2019). Tracking dynamic brain
    networks using high temporal resolution MEG measures of functional
    connectivity. Neuroimage, 200, 38-50.

    O’Neill, G. C., Barratt, E. L., Hunt, B. A., Tewarie, P. K., & Brookes, M.
    J. (2015). Measuring electrophysiological connectivity by power envelope
    correlation: a technical review on MEG methods. Physics in Medicine &
    Biology, 60(21), R271.

    Parameters
    ----------
    data : numpy.ndarray
        MEEG Signal. Allowed dimensions: [n_epochs, n_samples, n_channels] and
        [n_samples, n_channels].
    ort : bool
        If True, the signals on "data" will be orthogonalized before the
        computation of the instantaneous amplitude correlation.

    Returns
    -------
    iac : numpy 2D square matrix
        iac-based connectivity matrix.
        [n_epochs, n_channels, n_channels, n_samples].

    """
    #  Error check
    if not np.issubdtype(data.dtype, np.number):
        raise ValueError('data matrix contains non-numeric values')

    if not ort:
        if os.environ.get("MEDUSA_EXTRAS_GPU_TF") == "1" and \
                tensorflow_integration.check_tf_config(autoconfig=True):
            iac = __iac_gpu(data)
        else:
            iac = __iac_cpu(data)

    else:
        if os.environ.get("MEDUSA_EXTRAS_GPU_TF") == "1" and \
                tensorflow_integration.check_tf_config(autoconfig=True):
            iac = __iac_ort_gpu(data)
        else:
            iac = __iac_ort_cpu(data)

    return iac
