import numpy as np
from scipy import stats as sp_stats
from medusa import signal_orthogonalization as orthogonalizate
from medusa.transforms import hilbert
from medusa import pearson_corr_matrix as corr
import warnings


def __aec_gpu(data):
    """ This method implements the amplitude envelope correlation using GPU.

    REFERENCES: Liu, Z., Fukunaga, M., de Zwart, J. A., & Duyn, J. H. (2010).
    Large-scale spontaneous fluctuations and correlations in brain electrical
    activity observed with magnetoencephalography. Neuroimage, 51(1), 102-111.

    NOTE: See the orthogonalized version.

    Parameters
    ----------
    data : numpy 2D matrix
        MEEG Signal. [n_samples x n_channels].

    Returns
    -------
    aec : numpy 2D square matrix
        aec-based connectivity matrix. [n_channels x n_channels].

    """
    import tensorflow as tf
    from tensorflow_probability import stats as tfp_stats
    # Error check
    if type(data) != np.ndarray:
        raise ValueError("Parameter data must be of type numpy.ndarray")
    if data.shape[0] < data.shape[1]:
        warnings.warn("Warning: Signal dimensions flipped out. If you have more"
                      " samples than channels, comment this "
                      "line")
        data = data.T

    #  AEC computation
    hilb = hilbert(data)
    envelope = tf.math.abs(hilb) 
    env = tf.math.log(tf.math.square(envelope))
    aec = tfp_stats.correlation(env)
        
    return aec


def __aec_cpu(data):
    """ This method implements the amplitude envelope correlation using CPU.

    REFERENCES: Liu, Z., Fukunaga, M., de Zwart, J. A., & Duyn, J. H. (2010).
    Large-scale spontaneous fluctuations and correlations in brain electrical
    activity observed with magnetoencephalography. Neuroimage, 51(1), 102-111.

    NOTE: See the orthogonalized version.

    Parameters
    ----------
    data : numpy 2D matrix
        MEEG Signal. [n_samples x n_channels].

    Returns
    -------
    aec : numpy 2D square matrix
        aec-based connectivity matrix. [n_channels x n_channels].

    """
    # Error check
    if type(data) != np.ndarray:
        raise ValueError("Parameter data must be of type numpy.ndarray")
    if data.shape[0] < data.shape[1]:
        warnings.warn("Warning: Signal dimensions flipped out. If you have more"
                      " samples than channels, comment this "
                      "line")
        data = data.T

    #  Variable initialization
    aec = np.empty((len(data[1]), len(data[1])))
    aec[:] = np.nan

    # AEC computation
    hilb = hilbert(data)
    envelope = abs(hilb) 
    env = np.log(envelope**2)
    aec = corr.pearson_corr_matrix(env, env)
        
    return aec


def __aec_ort_gpu(data):
    """ This method implements the orthogonalized version of the amplitude
    envelope correlation using GPU. This orthogonalized version minimizes the
    spurious connectivity caused by common sources (zero-lag correlations).

    REFERENCES: Liu, Z., Fukunaga, M., de Zwart, J. A., & Duyn, J. H. (2010).
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
    data : numpy 2D matrix
        MEEG Signal. [n_samples x n_channels].

    Returns
    -------
    aec : numpy 2D square matrix
        aec-based connectivity matrix. [n_channels x n_channels].

    """
    import tensorflow as tf
    from tensorflow_probability import stats as tfp_stats
    # Error check
    if type(data) != np.ndarray:
        raise ValueError("Parameter data must be of type numpy.ndarray")
    if data.shape[0] < data.shape[1]:
        warnings.warn("Warning: Signal dimensions flipped out. If you have more"
                      " samples than channels, comment this "
                      "line")
        data = data.T

    # Variable initialization
    num_chan = len(data[1])
    aec = np.empty((num_chan, num_chan))
    aec[:] = np.nan
    
    # AEC Ort Calculation (CPU orthogonalization is much faster than GPU one)
    signal_ort = orthogonalizate.signal_orthogonalization_cpu(data, data)

    signal_ort_2 = tf.transpose(tf.reshape(tf.transpose(signal_ort),
                                           (num_chan*num_chan,
                                            len(signal_ort))))
    hilb_1 = hilbert(signal_ort_2)
    envelope_1 = tf.math.abs(hilb_1) 
    env = tf.math.log(tf.math.square(envelope_1))
    aec_tmp = tfp_stats.correlation(env)
    aec_tmp2 = tf.transpose(
        tf.reshape(
            tf.transpose(aec_tmp),
            (tf.cast(aec_tmp.shape[0]*aec_tmp.shape[0]/num_chan, tf.int32), -1)
        )
    )
    idx = tf.cast(tf.linspace(0, len(aec_tmp2[0])-1, num_chan), tf.int32)
    aec = tf.gather(aec_tmp2, idx, axis=1).numpy()

    # # Another way of calculating the AEC
    # for n_chan in range(0,num_chan):
    #     hilb_1 = hilbert(np.asarray(signal_ort[:,:,n_chan]))
    #     envelope_1 = tf.math.abs(hilb_1) 
    #     env = tf.math.log(tf.math.square(envelope_1) )
    #     aec[:,n_chan] = tf.transpose(tf.slice(tfp_stats.correlation(env),
    #                                       [0,n_chan],[len(data[0]),1]))
        
    # Orthogonalize A regarding B is not the same as orthogonalize B regarding 
    # A, so we average lower and upper triangular matrices to construct the 
    # symmetric matrix required for Orthogonalized AEC 

    aec_upper = tf.linalg.band_part(aec, 0, -1)
    aec_lower = tf.transpose(tf.linalg.band_part(aec, -1, 0))
    aec_ort = tf.math.divide(tf.math.add(aec_upper, aec_lower), 2)
    aux = tf.linalg.band_part(aec_ort, 0, -1) - tf.linalg.band_part(
        aec_ort, 0, 0)
    aec_ort = tf.math.abs(tf.math.add(aux, tf.transpose(aec_ort)))
        
    return aec_ort


def __aec_ort_cpu(data):
    """ This method implements the orthogonalized version of the amplitude
    envelope correlation using CPU. This orthogonalized version minimizes the
    spurious connectivity caused by common sources (zero-lag correlations).

    REFERENCES: Liu, Z., Fukunaga, M., de Zwart, J. A., & Duyn, J. H. (2010).
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
    data : numpy 2D matrix
        MEEG Signal. [n_samples x n_channels].

    Returns
    -------
    aec : numpy 2D square matrix
        aec-based connectivity matrix. [n_channels x n_channels].

    """
    # Error check
    if type(data) != np.ndarray:
        raise ValueError("Parameter data must be of type numpy.ndarray")
    if data.shape[0] < data.shape[1]:
        warnings.warn("Warning: Signal dimensions flipped out. If you have more"
                      " samples than channels, comment this "
                      "line")
        data = data.T

    # Variable initialization
    n_cha = len(data[1])
    aec = np.empty((n_cha, n_cha))
    aec[:] = np.nan

    # AEC Ort Calculation
    signal_ort = orthogonalizate.signal_orthogonalization_cpu(data, data)

    # 1st method of calculating the Ort AEC. Faster
    signal_ort_2 = np.transpose(np.reshape(np.transpose(signal_ort),
                                           (n_cha*n_cha,
                                            len(signal_ort))))
    hilb_1 = hilbert(signal_ort_2)
    envelope_1 = np.abs(hilb_1)
    env = np.log(np.square(envelope_1))
    aec_tmp = np.corrcoef(env, rowvar=False)
    aec_tmp2 = np.transpose(
        np.reshape(
            np.transpose(aec_tmp),
            (int(aec_tmp.shape[0]*aec_tmp.shape[0]/n_cha), -1)
        )
    )
    idx = np.linspace(0, aec_tmp2.shape[1]-1, n_cha).astype(np.int32)
    aec = aec_tmp2[:, idx]

    # 2nd method of calculating the Ort AEC
    # for cha in range(n_cha):
    #     hilb = hilbert(signal_ort[:, :, cha])
    #     env = np.log(np.square(np.abs(hilb)))
    #     aec[:, cha] = np.corrcoef(env.T)[:, cha]

    # Orthogonalize A regarding B is not the same as orthogonalize B regarding
    # A, so we average lower and upper triangular matrices to construct the
    # symmetric matrix required for Orthogonalized AEC
    aec_upper = np.triu(np.squeeze(aec))
    aec_lower = np.transpose(np.tril(np.squeeze(aec)))
    aec_ort = (aec_upper + aec_lower) / 2
    aec_ort = abs(np.triu(aec_ort, 1) + np.transpose(aec_ort))

    return aec_ort


def aec(data, ort=True):
    """ This method implements the amplitude envelope correlation (using GPU if
    available). Based on the "ort" param, the signals could be orthogonalized
    before the computation of the amplitude envelope correlation.

    REFERENCES: Liu, Z., Fukunaga, M., de Zwart, J. A., & Duyn, J. H. (2010).
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
    data : numpy 2D matrix
        MEEG Signal. [n_samples x n_channels].
    ort : bool
        If True, the signals on "data" will be orthogonalized before the
        computation of the amplitude envelope correlation.

    Returns
    -------
    aec : numpy 2D square matrix
        aec-based connectivity matrix. [n_channels x n_channels].

    """
    from medusa import tensorflow_integration
    #  Error check
    if not np.issubdtype(data.dtype, np.number):
        raise ValueError('data matrix contains non-numeric values') 

    if not ort:
        if tensorflow_integration.check_tf_config(autoconfig=True):
            aec = __aec_gpu(data)
        else:
            aec = __aec_cpu(data)

    else:
        if tensorflow_integration.check_tf_config(autoconfig=True):
            aec = __aec_ort_gpu(data)
        else:
            aec = __aec_ort_cpu(data)

    return aec


def __iac_gpu(data):
    """ This method implements the instantaneous amplitude correlation using
    GPU.

    REFERENCES: Tewarie, P., Liuzzi, L., O'Neill, G. C., Quinn, A. J., Griffa,
    A., Woolrich, M. W., ... & Brookes, M. J. (2019). Tracking dynamic brain
    networks using high temporal resolution MEG measures of functional
    connectivity. Neuroimage, 200, 38-50.

    NOTE: See the orthogonalized version. In the original paper, the
    orthogonalized version was used

    Parameters
    ----------
    data : numpy 2D matrix
        MEEG Signal. [n_samples x n_channels].

    Returns
    -------
    iac : numpy 2D square matrix
        iac-based connectivity matrix. [n_channels x n_channels].

    """
    import tensorflow as tf

    # Error check
    if type(data) != np.ndarray:
        raise ValueError("Parameter data must be of type numpy.ndarray")
    if data.shape[0] < data.shape[1]:
        warnings.warn("Warning: Signal dimensions flipped out. If you have more"
                      " samples than channels, comment this "
                      "line")
        data = data.T

    # Variable initialization
    n_chan = data.shape[1]
    n_samples = data.shape[0]

    # Z-score
    data = tf.math.divide(
        tf.math.subtract(data,
                         tf.math.reduce_mean(data, axis=0, keepdims=True)),
        tf.math.reduce_std(data, axis=0, keepdims=True))

    #  IAC computation
    hilb = hilbert(data)
    envelope = tf.math.abs(hilb)

    iac = tf.multiply(tf.tile(envelope, (1, n_chan)),
                      tf.transpose(tf.reshape(
                          tf.transpose(tf.tile(envelope, (n_chan, 1))),
                          (n_chan*n_chan, n_samples)))).numpy()
    iac = tf.reshape(tf.transpose(iac), (n_chan, n_chan, n_samples)).numpy()

    # Set diagonal to 0
    diag_mask = tf.ones((n_chan, n_chan))
    diag_mask = tf.linalg.set_diag(diag_mask, tf.zeros(diag_mask.shape[0:-1]),
                                   name=None)
    iac = tf.multiply(iac, tf.repeat(diag_mask[:, :, None], (iac.shape[2]),
                                     axis=2))

    return iac


def __iac_cpu(data):
    """ This method implements the instantaneous amplitude correlation using
    CPU.

    REFERENCES: Tewarie, P., Liuzzi, L., O'Neill, G. C., Quinn, A. J., Griffa,
    A., Woolrich, M. W., ... & Brookes, M. J. (2019). Tracking dynamic brain
    networks using high temporal resolution MEG measures of functional
    connectivity. Neuroimage, 200, 38-50.

    NOTE: See the orthogonalized version. In the original paper, the
    orthogonalized version was used

    Parameters
    ----------
    data : numpy 2D matrix
        MEEG Signal. [n_samples x n_channels].

    Returns
    -------
    iac : numpy 2D square matrix
        iac-based connectivity matrix. [n_channels x n_channels].

    """
    # Error check
    if type(data) != np.ndarray:
        raise ValueError("Parameter data must be of type numpy.ndarray")
    if data.shape[0] < data.shape[1]:
        warnings.warn("Warning: Signal dimensions flipped out. If you have more"
                      " samples than channels, comment this "
                      "line")
        data = data.T

    #  Variable initialization
    n_chan = data.shape[1]
    n_samples = data.shape[0]

    # IAC computation
    data = sp_stats.zscore(data, axis=0)

    hilb = hilbert(data)
    envelope = abs(hilb)
    iac = np.multiply(np.reshape(
        np.tile(envelope, (n_chan, 1)), (n_samples, n_chan*n_chan), order='F'),
        np.tile(envelope, (1, n_chan)))
    iac = np.reshape(iac.T, (n_chan, n_chan, n_samples))

    # Set diagonal to 0
    diag_mask = np.ones((n_chan, n_chan))
    np.fill_diagonal(diag_mask, 0)
    iac = iac * np.repeat(diag_mask[:, :, None], (iac.shape[2]), axis=2)

    return iac


def __iac_ort_gpu(data):
    """ This method implements the orthogonalized version of the instantaneous
    amplitude correlation using GPU. This orthogonalized version minimizes the
    spurious connectivity caused by common sources (zero-lag correlations).

    REFERENCES: Tewarie, P., Liuzzi, L., O'Neill, G. C., Quinn, A. J., Griffa,
    A., Woolrich, M. W., ... & Brookes, M. J. (2019). Tracking dynamic brain
    networks using high temporal resolution MEG measures of functional
    connectivity. Neuroimage, 200, 38-50.
    O’Neill, G. C., Barratt, E. L., Hunt, B. A., Tewarie, P. K., & Brookes, M.
    J. (2015). Measuring electrophysiological connectivity by power envelope
    correlation: a technical review on MEG methods. Physics in Medicine &
    Biology, 60(21), R271.

    Parameters
    ----------
    data : numpy 2D matrix
        MEEG Signal. [n_samples x n_channels].

    Returns
    -------
    iac : numpy 2D square matrix
        iac-based connectivity matrix. [n_channels x n_channels].

    """
    import tensorflow as tf
    from tensorflow_probability import stats as tfp_stats
    # Error check
    if type(data) != np.ndarray:
        raise ValueError("Parameter data must be of type numpy.ndarray")
    if data.shape[0] < data.shape[1]:
        warnings.warn("Warning: Signal dimensions flipped out. If you have more"
                      " samples than channels, comment this "
                      "line")
        data = data.T

    # Variable initialization
    n_cha = data.shape[1]
    n_samples = data.shape[0]
    # iac = np.empty((num_chan, num_chan))
    # iac[:] = np.nan

    # IAC Ort Calculation (CPU orthogonalization is much faster than GPU one)
    # Z-score
    data = tf.math.divide(
        tf.math.subtract(data,
                         tf.math.reduce_mean(data, axis=0, keepdims=True)),
        tf.math.reduce_std(data, axis=0, keepdims=True))

    signal_ort = orthogonalizate.signal_orthogonalization_cpu(data.numpy(),
                                                              data.numpy())

    signal_ort_2 = tf.transpose(tf.reshape(tf.transpose(signal_ort),
                                           (n_cha * n_cha,
                                            signal_ort.shape[0])))
    hilb_1 = hilbert(signal_ort_2)
    envelope = tf.math.abs(hilb_1)

    iac = tf.multiply(tf.tile(envelope, (1, n_cha**2)),
                      tf.transpose(tf.reshape(
                          tf.transpose(tf.tile(envelope, (n_cha**2, 1))),
                          ((n_cha*n_cha)**2, n_samples))))
    iac = tf.reshape(tf.transpose(iac), (n_cha**2, n_cha**2, n_samples))

    iac_tmp2 = tf.transpose(
        tf.reshape(
            tf.transpose(iac, (1, 0, 2)),
            (int(iac.shape[0] * iac.shape[0] / n_cha), -1, iac.shape[2])
        ), (1, 0, 2)
    )
    idx = tf.cast(tf.linspace(0, iac_tmp2.shape[1]-1, n_cha), dtype=tf.int32)
    iac = tf.gather(iac_tmp2, idx, axis=1)

    # Orthogonalize A regarding B is not the same as orthogonalize B regarding
    # A, so we average lower and upper triangular matrices to construct the
    # symmetric matrix required for Orthogonalized iac

    iac_upper = tf.linalg.band_part(tf.transpose(iac, (2, 0, 1)), 0, -1)
    iac_lower = tf.transpose(tf.linalg.band_part(tf.transpose(iac, (2, 0, 1)),
                                                 -1, 0), (0, 2, 1))
    iac_ort = tf.math.divide(tf.math.add(iac_upper, iac_lower), 2)
    aux = tf.linalg.band_part(iac_ort, 0, -1) - tf.linalg.band_part(iac_ort, 0,
                                                                    0)
    iac_ort = tf.math.abs(tf.math.add(aux, tf.transpose(aux, (0, 2, 1))))

    return tf.transpose(iac_ort, (1, 2, 0))


def __iac_ort_cpu(data):
    """ This method implements the orthogonalized version of the instantaneous
    amplitude correlation using GPU. This orthogonalized version minimizes the
    spurious connectivity caused by common sources (zero-lag correlations).

    REFERENCES: Tewarie, P., Liuzzi, L., O'Neill, G. C., Quinn, A. J., Griffa,
    A., Woolrich, M. W., ... & Brookes, M. J. (2019). Tracking dynamic brain
    networks using high temporal resolution MEG measures of functional
    connectivity. Neuroimage, 200, 38-50.
    O’Neill, G. C., Barratt, E. L., Hunt, B. A., Tewarie, P. K., & Brookes, M.
    J. (2015). Measuring electrophysiological connectivity by power envelope
    correlation: a technical review on MEG methods. Physics in Medicine &
    Biology, 60(21), R271.

    Parameters
    ----------
    data : numpy 2D matrix
        MEEG Signal. [n_samples x n_channels].

    Returns
    -------
    iac : numpy 2D square matrix
        iac-based connectivity matrix. [n_channels x n_channels].

    """
    # Error check
    if type(data) != np.ndarray:
        raise ValueError("Parameter data must be of type numpy.ndarray")
    if data.shape[0] < data.shape[1]:
        warnings.warn("Warning: Signal dimensions flipped out. If you have more"
                      " samples than channels, comment this "
                      "line")
        data = data.T

    # Variable initialization
    n_cha = data.shape[1]
    n_samples = data.shape[0]

    # AEC Ort Calculation
    data = sp_stats.zscore(data, axis=0)

    signal_ort = orthogonalizate.signal_orthogonalization_cpu(data, data)

    # 1st method of calculating the Ort AEC. Faster
    signal_ort_2 = np.transpose(np.reshape(np.transpose(signal_ort),
                                           (n_cha*n_cha,
                                            signal_ort.shape[0])))
    hilb_1 = hilbert(signal_ort_2)
    envelope_1 = np.abs(hilb_1)

    iac = np.multiply(np.reshape(np.tile(
        envelope_1, (n_cha**2, 1)), (n_samples, n_cha**2*n_cha**2), order='F'),
                      np.tile(envelope_1, (1, n_cha**2)))
    iac = np.reshape(iac.T, (n_cha**2, n_cha**2, n_samples))
    iac_tmp2 = np.transpose(
        np.reshape(
            np.transpose(iac, (1, 0, 2)),
            (int(iac.shape[0] * iac.shape[0] / n_cha), -1, iac.shape[2])
        ), (1, 0, 2)
    )
    idx = np.linspace(0, iac_tmp2.shape[1]-1, n_cha).astype(np.int32)
    iac = iac_tmp2[:, idx, :]

    # 2nd method of calculating the Ort AEC
    # iac = np.zeros((n_cha, n_cha, signal_ort.shape[0]))
    # for cha in range(n_cha):
    #     hilb = hilbert(signal_ort[:, :, cha])
    #     env = np.abs(hilb)
    #
    #     iac_tmp = np.multiply(np.reshape(np.tile(env, (n_cha, 1)), (n_samples,
    #                                       n_cha * n_cha), order='F'),
    #                       np.tile(env, (1, n_cha)))
    #     iac_tmp = np.reshape(iac_tmp.T, (n_cha, n_cha, n_samples))
    #
    #     iac[:, cha, :] = iac_tmp[:, cha, :]

    # Orthogonalize A regarding B is not the same as orthogonalize B regarding
    # A, so we average lower and upper triangular matrices to construct the
    # symmetric matrix required for Orthogonalized AEC

    iac_upper = np.triu(np.transpose(iac, (2, 0, 1)), k=1)
    iac_lower = np.transpose(np.tril(np.transpose(iac, (2, 0, 1)), k=-1), (0, 2,
                                                                           1))
    iac_ort = (iac_upper + iac_lower) / 2
    iac_ort = abs(np.triu(iac_ort, k=1) + np.transpose(iac_ort, (0, 2, 1)))

    return np.transpose(iac_ort, (1, 2, 0))


def iac(data, ort=True):
    """ This method implements the instantaneous amplitude correlation (using
    GPU if available). Based on the "ort" param, the signals could be
    orthogonalized before the computation of the amplitude envelope correlation.

    REFERENCES: Tewarie, P., Liuzzi, L., O'Neill, G. C., Quinn, A. J., Griffa,
    A., Woolrich, M. W., ... & Brookes, M. J. (2019). Tracking dynamic brain
    networks using high temporal resolution MEG measures of functional
    connectivity. Neuroimage, 200, 38-50.

    Parameters
    ----------
    data : numpy 2D matrix
        MEEG Signal. [n_samples x n_channels].
    ort : bool
        If True, the signals on "data" will be orthogonalized before the
        computation of the instantaneous amplitude correlation.

    Returns
    -------
    iac : numpy 2D square matrix
        iac-based connectivity matrix. [n_channels x n_channels].

    """
    from medusa import tensorflow_integration

    #  Error check
    if not np.issubdtype(data.dtype, np.number):
        raise ValueError('data matrix contains non-numeric values')

    if not ort:
        if tensorflow_integration.check_tf_config(autoconfig=True):
            iac = __iac_gpu(data)
        else:
            iac = __iac_cpu(data)

    else:
        if tensorflow_integration.check_tf_config(autoconfig=True):
            iac = __iac_ort_gpu(data)
        else:
            iac = __iac_ort_cpu(data)

    return iac
