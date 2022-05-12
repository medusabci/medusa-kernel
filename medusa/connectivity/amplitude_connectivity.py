import numpy as np
from medusa import signal_orthogonalization as orthogonalizate
from medusa.hilbert import hilbert
from medusa import pearson_corr_matrix as corr
import warnings


def __aec_gpu(data):
    """
    This function calculates the amplitude envelope correlation using GPU
    
    Parameters
    ----------
    data : numpy 2D matrix
        MEEG Signal. SamplesXChannel.
        
    Returns
    -------
    aec : numpy 2D matrix
        Array of size ChannelsXChannels containing aec values.
        
    """
    import tensorflow as tf
    import tensorflow_probability as tfp
    # ERROR CHECK
    if (type(data) != np.ndarray):
        raise ValueError("Parameter data must be of type numpy.ndarray")
    if data.shape[0] < data.shape[1]:
        warnings.warn("Warning: Signal dimensions flipped out. If you have more samples than channels, comment this "
                      "line")
        data = data.T

    #  AEC CALCULATION
    hilb = hilbert(data)
    envelope = tf.math.abs(hilb) 
    env = tf.math.log(tf.math.square(envelope))
    aec = tfp.stats.correlation(env)
        
    return aec


def __aec_cpu(data):
    """
    This function calculates the amplitude envelope correlation using CPU
    
    Parameters
    ----------
    data : numpy 2D matrix
        MEEG Signal. SamplesXChannel.
        
    Returns
    -------
    aec : numpy 2D matrix
        Array of size ChannelsXChannels containing aec values.
        
    """
    # ERROR CHECK
    if (type(data) != np.ndarray):
        raise ValueError("Parameter data must be of type numpy.ndarray")
    if data.shape[0] < data.shape[1]:
        warnings.warn("Warning: Signal dimensions flipped out. If you have more samples than channels, comment this "
                      "line")
        data = data.T

    #  VARIABLE INITIALIZATION
    aec = np.empty((len(data[1]), len(data[1])))
    aec[:] = np.nan

    # AEC CALCULATION
    hilb = hilbert(data)
    envelope = abs(hilb) 
    env = np.log(envelope**2)
    aec = corr.pearson_corr_matrix(env, env)
        
    return aec


def __aec_ort_gpu(data):
    """
    This function calculates the orthogonalized version of the amplitude 
    envelope correlation using GPU. This orthogonalized version minimizes the 
    spurious connectivity caused by common source
    
    Parameters
    ----------
    data : numpy 2D matrix
        MEEG Signal. SamplesXChannel.
        
    Returns
    -------
    aec : numpy 2D matrix
        Array of size ChannelsXChannels containing orthogonalized aec values.
        
    """
    import tensorflow as tf
    import tensorflow_probability as tfp
    # Error check
    if (type(data) != np.ndarray):
        raise ValueError("Parameter data must be of type numpy.ndarray")
    if data.shape[0] < data.shape[1]:
        warnings.warn("Warning: Signal dimensions flipped out. If you have more samples than channels, comment this "
                      "line")
        data = data.T


    # Variable inicialization
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
    env = tf.math.log(tf.math.square(envelope_1) )     
    aec_tmp = tfp.stats.correlation(env)
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
    #     aec[:,n_chan] = tf.transpose(tf.slice(tfp.stats.correlation(env),[0,n_chan],[len(data[0]),1]))
        
    # Orthogonalize A regarding B is not the same as orthogonalize B regarding 
    # A, so we average lower and upper triangular matrices to construct the 
    # symmetric matrix required for Orthogonalized AEC 

    aec_upper = tf.linalg.band_part(aec, 0, -1)
    aec_lower = tf.transpose(tf.linalg.band_part(aec, -1, 0))
    aec_ort = tf.math.divide(tf.math.add(aec_upper, aec_lower), 2)
    aux = tf.linalg.band_part(aec_ort, 0, -1) - tf.linalg.band_part(aec_ort, 0, 0)
    aec_ort = tf.math.abs(tf.math.add(aux, tf.transpose(aec_ort)))
        
    return aec_ort


def __aec_ort_cpu(data, verbose=False):
    """
    This function calculates the orthogonalized version of the amplitude 
    envelope correlation using CPU. This orthogonalized version minimizes the 
    spurious connectivity caused by common source
    
    Parameters
    ----------
    data : numpy 2D matrix
        MEEG Signal. SamplesXChannel.
        
    Returns
    -------
    aec : numpy 2D matrix
        Array of size ChannelsXChannels containing orthogonalized aec values.
        
    """
    # Error check
    if (type(data) != np.ndarray):
        raise ValueError("Parameter data must be of type numpy.ndarray")
    if data.shape[0] < data.shape[1]:
        warnings.warn("Warning: Signal dimensions flipped out. If you have more samples than channels, comment this "
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
    """
    Calculates the amplitude envelope correlation.

    Parameters
    ----------
    data : numpy 2D matrix
        Time series. SamplesXChannels.
    ort : bool
        Orthogonalize or not the signal: removes volume conduction effects.
        Default True
        
    Returns
    -------
    aec : numpy 2D matrix
        Array of size ChannelsXChannels containing aec values.    

    """
    from medusa import tensorflow_integration
    #  ERROR CHECK
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
    """
    This function calculates the amplitude envelope correlation using GPU

    Parameters
    ----------
    data : numpy 2D matrix
        MEEG Signal. SamplesXChannel.

    Returns
    -------
    iac : numpy 2D matrix
        Array of size ChannelsXChannels containing iac values.

    """
    import tensorflow as tf

    # ERROR CHECK
    if (type(data) != np.ndarray):
        raise ValueError("Parameter data must be of type numpy.ndarray")
    if data.shape[0] < data.shape[1]:
        warnings.warn("Warning: Signal dimensions flipped out. If you have more samples than channels, comment this "
                      "line")
        data = data.T

    # VARIABLE INITIALIZATION
    n_chan = data.shape[1]
    n_samples = data.shape[0]

    # Z-score
    data = tf.math.divide(tf.math.subtract(data,
                                           tf.math.reduce_mean(data, axis=0, keepdims=True)),
                          tf.math.reduce_std(data, axis=0, keepdims=True))

    #  IAC CALCULATION
    hilb = hilbert(data)
    envelope = tf.math.abs(hilb)

    iac = tf.multiply(tf.tile(envelope, (1, n_chan)),
                      tf.transpose(tf.reshape(tf.transpose(tf.tile(envelope, (n_chan, 1))), (n_chan*n_chan, n_samples)))).numpy()
    iac = tf.reshape(tf.transpose(iac), (n_chan, n_chan, n_samples)).numpy()

    # Set diagonal to 0
    diag_mask = tf.ones((n_chan, n_chan))
    diag_mask = tf.linalg.set_diag(diag_mask, tf.zeros(diag_mask.shape[0:-1]), name=None)
    iac = tf.multiply(iac, tf.repeat(diag_mask[:, :, None], (iac.shape[2]), axis=2))


    return iac


def __iac_cpu(data):
    """
    This function calculates the amplitude envelope correlation using CPU

    Parameters
    ----------
    data : numpy 2D matrix
        MEEG Signal. SamplesXChannel.

    Returns
    -------
    iac : numpy 2D matrix
        Array of size ChannelsXChannels containing iac values.

    """
    # ERROR CHECK
    if (type(data) != np.ndarray):
        raise ValueError("Parameter data must be of type numpy.ndarray")
    if data.shape[0] < data.shape[1]:
        warnings.warn("Warning: Signal dimensions flipped out. If you have more samples than channels, comment this "
                      "line")
        data = data.T

    #  VARIABLE INITIALIZATION
    n_chan = data.shape[1]
    n_samples = data.shape[0]

    # IAC CALCULATION
    data = scipy.stats.zscore(data, axis=0)

    hilb = hilbert(data)
    envelope = abs(hilb)
    iac = np.multiply(np.reshape(np.tile(envelope, (n_chan, 1)), (n_samples, n_chan*n_chan), order='F'),
                      np.tile(envelope, (1, n_chan)))
    iac = np.reshape(iac.T, (n_chan, n_chan, n_samples))

    # Set diagonal to 0
    diag_mask = np.ones((n_chan, n_chan))
    np.fill_diagonal(diag_mask, 0)
    iac = iac * np.repeat(diag_mask[:, :, None], (iac.shape[2]), axis=2)

    return iac


def __iac_ort_gpu(data):
    """
    This function calculates the orthogonalized version of the amplitude
    envelope correlation using GPU. This orthogonalized version minimizes the
    spurious connectivity caused by common source

    Parameters
    ----------
    data : numpy 2D matrix
        MEEG Signal. SamplesXChannel.

    Returns
    -------
    iac : numpy 2D matrix
        Array of size ChannelsXChannels containing orthogonalized iac values.

    """
    import tensorflow as tf
    import tensorflow_probability as tfp
    # Error check
    if (type(data) != np.ndarray):
        raise ValueError("Parameter data must be of type numpy.ndarray")
    if data.shape[0] < data.shape[1]:
        warnings.warn("Warning: Signal dimensions flipped out. If you have more samples than channels, comment this "
                      "line")
        data = data.T

    # Variable inicialization
    n_cha = data.shape[1]
    n_samples = data.shape[0]
    # iac = np.empty((num_chan, num_chan))
    # iac[:] = np.nan

    # IAC Ort Calculation (CPU orthogonalization is much faster than GPU one)
    # Z-score
    data = tf.math.divide(tf.math.subtract(data,
                                           tf.math.reduce_mean(data, axis=0, keepdims=True)),
                          tf.math.reduce_std(data, axis=0, keepdims=True))

    signal_ort = orthogonalizate.signal_orthogonalization_cpu(data.numpy(), data.numpy())

    signal_ort_2 = tf.transpose(tf.reshape(tf.transpose(signal_ort),
                                           (n_cha * n_cha,
                                            signal_ort.shape[0])))
    hilb_1 = hilbert(signal_ort_2)
    envelope = tf.math.abs(hilb_1)

    iac = tf.multiply(tf.tile(envelope, (1, n_cha**2)),
                      tf.transpose(tf.reshape(tf.transpose(tf.tile(envelope, (n_cha**2, 1))), ((n_cha*n_cha)**2, n_samples))))
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
    iac_lower = tf.transpose(tf.linalg.band_part(tf.transpose(iac, (2, 0, 1)), -1, 0), (0, 2, 1))
    iac_ort = tf.math.divide(tf.math.add(iac_upper, iac_lower), 2)
    aux = tf.linalg.band_part(iac_ort, 0, -1) - tf.linalg.band_part(iac_ort, 0, 0)
    iac_ort = tf.math.abs(tf.math.add(aux, tf.transpose(aux, (0, 2, 1))))

    return tf.transpose(iac_ort, (1, 2, 0))


def __iac_ort_cpu(data, verbose=False):
    """
    This function calculates the orthogonalized version of the amplitude
    envelope correlation using CPU. This orthogonalized version minimizes the
    spurious connectivity caused by common source

    Parameters
    ----------
    data : numpy 2D matrix
        MEEG Signal. SamplesXChannel.

    Returns
    -------
    iac : numpy 2D matrix
        Array of size ChannelsXChannels containing orthogonalized iac values.

    """
    # Error check
    if (type(data) != np.ndarray):
        raise ValueError("Parameter data must be of type numpy.ndarray")
    if data.shape[0] < data.shape[1]:
        warnings.warn("Warning: Signal dimensions flipped out. If you have more samples than channels, comment this "
                      "line")
        data = data.T

    # Variable initialization
    n_cha = data.shape[1]
    n_samples = data.shape[0]

    # AEC Ort Calculation
    data = scipy.stats.zscore(data, axis=0)

    signal_ort = orthogonalizate.signal_orthogonalization_cpu(data, data)

    # 1st method of calculating the Ort AEC. Faster
    signal_ort_2 = np.transpose(np.reshape(np.transpose(signal_ort),
                                           (n_cha*n_cha,
                                            signal_ort.shape[0])))
    hilb_1 = hilbert(signal_ort_2)
    envelope_1 = np.abs(hilb_1)

    iac = np.multiply(np.reshape(np.tile(envelope_1, (n_cha**2, 1)), (n_samples, n_cha**2*n_cha**2), order='F'),
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
    #     iac_tmp = np.multiply(np.reshape(np.tile(env, (n_cha, 1)), (n_samples, n_cha * n_cha), order='F'),
    #                       np.tile(env, (1, n_cha)))
    #     iac_tmp = np.reshape(iac_tmp.T, (n_cha, n_cha, n_samples))
    #
    #     iac[:, cha, :] = iac_tmp[:, cha, :]

    # Orthogonalize A regarding B is not the same as orthogonalize B regarding
    # A, so we average lower and upper triangular matrices to construct the
    # symmetric matrix required for Orthogonalized AEC

    iac_upper = np.triu(np.transpose(iac, (2, 0, 1)), k=1)
    iac_lower = np.transpose(np.tril(np.transpose(iac, (2, 0, 1)), k=-1), (0, 2, 1))
    iac_ort = (iac_upper + iac_lower) / 2
    iac_ort = abs(np.triu(iac_ort, k=1) + np.transpose(iac_ort, (0, 2, 1)))

    return np.transpose(iac_ort, (1, 2, 0))


def iac(data, ort=True):
    """
    Calculates the instantaneous amplitude correlation.

    Parameters
    ----------
    data : numpy 2D matrix
        Time series. SamplesXChannels.
    ort : bool
        Orthogonalize or not the signal: removes volume conduction effects.
        Default True

    Returns
    -------
    iac : numpy 2D matrix
        Array of size ChannelsXChannels containing iac values.

    """
    from medusa import tensorflow_integration

    #  ERROR CHECK
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


# def matrix_correlation(data, mode='Spearman', trunc=False):
#     """
#     Calculates the amplitude envelope correlation.
#
#     Parameters
#     ----------
#     data : numpy 2D matrix
#         Time series. SamplesXChannels.
#     mode : string
#         Pearson or Spearman. Default Spearman. NOTE: Pearson requires normally
#         distributed data
#     trunc : bool
#         If True, non-significant correlation values will be considered 0.
#         Default False
#
#     Returns
#     -------
#     correlation : numpy 2D matrix
#         Array of size ChannelsXChannels containing iac values.
#
#     """
#     #  ERROR CHECK
#     if not np.issubdtype(data.dtype, np.number):
#         raise ValueError('data matrix contains non-numeric values')
#     if not isinstance(trunc, bool):
#         raise ValueError('trunc must be a boolean variable')
#
#     if mode == 'Spearman':
#         n = data.shape[0]
#         corr = np.corrcoef(data, rowvar=False)
#         ab = n / 2 - 1
#         p_val = 2 * special.btdtr(ab, ab, 0.5 * (1 - abs(np.float64(corr))))
#
#         # corr, p_val = stats.spearmanr(data)
#     elif mode == 'Pearson':
#         corr = np.zeros((data.shape[1], data.shape[1]))
#         p_val = np.zeros((data.shape[1], data.shape[1]))
#         for i in range(data.shape[1]):
#             for j in range(data.shape[1]):
#                 corr[i, j], p_val[i, j] = stats.pearsonr(data[:, i], data[:, j])
#     else:
#         raise ValueError('Unknown correlation mode')
#
#     if trunc:
#         corr[p_val > 0.05] = 0
#
#     return corr, p_val


if __name__ == "__main__":
    import scipy.io
    import time
    import matplotlib.pyplot as plt
    mat = scipy.io.loadmat('Q:/VRG/0001_Control.mat')
    vector = np.array(mat["signal"])[0:8, 0:100]

    n_iter = 20

    times_cpu = []
    times_gpu = []
    times_cpu_o = []
    times_gpu_o = []
    for ii in range(n_iter):
        t0 = time.time()
        salida = __iac_cpu(vector)
        t1 = time.time()
        salida = salida[:, :, 0]
        plt.imshow(salida)
        # plt.clim(0, 0.3)
        plt.colorbar()
        plt.show()
        t2 = time.time()
        salida2 = __iac_gpu(vector)
        t3 = time.time()
        salida2 = salida2[:, :, 0]
        plt.imshow(salida2)
        # plt.clim(0, 0.3)
        plt.colorbar()
        plt.show()

        times_cpu.append(t1 - t0)
        times_gpu.append(t3 - t2)

        print('Time CPU: ', t1 - t0)
        print('Time GPU: ', t3 - t2)

        t0 = time.time()
        salida = __iac_ort_cpu(vector)
        t1 = time.time()
        salida = salida[:, :, 0]
        plt.imshow(salida)
        # plt.clim(0, 0.3)
        plt.colorbar()
        plt.show()
        t2 = time.time()
        salida2 = __iac_ort_gpu(vector)
        t3 = time.time()
        salida2 = salida2[:, :, 0]
        plt.imshow(salida2)
        # plt.clim(0, 0.3)
        plt.colorbar()
        plt.show()

        times_cpu_o.append(t1 - t0)
        times_gpu_o.append(t3 - t2)

        print('Time Ort. CPU: ', t1 - t0)
        print('Time Ort. GPU: ', t3 - t2)

    print('Mean time for CPU: ', np.mean(times_cpu[1:]))
    print('Mean time for CPU Ort: ', np.mean(times_cpu_o[1:]))
    print('Mean time for GPU: ', np.mean(times_gpu[1:]))
    print('Mean time for GPU Ort: ', np.mean(times_gpu_o[1:]))
