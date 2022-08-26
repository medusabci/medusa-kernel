import scipy.signal as sp_signal
import numpy as np
from medusa import transforms
from numba import jit
import warnings
import tensorflow as tf


@jit(nopython=True, cache=True, parallel=True)
def reshape_angles_loops(phase_data):
    """ Additional method require for the implementation of PLV, PLI, and wPLI
    in Numba. It receives the phases of the signal and return the PLV, PLI and
    wPLI connectivity matrices.

    NOTE: The shape of "phase_data" is [n_channels x n_samples], not the usual
    [n_samples x n_channels]

    Parameters
    ----------
    phase_data : numpy 2D matrix
        phases of the MEEG Signal. [n_channels x n_samples].

    Returns
    -------
    plv : numpy 2D square matrix
        plv-based connectivity matrix. [n_channels x n_channels].
    pli : numpy 2D square matrix
        pli-based connectivity matrix. [n_channels x n_channels].
    wpli : numpy 2D square matrix
        wpli-based connectivity matrix. [n_channels x n_channels].

    """
    n_cha = phase_data.shape[0]

    m = np.empty((phase_data.shape[0] * phase_data.shape[0],
                 phase_data.shape[1]))
    for i in range(n_cha):
        for j in range(n_cha):
            m[n_cha * i + j] = phase_data[i] - phase_data[j]

    n = np.empty((phase_data.shape[0] * phase_data.shape[0]))
    for i in range(m.shape[0]):
        n[i] = np.mean(np.sign(np.sin(m[i])))
    pli_vector = np.absolute(n)
    pli = np.reshape(pli_vector, (n_cha, n_cha))

    plv_vector = np.divide(
        np.absolute(np.sum(np.exp(1j * m), axis=1)),
        phase_data.shape[1])
    plv = np.reshape(plv_vector, (n_cha, n_cha))

    imz = np.sin(m)
    num = np.empty((phase_data.shape[0] * phase_data.shape[0]))
    den = np.empty((phase_data.shape[0] * phase_data.shape[0]))
    for i in range(m.shape[0]):
        num[i] = np.absolute(
                    np.mean(np.multiply(np.absolute(imz[i]), np.sign(imz[i]))))
        den[i] = np.mean(np.absolute(imz[i]))
    wpli_vector = np.divide(num, den)
    wpli = np.reshape(wpli_vector, (n_cha, n_cha))

    return plv, pli, wpli


def __phase_connectivity_numba(data):
    """ This method implements three phase-based connectivity parameters using
    Numba: PLV, PLI, and wPLI.

    REFERENCES: PLV: Mormann, F., Lehnertz, K., David, P., & Elger, C. E.
    (2000). Mean phase coherence as a measure for phase synchronization and its
    application to the EEG of epilepsy patients. Physica D: Nonlinear Phenomena,
    144(3-4), 358-369.
    PLI: Nolte, G., Bai, O., Wheaton, L., Mari, Z., Vorbach, S., & Hallett, M.
    (2004). Identifying true brain interaction from EEG data using the imaginary
    part of coherency. Clinical neurophysiology, 115(10), 2292-2307.
    wPLI: Vinck, M., Oostenveld, R., Van Wingerden, M., Battaglia, F., &
    Pennartz, C. M. (2011). An improved index of phase-synchronization for
    electrophysiological data in the presence of volume-conduction, noise and
    sample-size bias. Neuroimage, 55(4), 1548-1565.

    NOTE: PLV is sensitive to volume conduction effects

    Parameters
    ----------
    data : numpy 2D matrix
        MEEG Signal. [n_samples x n_channels].

    Returns
    -------
    plv : numpy 2D square matrix
        plv-based connectivity matrix. [n_channels x n_channels].
    pli : numpy 2D square matrix
        pli-based connectivity matrix. [n_channels x n_channels].
    wpli : numpy 2D square matrix
        wpli-based connectivity matrix. [n_channels x n_channels].

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
    num_chan = data.shape[1]

    # Connectivity computation
    phase_data = np.transpose(np.angle(sp_signal.hilbert(np.transpose(data))))
    phase_data = np.ascontiguousarray(phase_data.T)
    # angles_1 = np.reshape(np.tile(phase_data, (num_chan, 1)),
    #                       (len(phase_data), num_chan * num_chan),
    #                       order='F')
    # angles_2 = np.tile(phase_data, (1, num_chan))

    # pli_vector = abs(np.mean(np.sign(np.sin(angles_1 - angles_2)), axis=0))

    plv, pli, wpli, = reshape_angles_loops(phase_data)

    return plv, pli, wpli,


def __phase_connectivity_cpu(data):
    """ This method implements three phase-based connectivity parameters using
    CPU: PLV, PLI, and wPLI.

    REFERENCES: PLV: Mormann, F., Lehnertz, K., David, P., & Elger, C. E.
    (2000). Mean phase coherence as a measure for phase synchronization and its
    application to the EEG of epilepsy patients. Physica D: Nonlinear Phenomena,
    144(3-4), 358-369.
    PLI: Nolte, G., Bai, O., Wheaton, L., Mari, Z., Vorbach, S., & Hallett, M.
    (2004). Identifying true brain interaction from EEG data using the imaginary
    part of coherency. Clinical neurophysiology, 115(10), 2292-2307.
    wPLI: Vinck, M., Oostenveld, R., Van Wingerden, M., Battaglia, F., &
    Pennartz, C. M. (2011). An improved index of phase-synchronization for
    electrophysiological data in the presence of volume-conduction, noise and
    sample-size bias. Neuroimage, 55(4), 1548-1565.

    NOTE: PLV is sensitive to volume conduction effects

    Parameters
    ----------
    data : numpy 2D matrix
        MEEG Signal. [n_samples x n_channels].

    Returns
    -------
    plv : numpy 2D square matrix
        plv-based connectivity matrix. [n_channels x n_channels].
    pli : numpy 2D square matrix
        pli-based connectivity matrix. [n_channels x n_channels].
    wpli : numpy 2D square matrix
        wpli-based connectivity matrix. [n_channels x n_channels].

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
    num_chan = data.shape[1]

    # Connectivity computation
    phase_data = np.transpose(np.angle(sp_signal.hilbert(np.transpose(data))))
    phase_data = np.ascontiguousarray(phase_data)
    angles_1 = np.reshape(np.tile(phase_data, (num_chan, 1)),
                          (len(phase_data), num_chan * num_chan),
                          order='F')
    angles_2 = np.tile(phase_data, (1, num_chan))

    pli_vector = abs(np.mean(np.sign(np.sin(angles_1 - angles_2)), axis=0))
    pli = np.reshape(pli_vector, (num_chan, num_chan), order='F')

    plv_vector = np.divide(
        abs(np.sum(np.exp(1j * (angles_1 - angles_2)), axis=0)),
        data.shape[0])
    plv = np.reshape(plv_vector, (num_chan, num_chan), order='F')

    imz = np.sin(angles_1 - angles_2)
    with np.errstate(divide='ignore', invalid='ignore'):
        wpli_vector = np.divide(
            abs(np.mean(np.multiply(abs(imz), np.sign(imz)), axis=0)),
            np.mean(abs(imz), axis=0)
        )
    wpli = np.reshape(wpli_vector, (num_chan, num_chan), order='F')

    return plv, pli, wpli,


def __phase_connectivity_gpu(data):
    """ This method implements three phase-based connectivity parameters using
    GPU: PLV, PLI, and wPLI.

    REFERENCES: PLV: Mormann, F., Lehnertz, K., David, P., & Elger, C. E.
    (2000). Mean phase coherence as a measure for phase synchronization and its
    application to the EEG of epilepsy patients. Physica D: Nonlinear Phenomena,
    144(3-4), 358-369.
    PLI: Nolte, G., Bai, O., Wheaton, L., Mari, Z., Vorbach, S., & Hallett, M.
    (2004). Identifying true brain interaction from EEG data using the imaginary
    part of coherency. Clinical neurophysiology, 115(10), 2292-2307.
    wPLI: Vinck, M., Oostenveld, R., Van Wingerden, M., Battaglia, F., &
    Pennartz, C. M. (2011). An improved index of phase-synchronization for
    electrophysiological data in the presence of volume-conduction, noise and
    sample-size bias. Neuroimage, 55(4), 1548-1565.

    NOTE: PLV is sensitive to volume conduction effects

    Parameters
    ----------
    data : numpy 2D matrix
        MEEG Signal. [n_samples x n_channels].

    Returns
    -------
    plv : numpy 2D square matrix
        plv-based connectivity matrix. [n_channels x n_channels].
    pli : numpy 2D square matrix
        pli-based connectivity matrix. [n_channels x n_channels].
    wpli : numpy 2D square matrix
        wpli-based connectivity matrix. [n_channels x n_channels].

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
    num_chan = data.shape[1]

    # Connectivity computation
    phase_data = tf.math.angle(transforms.hilbert(data))

    angles_1 = tf.transpose(
                    tf.reshape(
                        tf.transpose(tf.tile(phase_data, (num_chan, 1))),
                        (num_chan * num_chan, len(phase_data)))
                )
    angles_2 = tf.tile(phase_data, (1, num_chan))

    pli_vector = tf.math.abs(
                    tf.math.reduce_mean(
                        tf.math.sign(
                            tf.math.sin(tf.math.subtract(angles_1, angles_2))),
                        axis=0))
    pli = tf.reshape(pli_vector, (num_chan, num_chan))

    plv_vector = tf.math.divide(
                    tf.math.abs(
                        tf.math.reduce_sum(
                            tf.math.exp(
                                tf.math.scalar_mul(
                                    1j,
                                    tf.cast(
                                        tf.math.subtract(angles_1, angles_2),
                                        'complex64'))),
                            axis=0)),
                    data.shape[0])
    plv = tf.reshape(plv_vector, (num_chan, num_chan))

    imz = tf.math.sin(tf.math.subtract(angles_1, angles_2))
    wpli_vector = tf.math.divide(
                    tf.math.abs(tf.math.reduce_mean(
                        tf.math.multiply(
                            tf.math.abs(imz),
                            tf.math.sign(imz)),
                        axis=0)),
                    tf.math.reduce_mean(tf.math.abs(imz), axis=0))
    wpli = tf.reshape(wpli_vector, (num_chan, num_chan))

    return plv, pli, wpli,


def phase_connectivity(data):
    """ This method implements three phase-based connectivity parameters: PLV,
    PLI, and wPLI.

    REFERENCES:
    PLV: Mormann, F., Lehnertz, K., David, P., & Elger, C. E.
    (2000). Mean phase coherence as a measure for phase synchronization and its
    application to the EEG of epilepsy patients. Physica D: Nonlinear Phenomena,
    144(3-4), 358-369.
    PLI: Nolte, G., Bai, O., Wheaton, L., Mari, Z., Vorbach, S., & Hallett, M.
    (2004). Identifying true brain interaction from EEG data using the imaginary
    part of coherency. Clinical neurophysiology, 115(10), 2292-2307.
    wPLI: Vinck, M., Oostenveld, R., Van Wingerden, M., Battaglia, F., &
    Pennartz, C. M. (2011). An improved index of phase-synchronization for
    electrophysiological data in the presence of volume-conduction, noise and
    sample-size bias. Neuroimage, 55(4), 1548-1565.

    NOTE: PLV is sensitive to volume conduction effects

    Parameters
    ----------
    data : numpy 2D matrix
        MEEG Signal. [n_samples x n_channels].

    Returns
    -------
    plv : numpy 2D square matrix
        plv-based connectivity matrix. [n_channels x n_channels].
    pli : numpy 2D square matrix
        pli-based connectivity matrix. [n_channels x n_channels].
    wpli : numpy 2D square matrix
        wpli-based connectivity matrix. [n_channels x n_channels].

    """
    from medusa import tensorflow_integration
    # Error check
    if not np.issubdtype(data.dtype, np.number):
        raise ValueError('data matrix contains non-numeric values')

    if tensorflow_integration.check_tf_config(autoconfig=True):
        plv, pli, wpli, = __phase_connectivity_gpu(data)
    else:
        # plv, pli, wpli, = __phase_connectivity_numba(data)
        plv, pli, wpli, = __phase_connectivity_cpu(data)
    # Remove nan values in wpli main diagonal
    wpli = tf.linalg.set_diag(wpli,np.ones(wpli.shape[0]))
    return plv, pli, wpli,
