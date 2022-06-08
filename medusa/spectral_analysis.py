import numpy as np


def normalize_psd(psd, norm='rel'):
    """
    Normalizes the PSD by the total power.

    :param psd: Power Spectral Density (PSD) of the signal with shape [samples], [samples x channels] or
                [epochs x samples x channels]. It assumes PSD is the one-sided spectrum.
    :type psd: numpy array or list

    :param norm: Normalization to be performed. Choose z for z-score or rel for relative power.
    :type norm: string

    """
    # Check errors
    if len(psd.shape) > 3:
        raise Exception('Parameter psd must have shape [samples], [samples x channels] or [epochs x samples x channels]')

    # Reshape
    if len(psd.shape) == 1:
        psd = psd.reshape(psd.shape[0], 1)

    if len(psd.shape) == 2:
        psd = psd.reshape(1, psd.shape[0], psd.shape[1])

    if norm == 'rel':
        p = np.sum(psd, axis=1, keepdims=True)
        psd_norm = psd / p
    elif norm == 'z':
        m = np.mean(psd, keepdims=True, axis=1)
        s = np.std(psd, keepdims=True, axis=1)
        psd_norm = (psd - m) / s
    else:
        raise Exception('Unknown normalization. Choose z or rel')

    return psd_norm
