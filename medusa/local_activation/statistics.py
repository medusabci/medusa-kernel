import numpy as np


def signed_r2(class1, class2, signed=True, axis=0):
    """ This function computes the basic form of the squared point biserial
    correlation coefficient (r2-value).

    Parameters
    ---------
    class1: list or numpy.ndarray
        Data that belongs to the first class
    class1: list or numpy.ndarray
        Data that belongs to the second class
    signed: bool (Optional, default=True)
        Controls if the sign should be mantained.
    axis: int (Optional, default=0)
        Dimension along which the r2-value is computed. Therefore,
        if class1 and class2 has dimensions of [observations x samples]
        and dim=0, the r2-value will have dimensions [1 x samples].

    Returns
    -------
    r2: numpy.ndarray
        (Signed) r2-value.
    """
    # Length of each class
    N1 = class1.shape[axis]
    N2 = class2.shape[axis]
    
    # Pre-computation
    all_data = np.concatenate((class1,class2), axis=axis)
    v = np.var(all_data, axis=axis)
    m_diff = np.mean(class1, axis=axis) - np.mean(class2, axis=axis)
    
    # Compute the sign if required
    sign = 1
    if signed:
        sign = np.sign(m_diff)
        sign[sign == 0] = 1
    
    # Final r2 value
    r2 = sign*N1*N2*np.power(m_diff, 2)/(v*np.power(N1+N2, 2))
    return r2
