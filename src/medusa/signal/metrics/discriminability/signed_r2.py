import numpy as np
from numpy.typing import NDArray

__all__ = ["signed_r2"]


def signed_r2(class1: NDArray, class2: NDArray, signed: bool = True,
              axis: int = 0) -> NDArray:
    """Compute the squared point-biserial correlation coefficient (r2-value).

    The r2-value is a measure of discriminability between two classes. It
    quantifies how much of the variance of the joint data is explained by the
    class labels along the requested ``axis``. When ``signed`` is True the
    output keeps the sign of the difference of class means, which indicates
    which class yields larger values.

    Parameters
    ----------
    class1
        (n_observations, ...). Data that belongs to the first class. The
        ``axis`` dimension indexes the observations.
    class2
        (n_observations, ...). Data that belongs to the second class. It must
        share the shape of ``class1`` on every dimension other than ``axis``.
    signed
        If True, the sign of the difference of class means is kept in the
        result. If False, the unsigned (always non-negative) r2-value is
        returned.
    axis
        Dimension along which the r2-value is computed. Therefore, if ``class1``
        and ``class2`` have dimensions ``(n_observations, n_samples)`` and
        ``axis=0``, the r2-value will have dimensions ``(n_samples,)``.

    Returns
    -------
    NDArray
        (Signed) r2-value, with the same shape as ``class1`` and ``class2`` but
        with the ``axis`` dimension removed.

    Examples
    --------
    >>> import numpy as np
    >>> from medusa.signal.metrics.discriminability.signed_r2 import signed_r2
    >>> rng = np.random.default_rng(42)
    >>> class1 = rng.normal(0.0, 1.0, size=(50, 8))
    >>> class2 = rng.normal(1.0, 1.0, size=(40, 8))
    >>> r2 = signed_r2(class1, class2, signed=True, axis=0)
    >>> r2.shape
    (8,)
    """
    # Sanitise inputs (dtype-preserving)
    class1 = np.asarray(class1)
    class2 = np.asarray(class2)

    # Length of each class
    N1 = class1.shape[axis]
    N2 = class2.shape[axis]

    # Pre-computation
    all_data = np.concatenate((class1, class2), axis=axis)
    v = np.var(all_data, axis=axis)
    m_diff = np.mean(class1, axis=axis) - np.mean(class2, axis=axis)

    # Compute the sign if required
    sign = 1
    if signed:
        sign = np.sign(m_diff)
        sign[sign == 0] = 1

    # Final r2 value
    r2 = sign * N1 * N2 * np.power(m_diff, 2) / (v * np.power(N1 + N2, 2))
    return r2
