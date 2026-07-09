"""Paradigm-agnostic BCI performance metrics (free functions)."""

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "itr",
]


def itr(accuracy: "float | NDArray", n_commands: int,
        selections_per_min: float) -> "float | NDArray":
    """Information transfer rate (ITR), in bits per minute.

    ITR is a common way to measure BCI speed. It has known limits for online
    systems. It assumes that every command is equally likely, that the system has no
    memory, that the paradigm is synchronous, and that the user cannot fix mistakes.

    Parameters
    ----------
    accuracy :
        System accuracy, from 0 to 1. May be a single number or an array. An array
        is handled element by element.
    n_commands :
        Number of commands the user can choose from. Must be greater than 1.
    selections_per_min :
        Number of selections made per minute.

    Returns
    -------
    float or numpy.ndarray
        ITR in bits per minute. Same shape as ``accuracy``.

    Examples
    --------
    >>> from medusa.pipelines.bci.performance import itr
    >>> round(itr(1.0, 4, 12), 2)
    24.0
    >>> round(itr(0.9, 4, 12), 2)
    16.47
    """
    acc = np.asarray(accuracy, dtype=float)
    log2_n = np.log2(n_commands)
    # Per-command bits via the standard Wolpaw formula, with the exact limits at the
    # boundaries (the log terms vanish at acc==0 / acc==1).
    with np.errstate(divide="ignore", invalid="ignore"):
        bits = (log2_n
                + acc * np.log2(acc)
                + (1 - acc) * np.log2((1 - acc) / (n_commands - 1)))
    bits = np.where(acc >= 1.0, log2_n, bits)
    bits = np.where(acc <= 0.0, 0.0, bits)
    result = bits * selections_per_min
    return float(result) if result.ndim == 0 else result
