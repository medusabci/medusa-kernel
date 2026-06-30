"""Core utilities used across the medusa-kernel/signal package."""

from collections.abc import Callable, Iterable
from threading import Thread
import warnings
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

__all__ = ["check_data_dims", "ThreadWithReturnValue"]

_RepType = Literal['time', 'time_segments',
                    'freq', 'freq_segments',
                    'time_freq', 'time_freq_segments']

_INSERTED_AXES: dict[_RepType, dict[int, tuple[int, ...]]] = {
    'time': {
        1: (1,),
        2: (),
    },
    'time_segments': {
        1: (0, 2),
        2: (0,),
        3: (),
    },
    'freq': {
        1: (1,),
        2: (),
    },
    'freq_segments': {
        1: (0, 2),
        2: (0,),
        3: (),
    },
    'time_freq': {
        2: (2,),
        3: (),
    },
    'time_freq_segments': {
        2: (0, 3),
        3: (0,),
        4: (),
    }
}


def check_data_dims(
    data: NDArray,
    rep_type: _RepType,
) -> tuple[NDArray, tuple[int, ...]]:
    """Validate and promote ``data`` to the canonical shape for ``rep_type``.

    The medusa-kernel signal contract recognises six representation
    types, each with a fixed canonical ndim and axis order:

    ========================  =====================================================  =====
    ``rep_type``              Canonical shape                                         ndim
    ========================  =====================================================  =====
    ``'time'``                ``(n_samples, n_channels)``                             2
    ``'time_segments'``       ``(n_segments, n_samples, n_channels)``                 3
    ``'freq'``                ``(n_frequencies, n_channels)``                         2
    ``'freq_segments'``       ``(n_segments, n_frequencies, n_channels)``             3
    ``'time_freq'``           ``(n_frequencies, n_times, n_channels)``                3
    ``'time_freq_segments'``  ``(n_segments, n_frequencies, n_times, n_channels)``    4
    ========================  =====================================================  =====

    Under-dimensioned inputs are promoted by inserting length-1 axes at
    the *segments* (leading) and/or *channel* (trailing) positions. The
    channel axis is always last; the segments axis (when present) is
    always first. Ambiguous shapes are resolved in favour of
    "channels-last": e.g. a 2-D input under ``rep_type='time_freq'`` is
    interpreted as ``(n_frequencies, n_times)`` and promoted to
    ``(n_frequencies, n_times, 1)``, not as
    ``(n_frequencies, n_channels)``.

    Parameters
    ----------
    data :
        Input array. Acceptable ndims depend on ``rep_type``; see Notes.
    rep_type :
        Signal representation. Must be one of the six values above.
        Unknown values raise :class:`ValueError`.

    Returns
    -------
    out : NDArray
        Array promoted to the canonical shape for ``rep_type``. Dtype is
        preserved. When ``data.ndim`` already matches the canonical
        ndim, this is the input array (after a single
        :func:`numpy.asarray` call) and ``inserted`` is ``()``.
    inserted : tuple[int, ...]
        Positions of the axes inserted during promotion, in the order
        used by :func:`numpy.expand_dims`. Pass this directly to
        :func:`numpy.squeeze` to restore the original shape once your
        computation is done (see Examples).

    Raises
    ------
    ValueError
        If ``data.ndim`` is incompatible with ``rep_type`` (e.g. a 4-D
        input with ``rep_type='time'`` or a 1-D input with
        ``rep_type='time_freq'``), or if ``rep_type`` is not one of the
        six recognised values.

    Warns
    -----
    UserWarning
        Emitted whenever axes are inserted by promotion. The warning is
        raised with ``stacklevel=2`` so its source location points to
        the caller of :func:`check_data_dims`. No warning is emitted
        when the input already has the canonical ndim.

    Notes
    -----
    Promotion rules:

    ========================  ===============  =================  ===================  ==================
    ``rep_type``              1-D              2-D                3-D                  4-D
    ========================  ===============  =================  ===================  ==================
    ``time``                  ``(n, 1)``       unchanged          error                error
    ``time_segments``         ``(1, n, 1)``    ``(1, n, ch)``     unchanged            error
    ``freq``                  ``(f, 1)``       unchanged          error                error
    ``freq_segments``         ``(1, f, 1)``    ``(1, f, ch)``     unchanged            error
    ``time_freq``             error            ``(f, t, 1)``      unchanged            error
    ``time_freq_segments``    error            ``(1, f, t, 1)``   ``(1, f, t, ch)``    unchanged
    ========================  ===============  =================  ===================  ==================

    Examples
    --------
    Promote a 1-D recording to canonical 2-D ``'time'`` shape:

    >>> import numpy as np
    >>> from medusa.core.utils import check_data_dims
    >>> out, inserted = check_data_dims(np.zeros(1000), rep_type='time')
    >>> out.shape, inserted
    ((1000, 1), (1,))

    Promote a 2-D ``(n_samples, n_channels)`` array to
    ``'time_segments'`` by prepending a singleton segments axis:

    >>> out, inserted = check_data_dims(
    ...     np.zeros((1000, 16)), rep_type='time_segments'
    ... )
    >>> out.shape, inserted
    ((1, 1000, 16), (0,))

    A 4-D array under ``'time_freq_segments'`` is already canonical, so
    nothing is inserted:

    >>> out, inserted = check_data_dims(
    ...     np.zeros((5, 40, 1000, 16)), rep_type='time_freq_segments'
    ... )
    >>> out.shape, inserted
    ((5, 40, 1000, 16), ())

    Restoring the original shape after processing — pass ``inserted``
    straight back to :func:`numpy.squeeze`:

    >>> data = np.zeros(1000)
    >>> out, inserted = check_data_dims(data, rep_type='time')
    >>> out.shape
    (1000, 1)
    >>> # ... do shape-preserving processing on `out` ...
    >>> restored = np.squeeze(out, axis=inserted) if inserted else out
    >>> restored.shape
    (1000,)
    """
    data = np.asarray(data)

    table = _INSERTED_AXES.get(rep_type)
    if table is None:
        raise ValueError(
            f"check_data_dims: unknown rep_type {rep_type!r}. "
            f"Expected one of {list(_INSERTED_AXES)}."
        )

    inserted = table.get(data.ndim)
    if inserted is None:
        raise ValueError(
            f"check_data_dims: input shape {data.shape} is not "
            f"acceptable for rep_type={rep_type!r}."
        )
    out = np.expand_dims(data, axis=inserted)
    if len(inserted) > 0:
        warnings.warn(
            f"check_data_dims: input shape {data.shape} was "
            f"automatically promoted to {out.shape} "
            f"(rep_type={rep_type!r}). Pass a {out.ndim}-D array to "
            f"suppress this warning.",
            UserWarning,
            stacklevel=2,
        )
    return out, inserted


class ThreadWithReturnValue(Thread):
    """:class:`Thread` subclass whose :meth:`join` returns the target's value.

    Behaves exactly like :class:`threading.Thread`, but the value returned by
    the ``target`` callable is captured and handed back by :meth:`join`,
    avoiding the usual shared-mutable-state workaround for collecting results
    from worker threads.

    Parameters
    ----------
    group
        Reserved for future extension; must be ``None`` (as in
        :class:`threading.Thread`).
    target
        Callable invoked in the worker thread. Its return value is captured
        and returned by :meth:`join`.
    name
        Thread name.
    args
        Positional arguments passed to ``target``.
    kwargs
        Keyword arguments passed to ``target``. ``None`` is treated as an
        empty mapping.

    Examples
    --------
    >>> from medusa.core.utils import ThreadWithReturnValue
    >>> t = ThreadWithReturnValue(target=sum, args=([1, 2, 3],))
    >>> t.start()
    >>> t.join()
    6
    """

    def __init__(self, group: None = None,
                 target: Callable[..., Any] | None = None,
                 name: str | None = None,
                 args: Iterable[Any] = (),
                 kwargs: dict[str, Any] | None = None) -> None:
        if kwargs is None:
            kwargs = {}
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self) -> None:
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)

    def join(self, *args: float | None) -> object:
        Thread.join(self, *args)
        return self._return