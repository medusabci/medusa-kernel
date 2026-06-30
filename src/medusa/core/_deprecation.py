"""Internal helpers for the medusa-kernel deprecation policy.

This is a *private* module (leading underscore): it is infrastructure for
emitting :class:`DeprecationWarning` consistently across the kernel, not part of
the public API itself. The policy it supports (documented in the Contributing
page): a public symbol scheduled for removal warns for one minor release and is
removed in the next.

Use :func:`warn_deprecated` from inside a function body when only part of a
call path is deprecated, or the :func:`deprecated` decorator to mark a whole
function or class.
"""

import functools
import warnings
from collections.abc import Callable
from typing import TypeVar

__all__ = ["deprecated", "warn_deprecated"]

_T = TypeVar("_T", bound=Callable[..., object])


def warn_deprecated(
    name: str,
    *,
    removed_in: str,
    instead: str | None = None,
    stacklevel: int = 2,
) -> None:
    """Emit a standardised :class:`DeprecationWarning`.

    Parameters
    ----------
    name
        Human-readable name of the deprecated symbol (e.g. ``"foo()"`` or
        ``"medusa.signal.bar"``).
    removed_in
        Version in which the symbol will be removed (e.g. ``"2.1"``).
    instead
        Optional replacement to point users to. When given, it is appended to
        the message as ``"; use <instead>"``.
    stacklevel
        Passed through to :func:`warnings.warn`. The default of ``2`` makes the
        warning point at the caller of the deprecated symbol rather than at this
        helper. Increase it when there is an extra wrapper frame in between.

    Examples
    --------
    >>> import warnings
    >>> with warnings.catch_warnings(record=True) as caught:
    ...     warnings.simplefilter("always")
    ...     warn_deprecated("foo()", removed_in="2.1", instead="bar()")
    >>> str(caught[0].message)
    'foo() is deprecated and will be removed in 2.1; use bar()'
    """
    message = f"{name} is deprecated and will be removed in {removed_in}"
    if instead is not None:
        message += f"; use {instead}"
    warnings.warn(message, DeprecationWarning, stacklevel=stacklevel)


def deprecated(
    *,
    removed_in: str,
    instead: str | None = None,
) -> Callable[[_T], _T]:
    """Mark a function or class as deprecated.

    Returns a decorator that wraps the target so that *every* call emits a
    :class:`DeprecationWarning` via :func:`warn_deprecated`, then delegates to
    the original object. The wrapper preserves the wrapped object's metadata
    via :func:`functools.wraps`.

    Parameters
    ----------
    removed_in
        Version in which the symbol will be removed (e.g. ``"2.1"``).
    instead
        Optional replacement to point users to.

    Returns
    -------
    Callable
        A decorator that wraps the deprecated function or class.

    Examples
    --------
    >>> import warnings
    >>> @deprecated(removed_in="2.1", instead="new_func()")
    ... def old_func(x):
    ...     return x + 1
    >>> with warnings.catch_warnings(record=True) as caught:
    ...     warnings.simplefilter("always")
    ...     old_func(1)
    2
    >>> str(caught[0].message)
    'old_func is deprecated and will be removed in 2.1; use new_func()'
    """

    def decorator(obj: _T) -> _T:
        @functools.wraps(obj)
        def wrapper(*args: object, **kwargs: object) -> object:
            warn_deprecated(
                obj.__name__,
                removed_in=removed_in,
                instead=instead,
                stacklevel=3,
            )
            return obj(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator
