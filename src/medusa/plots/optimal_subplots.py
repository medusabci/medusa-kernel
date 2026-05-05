import math


def factor(n):
    """ Computes an irreducible factorization of a number.

    Parameter
    ----------
    n : int
        Number to factorize

    Returns
    ---------
    list:
        List of factors
    """
    f = lambda n: (p := [
        next(i for i in range(2, n + 1) if n % i == 0)] if n > 1 else []) + (
                      f(n // p[0]) if p else [])
    return f(n)


def optimal_subplot_row_col(n_items):
    """ Computes the optimal arrangement of items in rows and columns.

    Parameter
    ----------
    n_items : int
        Number of items.

    Returns
    ---------
    int, int:
        Optimal number of rows and columns, respectively
    """
    f = factor(n_items)
    if len(f) == 1:
        f = factor(n_items - 1)
        m_ = round(len(f) / 2)
        sub_c = math.prod(f[:m_]) + 1
        sub_r = math.prod(f[m_:])
    else:
        m_ = round(len(f) / 2)
        sub_c = math.prod(f[:m_])
        sub_r = math.prod(f[m_:])
    return sub_r, sub_c
