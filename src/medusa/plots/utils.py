"""Shared plotting utilities for ``medusa.plots``.

Figure-level helpers used across the package:

- :func:`save_figure` — publication-ready export (transparent / solid background,
  300 dpi, tight bbox, auto-mkdir). Plotting functions return plain matplotlib
  values (``(ax, artists)``), so exporting is a one-liner over the parent figure:
  ``save_figure(ax.figure, "fig.png", transparent=True)``.
- :func:`optimal_grid` — rows/columns for an as-square-as-possible subplot grid.
- :func:`_add_colorbar` — the shared compact-colorbar helper behind the
  ``colorbar``/``cbar_*`` API of the color-scale plots.
- :func:`_resolve_colors` / :func:`_resolve_cmap` — normalize the per-series
  ``color`` and the ``cmap`` arguments shared by the multi-series / color-scale
  plots.

medusa-kernel owns no visual identity of its own: the palette, categorical
cycle, colormaps and Matplotlib ``rcParams`` all come from the ``medusa_style``
package (the MEDUSA-wide single source of truth). Plotting code calls it
directly where a color is needed — ``medusa_style.categorical_color(i)`` for a
series color, ``medusa_style.current_theme().plot_fg`` / ``.plot_bg``
for chrome, ``medusa_style.mpl.sequential_cmap()`` / ``diverging_cmap()`` for the
colormaps — rather than going through kernel-side wrappers.
"""

import math
from pathlib import Path
from typing import Any

import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Colormap, is_color_like
from matplotlib.figure import Figure

__all__ = ["save_figure", "optimal_grid"]


def save_figure(
    fig: Figure,
    path: str | Path,
    *,
    transparent: bool = False,
    facecolor: str | None = None,
    dpi: int = 300,
    bbox_inches: str | None = "tight",
    pad_inches: float = 0.05,
    **savefig_kwargs: Any,
) -> Path:
    """Save a figure as a publication-ready image.

    The format is inferred from the suffix; missing parent directories are created.

    Parameters
    ----------
    fig
        Figure to export.
    path
        Destination path; the suffix selects the format.
    transparent
        Export a transparent canvas (takes precedence over ``facecolor``).
    facecolor
        Solid background to force; the figure's current face color when ``None``.
    dpi
        Output resolution (default 300).
    bbox_inches, pad_inches
        Bounding-box mode (``'tight'`` trims whitespace) and its padding.
    **savefig_kwargs
        Forwarded to ``Figure.savefig``.

    Returns
    -------
    pathlib.Path
        The path that was written.

    Examples
    --------
    >>> save_figure(fig, "fig.png", transparent=True)  # doctest: +SKIP
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if transparent:
        fig.savefig(
            path,
            dpi=dpi,
            transparent=True,
            bbox_inches=bbox_inches,
            pad_inches=pad_inches,
            **savefig_kwargs,
        )
    else:
        fc = facecolor if facecolor is not None else fig.get_facecolor()
        fig.savefig(
            path,
            dpi=dpi,
            transparent=False,
            facecolor=fc,
            edgecolor=fc,
            bbox_inches=bbox_inches,
            pad_inches=pad_inches,
            **savefig_kwargs,
        )
    return path


def _prime_factors(n: int) -> list[int]:
    """Prime factorization of ``n`` (ascending). ``_prime_factors(1) == []``."""
    factors: list[int] = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors


def optimal_grid(n_items: int) -> tuple[int, int]:
    """Rows and columns for an as-square-as-possible subplot grid of ``n_items``.

    The prime factorization is split into two balanced halves whose products
    become the columns and rows. A prime count (which would otherwise give a
    ``1 x N`` strip) is gridded as ``n_items - 1`` with one extra column, leaving
    a single empty cell.

    Parameters
    ----------
    n_items
        Number of subplots to arrange (``>= 1``).

    Returns
    -------
    rows, cols : int
        Grid dimensions; ``rows * cols >= n_items``.

    Examples
    --------
    >>> optimal_grid(6)
    (3, 2)
    >>> optimal_grid(7)  # prime -> 3x3 with one empty cell
    (3, 3)
    """
    if n_items < 1:
        raise ValueError("n_items must be >= 1")
    factors = _prime_factors(n_items)
    if len(factors) == 1:  # n_items is prime -> grid n_items-1 + one extra column
        factors = _prime_factors(n_items - 1)
        m = round(len(factors) / 2)
        return math.prod(factors[m:]), math.prod(factors[:m]) + 1
    m = round(len(factors) / 2)
    return math.prod(factors[m:]), math.prod(factors[:m])


def _add_colorbar(mappable: ScalarMappable, ax: Axes, *,
                  label: str | None = None, fraction: float = 0.05,
                  pad: float = 0.02, cbar=None):
    """Add a compact colorbar to ``ax`` for ``mappable`` (or update an existing one).

    Shared by the color-scale plots (topography, connectivity, heatmap) so they
    expose the same ``colorbar``/``cbar_label``/``cbar_fraction``/``cbar_pad`` API.
    ``fraction`` is the share of the axes width given to the bar (matplotlib's
    default ``0.15`` is wider — hence a fat right margin); ``pad`` is the gap
    between the axes and the bar. Pass a previously-returned colorbar as ``cbar``
    to re-point it at a new ``mappable`` in place (e.g. after a live data change or
    a rebuild) instead of adding a second bar — this keeps the axes from shrinking
    again.

    Returns
    -------
    matplotlib.colorbar.Colorbar
        The created (or updated) colorbar.
    """
    if cbar is None:
        return ax.figure.colorbar(mappable, ax=ax, fraction=fraction, pad=pad,
                                  label=label)
    cbar.update_normal(mappable)
    return cbar


def _resolve_colors(color, n: int) -> list | None:
    """Resolve a per-series ``color`` argument to a list of ``n`` colors.

    ``None`` returns ``None`` (the caller applies its own default, e.g. the
    medusa-style categorical cycle); a single color is repeated ``n`` times; a
    sequence is returned as-is (must have length ``n``). Lets every multi-series
    plot override the default colors.
    """
    if color is None:
        return None
    if is_color_like(color):
        return [color] * n
    colors = list(color)
    if len(colors) != n:
        raise ValueError(
            f"color must be a single color or {n} colors, got {len(colors)}.")
    return colors


def _resolve_cmap(cmap, default: Colormap) -> Colormap:
    """Resolve a user colormap (name / object / ``None``) to a ``Colormap``.

    ``None`` falls back to ``default`` — an already-resolved ``Colormap`` (pass
    e.g. ``medusa_style.mpl.sequential_cmap()``); a string is looked up in
    ``matplotlib.colormaps``; a ``Colormap`` is returned as-is. Internal helper
    shared by the color-scale plots (topography, connectivity, heatmap).
    """
    if cmap is None:
        return default
    return mpl.colormaps[cmap] if isinstance(cmap, str) else cmap
