"""Reusable, publication-ready matplotlib visualizations for medusa-kernel.

Importing this package has **no side effects** on global matplotlib state. The
visual identity (palette, categorical cycle, colormaps, fonts, ``rcParams``) is
owned by the :mod:`medusa_style` package — the single source of truth shared
across the whole MEDUSA ecosystem — and these plots consume it directly:

>>> import medusa_style
>>> medusa_style.apply()              # theme every new matplotlib figure

medusa-kernel always defaults the active theme to **light** (publication-first),
regardless of medusa-style's own default. Importing this package selects it; a
host application (e.g. the dark medusa-platform GUI) can switch afterwards with
``medusa_style.use_theme("dark")``.

Plotting functions return plain matplotlib values — ``(ax, artists)`` — so you
restyle the named ``artists`` directly (e.g.
``medusa_style.categorical_color(0)``), add a colorbar with ``colorbar=True``, and
export with :func:`save_figure` (transparent-background support).

See ``plots/TODO.md`` for the 2.0 refactor roadmap.
"""

import medusa_style

from medusa.plots.scalp import plot_scalp
from medusa.plots.scalp_connectivity import ConnectivityPlot, plot_connectivity
from medusa.plots.scalp_topography import TopographicPlot, plot_topography
from medusa.plots.shaded_line import plot_shaded_line
from medusa.plots.time_heatmap import TimeHeatmapPlot, plot_timeheatmap
from medusa.plots.time_line import TimeLinePlot, plot_timeline
from medusa.plots.utils import optimal_grid, save_figure

# Kernel policy: the visual layer always defaults to the LIGHT theme. This only
# sets medusa-style's active theme (no global matplotlib mutation, so the
# no-side-effects-on-import contract above holds); a host can switch to "dark".
medusa_style.use_theme("light")

__all__ = [
    # Shared utilities (export / layout)
    "save_figure",
    "optimal_grid",
    # Scalp plots (head substrate + topography + connectivity)
    "plot_scalp",
    "plot_topography",
    "plot_connectivity",
    "TopographicPlot",
    "ConnectivityPlot",
    # Time-axis plots (line + heatmap)
    "plot_timeline",
    "plot_timeheatmap",
    "TimeLinePlot",
    "TimeHeatmapPlot",
    # Shaded-line summary (mean ± error/CI band; ERP, mean PSD, ...)
    "plot_shaded_line",
]
