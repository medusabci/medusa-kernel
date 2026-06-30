"""Plotting helpers for the BCI paradigm pipelines (ERP and MI).

These functions take the paradigm datasets / feature containers produced by
:mod:`medusa.pipelines.bci` and render quick, paradigm-specific diagnostics
(ERP waveforms, ERD/ERS maps, r-squared topographies). They consume matplotlib
directly and have no Qt dependency.
"""

from medusa.pipelines.bci.plots.erp import (
    plot_erp,
    plot_erp_from_erp_speller_dataset,
    compute_dev_epochs,
)
from medusa.pipelines.bci.plots.mi import (
    MIPlots,
    plot_erd_ers_time,
    plot_erd_ers_freq,
    plot_r2_topoplot,
)

__all__ = [
    # ERP plots
    "plot_erp",
    "plot_erp_from_erp_speller_dataset",
    "compute_dev_epochs",
    # MI plots
    "MIPlots",
    "plot_erd_ers_time",
    "plot_erd_ers_freq",
    "plot_r2_topoplot",
]
