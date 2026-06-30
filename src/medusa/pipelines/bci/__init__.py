"""Brain-Computer Interface (BCI) paradigm pipelines.

Each module bundles the data containers, processing methods (preprocessing /
feature extraction) and models for one BCI paradigm:

- :mod:`~medusa.pipelines.bci.cvep_spellers` — code-modulated VEP spellers.
- :mod:`~medusa.pipelines.bci.erp_spellers` — event-related potential spellers.
- :mod:`~medusa.pipelines.bci.mi_paradigms` — motor imagery paradigms.
- :mod:`~medusa.pipelines.bci.nft_paradigms` — neurofeedback paradigms.
- :mod:`~medusa.pipelines.bci.ssvep_spellers` — steady-state VEP spellers.

The paradigms intentionally share class names (every speller module defines its
own ``StandardPreprocessing`` / ``StandardFeatureExtraction`` and command
decoding helpers), so they are re-exported here as *namespaces* rather than
flattened into this package::

    from medusa.pipelines.bci import erp_spellers
    model = erp_spellers.CMDModelRLDA()

Paradigm-agnostic performance metrics live in
:mod:`~medusa.pipelines.bci.performance` (``itr`` is re-exported for
convenience) and plotting helpers in :mod:`~medusa.pipelines.bci.plots`.
"""

from medusa.pipelines.bci import (
    cvep_spellers,
    erp_spellers,
    mi_paradigms,
    nft_paradigms,
    ssvep_spellers,
    performance,
    plots,
)
from medusa.pipelines.bci.performance import itr

__all__ = [
    # Paradigm namespaces
    "cvep_spellers",
    "erp_spellers",
    "mi_paradigms",
    "nft_paradigms",
    "ssvep_spellers",
    # Performance metrics
    "performance",
    "itr",
    # Plotting helpers
    "plots",
]
