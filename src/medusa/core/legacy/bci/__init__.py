"""Legacy BCI paradigm *data* classes (1.x persistence compatibility).

Each module here holds only the ``ExperimentData`` subclass needed to *load*
a paradigm recording (cvep / erp-rcp / mi / ssvep / nft); processing, ML,
datasets and models stay in ``medusa.pipelines.bci``. Imported lazily by the
legacy loader via ``_LEGACY_MODULE_MAP``.
"""

from medusa.core.legacy.bci.cvep_spellers import CVEPSpellerData
from medusa.core.legacy.bci.erp_spellers import ERPSpellerData
from medusa.core.legacy.bci.mi_paradigms import MIData
from medusa.core.legacy.bci.ssvep_spellers import SSVEPSpellerData
from medusa.core.legacy.bci.nft_paradigms import NeurofeedbackData

__all__ = [
    "CVEPSpellerData",
    "ERPSpellerData",
    "MIData",
    "SSVEPSpellerData",
    "NeurofeedbackData",
]
