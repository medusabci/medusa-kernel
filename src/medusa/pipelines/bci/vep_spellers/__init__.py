"""Visual-evoked-potential (VEP) spellers: c-VEP, SSVEP and ERP/P300.

These paradigms share one stimulation model -- a per-command *codebook* (see
:mod:`~medusa.pipelines.bci.vep_spellers.encoding`) -- and split only by scoring strategy
(template-matching vs bit-wise reconstruction). Their Layer-1
:class:`~medusa.pipelines.base.DecodingPipeline`\\ s emit per-segment scores; the
Layer-2 ``VEPCommandDecoder`` maps scores to commands.

Both scoring families are here: bit-wise reconstruction (``BWRLDAPipeline``; ERP/P300 +
c-VEP-BWR) and template matching (``TMCCAPipeline``, with a ``reference`` mode -- calibration-
free SSVEP via sin/cos harmonics, or a calibrated learned template for c-VEP / SSVEP). Each
Layer-1 pipeline emits a cumulative ``(n_cycles, n_commands)`` score matrix and the
paradigm-agnostic ``VEPCommandDecoder`` selects the command per trial.
"""

from medusa.pipelines.bci.vep_spellers.data import (
    CommandInfo,
    SpellerData,
    SPELLER_EVENT_COLUMNS,
    validate_speller_events,
)
from medusa.pipelines.bci.vep_spellers.encoding import (
    generate_row_col_codebook,
    generate_freq_codebook,
    generate_mseq_codebook,
    generate_gold_codebook,
    get_optimal_frequencies,
    plot_codebook,
    LFSR,
    GOLD_CODES,
)
from medusa.pipelines.bci.vep_spellers.decoding import (
    BWRLDAPipeline,
    TMCCAPipeline,
    VEPCommandDecoder,
    bwr_labels,
    bwr_command_scores,
    tm_command_scores,
    select_commands,
    command_decoding_accuracy,
    command_decoding_accuracy_per_cycle,
)

__all__ = [
    # data model
    "CommandInfo",
    "SpellerData",
    "SPELLER_EVENT_COLUMNS",
    "validate_speller_events",
    # encoding (codebook framework)
    "generate_row_col_codebook",
    "generate_freq_codebook",
    "generate_mseq_codebook",
    "generate_gold_codebook",
    "get_optimal_frequencies",
    "plot_codebook",
    "LFSR",
    "GOLD_CODES",
    # decoding (L1 scoring pipelines + L2 command decoder)
    "BWRLDAPipeline",
    "TMCCAPipeline",
    "VEPCommandDecoder",
    "bwr_labels",
    "bwr_command_scores",
    "tm_command_scores",
    "select_commands",
    "command_decoding_accuracy",
    "command_decoding_accuracy_per_cycle",
]
