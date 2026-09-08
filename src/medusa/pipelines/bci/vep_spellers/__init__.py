"""Visual-evoked-potential (VEP) spellers: c-VEP, SSVEP and ERP/P300.

These paradigms share one stimulation model -- a per-command *codebook* (see
:mod:`~medusa.pipelines.bci.vep_spellers.encoding`) -- and split only by scoring strategy
(template-matching vs bit-wise reconstruction). Their Layer-1
:class:`~medusa.pipelines.base.DecodingPipeline`\\ s emit per-segment scores; the
Layer-2 ``select_commands`` maps scores to commands.

Both scoring families are here: bit-wise reconstruction (``BWRLDAPipeline``; ERP/P300 +
c-VEP-BWR) and template matching (``TMCCAPipeline``, with a ``reference`` mode named for what
the reference is made of -- ``synthetic_harmonics`` (calibration-free SSVEP),
``calibrated_template`` (learned template; c-VEP + SSVEP), or ``mixed_harmonics_template``
(the two fused, extended-CCA / eCCA; calibrated SSVEP)). Each
Layer-1 pipeline emits a cumulative ``(n_cycles, n_commands)`` score matrix and the
paradigm-agnostic ``select_commands`` picks the command per trial, reading the per-cycle
indices it needs from ``cycle_arrays(recording.events)``.

The ``decoding`` subpackage keeps one pipeline per module
(:mod:`~medusa.pipelines.bci.vep_spellers.decoding`). The deep BWR pipeline
``BWREEGInceptionPipeline`` (EEG-Inception v1/v2 frame classifier) lives in
:mod:`~medusa.pipelines.bci.vep_spellers.decoding.bwr_eeg_inception` and is imported
**lazily**: it needs torch, so it is resolved only when accessed
(``vep_spellers.BWREEGInceptionPipeline``), keeping ``import
medusa.pipelines.bci.vep_spellers`` torch-free for headless installs.
"""

from medusa.pipelines.bci.vep_spellers.data import (
    CommandInfo,
    SpellerData,
    SPELLER_EVENT_COLUMNS,
    validate_speller_events,
    cycle_arrays,
)
from medusa.pipelines.bci.vep_spellers.encoding import (
    generate_row_col_codebook,
    generate_freq_codebook,
    generate_mseq_codebook,
    generate_gold_codebook,
    generate_random_codebook,
    get_optimal_frequencies,
    plot_codebook,
    LFSR,
    GOLD_CODES,
)
from medusa.pipelines.bci.vep_spellers.decoding import (
    tm_cca_settings,
    zerocal_ssvep_settings,
    cal_ssvep_settings,
    cvep_settings,
    uniform_weights,
    decaying_power_law_weights,
    BWRLDAPipeline,
    TMCCAPipeline,
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
    "cycle_arrays",
    # encoding (codebook framework)
    "generate_row_col_codebook",
    "generate_freq_codebook",
    "generate_mseq_codebook",
    "generate_gold_codebook",
    "generate_random_codebook",
    "get_optimal_frequencies",
    "plot_codebook",
    "LFSR",
    "GOLD_CODES",
    # decoding (L1 scoring pipelines + L2 selector)
    "BWRLDAPipeline",
    "TMCCAPipeline",
    "tm_cca_settings",
    "zerocal_ssvep_settings",
    "cal_ssvep_settings",
    "cvep_settings",
    "uniform_weights",
    "decaying_power_law_weights",
    "bwr_labels",
    "bwr_command_scores",
    "tm_command_scores",
    "select_commands",
    "command_decoding_accuracy",
    "command_decoding_accuracy_per_cycle",
    # deep decoding + its stimulation profiles (torch-gated; resolved lazily
    # via __getattr__)
    "BWREEGInceptionPipeline",
    "bwr_eeg_inception_settings",
    "mseq_cvep_settings",
    "burst_cvep_settings",
]


#: Everything that lives in the torch-gated ``bwr_eeg_inception`` module: the pipeline and
#: its configuration profiles. Resolved on first access by ``__getattr__`` below, so
#: importing this package never pulls torch.
_TORCH_GATED = (
    "BWREEGInceptionPipeline",
    "bwr_eeg_inception_settings",
    "mseq_cvep_settings",
    "burst_cvep_settings",
)


def __getattr__(name: str):
    """Resolve the torch-gated deep pipeline and its profiles lazily (PEP 562).

    Importing this package must stay torch-free (headless installs), so everything in
    :mod:`~medusa.pipelines.bci.vep_spellers.decoding.bwr_eeg_inception` -- which pulls torch
    through its module -- is only imported when it is actually accessed.
    """
    if name in _TORCH_GATED:
        from medusa.pipelines.bci.vep_spellers.decoding import bwr_eeg_inception
        return getattr(bwr_eeg_inception, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}.")
