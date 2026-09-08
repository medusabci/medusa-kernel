"""Decoding for VEP spellers: Layer-1 scoring pipelines and the Layer-2 command selector.

Two layers, one shared contract, split across one module per pipeline plus small shared
helper modules:

* **Layer 1** turns EEG into a *cumulative* ``(n_cycles, n_commands)`` **score matrix**. Row
  ``i`` is the decision score for every command after cycle event ``i``, added up over that
  trial's cycles the way the model works best. Each pipeline is a **direct**
  :class:`~medusa.pipelines.base.DecodingPipeline` (no shared strategy base -- their feature
  paths have nothing in common), and lives in its own module:

  - :mod:`~medusa.pipelines.bci.vep_spellers.decoding.bwr_lda` --
    :class:`~medusa.pipelines.bci.vep_spellers.decoding.bwr_lda.BWRLDAPipeline`: classify each
    code frame with LDA, then correlate a command's code with the frame scores.
  - :mod:`~medusa.pipelines.bci.vep_spellers.decoding.bwr_eeg_inception` --
    :class:`~medusa.pipelines.bci.vep_spellers.decoding.bwr_eeg_inception.BWREEGInceptionPipeline`:
    the same BWR strategy with an EEG-Inception (v1 or v2) frame classifier (**torch-gated**,
    imported lazily so this package stays torch-free). Which band, window and kernel scales
    suit the data depends on the stimulation, so it ships **profiles** -- one ready settings
    tree per paradigm:
    :func:`~medusa.pipelines.bci.vep_spellers.decoding.bwr_eeg_inception.mseq_cvep_settings`
    (dense codes) and
    :func:`~medusa.pipelines.bci.vep_spellers.decoding.bwr_eeg_inception.burst_cvep_settings`
    (sparse bursts), or
    :func:`~medusa.pipelines.bci.vep_spellers.decoding.bwr_eeg_inception.bwr_eeg_inception_settings`
    to write your own.
  - :mod:`~medusa.pipelines.bci.vep_spellers.decoding.template_matching` --
    :class:`~medusa.pipelines.bci.vep_spellers.decoding.template_matching.TMCCAPipeline`:
    coherently average each command's cycle segments, then score against a synthetic
    harmonic bank (CCA, calibration-free SSVEP), a learned template (c-VEP / SSVEP), or the
    two fused (``mixed_harmonics_template``, extended-CCA / eCCA). It has no single sensible
    configuration, so it ships **profiles** -- one ready settings tree per paradigm:
    :func:`~medusa.pipelines.bci.vep_spellers.decoding.template_matching.zerocal_ssvep_settings`,
    :func:`~medusa.pipelines.bci.vep_spellers.decoding.template_matching.cal_ssvep_settings`
    and
    :func:`~medusa.pipelines.bci.vep_spellers.decoding.template_matching.cvep_settings`
    (or
    :func:`~medusa.pipelines.bci.vep_spellers.decoding.template_matching.tm_cca_settings`
    to write your own).

* **Layer 2** (:mod:`~medusa.pipelines.bci.vep_spellers.decoding.command_decoder`,
  :func:`~medusa.pipelines.bci.vep_spellers.decoding.command_decoder.select_commands`) is a
  *pure selector*: after each cycle it takes the argmax of the cumulative score row over that
  trial's available commands. It is paradigm-agnostic, because Layer 1 already did the
  family-specific accumulation. It is stateless and does no fitting, so it is a plain
  function rather than an object; pair it with
  :func:`~medusa.pipelines.bci.vep_spellers.data.cycle_arrays`, which reads the per-cycle
  trial and repetition indices out of a recording's events.

Shared building blocks: the family-level cumulative accumulators and BWR labels live in
:mod:`~medusa.pipelines.bci.vep_spellers.decoding.scores`
(:func:`~medusa.pipelines.bci.vep_spellers.decoding.scores.bwr_labels`,
:func:`~medusa.pipelines.bci.vep_spellers.decoding.scores.bwr_command_scores`,
:func:`~medusa.pipelines.bci.vep_spellers.decoding.scores.tm_command_scores`). Reading a
recording's events into the per-cycle arrays every layer consumes is
:func:`~medusa.pipelines.bci.vep_spellers.data.cycle_arrays`; the remaining model-agnostic
frame-onset and frequency-filtering plumbing lives in the private ``_common`` module.

Everything below is re-exported at the package level, so the historical import path
``from medusa.pipelines.bci.vep_spellers.decoding import BWRLDAPipeline`` keeps working.
"""

from __future__ import annotations

from medusa.pipelines.bci.vep_spellers.decoding.scores import (
    bwr_labels,
    bwr_command_scores,
    tm_command_scores,
)
from medusa.pipelines.bci.vep_spellers.decoding.command_decoder import (
    select_commands,
    command_decoding_accuracy,
    command_decoding_accuracy_per_cycle,
)
from medusa.pipelines.bci.vep_spellers.decoding.bwr_lda import BWRLDAPipeline
from medusa.pipelines.bci.vep_spellers.decoding.template_matching import (
    TMCCAPipeline,
    tm_cca_settings,
    zerocal_ssvep_settings,
    cal_ssvep_settings,
    cvep_settings,
    uniform_weights,
    decaying_power_law_weights,
)

__all__ = [
    # Layer-1 scoring pipelines
    "BWRLDAPipeline",
    "TMCCAPipeline",
    "BWREEGInceptionPipeline",   # torch-gated; resolved lazily via __getattr__
    # BWREEGInceptionPipeline configuration profiles (one stimulation paradigm each;
    # torch-gated too, so they are resolved lazily as well)
    "bwr_eeg_inception_settings",
    "mseq_cvep_settings",
    "burst_cvep_settings",
    # TMCCAPipeline configuration profiles (one paradigm each)
    "tm_cca_settings",
    "zerocal_ssvep_settings",
    "cal_ssvep_settings",
    "cvep_settings",
    # filter-bank score weights (build the list, then pass it in)
    "uniform_weights",
    "decaying_power_law_weights",
    # Layer-2 selector + metrics (pure functions)
    "bwr_labels",
    "bwr_command_scores",
    "tm_command_scores",
    "select_commands",
    "command_decoding_accuracy",
    "command_decoding_accuracy_per_cycle",
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
