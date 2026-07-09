"""Endogenous (self-generated) BCI paradigms: motor imagery and neurofeedback.

MI and neurofeedback are one family -- SMR-neurofeedback *is* motor-imagery feedback:
the same sensorimotor pipeline emits scores whether the app argmaxes them to a class
(MI) or streams a value (neurofeedback). They are Layer-1
:class:`~medusa.pipelines.base.DecodingPipeline`\\ s only (no stimulation codebook, no
command decoder); the scores are the output.

Populated in Phase 2 of the refactor (see ``pipelines/TODO.md``): ``data``
(``EndogenousData``), ``decoding`` (CSP / EEGSym / band-power / connectivity),
``analysis`` (ERD/ERS, SCP).
"""

__all__: list[str] = []
