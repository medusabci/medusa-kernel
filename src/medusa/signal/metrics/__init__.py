"""Signal metrics: signal -> scalar/vector extractions, grouped by family.
Buckets are organized by *operational question* (what the metric tells you),
not by mathematical lineage:
- ``spectral``        : what frequencies dominate?
- ``nonlinear``       : how regular / structured is the dynamics?
- ``discriminability``: how well does this feature separate classes?
- ``connectivity``    : how do channels relate to each other?
See ``TODO.md`` K1 for the rationale.
"""
from medusa.signal.metrics import (
    connectivity,
    discriminability,
    nonlinear,
    spectral,
)

__all__ = ["spectral", "nonlinear", "discriminability", "connectivity"]
