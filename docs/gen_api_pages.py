"""Auto-generate one ``docs/api/<module>.md`` page per public module.

Run by ``mkdocs-gen-files`` at build time. Walks ``src/medusa/`` and emits
a page per module that contains a single ``mkdocstrings`` ``:::`` directive,
plus a ``SUMMARY.md`` consumed by ``mkdocs-literate-nav``.

Modules / packages whose name starts with ``_`` are considered private and
skipped. ``__init__.py`` files are mapped to their package's index page
(which lists the package symbols plus a brief description). Leaf module pages
carry a ``toc_depth: 3`` front-matter so that the auto-generated member
headings are visible in the right-hand TOC.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import mkdocs_gen_files

SRC = Path(__file__).parent.parent / "src"
PACKAGE_ROOT = SRC / "medusa"

# ── Human-readable sub-package descriptions shown on index pages ─────────────
_PKG_DESCRIPTIONS: dict[str, str] = {
    "medusa": (
        "Root package — exposes `__version__`; import the subpackages directly "
        "(`medusa.signal`, `medusa.core`, …)."
    ),
    "medusa.core": (
        "Domain types, serialisation helpers, pipeline and profiling utilities."
    ),
    "medusa.signal": (
        "Low-level signal processing: filters, transforms, segmentation, "
        "artifact removal, signal generators."
    ),
    "medusa.signal.metrics": "Signal / biosignal metrics.",
    "medusa.signal.metrics.spectral": "Spectral metrics (band power, SEF, median frequency, …).",
    "medusa.signal.metrics.nonlinear": (
        "Nonlinear / information-theoretic metrics (entropy, LZC, …)."
    ),
    "medusa.signal.metrics.connectivity": (
        "Phase-based and amplitude-based connectivity metrics (PLV, PLI, wPLI, AEC, IAC)."
    ),
    "medusa.signal.metrics.discriminability": "Discriminability metrics (signed r²).",
    "medusa.graph": "Graph-theoretic metrics (clustering, path length, centrality, …).",
    "medusa.ml": (
        "Machine-learning utilities and PyTorch models "
        "(``torch_models`` backbones + classifiers)."
    ),
    "medusa.plots": (
        "Publication-ready matplotlib plots — scalp topographies & "
        "connectivity, time-series lines & heatmaps, shaded-line summaries. "
        "Styled via the shared ``medusa_style`` package."
    ),
    "medusa.widgets": "Qt-based interactive widgets (settings tree editor, time-plot viewer).",
    "medusa.pipelines": "High-level BCI paradigm pipelines.",
    "medusa.pipelines.bci": (
        "ERP, c-VEP, SSVEP, MI and NFT BCI paradigms, shared performance metrics."
    ),
}

nav = mkdocs_gen_files.Nav()

for path in sorted(PACKAGE_ROOT.rglob("*.py")):
    module_path = path.relative_to(SRC).with_suffix("")
    parts = list(module_path.parts)

    # Skip private modules / sub-packages anywhere in the path.
    if any(part.startswith("_") and part != "__init__" for part in parts):
        continue

    # Skip the internal 1.x load-only compatibility layer. `medusa.core.legacy`
    # is not public API (see the Contributing page) and must not appear in the
    # generated reference.
    if "legacy" in parts:
        continue

    # Skip dead/superseded styling modules. The kernel no longer owns styling
    # (it consumes the external `medusa_style` package); `plots/style.py` and
    # `plots/styles/**` are pending removal and mkdocstrings cannot even
    # collect them. Drop this guard once those files are deleted.
    if "plots" in parts and ("styles" in parts or parts[-1] == "style"):
        continue

    is_init = parts[-1] == "__init__"

    if is_init:
        parts = parts[:-1]
        if not parts:
            continue
        doc_path = Path("api", *parts, "index.md")
        nav_parts = tuple(parts[1:]) or (parts[0],)
    else:
        doc_path = Path("api", *parts).with_suffix(".md")
        nav_parts = tuple(parts[1:])

    if not nav_parts:
        nav_parts = ("Overview",)

    ident = ".".join(parts)

    nav[nav_parts] = doc_path.relative_to("api").as_posix()

    # ── Build the page content ────────────────────────────────────────────────
    desc = _PKG_DESCRIPTIONS.get(ident, "")

    if is_init:
        # Package index page: wider TOC (depth 2) + optional description.
        front_matter = textwrap.dedent("""\
            ---
            toc_depth: 2
            ---
        """)
        description_block = f"\n{desc}\n" if desc else ""
        content = (
            f"{front_matter}\n"
            f"# `{ident}`\n"
            f"{description_block}\n"
            f"::: {ident}\n"
        )
    else:
        # Leaf module page: deep TOC (depth 3) shows function/class members.
        front_matter = textwrap.dedent("""\
            ---
            toc_depth: 3
            ---
        """)
        content = (
            f"{front_matter}\n"
            f"# `{ident}`\n\n"
            f"::: {ident}\n"
        )

    with mkdocs_gen_files.open(doc_path, "w") as f:
        f.write(content)

    mkdocs_gen_files.set_edit_path(doc_path, path.relative_to(SRC.parent))

with mkdocs_gen_files.open("api/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())

