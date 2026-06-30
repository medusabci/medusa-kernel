---
hide:
  - navigation
  - toc
---

<div class="md-hero" markdown="1">

# medusa-kernel

Signal-processing core of the **MEDUSA©** ecosystem — a pure-Python
toolkit for biomedical signals (EEG, MEG, ECG, EMG, EOG, NIRS) used by
the MEDUSA Platform, the MEDUSA Analyzer, and any researcher who
`pip install`s it.

[Get started :material-arrow-right:](installation.md){ .md-button .md-button--primary }
[Browse tutorials :material-notebook-outline:](tutorials/index.md){ .md-button }
[GitHub :material-github:](https://github.com/medusabci/medusa-kernel){ .md-button }

</div>

## Why medusa-kernel

<div class="grid cards" markdown>

-   :material-function-variant:{ .lg .middle } **Functional architecture**

    ---
    Free functions on NumPy arrays — every input shows up in the
    signature. No hidden state on container classes.

-   :material-cube-outline:{ .lg .middle } **Clean modality split**

    ---
    `EEG`, `ECG`, `EMG`, `EOG`, `NIRS` keep their own rich schema instead
    of collapsing into a generic union container.

-   :material-graph-outline:{ .lg .middle } **From samples to BCI pipelines**

    ---
    Filters, transforms, signal & graph metrics, ML utilities and
    end-to-end ERP / c-VEP / SSVEP / MI / NFT paradigms — one library.

-   :material-rocket-launch:{ .lg .middle } **Just `pip install`**

    ---
    Headless-friendly, cross-platform (Linux / macOS / Windows). The Qt
    viewers (PySide6) ship as a core dependency — no extra needed.

</div>

## Quick links

<div class="grid cards" markdown>

-   :material-download:{ .lg .middle } **Installation**

    ---
    Install from PyPI, manage optional extras (`dev`, `docs`) and the
    manual PyTorch step.

    [:octicons-arrow-right-24: Read the install guide](installation.md)

-   :material-rocket-launch:{ .lg .middle } **Quickstart**

    ---
    A 5-minute end-to-end example: filter, extract a metric, plot.

    [:octicons-arrow-right-24: Get started](quickstart.md)

-   :material-notebook-outline:{ .lg .middle } **Tutorials**

    ---
    Curated Jupyter notebooks with one-click *Open in Colab* — algorithm
    basics, connectivity, ERP / c-VEP / MI BCI paradigms.

    [:octicons-arrow-right-24: Browse tutorials](tutorials/index.md)

-   :material-api:{ .lg .middle } **API reference**

    ---
    Auto-generated reference for every public module: `signal/`, `core/`,
    `pipelines/`, `ml/`, `graph/`, `plots/`, `widgets/`.

    [:octicons-arrow-right-24: Open API reference](api/medusa/index.md)

-   :material-source-pull:{ .lg .middle } **Contributing**

    ---
    Coding conventions, tests, public-API rules, license headers.

    [:octicons-arrow-right-24: Contribute](contributing.md)

</div>

## License

Apache License 2.0 — see `LICENSE` and `NOTICE` in the repository root.
