# Tutorials

A curated set of executable Jupyter notebooks that walk through real
analyses with `medusa-kernel`. Every tutorial ships with a one-click
**Open in Colab** button at the top of the page, so you can run it without
installing anything locally — the notebook will fetch its dataset directly
from this repository.

Prefer to run them locally? Each page also exposes a **Download .ipynb**
button.

## Getting started

<div class="grid cards" markdown>

-   :material-database:{ .lg .middle } **Recordings & streaming**

    ---

    The `core.data` runtime model: build a `Recording` (typed channels,
    signals, events, metadata), save/load across formats, and stream data
    live with a `Recorder`.

    [:octicons-arrow-right-24: Open tutorial](recordings_tut.ipynb)

-   :material-chart-box:{ .lg .middle } **Visualization**

    ---

    The reusable `medusa.plots` plots: the MEDUSA style, scalp maps
    (head, topography, connectivity), time-axis views (stacked traces,
    heatmaps) and mean ± CI bands.

    [:octicons-arrow-right-24: Open tutorial](visualization.ipynb)

-   :material-history:{ .lg .middle } **Loading legacy data**

    ---

    Read 1.x recordings (`.rec/.cvep/.mi/.rcp.bson`) with
    `medusa.core.legacy` and bridge them into the 2.0 `core.data` model so
    the modern signal, plotting and ML stack works on old data.

    [:octicons-arrow-right-24: Open tutorial](loading_legacy_data.ipynb)

</div>

## Signal analysis

<div class="grid cards" markdown>

-   :material-chart-line:{ .lg .middle } **Local activation**

    ---

    Spectral and nonlinear metrics applied to resting-state EEG: band
    power, spectral entropy, sample entropy, …

    [:octicons-arrow-right-24: Open tutorial](local_activation_analysis.ipynb)

-   :material-chart-bell-curve:{ .lg .middle } **Functional connectivity**

    ---

    Phase-based (PLV, PLI, wPLI) and amplitude-based (AEC) connectivity
    metrics, plus surrogate-based statistical thresholding.

    [:octicons-arrow-right-24: Open tutorial](connectivity_analysis.ipynb)

-   :material-eye-off:{ .lg .middle } **Artifact rejection**

    ---

    Remove ocular (EOG) artifacts from synthetic EEG using regression,
    ICA and amplitude-threshold epoch rejection, with topographic plots.

    [:octicons-arrow-right-24: Open tutorial](artifact_rejection.ipynb)

</div>

## Machine learning

<div class="grid cards" markdown>

-   :material-brain:{ .lg .middle } **Train a PyTorch classifier**

    ---

    Train an `EEGNet` backbone with `TorchClassifier` (a scikit-learn
    estimator) on a synthetic ERP dataset: fit, evaluate (confusion
    matrix, ROC), and save/load the model.

    [:octicons-arrow-right-24: Open tutorial](train_torch_classifier.ipynb)

</div>

## Data

Most tutorials generate their data on the fly with
`medusa.signal.generators`, so they run anywhere with no download. The
*Loading legacy data* tutorial reads the example 1.x recordings under
`docs/tutorials/data/legacy/{rest_eeg,cvep,mi,rcp_speller}/`. Those files,
and any artefacts the notebooks write to `docs/tutorials/results/`, are
git-ignored and excluded from the published site
(`mkdocs.yml :: exclude_docs`).

