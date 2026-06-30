# docs/tutorials/

Executable Jupyter notebooks that constitute the narrative documentation of
`medusa-kernel`. Each notebook is also a standalone teaching artefact: it
must run end-to-end on a fresh Colab kernel **and** render correctly as a
static page through MkDocs.

> Repository-wide rules: [`../../AGENTS.md`](../../AGENTS.md). MkDocs build
> contract: `tests.yml` runs `mkdocs build --strict` (see Kernel `TODO.md`
> K10). A malformed notebook fails the build and blocks the release.

---

## 1. Why this folder needs its own AGENTS.md

Tutorials are simultaneously:

- **Documentation pages.** They are rendered by `mkdocs-jupyter` (configured
  in `../../mkdocs.yml`, `plugins.mkdocs-jupyter`, `execute: false`). Their
  cell outputs are shipped *as committed* — the docs site does not
  re-execute them.
- **Runnable scripts.** Users open them in Colab via the "Open in Colab"
  banner or download the `.ipynb` to run locally.
- **JSON files committed to git.** They must be diff-friendly, IDE-friendly
  (PyCharm / VS Code notebook editors), and small enough to render.

Any deviation from the conventions below breaks at least one of those three
roles. The historical bug that motivated this file: a tutorial was
committed as *a single code cell whose `source` was the textual JSON of
the real notebook*. The file looked roughly like:

```jsonc
{
 "cells": [
  {
   "cell_type": "code",
   "source": [
    "\n",
    " \"cells\": [\n",                  // ← actual notebook embedded as text
    "  { \"cell_type\": \"markdown\", \"source\": [...] },\n",
    "  ...\n",
    " ],\n",
    " \"nbformat\": 4\n",
    "}\n"
   ],
   "outputs": [], "execution_count": null, "metadata": {}, "id": "…"
  }
 ],
 "metadata": {}, "nbformat": 4, "nbformat_minor": 5
}
```

It was technically valid JSON and even valid nbformat (one giant code
cell), so `mkdocs build` did not crash, but every renderer showed it as
raw JSON. **Do not let this happen again.** Validate every notebook you
touch (§4).

---

## 2. Canonical notebook structure

Every tutorial is a JSON document with this top-level shape:

```jsonc
{
  "cells": [ /* list of cell objects, in display order */ ],
  "metadata": {
    "kernelspec":    { "display_name": "Python 3", "language": "python", "name": "python3" },
    "language_info": { "name": "python" }
  },
  "nbformat": 4,
  "nbformat_minor": 5
}
```

Rules:

- `cells` is **always a flat list of cell objects**. Never nest a notebook
  inside a cell. Never wrap several cells in a single `source` string.
- `nbformat = 4`, `nbformat_minor = 5` for new tutorials (gives each cell a
  stable `id`, which improves diffs). Existing files at minor `1` are
  acceptable but should be bumped opportunistically when re-saved.
- Top-level `metadata` carries only `kernelspec` and `language_info`. Do
  not embed PyCharm/VS Code per-cell metadata (e.g.
  `{"pycharm": {"name": "#%% cd\n"}}`) — JupyterLab strips these on save,
  but PyCharm reintroduces them; prefer cleaning on commit (§5).
- The file must be encoded as **UTF-8 without BOM**, end with a single
  `\n`, and parse with `json.load(...)` *and* `nbformat.validate(...)`
  (§4).

### 2.1 Cell objects

**Markdown cell:**

```jsonc
{
  "cell_type": "markdown",
  "id": "8f3c…",                 // 8-hex string, unique inside the notebook (nbformat ≥ 4.5)
  "metadata": {},
  "source": [
    "## Section heading\n",
    "First paragraph of prose…\n"
  ]
}
```

**Code cell:**

```jsonc
{
  "cell_type": "code",
  "id": "b7fc…",
  "metadata": {},
  "execution_count": null,        // or an int if you want to show "In [n]:"
  "outputs": [ /* list of output objects produced by Jupyter */ ],
  "source": [
    "import numpy as np\n",
    "x = np.arange(10)\n"
  ]
}
```

Conventions:

- `source` is **a list of strings, one per line, each ending in `\n`**
  (except optionally the last). Do not collapse it into a single string —
  it ruins git diffs.
- `outputs` is a list of `stream` / `display_data` / `execute_result` /
  `error` objects (the standard nbformat schema). Keep only outputs that
  add pedagogical value (figures, key prints). Strip noisy warnings, full
  tracebacks, and any cell whose output is just `<Figure size … at 0x…>`
  garbage.
- `metadata` should be `{}` unless you intentionally use one of the
  `mkdocs-jupyter` tags (§3.3).
- `execution_count` may be `null` (preferred for committed tutorials, so
  Colab users don't see misleading `In [37]:` numbers) or a contiguous
  count starting at 1.

### 2.2 What a healthy tutorial looks like, in numbers

For reference, the current tutorials sit in this range:

| Tutorial                              | cells | md  | code |
|---------------------------------------|-----:|----:|----:|
| `recordings_tut.ipynb`                |  16  |  9  |  7  |
| `visualization.ipynb`                 |  17  |  9  |  8  |
| `loading_legacy_data.ipynb`           |  16  |  9  |  7  |
| `local_activation_analysis.ipynb`     |  25  | 14  | 11  |
| `connectivity_analysis.ipynb`         |  17  | 10  |  7  |
| `artifact_rejection.ipynb`            |  24  | 11  | 13  |
| `train_torch_classifier.ipynb`        |  20  | 11  |  9  |

A tutorial with **1 cell of either type** is almost certainly malformed.
A tutorial > ~1.5 MB is almost certainly carrying junk (embedded base64
binary outputs, an embedded notebook, accidental dataset dump). Investigate
before committing.

---

## 3. Narrative template

All published tutorials follow the same skeleton. Reuse it for every new
one.

### 3.1 Skeleton

```
1. # <Title>                                  (markdown — H1, one line)
   Intro paragraph + "In this notebook you will learn: …" bullet list.

2. ## Imports                                  (markdown)
   Two paragraphs:
     - install instruction (Colab only)
     - clone instruction (only if the notebook needs the example data
       under docs/tutorials/data/legacy/)
   Use the Colab-only admonition:
     <div class="alert alert-block alert-danger">
     Important: execute the following cell <b>only</b> if you're using
     Google Colab!
     </div>

3. !pip install …                              (code, Colab bootstrap)
   !git clone https://github.com/medusabci/medusa-kernel.git  # if needed

4. Import the modules…                         (markdown, one-liner)

5. <Python imports>                            (code)

6. ## <Step 1>                                 (markdown)  e.g. "Load EEG recording"
7. <code>
8. ## <Step 2>                                 (markdown)
9. <code>
…
N-1. ## Conclusion / Next steps                (markdown)
     - Bullet list summarising what was learnt.
     - "See you in the next MEDUSA tutorial!" sign-off.
```

### 3.2 Style rules for prose

- **English only.** All tutorials, docstrings and docs are in English.
- Use **ATX headings** (`#`, `##`, `###`). One single H1 (the title) per
  notebook; everything else is H2 or deeper.
- Wrap markdown text at ~72–80 characters per line. Long lines hurt
  side-by-side diffs and PyCharm's notebook editor.
- Refer to symbols with backticks: `` `medusa.signal.frequency_filtering.IIRFilter` ``.
  Use the **fully-qualified import path** the user is expected to type.
- Cite parameter names with the canonical Kernel names (`signal`, `fs`,
  `n_channels`, `band`, …; see `../../AGENTS.md` §5).
- Keep paragraphs short. Prefer bullet lists for enumerations.

### 3.3 Tags supported by `mkdocs-jupyter`

The site config (`../../mkdocs.yml`) enables these cell tags:

| Tag           | Effect on the published page                  |
|---------------|-----------------------------------------------|
| `hide-input`  | Hide the code, keep the output.               |
| `hide-output` | Run/show the code, hide its output.           |
| `hide-cell`   | Drop the cell entirely from the rendered page.|

Set them under `cell.metadata.tags`:

```jsonc
"metadata": { "tags": ["hide-input"] }
```

Use sparingly. The default is to publish both code and output.

---

## 4. The validation contract

Before committing a notebook, it **must** pass all of these:

```powershell
# 1. Parse as JSON
python -c "import json; json.load(open('docs/tutorials/<file>.ipynb', encoding='utf-8'))"

# 2. Validate against the nbformat schema
python -c "import nbformat; nbformat.validate(nbformat.read('docs/tutorials/<file>.ipynb', as_version=4))"

# 3. Build the docs with the same strictness as CI
mkdocs build --strict
```

A failure in any of the three is a blocker. The MkDocs build is the
ultimate gate — it's what `tests.yml` runs.

Additional smoke checks worth running for any non-trivial edit:

- `cells` is a list with at least one markdown and one code cell.
- No cell has `source` that, when joined, *parses as a JSON object with a
  `cells` key* — that's the embedded-notebook bug from §1.
- File size is plausible (typically < 1.5 MB; see §2.2).
- Top-level keys are exactly `{"cells", "metadata", "nbformat", "nbformat_minor"}`.

A one-liner that catches the most common breakage:

```powershell
python -c "import json,sys; nb=json.load(open(sys.argv[1],encoding='utf-8')); assert isinstance(nb['cells'],list) and len(nb['cells'])>1, 'suspicious single-cell notebook'; [json.loads('{'+''.join(c['source'])) for c in nb['cells'] if c['cell_type']=='code' and ''.join(c['source']).lstrip().startswith('\"cells\"')] and sys.exit('Embedded-notebook bug detected')" docs/tutorials/<file>.ipynb
```

---

## 5. Hygiene rules

- **Strip noisy metadata** before committing. The simplest path:

  ```powershell
  jupyter nbconvert --to notebook --inplace docs/tutorials/<file>.ipynb
  ```

  This re-serialises the notebook through the canonical writer and drops
  IDE-specific cell metadata.

- **Be conservative with outputs.** Cell outputs are committed (so the
  static site has something to show without executing anything), but huge
  base64 images bloat the repo. If a plot is purely decorative, consider
  marking the cell `hide-output` or replacing it with a smaller figure.

- **Deterministic outputs.** Seed RNGs (`np.random.seed(0)`, `torch.manual_seed(0)`)
  so re-executions produce diff-friendly outputs.

- **No absolute paths.** Tutorials run on three environments (the
  contributor's box, Colab, and CI). Always resolve dataset paths relative
  to the notebook:

  ```python
  from pathlib import Path
  HERE = Path.cwd()                 # notebook dir, both in Jupyter and Colab
  rec_path = HERE / "rest_eeg" / "subject01.rec.bson"
  ```

  In Colab the bootstrap cell `git clone`s the repo and `%cd`s into
  `medusa-kernel/docs/tutorials/`, which keeps the same relative layout
  as a local checkout.

- **No OS assumptions.** Use `pathlib`, forward slashes in markdown,
  never `\`. Tutorials must work on Linux/macOS/Windows.

- **`!pip install medusa-kernel` only.** Do not pin a version unless the
  notebook genuinely needs a feature that landed in a specific release —
  in which case use `!pip install 'medusa-kernel>=X.Y'` and add a note
  near the top (`*Updated for medusa-kernel X.Y.Z*` is the convention
  used by `local_activation_analysis.ipynb`).

- **Don't import from `medusa-platform`, `medusa-analyzer` or apps.**
  Tutorials are part of `medusa-kernel`'s docs; they may only depend on
  `medusa-kernel` itself plus the standard scientific stack.

---

## 6. Data and generated artefacts

- **Prefer synthetic data.** When a tutorial only needs a signal to
  demonstrate an API, generate it with `medusa.signal.generators`
  (`EEGSignalGenerator`, `EOGSignalGenerator`, …) seeded for
  reproducibility. Such notebooks run anywhere with no download — the
  default for new tutorials.
- **Example legacy recordings** live under
  `docs/tutorials/data/legacy/{rest_eeg,cvep,mi,rcp_speller}/` (1.x
  `*.bson` files). They are **git-ignored** (`data/` in the folder
  `.gitignore`) and resolved relative to the notebook at runtime; a
  notebook that uses them should **degrade gracefully** if they are absent
  (see `loading_legacy_data.ipynb`).
- **Generated artefacts** (figures, saved recordings/models) go under
  `docs/tutorials/results/`, which is also git-ignored. Create it with
  `Path('results').mkdir(exist_ok=True)` and write there
  (`mp.save_figure(fig, 'results/…png')`, `recording.save('results/…')`).
- **Excluded from the published site** by `mkdocs.yml :: exclude_docs`
  (entries `tutorials/data/`, `tutorials/results/`, `tutorials/*.bson`,
  `tutorials/*.mdl`, `tutorials/*.py`).
- Do not commit a new dataset just to support a tutorial unless it is
  small (< ~1 MB) and licensable. Larger datasets should be synthesised, or
  downloaded by the notebook at runtime from a stable URL.

---

## 7. Registering a new tutorial

A new `.ipynb` only goes live once **all three** of these are updated:

1. **`docs/tutorials/index.md`** — add a card under the appropriate
   section (Getting started / Signal analysis / BCI paradigms). Follow
   the card-grid pattern already in the file (icon + title + 1–2 line
   description + arrow link).
2. **`mkdocs.yml`** — add the entry under `nav: Tutorials:` in the
   matching subgroup, with a human-readable label:

   ```yaml
   - Tutorials:
       - tutorials/index.md
       - Signal analysis:
           - "My new tutorial": tutorials/my_new_tutorial.ipynb
   ```

3. **The notebook itself** — passes §4 validation and follows §3 template.

If the tutorial pulls a dataset, also extend `mkdocs.yml :: exclude_docs`
so the raw data is not copied into the published site.

---

## 8. File-naming convention

- Lowercase `snake_case`, ending in `.ipynb`.
- Either descriptive (`artifact_rejection`, `connectivity_analysis`,
  `local_activation_analysis`) or paradigm + `_tut` /
  `_spellers_tut` for BCI walkthroughs (`erp_spellers_tut`,
  `cvep_spellers_tut`, `motor_imagery_analysis_tut`).
- Avoid spaces, dashes, version suffixes (`_v2`, `_new`, `_final`). When
  a tutorial is overhauled, rewrite it in place and bump the
  `*Updated for medusa-kernel X.Y.Z*` line at the top.

---

## 9. Quick checklist before opening a PR

- [ ] Notebook parses as JSON and `nbformat.validate` passes.
- [ ] `cells` is a flat list, no notebook embedded in a `source` string.
- [ ] Top-level keys = `{cells, metadata, nbformat, nbformat_minor}`.
- [ ] At least one markdown and one code cell, narrative follows §3.
- [ ] H1 title is the first cell; no other H1 in the notebook.
- [ ] `!pip install medusa-kernel` (and optional `!git clone …`) inside
      a Colab-only admonition block.
- [ ] All imports work on a clean `pip install medusa-kernel` env.
- [ ] No absolute paths; data resolved with `pathlib`.
- [ ] Cell metadata is `{}` (no IDE residue) and outputs are reviewed.
- [ ] File size is plausible (typically < 1.5 MB).
- [ ] Listed in `docs/tutorials/index.md` and in `mkdocs.yml` nav.
- [ ] `mkdocs build --strict` succeeds locally.

