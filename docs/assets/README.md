# `docs/assets/` — brand & art for the docs site

Drop MEDUSA brand assets here. Anything in this folder is copied verbatim
into the published site at `https://…/assets/<filename>`.

The visual identity of the docs site mirrors the official MEDUSA©
website (a static copy lives under `docs/web/` purely as a design
reference; it is excluded from the published site by `mkdocs.yml`).

## Files the site is wired to read

| File                     | Where it shows up                              | Source |
|--------------------------|-------------------------------------------------|--------|
| `logo-white.png`         | Top-left of the header (white logo on the MEDUSA-blue chrome) | Same artwork the official site uses on its dark navy nav bar. |
| `logo-01.png`            | (kept available for future use — light backgrounds, hero, README, etc.) | Official colour logo from the website. |
| `medusa_task_icon.png`   | Browser-tab favicon                             | The MEDUSA task / app mark — the square version of the brand. |
| `medusa_task_icon.ico`   | (kept for legacy / desktop integrations)        | Multi-size ICO of the same mark; not consumed by the site today. |

To swap any asset, replace the file in place — or drop a new one and
update the two paths in `mkdocs.yml :: theme.{logo,favicon}`.

## Brand colour palette (official MEDUSA© values)

The palette is derived from the website CSS
(`docs/web/MEDUSA©_files/medusa.css`) and lives in
`docs/stylesheets/extra.css` — single source of truth, easy to tweak:

| Role | Light mode | Dark mode | Source variable in `medusa.css` |
|---|---|---|---|
| Primary (header / chrome)   | `#141eb0` | `#00a3ff` | `--color-primary-medusablue` / `--color-primary` |
| Primary — light end         | `#00a3ff` | `#46c0ff` | `--color-primary` |
| Primary — dark end (footer) | `#0a1170` | `#141eb0` | derived (deepened from `--color-primary-medusablue`) |
| Accent (CTA / focus)        | `#f6412d` | `#ff6b3a` | `--color-primary-medusaorange` |
| Hero gradient               | `linear-gradient(to right bottom, #2899d8, #00a3ff)` | same (cyan stays vivid on dark) | `--gradient-one` |
| Dark surface                | n/a       | `#151521` / `#242435` | `--background-color-3` / `--background-color-4` |
| Footer band                 | `#1d293f` (deep navy) | same | `--color-primary-alta` family |
| Body font                   | `Rubik`   | `Rubik`   | Google Fonts CSS2 served by the website (`<link href="…/css2?family=Rubik:…">`) |

Edit the `:root` and `[data-md-color-scheme="slate"]` blocks in
`extra.css` to change them; the header, hero, card-grid hover, tutorial
banner and footer all pick up the change automatically.

## Adding more assets

* **Hero illustration / screenshots / diagrams:** drop them here and link
  with `![alt](../assets/<file>.png){ width="640" }` from any `.md` page.
* **Per-page art** (e.g. a tutorial-specific banner): keep it next to the
  page (`docs/tutorials/myimg.png`) so it co-locates with the content.
* **Multi-resolution favicons / Apple touch icons:** add the files
  here and either point `theme.favicon` at the right one or override
  the `<head>` via `overrides/main.html` (Material's
  [theme extension hook](https://squidfunk.github.io/mkdocs-material/customization/)).

## Building locally to preview a swap

```bash
uv run mkdocs serve      # http://127.0.0.1:8000 with live-reload
uv run mkdocs build      # one-off static build into ./site/
```
