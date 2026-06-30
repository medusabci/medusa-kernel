"""MkDocs hooks for the medusa-kernel docs site.

Implements:

* ``on_page_content``: prepends an "Open in Colab" + "View on GitHub" +
  "Download .ipynb" button row to every Jupyter-notebook page (any source
  file ending in ``.ipynb`` rendered by ``mkdocs-jupyter``). The plugin
  bypasses the Markdown rendering stage and emits HTML directly via
  nbconvert, so we cannot rely on Material icon shortcodes here — the
  icons are inlined as SVG paths from the open-source Material Design
  Icons set.

The repository slug and branch are read from ``extra.colab`` in
``mkdocs.yml`` so they stay in sync with the ``mike`` deploy aliases.

The hook is registered via the ``hooks:`` key in ``mkdocs.yml``; no
plugin boilerplate is needed.
"""

from __future__ import annotations

from typing import Any

# Inline SVG icons (Material Design Icons — Apache 2.0 / MIT compatible).
# Sized to inherit the surrounding text colour via ``currentColor``.
_ICON_GITHUB = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
    'width="16" height="16" fill="currentColor" '
    'style="vertical-align:-3px;margin-right:.35rem;">'
    '<path d="M12 .3a12 12 0 0 0-3.8 23.4c.6.1.8-.3.8-.6v-2.2c-3.3.7-4-1.6-4-1.6-.6-1.4-1.4-1.8-1.4-1.8-1.1-.7.1-.7.1-.7 1.2.1 1.9 1.2 1.9 1.2 1.1 1.9 2.9 1.4 3.6 1 .1-.8.4-1.4.8-1.7-2.7-.3-5.5-1.3-5.5-6 0-1.3.5-2.4 1.2-3.2-.1-.3-.5-1.5.1-3.2 0 0 1-.3 3.3 1.2a11.5 11.5 0 0 1 6 0c2.3-1.5 3.3-1.2 3.3-1.2.6 1.7.2 2.9.1 3.2.8.8 1.2 1.9 1.2 3.2 0 4.6-2.8 5.6-5.5 5.9.5.4.9 1.2.9 2.3v3.4c0 .3.2.7.8.6A12 12 0 0 0 12 .3"/>'
    '</svg>'
)
_ICON_DOWNLOAD = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
    'width="16" height="16" fill="currentColor" '
    'style="vertical-align:-3px;margin-right:.35rem;">'
    '<path d="M5 20h14v-2H5m14-9h-4V3H9v6H5l7 7 7-7Z"/>'
    '</svg>'
)


def _colab_badge(repo: str, branch: str, src_path: str) -> str:
    """Return an HTML banner with Colab + GitHub + Download buttons."""
    nb_path = src_path.replace("\\", "/")
    colab_url = (
        f"https://colab.research.google.com/github/{repo}/blob/{branch}/"
        f"{nb_path}"
    )
    github_url = f"https://github.com/{repo}/blob/{branch}/{nb_path}"
    raw_url = f"https://raw.githubusercontent.com/{repo}/{branch}/{nb_path}"

    btn = (
        "padding:.3rem .7rem;font-size:.75rem;text-decoration:none;"
        "display:inline-flex;align-items:center;"
    )
    return (
        '<div class="md-tutorial-banner" '
        'style="display:flex;flex-wrap:wrap;gap:.6rem;align-items:center;'
        'margin:0 0 1.25rem 0;padding:.7rem .9rem;border-radius:.4rem;'
        'background:var(--md-code-bg-color);'
        'border:1px solid var(--md-default-fg-color--lightest);">'
        '<span style="font-weight:600;margin-right:.25rem;">'
        'Run this tutorial:</span>'
        f'<a href="{colab_url}" target="_blank" rel="noopener" '
        f'class="md-button md-button--primary" style="{btn}">'
        '<img src="https://colab.research.google.com/assets/colab-badge.svg" '
        'alt="Open in Colab" style="height:1.25rem;display:block;"></a>'
        f'<a href="{github_url}" target="_blank" rel="noopener" '
        f'class="md-button" style="{btn}">'
        f'{_ICON_GITHUB}View on GitHub</a>'
        f'<a href="{raw_url}" target="_blank" rel="noopener" download '
        f'class="md-button" style="{btn}">'
        f'{_ICON_DOWNLOAD}Download .ipynb</a>'
        '</div>\n'
    )


def on_page_content(html: str, page: Any, config: Any, files: Any) -> str:  # noqa: ARG001
    """Inject the Colab/GitHub banner at the top of every notebook page."""
    src_path = page.file.src_path
    if not src_path.endswith(".ipynb"):
        return html

    cfg = (config.get("extra") or {}).get("colab") or {}
    repo = cfg.get("repo", "medusabci/medusa-kernel")
    branch = cfg.get("branch", "main")

    return _colab_badge(repo, branch, src_path) + html

