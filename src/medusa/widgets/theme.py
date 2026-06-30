"""Shared Qt theme for :mod:`medusa.widgets` — the Qt side of the MEDUSA identity.

Every color here is derived from :mod:`medusa.plots.style` (the single source of
truth); :mod:`medusa.plots` stays Qt-free, and the widgets import the brand from
this thin Qt-facing layer instead of hand-spelling ``rgba(0, 163, 255, ...)`` or
copy-pasting QSS. It provides:

- **color tokens** (QSS-ready strings) for the brand accent (selection / hover),
  theme-neutral hover/border tints, muted secondary text and the error color;
- **geometry tokens** (radii, control padding, icon size) shared by the widgets;
- **QSS builders** (:func:`toolbar_qss`, :func:`panel_qss`, :func:`filmstrip_qss`,
  :func:`muted_label_qss`, :func:`error_field_qss`) — one home for chrome that was
  previously duplicated across widgets;
- **typography**: the bundled Rubik faces registered with Qt
  (:func:`register_qt_fonts`) and applied window-scoped (:func:`style_window`), so
  widgets match the figures without touching a shared ``QApplication``.

Importing this module has **no side effects**; fonts register lazily on first use.
"""

from functools import lru_cache
from importlib.resources import as_file, files
from pathlib import Path

from PySide6.QtGui import QColor, QFontDatabase

from medusa.plots.style import ACCENT, ERROR, MUTED, rgba_css

__all__ = [
    "ACCENT",
    "ERROR",
    "MUTED_TEXT",
    "SELECTION_TINT",
    "SELECTION_TINT_STRONG",
    "ACCENT_HOVER",
    "HOVER_NEUTRAL",
    "HOVER_NEUTRAL_SOFT",
    "BORDER_NEUTRAL",
    "BORDER_NEUTRAL_SOFT",
    "RADIUS_CARD",
    "RADIUS_CONTROL",
    "RADIUS_PANEL",
    "ICON_PX",
    "TITLE_WEIGHT",
    "FONT_FAMILY",
    "selection_color",
    "register_qt_fonts",
    "style_window",
    "toolbar_qss",
    "panel_qss",
    "filmstrip_qss",
    "muted_label_qss",
    "error_field_qss",
]

# --------------------------------------------------------------------------- #
# Color tokens (QSS-ready)
# --------------------------------------------------------------------------- #
#: Secondary / caption text (status lines, hints). Brand-neutral slate grey.
MUTED_TEXT: str = MUTED
#: Brand-accent tints for selection / interactive hover (translucent so they
#: *tint* the OS theme rather than paint over it).
SELECTION_TINT: str = rgba_css(ACCENT, 0.16)
SELECTION_TINT_STRONG: str = rgba_css(ACCENT, 0.20)
ACCENT_HOVER: str = rgba_css(ACCENT, 0.16)
#: Theme-neutral tints (mid-grey alpha): follow the OS light/dark theme; used for
#: hovers/borders that should read as chrome, not brand.
HOVER_NEUTRAL: str = "rgba(127, 127, 127, 0.12)"
HOVER_NEUTRAL_SOFT: str = "rgba(127, 127, 127, 0.10)"
BORDER_NEUTRAL: str = "rgba(127, 127, 127, 0.30)"
BORDER_NEUTRAL_SOFT: str = "rgba(127, 127, 127, 0.25)"

# --------------------------------------------------------------------------- #
# Geometry tokens (px) shared across widgets
# --------------------------------------------------------------------------- #
RADIUS_CARD: int = 8        # filmstrip cards
RADIUS_CONTROL: int = 5     # tool buttons
RADIUS_PANEL: int = 6       # group boxes / panel buttons
ICON_PX: int = 18           # toolbar / control icon size
TITLE_WEIGHT: int = 600     # semibold (matches the figure title weight)

#: The brand typeface (bundled with the plots package; registered on first use).
FONT_FAMILY: str = "Rubik"


# --------------------------------------------------------------------------- #
# Colors as QColor
# --------------------------------------------------------------------------- #
def selection_color(lighten: float = 0.55) -> QColor:
    """Brand-accent selection color (the cyan, lightened toward white).

    Used for the tree/list selection highlight so a selected row reads as MEDUSA
    brand rather than a generic OS highlight. ``lighten`` blends toward white
    (0 = pure accent, 1 = white) to keep selected text legible.
    """
    c = QColor(ACCENT)
    return QColor(round(c.red() + (255 - c.red()) * lighten),
                  round(c.green() + (255 - c.green()) * lighten),
                  round(c.blue() + (255 - c.blue()) * lighten))


# --------------------------------------------------------------------------- #
# Typography
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def register_qt_fonts() -> None:
    """Register the bundled Rubik weights with Qt (idempotent, lazy).

    Loads the same TTFs shipped for matplotlib (under ``medusa.plots`` package
    data) into Qt's font database, so widget chrome can use ``Rubik`` on any
    machine without a system install.
    """
    fonts_dir = files("medusa.plots").joinpath("styles/fonts")
    with as_file(fonts_dir) as path:
        for ttf in sorted(Path(path).glob("*.ttf")):
            QFontDatabase.addApplicationFont(str(ttf))


def style_window(widget) -> None:
    """Apply the MEDUSA typeface to a top-level ``widget`` (window-scoped).

    Sets Rubik as the window's font (children inherit it) while keeping the
    current point size, so the whole window matches the figures — without
    mutating a shared ``QApplication`` palette/font that a host app may own.
    """
    register_qt_fonts()
    font = widget.font()
    font.setFamily(FONT_FAMILY)
    widget.setFont(font)


# --------------------------------------------------------------------------- #
# QSS builders (single home for chrome previously duplicated across widgets)
# --------------------------------------------------------------------------- #
def toolbar_qss() -> str:
    """QSS for a flat toolbar with rounded, neutral-hover tool buttons."""
    return (
        f"QToolBar {{ spacing: 6px; padding: 3px; border: none; }}"
        f"QToolButton {{ padding: 4px 9px; border-radius: {RADIUS_CONTROL}px; }}"
        f"QToolButton:hover {{ background: {HOVER_NEUTRAL}; }}"
    )


def panel_qss() -> str:
    """QSS for a control panel: rounded group boxes + accent-hover buttons."""
    return f"""
QGroupBox {{ font-weight: {TITLE_WEIGHT}; border: 1px solid {BORDER_NEUTRAL_SOFT};
            border-radius: {RADIUS_PANEL}px; margin-top: 8px;
            padding: 6px 4px 4px 4px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 3px; }}
QPushButton {{ padding: 3px 8px; border-radius: {RADIUS_CONTROL}px;
              border: 1px solid {BORDER_NEUTRAL}; }}
QPushButton:hover {{ background: {ACCENT_HOVER}; }}
"""


def filmstrip_qss() -> str:
    """QSS for a borderless thumbnail filmstrip with a brand selection tint.

    Selection is toggled via the ``selected`` dynamic property (no stylesheet
    swap), so nothing shifts between states.
    """
    return f"""
QListWidget {{ border: none; outline: 0; background: palette(base); }}
QListWidget::item {{ border: none; }}
QListWidget::item:selected {{ background: transparent; }}
QWidget#card {{ border-radius: {RADIUS_CARD}px; background: transparent; }}
QWidget#card:hover {{ background: {HOVER_NEUTRAL_SOFT}; }}
QWidget#card[selected="true"] {{ background: {SELECTION_TINT}; }}
QWidget#card[selected="true"]:hover {{ background: {SELECTION_TINT_STRONG}; }}
QLabel#cardtitle {{ font-weight: {TITLE_WEIGHT}; }}
QLabel#cardthumb {{ border: 1px solid {BORDER_NEUTRAL}; }}
"""


def muted_label_qss(extra: str = "") -> str:
    """QSS for muted secondary text; append ``extra`` declarations if needed.

    Examples
    --------
    >>> muted_label_qss("padding-right: 10px;")  # doctest: +SKIP
    'color: #6c7480; padding-right: 10px;'
    """
    return f"color: {MUTED_TEXT}; {extra}".strip()


def error_field_qss() -> str:
    """QSS that tints a ``QLineEdit``'s text with the brand error color."""
    return f"QLineEdit {{ color: {ERROR}; }}"
