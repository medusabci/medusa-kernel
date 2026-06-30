"""Foundation tests for ``medusa.plots``: public API surface, publication
export, layout helpers, the shared argument resolvers, and the integration
with ``medusa_style`` (the styling single source of truth).
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import medusa_style
import pytest

import medusa.plots as mp
from medusa.plots.utils import _resolve_cmap, _resolve_colors


@pytest.fixture(autouse=True)
def _isolate():
    """Isolate each test from global rcParams / active-theme mutations."""
    snapshot = matplotlib.rcParams.copy()
    medusa_style.use_theme("light")
    try:
        yield
    finally:
        matplotlib.rcParams.update(snapshot)
        plt.close("all")
        medusa_style.use_theme("light")


# --- public API surface -----------------------------------------------------

def test_public_api_surface():
    for name in mp.__all__:
        assert hasattr(mp, name), name


def test_no_style_symbols_exported():
    """Styling lives in medusa_style now, not in medusa.plots: none of the old
    kernel style symbols are public or even present (``medusa_style`` itself is
    the imported SSOT package, so it is intentionally excluded here)."""
    gone = ("BRAND", "CATEGORICAL", "use_style", "medusa_style",
            "brand_color", "categorical_colors", "colormap",
            "INK", "EVENT", "PAPER", "SEQUENTIAL", "DIVERGING")
    assert not (set(gone) & set(mp.__all__))   # none are part of the public API
    # the removed kernel-defined symbols must not even resolve as attributes
    # (``medusa_style`` is excluded: it is the imported SSOT package)
    for name in (n for n in gone if n != "medusa_style"):
        assert not hasattr(mp, name)


def test_import_does_not_touch_global_rcparams():
    """Importing medusa.plots must not mutate global matplotlib rcParams."""
    assert matplotlib.rcParams["axes.spines.top"] is True


def test_kernel_defaults_to_light_theme():
    """Importing the plotting layer selects the LIGHT medusa-style theme."""
    import subprocess
    import sys

    script = ("import medusa.plots, medusa_style\n"
              "print(medusa_style.current_theme().name)\n")
    proc = subprocess.run([sys.executable, "-"], input=script,
                          capture_output=True, text=True, check=True)
    assert proc.stdout.strip().splitlines()[-1] == "light"


# --- medusa-style integration -----------------------------------------------

def test_categorical_color_is_a_data_color():
    """Series colors come from the theme-independent categorical cycle."""
    c0 = medusa_style.categorical_color(0)
    assert c0.startswith("#")
    # theme-independent: same color in light and dark
    medusa_style.use_theme("dark")
    assert medusa_style.categorical_color(0) == c0
    medusa_style.use_theme("light")


def test_sequential_and_diverging_cmaps_register():
    seq = medusa_style.mpl.sequential_cmap()
    div = medusa_style.mpl.diverging_cmap()
    assert seq.name == "medusa_sequential"
    assert div.name == "medusa_diverging"


# --- publication export ------------------------------------------------------

def test_save_figure_opaque_vs_transparent(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [1, 0])
    opaque = mp.save_figure(fig, tmp_path / "opaque.png")
    transp = mp.save_figure(fig, tmp_path / "transp.png", transparent=True)
    op = plt.imread(opaque)
    tr = plt.imread(transp)
    assert op[0, 0, 3] > 0.5      # opaque corner
    assert tr[0, 0, 3] < 0.5      # transparent corner


def test_save_figure_creates_parent_dirs(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    out = mp.save_figure(fig, tmp_path / "deep" / "nested" / "fig.png")
    assert out.exists() and out.stat().st_size > 0


# --- layout + argument resolvers --------------------------------------------

def test_optimal_grid():
    assert mp.optimal_grid(6) == (3, 2)
    rows, cols = mp.optimal_grid(7)        # prime -> one empty cell
    assert rows * cols >= 7
    with pytest.raises(ValueError):
        mp.optimal_grid(0)


def test_resolve_colors():
    assert _resolve_colors(None, 3) is None        # caller applies default
    assert _resolve_colors("red", 3) == ["red"] * 3
    assert _resolve_colors(["a", "b"], 2) == ["a", "b"]
    with pytest.raises(ValueError):
        _resolve_colors(["a", "b"], 3)


def test_resolve_cmap():
    seq = medusa_style.mpl.sequential_cmap()
    assert _resolve_cmap(None, seq) is seq                 # None -> default
    assert _resolve_cmap("viridis", seq).name == "viridis"  # str -> lookup
    assert _resolve_cmap(seq, None) is seq                 # object -> passthrough
