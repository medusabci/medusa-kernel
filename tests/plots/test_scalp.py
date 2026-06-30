"""Tests for the scalp plots: head substrate, topography, connectivity.

Inputs are plain arrays + a :class:`medusa.core.data.ChannelSet`; each plot
takes a caller-supplied ``ax`` and returns ``(ax, artists)``. Chrome colors
default to the active ``medusa_style`` theme; colormaps default to the SSOT
sequential / diverging maps.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import medusa_style
import numpy as np
import pytest
from matplotlib.colors import to_hex

from medusa.core.data import ChannelSet
from medusa.plots import (
    ConnectivityPlot,
    TopographicPlot,
    plot_connectivity,
    plot_scalp,
    plot_topography,
)

_LABELS = ["Fz", "Cz", "Pz", "Oz", "C3", "C4", "P3", "P4"]


@pytest.fixture(autouse=True)
def _isolate():
    medusa_style.use_theme("light")
    yield
    plt.close("all")
    medusa_style.use_theme("light")


@pytest.fixture
def channel_set():
    return ChannelSet().add_unipolar_eeg_channels(_LABELS)


@pytest.fixture
def values():
    return np.random.default_rng(0).standard_normal(len(_LABELS))


# --- head substrate ---------------------------------------------------------

def test_plot_scalp_returns_named_artists(channel_set):
    _, ax = plt.subplots()
    out_ax, artists = plot_scalp(channel_set, ax, show_labels=True)
    assert out_ax is ax
    assert "head-line" in artists and "sensors" in artists
    assert "labels" in artists and len(artists["labels"]) == len(_LABELS)


def test_plot_scalp_chrome_defaults_to_theme_foreground(channel_set):
    """Head outline color defaults to the active theme's plot_fg."""
    _, ax = plt.subplots()
    _, artists = plot_scalp(channel_set, ax)
    fg = medusa_style.current_theme().palette.plot_fg
    assert to_hex(artists["head-line"].get_color()).lower() == fg.lower()


# --- topography -------------------------------------------------------------

def test_plot_topography_interpolated(channel_set, values):
    _, ax = plt.subplots()
    _, artists = plot_topography(values, channel_set, ax, colorbar=True)
    assert "image" in artists                  # interpolated surface
    assert artists["image"].get_cmap().name == "medusa_sequential"


def test_plot_topography_discrete(channel_set, values):
    _, ax = plt.subplots()
    _, artists = plot_topography(values, channel_set, ax, interpolate=False)
    assert "discs" in artists and len(artists["discs"]) == len(_LABELS)


def test_topographic_plot_set_data_updates_in_place(channel_set, values):
    _, ax = plt.subplots()
    tp = TopographicPlot(channel_set, ax)
    updated = tp.set_data(values)
    assert updated and tp.mappable is not None
    img = tp.artists.get("image")
    tp.set_data(values * 2)                     # in-place update reuses the image
    assert tp.artists.get("image") is img


# --- connectivity -----------------------------------------------------------

def test_plot_connectivity_edges(channel_set):
    adj = np.corrcoef(np.random.default_rng(1).standard_normal(
        (len(_LABELS), 200)))
    _, ax = plt.subplots()
    _, artists = plot_connectivity(adj, channel_set, ax, threshold=50,
                                   colorbar=True)
    assert "edges" in artists


def test_connectivity_default_cmap_is_diverging(channel_set):
    _, ax = plt.subplots()
    cp = ConnectivityPlot(channel_set, ax)
    assert cp._cmap.name == "medusa_diverging"
