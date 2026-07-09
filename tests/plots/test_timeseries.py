"""Tests for the time-axis plots: stacked traces, heatmaps and shaded-line
summaries. Plain-array inputs, caller-supplied ``ax``, ``(ax, artists)`` return.
Signal traces use DATA colors (the medusa-style categorical cycle), never the
foreground/text color.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import medusa_style
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_hex

from medusa.plots import (
    TimeHeatmapPlot,
    TimeLinePlot,
    plot_shaded_line,
    plot_timeheatmap,
    plot_timeline,
)


@pytest.fixture(autouse=True)
def _isolate():
    medusa_style.use_theme("light")
    yield
    plt.close("all")
    medusa_style.use_theme("light")


@pytest.fixture
def signal():
    return np.random.default_rng(0).standard_normal((1000, 4))


# --- timeline ---------------------------------------------------------------

def test_plot_timeline_lines_per_channel(signal):
    _, ax = plt.subplots()
    _, artists = plot_timeline(signal, ax, fs=250.0)
    assert len(artists["lines"]) == signal.shape[1]


def test_default_trace_uses_a_data_color(signal):
    """The default single-signal trace is categorical_color(0), NOT plot_fg."""
    _, ax = plt.subplots()
    _, artists = plot_timeline(signal, ax, fs=250.0)
    data_color = medusa_style.categorical_color(0)
    fg = medusa_style.current_theme().plot_fg
    assert to_hex(artists["lines"][0].get_color()).lower() == data_color.lower()
    assert data_color.lower() != fg.lower()    # data != foreground/text color


def test_overlay_uses_next_categorical_color(signal):
    _, ax = plt.subplots()
    tl = TimeLinePlot(ax, fs=250.0)
    tl.set_data(signal, label="raw")
    tl.add_overlay(signal + 0.5, label="filtered")
    overlay_color = to_hex(tl._layers[1]["lines"][0].get_color()).lower()
    assert overlay_color == medusa_style.categorical_color(1).lower()


def test_events_overlay(signal):
    _, ax = plt.subplots()
    events = pd.DataFrame({"onset": [1.0, 2.0], "duration": [0.0, 0.5]})
    _, artists = plot_timeline(signal, ax, fs=250.0, events=events)
    assert artists.get("events")               # onset line + span drawn


def test_segmented_input_draws_boundaries(signal):
    seg = signal[:900].reshape(3, 300, 4)       # 3 segments
    _, ax = plt.subplots()
    _, artists = plot_timeline(seg, ax, fs=250.0)
    assert len(artists["boundaries"]) == 2      # one line per join


# --- heatmap ----------------------------------------------------------------

def test_plot_timeheatmap_image_and_cmap(signal):
    _, ax = plt.subplots()
    _, artists = plot_timeheatmap(signal, ax, fs=250.0)
    assert "image" in artists
    assert artists["image"].get_cmap().name == "medusa_sequential"


def test_timeheatmap_set_data_updates(signal):
    _, ax = plt.subplots()
    hm = TimeHeatmapPlot(ax, fs=250.0)
    updated = hm.set_data(signal)
    assert updated and hm.mappable is not None


# --- shaded line ------------------------------------------------------------

def test_plot_shaded_line_mean_and_band():
    obs = np.random.default_rng(2).standard_normal((30, 100, 3))
    _, ax = plt.subplots()
    _, artists = plot_shaded_line(obs, ax, error="ci95",
                                  series_labels=["a", "b", "c"])
    assert len(artists["mean"]) == 3 and len(artists["band"]) == 3
    # per-series data colors
    assert to_hex(artists["mean"][1].get_color()).lower() == \
        medusa_style.categorical_color(1).lower()
