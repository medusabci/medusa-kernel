import numpy as np
from matplotlib.figure import Figure
from medusa.widgets.plot_visualizer import PlotVisualizer

rng = np.random.default_rng(0)
viz = PlotVisualizer()

for k in range(6):
    fig = Figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    if k % 2:
        ax.imshow(rng.random((40, 120)), aspect="auto", cmap="viridis")
        ax.set_title(f"Heatmap {k + 1}")
    else:
        ax.plot(np.cumsum(rng.standard_normal(300)))
        ax.set_title(f"Random walk {k + 1}")
    viz.add_figure(fig)

viz.show_figs()