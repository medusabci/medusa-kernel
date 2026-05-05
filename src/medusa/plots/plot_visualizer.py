import sys
from PySide6.QtWidgets import *
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt


class PlotVisualizerWindow(QMainWindow):

    def __init__(self):

        super().__init__()

        # Set the title and geometry of the main window
        self.setWindowTitle("Plot Visualizer")

        # Create a horizontal layout for the control buttons (Next, Previous)
        change_fig_button_layout = QHBoxLayout()
        # Create the Previous button
        prev_button = QPushButton("Previous", self)
        prev_button.clicked.connect(self.show_previous_figure)
        change_fig_button_layout.addWidget(prev_button)
        # Create the Next button
        next_button = QPushButton("Next", self)
        next_button.clicked.connect(self.show_next_figure)
        change_fig_button_layout.addWidget(next_button)

        # Create a stacked layout for handling multiple figures
        self.stacked_layout = QStackedLayout()

        # Create the Export button
        export_fig_layout = QHBoxLayout()
        export_button = QPushButton("Export Figure", self)
        export_button.clicked.connect(self.export_figure)
        export_fig_layout.addWidget(export_button)

        # Create a central widget and set the layout
        widget = QWidget()
        layout = QVBoxLayout()
        layout.addLayout(change_fig_button_layout)
        layout.addLayout(self.stacked_layout)
        layout.addLayout(export_fig_layout)
        widget.setLayout(layout)
        self.setCentralWidget(widget)

    def add_figure(self, fig, fixed_size=False):
        """Add a new matplotlib figure to the stacked layout."""
        # Create canvas
        canvas = FigureCanvas(fig)
        # Set fixed size if wanted
        if fixed_size:
            fig_width, fig_height = fig.get_size_inches()
            dpi = fig.get_dpi()
            canvas.setFixedSize(int(fig_width * dpi), int(fig_height * dpi))
        # Add canvas
        self.stacked_layout.addWidget(canvas)
        self.stacked_layout.setCurrentWidget(canvas)
        canvas.draw()

    def show_next_figure(self):
        """Show the next figure in the stacked layout."""
        current_index = self.stacked_layout.currentIndex()
        next_index = (current_index + 1) % self.stacked_layout.count()
        self.stacked_layout.setCurrentIndex(next_index)

    def show_previous_figure(self):
        """Show the previous figure in the stacked layout."""
        current_index = self.stacked_layout.currentIndex()
        prev_index = ((current_index - 1 + self.stacked_layout.count()) %
                      self.stacked_layout.count())
        self.stacked_layout.setCurrentIndex(prev_index)

    def export_figure(self):
        """Export the current figure with custom settings."""
        # Get the current canvas
        current_canvas = self.stacked_layout.currentWidget()
        if not current_canvas:
            QMessageBox.warning(
                self, "No Figure", "There is no figure to export.")
            return  # No figure to export

        # Get export parameters from the user
        filt = ("PNG Files (*.png);;JPEG Files (*.jpg);;"
                "PDF Files (*.pdf);;SVG Files (*.svg)")
        file_path, selected_format = QFileDialog.getSaveFileName(
            self, "Save Figure", "", filt)
        if not file_path:
            return  # User canceled

        # Ask for figure size (in inches)
        fig_width, ok1 = QInputDialog.getDouble(
            self, "Figure Width",
            "Enter figure width (in inches):",
            6, 1, 100, 2)
        fig_height, ok2 = QInputDialog.getDouble(
            self, "Figure Height",
            "Enter figure height (in inches):",
            4, 1, 100, 2)
        if not (ok1 and ok2):
            return  # User canceled

        # Ask for DPI
        dpi, ok3 = QInputDialog.getInt(
            self, "DPI", "Enter DPI:",
            100, 50, 1000, 10)
        if not ok3:
            return  # User canceled

        # Ask for background color (only if transparency is not selected)
        transparent = self.ask_for_transparency()
        if not transparent:
            bg_color = QColorDialog.getColor()
            if not bg_color.isValid():
                return  # User canceled

        # Save the figure with the chosen parameters
        figure = current_canvas.figure
        figure.set_size_inches(fig_width, fig_height)  # Set figure size

        # Get the file format (extension from file dialog selection)
        if selected_format.endswith("png"):
            format = "png"
        elif selected_format.endswith("jpg"):
            format = "jpg"
        elif selected_format.endswith("pdf"):
            format = "pdf"
        elif selected_format.endswith("svg"):
            format = "svg"
        else:
            format = "png"  # Default to PNG if something goes wrong

        # Save with custom settings
        save_kwargs = {
            "dpi": dpi,
            "format": format,
            "transparent": transparent
        }
        # Set background color if not transparent
        if not transparent:
            save_kwargs["facecolor"] = bg_color.name()

        figure.savefig(file_path, **save_kwargs)  # Save figure

    def ask_for_transparency(self):
        """Ask the user whether they want the figure to be saved with a transparent background."""
        checkbox = QCheckBox("Save with transparent background?")
        checkbox.setChecked(False)

        # Show a dialog with the checkbox
        result = QMessageBox.question(self, "Transparency",
                                      "Do you want a transparent background?",
                                      QMessageBox.Yes | QMessageBox.No)

        if result == QMessageBox.Yes:
            return True
        else:
            return False


class PlotVisualizer:

    def __init__(self):
        self.app = QApplication(sys.argv)
        self.window = PlotVisualizerWindow()

    def add_figure(self, fig):
        self.window.add_figure(fig)

    def show_figs(self):
        self.window.show()
        self.app.exec()

    def clear(self):
        while self.window.stacked_layout.count():
            widget = self.window.stacked_layout.widget(0)
            self.window.stacked_layout.removeWidget(widget)
            widget.deleteLater()

if __name__ == '__main__':

    from matplotlib.figure import Figure
    import numpy as np

    # Create visualizer
    visualizer = PlotVisualizer()

    # # Create figure
    # with plt.style.context('seaborn-v0_8'):
    #     fig = Figure()
    #     ax = fig.add_subplot(111)
    #     ax.plot([0, 1, 2, 3], [0, 1, 4, 9])
    #     visualizer.add_figure(fig)
    #
    # # Create figure
    # fig2 = Figure()
    # ax2 = fig2.add_subplot(111)
    # ax2.plot([0, 1, 2, 3, 4], [0, 1, 4, 9, 1])
    # visualizer.add_figure(fig2)

    # Create figure
    fig3 = Figure()
    ax3 = fig3.add_subplot(111)
    spectrogram = np.random.rand(251, 35)
    times = np.linspace(0, 10, spectrogram.shape[1])
    frequencies = np.linspace(0, 5000, spectrogram.shape[0])
    img = ax3.imshow(spectrogram, aspect='auto', cmap='viridis',
                    interpolation='none',
                    extent=(times.min(), times.max(),
                            frequencies.min(),frequencies.max()))
    # Add labels
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('Frequency (Hz)')
    ax3.set_title('Spectrogram')

    # Add a colorbar to show the color scale
    fig3.colorbar(img, ax=ax3)
    visualizer.add_figure(fig3)

    # Show figures
    visualizer.show_figs()