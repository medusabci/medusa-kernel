# Built-in imports
import sys
# External imports
from PySide6 import QtCore
from PySide6.QtWidgets import *
# Medusa imports
from medusa.core.settings_tree import SettingsTree


class TextToTreeItem:
    def __init__(self):
        self.text_list = []
        self.titem_list = []

    def append(self, text_list, titem):
        for text in text_list:
            self.text_list.append(text)
            self.titem_list.append(titem)

    def find(self, find_str):
        find_str = find_str.lower()  # Convert search string to lowercase
        return [self.titem_list[i] for i, s in
                enumerate(self.text_list) if find_str in s.lower()]


class SettingsTreeWidget(QWidget):
    """Visualizes a settings tree using a hierarchical view.

    This widget displays a tree structure from a JSON-compatible dictionary
    or list, allowing user interaction and editing of values.

    Parameters
    ----------
    jdata : dict or list or SettingsTree
        The data to visualize. It can be a dictionary, a list, or a
        `SettingsTree` object.

    Attributes
    ----------
    find_box : QLineEdit
        The search input box.
    tree_widget : QTreeWidget
        The widget that displays the tree.
    text_to_titem : TextToTreeItem
        A helper object for searching text in the tree.
    find_str : str
        The current search string.
    found_titem_list : list
        A list of tree items that match the search.
    found_idx : int
        The index of the currently highlighted search result.
    jdata : dict or list
        The data being visualized.
    """
    def __init__(self, jdata):
        """Initializes the SettingsTreeWidget.

        Parameters
        ----------
        jdata : dict or list or SettingsTree
            The data to visualize.
        """
        super(SettingsTreeWidget, self).__init__()

        if isinstance(jdata, SettingsTree):
            jdata = jdata.to_serializable_obj()

        self.find_box = None
        self.tree_widget = None
        self.text_to_titem = TextToTreeItem()
        self.find_str = ""
        self.found_titem_list = []
        self.found_idx = 0
        self.jdata = jdata

        # Find UI
        find_layout = self.make_find_ui()

        # Tree Widget
        self.tree_widget = QTreeWidget()
        self.tree_widget.setHeaderLabels(["Key", "Value", "Info"])
        self.tree_widget.header().setSectionResizeMode(QHeaderView.Stretch)

        # Populate Tree
        self.recurse_jdata(self.jdata, self.tree_widget)

        # Layout
        layout = QHBoxLayout()
        layout.addWidget(self.tree_widget)
        gbox = QGroupBox()
        gbox.setLayout(layout)
        layout2 = QVBoxLayout()
        layout2.addLayout(find_layout)
        layout2.addWidget(gbox)
        self.setLayout(layout2)

    def make_find_ui(self):
        """Creates the search UI elements.

        Returns
        -------
        QHBoxLayout
            A layout containing the search input box and find button.
        """
        # todo: make tree widget and search bar independent. This should be
        #  removed from this class
        # Text box
        self.find_box = QLineEdit()
        self.find_box.returnPressed.connect(self.find_button_clicked)
        # Find Button
        find_button = QPushButton("Find")
        find_button.clicked.connect(self.find_button_clicked)

        layout = QHBoxLayout()
        layout.addWidget(self.find_box)
        layout.addWidget(find_button)

        return layout

    def find_button_clicked(self):
        """Handles the click event of the find button.

        Searches for the text in the `find_box` within the tree and
        highlights the matches.
        """
        find_str = self.find_box.text()
        if not find_str:
            return

        if find_str != self.find_str:
            self.find_str = find_str
            self.found_titem_list = self.text_to_titem.find(self.find_str)
            self.found_idx = 0
        else:
            item_num = len(self.found_titem_list)
            self.found_idx = (self.found_idx + 1) % item_num

        if self.found_titem_list:
            self.tree_widget.setCurrentItem(self.found_titem_list[self.found_idx])
        else:
            QMessageBox.warning(self, "Search", "No matches found.")

    def recurse_jdata(self, jdata, tree_widget):
        """Recursively populates the tree widget with data.

        Parameters
        ----------
        jdata : dict or list
            The data to add to the tree.
        tree_widget : QTreeWidget or QTreeWidgetItem
            The parent widget or item to which new rows will be added.
        """
        if isinstance(jdata, dict):
            for data in jdata.values():
                self.tree_add_row(data, tree_widget)
        elif isinstance(jdata, list):
            for data in jdata:
                self.tree_add_row(data, tree_widget)

    def tree_add_row(self, data, tree_widget, delete=False):
        """Adds a single row to the tree widget.

        This method creates and configures a `QTreeWidgetItem` based on the
        provided data, including setting up input widgets for values.

        Parameters
        ----------
        data : dict
            The data for the row.
        tree_widget : QTreeWidget or QTreeWidgetItem
            The parent widget or item.
        delete : bool, optional
            If True, a delete button is added to the row. The default is False.
        """
        text_list = []

        # Obtain the necessary fields
        key = data.get("key", "")
        value = data.get("value", None)
        info = data.get("info", None)
        input_format = data.get("input_format", None)
        value_range = data.get("value_range", None)
        value_options = data.get("value_options", None)
        items = data.get("items", None)

        # Set input format
        if input_format is None:
            if isinstance(value, bool):
                input_format = "checkbox"
            elif isinstance(value, list):
                input_format = "list"
            elif isinstance(value, int):
                input_format = "combobox" if value_options else "spinbox"
            elif isinstance(value, float):
                input_format = "combobox" if value_options else "doublespinbox"
            elif isinstance(value, str):
                input_format = "combobox" if value_options else "lineedit"
        else:
            input_format = input_format.lower()

        text_list.append(key)

        # Add the row item
        row_item = QTreeWidgetItem(tree_widget)

        if delete:
            key_widget = QWidget()
            key_layout = QHBoxLayout()
            key_layout.setContentsMargins(0, 0, 0, 0)
            key_label = QLabel(str(key))
            remove_button = QPushButton("-")
            remove_button.setFixedWidth(30)
            remove_button.clicked.connect(lambda: self.remove_list_item(row_item))
            key_layout.addWidget(key_label)
            key_layout.addWidget(remove_button)
            key_widget.setLayout(key_layout)
            self.tree_widget.setItemWidget(row_item, 0, key_widget)
        else:
            key_label = QLabel(str(key))
            self.tree_widget.setItemWidget(row_item, 0, key_label)

        if info is not None:
            info_label = QLabel(str(info))
            info_label.setStyleSheet("padding-left: 10px; border:none")
            scroll_area = QScrollArea()
            scroll_area.setWidget(info_label)
            scroll_area.setWidgetResizable(True)
            scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
            scroll_area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
            scroll_area.setAlignment(QtCore.Qt.AlignTop)
            scroll_area.setStyleSheet("""
                QScrollArea {
                    border: none;
                    padding: 0px;
                    background: transparent;
                }
                QScrollBar:horizontal {
                    height: 5px;
                    background: transparent;
                    border: none;
                }
                QScrollBar::handle:horizontal {
                    background: #a0a0a0;
                    min-width: 20px;
                    border-radius: 2px;
                }
            """)
            self.tree_widget.setItemWidget(row_item, 2, scroll_area)

        # Add widgets based on input_format for the 'Value' column
        if input_format == "combobox":
            assert value_options is not None, \
                'Options list must not be empty'
            value = str(value)
            value_options = [str(option) for option in value_options]
            combobox = QComboBox()
            combobox.addItems(value_options)
            combobox.setCurrentIndex(value_options.index(value))
            self.tree_widget.setItemWidget(row_item, 1, combobox)
        elif input_format == "checkbox":
            assert isinstance(value, bool), \
                'For the selected input format default value must be of type %s' % bool
            checkbox = QCheckBox()
            checkbox.setChecked(value)
            self.tree_widget.setItemWidget(row_item, 1, checkbox)
        elif input_format == "spinbox":
            assert isinstance(value, int), \
                'For the selected input format default value must be of type %s' % int
            spinbox = QSpinBox()
            if value_range:
                low_lim = value_range[0] if value_range[0] is not None else -1000000000
                upper_lim = value_range[1] if value_range[1] is not None else 1000000000
                spinbox.setRange(low_lim, upper_lim)
            else:
                spinbox.setRange(-1000000000, 1000000000)
            spinbox.setValue(value)
            self.tree_widget.setItemWidget(row_item, 1, spinbox)
        elif input_format == "doublespinbox":
            assert isinstance(value, float), \
                'For the selected input format default value must be of type %s' % float
            float_spinbox = QDoubleSpinBox()
            if value_range:
                low_lim = value_range[0] if value_range[0] is not None else -1000000000
                upper_lim = value_range[1] if value_range[1] is not None else 1000000000
                float_spinbox.setRange(low_lim, upper_lim)
            else:
                float_spinbox.setRange(-1000000000, 1000000000)
            float_spinbox.setValue(value)
            self.tree_widget.setItemWidget(row_item, 1, float_spinbox)
        elif input_format == "lineedit":
            line_edit = QLineEdit()
            line_edit.setText(str(value))
            self.tree_widget.setItemWidget(row_item, 1, line_edit)
        elif input_format == "list":
            for idx, list_item in enumerate(value):
                subkey = f"{key}[{idx}]"
                subdata = {
                    "key": subkey,
                    "value": list_item
                }

                if isinstance(list_item, bool):
                    subdata["input_format"] = "checkbox"
                elif isinstance(list_item, int):
                    subdata["input_format"] = "spinbox"
                elif isinstance(list_item, float):
                    subdata["input_format"] = "doublespinbox"
                elif isinstance(list_item, str):
                    subdata["input_format"] = "lineedit"
                elif isinstance(list_item, list):
                    subdata["input_format"] = "list"

                self.tree_add_row(subdata, row_item, delete=True)

            # Add "Add" button at the bottom
            button_container = QWidget()
            button_layout = QHBoxLayout()
            add_button = QPushButton("Add")
            button_layout.addWidget(add_button)
            button_layout.setContentsMargins(0, 0, 0, 0)
            button_container.setLayout(button_layout)
            button_item = QTreeWidgetItem(row_item)
            self.tree_widget.setItemWidget(button_item, 0, button_container)
            add_button.clicked.connect(lambda: self.add_button_clicked(row_item, key))

        if items:
            self.recurse_jdata(items, row_item)

        if isinstance(tree_widget, QTreeWidget):
            tree_widget.addTopLevelItem(row_item)
        else:
            tree_widget.addChild(row_item)

        self.text_to_titem.append(text_list, row_item)

    def add_button_clicked(self, row_item, parent_key):
        """Handles the click event of the 'Add' button for a list.

        Opens a dialog to choose a new item type and adds it to the list.

        Parameters
        ----------
        row_item : QTreeWidgetItem
            The parent item in the tree where the new item will be added.
        parent_key : str
            The base key for the new item.
        """
        items = ["str", "int", "float", "bool", "list"]
        item_type, ok = QInputDialog.getItem(self,
                                        "Choose item type",
                                        "Choose item type",
                                        items,
                                        0,
                                        False)
        if ok:
            type_defaults = {
                "str": ("", "lineedit"),
                "int": (0, "spinbox"),
                "float": (0.0, "doublespinbox"),
                "bool": (False, "checkbox"),
                "list": ([], "list")
            }

            value, input_format = type_defaults[item_type]
            new_key = f"{parent_key}[{row_item.childCount()-1}]"
            new_data = {
                "key": new_key,
                "value": value,
                "input_format": input_format
            }

            self.tree_add_row(new_data, row_item, delete=True)

            #Take 'Add' button to the end of the list
            row_item.takeChild(row_item.childCount()-2)
            button_container = QWidget()
            button_layout = QHBoxLayout()
            add_button = QPushButton("Add")
            button_layout.addWidget(add_button)
            button_layout.setContentsMargins(0, 0, 0, 0)
            button_container.setLayout(button_layout)
            button_item = QTreeWidgetItem(row_item)
            self.tree_widget.setItemWidget(button_item, 0, button_container)
            add_button.clicked.connect(lambda: self.add_button_clicked(row_item, parent_key))

            self.reindex_list_items(row_item, parent_key)

    def remove_list_item(self, item):
        """Removes an item from a list in the tree.

        Parameters
        ----------
        item : QTreeWidgetItem
            The item to remove.
        """
        parent_item = item.parent()
        if parent_item:
            parent_item.removeChild(item)
            self.reindex_list_items(parent_item)

    def reindex_list_items(self, parent_item, parent_key=None):
        """Re-indexes the keys of list items after an addition or removal.

        Parameters
        ----------
        parent_item : QTreeWidgetItem
            The parent item whose children need re-indexing.
        parent_key : str, optional
            The base key for the children. If None, it's inferred from the
            first child. The default is None.
        """
        child_count = parent_item.childCount()-1
        index = 0
        for i in range(child_count):
            child = parent_item.child(i)
            label_widget = self.tree_widget.itemWidget(child, 0)
            label = label_widget.findChild(QLabel)
            if parent_key is None:
                last_bracket = label.text().rfind("[")
                parent_key = label.text()[:last_bracket]
            new_key = f"{parent_key}[{index}]"
            label.setText(new_key)
            index += 1
            self.reindex_list_items(child, new_key)


class TreeViewer(QMainWindow):
    """A main window for displaying the SettingsTreeWidget.

    This class hosts the `SettingsTreeWidget` in a `QMainWindow`.

    Parameters
    ----------
    jdata : dict or list
        The JSON-compatible data to be visualized.
    """
    def __init__(self, jdata):
        """Initializes the TreeViewer.

        Parameters
        ----------
        jdata : dict or list
            The data to be visualized in the tree view.
        """
        super(TreeViewer, self).__init__()

        json_view = SettingsTreeWidget(jdata)

        self.setCentralWidget(json_view)
        self.setWindowTitle("Settings tree viewer")
        self.resize(1000, 600)
        self.show()


if __name__ == "__main__":

    settings = SettingsTree()
    settings.add_item("update_rate", value=0.2,
                      info="Update rate (s) of the plot",
                      value_range=[0, None])
    freq_filt = settings.add_item("frequency_filter")
    freq_filt.add_item("apply", value=True,
                       info="Apply IIR filter in real-time")
    freq_filt.add_item("type", value="highpass",
                       value_options=["highpass", "lowpass", "bandpass",
                                      "stopband"], info="Filter type")
    freq_filt.add_item("cutoff_freq", value=[1.0],
                       info="List with one cutoff for highpass/lowpass, "
                            "two for bandpass/stopband")
    freq_filt.add_item("order", value=5,
                       info="Order of the filter (the higher, "
                            "the greater computational cost)",
                       value_range=[1, None])


    app = QApplication(sys.argv)
    window = TreeViewer(settings.to_serializable_obj())
    window.show()
    sys.exit(app.exec())
