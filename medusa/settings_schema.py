# Built-in imports
import sys
# External imports
from PySide6 import QtCore
from PySide6.QtWidgets import *
# Medusa imports
from .components import SerializableComponent


class SettingsTree(SerializableComponent):
    """
    SettingsTree is a utility class for building and managing hierarchical tree
    structures in a JSON-compatible format.
    """
    def __init__(self, tree=None):
        # Initialize the tree structure.
        # If a tree is provided, use it; otherwise, start with an empty list.
        self.tree = tree if tree is not None else []

    def add_item(self, key, default_value=None, info=None, input_format=None,
                 value_range=None, value_options=None):
        """
        Adds a new item (or sub-item) to the current tree structure.

        Parameters
        ----------
        key : str
            The key name of the item.
        default_value : {str, int, float, bool, list}, optional
            Default value for this item. Can be a primitive type or a list.
        info : str, optional
            Help text or description to be displayed.
        input_format : {'checkbox', 'spinbox', 'doublespinbox', 'lineedit', 'combobox'}, optional
            Type of UI control associated with this item.
        value_range : list, optional
            List indicating the [min, max] for numeric inputs. Use `None` to indicate no bound on that side.
        value_options : list, optional
            A list of allowed options (used for combobox inputs).

        Returns
        -------
        SettingsTree
            A `SettingsTree` instance wrapping the added item.
        """

        # Ensure the key is a string, convert if necessary
        if key is not None:
            if not isinstance(key, str):
                try:
                    key = str(key)
                except Exception as e:
                    print(f"Error: 'key' must be a string. Cannot convert "
                          f"{key} to string.")
                    return None

        # Validate all provided values
        if default_value is not None and not self.validate_default_value(default_value):
            return None
        info = self.validate_info(info)
        input_format = self.validate_input_format(input_format)
        value_range = self.validate_value_range(value_range)
        value_options = self.validate_value_options(value_options)

        # Build the item dictionary
        item = {
            'key': key,
            'default_value': default_value,
            'info': info,
            'input_format': input_format,
            'value_range': value_range,
            'value_options': value_options,
        }

        # Remove any keys with None values to keep the dictionary clean
        item = {k: v for k, v in item.items() if v is not None}

        # Add the new item to the tree depending on its type
        if isinstance(self.tree, dict):
            # If this is a branch, append to its 'items' list (create it if necessary)
            self.tree.setdefault('items', []).append(item)
        elif isinstance(self.tree, list):
            # If this is a top-level list, simply append the item
            self.tree.append(item)

        return SettingsTree(item)

    def get_item(self, *keys):
        """
        Recursively retrieves a nested item from the tree using a sequence of keys.

        Parameters:
            *keys: Sequence of keys to navigate through nested items.

        Returns:
            SettingsTree: A SettingsTree instance wrapping the matched item (or sub-item).
        """
        current_node = self.tree
        for key in keys:
            found = None
            items = current_node.get('items', []) if isinstance(current_node, dict) else current_node
            for item in items:
                if item.get('key') == key:
                    found = item
                    break
            if found is None:
                raise KeyError(f"Key '{key}' not found in the tree.")
            current_node = found
        return SettingsTree(current_node)

    def edit_item(self, default_value=None, info=None, input_format=None, value_range=None, value_options=None):
        """
        Edits a SettingsTree instance.

        Parameters:
            default_value (str, int, float, bool or list, optional): Updated default value for this item.
            info (str, optional): Updated help text or description to be displayed.
            input_format (str, optional): Updated UI control type ('checkbox', 'spinbox', 'doublespinbox', 'lineedit', 'combobox').
            value_range (list, optional): Updated list indicating the [min, max] for numeric inputs.
            value_options (list, optional): Updated list of allowed options (used for combobox).

        Returns:
            SettingsTree: The current SettingsTree instance after editing
        """
        tree = self.tree
        if not isinstance(tree, dict):
            raise TypeError("SettingsTree must wrap a dictionary to be editable.")

        if default_value is not None: tree['default_value'] = default_value if self.validate_default_value(
            default_value) else tree.get('default_value')
        if info is not None: tree['info'] = self.validate_info(info)
        if input_format is not None: tree['input_format'] = self.validate_input_format(input_format)
        if value_range is not None: tree['value_range'] = self.validate_value_range(value_range)
        if value_options is not None: tree['value_options'] = self.validate_value_options(value_options)
        return self

    def update_tree_from_widget(self, tree_widget: QTreeWidget):
        """
        Updates the SettingsTree dictionary with values from a QTreeWidget object.
        """

        def extract_value_from_widget(widget):
            if isinstance(widget, QComboBox):
                return widget.currentText()
            elif isinstance(widget, QCheckBox):
                return widget.isChecked()
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                return widget.value()
            elif isinstance(widget, QLineEdit):
                return widget.text()
            return None

        def extract_value_from_list(parent_item):
            new_list = []
            for i in range(parent_item.childCount()-1):
                child = parent_item.child(i)
                value_widget = tree_widget.itemWidget(child, 1)
                if value_widget is not None:
                    value = extract_value_from_widget(value_widget)
                    new_list.append(value)
                else:
                    sublist = extract_value_from_list(child)
                    new_list.append(sublist)
            return new_list

        def traverse_tree_item(item, node):
            # Retrieve the widget associated with the "Value" column for the current item
            widget = tree_widget.itemWidget(item, 1)

            # Extract values from the widget
            if widget is not None:
                value = extract_value_from_widget(widget)
                SettingsTree(node).edit_item(default_value=value)
            elif isinstance(node.get("default_value"), list):
                value = extract_value_from_list(item)
                SettingsTree(node).edit_item(default_value=value)

            # Recursively process child items
            if "items" in node and item.childCount() > 0:
                for j in range(item.childCount()):
                    child_item = item.child(j)
                    child_node = node["items"][j]
                    traverse_tree_item(child_item, child_node)

        # Traverse all top-level items
        if isinstance(self.tree, list):
            for i in range(tree_widget.topLevelItemCount()):
                item = tree_widget.topLevelItem(i)
                node = self.tree[i]
                traverse_tree_item(item, node)
        elif isinstance(self.tree, dict) and "items" in self.tree:
            for i in range(tree_widget.topLevelItemCount()):
                item = tree_widget.topLevelItem(i)
                node = self.tree["items"][i]
                traverse_tree_item(item, node)
        return self

    def validate_default_value(self, default_value):
        # Validate that the default_value is one of the allowed types
        valid_types = (str, int, float, bool, list)
        if not isinstance(default_value, valid_types):
            print(f"Error: 'default_value' must be of type: string, int, float, bool, list.")
            return False
        return True

    def validate_info(self, info):
        # Validate that 'info' is a string, otherwise keep it as None
        if info is not None and not isinstance(info, str):
            print("Warning: 'info' must be a string describing the element. Keeping it as None.")
            return None
        return info

    def validate_input_format(self, input_format, default_value=None, value_options=None):
        # Validate that 'input_format' is one of the allowed types and meets specific requirements
        valid_formats = ['checkbox', 'spinbox', 'doublespinbox', 'lineedit', 'combobox']

        input_format = input_format.lower() if input_format is not None else None

        if input_format is not None and (input_format not in valid_formats or not isinstance(input_format, str)):
            print("Warning: 'input_format' must be one of the following options: 'CheckBox', 'SpinBox', 'DoubleSpinBox', 'LineEdit', 'ComboBox'. Keeping it as None.")
            return None

        if input_format == "combobox" and value_options is None:
            print("Warning: 'ComboBox' requires 'value_options' to be specified. Keeping input format as None.")
            return None
        if input_format == "checkbox" and not isinstance(default_value, bool):
            print("Warning: 'CheckBox' requires 'default_value' to be a boolean (True/False). Keeping input format as None.")
            return None
        if input_format == "spinbox" and not isinstance(default_value, int):
            print("Warning: 'SpinBox' requires 'default_value' to be an integer. Keeping input format as None.")
            return None
        if input_format == "doublespinbox" and not isinstance(default_value, float):
            print("Warning: 'DoubleSpinBox' requires 'default_value' to be a float. Keeping input format as None.")
            return None

        return input_format

    def validate_value_range(self, value_range):
        # Validate that 'value_range' is a list or array with exactly two elements
        if value_range is not None:
            if not (isinstance(value_range, list)):
                print(
                    "Warning: 'value_range' must be a list with exactly two elements determining the upper and lower limits the value can acquire. Keeping range without bounds.")
                return None
            if len(value_range) != 2:
                print("Warning: 'value_range' must have exactly two elements: upper and lower limits the value can acquire. Keeping range without bounds.")
                return None
        return value_range

    def validate_value_options(self, value_options):
        # Validate that 'value_options' is a list
        if value_options is not None and not isinstance(value_options, list):
            print("Warning: 'value_options' must be a list containing the available options. The list will be kept empty.")
            return None
        return value_options

    def to_serializable_obj(self):
        return self.tree

    @classmethod
    def from_serializable_obj(cls, data):
        return cls(data)


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
    """
    A QWidget-based class that visualizes a JSON-compatible dictionary or list using a hierarchical tree view.

    Parameters:
        jdata (dict or list): A JSON-compatible dictionary or list containing for each item: key, default value
        (optional), input format (optional), value range (optional), value options (optional) and sub-items (optional).
    """
    def __init__(self, jdata):
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
        if isinstance(jdata, dict):
            for data in jdata.values():
                self.tree_add_row(data, tree_widget)
        elif isinstance(jdata, list):
            for data in jdata:
                self.tree_add_row(data, tree_widget)

    def tree_add_row(self, data, tree_widget, delete=False):
        text_list = []

        # Obtain the necessary fields
        key = data.get("key", "")
        default_value = data.get("default_value", None)
        info = data.get("info", None)
        input_format = data.get("input_format", None)
        value_range = data.get("value_range", None)
        value_options = data.get("value_options", None)
        items = data.get("items", None)

        # Set input format
        if input_format is None:
            if isinstance(default_value, bool):
                input_format = "checkbox"
            elif isinstance(default_value, list):
                input_format = "list"
            elif isinstance(default_value, int):
                input_format = "combobox" if value_options else "spinbox"
            elif isinstance(default_value, float):
                input_format = "combobox" if value_options else "doublespinbox"
            elif isinstance(default_value, str):
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
            default_value = str(default_value)
            value_options = [str(option) for option in value_options]
            combobox = QComboBox()
            combobox.addItems(value_options)
            combobox.setCurrentIndex(value_options.index(default_value))
            self.tree_widget.setItemWidget(row_item, 1, combobox)
        elif input_format == "checkbox":
            assert isinstance(default_value, bool), \
                'For the selected input format default value must be of type %s' % bool
            checkbox = QCheckBox()
            checkbox.setChecked(default_value)
            self.tree_widget.setItemWidget(row_item, 1, checkbox)
        elif input_format == "spinbox":
            assert isinstance(default_value, int), \
                'For the selected input format default value must be of type %s' % int
            spinbox = QSpinBox()
            if value_range:
                low_lim = value_range[0] if value_range[0] is not None else -1000000000
                upper_lim = value_range[1] if value_range[1] is not None else 1000000000
                spinbox.setRange(low_lim, upper_lim)
            else:
                spinbox.setRange(-1000000000, 1000000000)
            spinbox.setValue(default_value)
            self.tree_widget.setItemWidget(row_item, 1, spinbox)
        elif input_format == "doublespinbox":
            assert isinstance(default_value, float), \
                'For the selected input format default value must be of type %s' % float
            float_spinbox = QDoubleSpinBox()
            if value_range:
                low_lim = value_range[0] if value_range[0] is not None else -1000000000
                upper_lim = value_range[1] if value_range[1] is not None else 1000000000
                float_spinbox.setRange(low_lim, upper_lim)
            else:
                float_spinbox.setRange(-1000000000, 1000000000)
            float_spinbox.setValue(default_value)
            self.tree_widget.setItemWidget(row_item, 1, float_spinbox)
        elif input_format == "lineedit":
            line_edit = QLineEdit()
            line_edit.setText(str(default_value))
            self.tree_widget.setItemWidget(row_item, 1, line_edit)
        elif input_format == "list":
            for idx, list_item in enumerate(default_value):
                subkey = f"{key}[{idx}]"
                subdata = {
                    "key": subkey,
                    "default_value": list_item
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

            default_value, input_format = type_defaults[item_type]
            new_key = f"{parent_key}[{row_item.childCount()-1}]"
            new_data = {
                "key": new_key,
                "default_value": default_value,
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
        parent_item = item.parent()
        if parent_item:
            parent_item.removeChild(item)
            self.reindex_list_items(parent_item)

    def reindex_list_items(self, parent_item, parent_key=None):
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
    """
        A main window class that hosts the SettingsTreeWidget widget.

        Parameters:
            jdata (dict or list): The JSON-compatible data to be visualized in the tree view.
        """
    def __init__(self, jdata):
        super(TreeViewer, self).__init__()

        json_view = SettingsTreeWidget(jdata)

        self.setCentralWidget(json_view)
        self.setWindowTitle("Settings tree viewer")
        self.resize(1000, 600)
        self.show()


if __name__ == "__main__":

    settings = SettingsTree()
    settings.add_item("update_rate", default_value=0.2,
                      info="Update rate (s) of the plot",
                      value_range=[0, None])
    freq_filt = settings.add_item("frequency_filter")
    freq_filt.add_item("apply", default_value=True,
                       info="Apply IIR filter in real-time")
    freq_filt.add_item("type", default_value="highpass",
                       value_options=["highpass", "lowpass", "bandpass",
                                      "stopband"], info="Filter type")
    freq_filt.add_item("cutoff_freq", default_value=[1.0],
                       info="List with one cutoff for highpass/lowpass, "
                            "two for bandpass/stopband")
    freq_filt.add_item("order", default_value=5,
                       info="Order of the filter (the higher, "
                            "the greater computational cost)",
                       value_range=[1, None])


    app = QApplication(sys.argv)
    window = TreeViewer(settings.to_serializable_obj())
    window.show()
    sys.exit(app.exec())


