# Built-in imports
import sys
# External imports
from PySide6 import QtCore
from PySide6.QtWidgets import *
# Medusa imports
from .components import SerializableComponent


class SettingsTree(SerializableComponent):
    """Manages hierarchical tree structures for settings.

    This class provides utilities to build and manage settings trees
    in a JSON-compatible format.

    Parameters
    ----------
    tree : list or dict, optional
        The initial tree structure. If not provided, an empty list is used.
        The default is None, which creates an empty list.

    Attributes
    ----------
    tree : list or dict
        The internal representation of the tree structure.

    Methods
    -------
    add_item(key, default_value=None, info=None, input_format=None, value_range=None, value_options=None)
        Adds a new item or sub-item to the tree.
    get_item(*keys)
        Recursively retrieves a nested item using a sequence of keys.
    edit_item(default_value=None, info=None, input_format=None, value_range=None, value_options=None)
        Edits the current item in the tree.
    update_tree_from_widget(tree_widget)
        Updates the tree with values from a QTreeWidget.
    validate_default_value(default_value)
        Validates the type of the default value.
    validate_info(info)
        Validates the info field.
    validate_input_format(input_format, default_value=None, value_options=None)
        Validates the input format.
    validate_value_range(value_range)
        Validates the value range.
    validate_value_options(value_options)
        Validates the value options.
    to_serializable_obj()
        Returns the tree as a serializable object.
    from_serializable_obj(data)
        Creates a SettingsTree from a serializable object.

    Examples
    --------
    >>> settings = SettingsTree()
    >>> settings.add_item("update_rate", default_value=0.2,
    ...                   info="Update rate (s) of the plot",
    ...                   value_range=[0, None])
    >>> freq_filt = settings.add_item("frequency_filter")
    >>> freq_filt.add_item("apply", default_value=True,
    ...                    info="Apply IIR filter in real-time")
    >>> freq_filt.add_item("type", default_value="highpass",
    ...                    value_options=["highpass", "lowpass", "bandpass",
    ...                                   "stopband"], info="Filter type")
    >>> freq_filt.add_item("cutoff_freq", default_value=[1.0],
    ...                    info="List with one cutoff for highpass/lowpass, "
    ...                         "two for bandpass/stopband")
    >>> freq_filt.add_item("order", default_value=5,
    ...                    info="Order of the filter (the higher, "
    ...                         "the greater computational cost)",
    ...                    value_range=[1, None])
    """
    def __init__(self, tree=None):
        """Initializes the SettingsTree.

        Parameters
        ----------
        tree : list or dict, optional
            The initial tree structure. If not provided, an empty list is used.
            The default is None.

        """
        # Initialize the tree structure.
        # If a tree is provided, use it; otherwise, start with an empty list.
        self.tree = tree if tree is not None else []

    def add_item(self, key, default_value=None, info=None, input_format=None,
                 value_range=None, value_options=None):
        """Adds a new item or sub-item to the tree.

        Parameters
        ----------
        key : str
            The key name of the item.
        default_value : str, int, float, bool or list, optional
            Default value for the item. The default is None.
        info : str, optional
            A description or help text for the item. The default is None.
        input_format : {'checkbox', 'spinbox', 'doublespinbox', 'lineedit', 'combobox'}, optional
            The type of UI control for the item. The default is None.
        value_range : list of [min, max], optional
            The minimum and maximum limits for numeric inputs. Use `None` for
            no bound. The default is None.
        value_options : list, optional
            A list of allowed options for 'combobox' inputs. The default is None.

        Returns
        -------
        SettingsTree or None
            A `SettingsTree` instance wrapping the added item, or None if an
            error occurs.

        Raises
        ------
        TypeError
            If `key` cannot be converted to string.
        ValueError
            If `default_value` is not valid.

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
        """Retrieves a nested item from the tree.

        This method recursively navigates the tree using a sequence of keys.

        Parameters
        ----------
        *keys : str
            A sequence of keys to navigate to the desired item.

        Returns
        -------
        SettingsTree
            A `SettingsTree` instance wrapping the found item.

        Raises
        ------
        KeyError
            If a key in the sequence is not found.

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
        """Edits the properties of the current item.

        Parameters
        ----------
        default_value : str, int, float, bool or list, optional
            The new default value for the item. The default is None.
        info : str, optional
            The new description or help text. The default is None.
        input_format : {'checkbox', 'spinbox', 'doublespinbox', 'lineedit', 'combobox'}, optional
            The new UI control type. The default is None.
        value_range : list of [min, max], optional
            The new numeric limits. The default is None.
        value_options : list, optional
            The new list of options for a 'combobox'. The default is None.

        Returns
        -------
        SettingsTree
            The current `SettingsTree` instance.

        Raises
        ------
        TypeError
            If the wrapped tree is not a dictionary and cannot be edited.

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
        """Updates the tree with values from a QTreeWidget.

        This method traverses a QTreeWidget and the internal tree, updating
        the 'default_value' in the tree with values from the widget's input
        controls.

        Parameters
        ----------
        tree_widget : QTreeWidget
            The widget containing the user-edited values.

        Returns
        -------
        SettingsTree
            The current `SettingsTree` instance.

        """
        def extract_value_from_widget(widget):
            """Extracts the value from a given widget.

            Parameters
            ----------
            widget : QWidget
                The widget from which to extract the value.

            Returns
            -------
            object or None
                The value from the widget, or None if the widget type is
                not supported.

            """
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
            """Extracts values from a list-like structure in the widget.

            Parameters
            ----------
            parent_item : QTreeWidgetItem
                The parent item in the tree widget.

            Returns
            -------
            list
                A list of values extracted from the child items.

            """
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
            """Traverses a tree item and its corresponding node to update values.

            Parameters
            ----------
            item : QTreeWidgetItem
                The item in the QTreeWidget.
            node : dict
                The corresponding node in the SettingsTree.

            """
            # Retrieve the widget associated with the "Value" column for the
            # current item
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
        """Validates the type of the default value.

        Parameters
        ----------
        default_value : any
            The value to validate.

        Returns
        -------
        bool
            True if the value is a valid type (str, int, float, bool, list),
            False otherwise.

        """
        # Validate that the default_value is one of the allowed types
        valid_types = (str, int, float, bool, list)
        if not isinstance(default_value, valid_types):
            print(f"Error: 'default_value' must be of type: string, int, float, bool, list.")
            return False
        return True

    def validate_info(self, info):
        """Validates that the info field is a string.

        Parameters
        ----------
        info : str, optional
            The info text to validate.

        Returns
        -------
        str or None
            The original `info` if it is a string, otherwise `None`.

        """
        # Validate that 'info' is a string, otherwise keep it as None
        if info is not None and not isinstance(info, str):
            print("Warning: 'info' must be a string describing the element. "
                  "Keeping it as None.")
            return None
        return info

    def validate_input_format(self, input_format, default_value=None, value_options=None):
        """Validates the input format and its consistency.

        Checks if the `input_format` is compatible with the `default_value`
        and `value_options`.

        Parameters
        ----------
        input_format : str, optional
            The UI control type to validate.
        default_value : any, optional
            The default value to check for type consistency.
        value_options : list, optional
            The options required for 'combobox' format.

        Returns
        -------
        str or None
            The validated `input_format` if valid, otherwise `None`.

        """
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
        """Validates that the value range is a list of two elements.

        Parameters
        ----------
        value_range : list, optional
            The value range to validate.

        Returns
        -------
        list or None
            The validated `value_range` if valid, otherwise `None`.

        """
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
        """Validates that the value options is a list.

        Parameters
        ----------
        value_options : list, optional
            The value options to validate.

        Returns
        -------
        list or None
            The validated `value_options` if it is a list, otherwise `None`.

        """
        # Validate that 'value_options' is a list
        if value_options is not None and not isinstance(value_options, list):
            print("Warning: 'value_options' must be a list containing the available options. The list will be kept empty.")
            return None
        return value_options

    def to_serializable_obj(self):
        """Returns the tree as a serializable object.

        Returns
        -------
        list or dict
            The serializable representation of the tree.

        """
        return self.tree

    @classmethod
    def from_serializable_obj(cls, data):
        """Creates a SettingsTree from a serializable object.

        Parameters
        ----------
        data : list or dict
            The data to create the tree from.

        Returns
        -------
        SettingsTree
            A new `SettingsTree` instance.

        """
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
