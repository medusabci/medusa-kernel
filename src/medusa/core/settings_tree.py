from medusa.components import SerializableComponent


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
    remove_item(*keys)
        Removes a nested item from the tree using a sequence of keys
    update_tree_from_widget(tree_widget)
        Updates the tree with values from a QTreeWidget. Requires PySide6
        (install with the ``[widgets]`` extra).
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

    def add_item(self, key, value=None, info=None, input_format=None,
                 value_range=None, value_options=None):
        """Adds a new item or sub-item to the tree.

        Parameters
        ----------
        key : str
            The key name of the item.
        value : {str, int, float, bool, list}, optional
            Value for this item. Can be a primitive type or a list.
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
        if value is not None and not self.validate_value(value):
            return None
        info = self.validate_info(info)
        input_format = self.validate_input_format(input_format)
        value_range = self.validate_value_range(value_range)
        value_options = self.validate_value_options(value_options)

        # Build the item dictionary
        item = {
            'key': key,
            'value': value,
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

    def edit_item(self, value=None, info=None, input_format=None, value_range=None, value_options=None):
        """
        Edits the properties of the current item.

        Parameters
        ----------
        value : str, int, float, bool or list, optional
            The new value for the item. The default is None.
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

        if value is not None: tree['value'] = value if self.validate_value(
            value) else tree.get('value')
        if info is not None: tree['info'] = self.validate_info(info)
        if input_format is not None: tree['input_format'] = self.validate_input_format(input_format)
        if value_range is not None: tree['value_range'] = self.validate_value_range(value_range)
        if value_options is not None: tree['value_options'] = self.validate_value_options(value_options)
        return self

    def remove_item(self, *keys):
        """
        Removes a nested item from the tree using a sequence of keys.

        Parameters:
            *keys: Sequence of keys to navigate through nested items.
        """
        current_node = self.tree
        parent_items = None
        index_to_delete = None
        for key in keys:
            found = None
            items = current_node.get('items', []) if isinstance(current_node, dict) else current_node
            for idx, item in enumerate(items):
                if item.get('key') == key:
                    found = item
                    parent_items = items
                    index_to_delete = idx
                    break
            if found is None:
                raise KeyError(f"Key '{key}' not found in the tree.")
            current_node = found
        if parent_items is not None and index_to_delete is not None:
            del parent_items[index_to_delete]

    def get_item_value(self, *keys):
        item = self.get_item(*keys)
        return item.to_serializable_obj()['value']

    def update_tree_from_widget(self, tree_widget):
        """Updates the tree with values from a QTreeWidget.

        This method traverses a QTreeWidget and the internal tree, updating
        the 'default_value' in the tree with values from the widget's input
        controls.

        Requires PySide6 (install with the ``[widgets]`` extra).

        Parameters
        ----------
        tree_widget : QTreeWidget
            The widget containing the user-edited values.

        Returns
        -------
        SettingsTree
            The current `SettingsTree` instance.

        """
        try:
            from PySide6.QtWidgets import (
                QComboBox, QCheckBox, QSpinBox, QDoubleSpinBox, QLineEdit,
            )
        except ImportError as e:
            raise ImportError(
                "SettingsTree.update_tree_from_widget requires PySide6. "
                "Install with: pip install medusa-kernel[widgets]"
            ) from e

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
            # Retrieve the widget associated with the "Value" column for the
            # current item
            widget = tree_widget.itemWidget(item, 1)

            # Extract values from the widget
            if widget is not None:
                value = extract_value_from_widget(widget)
                SettingsTree(node).edit_item(value=value)
            elif isinstance(node.get("value"), list):
                value = extract_value_from_list(item)
                SettingsTree(node).edit_item(value=value)

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

    def validate_value(self, value):
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
        # Validate that the value is one of the allowed types
        valid_types = (str, int, float, bool, list)
        if not isinstance(value, valid_types):
            print(f"Error: 'value' must be of type: string, int, float, bool, list.")
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

    def validate_input_format(self, input_format, value=None, value_options=None):
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
        if input_format == "checkbox" and not isinstance(value, bool):
            print("Warning: 'CheckBox' requires 'value' to be a boolean (True/False). Keeping input format as None.")
            return None
        if input_format == "spinbox" and not isinstance(value, int):
            print("Warning: 'SpinBox' requires 'value' to be an integer. Keeping input format as None.")
            return None
        if input_format == "doublespinbox" and not isinstance(value, float):
            print("Warning: 'DoubleSpinBox' requires 'value' to be a float. Keeping input format as None.")
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
