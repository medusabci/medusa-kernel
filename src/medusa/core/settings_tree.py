"""Typed, hierarchical configuration schema (Qt-free).

A :class:`SettingsTree` is a tree of named *items*. Each item carries a
``value`` (and the developer ``default`` it was built with), optional help
``info``, an optional ``input_format`` render hint, and optional constraints
(``value_range`` for numerics, ``value_options`` for enums, ``element_schema``
for the elements of a list value). Items may nest: an item created with
:meth:`SettingsTree.add_group` (or any item that gains children) becomes a
*branch*; items with a value are *leaves*.

The class is the schema half of MEDUSA's "build it, edit it, read it back"
loop:

* build a schema with :meth:`add_item`/:meth:`add_group` (or declaratively from
  a plain dict with :meth:`from_dict`),
* render and edit it with ``medusa.widgets.settings_tree`` (the Qt half), then
  read the edited tree back with ``SettingsTreeWidget.get_settings()``,
* consume the result as a plain nested ``{key: value}`` config with
  :meth:`to_dict`.

This module imports no Qt and persists only to JSON (:meth:`to_json` /
:meth:`from_json`).

Examples
--------
>>> settings = SettingsTree()
>>> settings.add_item("update_rate", value=0.2,
...                   info="Update rate (s) of the plot", value_range=[0, None])
>>> freq = settings.add_group("frequency_filter", info="Real-time IIR filter")
>>> freq.add_item("apply", value=True, info="Apply the filter")
>>> freq.add_item("type", value="highpass",
...               value_options=["highpass", "lowpass", "bandpass", "stopband"])
>>> freq.add_item("order", value=5, value_range=[1, None])
>>> settings.to_dict()
{'update_rate': 0.2, 'frequency_filter': {'apply': True, 'type': 'highpass', 'order': 5}}
"""

from __future__ import annotations

import json
import warnings
from typing import Any, Iterator, Optional, Union

__all__ = ["SettingsTree", "infer_input_format", "INPUT_FORMATS"]

#: A leaf value: a JSON primitive or a list of them.
Primitive = Union[str, int, float, bool, list]

#: The render hints understood by the Qt editor. ``input_format`` is almost
#: always derivable from the value's type (see :func:`infer_input_format`); it
#: is stored on a node only when it overrides that default.
INPUT_FORMATS = ("checkbox", "spinbox", "doublespinbox", "lineedit",
                 "combobox", "list")

_PRIMITIVE_TYPES = (str, int, float, bool, list)


# ---------------------------------------------------------------------------
# Module-level helpers (single source of truth, shared with the Qt widget)
# ---------------------------------------------------------------------------
def infer_input_format(value: Any,
                       value_options: Optional[list] = None) -> Optional[str]:
    """Infer the editor input format from a value's type.

    This is the single source of truth shared by the schema (to fill a node's
    render hint) and by the Qt widget (to build the right editor). The presence
    of ``value_options`` always means a ``combobox``; otherwise the format
    follows the Python type.

    Parameters
    ----------
    value : Any
        The leaf value.
    value_options : list, optional
        Allowed options; when given, the format is ``combobox``.

    Returns
    -------
    str or None
        One of :data:`INPUT_FORMATS`, or ``None`` if ``value`` has no editor
        (e.g. a value-less branch).
    """
    if value_options is not None:
        return "combobox"
    # bool must be checked before int (bool is a subclass of int).
    if isinstance(value, bool):
        return "checkbox"
    if isinstance(value, int):
        return "spinbox"
    if isinstance(value, float):
        return "doublespinbox"
    if isinstance(value, str):
        return "lineedit"
    if isinstance(value, list):
        return "list"
    return None


def _validate_key(key: Any) -> str:
    if key is None:
        raise TypeError("'key' is required and cannot be None.")
    if isinstance(key, str):
        return key
    try:
        return str(key)
    except Exception as exc:  # pragma: no cover - str() rarely raises
        raise TypeError(f"'key' must be a string; cannot convert {key!r}.") \
            from exc


def _validate_value(value: Any) -> Primitive:
    if not isinstance(value, _PRIMITIVE_TYPES):
        raise TypeError(
            "'value' must be one of str, int, float, bool, list; got "
            f"{type(value).__name__}.")
    return value


def _validate_info(info: Any) -> Optional[str]:
    if info is not None and not isinstance(info, str):
        raise TypeError("'info' must be a string.")
    return info


def _validate_value_range(value_range: Any) -> Optional[list]:
    if value_range is None:
        return None
    if not isinstance(value_range, list) or len(value_range) != 2:
        raise ValueError(
            "'value_range' must be a list of exactly two elements "
            "[min, max] (use None for an unbounded side).")
    return value_range


def _validate_value_options(value_options: Any) -> Optional[list]:
    if value_options is None:
        return None
    if not isinstance(value_options, list):
        raise ValueError("'value_options' must be a list of allowed options.")
    return value_options


def _validate_element_schema(element_schema: Any) -> Optional[dict]:
    if element_schema is None:
        return None
    if not isinstance(element_schema, dict):
        raise TypeError(
            "'element_schema' must be a dict describing the list elements, "
            "e.g. {'input_format': 'spinbox', 'value_range': [0, None]}.")
    # Validate the nested constraints the same way leaf constraints are.
    _validate_value_range(element_schema.get("value_range"))
    _validate_value_options(element_schema.get("value_options"))
    return element_schema


def _check_format_consistency(fmt: str, value: Any,
                              value_options: Optional[list]) -> None:
    """Raise ``ValueError`` if an explicit ``fmt`` is incompatible with value."""
    if fmt == "combobox" and value_options is None:
        raise ValueError("input_format 'combobox' requires 'value_options'.")
    if fmt == "checkbox" and not isinstance(value, bool):
        raise ValueError("input_format 'checkbox' requires a bool value.")
    if fmt == "spinbox" and (not isinstance(value, int)
                             or isinstance(value, bool)):
        raise ValueError("input_format 'spinbox' requires an int value.")
    if fmt == "doublespinbox" and not isinstance(value, float):
        raise ValueError("input_format 'doublespinbox' requires a float value.")
    if fmt == "list" and not isinstance(value, list):
        raise ValueError("input_format 'list' requires a list value.")


def _resolve_input_format(input_format: Optional[str], value: Any,
                          value_options: Optional[list]) -> Optional[str]:
    """Validate an explicit ``input_format`` and keep it only if it overrides.

    Returns the format to store on the node, or ``None`` to let the widget
    infer it. An explicit format inconsistent with ``value``/``value_options``
    raises ``ValueError`` (a schema-author error).
    """
    if input_format is None:
        return None
    fmt = input_format.lower()
    if fmt not in INPUT_FORMATS:
        raise ValueError(
            f"'input_format' must be one of {INPUT_FORMATS}; got "
            f"{input_format!r}.")
    _check_format_consistency(fmt, value, value_options)
    # Persist only when it carries information beyond plain type inference.
    return None if fmt == infer_input_format(value, value_options) else fmt


def _warn_out_of_constraints(key: str, value: Any,
                             value_range: Optional[list],
                             value_options: Optional[list]) -> None:
    """Soft check: warn (do not raise) if a value violates its own constraints.

    Building/editing a value outside its declared range/options is tolerated so
    schemas can be assembled incrementally; :meth:`SettingsTree.validate` and
    :meth:`SettingsTree.coerce` are the explicit tools to surface/fix it.
    """
    if value is None:
        return
    if value_options is not None and value not in value_options:
        warnings.warn(
            f"value {value!r} for {key!r} is not in value_options "
            f"{value_options}.", stacklevel=3)
    if (value_range is not None and isinstance(value, (int, float))
            and not isinstance(value, bool)):
        low, high = value_range
        if (low is not None and value < low) or \
                (high is not None and value > high):
            warnings.warn(
                f"value {value!r} for {key!r} is outside value_range "
                f"{value_range}.", stacklevel=3)


# ---------------------------------------------------------------------------
# SettingsTree
# ---------------------------------------------------------------------------
class SettingsTree:
    """A hierarchical, typed configuration schema.

    An instance wraps a single *node* dict. The root is a value-less group
    whose children are the top-level items; :meth:`add_item`/:meth:`add_group`
    return wrappers around the freshly created child nodes so building can be
    chained.

    Parameters
    ----------
    tree : dict, optional
        An existing node dict to wrap (used internally and by
        :meth:`from_serializable_obj`). When omitted, a new empty root group is
        created.

    Notes
    -----
    The on-disk node shape is::

        {
            "key": str,                # omitted on the root
            "default": <primitive>,    # value the item was built with
            "value":   <primitive>,    # current value (== default until edited)
            "info": str,               # optional help text
            "input_format": str,       # optional; only when it overrides type
            "value_range": [min, max], # optional numeric bounds (None = open)
            "value_options": [...],    # optional enum
            "element_schema": {...},   # optional template for list elements
            "items": [ <node>, ... ],  # present iff this node is a branch
        }
    """

    def __init__(self, tree: Optional[dict] = None):
        self.tree: dict = {"items": []} if tree is None else tree

    # -- node accessors -----------------------------------------------------
    @property
    def key(self) -> Optional[str]:
        """The node's key (``None`` for the root)."""
        return self.tree.get("key")

    @property
    def value(self) -> Optional[Primitive]:
        """The node's current value (falls back to its default)."""
        return self.tree.get("value", self.tree.get("default"))

    @value.setter
    def value(self, new_value: Primitive) -> None:
        self.edit_item(value=new_value)

    @property
    def default(self) -> Optional[Primitive]:
        """The node's build-time default value."""
        return self.tree.get("default")

    @property
    def info(self) -> Optional[str]:
        """The node's help text, if any."""
        return self.tree.get("info")

    @property
    def is_group(self) -> bool:
        """Whether the node is a branch (has children)."""
        return "items" in self.tree

    def _child_nodes(self) -> list:
        return self.tree.get("items", [])

    # -- building -----------------------------------------------------------
    def add_item(self, key: str, value: Optional[Primitive] = None,
                 info: Optional[str] = None,
                 input_format: Optional[str] = None,
                 value_range: Optional[list] = None,
                 value_options: Optional[list] = None,
                 element_schema: Optional[dict] = None) -> "SettingsTree":
        """Add a leaf item under this node and return it (for chaining).

        Parameters
        ----------
        key : str
            Item name (unique among its siblings).
        value : str, int, float, bool or list, optional
            The item's value; also captured as its ``default``.
        info : str, optional
            Help text shown next to the item.
        input_format : {'checkbox', 'spinbox', 'doublespinbox', 'lineedit', \
'combobox', 'list'}, optional
            Editor hint. Usually inferred from ``value``; pass it only to force
            a non-default editor (e.g. a ``combobox`` over a free value). Must
            be consistent with ``value``/``value_options``.
        value_range : list of [min, max], optional
            Numeric bounds; use ``None`` for an unbounded side.
        value_options : list, optional
            Allowed options (turns the editor into a ``combobox``).
        element_schema : dict, optional
            For a list ``value``, a template applied to every element, e.g.
            ``{'input_format': 'spinbox', 'value_range': [0, None]}``.

        Returns
        -------
        SettingsTree
            A wrapper around the newly added item.

        Raises
        ------
        TypeError
            If ``key``/``value``/``info``/``element_schema`` have invalid types.
        ValueError
            If ``key`` duplicates a sibling, or ``input_format``/``value_range``/
            ``value_options`` are invalid or inconsistent.
        """
        node = self._build_node(key, value, info, input_format, value_range,
                                value_options, element_schema)
        self._append(node)
        return SettingsTree(node)

    def add_group(self, key: str,
                  info: Optional[str] = None) -> "SettingsTree":
        """Add a branch (a value-less container) and return it for chaining.

        Parameters
        ----------
        key : str
            Group name (unique among its siblings).
        info : str, optional
            Help text for the group.

        Returns
        -------
        SettingsTree
            A wrapper around the new group; add children to it.
        """
        key = _validate_key(key)
        info = _validate_info(info)
        self._reject_duplicate(key)
        node: dict = {"key": key, "items": []}
        if info is not None:
            node["info"] = info
        self._append(node)
        return SettingsTree(node)

    @staticmethod
    def _build_node(key, value, info, input_format, value_range,
                    value_options, element_schema) -> dict:
        key = _validate_key(key)
        if value is not None:
            value = _validate_value(value)
        info = _validate_info(info)
        value_range = _validate_value_range(value_range)
        value_options = _validate_value_options(value_options)
        element_schema = _validate_element_schema(element_schema)
        stored_format = _resolve_input_format(input_format, value,
                                              value_options)
        _warn_out_of_constraints(key, value, value_range, value_options)

        node: dict = {"key": key}
        if value is not None:
            node["default"] = value
            node["value"] = value
        if info is not None:
            node["info"] = info
        if stored_format is not None:
            node["input_format"] = stored_format
        if value_range is not None:
            node["value_range"] = value_range
        if value_options is not None:
            node["value_options"] = value_options
        if element_schema is not None:
            node["element_schema"] = element_schema
        return node

    def _reject_duplicate(self, key: str) -> None:
        if any(child.get("key") == key for child in self._child_nodes()):
            raise ValueError(
                f"duplicate key {key!r}: an item with this key already exists "
                "at this level.")

    def _append(self, node: dict) -> None:
        self._reject_duplicate(node["key"])
        self.tree.setdefault("items", []).append(node)

    # -- navigation ---------------------------------------------------------
    def get_item(self, *keys: str) -> "SettingsTree":
        """Return the nested item reached by following ``keys``.

        Raises
        ------
        KeyError
            If any key in the path is not found.
        """
        node = self.tree
        for key in keys:
            for child in node.get("items", []):
                if child.get("key") == key:
                    node = child
                    break
            else:
                raise KeyError(f"key {key!r} not found in the tree.")
        return SettingsTree(node)

    def get_item_value(self, *keys: str, default: Any = None) -> Any:
        """Return the value of the nested item reached by following ``keys``.

        For a branch, returns its nested ``{key: value}`` dict
        (:meth:`to_dict`). For a leaf with no value, returns ``default``.

        Raises
        ------
        KeyError
            If any key in the path is not found.
        """
        item = self.get_item(*keys)
        if item.is_group:
            return item.to_dict()
        return item.tree.get("value", item.tree.get("default", default))

    def remove_item(self, *keys: str) -> "SettingsTree":
        """Remove the nested item reached by following ``keys``.

        Returns ``self`` for chaining.

        Raises
        ------
        KeyError
            If any key in the path is not found.
        """
        if not keys:
            raise KeyError("remove_item requires at least one key.")
        parent = self.get_item(*keys[:-1]) if keys[:-1] else self
        siblings = parent.tree.get("items", [])
        for idx, child in enumerate(siblings):
            if child.get("key") == keys[-1]:
                del siblings[idx]
                return self
        raise KeyError(f"key {keys[-1]!r} not found in the tree.")

    # -- editing ------------------------------------------------------------
    def edit_item(self, value: Optional[Primitive] = None,
                  info: Optional[str] = None,
                  input_format: Optional[str] = None,
                  value_range: Optional[list] = None,
                  value_options: Optional[list] = None,
                  element_schema: Optional[dict] = None) -> "SettingsTree":
        """Edit this node's fields in place (only the given ones change).

        Editing ``value`` changes the current value but not the ``default``
        (use :meth:`reset` to restore the default).

        Returns
        -------
        SettingsTree
            ``self``.
        """
        node = self.tree
        if value is not None:
            node["value"] = _validate_value(value)
        if info is not None:
            node["info"] = _validate_info(info)
        if value_range is not None:
            node["value_range"] = _validate_value_range(value_range)
        if value_options is not None:
            node["value_options"] = _validate_value_options(value_options)
        if element_schema is not None:
            node["element_schema"] = _validate_element_schema(element_schema)
        if input_format is not None:
            eff_value = node.get("value", node.get("default"))
            eff_options = node.get("value_options")
            stored = _resolve_input_format(input_format, eff_value, eff_options)
            if stored is None:
                node.pop("input_format", None)
            else:
                node["input_format"] = stored
        _warn_out_of_constraints(node.get("key"),
                                 node.get("value", node.get("default")),
                                 node.get("value_range"),
                                 node.get("value_options"))
        return self

    def set_value(self, *keys: str, value: Primitive) -> "SettingsTree":
        """Navigate to ``keys`` and set its value in one call.

        Returns ``self`` for chaining.

        Raises
        ------
        KeyError
            If any key in the path is not found.
        """
        self.get_item(*keys).edit_item(value=value)
        return self

    def reset(self, *keys: str) -> "SettingsTree":
        """Restore values to their build-time defaults (deep).

        With no ``keys``, resets this whole subtree; otherwise resets the item
        at ``keys`` and its descendants. Returns ``self``.
        """
        target = self.get_item(*keys) if keys else self
        self._reset_node(target.tree)
        return self

    @classmethod
    def _reset_node(cls, node: dict) -> None:
        if "items" in node:
            for child in node["items"]:
                cls._reset_node(child)
        elif "default" in node:
            node["value"] = node["default"]

    # -- plain-dict bridge --------------------------------------------------
    def to_dict(self) -> dict:
        """Collapse this subtree into a plain nested ``{key: value}`` config.

        Branches become nested dicts; leaves map to their current value. This
        is the lossy *values* projection consumers feed into pipelines (distinct
        from :meth:`to_serializable_obj`, the lossless schema).

        Raises
        ------
        ValueError
            If two siblings share a key (the projection would be ambiguous).
        """
        config: dict = {}
        for child in self._child_nodes():
            key = child.get("key")
            if key in config:
                raise ValueError(
                    f"duplicate key {key!r}: cannot project to a plain dict.")
            if "items" in child:
                config[key] = SettingsTree(child).to_dict()
            else:
                config[key] = child.get("value", child.get("default"))
        return config

    #: Alias of :meth:`to_dict`.
    values = to_dict

    @classmethod
    def from_dict(cls, data: dict,
                  infos: Optional[dict] = None) -> "SettingsTree":
        """Build a schema from a plain nested dict, inferring input formats.

        Nested dicts become groups; scalars/lists become leaves whose editor is
        inferred from the value type. An optional ``infos`` dict mirrors
        ``data``'s structure with help strings for leaves.

        Parameters
        ----------
        data : dict
            Nested ``{key: value}`` configuration.
        infos : dict, optional
            Parallel nested dict of help strings (string at a leaf key, nested
            dict at a group key).

        Returns
        -------
        SettingsTree
            A new schema. ``SettingsTree.from_dict(d).to_dict() == d``.
        """
        tree = cls()
        cls._populate(tree, data, infos or {})
        return tree

    @staticmethod
    def _populate(parent: "SettingsTree", data: dict, infos: dict) -> None:
        for key, value in data.items():
            sub_info = infos.get(key) if isinstance(infos, dict) else None
            if isinstance(value, dict):
                group = parent.add_group(
                    key, info=sub_info if isinstance(sub_info, str) else None)
                SettingsTree._populate(
                    group, value,
                    sub_info if isinstance(sub_info, dict) else {})
            else:
                parent.add_item(
                    key, value=value,
                    info=sub_info if isinstance(sub_info, str) else None)

    def update_from_dict(self, data: dict) -> "SettingsTree":
        """Bulk-set current values from a plain nested dict (schema preserved).

        Keys absent from the schema are skipped with a warning. Returns
        ``self``.
        """
        for key, value in data.items():
            try:
                item = self.get_item(key)
            except KeyError:
                warnings.warn(f"update_from_dict: unknown key {key!r}; "
                              "skipping.", stacklevel=2)
                continue
            if isinstance(value, dict) and item.is_group:
                item.update_from_dict(value)
            else:
                item.edit_item(value=value)
        return self

    def user_overrides(self) -> dict:
        """Return only the values that differ from their defaults (nested)."""
        overrides: dict = {}
        for child in self._child_nodes():
            key = child.get("key")
            if "items" in child:
                nested = SettingsTree(child).user_overrides()
                if nested:
                    overrides[key] = nested
            elif "default" in child and \
                    child.get("value", child["default"]) != child["default"]:
                overrides[key] = child.get("value")
        return overrides

    # -- validation / coercion ---------------------------------------------
    def validate(self) -> list:
        """Return constraint violations as ``(path, reason)`` tuples.

        Checks each leaf's current value against its ``value_options`` and
        ``value_range``. An empty list means the tree is valid.
        """
        violations: list = []
        self._validate_node(self.tree, [], violations)
        return violations

    @classmethod
    def _validate_node(cls, node: dict, path: list, out: list) -> None:
        if "items" in node:
            for child in node["items"]:
                cls._validate_node(child, path + [child.get("key")], out)
            return
        value = node.get("value", node.get("default"))
        dotted = ".".join(str(p) for p in path)
        options = node.get("value_options")
        if options is not None and value not in options:
            out.append((dotted, f"{value!r} not in options {options}"))
        rng = node.get("value_range")
        if rng is not None and isinstance(value, (int, float)) \
                and not isinstance(value, bool):
            low, high = rng
            if low is not None and value < low:
                out.append((dotted, f"{value!r} < min {low}"))
            if high is not None and value > high:
                out.append((dotted, f"{value!r} > max {high}"))

    def coerce(self, clamp: bool = True,
               snap_options: bool = True) -> "SettingsTree":
        """Force values into their declared constraints (explicit, in place).

        Parameters
        ----------
        clamp : bool, default True
            Clamp out-of-range numerics to the nearest bound.
        snap_options : bool, default True
            Replace an out-of-options value with the item's default (if valid)
            else the first option.

        Returns
        -------
        SettingsTree
            ``self``.
        """
        self._coerce_node(self.tree, clamp, snap_options)
        return self

    @classmethod
    def _coerce_node(cls, node: dict, clamp: bool, snap_options: bool) -> None:
        if "items" in node:
            for child in node["items"]:
                cls._coerce_node(child, clamp, snap_options)
            return
        value = node.get("value", node.get("default"))
        options = node.get("value_options")
        if snap_options and options and value not in options:
            default = node.get("default")
            node["value"] = default if default in options else options[0]
            value = node["value"]
        rng = node.get("value_range")
        if clamp and rng is not None and isinstance(value, (int, float)) \
                and not isinstance(value, bool):
            low, high = rng
            if low is not None and value < low:
                node["value"] = low
            elif high is not None and value > high:
                node["value"] = high

    # -- serialization (JSON only) -----------------------------------------
    def to_serializable_obj(self) -> dict:
        """Return the lossless node dict (the canonical schema)."""
        return self.tree

    @classmethod
    def from_serializable_obj(cls, data: dict) -> "SettingsTree":
        """Rebuild a :class:`SettingsTree` from :meth:`to_serializable_obj`."""
        return cls(data)

    def to_json(self, path: str, encoding: str = "utf-8",
                indent: int = 4) -> None:
        """Write the schema to a JSON file."""
        with open(path, "w", encoding=encoding) as f:
            json.dump(self.tree, f, indent=indent)

    @classmethod
    def from_json(cls, path: str, encoding: str = "utf-8") -> "SettingsTree":
        """Read a schema from a JSON file written by :meth:`to_json`."""
        with open(path, "r", encoding=encoding) as f:
            return cls(json.load(f))

    # -- dunders ------------------------------------------------------------
    def __getitem__(self, key: str) -> "SettingsTree":
        return self.get_item(key)

    def __contains__(self, key: str) -> bool:
        return any(child.get("key") == key for child in self._child_nodes())

    def __iter__(self) -> Iterator["SettingsTree"]:
        return (SettingsTree(child) for child in self._child_nodes())

    def __len__(self) -> int:
        return len(self._child_nodes())

    def __repr__(self) -> str:
        if self.is_group:
            keys = [child.get("key") for child in self._child_nodes()]
            label = "root" if self.key is None else repr(self.key)
            return f"SettingsTree({label}, items={keys})"
        return f"SettingsTree(item {self.key!r}={self.value!r})"
