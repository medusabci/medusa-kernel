"""Typed, hierarchical configuration schema (Qt-free).

A :class:`SettingsTree` is a tree of named *items*. Each item carries a
``value`` (and the developer ``default`` it was built with), optional help
``info``, an optional ``input_format`` render hint, and optional constraints
(``value_range`` for numerics, ``value_options`` for enums, ``element_schema``
for the elements of a list value). Items may nest: an item created with
:meth:`SettingsTree.add_group` (or any item that gains children) becomes a
*branch*; items with a value are *leaves*. A :meth:`SettingsTree.add_group_list`
is a *group-list* -- a variable-length list of same-schema groups (typed,
per-element validated) that projects to a plain list of dicts in :meth:`to_dict`.

The class is the schema half of MEDUSA's "build it, edit it, read it back"
loop:

* build a schema with :meth:`add_item`/:meth:`add_group` (or declaratively from
  a plain dict with :meth:`from_dict`),
* render and edit it with ``medusa.widgets.settings_tree`` (the Qt half), then
  read the edited tree back with ``SettingsTreeWidget.get_settings()``,
* consume the result as a plain nested ``{key: value}`` config with
  :meth:`to_dict`,
* and, at any point, look at it from a terminal with :meth:`print_tree`
  (or just ``print(settings)``).

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

import copy
import io
import json
import warnings
from abc import ABC, abstractmethod
from typing import Any, Iterator, Optional, Union

__all__ = ["SettingsTree", "Configurable", "infer_input_format", "INPUT_FORMATS"]

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
# Console rendering helpers (the text half of SettingsTree.print_tree)
# ---------------------------------------------------------------------------
class _AsciiStringIO(io.StringIO):
    """A capture buffer that declares an ASCII encoding.

    ``rich`` picks its tree guides from the encoding its output file reports.
    A plain :class:`io.StringIO` reports none, so ``rich`` assumes UTF-8 and
    draws the guides with box-drawing characters (``|-- `` becomes ``├── ``),
    which then crash on a console whose code page cannot encode them (a common
    case on Windows).
    Declaring ASCII keeps :meth:`SettingsTree.describe` printable everywhere
    and identical on every platform.
    """

    encoding = "ascii"


#: ``rich`` styles of the console rendering.
_STYLE_KEY = "bold"
_STYLE_VALUE = "cyan"
_STYLE_EDITED = "bold yellow"       # a value that differs from its default
_STYLE_META = "dim"
_STYLE_INFO = "dim italic"


def _format_range(value_range: list) -> str:
    """Render a ``value_range`` as ``'1..10'``, ``'>= 0'`` or ``'<= 5'``."""
    low, high = value_range
    if low is not None and high is not None:
        return f"{low}..{high}"
    if low is not None:
        return f">= {low}"
    if high is not None:
        return f"<= {high}"
    return "any"


def _node_meta(node: dict) -> list:
    """The ``detail='full'`` annotations of a leaf, in display order.

    The ``default`` is listed only when the current value differs from it, so a
    reader sees at a glance which items were edited.
    """
    meta = []
    value = node.get("value", node.get("default"))
    if "default" in node and node["default"] != value:
        meta.append(f"default: {node['default']!r}")
    options = node.get("value_options")
    if options is not None:
        meta.append("options: " + ", ".join(repr(o) for o in options))
    if node.get("value_range") is not None:
        meta.append("range: " + _format_range(node["value_range"]))
    if node.get("input_format") is not None:
        meta.append(f"format: {node['input_format']}")
    if node.get("element_schema") is not None:
        meta.append(f"element_schema: {node['element_schema']}")
    return meta


def _leaf_parts(node: dict, detail: str) -> list:
    """Build the ``(text, style)`` parts of a leaf line."""
    value = node.get("value", node.get("default"))
    edited = "default" in node and node["default"] != value
    parts: list = [(str(node.get("key")), _STYLE_KEY), " = "]
    if "value" not in node and "default" not in node:    # a required, unset item
        parts.append(("<not set>", _STYLE_INFO))
    else:
        parts.append((repr(value), _STYLE_EDITED if edited else _STYLE_VALUE))
    if detail == "full":
        meta = _node_meta(node)
        if meta:
            parts.append(("  [" + " | ".join(meta) + "]", _STYLE_META))
        if node.get("info"):
            parts.append(("  # " + node["info"], _STYLE_INFO))
    return parts


def _branch_parts(node: dict, detail: str) -> list:
    """Build the ``(text, style)`` parts of a group (or group-list) line."""
    parts: list = [(str(node.get("key")), _STYLE_KEY)]
    if "element_group" in node:
        n = len(node.get("items", []))
        parts.append((f"  (group-list, {n} element{'' if n == 1 else 's'})",
                      _STYLE_META))
    if detail == "full" and node.get("info"):
        parts.append(("  # " + node["info"], _STYLE_INFO))
    return parts


def _add_nodes(branch, node: dict, detail: str) -> None:
    """Add ``node``'s children to a ``rich`` tree branch (recursive)."""
    from rich.text import Text
    if "element_group" in node:                          # group-list
        elements = node.get("items", [])
        if detail == "full" and not elements:            # empty: show the schema
            _add_nodes(branch.add(Text("<template>", style=_STYLE_INFO)),
                       node["element_group"], detail)
        for i, element in enumerate(elements):
            _add_nodes(branch.add(Text(f"[{i}]", style=_STYLE_META)),
                       element, detail)
        return
    for child in node.get("items", []):
        if "items" in child:                             # group or group-list
            _add_nodes(branch.add(Text.assemble(*_branch_parts(child, detail))),
                       child, detail)
        else:
            branch.add(Text.assemble(*_leaf_parts(child, detail)))


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
            "element_group": {...},    # group-list only: the element (group) template
            "items": [ <node>, ... ],  # branch children, or (group-list) element instances
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

    @property
    def is_group_list(self) -> bool:
        """Whether the node is a *group-list* (a repeated group; see :meth:`add_group_list`)."""
        return "element_group" in self.tree

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

    # -- group-lists (a variable-length list of same-schema groups) ----------
    def add_group_list(self, key: str,
                       info: Optional[str] = None) -> "SettingsTree":
        """Add a *group-list*: a variable-length list whose elements are groups.

        Every element shares one schema -- the **element template** -- which you populate
        once via :attr:`element` (with the ordinary :meth:`add_item`). Append instances
        (clones of the template) with :meth:`add_element`. In the value projection
        (:meth:`to_dict`) a group-list becomes a **list of dicts**, one per element, so
        consumers read it like any list value while the schema stays typed, per-element
        validated (:meth:`validate`) and editable (add/remove elements).

        Returns
        -------
        SettingsTree
            A wrapper around the new group-list node.
        """
        key = _validate_key(key)
        info = _validate_info(info)
        self._reject_duplicate(key)
        node: dict = {"key": key, "input_format": "group_list",
                      "element_group": {"items": []}, "items": []}
        if info is not None:
            node["info"] = info
        self._append(node)
        return SettingsTree(node)

    @property
    def element(self) -> "SettingsTree":
        """The element **template** group of a group-list; populate it with :meth:`add_item`."""
        if not self.is_group_list:
            raise TypeError("'element' is only available on a group-list (add_group_list).")
        return SettingsTree(self.tree["element_group"])

    @property
    def elements(self) -> "list[SettingsTree]":
        """The group-list's current element instances, as wrappers (in order)."""
        if not self.is_group_list:
            raise TypeError("'elements' is only available on a group-list (add_group_list).")
        return [SettingsTree(e) for e in self.tree["items"]]

    def add_element(self, values: Optional[dict] = None) -> "SettingsTree":
        """Append one element (a clone of the template) to a group-list; return its wrapper.

        ``values`` optionally overrides some of the element's leaf values (the rest keep the
        template defaults). A plain append -- it does **not** touch the group-list's default
        (that baseline is captured once by :meth:`snapshot_defaults`), so it is safe both to
        seed the schema and to add at edit time.
        """
        if not self.is_group_list:
            raise TypeError("add_element is only valid on a group-list (add_group_list).")
        element = copy.deepcopy(self.tree["element_group"])       # {"items": [leaf clones]}
        self.tree["items"].append(element)
        handle = SettingsTree(element)
        if values:
            handle.update_from_dict(values)
        return handle

    def set_elements(self, values_list: list) -> "SettingsTree":
        """Replace a group-list's elements from a list of value dicts. Returns ``self``.

        Each dict builds one element (a clone of the template with those overrides). Like
        :meth:`add_element`/:meth:`remove_element`, an edit -- it does not touch the default.
        """
        if not self.is_group_list:
            raise TypeError("set_elements is only valid on a group-list (add_group_list).")
        self.tree["items"] = []
        for values in values_list:
            self.add_element(values if isinstance(values, dict) else None)
        return self

    def remove_element(self, index: int) -> "SettingsTree":
        """Remove the element at ``index`` from a group-list. Returns ``self``."""
        if not self.is_group_list:
            raise TypeError("remove_element is only valid on a group-list (add_group_list).")
        del self.tree["items"][index]
        return self

    def snapshot_defaults(self) -> "SettingsTree":
        """Capture the current elements of every group-list as its default (idempotent).

        A group-list has no scalar ``default``; call this **once after building the schema and
        before editing** so :meth:`reset` / :meth:`user_overrides` have a baseline (the
        :class:`Configurable` mixin does it automatically). Group-lists that already have a
        default are left untouched. Returns ``self``.
        """
        self._snapshot_defaults_node(self.tree)
        return self

    @classmethod
    def _snapshot_defaults_node(cls, node: dict) -> None:
        if "element_group" in node:                 # group-list -> snapshot its element list
            # deep-copy so the baseline never aliases (shares list objects with) live elements
            node.setdefault("default", copy.deepcopy(SettingsTree(node).to_dict()))
            cls._snapshot_defaults_node(node["element_group"])   # template (for future clones)
            for element in node.get("items", []):                # and any nested group-lists
                cls._snapshot_defaults_node(element)
            return
        for child in node.get("items", []):
            cls._snapshot_defaults_node(child)

    def set_defaults_from_values(self) -> "SettingsTree":
        """Make the current values the defaults of this subtree (deep). Returns ``self``.

        :meth:`snapshot_defaults` only fills in a group-list baseline that is *missing*; this
        **overwrites** every default with the value next to it. Call it at the end of a builder
        that seeds a variant of a schema (a paradigm profile, say), so that the variant's own
        choices count as defaults: :meth:`reset` then restores *that* configuration, and
        :meth:`user_overrides` reports only what the user changed on top of it. Leaves that
        have no value (a required choice) are left alone, so they stay required.
        """
        self._defaults_from_values(self.tree)
        return self

    @classmethod
    def _defaults_from_values(cls, node: dict) -> None:
        if "element_group" in node:                 # group-list: baseline the elements, then the list
            for element in node.get("items", []):
                cls._defaults_from_values(element)
            node["default"] = copy.deepcopy(SettingsTree(node).to_dict())
            return
        if "items" in node:
            for child in node["items"]:
                cls._defaults_from_values(child)
            return
        if "value" in node:                         # deep-copy: a list value must not alias
            node["default"] = copy.deepcopy(node["value"])

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
        if "element_group" in node:                     # group-list: restore default elements
            if "default" in node:                       # (no baseline -> nothing to restore)
                SettingsTree(node).set_elements(node["default"])
        elif "items" in node:
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
        if self.is_group_list:                          # a group-list projects to a list of dicts
            return [SettingsTree(e).to_dict() for e in self.tree.get("items", [])]
        config: dict = {}
        for child in self._child_nodes():
            key = child.get("key")
            if key in config:
                raise ValueError(
                    f"duplicate key {key!r}: cannot project to a plain dict.")
            if "items" in child:                        # nested group or group-list (recurses)
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
            if item.is_group_list:
                if not isinstance(value, list):
                    raise ValueError(
                        f"{key!r} is a group-list; expected a list of dicts, got "
                        f"{type(value).__name__}.")
                item.set_elements(value)
            elif isinstance(value, dict) and item.is_group:
                item.update_from_dict(value)
            else:
                item.edit_item(value=value)
        return self

    def user_overrides(self) -> dict:
        """Return only the values that differ from their defaults (nested)."""
        overrides: dict = {}
        for child in self._child_nodes():
            key = child.get("key")
            if "element_group" in child:                    # group-list: differs from default?
                if "default" in child:
                    current = SettingsTree(child).to_dict()
                    if current != child["default"]:
                        overrides[key] = current
            elif "items" in child:
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
        if "element_group" in node:                     # group-list: validate each element
            for i, element in enumerate(node.get("items", [])):
                cls._validate_node(element, path + [str(i)], out)
            return
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

    # -- console rendering --------------------------------------------------
    def print_tree(self, detail: str = "values",
                   title: Optional[str] = None) -> None:
        """Print this subtree to the terminal as an indented, colored tree.

        The quickest way to see what a configuration currently holds::

            >>> settings.print_tree()
            SettingsTree
            ├── update_rate = 0.2
            └── frequency_filter
                ├── apply = True
                └── order = 5

        Values that differ from their default are highlighted. Use
        ``detail='full'`` to also see where each value may move (its range or
        options), what it was built with, and what it is for.

        Parameters
        ----------
        detail : {'values', 'full'}, default 'values'
            ``'values'`` prints only ``key = value``, i.e. the configuration as
            :meth:`to_dict` projects it. ``'full'`` adds each item's
            constraints (``value_range``, ``value_options``,
            ``element_schema``), its editor hint (``input_format``), its help
            text (``info``), and its ``default`` -- the default is shown only
            when the current value differs from it, so it marks what has been
            edited. An empty group-list also shows its element template.
        title : str, optional
            Text of the root line. Defaults to the node key, or
            ``'SettingsTree'`` for the root.

        Raises
        ------
        ValueError
            If ``detail`` is not ``'values'`` or ``'full'``.

        See Also
        --------
        describe : the same rendering, returned as a plain string.
        """
        from rich.console import Console
        Console().print(self._rich_tree(detail, title))

    def describe(self, detail: str = "values",
                 title: Optional[str] = None) -> str:
        """Return what :meth:`print_tree` prints, as plain text (no colors).

        Use it to log a configuration, to embed one in a report, or to compare
        two of them. ``str(settings)`` is this method with its defaults.

        Parameters
        ----------
        detail : {'values', 'full'}, default 'values'
            See :meth:`print_tree`.
        title : str, optional
            See :meth:`print_tree`.

        Returns
        -------
        str
            The tree, drawn with ASCII guides, wrapped at 100 columns and
            without a trailing newline.
        """
        from rich.console import Console
        console = Console(file=_AsciiStringIO(), width=100, no_color=True,
                          legacy_windows=False)
        console.print(self._rich_tree(detail, title))
        return console.file.getvalue().rstrip("\n")

    def _rich_tree(self, detail: str, title: Optional[str]):
        """Build the ``rich.tree.Tree`` shared by print_tree/describe."""
        from rich.text import Text
        from rich.tree import Tree
        if detail not in ("values", "full"):
            raise ValueError("'detail' must be 'values' or 'full'; got "
                             f"{detail!r}.")
        if title is None:
            title = "SettingsTree" if self.key is None else str(self.key)
        tree = Tree(Text(title, style=_STYLE_KEY), guide_style=_STYLE_META)
        _add_nodes(tree, self.tree, detail)
        return tree

    # -- dunders ------------------------------------------------------------
    def __getitem__(self, key: str) -> "SettingsTree":
        return self.get_item(key)

    def __contains__(self, key: str) -> bool:
        return any(child.get("key") == key for child in self._child_nodes())

    def __iter__(self) -> Iterator["SettingsTree"]:
        return (SettingsTree(child) for child in self._child_nodes())

    def __len__(self) -> int:
        return len(self._child_nodes())

    def __str__(self) -> str:
        return self.describe()

    def __repr__(self) -> str:
        if self.is_group_list:
            return (f"SettingsTree(group-list {self.key!r}, "
                    f"{len(self.tree.get('items', []))} elements)")
        if self.is_group:
            keys = [child.get("key") for child in self._child_nodes()]
            label = "root" if self.key is None else repr(self.key)
            return f"SettingsTree({label}, items={keys})"
        return f"SettingsTree(item {self.key!r}={self.value!r})"


class Configurable(ABC):
    """A ``SettingsTree`` configuration mixin (for pipelines and command decoders).

    A subclass declares its configuration **once** in :meth:`default_settings`. That
    method returns a :class:`SettingsTree` schema with the defaults and their constraints.
    Each instance keeps its own *live, editable* copy in :attr:`settings`. The Qt editor
    (``widgets.settings_tree``) edits this copy **in place**, with no round-trip through
    the class. :attr:`cfg` reads the current values back as a plain dict.

    You may pass the following to the constructor. All are optional and are applied on
    top of the defaults:

    * as the first argument, a full :class:`SettingsTree` (for example an edited or a
      saved one), or a plain ``dict`` of values;
    * ``**overrides`` keyword values, for a quick change in code.

    Examples
    --------
    >>> class MyConfig(Configurable):
    ...     @classmethod
    ...     def default_settings(cls):
    ...         s = SettingsTree()
    ...         s.add_item("filt_order", value=5, value_range=[1, None])
    ...         return s
    >>> obj = MyConfig(filt_order=7)                        # kwargs override the defaults
    >>> _ = obj.settings.set_value("filt_order", value=3)   # live edit (or via the GUI)
    >>> obj.cfg["filt_order"]
    3
    """

    @classmethod
    @abstractmethod
    def default_settings(cls) -> SettingsTree:
        """Return a fresh configuration schema (defaults + constraints) for this class."""

    def __init__(self, settings: "SettingsTree | dict | None" = None,
                 **overrides) -> None:
        self.settings = (settings if isinstance(settings, SettingsTree)
                         else self.default_settings())
        self.settings.snapshot_defaults()   # baseline any group-list defaults before editing
        if isinstance(settings, dict):
            self.settings.update_from_dict(settings)
        if overrides:
            self._reject_unknown(overrides)
            self.settings.update_from_dict(overrides)
        self._check_settings()

    def _reject_unknown(self, overrides: dict) -> None:
        valid = set(self.settings.to_dict())
        unknown = sorted(k for k in overrides if k not in valid)
        if unknown:
            raise TypeError(
                f"unknown setting(s) {unknown} for {type(self).__name__}; "
                f"valid keys: {sorted(valid)}.")

    def _check_settings(self) -> None:
        issues = self.settings.validate()
        if issues:
            raise ValueError(
                f"invalid settings for {type(self).__name__}: {issues}")

    @property
    def cfg(self) -> dict:
        """Current configuration values as a fresh dict from the live settings tree."""
        return self.settings.to_dict()
