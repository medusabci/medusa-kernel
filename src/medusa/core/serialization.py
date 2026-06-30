# Built-in imports
import json, bson
import warnings
import importlib
from abc import ABC, abstractmethod
import copy

# External imports
import numpy as np
import scipy.io
import dill

__all__ = [
    "SerializableComponent",
    "PickleableComponent",
    "ensure_list",
    "tag_component",
    "load_component",
]


class SerializableComponent(ABC):
    """Base class for components persisted to multiplatform formats.

    Subclasses implement :meth:`to_serializable_obj` / :meth:`from_serializable_obj`
    (a dict/list of primitive types) and inherit :meth:`save` / :meth:`load` for the
    ``bson``/``json``/``mat``/``pickle`` formats. Used by types that must round-trip
    across platforms, e.g. recordings.
    """
    @abstractmethod
    def to_serializable_obj(self):
        """Return a serializable object (dict/list of primitive types) holding the
        class's relevant attributes."""
        raise NotImplemented

    @classmethod
    @abstractmethod
    def from_serializable_obj(cls, data):
        """Rebuild an instance from the object produced by
        :meth:`to_serializable_obj`."""
        raise NotImplemented

    @staticmethod
    def __none_to_null(obj):
        """This function iterates over the attributes of the an object and
        converts all None objects to 'null' to avoid problems with
        scipy.io.savemat"""
        if isinstance(obj, dict):
            for k, v in obj.items():
                if hasattr(v, '__dict__'):
                    v = SerializableComponent.__none_to_null(v.__dict__)
                elif isinstance(v, dict) or isinstance(v, list):
                    v = SerializableComponent.__none_to_null(v)
                if v is None:
                    obj[k] = 'null'
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                if hasattr(v, '__dict__'):
                    v = SerializableComponent.__none_to_null(v.__dict__)
                elif isinstance(v, dict) or isinstance(v, list):
                    v = SerializableComponent.__none_to_null(v)
                if v is None:
                    obj[i] = 'null'
        return obj

    @staticmethod
    def _str_lists_to_cells(obj):
        """Convert all-string lists to object arrays for ``scipy.io.savemat``.

        ``savemat`` writes a Python ``list[str]`` as a 2-D char *matrix*, which
        right-pads every shorter string with spaces to the longest one (so
        ``["go", "nogo"]`` round-trips as ``["go  ", "nogo"]``). Storing the list as
        an object array makes ``savemat`` emit a MATLAB *cell* array instead, which
        preserves each string's exact length. Recurses through dicts/lists; all
        other values (incl. numeric lists) are untouched, and the result
        round-trips through ``_sanitize_and_reconstruct`` on load.
        """
        if isinstance(obj, dict):
            return {k: SerializableComponent._str_lists_to_cells(v)
                    for k, v in obj.items()}
        if isinstance(obj, list):
            items = [SerializableComponent._str_lists_to_cells(v) for v in obj]
            if items and all(isinstance(v, str) for v in items):
                return np.array(items, dtype=object)
            return items
        return obj

    @staticmethod
    def __null_to_none(obj):
        """This function iterates over the attributes of the an object and
        converts all 'null' objects to None to restore the Python original
        representation"""
        if isinstance(obj, dict):
            for k, v in obj.items():
                if hasattr(v, '__dict__'):
                    v = SerializableComponent.__null_to_none(v.__dict__)
                elif isinstance(v, dict) or isinstance(v, list):
                    v = SerializableComponent.__null_to_none(v)
                try:
                    if v == 'null':
                        obj[k] = None
                except ValueError as e:
                    # Some class do not admit comparison with strings (ndarrays)
                    pass
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                if hasattr(v, '__dict__'):
                    v = SerializableComponent.__null_to_none(v.__dict__)
                elif isinstance(v, dict) or isinstance(v, list):
                    v = SerializableComponent.__null_to_none(v)
                try:
                    if v == 'null':
                        obj[i] = None
                except ValueError as e:
                    # Some class do not admit comparison with strings (ndarrays)
                    pass
        return obj

    def save(self, path, data_format=None):
        """Save the component; the format is ``data_format`` or the path extension.

        Formats: ``bson`` (efficient binary, multiplatform), ``json``
        (human-readable, multiplatform, larger files), ``mat`` (MATLAB binary,
        allows numpy types, limited multiplatform support), ``pickle`` (easy but
        Python-only).

        Parameters
        ----------
        path
            File path; the format is taken from its extension when ``data_format``
            is ``None``.
        data_format
            One of ``bson``/``json``/``mat``/``pickle`` (overrides the extension).
        """
        # Decode format
        if data_format is None:
            df = path.split('.')[-1]
        else:
            df = data_format

        if df == 'pickle' or df == 'pkl':
            self.save_to_pickle(path)
        elif df == 'bson':
            self.save_to_bson(path)
        elif df == 'json':
            self.save_to_json(path)
        elif df == 'mat':
            self.save_to_mat(path)
        else:
            raise ValueError('Format %s is not available yet' % df)

    def save_to_bson(self, path):
        """Saves the class attributes in BSON format"""
        with open(path, 'wb') as f:
            f.write(bson.dumps(self.to_serializable_obj()))

    def save_to_json(self, path, encoding='utf-8', indent=4):
        """Saves the class attributes in JSON format"""
        with open(path, 'w', encoding=encoding) as f:
            json.dump(self.to_serializable_obj(), f, indent=indent)

    def save_to_mat(self, path, avoid_none_objects=True):
        """Save to a MATLAB ``.mat`` file via scipy.

        Parameters
        ----------
        path
            File path.
        avoid_none_objects
            Replace ``None`` with ``'null'`` before saving (``scipy.io.savemat``
            cannot store ``None``); restored on load. Slower -- set ``False`` if
            ``None`` values are already removed.
        """
        ser_obj = self.to_serializable_obj()
        if avoid_none_objects:
            warnings.warn('Option avoid_none_objects may slow this process. '
                          'Consider removing None objects manually before '
                          'calling this function to save time')
            ser_obj = self.__none_to_null(ser_obj)
        # Store variable-length string lists as cell arrays, not padded char
        # matrices (otherwise scipy right-pads shorter strings with spaces).
        ser_obj = self._str_lists_to_cells(ser_obj)
        scipy.io.savemat(path, mdict=ser_obj)

    def save_to_pickle(self, path, protocol=0):
        """Saves the class using dill into pickle format"""
        with open(path, 'wb') as f:
            dill.dump(self.to_serializable_obj(), f, protocol=protocol)

    # No generic save_to_hdf5/load_from_hdf5 here: unlike the formats above (built
    # on to_serializable_obj, which list-ifies arrays), a useful HDF5 store needs
    # per-component knowledge of which fields are large arrays. Components with big
    # array payloads add it themselves (e.g. Recording, via core.data.streaming).

    @classmethod
    def load(cls, path, data_format=None):
        """Load a component; the format is ``data_format`` or the path extension.

        Parameters
        ----------
        path
            File path.
        data_format
            One of ``bson``/``json``/``mat``/``pickle`` (overrides the extension).

        Returns
        -------
        SerializableComponent
            The reconstructed instance.
        """
        # Check extension
        if data_format is None:
            df = path.split('.')[-1]
        else:
            df = data_format
        # Load file
        if df == 'pickle' or df == 'pkl':
            return cls.load_from_bson(path)
        elif df == 'bson':
            return cls.load_from_bson(path)
        elif df == 'json':
            return cls.load_from_json(path)
        elif df == 'mat':
            return cls.load_from_mat(path)
        else:
            raise TypeError('Unknown file format %s' % df)

    @classmethod
    def load_from_bson(cls, path):
        with open(path, 'rb') as f:
            ser_obj_dict = bson.loads(f.read())
        return cls.from_serializable_obj(ser_obj_dict)

    @classmethod
    def load_from_json(cls, path, encoding='utf-8'):
        with open(path, 'r', encoding=encoding) as f:
            ser_obj_dict = json.load(f)
        return cls.from_serializable_obj(ser_obj_dict)

    @classmethod
    def load_from_mat(cls, path, squeeze_me=True, simplify_cells=True,
                      restore_none_objects=True):
        """Load a ``.mat`` file via scipy and rebuild the component.

        Parameters
        ----------
        path
            File path.
        squeeze_me, simplify_cells
            Passed to :func:`scipy.io.loadmat` (collapse size-1 axes; simplify
            cell/struct arrays).
        restore_none_objects
            Restore ``'null'`` strings back to ``None`` (inverse of
            ``save_to_mat``). Slower -- set ``False`` if not needed.
        """
        ser_obj_dict = scipy.io.loadmat(path, squeeze_me=squeeze_me,
                                        simplify_cells=simplify_cells)
        if restore_none_objects:
            warnings.warn('Option restore_none_objects may slow this process. '
                          'Consider removing "null" strings manually and '
                          'substitute them for None objects before calling '
                          'this function to save time')
            ser_obj_dict = cls.__none_to_null(ser_obj_dict)

        def _sanitize_and_reconstruct(data):
            if isinstance(data, dict):
                # Clean MATLAB metadata (e.g., __header__, __version__, __globals__)
                clean_dict = {k: _sanitize_and_reconstruct(v)
                              for k, v in data.items() if not k.startswith('__')}
                return clean_dict

            elif isinstance(data, np.ndarray):
                # Convert problematic arrays to native lists

                # If it's an array of objects (typical when loading MATLAB structs)
                if data.dtype == 'O':
                    return [_sanitize_and_reconstruct(i) for i in data.tolist()]

                return data.tolist()

            elif isinstance(data, list):
                return [_sanitize_and_reconstruct(i) for i in data]

            return data

        # Apply the sanitizer to the loaded data
        sane_dict = _sanitize_and_reconstruct(cls.__null_to_none(ser_obj_dict))
        return cls.from_serializable_obj(cls.__null_to_none(sane_dict))

    @classmethod
    def load_from_pickle(cls, path):
        with open(path, 'rb') as f:
            cmp = dill.load(f)
        return cmp


class PickleableComponent(ABC):
    """Base class for components persisted with ``dill`` (Python-only).

    For types that need persistence but not multiplatform interoperability (e.g.
    signal-processing methods). Subclasses implement :meth:`to_pickleable_obj` /
    :meth:`from_pickleable_obj`.
    """
    @abstractmethod
    def to_pickleable_obj(self):
        """Return a pickleable representation of the instance.

        Usually the instance itself is pickleable (medusa methods, sklearn
        classifiers); override when it is not (e.g. keras models).

        Returns
        -------
        object
            Pickleable representation of the instance.
        """
        raise NotImplemented

    @classmethod
    @abstractmethod
    def from_pickleable_obj(cls, pickleable_obj):
        """Rebuild an instance from the object returned by :meth:`to_pickleable_obj`.

        By default that object *is* the instance; override when
        :meth:`to_pickleable_obj` returns something else (e.g. keras weights).

        Parameters
        ----------
        pickleable_obj
            The representation produced by :meth:`to_pickleable_obj`.

        Returns
        -------
        PickleableComponent
            The reconstructed instance.
        """
        raise NotImplemented

    def save(self, path, protocol=0):
        """Saves the class using dill into pickle format"""
        with open(path, 'wb') as f:
            dill.dump(self.to_pickleable_obj(), f, protocol=protocol)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            pickleable_obj = dill.load(f)
        return cls.from_pickleable_obj(pickleable_obj)


def ensure_list(value):
    """Coerce ``value`` to a ``list`` (wrapping a non-list in a one-element list).

    Defensive helper for ``from_serializable_obj``. ``scipy.io.loadmat`` with
    ``squeeze_me=True`` collapses a size-1 array to a scalar, so a one-element
    serialized list (e.g. a recording with a single event, channel or sensor) comes
    back from ``.mat`` as a bare value instead of a list. Components whose schema
    expects a list call this so the ``.mat`` round-trip matches bson/json/hdf5.
    Idempotent: an existing list is returned unchanged.
    """
    return value if isinstance(value, list) else [value]


def tag_component(obj):
    """Serialise a :class:`SerializableComponent` with a class-dispatch tag.

    Returns ``{module_name, class_name, component_data}`` where ``component_data``
    is the component's own ``to_serializable_obj()``. The tag lets a heterogeneous
    container (e.g. ``Recording.data``, ``Recording.experiment``) reconstruct the
    *concrete* class on load without knowing it in advance -- see
    :func:`load_component`.
    """
    return {
        "module_name": type(obj).__module__,
        "class_name": type(obj).__name__,
        "component_data": obj.to_serializable_obj(),
    }


def load_component(tagged):
    """Rebuild a component from a :func:`tag_component` dict.

    Imports ``module_name``, resolves ``class_name``, and delegates to its
    ``from_serializable_obj``.

    Raises
    ------
    ImportError
        If the recorded module/class is not importable. The class must be on the
        import path (installed or already imported) for the load to succeed.
    """
    try:
        module = importlib.import_module(tagged["module_name"])
        cls = getattr(module, tagged["class_name"])
    except (ImportError, AttributeError) as exc:
        raise ImportError(
            f"cannot resolve serialized class {tagged.get('class_name')!r} from "
            f"module {tagged.get('module_name')!r}; is it importable here?"
        ) from exc
    return cls.from_serializable_obj(tagged["component_data"])
