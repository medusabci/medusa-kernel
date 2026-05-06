# Built-in imports
import json, bson
import warnings
from abc import ABC, abstractmethod
import copy

# External imports
import numpy as np
import scipy.io
import dill


class SerializableComponent(ABC):
    """Skeleton class for serializable components. These components must
    implement functions to transform the class to multiplatform formats,
    such as json, bson and mat. It must be used in classes that need persistence
    across multple platforms (i.e., recordings)
    """
    @abstractmethod
    def to_serializable_obj(self):
        """This function must return a serializable object (list or dict of
        primitive types) containing the relevant attributes of the class
        """
        raise NotImplemented

    @classmethod
    @abstractmethod
    def from_serializable_obj(cls, data):
        """This function must return an instance of the class from a
        serializable (list or dict of primitive types)"""
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
        """Saves the component to the specified format.

        Compatible formats:

        - bson: This format is safe, efficient, easy to use and multiplatform.
            Thus, it comes with  advantages in comparison to other formats.
            BSON format requires serializable classes to python primary types.
        - json: This format is safe, human readable and multiplatform, widely
            used for web applications. Nevertheless, files are encoded in utf-8
            and thus occupy more space. JSON format requires serializable
            classes to python primary types.
        - mat: This is a binary format widely used in research for its
            compatibility with Matlab. Very powerful and safe, but lacks of
            wide multiplatform compatibility. MAT format requires serializable
            classes, but allows numpy types.
        - pickle: This format is easy to use but lacks of multiplatform
            interoperability and it's not very efficient.

        Parameters
        ----------
        path: str
            File path. If data_format is None, The data format will be
            automatically decoded from the path extension.
        data_format: str
            Format to save the recording. Current supported formats:
        """
        # Decode format
        if data_format is None:
            df = path.split('.')[-1]
        else:
            df = data_format

        if df == 'pickle' or df == 'pkl':
            return self.save_to_pickle(path)
        elif df == 'bson':
            return self.save_to_bson(path)
        elif df == 'json':
            return self.save_to_json(path)
        elif df == 'mat':
            return self.save_to_mat(path)
        elif df == 'hdf5' or df == 'h5':
            raise NotImplemented
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
        """Save the class in a MATLAB .mat file using scipy

        Parameters
        ----------
        path: str
            Path to file
        avoid_none_objects: bool
            If True, it ensures that all None objects are removed from the
            object to save to avoid scipy.io.savemat error with this type.
            Nonetheless, it is computationally expensive, so it is better to
            leave to False and ensure manually.
        """
        ser_obj = self.to_serializable_obj()
        if avoid_none_objects:
            warnings.warn('Option avoid_none_objects may slow this process. '
                          'Consider removing None objects manually before '
                          'calling this function to save time')
            ser_obj = self.__none_to_null(ser_obj)
        scipy.io.savemat(path, mdict=ser_obj)

    def save_to_pickle(self, path, protocol=0):
        """Saves the class using dill into pickle format"""
        with open(path, 'wb') as f:
            dill.dump(self.to_serializable_obj(), f, protocol=protocol)

    @classmethod
    def load(cls, path, data_format=None):
        """Loads the file with the correct data structures

        Parameters
        ----------
        path : str
            File path
        data_format : None or str
            File format. If None, the format will be given by the file extension

        Returns
        -------
        Recording
            Recording class with the correct data structures
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
        elif df == 'hdf5' or df == 'h5':
            raise NotImplemented
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
        """Load a mat file using scipy and restore its original class

        Parameters
        ----------
        path: str
            Path to file
        restore_none_objects: bool
            If True, it ensures that all 'null' strings are restored as None
            objects in case that these objects were removed upon saving.
            Nonetheless, it is computationally expensive, so it is better to
            leave to False and ensure manually.
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
    """Skeleton class for pickleable components. These components must
    implement functions to transform the class to a pickleable object using
    dill package. It must be used in classes that need persistence but only make
    sense in Python and thus, they do not require multiplatform compatibility
    (i.e., signal processing methods)
    """
    @abstractmethod
    def to_pickleable_obj(self):
        """Returns a pickleable representation of the class. In most cases,
        the instance of the class is directly pickleable (e.g., all medusa
        methods, sklearn classifiers), but this may not be the case for some
        methods (i.e., keras models). Therefore, this function must be
        overridden in such cases.

        Returns
        -------
        representation: object
            Pickleable representation of the instance.name
        """
        raise NotImplemented

    @classmethod
    @abstractmethod
    def from_pickleable_obj(cls, pickleable_obj):
        """Returns the instance of the unpickled version of the pickleable
        representation given by function to_pickleable_representation.
        Therefore, this parameter is, by default, an instance of the class
        and no additional treatment is required. In some cases (i.e.,
        keras models), the pickleable_representation may not be the instance,
        but some other pickleable format with the required information of the
        method to reinstantiate the instance itself (i.e., weights for
        keras models). In such cases, this function must be overriden

        Parameters
        ----------
        pickleable_obj: object
            Pickleable representation of the processing method instance.

        Returns
        -------
        instance: PickleableComponent
            Instance of the component
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
