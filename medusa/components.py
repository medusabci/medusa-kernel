# Built-in imports
import json, bson
import warnings
from abc import ABC, abstractmethod
import sys, inspect
import copy, collections
from threading import Thread

# External imports
import numpy as np
import scipy.io
import dill

# Medusa imports
from medusa.performance_analysis import perf_analysis


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
        return cls.from_serializable_obj(cls.__null_to_none(ser_obj_dict))

    @classmethod
    def load_from_pickle(cls, path):
        with open(path, 'rb') as f:
            cmp = dill.load(f)
        return cmp


class SettingsTreeItem(SerializableComponent):
    """General class to represent settings.
    """
    def __init__(self, key, info, value_type=None, value=None):
        """Class constructor.

        Parameters
        ----------
        key: str
            Tree item key
        info: str
            Information about this item
        value_type: str ['string'|'number'|'integer'|'boolean'|'dict'|'list'], optional
            Type of the data stored in attribute value. Leave to None if the
            item is going to be a tree.
        value: str, int, float, bool, dict or list, optional
            Tree item value. It must be one of the JSON types to be compatible
            with serialization. Leave to None if the item is going to be a tree.
        """
        # Init attributes
        self.key = key
        self.info = info
        self.value_type = None
        self.value = None
        self.items = list()
        # Set data
        if value_type is not None:
            self.set_data(value_type, value)

    def set_data(self, value_type, value):
        """Adds tree item to the tree. Use this function to build a custom tree.

        Parameters
        ----------
        value_type: str or list ['string'|'number'|'boolean'|'dict'|'list']
            Type of the data stored in attribute value. If a list is provided,
            several data types are accepted for attribute value.
        value: str, int, float, bool, dict or list
            Tree item value. It must be one of the JSON types to be compatible
            with serialization. If list or dict, the items must be of type
            SettingsTreeItem.
        """
        # Check errors
        orig_value_type = value_type
        value_type = [value_type] if not isinstance(value_type, list) \
            else value_type
        for t in value_type:
            if t == 'string':
                if value is not None:
                    assert isinstance(value, str), \
                        'Parameter value must be of type %s' % str
            elif t == 'number':
                if value is not None:
                    assert isinstance(value, int) or isinstance(value, float), \
                        'Parameter value must be of types %s or %s' % \
                        (int, float)
            elif t == 'integer':
                if value is not None:
                    assert isinstance(value, int), \
                        'Parameter value must be of types %s or %s' % \
                        (int, float)
            elif t == 'boolean':
                if value is not None:
                    assert isinstance(value, bool), \
                        'Parameter value must be of type %s' % bool
            elif t == 'list':
                if value is not None:
                    assert isinstance(value, list), \
                        'Parameter value must be of type %s' % list
                    for v in value:
                        assert isinstance(v, SettingsTreeItem), \
                            'All items must be of type %s' % SettingsTreeItem
                        assert not v.is_tree(), 'Items cannot be trees. Use ' \
                                                'add item instead!'
            elif t == 'dict':
                if value is not None:
                    assert isinstance(value, dict), \
                        'Parameter value must be of type %s' % dict
                    for v in value.values():
                        assert isinstance(v, SettingsTreeItem), \
                            'All items must be of type %s' % SettingsTreeItem
                        assert not v.is_tree(), 'Items cannot be trees. Use ' \
                                                'add item instead!'
            else:
                raise ValueError('Unknown value_type. Read the docs!')
        # Set data
        self.value_type = orig_value_type
        self.value = value
        self.items = list()

    def add_item(self, item):
        """Adds tree item to the tree. Use this function to build a custom tree.
        Take into account that if this function is used, attributes value and
        type will be set to None.

        Parameters
        ----------
        item: SettingsTreeItem
            Tree item to add
        """
        if not isinstance(item, SettingsTreeItem):
            raise ValueError('Parameter item must be of type %s' %
                             type(SettingsTreeItem))
        self.items.append(item)
        self.value_type = None
        self.value = None

    def count_items(self):
        return len(self.items)

    def is_tree(self):
        return len(self.items) > 0

    def to_serializable_obj(self):
        # Get serialized value
        if self.value_type == 'dict':
            value = dict()
            for k, v in self.value.items():
                value[k] = v.to_serializable_obj()
        elif self.value_type == 'list':
            value = list()
            for v in self.value:
                value.append(v.to_serializable_obj())
        else:
            value = self.value
        # Serialize
        data = {
            'key': self.key,
            'value': value,
            'value_type': self.value_type,
            'info': self.info,
            'items': [item.to_serializable_obj() for item in self.items]
        }
        return data

    @classmethod
    def from_serializable_obj(cls, data):
        # Get desserialized value
        if data['value_type'] == 'dict':
            value = dict()
            for k, v in data['value'].items():
                value[k] = SettingsTreeItem.from_serializable_obj(v)
        elif data['value_type'] == 'list':
            value = list()
            for v in data['value']:
                value.append(SettingsTreeItem.from_serializable_obj(v))
        else:
            value = data['value']
        # Create item
        tree_item = cls(data['key'], data['info'],
                        data['value_type'], value)
        for serialized_item in data['items']:
            tree_item.add_item(SettingsTreeItem.from_serializable_obj(
                serialized_item))
        return tree_item


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


class Recording(SerializableComponent):
    """
    Class intended to save the data from one recording. It implements all
    necessary methods to save and load from several formats. It accepts
    different kinds of data: experiment data, which saves all the information
    about the experiment (e.g., events); biosignal data (e.g., EEG, MEG, NIRS),
    bioimaging data (e.g., fMRI, MRI); and custom data (e.g., photos, videos,
    audio). Temporal data must be must be synchronized with the reference.
    To assure multiplatform interoperability, this class must be serializable
    using python primitive types.
    """
    def __init__(self, subject_id, recording_id=None, description=None,
                 source=None, date=None, **kwargs):
        """Recording dataset constructor. Custom useful parameters can be
        provided to save in the class.

        Parameters
        ----------
        subject_id : int or str
            Subject identifier
        recording_id : str or None
            Identifier of the recording for automated processing or easy
            identification
        description : str or None
            Description of this recording. Useful to write comments (e.g., the
            subject moved a lot, the experiment was interrupted, etc)
        source : str or None
            Source of the data, such as software, equipment, experiment, etc
        kwargs : custom key-value parameters
            Other useful parameters (e.g., software version, research team,
            laboratory, etc)
        """

        # Standard attributes
        self.subject_id = subject_id
        self.recording_id = recording_id
        self.description = description
        self.source = source
        self.date = date

        # Data variables
        self.experiments = dict()
        self.biosignals = dict()
        self.bioimaging = dict()
        self.custom_data = dict()

        # Set the specified arguments
        for key, value in kwargs.items():
            setattr(self, key, value)

    def add_experiment_data(self, experiment_data, key=None):
        """Adds the experiment data of this recording. Each experiment should
        have a predefined class that must be instantiated before. Several
        classes are defined within medusa core, but it also can be a custom
        experiment.

        Parameters
        ----------
        experiment_data : experiment class
            Instance of an experiment class. This class can be custom if it is
            serializable, but it is recommended to use the classes provided
            by medusa core in different modules (e.g., bci.erp_paradigms.rcp)
        key: str
            Custom name for this experiment. If not provided, the experiment
            will be saved in an attribute according to its type (e.g., rcp,
            cake paradigm, etc). This parameter is useful if several experiments
            of the same type are added to this recording
        """
        # Check errors
        if not issubclass(type(experiment_data), ExperimentData):
            raise TypeError('Parameter experiment_data must subclass '
                            'medusa.io.ExperimentData')
        # Check type
        experiment_module_name = type(experiment_data).__module__
        experiment_class_name = type(experiment_data).__name__
        att = experiment_class_name.lower() if key is None else key
        if isinstance(experiment_data, CustomExperimentData):
            warnings.warn('Unspecific experiment data %s. Some high-level '
                          'functions may not work' % type(experiment_data))
        # Check key
        if hasattr(self, att):
            raise ValueError('This recording already has an attribute with key '
                             '%s' % att)
        # Add experiment
        setattr(self, att, experiment_data)
        self.experiments[att] = {
            'module_name': experiment_module_name,
            'class_name': experiment_class_name
        }

    def add_biosignal(self, biosignal, key=None):
        """Adds a biosignal recording. Each biosignal has predefined classes
        that must be instantiated before (e.g., EEG, MEG)

        Parameters
        ----------
        biosignal : biosignal class
            Instance of the biosignal class. This class must be serializable.
            Current available: EEG, MEG.
        key: str
            Custom name for this biosignal. If not provided, the biosignal will
            be saved in an attribute according to its type in lowercase
            (e.g., eeg, meg, etc). This parameter is  useful if several
            biosignals of the same type are added to this recording
        """
        # Check errors
        if not issubclass(type(biosignal), BiosignalData):
            raise TypeError('Parameter biosignal must subclass '
                            'medusa.io.BiosignalData')
        # Check type
        biosignal_module_name = type(biosignal).__module__
        biosignal_class_name = type(biosignal).__name__
        att = biosignal_class_name.lower() if key is None else key
        if isinstance(biosignal, CustomBiosignalData):
            warnings.warn('Unspecific biosignal %s. Some high-level functions '
                          'may not work' % type(biosignal))
        # Check key
        if hasattr(self, att):
            raise ValueError('This recording already contains an attribute '
                             'with key %s' % att)
        # Add biosignal
        setattr(self, att, biosignal)
        self.biosignals[att] = {
            'module_name': biosignal_module_name,
            'class_name': biosignal_class_name
        }

    def add_bioimaging(self, bioimaging, key=None):
        # TODO: Create BioimagingData class
        raise NotImplemented

    def add_custom_data(self, data, key=None):
        # TODO: Create CustomData class
        raise NotImplemented

    def cast_biosignal(self, key, biosignal_class):
        """This function casts a biosignal to the class passed in
        biosignal_class
        """
        biosignal_module_name = biosignal_class.__module__
        biosignal_class_name = biosignal_class.__name__
        # Check errors
        if not issubclass(biosignal_class, BiosignalData):
            raise TypeError('Class %s must subclass medusa.io.Biosignal' %
                            biosignal_class_name)
        biosignal = getattr(self, key)
        biosignal_dict = biosignal.to_serializable_obj()
        setattr(self, key, biosignal_class.from_serializable_obj(biosignal_dict))
        self.biosignals[key] = {
            'module_name': biosignal_module_name,
            'class_name': biosignal_class_name
        }

    def cast_experiment(self, key, experiment_class):
        """This function casts an experiment of recording run to the class
        passed in experiment_class
        """
        exp_module_name = experiment_class.__module__
        exp_class_name = experiment_class.__name__
        # Check errors
        if not issubclass(experiment_class, ExperimentData):
            raise TypeError('Class %s must subclass medusa.io.ExperimentData' %
                            exp_class_name)
        experiment_data = getattr(self, key)
        experiment_data_dict = experiment_data.to_serializable_obj()
        setattr(self, key, experiment_class.from_serializable_obj(experiment_data_dict))
        self.experiments[key] = {
            'module_name': exp_module_name,
            'class_name': exp_class_name
        }

    def rename_attribute(self, old_key, new_key):
        """Rename an attribute. Useful to unify attribute names on fly while
        creating a dataset.

        Parameters
        ----------
        old_key : str
            Old attribute key
        new_key : str
            New attribute key
        """
        self.__dict__[new_key] = self.__dict__.pop(old_key)

    def get_biosignals_with_class_name(self, biosignal_class_name):
        """This function returns the biosignals with a specific class name

        Parameters
        ----------
        biosignal_class_name: str
            Class name of the biosignal (e.g., "EEG")
        """
        biosignals = dict()
        for key, value in self.biosignals.items():
            if value['class_name'] == biosignal_class_name:
                biosignals[key] = getattr(self, key)
        return biosignals

    def get_experiments_with_class_name(self, exp_class_name):
        """This function returns the experiments with a specific class name

        Parameters
        ----------
        exp_class_name: str
            Class name of the experiment (e.g., "ERPSpellerData")
        """
        experiments = dict()
        for key, value in self.experiments.items():
            if value['class_name'] == exp_class_name:
                experiments[key] = getattr(self, key)
        return experiments

    def to_serializable_obj(self):
        """This function returns a serializable dict (primitive types)
        containing the attributes of the class
        """
        rec_dict = self.__dict__
        # Process biosginals
        for key in self.biosignals:
            biosignal = getattr(self, key)
            rec_dict[key] = biosignal.to_serializable_obj()
        # Process experiments
        for key in self.experiments:
            experiments = getattr(self, key)
            rec_dict[key] = experiments.to_serializable_obj()
        return rec_dict

    @classmethod
    def from_serializable_obj(cls, rec_dict):
        """Function that loads the class from a python dictionary
        """
        # Handle biosignals
        if 'biosignals' in rec_dict:
            for biosignal_key, biosignal_dict in rec_dict['biosignals'].\
                    items():
                try:
                    module = sys.modules[biosignal_dict['module_name']]
                    obj = getattr(module, biosignal_dict['class_name'])
                except KeyError:
                    raise ImportError('Biosignal class %s not found in module '
                                      '%s. This class must be reachable in '
                                      ' this module or defined in the main '
                                      'program. Did you import the module %s'
                                      ' before using this function?'
                                      % (biosignal_dict['class_name'],
                                         biosignal_dict['module_name'],
                                         biosignal_dict['module_name']))
                rec_dict[biosignal_key] = \
                    obj.from_serializable_obj(rec_dict[biosignal_key])
        # Handle experiments
        if 'experiments' in rec_dict:
            for exp_key, exp_dict in rec_dict['experiments'].items():
                try:
                    module = sys.modules[exp_dict['module_name']]
                    obj = getattr(module, exp_dict['class_name'])
                except KeyError:
                    raise ImportError('Experiment class %s not found in module '
                                      '%s. This class must be reachable in '
                                      'this module or defined in the main '
                                      'program. Did you import the module %s '
                                      'before using this function?'
                                      % (exp_dict['class_name'],
                                         exp_dict['module_name'],
                                         exp_dict['module_name']))
                rec_dict[exp_key] = obj.from_serializable_obj(rec_dict[exp_key])
        # Instantiate class
        return cls(**rec_dict)


class BiosignalData(SerializableComponent):
    """Skeleton class for biosignals
    """

    @abstractmethod
    def to_serializable_obj(self):
        """This function must return a serializable dict (primitive types)
        containing the relevant attributes of the class
        """
        pass

    @classmethod
    @abstractmethod
    def from_serializable_obj(cls, dict_data):
        """This function must return an instance of the class from a
        serializable dict (primitive types)"""
        pass


class CustomBiosignalData(BiosignalData):
    """Custom biosignal data class. This class does not check the arguments and
    provides less functionality that more specific classes. It should
    only be used for custom signals that do not fit in other data classes
    """

    def __init__(self, **kwargs):
        """CustomBiosginal constructor

        Parameters
        ----------
        kwargs: kwargs
            Key-value arguments to be saved in the class. This general class
            does not check anything
        """
        # Set the specified arguments
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_serializable_obj(self):
        rec_dict = self.__dict__
        for key in rec_dict.keys():
            if type(rec_dict[key]) == np.ndarray:
                rec_dict[key] = rec_dict[key].tolist()
        return rec_dict

    @classmethod
    def from_serializable_obj(cls, dict_data):
        return cls(**dict_data)


class ExperimentData(SerializableComponent):

    """Skeleton class for experiment data
    """

    @abstractmethod
    def to_serializable_obj(self):
        """This function must return a serializable dict (primitive types)
        containing the relevant attributes of the class
        """
        pass

    @classmethod
    @abstractmethod
    def from_serializable_obj(cls, dict_data):
        """This function must return an instance of the class from a
        serializable dict (primitive types)
        """
        pass


class CustomExperimentData(ExperimentData):
    """Custom experiment data class. This class does not check the arguments and
    provides less functionality that a proper experiment class. It should
    only be used for custom experiments that do not fit in other experiment
    data classes
    """
    def __init__(self, **kwargs):
        """CustomExperimentData constructor

        Parameters
        ----------
        kwargs: kwargs
            Key-value arguments to be saved in the class. This general class
            does not check anything
        """
        # Set the specified arguments
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_serializable_obj(self):
        rec_dict = self.__dict__
        for key in rec_dict.keys():
            if type(rec_dict[key]) == np.ndarray:
                rec_dict[key] = rec_dict[key].tolist()
        return rec_dict

    @classmethod
    def from_serializable_obj(cls, dict_data):
        return cls(**dict_data)


class ConsistencyChecker(SerializableComponent):
    """Class that provides functionality to check consistency across recordings
    to build a dataset
    """

    def __init__(self):
        self.__rules = list()

    def add_consistency_rule(self, rule, rule_params, parent=None):
        """Adds a consistency check for the specified attribute It provides 2
        levels of consistency using parameter key, enough to check attributes
        inside biosignal or experiments classes

        Parameters
        ----------
        rule : str {'check-attribute-type'|'check-attribute-value'|
            'check-values-in-attribute'|'check-if-attribute-exists'|
            'check-if-type-exists'}
                Check mode of this attribute. Modes:
                    - check-attribute-type: checks if the attribute has the type
                        specified in parameter check_value.
                    - check-attribute-value: checks if the attribute has the
                        value specified in parameter check_value
                    - check-values-in-attribute: checks if the attribute
                        contains the values (the attribute must support in
                        operation). It can check keys in dicts or values in
                        lists or sets.
                    - check-attribute: checks if the attribute exists
                    - check-type: checks if the class contains attributes with
                        the specified type. Use operator to define establish
                        rules about the number of attributes allowed with the
                        specified type
        rule_params : dict
            Specifies the rule params. Depending on the rule, it must contain
            the following key-value pairs:
                - check-attribute-type: {attribute: str, type: class or list}.
                    If type is list, it will be checked that the attribute is of
                    one of the types defined in the list
                - check-attribute-value: {attribute: str, value: obj}
                - check-values-in-attribute: {attribute: str, values: list}
                - check-attribute: {attribute: str}
                - check-type: {type: class, limit: int, operator: str {'<'|'>'|'
                    <='|'>='|'=='|'!='}
        parent : str or None
            Checks the rule inside specified parent. If None, the parent is the
            recording itself. Therefore, the parent must be a class. This
            parameter designed to allow check rules inside biosignals or
            experiments class. If the parent is in deeper levels, use points to
            define the parent. For example, you can check the labels of the
            channels in an EEG recording setting this parameter as
            eeg.channel_set
        """
        # Check to avoid errors
        if rule == 'check-attribute-type':
            if not all(k in rule_params for k in ['attribute', 'type']):
                raise ValueError('Rule params must contain keys (attribute, '
                                 'type) for rule %s' % rule)
        elif rule == 'check-attribute-value':
            if not all(k in rule_params for k in ['attribute', 'value']):
                raise ValueError('Rule params must contain keys (attribute, '
                                 'value) for rule %s' % rule)
        elif rule == 'check-values-in-attribute':
            if not all(k in rule_params for k in ['attribute', 'values']):
                raise ValueError('Rule params must contain keys (attribute, '
                                 'values) for rule %s' % rule)
        elif rule == 'check-attribute':
            if not all(k in rule_params for k in ['attribute']):
                raise ValueError('Rule params must contain keys (attribute) '
                                 'for rule %s' % rule)
        elif rule == 'check-type':
            if not all(k in rule_params for k in ['type', 'limit', 'operator']):
                raise ValueError('Rule params must contain keys (type) for '
                                 'rule %s' % rule)
            if rule_params['operator'] not in {'<', '>', '<=', '>=', '==', '!='}:
                raise ValueError("Unknown operator %s. Possible operators: "
                                 "{'<'|'>'|'<='|'>='|'=='|'!='}" %
                                 rule_params['operator'])
        else:
            raise ValueError("Unknown rule. Possible rules: "
                             "{'check-attribute-type'|'check-attribute-value'|"
                             "'check-values-in-attribute'|"
                             "'check-if-attribute-exists'|"
                             "'check-if-type-exists'}")
        # Save rule
        self.__rules.append({'rule': rule, 'rule_params': rule_params,
                             'parent': parent})

    def check_consistency(self, recording):
        """Checks the consistency of a recording according to the current rules

        Parameters
        ----------
        recording : Recording
            Recording to be checked
        """
        # Check general attributes
        for r in self.__rules:
            rule = r['rule']
            rule_params = r['rule_params']
            if r['parent'] is None:
                parent = recording
            else:
                parent = recording
                for p in r['parent'].split('.'):
                    parent = getattr(parent, p)
            if rule == 'check-attribute-type':
                attribute = getattr(parent, rule_params['attribute'])
                if type(rule_params['type']) == list:
                    check = False
                    for t in rule_params['type']:
                        if isinstance(attribute, t):
                            check = True
                    if not check:
                        raise TypeError('Type of attribute %s must be one '
                                        'of %s' % (rule_params['attribute'],
                                         str(rule_params['type'])))
                else:
                    if not isinstance(attribute, rule_params['type']):
                        raise TypeError('Type of attribute %s must be %s' %
                                        (rule_params['attribute'],
                                         str(rule_params['type'])))
            elif rule == 'check-attribute-value':
                attribute = getattr(parent, rule_params['attribute'])
                if attribute != rule_params['value']:
                    raise ValueError('Value of attribute %s must be %s' %
                                     (rule_params['attribute'],
                                      str(rule_params['value'])))
            elif rule == 'check-values-in-attribute':
                attribute = getattr(parent, rule_params['attribute'])
                for val in rule_params['values']:
                    if val not in attribute:
                        raise ValueError('Parameter %s must contain value %s' %
                                         (rule_params['attribute'],
                                          str(rule_params['values'])))
            elif rule == 'check-attribute':
                if not hasattr(parent, rule_params['attribute']):
                    raise ValueError('Attribute %s does not exist' %
                                     rule_params['attribute'])
            elif rule == 'check-type':
                # Get number of attributes with type
                n = 0
                for key, val in parent.__dict__.items():
                    if isinstance(val, rule_params['type']):
                        n += 1
                # Check
                if not self.__numeric_check(n, rule_params['limit'],
                                            rule_params['operator']):
                    raise ValueError('Number of attributes with type %s does '
                                     'not meet the rule (%i %s %i)' %
                                     (rule_params['type'], n,
                                      rule_params['operator'],
                                      rule_params['limit']))

    @staticmethod
    def __numeric_check(number, limit, operator):
        result = True
        if operator == '<':
            if number >= limit:
                result = False
        elif operator == '>':
            if number <= limit:
                result = False
        elif operator == '<=':
            if number > limit:
                result = False
        elif operator == '>=':
            if number < limit:
                result = False
        elif operator == '==':
            if number != limit:
                result = False
        elif operator == '!=':
            if number == limit:
                result = False
        else:
            raise ValueError("Unknown operator %s. Possible operators: "
                             "{'<'|'>'|'<='|'>='|'=='|'!='}" % operator)

        return result

    def to_serializable_obj(self):
        return self.__dict__

    @classmethod
    def from_serializable_obj(cls, dict_data):
        inst = cls()
        inst.__dict__.update(dict_data)
        return inst


class Dataset(ABC):
    """Class to handle multiple recordings maintaining consistency"""

    def __init__(self, consistency_checker=None):
        """Class constructor

        Parameters
        ----------
        consistency_checker : ConsistencyChecker
            Consistency checker for this dataset.
        """
        self.consistency_checker = consistency_checker
        self.recordings = list()

    def add_recordings(self, recordings):
        """Adds one or more recordings to the dataset, checking the consistency

        Parameters
        ----------
        recordings : list or medusa.data_structures.Recording
            List containing the paths to recording files or instances of
            Recording class
        """
        # Avoid errors
        recordings = [recordings] if type(recordings) != list else recordings
        if len(recordings) == 0:
            raise ValueError('Parameter recordings is empty!')
        # Add recordings
        for r in recordings:
            # Check if recording is instance of Recording of path
            if type(r) == str:
                recording = Recording.load(r)
            elif type(r) == Recording:
                recording = r
            else:
                raise TypeError('Error at index %i: type has to be %s or %s' %
                                (recordings.index(r),
                                 type(str),
                                 type(Recording)))
            # Check consistency
            if self.consistency_checker is not None:
                self.consistency_checker.check_consistency(recording)
            # Append recording
            self.recordings.append(
                self.custom_operations_on_recordings(recording)
            )

    def custom_operations_on_recordings(self, recording):
        """Function add_recordings calls this function before adding each
        recording to the dataset. Implement this method in custom classes to
        have personalized behaviour (e.g., change the channel set)

        Parameters
        ----------
        recording : subclass of Recording
            Recording that will be changed. It can also be a subclass of
            Recording

        Returns
        -------
        recording : Recording
            Modified recording
        """
        return recording


class ProcessingMethod(PickleableComponent):
    """Skeleton class for processing methods. This class implements some
    useful features that allow the implementations of Algorithms,
    a key component of medusa.

    Check this `tutorial <http://www.example.com>`_ to better understand the
    usage of this class.
    """
    def __init__(self, **kwargs):
        """ProcessingMethod constructor

        Parameters
        ----------
        kwargs:
            Key-value arguments that define the exposed methods and output
            signature. This is used by class Algorithm for a correct
            implementation of signal processing pipelines.
        """
        # Get class funcs
        funcs = self.__get_methods()
        # Check errors
        for key, val in kwargs.items():
            if not key in funcs:
                raise TypeError('Method %s is not defined' % key)

            if not isinstance(val, list):
                raise TypeError('Value for method %s must be a list of str '
                                'with its output signature. ')
            for out in val:
                if not isinstance(out, str):
                    raise TypeError('Value for method %s must be a list of str '
                                    'with its output signature. ')
        self.exp_methods = kwargs

    def __get_methods(self):
        return [func for func in dir(self) if callable(getattr(self, func))]

    def get_exposed_methods(self):
        return self.exp_methods

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
        return self

    @classmethod
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
        instance: ProcessingMethod
            Instance of the processing method
        """
        return pickleable_obj


class ProcessingFuncWrapper(ProcessingMethod):
    """ProcessingMethod wrapper for processing functions. Use to add a
    processing function to an algorithm

    Check this `tutorial <http://www.example.com>`_ to better understand the
    usage of this class.
    """
    def __init__(self, func, outputs, **kwargs):
        """ProcessingFuncWrapper constructor

        Parameters
        ----------
        func: callable
            Function that will be implemented
        outputs: list
            Output signature of the method (output variables). This is used by
            class Algorithm for a correct implementation of signal processing
            pipelines.
        """
        # Check errors
        if not callable(func):
            raise TypeError('Parameter experiment_data must be callable')
        # Variables
        self.func_name = func.__name__
        self.module_name = func.__module__
        # Set func
        setattr(self, self.func_name, func)
        # setattr(self, self.func_name, self.set_defaults(func, **kwargs))
        # Call super
        super().__init__(**{self.func_name: outputs})


class ProcessingClassWrapper(ProcessingMethod):
    """ProcessingMethod wrapper for external classes (e.g., sklearn classifier).
    Use it to add an instance of the desired class to an algorithm. When
    designing your pipeline, take into account that the input signature
    (arguments) of the methods will be inherited from the original class.

    DISCLAIMER: This wrapper may not work with all classes, since it uses
    some hacking to bind the methods and attributes of the original instance
    to this wrapper, changing the original type. Additionally, it is assumed
    that the instance is pickleable. If this is not the case, or something
    doesn't work, you'll have to design your own wrapper subclassing
    ProcessingMethod, which is also very easy and quick.

    Check this `tutorial <http://www.example.com>`_ to better understand the
    usage of this class.
    """
    def __init__(self, instance, **kwargs):
        """ProcessingClassWrapper constructor

        Parameters
        ----------
        instance: object
            Instance of the class that will be implemented
        kwargs:
            Key-value arguments that define the exposed methods and output
            signature. This is used by class Algorithm for a correct
            implementation of signal processing pipelines.
        """
        # Inherit attributes from instance
        for k, v in inspect.getmembers(instance):
            if k.startswith('__') and k.endswith('__'):
                continue
            setattr(self, k, v)
        # Set useful variables
        self.class_name = type(instance).__name__
        self.module_name = instance.__module__
        # Call super
        super().__init__(**kwargs)

    def to_pickleable_obj(self):
        # TODO: workaround for error: TypeError: cannot pickle '_abc_data'
        #  object. It would be better to find another solution...
        self._abc_impl = None
        return self


class PipelineConnector:
    """Auxiliary class to define connections between stages of a pipeline

    Check this `tutorial <http://www.example.com>`_ to better understand the
    usage of this class.
    """

    def __init__(self, method_uid, output_key, conn_exp=None):
        """PipelineConnector constructor

        Parameters
        ----------
        method_uid: int
            Unique method identifier of method whose output will be connected.
        output_key: str
            Key of the output of method_id that will be passed. Useful when a
            method returns several variables, but only 1 is useful as input
            to other stage. If None, the output will be passed straightaway.
        conn_exp: callable
            Expresion that transforms the connected variable in some way.
            Fore instance, select a certain key from a dictionary, reshape an
            array, etc.
        """
        # Check errors
        if conn_exp is not None and not callable(conn_exp):
            raise TypeError('Parameter conn_exp must be callable or None')

        self.method_uid = method_uid
        self.output_key = output_key
        self.conn_exp = conn_exp

    def to_dict(self):
        return self.__dict__

    @staticmethod
    def from_dict(dict_data):
        return PipelineConnector(**dict_data)


class Pipeline:
    """Pipeline that defines the tasks and connections between methods of a
    signal processing task. This class does not check if the connections are
    valid. This is done by Algorithm class, which compiles the connections with
    the available methods

    Check this `tutorial <http://www.example.com>`_ to better understand the
    usage of this class.
    """
    def __init__(self):
        """Pipeline constructor
        """
        self.connections = []

    def input(self, args):
        """Defines the input arguments of the pipeline

        Parameters
        ----------
        args: list of str
            List of input arguments to the pipeline
        """
        kwargs = dict.fromkeys(args)
        if len(self.connections) == 0:
            self.connections.append(('input', kwargs))
        else:
            self.connections[0] = ('input', kwargs)
        return 0

    def add(self, method_func_key, **kwargs):
        """Adds a method to the pipeline

        Parameters
        ----------
        method_func_key: str
            Method identifier and function to be executed, separated by
            semicolon. Example: fir_filter:fit
        kwargs:
            Key-value arguments defining the input arguments of the methods.
            The key specifies the input argument. The value can be a static
            value (i.e., int, float, object instance) or a connection to the
            output of another stage of the pipeline. In this case, use method
            conn_to
        """
        if len(self.connections) == 0:
            raise ValueError('Call function input first')
        uid = len(self.connections)
        self.connections.append((method_func_key, kwargs))
        return uid

    def conn_to(self, uid, out_key, conn_exp=None):
        """Returns a PipelineConnector object that defines a connection
        between the input of a method and the ouput of a previous stage of
        the pipeline.

        Parameters
        ----------
        uid: int
            Stage unique id returned by input or add methods.
        out_key: str
            Key of the output of the method given by uid that will be
            connected to the input argument.
        conn_exp: callable
            Expresion that transforms the connected variable in some way.
            Fore instance, select a certain key from a dictionary, reshape an
            array, etc.
        """
        if uid >= len(self.connections):
            raise ValueError('Incorrect uid parameter. The connection must '
                             'be with a previous step of the pipeline.')
        return PipelineConnector(uid, out_key, conn_exp)


class Algorithm(ProcessingMethod):
    """Algorithm class is the main tool within medusa to implement standalone
    processing algorithms that can be shared as a simple file, supporting
    third-party libraries, such as sklearn. It allows persistence to save the
    algorithm and its state or use it later using dill package. Take into
    account that the algorithm needs access to the original classes and methods
    in order to be reconstructed.

    Check this `tutorial <http://www.example.com>`_ to better understand the
    usage of this class.
    """
    def __init__(self, **kwargs):
        super().__init__(exec_pipeline=['results'], **kwargs)
        self.methods = dict()
        self.pipelines = dict()

    def add_method(self, method_key, method_instance):
        if not isinstance(method_key, str):
            raise TypeError('Parameter method_id must be of type str')
        if not issubclass(type(method_instance), ProcessingMethod):
            raise TypeError('Parameter method_instance must be subclass of %s'
                            % str(type(ProcessingMethod)))
        method_dict = {
            'module_name': method_instance.__module__,
            'class_name': type(method_instance).__name__,
            'instance': method_instance
        }
        self.methods[method_key] = method_dict

    def add_pipeline(self, pipeline_key, pipeline_instance):
        if not isinstance(pipeline_key, str):
            raise TypeError('Parameter pipeline_key must be of type str')
        if not issubclass(type(pipeline_instance), Pipeline):
            raise TypeError('Parameter pipeline_instance must be subclass of %s'
                            % str(type(Pipeline)))
        self.pipelines[pipeline_key] = \
            self.__compile_pipeline(pipeline_instance)

    def __compile_pipeline(self, pipeline):
        connections = copy.deepcopy(pipeline.connections)
        parsed_connections = list()
        for conn in connections:
            # Method to connect
            conn_method_func = conn[0]
            conn_method_params = conn[1]
            # Take care with methods
            if len(conn_method_func.split(':')) < 2:
                conn_method_func = ':'.join([conn_method_func]*2)
            # Get id and func
            conn_method_func_split = conn_method_func.split(':')
            conn_method_key = conn_method_func_split[0]
            conn_method_func_key = conn_method_func_split[1]
            for param_key, param_value in conn_method_params.items():
                if conn_method_key != 'input':
                    try:
                        # Inspect function
                        ins = inspect.getfullargspec(
                            getattr(self.methods[conn_method_key]['instance'],
                                    conn_method_func_key)
                        )
                    except AttributeError as e:
                        raise AttributeError(
                            'Function %s is not defined in method %s.' %
                            (conn_method_func_key, conn_method_key)
                        )

                    # Check that the argument exists
                    if param_key not in ins.args:
                        if ins.varkw is None:
                            raise KeyError(
                                'Input %s is not defined in method %s. '
                                'Available inputs: %s' %
                                (param_key, conn_method_func, ins.args)
                            )

                # Check connection
                is_connector = isinstance(param_value, PipelineConnector)
                if is_connector:
                    # Get out_method_key_func
                    out_method_key_func = connections[param_value.method_uid][0]
                    # Take care
                    if len(out_method_key_func.split(':')) < 2:
                        out_method_key_func = ':'.join([out_method_key_func]*2)
                    # Check that the output exists
                    out_method_key_func_split = out_method_key_func.split(':')
                    out_method_key = out_method_key_func_split[0]
                    out_method_func = out_method_key_func_split[1]
                    if out_method_key != 'input':
                        # Check that the method has been added to the algorithm
                        if out_method_key not in self.methods:
                            raise KeyError('Method %s has not been added to '
                                           'the algorithm.' %
                                           out_method_key_func)

                        # Check exposed methods and outputs
                        out_exp_methods = \
                            self.methods[out_method_key]['instance'].exp_methods
                        try:
                            out_exp_method = out_exp_methods[out_method_func]
                        except KeyError as e:
                            raise KeyError('Method %s is not exposed' %
                                           out_method_key_func)

                        if param_value.output_key not in out_exp_method:
                            raise KeyError('Output %s from method %s is not '
                                           'exposed. Available: %s' %
                                           (param_value.output_key,
                                            out_method_key_func,
                                            str(out_exp_method)))
                    else:
                        # Get input keys
                        input_keys = list(parsed_connections[0][1].keys())
                        if param_value.output_key not in input_keys:
                            raise KeyError('Output %s from method %s is not '
                                           'exposed. Available: %s' %
                                           (param_value.output_key,
                                            out_method_key_func,
                                            str(input_keys)))

                    param_value = {
                        'connector': is_connector,
                        'value': param_value.to_dict()
                    }
                else:
                    param_value = {
                        'connector': is_connector,
                        'value': param_value
                    }
                conn_method_params[param_key] = param_value
            parsed_connections.append(
                (conn_method_func, conn_method_params)
            )
        # Delete the first stage, which is not a method but the input of the
        # pipeline. Parsed connections only has to store the applied methods.
        # parsed_connections.pop(0)
        return parsed_connections

    @staticmethod
    def __get_inputs(method_key_func, input_map, exec_methods):
        """ Gets the inputs for the next method"""
        inputs = {}
        for inp_key, inp_value in input_map.items():
            if inp_value['connector']:
                res_method_uid = inp_value['value']['method_uid']
                res_key = inp_value['value']['output_key']
                res_exp = inp_value['value']['conn_exp']
                res_method_dict = exec_methods[res_method_uid]
                try:
                    inputs[inp_key] = res_method_dict['res'][res_key]
                    # Evaluate connector expression
                    if res_exp is not None:
                        inputs[inp_key] = res_exp(inputs[inp_key])
                except KeyError:
                    raise KeyError('Input %s to %s not available from %s. '
                                   'Available: %s' %
                                   (res_key, method_key_func,
                                    res_method_dict['key'],
                                    str(list(res_method_dict['res'].keys()))))
            else:
                inputs[inp_key] = inp_value['value']
        return inputs

    @staticmethod
    def __map_output_to_dict(method_key_func, method, func, output):
        try:
            if not isinstance(output, list) and not isinstance(output, tuple):
                output = [output]
            # Map outputs
            out_dict = {}
            for i, key in enumerate(method.exp_methods[func]):
                out_dict[key] = output[i]
        except KeyError as e:
            raise KeyError('Function %s was not found. It has been exposed?'
                           % (method_key_func))
        except IndexError as e:
            raise IndexError('Error mapping outputs of %s. Check the outputs.'
                             % (method_key_func))
        return out_dict

    def exec_pipeline(self, pipeline_key, **kwargs):
        """ Execute pipeline"""
        # Check kwargs
        in_kwargs = self.pipelines[pipeline_key][0][1]
        if list(in_kwargs.keys()) != list(kwargs.keys()):
            raise ValueError('Wrong input. Specified args: %s' %
                             str(list(in_kwargs.keys())))

        # Init
        results = collections.OrderedDict()
        results[0] = {'key': 'input', 'res': kwargs, 'perf': None}

        # Execute pipeline
        for s in range(1, len(self.pipelines[pipeline_key])):

            # Stage (method_key_func, input_map)
            method_key_func = self.pipelines[pipeline_key][s][0]
            input_map = self.pipelines[pipeline_key][s][1]

            # Get inputs
            inputs = self.__get_inputs(method_key_func, input_map, results)

            # Method
            method_key_func_split = method_key_func.split(':')
            method_key = method_key_func_split[0]
            method_func = method_key_func_split[1]

            # Get method instance
            method = self.methods[method_key]['instance']
            func = perf_analysis(getattr(method, method_func))
            out, perf_profile = func(**inputs)
            out_dict = self.__map_output_to_dict(method_key_func, method,
                                                 method_func, out)
            # Append results
            results[s] = {'key': method_key_func,
                          'res': out_dict,
                          'perf': perf_profile}

        return results

    def get_inst(self, method_key):
        """Returns the instance of a method given the key"""
        return self.methods[method_key]['instance']

    def to_pickleable_obj(self):
        # Get pickleable objects of the methods
        for method_key, method_dict in self.methods.items():
            self.methods[method_key]['instance'] = \
                method_dict['instance'].to_pickleable_obj()
        return self

    @classmethod
    def from_pickleable_obj(cls, alg):
        # Reconstruct methods
        for method_key, method_dict in alg.methods.items():
            # Check if the obj is already a ProcessingMethod instance
            if not issubclass(type(method_dict['instance']), ProcessingMethod):
                # Load class
                try:
                    module = sys.modules[method_dict['module_name']]
                    obj = getattr(module, method_dict['class_name'])
                except KeyError as e:
                    raise ImportError(
                        'Class %s has not been found in module %s. '
                        'This object must be reachable in this '
                        'module or defined in the main program. Did you import '
                        'the module %s before using this function?'
                        % (method_dict['class_name'],
                           method_dict['module_name'],
                           method_dict['module_name'])
                    )
                # Load instance from pickleable object
                alg.methods[method_key]['instance'] = \
                    obj.from_pickleable_obj(method_dict['instance'])
        return alg


class ThreadWithReturnValue(Thread):
    """This class inherits from thread class and allows getting
     the target function return"""
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)

    def join(self, *args):
        Thread.join(self, *args)
        return self._return
