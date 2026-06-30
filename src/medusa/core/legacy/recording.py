# Built-in imports
import sys
import warnings
import importlib

# External imports

# Medusa imports
from medusa.core.serialization import SerializableComponent
from medusa.core.legacy.signal import Signal, CustomSignal
from medusa.core.legacy.experiment import ExperimentData, CustomExperimentData


# Backward-compatibility map for module renames between Kernel <2.0 (flat
# layout) and the post-K1 hierarchical layout. Used when loading legacy
# .rec.* files whose serialized `module_name` field still references the
# original module path. Update this table when modules are moved.
_LEGACY_MODULE_MAP = {
    # 1.x biosignal classes (EEG/MEG/ECG/...) now live under the compat tree.
    # 1.x serialized `module_name` is the package path `medusa.meeg.meeg`
    # (some builds wrote the bare package `medusa.meeg`); map both.
    'medusa.meeg.meeg': 'medusa.core.legacy.biosignals.eeg.eeg',
    'medusa.meeg': 'medusa.core.legacy.biosignals.eeg.eeg',
    'medusa.ecg.ecg': 'medusa.core.legacy.biosignals.ecg.ecg',
    'medusa.ecg': 'medusa.core.legacy.biosignals.ecg.ecg',
    # `medusa.components` was the 1.x dumping ground for Recording,
    # ExperimentData, CustomExperimentData, ConsistencyChecker, etc. Old
    # serialized recordings reference it via `module_name`. Repoint to the
    # compat `Recording` module, whose namespace also re-exports the experiment
    # bases (`ExperimentData`/`CustomExperimentData`) and signal bases
    # (`Signal`/`CustomSignal`); the resolver only needs *some* module that
    # exposes the recorded class name.
    'medusa.components': 'medusa.core.legacy.recording',
    'medusa.io': 'medusa.core.legacy.recording',
    # 1.x BCI paradigm *data* classes (CVEPSpellerData / ERPSpellerData / MIData
    # / SSVEPSpellerData / NeurofeedbackData). Remapped to the lightweight compat
    # copies under `core.legacy.bci` -- NOT the full `pipelines.bci.*` modules,
    # which pull in ml/processing deps and must not be on the load path.
    'medusa.bci.erp_spellers': 'medusa.core.legacy.bci.erp_spellers',
    'medusa.bci.cvep_spellers': 'medusa.core.legacy.bci.cvep_spellers',
    'medusa.bci.ssvep_spellers': 'medusa.core.legacy.bci.ssvep_spellers',
    'medusa.bci.mi_paradigms': 'medusa.core.legacy.bci.mi_paradigms',
    'medusa.bci.nft_paradigms': 'medusa.core.legacy.bci.nft_paradigms',
}


def _resolve_serialized_module(module_name):
    """Resolve a (possibly legacy) module path to an imported module.

    Tries the recorded path first; if it is not currently importable,
    falls back to ``_LEGACY_MODULE_MAP`` to honor the public contract for
    loading recordings produced by older Kernel versions.
    """
    candidates = [module_name]
    if module_name in _LEGACY_MODULE_MAP:
        candidates.append(_LEGACY_MODULE_MAP[module_name])
    for candidate in candidates:
        if candidate in sys.modules:
            return sys.modules[candidate]
        try:
            return importlib.import_module(candidate)
        except ImportError:
            continue
    raise KeyError(module_name)




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
                            'ExperimentData')
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
        if not isinstance(biosignal, Signal):
            raise TypeError('Parameter biosignal must subclass Signal')
        # Check type
        biosignal_module_name = type(biosignal).__module__
        biosignal_class_name = type(biosignal).__name__
        att = biosignal_class_name.lower() if key is None else key
        if isinstance(biosignal, CustomSignal):
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
        if not issubclass(biosignal_class, Signal):
            raise TypeError('Class %s must subclass Signal' %
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
            raise TypeError('Class %s must subclass ExperimentData' %
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
                    module = _resolve_serialized_module(
                        biosignal_dict['module_name'])
                    obj = getattr(module, biosignal_dict['class_name'])
                except (KeyError, AttributeError):
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
                    module = _resolve_serialized_module(
                        exp_dict['module_name'])
                    obj = getattr(module, exp_dict['class_name'])
                except (KeyError, AttributeError):
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
