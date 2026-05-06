# Built-in imports
import sys, inspect, copy, collections

# External imports
import dill

# Medusa imports
from medusa.core.serialization import PickleableComponent
from medusa.core.profiling import perf_analysis


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
