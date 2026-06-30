# Built-in imports
from abc import abstractmethod

# External imports
import numpy as np

# Medusa imports
from medusa.core.serialization import SerializableComponent


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
