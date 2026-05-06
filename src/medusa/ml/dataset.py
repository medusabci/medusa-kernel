# Built-in imports
from abc import ABC

# Medusa imports
from medusa.core.data.recording import Recording, ConsistencyChecker


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
