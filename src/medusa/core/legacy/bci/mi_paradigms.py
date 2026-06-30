"""Legacy motor imagery experiment-data class (1.x persistence compatibility).

Data-loading only. ``MIData`` is copied verbatim from
``medusa.pipelines.bci.mi_paradigms`` and stripped of all processing / ML /
dataset / model code, so the 2.0 kernel can *load* legacy ``.mi.bson``
recordings without importing the heavy pipeline module. The legacy module
name (``medusa.bci.mi_paradigms``) is remapped here in
``medusa.core.legacy.recording._LEGACY_MODULE_MAP``.
"""

import numpy as np

from medusa.core.legacy.experiment import ExperimentData


class MIData(ExperimentData):
    # TODO: Check everything

    """Class with the necessary attributes to define motor imagery (MI)
    experiments. It provides compatibility with high-level functions for this
    MI paradigms in BCI module.
    """

    def __init__(self, mode, onsets, w_trial_t,
                 calibration_onset_w=None, w_preparation_t=None, w_rest_t=None,
                 mi_labels=None, mi_labels_info=None, mi_result=None,
                 paradigm_info=None, **kwargs):
        """MIData constructor

        Parameters
        ----------
        mode : str {"train"|"test"|"guided test"}
            Mode of this run.
        onsets : list or numpy.ndarray [n_stim x 1]
            Timestamp of the cue with respect to the EEG signal (just when the
            motor imagery starts).
        w_trial_t: list [start, end]
            Temporal window of the motor imagery with respect to each onset in
            ms. For example, if  w_trial_t = [500, 4000] the subject was
            performing the motor imagery task from 500ms to 4000ms after
            the onset.
        calibration_onset_w: list [start, end]
            Timestamps of the onsets regarding the calibration window,
            if exists, respect to the EEG signal.
        w_preparation_t: list [start, end]
            Temporal window of the preparation time (no motor imagery),
            if exists, with respect to each onset in ms.
        w_rest_t: list [start, end]
            Temporal window of the rest time (no motor imagery),
            if exists, with respect to each onset in ms.
        mi_labels : list or numpy.ndarray [n_mi_labels x 1]
            Only in train mode. Contains the mi labels of each stimulation,
            as many as classes in the experiment.
        mi_labels_info : dict
            Contains the description of the mi labels.
            Example:
                mi_labels_info =
                    {0: "Rest",
                    1: "Left_hand",
                    2: "Right_hand"}
        mi_result : list or numpy.ndarray [n_mi_labels x 1]
            Result of this run. Each position contains the data of the
            selected target at each trial.
        paradigm_info: dict()
            Recommended but not mandatory. Use this variable to keep the
            information regarding the different timings of the paradigm.
        kwargs : kwargs
            Custom arguments that will also be saved in the class
            (e.g., timings, calibration gaps, etc.)
        """

        # Check errors
        mode = mode.lower()
        if mode == 'train':
            if mi_labels is None:
                raise ValueError('Attribute "mi_labels" must be provided in '
                                 'train mode')

        # Standard attributes
        self.mode = mode
        self.onsets = np.array(onsets)
        self.w_trial_t = np.array(w_trial_t)
        self.calibration_onset_w = np.array(calibration_onset_w)
        self.w_preparation_t = np.array(w_preparation_t)
        self.w_rest_t = np.array(w_rest_t)
        self.mi_labels = np.array(mi_labels) if mi_labels is not None else \
            mi_labels
        self.mi_labels_info = mi_labels_info
        self.mi_result = np.array(mi_result) if mi_result is not None else \
            mi_result
        self.paradigm_info = paradigm_info

        # Optional attributes
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
