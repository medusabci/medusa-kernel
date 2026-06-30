"""Legacy neurofeedback experiment-data class (1.x persistence compatibility).

Data-loading only. ``NeurofeedbackData`` is copied verbatim from
``medusa.pipelines.bci.nft_paradigms`` and stripped of all processing / ML /
dataset / model code, so the 2.0 kernel can *load* legacy ``.nft.bson``
recordings without importing the heavy pipeline module. The legacy module
name (``medusa.bci.nft_paradigms``) is remapped here in
``medusa.core.legacy.recording._LEGACY_MODULE_MAP``.
"""

import numpy as np

from medusa.core.legacy.experiment import ExperimentData


class NeurofeedbackData(ExperimentData):
    """Experiment info class for Neurofeedback training experiments. It records
    the important events that take place during a Neurofeedback run,
    allowing offline analysis."""

    def __init__(self, run_onsets, run_durations, run_success, run_pauses,
                 run_restarts, medusa_nft_app_settings, nft_values, nft_times,
                 nft_baseline, **kwargs):

        self.run_onsets = run_onsets
        self.run_durations = run_durations
        self.run_success = run_success
        self.run_pauses = run_pauses
        self.run_restarts = run_restarts
        self.medusa_nft_app_settings = medusa_nft_app_settings
        self.nft_values = nft_values
        self.nft_times = nft_times
        self.nft_baseline = nft_baseline

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
