"""Legacy c-VEP speller experiment-data class (1.x persistence compatibility).

Data-loading only. ``CVEPSpellerData`` is copied verbatim from
``medusa.pipelines.bci.cvep_spellers`` and stripped of all processing / ML /
dataset / model code, so the 2.0 kernel can *load* legacy ``.cvep.bson``
recordings without importing the heavy pipeline module. The legacy module
name (``medusa.bci.cvep_spellers``) is remapped here in
``medusa.core.legacy.recording._LEGACY_MODULE_MAP``.
"""

import numpy as np

from medusa.core.legacy.experiment import ExperimentData


class CVEPSpellerData(ExperimentData):
    """Experiment info class for c-VEP-based spellers. It supports nested
    multi-level paradigms. This unified class can be used to represent a run
    of every c-VEP stimulation paradigm designed to date, and is the expected
    class for feature extraction and command decoding functions of the module
    medusa.bci.cvep_paradigms. It is complicated, but powerful so.. use it well!
    """

    def __init__(self, mode, paradigm_conf, commands_info, onsets, command_idx,
                 unit_idx, level_idx, matrix_idx, cycle_idx, trial_idx,
                 spell_result, fps_resolution, spell_target=None,
                 raster_events=None, **kwargs):

        # Check errors
        if mode not in ('train', 'test'):
            raise ValueError('Unknown mode. Possible values {train, test}')

        # Standard attributes
        self.mode = mode
        self.paradigm_conf = paradigm_conf
        self.commands_info = commands_info
        self.onsets = onsets
        self.command_idx = command_idx
        self.unit_idx = unit_idx
        self.level_idx = level_idx
        self.matrix_idx = matrix_idx
        self.cycle_idx = cycle_idx
        self.trial_idx = trial_idx
        self.spell_result = spell_result
        self.fps_resolution = fps_resolution
        self.spell_target = spell_target
        self.raster_events = raster_events

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
