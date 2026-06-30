"""Legacy SSVEP speller experiment-data class (1.x persistence compatibility).

Data-loading only. ``SSVEPSpellerData`` is copied verbatim from
``medusa.pipelines.bci.ssvep_spellers`` and stripped of all processing / ML /
dataset / model code, so the 2.0 kernel can *load* legacy ``.ssvep.bson``
recordings without importing the heavy pipeline module. The legacy module
name (``medusa.bci.ssvep_spellers``) is remapped here in
``medusa.core.legacy.recording._LEGACY_MODULE_MAP``.
"""

import numpy as np

from medusa.core.legacy.experiment import ExperimentData


class SSVEPSpellerData(ExperimentData):
    """Experiment info class for SSVEP-based spellers. It supports nested
    multi-matrix multi-level paradigms. This unified class can be used to
    represent a run of every SSVEP stimulation paradigm designed to date,
    and is the expected class for feature extraction and command decoding
    functions of the module medusa.bci.ssvep_spellers. It is complicated,
    but powerful so.. use it well!
    """

    def __init__(self, mode, paradigm_conf, commands_info, onsets,
                 unit_idx, level_idx, matrix_idx, trial_idx,
                 cmd_model, csd_model, spell_result, control_state_result,
                 fps_resolution, stim_time, stim_freq_range, spell_target=None,
                 control_state_target=None, **kwargs):

        # Check errors
        mode = mode.lower()
        if mode not in ('train', 'test'):
            raise ValueError('Unknown mode. Possible values {train, test}')

        # Standard attributes
        self.mode = mode
        self.paradigm_conf = paradigm_conf
        self.commands_info = commands_info
        self.onsets = onsets
        self.unit_idx = unit_idx
        self.level_idx = level_idx
        self.matrix_idx = matrix_idx
        self.trial_idx = trial_idx
        self.cmd_model = cmd_model
        self.csd_model = csd_model
        self.spell_result = spell_result
        self.control_state_result = control_state_result
        self.fps_resolution = fps_resolution
        self.stim_time = stim_time
        self.stim_freq_range = stim_freq_range
        self.spell_target = spell_target
        self.control_state_target = control_state_target

        # Optional attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

    @staticmethod
    def get_paradigm_conf_for_freq_enc(matrix_dims, commands_info=None):
        """Simple frequency encoding paradigm with no levels

        Example of a frequency encoding paradigm with 4 commands:

                paradigm_conf = [
                    # Matrices
                    [
                        # Units
                        [
                            # Groups
                            [
                                # Batches
                                [0, 1, 2, 3, 4]
                            ]
                        ]
                    ]
                ]
        """
        matrix_dims = np.array(matrix_dims)
        paradigm_conf = list()
        commands_info_list = list()

        for m in range(matrix_dims.shape[0]):
            # Commands matrix
            n_rows = matrix_dims[m, 0]
            n_cols = matrix_dims[m, 1]
            commands_ids = np.arange(n_rows * n_cols).tolist()
            # Paradigm conf. Groups and batches are not necessary for SSVEP
            # spellers, only matrices and units for multilevel paradigms.
            paradigm_conf.append(list())                    # Matrix
            paradigm_conf[m].append(list())                 # Unit
            paradigm_conf[m][0].append(commands_ids)        # Group
            # Commands info
            if commands_info is None:
                cmd_info_values = [dict() for i in commands_ids]
            else:
                cmd_info_values = np.array(commands_info[m]).flatten()
                cmd_info_values = cmd_info_values.tolist()
            commands_info_list.append(dict(zip(commands_ids, cmd_info_values)))

        return paradigm_conf, commands_info_list

    def to_serializable_obj(self):
        rec_dict = self.__dict__
        for key in rec_dict.keys():
            if type(rec_dict[key]) == np.ndarray:
                rec_dict[key] = rec_dict[key].tolist()
        return rec_dict

    @classmethod
    def from_serializable_obj(cls, dict_data):
        return cls(**dict_data)
