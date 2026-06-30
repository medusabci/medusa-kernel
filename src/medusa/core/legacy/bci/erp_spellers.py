"""Legacy ERP / RCP speller experiment-data class (1.x persistence compatibility).

Data-loading only. ``ERPSpellerData`` is copied verbatim from
``medusa.pipelines.bci.erp_spellers`` and stripped of all processing / ML /
dataset / model code, so the 2.0 kernel can *load* legacy ``.rcp.bson``
recordings without importing the heavy pipeline module. The legacy module
name (``medusa.bci.erp_spellers``) is remapped here in
``medusa.core.legacy.recording._LEGACY_MODULE_MAP``.
"""

import numpy as np

from medusa.core.legacy.experiment import ExperimentData


class ERPSpellerData(ExperimentData):
    """Experiment info class for ERP-based spellers. It supports nested
    multi-level paradigms. This unified class can be used to represent a run
    of every ERP stimulation paradigm designed to date, and is the expected
    class for feature extraction and command decoding functions of the module
    medusa.bci.erp_paradigms. It is complicated, but powerful so.. use it well!
    """

    def __init__(self, mode, paradigm_conf, commands_info, onsets, batch_idx,
                 group_idx, unit_idx, level_idx, matrix_idx, sequence_idx,
                 trial_idx, spell_result, control_state_result,
                 spell_target=None, control_state_target=None, **kwargs):

        """ERPSpellerData constructor

        Parameters
        ----------
        mode : str
            Mode of this run. Values: {"train"|"test"}
        paradigm_conf :  list
            This parameter describes the paradigm configuration for the
            experiment. The array must have shape [n_matrices x n_units x
            n_groups x n_batches x n_commands/batch]. The matrix is the maximum
            entity of the paradigm and only one can be used in each trial.
            The units are smaller entities that are used in multi-level
            paradigms, such as the Hex-O-spell (HOS) paradigm [1]. In this
            case, each level can use a different unit, affecting the selected
            command for the trial. For instance, in the HOS paradigm,
            you should define 1 matrix with 7 units, one for the initial menu
            and 6 for the second level of each command (letters).
            Importantly, commands must be unequivocally defined in each matrix.
            Therefore, units cannot share command identifiers. Then, the groups
            describe aggregations of commands that are highlighted at the
            same time. For instance, the row-column paradigm (RCP) [2]
            has 2 groups of commands (i.e., rows and columns), whereas the
            HOS has only 1 (i.e., each command is highlighted individually).
            Finally, batches contain the commands IDs defined in each group.
            In an RCP matrix of 6x6, each of the 2 groups has 6 batches,
            corresponding to the rows and columns. This structure supports
            nested multi-level matrices, providing compatibility with all
            paradigms to date and setting a general framework for feature
            extraction and command decoding functions. The relationship between
            the command IDs and the letters or actions should be defined in
            other variable, but it is not necessary for signal processing.

            Example of 2x2 RCP paradigm:

                rcp_conf = [
                    # Matrices
                    [
                        # Units
                        [
                            # Groups
                            [
                                # Batches
                                [0, 1],
                                [2, 3]
                            ],
                            [
                                [0, 2],
                                [1, 3]
                            ]
                        ]
                    ]
                ]

            Example of HOS paradigm:

                hos_conf = [
                    # Matrices
                    [
                        # Units
                        [
                            # Groups
                            [
                                # Batches
                                [0], [1], [2], [3], [4], [5]
                            ],
                        ],
                        [
                            [
                                [6], [7], [8], [9], [10], [11]
                            ],
                        ],
                        [
                            [
                                [12], [13], [14], [15], [16], [17]
                            ],
                        ],
                        [
                            [
                                [18], [19], [20], [21], [22], [23]
                            ],
                        ],
                        [
                            [
                                [24], [25], [26], [27], [28], [29]
                            ],
                        ]
                    ]
                ]
        commands_info : list
            List containing the command information per matrix. Each
            position must be a dict, whose keys are the command ids used in
            paradigm_conf. The value must be another dict containing important
            information about each command (e.g., label, text, action, icon
            path, etc). This information may be different for different use
            cases, but must be serializable (i.e., contain primitive types).
            Shape [n_matrices x n_commands].
        onsets : list or numpy.ndarray
            Timestamp of each stimulation. This timestamps have to be
            synchronized with the EEG (or other biosignal) timestamps in
            order to assure a correct functioning of all medusa functions.
            Shape: [n_stim x 1]
        batch_idx : list or numpy.ndarray
            Index of the highlighted batch for each stimulation. A batch
            represents the highlighted commands in each  stimulation. For
            example in the row-col paradigm (RCP) represents each row and
            column. Shape: [n_stim x 1]
        group_idx : list or numpy.ndarray
            Index of the group that has been highlighted. Groups represent the
            different aggregations of batches. Between batches of different
            groups, 1 command must be common. For example in the RCP there
            are 2 groups: rows and columns. In this paradigm, between each
            pair of batches (e.g., row=2, col=4), there is only one command
            in common. Shape: [n_stim x 1]
        unit_idx: list or numpy.ndarray
            Index of the unit used in each stimulation. Units are low level
            entities used in multi-level paradigms, such as HOS paradigm [1].
            For each level, only 1 unit can be used. As the trial may have
            several layers, several units can be used in 1 trial. For
            instance, in the HOS, the first unit is the main menu. The other
            6 units are each of the lower level entities that are displayed
            in the second level of stimulation. Shape: [n_stim x 1]
        level_idx : list or numpy.ndarray
            Index of the level of each stimulation. Levels represent each
            one of the selections that must be made before a trial is
            finished. For example, in the Hex-O-spell paradigm there are 2
            levels (see [1]). Shape: [n_stim x 1]
        matrix_idx : list or numpy.ndarray
            Index of the matrix used in each stimulation. Each matrix can
            contain several levels. The matrix has to be the same accross the
            entire trial. Shape: [n_stim x 1]
        sequence_idx : list or numpy.ndarray
            Index of the sequence for each stimulation. A sequence
            represents a round of stimulation: all commands have been
            highlighted 1 time. This class support dynamic stopping in
            different levels. Shape: [n_stim x 1]
        trial_idx : list or numpy.ndarray
            Index of the trial for each stimulation. A trial represents
            the selection of a final command. Depending on the number of levels,
            the final selection takes N intermediate selections.
        spell_result : list or numpy.ndarray
            Spell result of the run. Each position contains the matrix and
            command id that identifies the selected command per trial and
            level. Shape [n_trials x n_levels x 2]. Eg., in the RCP:
            [[[matrix_idx, cmd_id], [matrix_idx, cmd_id]]].
        control_state_result : list or numpy.ndarray
            Control state result of the run. Each position contains the
            detected control state of the user per trial (0 -> non-control,
            1-> control). Shape: [n_trials x 1]. Values {0|1}
        spell_target : list or numpy.ndarray or None
            Spell target of the run. Each position contains the matrix and
            command id per level that identifies the target command of the
            trial. Shape [n_trials x n_levels x 2]. Eg., in the RCP:
            [[[matrix_idx, cmd_id], [matrix_idx, cmd_id], etc]].
        control_state_target : list or numpy.ndarray or None
            Control state target of the run. Each position contains the
            target control state of the user per trial (0 -> non-control,
            1-> control). Shape: [n_trials x 1]. Values {0|1}
        kwargs : kwargs
            Custom arguments that will also be saved in the class


        References
        ----------
        [1] Blankertz, B., Dornhege, G., Krauledat, M., Schröder,
        M., Williamson, J., Murray-Smith, R., & Müller, K. R. (2006). The
        Berlin Brain-Computer Interface presents the novel mental typewriter
        Hex-o-Spell.

        [2] Farwell, L. A., & Donchin, E. (1988). Talking off the top of your
        head: toward a mental prosthesis utilizing event-related brain
        potentials. Electroencephalography and clinical Neurophysiology,
        70(6), 510-523.
        """
        # Check errors
        mode = mode.lower()
        if mode not in ('train', 'test'):
            raise ValueError('Unknown mode. Possible values {train, test}')

        # Standard attributes
        self.mode = mode
        self.paradigm_conf = paradigm_conf
        self.commands_info = commands_info
        self.onsets = onsets
        self.batch_idx = batch_idx
        self.group_idx = group_idx
        self.unit_idx = unit_idx
        self.level_idx = level_idx
        self.matrix_idx = matrix_idx
        self.sequence_idx = sequence_idx
        self.trial_idx = trial_idx
        self.spell_result = spell_result
        self.control_state_result = control_state_result
        self.spell_target = spell_target
        self.control_state_target = control_state_target
        self.erp_labels = self.compute_erp_labels() \
            if mode == 'train' else None
        self.control_state_labels = self.compute_control_state_labels() \
            if mode == 'train' else None
        # Optional attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

    def get_batches_associated_to_cmd(self, matrix_idx, command_idx):
        """This function returns the batches associated to the given command
        as defined in the paradigm configuration

        Parameters
        ----------
        matrix_idx: int
            Matrix of the command
        command_idx: int
            Index of the command as defined in attribute commands_info
        """
        tg_ids_batches_list = list()
        for u, unit in enumerate(self.paradigm_conf[matrix_idx]):
            for g, group in enumerate(unit):
                for b, batch in enumerate(group):
                    if command_idx in batch:
                        tg_ids_batches_list.append(
                            [matrix_idx, u, g, b])
        return tg_ids_batches_list

    def compute_erp_labels(self):
        """
        This function computes the erp label vector (0 if the epoch doesn't
        have ERP, 1 if the epoch have ERP).
        """
        # Convert to numpy array
        batch_idx = np.array(self.batch_idx)
        group_idx = np.array(self.group_idx)
        unit_idx = np.array(self.unit_idx)
        level_idx = np.array(self.level_idx)
        matrix_idx = np.array(self.matrix_idx)
        trial_idx = np.array(self.trial_idx)
        # Get batches associated to each target
        target_ids_batches = dict()
        for t, trial_target in enumerate(self.spell_target):
            for l, level_target in enumerate(trial_target):
                tg_matrix_idx = level_target[0]
                tg_id = level_target[1]
                target_ids_batches[(t, l)] = self.get_batches_associated_to_cmd(
                    tg_matrix_idx, tg_id)
        # Compute erp_labels
        erp_labels = np.zeros((len(batch_idx))).astype(int)
        for key, batches in target_ids_batches.items():
            for batch in batches:
                erp_labels_idx = np.ones((len(batch_idx))).astype(int)
                erp_labels_idx = np.logical_and(erp_labels_idx,
                                                trial_idx == key[0])
                erp_labels_idx = np.logical_and(erp_labels_idx,
                                                level_idx == key[1])
                erp_labels_idx = np.logical_and(erp_labels_idx,
                                                matrix_idx == batch[0])
                erp_labels_idx = np.logical_and(erp_labels_idx,
                                                unit_idx == batch[1])
                erp_labels_idx = np.logical_and(erp_labels_idx,
                                                group_idx == batch[2])
                erp_labels_idx = np.logical_and(erp_labels_idx,
                                                batch_idx == batch[3])
                erp_labels[erp_labels_idx] = 1
        return erp_labels

    def compute_control_state_labels(self):
        """
        This function computes the control state label vector (0 -> non-control
        state, 1 -> control state).
        """
        trial_idx = np.array(self.trial_idx)
        labels = np.zeros_like(trial_idx).astype(int)
        for t, trial in enumerate(np.unique(trial_idx)):
            labels[trial_idx == trial] = self.control_state_target[t]
        return labels

    @staticmethod
    def get_paradigm_conf_for_rcp(matrix_dims, commands_info_rcp=None):
        """Returns the paradigm configuration matrix for the row-column paradigm
        (RCP) experiment [1]

        Parameters
        ----------
        matrix_dims : list or np.array
            Array containing the dimensions of the matrices used in the
            experiment. For each matrix, the first position is the number of
            rows and the second the number of columns. Shape [n_matrices x 2]
        commands_info_rcp: list or None
            Array containing the dict info of each command, arranged in 2D
            matrices. Shape [n_matrices x n_rows x n_cols]

        Returns
        -------
        paradigm_conf : list
            Array with the paradigm configuration for an RCP paradigm
        commands_info : list
            Flattened version of commands_info input. It contains the command
            information corresponding to paradigm_conf. If input parameter
            commands_info is None, this output will be a skeleton with an empty
            dict for each command. If input commands_info is provided, it will
            be incorporated in the output

        References
        ----------
        [1] Farwell, L. A., & Donchin, E. (1988). Talking off the top of your
        head: toward a mental prosthesis utilizing event-related brain
        potentials. Electroencephalography and clinical Neurophysiology,
        70(6), 510-523.
        """
        # Paradigm conf
        matrix_dims = np.array(matrix_dims)
        paradigm_conf = list()
        commands_info = list()
        for m in range(matrix_dims.shape[0]):
            # Commands matrix
            n_rows = matrix_dims[m, 0]
            n_cols = matrix_dims[m, 1]
            commands_ids = np.arange(n_rows * n_cols)
            matrix = commands_ids.reshape((n_rows, n_cols))
            # Paradigm conf
            paradigm_conf.append(list())                    # Matrix
            paradigm_conf[m].append(list())                 # Unit
            paradigm_conf[m][0].append(matrix.tolist())     # Append group rows
            paradigm_conf[m][0].append(matrix.T.tolist())   # Append group cols
            # Commands info
            cmd_info_keys = commands_ids.tolist()
            if commands_info_rcp is None:
                cmd_info_values = [dict() for i in cmd_info_keys]
            else:
                cmd_info_values = np.array(commands_info_rcp[m]).flatten()
                cmd_info_values = cmd_info_values.tolist()
            commands_info.append(dict(zip(cmd_info_keys, cmd_info_values)))

        return paradigm_conf, commands_info

    @staticmethod
    def get_paradigm_conf_for_hox(matrix_dims, commands_info_hox=None):
        """Returns the paradigm configuration matrix for the Hex-O-Speller (HOX)
        or cake paradigms from the Berlin BCI Group [1]. This paradigm has 2
        levels of selection with 6 commands in each unit.

        Parameters
        ----------
        matrix_dims : list or np.array
            Array containing the dimensions of the matrices used in the
            experiment. For each matrix, the first position is the number of
            commands of the first level and the second the number of commands
            of the second level (typically both are 6). Shape [n_matrices x 2]
        commands_info_hox: list or None
            Array containing the dict info of each command. The first
            dimension are the matrices, the second dimension represent the
            units, and the third dimension contains the dictionary with the
            info of each command. Typically, this paradigm has 7 units of 6
            commands each. As defined by the Berlin BCI group: 1 menu matrix and
            6 matrix for each group of 6 letters. Therefore, with this
            setup, this array has shape [n_matrices x 7 x 6]

        Returns
        -------
        paradigm_conf : list
            Array with the paradigm configuration for an RCP paradigm
        commands_info : list
            Flattened version of commands_info input. It contains the command
            information corresponding to paradigm_conf. If input parameter
            commands_info is None, this output will be a skeleton with an empty
            dict for each command. If input commands_info is provided, it will
            be incorporated in the output

        References
        ----------
        [1] Blankertz, B., Dornhege, G., Krauledat, M., Schröder,
        M., Williamson, J., Murray-Smith, R., & Müller, K. R. (2006). The
        Berlin Brain-Computer Interface presents the novel mental typewriter
        Hex-o-Spell.
        """
        # Paradigm conf
        matrix_dims = np.array(matrix_dims)
        paradigm_conf = list()
        commands_info = list()
        for m in range(matrix_dims.shape[0]):
            # Useful variables
            n_cmd_l1 = matrix_dims[m, 0]
            n_cmd_l2 = matrix_dims[m, 0]
            units = list()
            commands_ids = list()
            # First unit (level 1)
            cmd_ids_u1 = list(range(n_cmd_l1))
            units.append([[[int(i)] for i in cmd_ids_u1]])
            commands_ids += cmd_ids_u1
            # Rest of units (level 2)
            for u in range(n_cmd_l1):
                cmd_ids_ux = list(range(
                    commands_ids[-1] + 1, commands_ids[-1] + 1 + n_cmd_l2
                ))
                units.append([[[int(i)] for i in cmd_ids_ux]])
                commands_ids += cmd_ids_ux
            paradigm_conf.append(units)
            # Commands info
            if commands_info_hox is None:
                cmd_info_values = [dict() for __ in commands_ids]
            else:
                cmd_info_values = np.array(commands_info_hox[m]).flatten()
                cmd_info_values = cmd_info_values.tolist()
            commands_info.append(dict(zip(commands_ids, cmd_info_values)))

        return paradigm_conf, commands_info

    def to_serializable_obj(self):
        rec_dict = self.__dict__
        for key in rec_dict.keys():
            if type(rec_dict[key]) == np.ndarray:
                rec_dict[key] = rec_dict[key].tolist()
        return rec_dict

    @classmethod
    def from_serializable_obj(cls, dict_data):
        return cls(**dict_data)
