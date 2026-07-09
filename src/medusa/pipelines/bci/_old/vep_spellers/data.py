from dataclasses import dataclass, field

import numpy as np


@dataclass
class CommandInfo:
    """Base description of a speller command (paradigm-agnostic).

    ``content`` / ``content_type`` describe what is shown for the command (e.g.
    the letter ``'A'``); they default to ``None`` because the codebook
    generators do not know them — the application fills them in. Paradigm-
    specific fields live on the subclasses below.
    """

    uid: str
    code: list
    content: str = None
    content_type: str = None
    layout_pos: object = None
    action: str = None
    # Paradigm-specific attributes, e.g. {'row': 0, 'col': 2} for a row-col
    # speller or {'freq': 12.0} for a frequency-coded speller.
    extra: dict = field(default_factory=dict)


class SpellerData:

    def __init__(
            self,
            mode,
            paradigm_conf,
            commands_info,
            onsets,
            cycle_idx,
            trial_idx,
            trial_available_cmmds,
            control_result,
            spell_result,
            fps_resolution,
            control_target=None,
            spell_target=None,
            **kwargs
    ):

        # Check errors
        if mode not in ('train', 'test'):
            raise ValueError('Unknown mode. Possible values {train, test}')

        # Standard attributes
        self.mode = mode
        self.paradigm_conf = paradigm_conf
        self.commands_info = commands_info
        self.onsets = onsets
        self.cycle_idx = cycle_idx
        self.trial_idx = trial_idx
        self.trial_available_cmmds = trial_available_cmmds
        self.control_result = control_result
        self.spell_result = spell_result
        self.fps_resolution = fps_resolution
        self.control_target = control_target
        self.spell_target = spell_target

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

















