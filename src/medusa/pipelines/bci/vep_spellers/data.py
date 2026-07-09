"""Application data model for VEP spellers.

A speller needs a little run metadata that BIDS does not cover. It lives in
:class:`SpellerData`, stored in ``Recording.experiment``. This is the *codebook* (one
:class:`CommandInfo` per command, each carrying its per-frame ``code``), the paradigm
and scenario configuration (``paradigm_conf``: speller layout and stimuli), the display
refresh rate, the *available commands* per trial, and the spelling and control-state
targets and results.

Everything that is time-locked lives in ``Recording.events`` instead. There is **one
row per stimulation cycle**. Its ``onset`` marks the start of the cycle, and its
``code_idx`` says which code that cycle showed (see :data:`SPELLER_EVENT_COLUMNS` and
:func:`validate_speller_events`). The per-frame timing comes from the codebook and
``fps_resolution``. So nothing here repeats the
:class:`~medusa.core.data.signal.Signal` or the
:class:`~medusa.core.data.events.Events`.

Every command has a code of shape ``(n_codes, n_frames)``: one row per selectable code,
each a per-frame stimulation pattern, where a frame is one display refresh. There is
only **one** representation. Single-code paradigms (c-VEP, SSVEP) have ``n_codes == 1``
and ``code_idx == 0``. A multi-code paradigm (an RCP/P300 speller that re-randomises its
flash order each cycle) stores one code per order and picks it per cycle with
``code_idx``.
"""

from dataclasses import dataclass, field, asdict

import numpy as np

from medusa.core.serialization import SerializableComponent, ensure_list
from medusa.core.data.recording import Recording
from medusa.core.data.events import Events

__all__ = [
    "CommandInfo",
    "SpellerData",
    "SPELLER_EVENT_COLUMNS",
    "validate_speller_events",
]

#: The required ``Events`` columns for VEP spellers, as pandas nullable ``Int64``.
#: There is **one stimulation event (row) per cycle**. Its ``onset`` marks the start of
#: that codebook cycle. The per-frame timing inside the cycle comes from the code length
#: and :attr:`SpellerData.fps_resolution` (frame ``f`` is at ``onset + f /
#: fps_resolution``). ``code_idx`` is the row of each command's ``(n_codes, n_frames)``
#: code that this cycle showed (always ``0`` for single-code paradigms: c-VEP, SSVEP).
SPELLER_EVENT_COLUMNS = {
    "trial_idx": "Int64",       # selection (one spelled symbol)
    "cycle_idx": "Int64",       # repetition index of the codebook within a trial
    "code_idx": "Int64",        # which code (0 for single-code paradigms) the cycle showed
}


def _ensure_2d_code(code, uid=""):
    """Coerce a command ``code`` to a rectangular 2-D ``(n_codes, n_frames)`` list.

    A 1-D code is treated as one code and wrapped to ``(1, n_frames)``, so there is only
    ever one internal shape. (A ``.mat`` round-trip can squeeze a single-code command's
    code down to 1-D; this handles that case too.) A ragged or higher-rank code is
    rejected.
    """
    arr = np.asarray(code)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(
            f"command {uid!r} code must be 1-D or 2-D (n_codes, n_frames), got shape "
            f"{arr.shape}.")
    if arr.dtype == object:
        raise ValueError(
            f"command {uid!r} code is ragged; every code must have equal length.")
    return arr.tolist()


@dataclass
class CommandInfo:
    """One speller command (paradigm-agnostic), with its codebook rows.

    ``code`` is **always** 2-D ``(n_codes, n_frames)``: one row per selectable code, each
    a per-frame stimulation pattern (a frame is one display refresh). Single-code
    paradigms (c-VEP, SSVEP) have ``n_codes == 1``. A multi-code paradigm (an RCP/P300
    speller that re-randomises its flash order each cycle) stores one code per order. A
    1-D ``code`` is coerced to ``(1, n_frames)`` at construction, so callers never have to
    treat single-code and multi-code apart.

    ``content`` and ``content_type`` describe what is shown (for example the letter
    ``"A"``). They default to ``None`` because the codebook generators do not know them;
    the application fills them in. Paradigm-specific fields (for example ``{'row': 0,
    'col': 2}`` or ``{'stim_freq': 12.0}``) live in ``extra``.
    """

    uid: str
    code: list
    content: "str | None" = None
    content_type: "str | None" = None
    layout_pos: object = None
    action: "str | None" = None
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        self.code = _ensure_2d_code(self.code, self.uid)

    def to_record(self) -> dict:
        """Return this command as a plain ``dict`` (``.mat``-safe: a record, not an int-keyed struct)."""
        rec = asdict(self)
        rec["code"] = np.asarray(self.code).tolist()    # list of lists (2-D)
        return rec

    @classmethod
    def from_record(cls, record: dict) -> "CommandInfo":
        """Rebuild a :class:`CommandInfo` from a plain record dict (re-coerces ``code`` to 2-D)."""
        return cls(**dict(record))                       # __post_init__ re-coerces to 2-D


class SpellerData(SerializableComponent):
    """VEP-speller run metadata for ``Recording.experiment``.

    Holds **no** signal, onsets or per-event index arrays -- those live in the
    ``Recording``'s :class:`~medusa.core.data.signal.Signal` and
    :class:`~medusa.core.data.events.Events`.

    Parameters
    ----------
    mode :
        ``"train"`` or ``"test"``.
    paradigm_conf :
        Free-form scenario and GUI configuration shown to the user (command layout,
        stimuli, and so on), built from simple types. Decoding never reads it. The only
        thing decoding needs about the layout is ``trial_available_cmmds``.
    commands_info :
        The codebook: ``{uid: CommandInfo}`` (each command carries its per-frame
        ``code``). Saved as a list of records (``.mat``-safe).
    fps_resolution :
        Display refresh rate (Hz) the codebook was built for. Together with the per-cycle
        event onsets, it fixes the per-frame timing inside a cycle (frame ``f`` is at
        ``onset + f / fps_resolution``).
    trial_available_cmmds :
        Per-trial list of the command ``uid``\\ s the user could select, as
        ``list[list[str]]`` indexed by ``trial_idx``. The speller GUI sets it at the start
        of each trial, from the layout and the actions taken so far. Command decoding then
        limits each trial's choice to its available commands. This is how multi-matrix and
        dynamic paradigms work, with no matrix bookkeeping in the decoder. ``None`` means
        every command is available in every trial.
    spell_target, spell_result :
        Target and decoded command per trial (``uid``\\ s), when available.
    control_state_target, control_state_result :
        Target and decoded control state per trial, when the paradigm uses one.

    Raises
    ------
    ValueError
        If ``mode`` is not ``"train"`` or ``"test"``.
    """

    def __init__(self, *, mode, paradigm_conf, commands_info, fps_resolution,
                 trial_available_cmmds=None, spell_target=None, spell_result=None,
                 control_state_target=None, control_state_result=None):
        if mode not in ("train", "test"):
            raise ValueError(f"mode must be 'train' or 'test', got {mode!r}.")
        self.mode = mode
        self.paradigm_conf = paradigm_conf
        self.commands_info = commands_info
        self.fps_resolution = fps_resolution
        self.trial_available_cmmds = trial_available_cmmds
        self.spell_target = spell_target
        self.spell_result = spell_result
        self.control_state_target = control_state_target
        self.control_state_result = control_state_result

    def __repr__(self) -> str:
        return (f"SpellerData(mode={self.mode!r}, "
                f"n_commands={len(self.commands_info)}, "
                f"fps_resolution={self.fps_resolution})")

    # ------------------------------------------------------------------
    # SerializableComponent contract  (commands_info <-> list-of-records)
    # ------------------------------------------------------------------
    def to_serializable_obj(self) -> dict:
        """Convert to a plain, ``.mat``-safe dict (``commands_info`` becomes a list of records)."""
        return {
            "mode": self.mode,
            "paradigm_conf": self.paradigm_conf,
            "commands_info": [c.to_record() for c in self.commands_info.values()],
            "fps_resolution": self.fps_resolution,
            # Flat (trial_idx, uid) records: a nested list[list[str]] does not survive
            # the .mat round-trip, but a flat list of records does (like commands_info).
            "trial_available_cmmds": (
                None if self.trial_available_cmmds is None else
                [{"trial_idx": t, "uid": str(u)}
                 for t, uids in enumerate(self.trial_available_cmmds) for u in uids]),
            "spell_target": _listify(self.spell_target),
            "spell_result": _listify(self.spell_result),
            "control_state_target": _listify(self.control_state_target),
            "control_state_result": _listify(self.control_state_result),
        }

    @classmethod
    def from_serializable_obj(cls, data: dict) -> "SpellerData":
        """Rebuild a :class:`SpellerData` from its serialisable dict.

        Regroups the flat command and trial records back into their nested form.
        """
        data = dict(data)
        # A single-command run round-trips through .mat with the list squeezed away.
        records = [CommandInfo.from_record(r)
                   for r in ensure_list(data.get("commands_info") or [])]
        commands_info = {c.uid: c for c in records}
        available = data.get("trial_available_cmmds")
        if available is not None:
            # Regroup the flat (trial_idx, uid) records into list[list[str]].
            by_trial: "dict[int, list[str]]" = {}
            for r in ensure_list(available):
                by_trial.setdefault(int(r["trial_idx"]), []).append(str(r["uid"]))
            n_trials = max(by_trial) + 1 if by_trial else 0
            available = [by_trial.get(t, []) for t in range(n_trials)]
        return cls(
            mode=data.get("mode"),
            paradigm_conf=data.get("paradigm_conf"),
            commands_info=commands_info,
            fps_resolution=data.get("fps_resolution"),
            trial_available_cmmds=available,
            spell_target=data.get("spell_target"),
            spell_result=data.get("spell_result"),
            control_state_target=data.get("control_state_target"),
            control_state_result=data.get("control_state_result"),
        )

    # ------------------------------------------------------------------
    # Recording bridge
    # ------------------------------------------------------------------
    @classmethod
    def from_recording(cls, recording: Recording) -> "SpellerData":
        """Return the recording's :class:`SpellerData`; raise ``TypeError`` if it is not one."""
        exp = recording.experiment
        if not isinstance(exp, cls):
            raise TypeError(
                f"recording.experiment is {type(exp).__name__}, not {cls.__name__}.")
        return exp

    def to_recording(self, recording: Recording) -> Recording:
        """Attach ``self`` as ``recording.experiment`` and return the recording."""
        return recording.set_experiment(self)

    # ------------------------------------------------------------------
    # Codebook convenience
    # ------------------------------------------------------------------
    @property
    def codes(self) -> np.ndarray:
        """The full ``(n_commands, n_codes, n_frames)`` code tensor (in row order).

        Every command's ``code`` is 2-D ``(n_codes, n_frames)``, so stacking them is 3-D.
        Decoding indexes it per cycle as ``codes[:, code_idx, :]`` (c-VEP and SSVEP simply
        have ``n_codes == 1`` and ``code_idx == 0``).
        """
        arr = np.array([np.asarray(c.code) for c in self.commands_info.values()])
        if arr.dtype == object:
            raise ValueError(
                "commands have inconsistent code shapes; every command needs the same "
                "(n_codes, n_frames).")
        return arr

    @property
    def codebook(self) -> np.ndarray:
        """The ``(n_commands, n_frames)`` first-code matrix (``codes[:, 0, :]``).

        A convenience view: the single code of c-VEP or SSVEP, or an RCP cycle-0 snapshot.
        Handy for :func:`~medusa.pipelines.bci.vep_spellers.encoding.plot_codebook` and for
        inspection. Decoding uses :attr:`codes`.
        """
        return self.codes[:, 0, :]

    @property
    def command_uids(self) -> "list[str]":
        """Command ``uid``\\ s in codebook-row order."""
        return list(self.commands_info)


def _listify(value):
    """None-safe ndarray/list -> list for serialisation."""
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    return list(value) if isinstance(value, (list, tuple)) else value


def validate_speller_events(events: Events) -> None:
    """Check that ``events`` is a usable VEP-speller timeline.

    The events are the stimulation-cycle timeline: one row per cycle (its ``onset`` marks
    the cycle start). All of :data:`SPELLER_EVENT_COLUMNS` (``trial_idx``, ``cycle_idx``,
    ``code_idx``) are required. A cycle row is a row with a non-null ``cycle_idx``. The
    decoder ignores any other rows in a mixed timeline.

    Raises
    ------
    ValueError
        If ``events`` is ``None``, a required column is missing, or there is no
        stimulation cycle (every ``cycle_idx`` is null).
    """
    if events is None:
        raise ValueError("recording has no events timeline.")
    cols = set(events.column_names)
    missing = [c for c in SPELLER_EVENT_COLUMNS if c not in cols]
    if missing:
        raise ValueError(
            f"events lack mandatory speller columns {missing}; required: "
            f"{list(SPELLER_EVENT_COLUMNS)}.")
    if not events.df["cycle_idx"].notna().any():
        raise ValueError("events contain no stimulation cycles (all cycle_idx are n/a).")
