"""Convert legacy (1.x) recordings to the 2.0 :class:`~medusa.core.data.recording.Recording`.

Migration helpers that read an old serialized recording (loaded with
:class:`medusa.core.legacy.recording.Recording`) and rebuild it on the 2.0 BIDS-aligned
data model. The 2.0 application containers some of them target live in
``medusa.pipelines`` (e.g. the VEP-speller :class:`SpellerData`), so those imports are
**local** to each function -- this compat module has no import-time dependency on the
higher ``pipelines`` layer.

:func:`recorder_recording_to_v2` covers plain *Recorder*-app runs (``.rec.*``: any
biosignals plus the manual ``marks`` annotations) and needs only ``medusa.core.data``;
:func:`cvep_recording_to_v2` and :func:`rcp_recording_to_v2` cover the BCI spellers.
"""

import warnings

import numpy as np

__all__ = ["recorder_recording_to_v2", "cvep_recording_to_v2",
           "rcp_recording_to_v2"]


def _find_attr(legacy_recording, registry_name: str, class_name: str):
    """Return the (single) legacy sub-object whose ``class_name`` matches, or raise."""
    registry = getattr(legacy_recording, registry_name, {})
    keys = [k for k, meta in registry.items() if meta.get("class_name") == class_name]
    if not keys:
        raise ValueError(
            f"legacy recording has no {class_name} in {registry_name} "
            f"(found: {[m.get('class_name') for m in registry.values()]}).")
    return getattr(legacy_recording, keys[0])


def _bids_label(value, fallback="01"):
    """Reduce a free-form legacy id to a valid BIDS label (``[0-9a-zA-Z]+``).

    Legacy ``subject_id``/``session_id`` are free-form (e.g. ``"Session_8"``), but a
    BIDS entity label admits only letters and digits. Strip everything else; fall back
    to ``fallback`` if nothing remains.
    """
    if value is None:
        return None
    label = "".join(ch for ch in str(value) if ch.isalnum())
    return label or fallback


def recorder_recording_to_v2(legacy_recording, *, task="rest", subject=None,
                             session=None, run=None, pair_conditions=True,
                             zero_time_origin=True, task_name=None):
    """Convert a legacy MEDUSA *Recorder* recording (``.rec.*``) to a 2.0 :class:`Recording`.

    The Recorder app writes plain biosignal runs -- resting-state EEG being the common
    case, but any streamed modality (ECG, EMG, EOG, ...) is supported -- together with
    an optional ``marks`` experiment holding the annotations logged by hand during
    acquisition: **conditions** (named block states, e.g. eyes-closed / eyes-open) and
    discrete **events** (blink / movement / noise). This maps

    - every biosignal in ``recording.biosignals`` to a
      :class:`~medusa.core.data.signal.Signal` (an ``EEG`` stream gets located sensors
      resolved from its labels, so topographic plotting works; every other modality gets
      generic typed channels);
    - the ``marks`` conditions and events to one BIDS
      :class:`~medusa.core.data.events.Events` timeline with columns ``onset`` /
      ``duration`` / ``trial_type`` (the condition/event *name*) / ``mark_type``
      (``"condition"`` or ``"event"``) / ``value`` (the raw integer label); and
    - the recorder ``app_settings`` (the name<->label dictionaries and the recording
      plan) plus acquisition metadata to ``Recording.experiment``.

    Conditions are logged as a **start mark and an end mark that share the same label**,
    so by default consecutive same-label condition marks are paired into a single event
    whose ``duration`` spans the block (e.g. one 300-s ``eyes-closed`` event). A mark left
    without a same-label partner is emitted as an instantaneous event (``duration`` 0)
    with a warning. Discrete events are always instantaneous. Set ``pair_conditions`` to
    ``False`` to emit every condition mark instantaneously instead.

    Parameters
    ----------
    legacy_recording : medusa.core.legacy.recording.Recording
        A recording loaded from a legacy ``.rec.*`` file.
    task : str, optional
        BIDS ``task`` label for the new recording (default ``"rest"``). A caller-chosen
        analysis label: it is validated, not sanitized -- an invalid value (spaces,
        ``_``, ``-``) raises. Letters and digits only.
    subject : str or None, optional
        Override the ``sub`` label. ``None`` (default) sanitizes the legacy
        ``subject_id`` to a valid BIDS label (falling back to ``"01"`` if nothing usable
        remains). Useful when the legacy id is wrong or when bulk-converting a study.
    session : str or int or None, optional
        BIDS ``ses`` label. ``None`` (default) uses the legacy ``session_id`` when
        present. Like ``subject``, an identity field: it is sanitized, not validated.
    run : str or int or None, optional
        Optional BIDS ``run`` index (validated, like ``task``).
    pair_conditions : bool, optional
        Pair consecutive same-label condition marks into interval events (default
        ``True``). ``False`` keeps every condition mark instantaneous.
    zero_time_origin : bool, optional
        Shift the time axis so the first recorded sample is ``t = 0`` and event onsets
        are relative to it (default ``True``, the BIDS convention). ``False`` keeps the
        original absolute (LSL) timestamps on both the signals and the events. Either
        way, the absolute first-sample timestamp is recorded in
        ``Recording.experiment["time_origin"]``.
    task_name : str or None, optional
        Human-readable ``TaskName`` written to every stream's sidecar. Defaults to
        ``task`` when ``None``.

    Returns
    -------
    medusa.core.data.recording.Recording
        The converted recording: one :class:`Signal` per biosignal, a manual-marks
        :class:`Events` timeline (``None`` when the file carries no marks), and the
        recorder metadata in ``experiment``.
    """
    from medusa.core.data import (Recording, BidsInfo, Signal, ChannelSet, Channel)

    bios = [(key, getattr(legacy_recording, key), meta.get("class_name"))
            for key, meta in getattr(legacy_recording, "biosignals", {}).items()]
    if not bios:
        raise ValueError("legacy recording has no biosignals to convert.")

    # -- Time origin: earliest first-sample across streams (shared LSL clock). --
    # ``abs_origin`` is the true absolute (LSL) start, always recorded in the
    # experiment; ``shift`` is what we actually subtract (0 unless rezeroing).
    firsts = [float(np.asarray(bio.times)[0])
              for _, bio, _ in bios
              if getattr(bio, "times", None) is not None
              and np.asarray(bio.times).size]
    abs_origin = min(firsts) if firsts else 0.0
    shift = abs_origin if zero_time_origin else 0.0

    # -- Biosignals -> Signals (EEG gets located sensors; others stay generic) --
    signals = {}
    for key, bio, class_name in bios:
        signal_arr = np.asarray(bio.signal)
        if signal_arr.ndim < 2:   # a 1-D single-channel stream -> [n_samples x 1]
            signal_arr = signal_arr.reshape(-1, 1)
        n_cha = signal_arr.shape[1]
        labels = getattr(getattr(bio, "channel_set", None), "l_cha", None)
        if not labels or len(labels) != n_cha:
            labels = [f"ch{i + 1}" for i in range(n_cha)]
        if str(class_name).upper() == "EEG":
            channel_set = ChannelSet().add_unipolar_eeg_channels(list(labels))
        else:
            channel_set = ChannelSet().add_channels(
                [Channel(lab, ch_type=str(class_name).upper(), unit="n/a")
                 for lab in labels])
        times = getattr(bio, "times", None)
        if times is not None and np.asarray(times).size:
            times = np.asarray(times, dtype=float) - shift
        else:
            times = None   # let Signal synthesize a regular axis from fs
        signals[key] = Signal(signal_arr, fs=float(bio.fs),
                              channel_set=channel_set, times=times)

    # -- Manual marks -> a single Events timeline ----------------------------
    marks = _find_marks(legacy_recording)
    events = _marks_to_events(marks, shift, pair_conditions) \
        if marks is not None else None

    # -- Assemble the 2.0 Recording ------------------------------------------
    # Identity fields (subject/session) are sanitized from possibly-messy legacy
    # values; subject always yields a label (fallback "01"). task/run are caller
    # analysis args and are validated (not sanitized) by BidsInfo.
    subject = _bids_label(legacy_recording.subject_id if subject is None
                          else subject) or "01"
    if session is None:
        session = getattr(legacy_recording, "session_id", None)
    bids = BidsInfo(subject=subject,
                    session=_bids_label(session, fallback=None) if session else None,
                    task=task, run=run)
    rec = Recording(bids)
    for key, signal in signals.items():
        rec.add_signal(key, signal)
    if events is not None:
        rec.set_events(events)
    rec.set_experiment(_recorder_experiment(legacy_recording, marks, abs_origin))
    rec.set_sidecar(TaskName=task_name if task_name is not None else task)
    return rec


def _find_marks(legacy_recording):
    """Return the legacy *marks* experiment object, or ``None`` if the run has none.

    The Recorder writes its annotations as a ``CustomExperimentData`` (usually keyed
    ``"marks"``). Rather than trust the key, this duck-types: it returns the first
    experiment exposing any of the four marks arrays.
    """
    marks_attrs = ("conditions_times", "conditions_labels",
                   "events_times", "events_labels")
    for key in getattr(legacy_recording, "experiments", {}):
        exp = getattr(legacy_recording, key)
        if any(hasattr(exp, attr) for attr in marks_attrs):
            return exp
    return None


def _invert_label_dict(spec):
    """``{name: {"label": int, ...}}`` -> ``{int_label: name}`` (recorder app_settings)."""
    out = {}
    for name, info in (spec or {}).items():
        if isinstance(info, dict) and "label" in info:
            try:
                out[int(info["label"])] = name
            except (TypeError, ValueError):
                continue   # non-integer label: fall back to the raw value as trial_type
    return out


def _pair_condition_marks(points, pair_conditions):
    """Turn condition ``(label, time)`` marks into ``(label, onset, duration)`` rows.

    With ``pair_conditions``, consecutive time-ordered marks that share a label are
    read as a block ``(start, end)`` and collapse to one row of ``duration = end -
    start``; a mark with no same-label successor stays instantaneous (``duration`` 0)
    and warns. Without it, every mark is instantaneous.
    """
    pts = sorted(points, key=lambda p: p[1])
    if not pair_conditions:
        return [(lab, t, 0.0) for lab, t in pts]
    rows, i = [], 0
    while i < len(pts):
        lab, start = pts[i]
        if i + 1 < len(pts) and pts[i + 1][0] == lab:
            rows.append((lab, start, pts[i + 1][1] - start))
            i += 2
        else:
            warnings.warn(
                f"condition mark {lab!r} at t={start:.3f} has no matching end mark; "
                f"emitting it as an instantaneous event (duration 0).")
            rows.append((lab, start, 0.0))
            i += 1
    return rows


def _marks_to_events(marks, shift, pair_conditions):
    """Build the BIDS :class:`Events` timeline from the recorder marks.

    ``shift`` (seconds) is subtracted from every absolute mark time so the onsets share
    the (possibly rezeroed) origin of the signals. The column descriptions
    (``events.json`` levels) are attached to the returned :class:`Events`. Returns
    ``None`` when there are no marks at all.
    """
    from medusa.core.data import Events

    cond_labels = list(getattr(marks, "conditions_labels", None) or [])
    cond_times = list(getattr(marks, "conditions_times", None) or [])
    ev_labels = list(getattr(marks, "events_labels", None) or [])
    ev_times = list(getattr(marks, "events_times", None) or [])
    if not cond_labels and not ev_labels:
        return None

    app_settings = getattr(marks, "app_settings", None) or {}
    cond_names = _invert_label_dict(app_settings.get("conditions"))
    ev_names = _invert_label_dict(app_settings.get("events"))

    records = []
    for lab, onset, dur in _pair_condition_marks(
            list(zip(cond_labels, cond_times)), pair_conditions):
        records.append({
            "onset": onset - shift, "duration": float(dur),
            "trial_type": cond_names.get(int(lab), str(lab)),
            "mark_type": "condition", "value": int(lab)})
    for lab, onset in zip(ev_labels, ev_times):
        records.append({
            "onset": onset - shift, "duration": 0.0,
            "trial_type": ev_names.get(int(lab), str(lab)),
            "mark_type": "event", "value": int(lab)})
    records.sort(key=lambda r: r["onset"])   # append() warns on out-of-order onsets

    levels = {}
    for spec in (app_settings.get("conditions"), app_settings.get("events")):
        for name, info in (spec or {}).items():
            levels.setdefault(name, (info or {}).get("desc-name", name)
                              if isinstance(info, dict) else name)
    descriptions = {
        "trial_type": {
            "Description": "Condition or discrete event marked during the recording.",
            **({"Levels": levels} if levels else {})},
        "mark_type": {
            "Description": "'condition' for a block state (paired start/end marks); "
                           "'event' for a discrete point event."},
        "value": {"Description": "Original integer label from the recorder app."},
    }

    events = Events(optional_columns={"trial_type": str, "mark_type": str,
                                      "value": "Int64"},
                    descriptions=descriptions)
    if records:
        events.append(records)
    return events


def _recorder_experiment(legacy_recording, marks, abs_origin):
    """Free-form ``Recording.experiment`` dict: recorder settings + acquisition metadata."""
    meta_keys = ("subject_id", "recording_id", "description", "source", "date",
                 "study_id", "session_id", "path")
    experiment = {
        "kind": "recorder-marks",
        "time_origin": abs_origin,
        "source_metadata": {k: getattr(legacy_recording, k, None)
                            for k in meta_keys},
    }
    if marks is not None and getattr(marks, "app_settings", None):
        experiment["app_settings"] = marks.app_settings
    return experiment


def cvep_recording_to_v2(legacy_recording, *, spell_target=None,
                         signal_key="eeg", task="cvep"):
    """Convert a legacy c-VEP speller recording to a 2.0 :class:`Recording`.

    Maps the legacy ``EEG`` biosignal to a :class:`~medusa.core.data.signal.Signal`, the
    ``CVEPSpellerData`` experiment to a VEP
    :class:`~medusa.pipelines.bci.vep_spellers.data.SpellerData` (the codebook, ``paradigm_conf``,
    per-trial available commands, targets) stored in ``Recording.experiment``, and the
    per-cycle stimulation ``onsets`` to a 2.0 :class:`~medusa.core.data.events.Events`
    timeline (one row per cycle, with ``trial_idx`` / ``cycle_idx``).

    Parameters
    ----------
    legacy_recording : medusa.core.legacy.recording.Recording
        A recording loaded from a legacy ``.cvep.*`` file.
    spell_target : list of str or None, optional
        Ground-truth target command ``uid`` per trial (``trial_idx`` order). ``None``
        auto-derives it **only** for calibration runs where every trial has a single
        available command (each trial's target is that command); for multi-command test
        runs the target is not stored in the file, so pass it (e.g. mapped from a labels
        file via each command's ``content``).
    signal_key : str, optional
        Key under which the EEG stream is stored in ``Recording.data`` (default ``"eeg"``).
    task : str, optional
        BIDS ``task`` label for the new recording (default ``"cvep"``).

    Returns
    -------
    medusa.core.data.recording.Recording
        The converted recording, with a :class:`SpellerData` experiment and a
        per-cycle events timeline -- ready for :mod:`medusa.pipelines.bci.vep_spellers`.
    """
    from medusa.core.data import (Recording, BidsInfo, Signal, ChannelSet, Events)
    from medusa.pipelines.bci.vep_spellers.data import SpellerData, CommandInfo

    eeg = _find_attr(legacy_recording, "biosignals", "EEG")
    exp = _find_attr(legacy_recording, "experiments", "CVEPSpellerData")

    # -- Signal (keep the absolute timestamps so the onsets align) -----------
    channel_set = ChannelSet().add_unipolar_eeg_channels(list(eeg.channel_set.l_cha))
    signal = Signal(np.asarray(eeg.signal), fs=float(eeg.fs), channel_set=channel_set,
                    times=np.asarray(eeg.times))

    # -- Codebook (single matrix): {uid: CommandInfo(code=sequence)} ---------
    matrix0 = exp.commands_info[0]
    commands_info = {
        str(uid): CommandInfo(
            uid=str(uid), code=list(info["sequence"]),
            content=info.get("text", info.get("label")),
            extra={k: info[k] for k in ("row", "col") if k in info})
        for uid, info in matrix0.items()
    }

    # -- Events: one row per stimulation cycle -------------------------------
    trial = np.asarray(exp.trial_idx, dtype=int)
    cycle = np.asarray(exp.cycle_idx, dtype=int)
    onsets = np.asarray(exp.onsets, dtype=float)
    events = Events(optional_columns={"trial_idx": "Int64", "cycle_idx": "Int64",
                                      "code_idx": "Int64"})
    events.append([{"onset": float(o), "duration": 0.0,
                    "trial_idx": int(t), "cycle_idx": int(c), "code_idx": 0}
                   for o, t, c in zip(onsets, trial, cycle)])

    # -- Per-trial available commands (from paradigm_conf[matrix][level][unit]) --
    m_idx = np.asarray(exp.matrix_idx, dtype=int)
    l_idx = np.asarray(exp.level_idx, dtype=int)
    u_idx = np.asarray(exp.unit_idx, dtype=int)
    n_trials = int(trial.max()) + 1 if len(trial) else 0
    available = []
    for t in range(n_trials):
        pos = int(np.where(trial == t)[0][0])
        cmds = exp.paradigm_conf[m_idx[pos]][l_idx[pos]][u_idx[pos]]
        available.append([str(c) for c in cmds])

    # -- Targets: auto-derive for single-command calibration runs ------------
    if spell_target is None and available and all(len(a) == 1 for a in available):
        spell_target = [a[0] for a in available]

    speller = SpellerData(
        mode=exp.mode, paradigm_conf=exp.paradigm_conf, commands_info=commands_info,
        fps_resolution=exp.fps_resolution, trial_available_cmmds=available,
        spell_target=spell_target,
        spell_result=(exp.spell_result if exp.spell_result else None))

    rec = Recording(BidsInfo(subject=str(legacy_recording.subject_id) or "legacy",
                             task=task))
    rec.add_signal(signal_key, signal).set_events(events)
    speller.to_recording(rec)
    return rec


def rcp_recording_to_v2(legacy_recording, *, spell_target=None, signal_key="eeg",
                        task="rcp", fps=60.0, t_stim=None):
    """Convert a legacy RCP / ERP speller recording to a 2.0 :class:`Recording`.

    Unlike c-VEP, an ERP/RCP speller has **no fixed periodic codebook**: it flashes groups
    (rows and columns) in a fresh random order every cycle. This rebuilds, per cycle, a
    per-command **display-frame code** ``(n_codes, n_frames)`` from the legacy per-flash
    ``onsets`` + ``group_idx`` / ``batch_idx`` + ``paradigm_conf`` (a command is lit at a
    flash ``(group g, batch b)`` iff it is in ``paradigm_conf[matrix][unit][g][b]``), and
    stores one events row per cycle carrying the ``code_idx`` that cycle presented.

    Legacy runs were not frame-locked (no stored ``t_stim`` / ``t_isi``), so the frame grid
    is a **best effort**: a frame is one ``1/fps`` display refresh; each flash is quantized
    to its nearest frame ``round((onset - cycle_onset) * fps)`` and held on for ``t_stim``
    (defaulting to one median inter-flash interval, i.e. the flash owns its whole slot). If
    you know the real highlight duration, pass ``t_stim`` (seconds).

    All legacy cycles have distinct flash orders, so ``n_codes == n_cycles`` (one code per
    cycle, ``code_idx`` running ``0, 1, 2, ...``); that is expected, not a bug.

    Parameters
    ----------
    legacy_recording : medusa.core.legacy.recording.Recording
        A recording loaded from a legacy ``.rcp.*`` file (an ``ERPSpellerData`` experiment).
    spell_target : list of str or None, optional
        Ground-truth target command ``uid`` per trial (``trial_idx`` order). ``None``
        derives it from the file's ``spell_target`` when present.
    signal_key : str, optional
        Key under which the EEG stream is stored (default ``"eeg"``).
    task : str, optional
        BIDS ``task`` label for the new recording (default ``"rcp"``).
    fps : float, optional
        Assumed display refresh rate for the frame grid (default ``60.0``).
    t_stim : float or None, optional
        Flash on-time in seconds; ``None`` (default) holds each flash on for its whole slot
        (one median inter-flash interval).

    Returns
    -------
    medusa.core.data.recording.Recording
        The converted recording, with a :class:`SpellerData` experiment (multi-code codes)
        and a per-cycle events timeline (``trial_idx`` / ``cycle_idx`` / ``code_idx``).
    """
    from medusa.core.data import (Recording, BidsInfo, Signal, ChannelSet, Events)
    from medusa.pipelines.bci.vep_spellers.data import SpellerData, CommandInfo

    eeg = _find_attr(legacy_recording, "biosignals", "EEG")
    exp = _find_attr(legacy_recording, "experiments", "ERPSpellerData")

    # -- Signal (keep the absolute timestamps so the onsets align) -----------
    channel_set = ChannelSet().add_unipolar_eeg_channels(list(eeg.channel_set.l_cha))
    signal = Signal(np.asarray(eeg.signal), fs=float(eeg.fs), channel_set=channel_set,
                    times=np.asarray(eeg.times))

    # -- Per-flash legacy arrays --------------------------------------------
    onsets = np.asarray(exp.onsets, dtype=float)
    trial = np.asarray(exp.trial_idx, dtype=int)
    seq = np.asarray(exp.sequence_idx, dtype=int)
    grp = np.asarray(exp.group_idx, dtype=int)
    bat = np.asarray(exp.batch_idx, dtype=int)
    mat = np.asarray(exp.matrix_idx, dtype=int)
    unit = np.asarray(exp.unit_idx, dtype=int)
    pconf = exp.paradigm_conf

    # -- Commands (single matrix): uid list + display content ---------------
    matrix0 = exp.commands_info[0]
    cmd_uids = [str(i) for i in sorted(int(k) for k in matrix0.keys())]
    cmd_row = {u: i for i, u in enumerate(cmd_uids)}
    n_cmd = len(cmd_uids)

    # -- Group flashes into cycles = (trial, sequence), ordered in time ------
    cycles = {}
    for i in range(len(onsets)):
        cycles.setdefault((int(trial[i]), int(seq[i])), []).append(i)
    cyc_keys = sorted(cycles)
    for k in cyc_keys:
        cycles[k].sort(key=lambda i: onsets[i])
    slot_counts = {len(v) for v in cycles.values()}
    if len(slot_counts) != 1:
        raise ValueError(
            f"cycles have varying flash counts {sorted(slot_counts)}; expected a constant "
            f"number of stimulation groups per sequence.")

    # -- Frame grid (best effort): quantize flashes to whole 1/fps frames ----
    within_dt = [d for k in cyc_keys for d in np.diff(onsets[cycles[k]])]
    soa = float(np.median(within_dt))
    slot_frames = max(1, int(round(soa * fps)))
    stim_frames = slot_frames if t_stim is None else max(1, int(round(t_stim * fps)))
    cyc_offsets = {k: np.round((onsets[cycles[k]] - onsets[cycles[k][0]]) * fps).astype(int)
                   for k in cyc_keys}
    n_frames = max(int(o.max()) for o in cyc_offsets.values()) + stim_frames

    # -- Per-cycle codes: (n_commands, n_cycles, n_frames) -------------------
    n_codes = len(cyc_keys)
    cycle_code_idx = {k: i for i, k in enumerate(cyc_keys)}
    codes = np.zeros((n_cmd, n_codes, n_frames), dtype=np.uint8)
    for code_i, k in enumerate(cyc_keys):
        for p, i in enumerate(cycles[k]):
            lit = pconf[int(mat[i])][int(unit[i])][int(grp[i])][int(bat[i])]
            f0 = int(cyc_offsets[k][p])
            for c in lit:
                codes[cmd_row[str(c)], code_i, f0:f0 + stim_frames] = 1

    commands_info = {
        uid: CommandInfo(uid=uid, code=codes[cmd_row[uid]],
                         content=matrix0[uid].get("text", matrix0[uid].get("label")),
                         content_type="text")
        for uid in cmd_uids
    }

    # -- Events: one row per cycle (onset = first flash of the cycle) --------
    events = Events(optional_columns={"trial_idx": "Int64", "cycle_idx": "Int64",
                                      "code_idx": "Int64"})
    events.append([{"onset": float(onsets[cycles[k][0]]), "duration": 0.0,
                    "trial_idx": int(k[0]), "cycle_idx": int(k[1]),
                    "code_idx": int(cycle_code_idx[k])}
                   for k in cyc_keys])

    # -- Targets / control state (per trial) --------------------------------
    if spell_target is None and getattr(exp, "spell_target", None):
        spell_target = [str(tr[0][1]) for tr in exp.spell_target]
    spell_result = ([str(tr[0][1]) for tr in exp.spell_result]
                    if getattr(exp, "spell_result", None) else None)

    speller = SpellerData(
        mode=exp.mode,
        paradigm_conf=pconf,
        commands_info=commands_info,
        fps_resolution=fps,
        trial_available_cmmds=None,
        spell_target=spell_target,
        spell_result=spell_result,
        control_state_target=_aslist(getattr(exp, "control_state_target", None)),
        control_state_result=_aslist(getattr(exp, "control_state_result", None)))

    rec = Recording(BidsInfo(
        subject=str(legacy_recording.subject_id) or "legacy",
        task=task))
    rec.add_signal(signal_key, signal).set_events(events)
    speller.to_recording(rec)
    return rec


def mi_recording_to_v2(legacy_recording, *, signal_key="eeg", task="mi"):
    """Convert a legacy motor-imagery recording to a 2.0 :class:`Recording`.

    Maps the legacy ``EEG`` biosignal to a :class:`~medusa.core.data.signal.Signal` (keeping
    the absolute timestamps so the cue onsets align) and the ``MIData`` experiment to a
    **trial timeline**: one :class:`~medusa.core.data.events.Events` row per motor-imagery
    trial, carrying ``trial_idx`` and the integer class ``label``. That is the shared trial
    contract of :mod:`~medusa.pipelines.bci.trial_events`, ready for
    :mod:`~medusa.pipelines.bci.motor_decoding`.

    Motor decoding needs **no data class** -- the trials are independent, and their timing and
    class live in the events. The remaining run metadata that the decoder never reads (the
    ``label`` -> class-name map, the run ``mode``, and the paradigm timings) is kept as a plain
    provenance ``dict`` in ``Recording.experiment`` so nothing is lost on conversion.

    Parameters
    ----------
    legacy_recording : medusa.core.legacy.recording.Recording
        A recording loaded from a legacy ``.mi.*`` file (an ``MIData`` experiment).
    signal_key : str, optional
        Key under which the EEG stream is stored (default ``"eeg"``).
    task : str, optional
        BIDS ``task`` label for the new recording (default ``"mi"``).

    Returns
    -------
    medusa.core.data.recording.Recording
        The converted recording: an EEG signal, a per-trial events timeline (``trial_idx`` /
        ``label``), and a plain-dict provenance experiment.
    """
    from medusa.core.data import (Recording, BidsInfo, Signal, ChannelSet, Events)

    eeg = _find_attr(legacy_recording, "biosignals", "EEG")
    exp = _find_attr(legacy_recording, "experiments", "MIData")

    # -- Signal (keep the absolute timestamps so the cue onsets align) -------
    channel_set = ChannelSet().add_unipolar_eeg_channels(list(eeg.channel_set.l_cha))
    signal = Signal(np.asarray(eeg.signal), fs=float(eeg.fs), channel_set=channel_set,
                    times=np.asarray(eeg.times))

    # -- Events: one row per MI trial (onset = cue, label = class) -----------
    onsets = np.asarray(exp.onsets, dtype=float)
    labels = np.asarray(exp.mi_labels) if getattr(exp, "mi_labels", None) is not None else None
    labelled = labels is not None and len(labels) == len(onsets)
    w = _aslist(getattr(exp, "w_trial_t", None))
    duration = (float(w[1]) - float(w[0])) / 1000.0 if w and len(w) == 2 else 0.0  # ms -> s
    events = Events(optional_columns={"trial_idx": "Int64", "label": "Int64"})
    events.append([
        {"onset": float(o), "duration": duration, "trial_idx": int(t),
         **({"label": int(labels[t])} if labelled else {})}
        for t, o in enumerate(onsets)])

    # -- Provenance (a plain dict; the decoder never reads it) ---------------
    label_names = {int(k): v for k, v in (getattr(exp, "mi_labels_info", None) or {}).items()}
    experiment = {
        "paradigm": "motor_imagery",
        "mode": getattr(exp, "mode", None),
        "label_names": label_names,
        "w_trial_t": w,
        "paradigm_info": getattr(exp, "paradigm_info", None),
    }

    rec = Recording(BidsInfo(subject=str(legacy_recording.subject_id) or "legacy", task=task))
    rec.add_signal(signal_key, signal).set_events(events)
    rec.set_experiment(experiment)
    return rec


def _aslist(value):
    """None-safe ndarray/list -> plain list (for the ``.mat``-safe SpellerData fields)."""
    if value is None:
        return None
    return np.asarray(value).tolist()
