"""Convert legacy (1.x) recordings to the 2.0 :class:`~medusa.core.data.recording.Recording`.

Migration helpers that read an old serialized recording (loaded with
:class:`medusa.core.legacy.recording.Recording`) and rebuild it on the 2.0 BIDS-aligned
data model. The 2.0 application containers some of them target live in
``medusa.pipelines`` (e.g. the VEP-speller :class:`SpellerData`), so those imports are
**local** to each function -- this compat module has no import-time dependency on the
higher ``pipelines`` layer.

:func:`recorder_recording_to_v2` covers plain *Recorder*-app runs (``.rec.*``: any
biosignals plus the manual ``marks`` annotations) and needs only ``medusa.core.data``;
:func:`cvep_recording_to_v2` and :func:`rcp_recording_to_v2` cover the BCI spellers;
:func:`mi_recording_to_v2` and :func:`edubiomat_recording_to_v2` cover the trial-based
runs, which need no data class at all (their trials live in the events timeline).
"""

import warnings

import numpy as np

__all__ = ["recorder_recording_to_v2", "cvep_recording_to_v2",
           "rcp_recording_to_v2", "mi_recording_to_v2",
           "edubiomat_recording_to_v2", "EDUBIOMAT_TASK_LABELS",
           "EDUBIOMAT_EVENT_COLUMNS"]


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


def _legacy_biosignals(legacy_recording):
    """``[(key, biosignal, class_name)]`` for every stream in the legacy file."""
    bios = [(key, getattr(legacy_recording, key), meta.get("class_name"))
            for key, meta in getattr(legacy_recording, "biosignals", {}).items()]
    if not bios:
        raise ValueError("legacy recording has no biosignals to convert.")
    return bios


def _time_origin(bios):
    """First-sample timestamp of the run: the earliest across streams.

    All streams of a legacy recording are timestamped on the same LSL clock, so the
    earliest first sample is the absolute start of the run.
    """
    firsts = [float(np.asarray(bio.times)[0])
              for _, bio, _ in bios
              if getattr(bio, "times", None) is not None
              and np.asarray(bio.times).size]
    return min(firsts) if firsts else 0.0


def _biosignals_to_signals(bios, shift=0.0):
    """Every legacy biosignal -> a 2.0 :class:`Signal`, keyed as in the legacy file.

    An ``EEG`` stream gets located sensors resolved from its labels, so topographic
    plotting works; every other modality gets generic typed channels. ``shift``
    (seconds) is subtracted from the absolute (LSL) timestamps: pass the run origin
    to rezero the time axis, or leave it at 0 to keep the original clock (which is
    what a paradigm whose event onsets are absolute timestamps needs).
    """
    from medusa.core.data import (Signal, ChannelSet, Channel, BIDS_CHANNEL_TYPES)

    signals = {}
    for key, bio, class_name in bios:
        signal_arr = np.asarray(bio.signal)
        if signal_arr.ndim < 2:   # a 1-D single-channel stream -> [n_samples x 1]
            signal_arr = signal_arr.reshape(-1, 1)
        n_cha = signal_arr.shape[1]
        labels = getattr(getattr(bio, "channel_set", None), "l_cha", None)
        if not labels or len(labels) != n_cha:
            labels = [f"ch{i + 1}" for i in range(n_cha)]
        # The 1.x class name already *is* the BIDS type for a typed modality
        # (EEG/ECG/...), but the Recorder writes every untyped stream as a free-form
        # CustomBiosignalData keyed by its modality instead (GSR, PPG, ...), so read
        # the type off the key when the class name says nothing.
        ch_type = next((str(c).upper() for c in (class_name, key)
                        if c and str(c).upper() in BIDS_CHANNEL_TYPES), "OTHER")
        if ch_type == "EEG":
            channel_set = ChannelSet().add_unipolar_eeg_channels(list(labels))
        else:
            channel_set = ChannelSet().add_channels(
                [Channel(lab, ch_type=ch_type, unit="n/a") for lab in labels])
        times = getattr(bio, "times", None)
        if times is not None and np.asarray(times).size:
            times = np.asarray(times, dtype=float) - shift
        else:
            times = None   # let Signal synthesize a regular axis from fs
        signals[key] = Signal(signal_arr, fs=float(bio.fs),
                              channel_set=channel_set, times=times)
    return signals


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
    from medusa.core.data import Recording, BidsInfo

    bios = _legacy_biosignals(legacy_recording)

    # -- Time origin: earliest first-sample across streams (shared LSL clock). --
    # ``abs_origin`` is the true absolute (LSL) start, always recorded in the
    # experiment; ``shift`` is what we actually subtract (0 unless rezeroing).
    abs_origin = _time_origin(bios)
    shift = abs_origin if zero_time_origin else 0.0
    signals = _biosignals_to_signals(bios, shift)

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


#: BIDS ``task`` label a run of each edubiomat test mode gets when the caller passes
#: none. Its keys are the two test modes, as ``app_settings["test"]`` spells them, and
#: they are the only modes this converter accepts.
#:
#: The mode belongs in ``task``, not in ``acq``: the two modes show different stimuli,
#: trial by trial, and even fill different event columns, which is what a BIDS *task* is;
#: ``acq`` marks a change of acquisition parameters for the same task, and this kernel
#: already spends it on the data key (:meth:`Recording.bids_basename`), which would
#: override a mode put there. The labels are camelCase because a BIDS label holds letters
#: and digits only, so there is no separator to put between the two words.
EDUBIOMAT_TASK_LABELS = {
    "images": "edubiomatImages",
    "questions": "edubiomatQuestions",
}

#: Columns of the events timeline :func:`edubiomat_recording_to_v2` writes, whatever the
#: mode: one row per trial. A column the mode has no value for is ``n/a`` on every row (an
#: image run has no ``correct`` or ``feedback_onset``; a question run has no ``label``), so
#: the two modes share one schema and their timelines stay comparable.
EDUBIOMAT_EVENT_COLUMNS = {
    "trial_idx": "Int64",
    "trial_type": str,
    "stim_id": str,
    "label": "Int64",
    "response": str,
    "response_onset": "float64",
    "correct": "Int64",
    "feedback_onset": "float64",
}


def edubiomat_recording_to_v2(legacy_recording, *, task=None, subject=None,
                              session=None, run=None):
    """Convert a legacy *edubiomat* recording (``.rec.*``) to a 2.0 :class:`Recording`.

    The edubiomat app runs an educational test while the EEG is recorded. It has two test
    modes, and both are trial-based: one stimulus per trial, one response per trial.

    * ``"images"`` shows an image for ``t_sti`` ms and then takes a response. The image
      file name may carry the image's class (``8-0.png`` is image 8 of class 0), which the
      app stores in ``img_class``.
    * ``"questions"`` asks a question, takes the answer the subject chose, and then shows
      whether it was right.

    Like motor imagery, this paradigm needs **no data class**: the trials are independent,
    and everything time-locked lives in the events. So this maps

    - every biosignal in ``recording.biosignals`` to a
      :class:`~medusa.core.data.signal.Signal`, keeping the absolute timestamps so the
      trial onsets align with the samples (the app streams EEG together with the
      autonomic modalities that carry the stress response -- ECG, GSR, PPG);
    - every trial to one :class:`~medusa.core.data.events.Events` row
      (:data:`EDUBIOMAT_EVENT_COLUMNS`), with the stimulus onset as ``onset`` and the
      configured ``t_sti`` as ``duration``; and
    - the app settings and the raw per-trial records to a plain provenance ``dict`` in
      ``Recording.experiment``, so nothing the columns leave out is lost.

    An ``images`` run comes out as a valid trial-decoding recording in the sense of
    :mod:`~medusa.pipelines.bci.trial_events` (``trial_idx`` + ``label``), so the image
    class can be decoded from the EEG straight away -- as long as the file names carry a
    class. A ``questions`` run has no integer class, so its ``label`` is ``n/a``
    throughout; what it has instead is ``correct`` and ``feedback_onset``, which is what a
    feedback-locked analysis needs.

    Three moments of a trial are time-locked, and all three keep the clock of ``onset``:
    the stimulus (``onset``), the response (``response_onset``) and, for a question, the
    feedback (``feedback_onset``). ``duration`` is the **configured** stimulus time from
    ``app_settings["t_sti"]``, not a measured one: the app does not record when the
    stimulus actually went away.

    Parameters
    ----------
    legacy_recording : medusa.core.legacy.recording.Recording
        A recording loaded from a legacy edubiomat ``.rec.*`` file. Its experiment is a
        generic ``CustomExperimentData``, so it is found by its ``app_settings["test"]``
        rather than by a class name.
    task : str or None, optional
        BIDS ``task`` label for the new recording. ``None`` (default) takes the label of
        the mode the file holds from :data:`EDUBIOMAT_TASK_LABELS`, so an image run and a
        question run never collide in one dataset. Letters and digits only; it is
        validated, not sanitized.
    subject : str or None, optional
        Override the ``sub`` label. ``None`` (default) sanitizes the legacy ``subject_id``
        to a valid BIDS label, falling back to ``"01"`` when nothing usable remains -- as
        it does for a run recorded without a subject id.
    session : str or int or None, optional
        BIDS ``ses`` label. ``None`` (default) uses the legacy ``session_id`` when the file
        has one. Like ``subject``, it is sanitized, not validated.
    run : str or int or None, optional
        Optional BIDS ``run`` index (validated, like ``task``).

    Returns
    -------
    medusa.core.data.recording.Recording
        The converted recording: one :class:`Signal` per biosignal (keyed as in the
        legacy file), a per-trial events timeline, and a plain-dict provenance
        experiment.

    Raises
    ------
    ValueError
        If the file holds no biosignals, or no experiment whose ``app_settings["test"]``
        names an edubiomat mode.
    """
    from medusa.core.data import Recording, BidsInfo, Events

    exp = _find_edubiomat_experiment(legacy_recording)
    settings = dict(getattr(exp, "app_settings", None) or {})
    mode = settings.get("test")
    trials = list(getattr(exp, "data", None) or [])

    # -- Signals (keep the absolute timestamps: the trial onsets are on that clock) --
    signals = _biosignals_to_signals(_legacy_biosignals(legacy_recording))

    # -- Events: one row per trial (onset = stimulus, duration = configured t_sti) --
    duration = float(settings.get("t_sti") or 0.0) / 1000.0   # ms -> s
    # An explicit map, not an if/else: a mode added to EDUBIOMAT_TASK_LABELS without a
    # row builder then raises here instead of quietly being read as the other mode.
    build_row = {"images": _image_trial_row, "questions": _question_trial_row}[mode]
    records = [build_row(trial, i, duration) for i, trial in enumerate(trials)]
    records.sort(key=lambda r: r["onset"])   # append() warns on out-of-order onsets
    events = Events(optional_columns=dict(EDUBIOMAT_EVENT_COLUMNS),
                    descriptions=_edubiomat_event_descriptions(mode))

    initial_time = signals['eeg'].times[0]
    for record in records:
        record['onset'] = record['onset'] - initial_time
        record['response_onset'] = record['response_onset'] - initial_time

    if records:
        events.append(records)

    # -- Provenance (a plain dict; keeping the raw trials loses nothing) -----
    experiment = {
        "paradigm": "edubiomat",
        "mode": mode,
        "app_settings": settings,
        "trials": trials,
    }

    # Identity fields are sanitized from possibly-empty legacy values (an edubiomat run is
    # often recorded with no subject id at all); task/run are caller args and are validated.
    subject = _bids_label(legacy_recording.subject_id if subject is None
                          else subject) or "01"
    task = EDUBIOMAT_TASK_LABELS[mode] if task is None else task
    if session is None:
        session = getattr(legacy_recording, "session_id", None)
    rec = Recording(BidsInfo(
        subject=subject,
        session=_bids_label(session, fallback=None) if session else None,
        task=task, run=run))
    for key, signal in signals.items():
        rec.add_signal(key, signal)
    rec.set_events(events)
    rec.set_experiment(experiment)
    rec.set_sidecar(TaskName=f"edubiomat {mode}")
    return rec


def _find_edubiomat_experiment(legacy_recording):
    """Return the legacy edubiomat experiment object, or raise.

    The app writes a generic ``CustomExperimentData``, so its class name says nothing about
    which app produced it. This duck-types instead: the edubiomat experiment is the one
    whose ``app_settings["test"]`` names a mode and that carries the per-trial ``data``.
    """
    seen = []
    for key in getattr(legacy_recording, "experiments", {}):
        exp = getattr(legacy_recording, key)
        settings = getattr(exp, "app_settings", None) or {}
        test = settings.get("test") if isinstance(settings, dict) else None
        if hasattr(exp, "data") and test in EDUBIOMAT_TASK_LABELS:
            return exp
        seen.append(test)
    raise ValueError(
        f"legacy recording has no edubiomat experiment: none of its experiments has a "
        f"'data' list and an app_settings['test'] in {list(EDUBIOMAT_TASK_LABELS)} "
        f"(found tests: {seen}).")


def _image_trial_row(trial, index, duration):
    """The events record of one ``images`` trial."""
    return {
        "onset": float(trial["onset_time"]),
        "duration": duration,
        "trial_idx": int(trial.get("trial", index)),
        "trial_type": "image",
        "stim_id": _file_name(trial.get("img_path")),
        "label": _or_none(trial.get("img_class"), int),
        "response": _response_text(trial.get("response")),
        "response_onset": _or_none(trial.get("response_time"), float),
        "correct": None,         # the file does not say whether a response was right
        "feedback_onset": None,  # this mode shows no feedback
    }


def _question_trial_row(trial, index, duration):
    """The events record of one ``questions`` trial."""
    question = trial.get("question") or {}
    return {
        "onset": float(trial["onset_time"]),
        "duration": duration,
        "trial_idx": int(trial.get("trial", index)),
        "trial_type": "question",
        "stim_id": question.get("question-text"),
        "label": None,           # a question has no integer class
        "response": _response_text(trial.get("answer")),
        "response_onset": _or_none(trial.get("answer_time"), float),
        "correct": _or_none(trial.get("feedback"), int),
        "feedback_onset": _or_none(trial.get("feedback_time"), float),
    }


def _edubiomat_event_descriptions(mode):
    """``events.json`` column descriptions, written for the mode that produced them."""
    images = mode == "images"
    return {
        "trial_idx": {"Description": "Trial number within the run."},
        "trial_type": {
            "Description": "Kind of stimulus the trial showed.",
            "Levels": {"image": "An image was shown.",
                       "question": "A question was asked."}},
        "stim_id": {"Description": "Image file name shown on this trial." if images else
                    "Text of the question asked on this trial."},
        "label": {"Description":
                  "Class of the image, as its file name carries it; n/a when the name "
                  "carries none." if images else
                  "Not used by the questions mode: a question has no integer class."},
        "response": {"Description":
                     "Response the subject gave, as the app recorded it (1 or -1)."
                     if images else
                     "Text of the answer option the subject chose."},
        "response_onset": {
            "Description": "Time the subject responded, on the same clock as onset; "
                           "n/a when the trial got no response.",
            "Units": "s"},
        "correct": {"Description":
                    "Not used by the images mode: the file does not record whether a "
                    "response was right." if images else
                    "1 when the chosen answer was the correct one, 0 when it was not."},
        "feedback_onset": {
            "Description": "Not used by the images mode: it shows no feedback."
                           if images else
                           "Time the feedback was shown, on the same clock as onset.",
            "Units": "s"},
    }


def _file_name(path):
    """File name of a legacy path, whatever separator it was written with; ``None``-safe."""
    if not path:
        return None
    return str(path).replace("\\", "/").rsplit("/", 1)[-1]


def _response_text(value):
    """Render a legacy response as the text the ``response`` column holds.

    One string column carries both modes: the questions mode answers with the text of an
    option, the images mode with a number. An integral float loses its ``.0`` on the way,
    so a response reads as ``1`` / ``-1`` rather than ``1.0`` / ``-1.0``.
    """
    if value is None:
        return None
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _or_none(value, cast):
    """``cast(value)``, or ``None`` when the legacy field is absent or null (-> ``n/a``)."""
    return None if value is None else cast(value)


def _aslist(value):
    """None-safe ndarray/list -> plain list (for the ``.mat``-safe SpellerData fields)."""
    if value is None:
        return None
    return np.asarray(value).tolist()
