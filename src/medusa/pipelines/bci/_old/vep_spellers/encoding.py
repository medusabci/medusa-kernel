import warnings

import numpy as np

from .data import CommandInfo


def _seconds_to_frames(t_seconds, fps_resolution, param_name):
    """Quantize a duration in seconds to a whole number of display frames.

    A monitor can only update its content at frame boundaries, so any
    requested duration is rounded to the nearest multiple of the frame period
    ``1 / fps_resolution``. When the requested duration is not an exact
    multiple of that period, a :class:`UserWarning` is emitted reporting the
    duration that will actually be used.

    Parameters
    ----------
    t_seconds : float
        Requested duration, in seconds.
    fps_resolution : float
        Display refresh rate, in Hz (frames per second).
    param_name : str
        Name of the duration being converted; used only to build a more
        informative warning message (e.g. ``'t_stim'``).

    Returns
    -------
    int
        Duration expressed as a whole number of frames (rounded to nearest).
    """
    n_frames_exact = t_seconds * fps_resolution
    n_frames = int(round(n_frames_exact))
    # Warn only when the requested duration cannot be represented exactly at
    # this refresh rate and has therefore been approximated.
    if not np.isclose(n_frames_exact, n_frames):
        actual_t = n_frames / fps_resolution
        warnings.warn(
            f"{param_name}={t_seconds:g}s is not a multiple of the frame "
            f"period (1/{fps_resolution:g}Hz = "
            f"{1e3 / fps_resolution:.3f}ms): approximating to {n_frames} "
            f"frame(s) = {actual_t:g}s.",
            stacklevel=2,
        )
    return n_frames


def _make_commands_info(commands_info, codebook, extras):
    """Attach codebook rows to a ``{uid: CommandInfo}`` dict.

    Each codebook row is matched to one command in iteration order. When
    ``commands_info`` is ``None`` a fresh dict is built (one
    :class:`~spellers.data.CommandInfo` per row, ``uid = str(index)``, ``code``
    set from the row, and ``extras[index]`` stored in ``extra``). Otherwise only
    the ``code`` of each existing command is updated in place; ``extras`` is
    ignored and the command count must match the number of rows.

    Parameters
    ----------
    commands_info : dict of {str: CommandInfo} or None
        Commands to update in place, or ``None`` to build a default dict.
    codebook : numpy.ndarray
        Per-command code matrix of shape ``(n_commands, n_frames)``.
    extras : list of dict
        Paradigm-specific metadata (e.g. ``{'row': 0, 'col': 2}`` or
        ``{'stim_freq': 12.0}``) for each command, used only when building a
        default dict.

    Returns
    -------
    dict of {str: CommandInfo}
        The updated or newly built command dict.
    """
    n = len(codebook)
    if commands_info is None:
        return {
            str(i): CommandInfo(uid=str(i), code=codebook[i].tolist(),
                                extra=dict(extras[i]))
            for i in range(n)
        }
    if len(commands_info) != n:
        raise ValueError(
            f"commands_info has {len(commands_info)} entries but the codebook "
            f"has {n} commands; their counts must match.")
    for cmd, row in zip(commands_info.values(), codebook):
        cmd.code = row.tolist()
    return commands_info


def generate_row_col_codebook(
        n_rows, n_cols,
        t_stim,
        t_isi,
        fps_resolution,
        seed=None,
        commands_info=None
):
    """Generate the codebook of a row-column speller.

    Commands are laid out on an ``n_rows`` x ``n_cols`` grid and stimulated by
    *groups*: every row and every column forms one stimulation group, for a
    total of ``n_rows + n_cols`` groups. The groups are activated one after
    another in random order; while a group is active, every command belonging
    to it is set to ``1`` for ``t_stim`` seconds, followed by a silent
    inter-stimulus interval (ISI) of ``t_isi`` seconds during which every
    command is ``0``.

    Parameters
    ----------
    n_rows, n_cols : int
        Number of rows and columns of the command grid (both > 0). The speller
        exposes ``n_rows * n_cols`` commands.
    t_stim : float
        Stimulation time per group, in seconds (> 0).
    t_isi : float
        Inter-stimulus interval after each stimulation, in seconds (>= 0).
    fps_resolution : float
        Display refresh rate, in Hz (> 0). Durations are quantized to whole
        frames at this rate; non-representable durations are approximated and
        a warning is emitted (see Notes).
    seed : int or numpy.random.Generator or None, optional
        Seed or generator controlling the random order of the stimulation
        groups. Pass an int (or a ``Generator``) for a reproducible codebook;
        ``None`` (default) draws a fresh random order on every call.
    commands_info : dict of {str: CommandInfo} or None, optional
        Commands to attach the generated codes to. If a dict is given, each
        command's ``code`` is updated in place from the codebook, matched to
        rows in iteration order (the dict must hold exactly ``n_rows * n_cols``
        commands). If ``None`` (default), a fresh dict is built with one
        :class:`~spellers.data.CommandInfo` per command — ``uid = '0'`` ..
        ``str(n_rows * n_cols - 1)`` (row-major), its ``code``, and the grid
        position in ``extra`` as ``{'row': ..., 'col': ...}``.

    Returns
    -------
    commands_info : dict of {str: CommandInfo}
        Mapping ``uid -> CommandInfo`` — the dict passed in (with codes updated
        in place) or a freshly built default one. Each command's ``code`` is the
        binary row of the codebook (length
        ``code_len = (n_rows + n_cols) * (n_frames_stim + n_frames_isi)``,
        ``1`` while the command is stimulated).

    Warns
    -----
    UserWarning
        If ``t_stim`` or ``t_isi`` is not an exact multiple of the frame
        period and is therefore approximated to the nearest whole frame.

    Raises
    ------
    ValueError
        If any argument is out of range, or if ``t_stim`` rounds to 0 frames
        at the given refresh rate (which would yield an all-zero codebook).

    Notes
    -----
    Because stimuli can only switch at frame boundaries, the effective
    durations are ``round(t * fps_resolution)`` frames. For example, at 60 Hz a
    requested ``t_stim`` of 75 ms (4.5 frames) is approximated to 4 frames
    (~66.7 ms).
    """
    # --- Validate inputs ---------------------------------------------------
    if not (isinstance(n_rows, (int, np.integer)) and n_rows > 0):
        raise ValueError(f"n_rows must be a positive integer, got {n_rows!r}.")
    if not (isinstance(n_cols, (int, np.integer)) and n_cols > 0):
        raise ValueError(f"n_cols must be a positive integer, got {n_cols!r}.")
    if not fps_resolution > 0:
        raise ValueError(f"fps_resolution must be > 0, got {fps_resolution!r}.")
    if not t_stim > 0:
        raise ValueError(f"t_stim must be > 0, got {t_stim!r}.")
    if t_isi < 0:
        raise ValueError(f"t_isi must be >= 0, got {t_isi!r}.")

    # --- Grid / group bookkeeping -----------------------------------------
    n_cmd = n_rows * n_cols           # total number of commands
    n_stim_groups = n_rows + n_cols   # one stimulation group per row and column
    # Command ids arranged on the grid in row-major order, e.g. for 3x3:
    #   [[0 1 2], [3 4 5], [6 7 8]]
    cmd_layout = np.arange(n_cmd).reshape(n_rows, n_cols)

    # --- Time -> frames (quantized to the refresh rate, warns if inexact) --
    n_frames_stim = _seconds_to_frames(t_stim, fps_resolution, 't_stim')
    n_frames_isi = _seconds_to_frames(t_isi, fps_resolution, 't_isi')
    if n_frames_stim == 0:
        raise ValueError(
            f"t_stim={t_stim:g}s rounds to 0 frames at {fps_resolution:g}Hz; "
            f"the resulting codebook would contain no stimulation."
        )
    n_frames_cycle = n_frames_stim + n_frames_isi  # one stim + isi per group
    code_len = n_stim_groups * n_frames_cycle      # total codebook length

    # --- Random presentation order of the groups ---------------------------
    # default_rng leaves the global NumPy RNG untouched; pass ``seed`` to make
    # the codebook reproducible.
    rng = np.random.default_rng(seed)
    stim_groups = rng.permutation(n_stim_groups)

    # --- Fill the codebook -------------------------------------------------
    # Groups [0, n_rows) are rows; [n_rows, n_rows + n_cols) are columns.
    codebook = np.zeros((n_cmd, code_len), dtype=np.uint8)
    for slot, grp in enumerate(stim_groups):
        if grp < n_rows:
            stim_cmds = cmd_layout[grp, :]           # commands in that row
        else:
            stim_cmds = cmd_layout[:, grp - n_rows]  # commands in that column
        stim_start = slot * n_frames_cycle
        codebook[stim_cmds, stim_start:stim_start + n_frames_stim] = 1

    # Build a default {uid: CommandInfo} dict, or update the one provided.
    extras = [{'row': c // n_cols, 'col': c % n_cols} for c in range(n_cmd)]
    return _make_commands_info(commands_info, codebook, extras)


def _get_quantization_bins(base):
    """Return the ``base - 1`` bin edges that split [-1, 1] into ``base`` levels.

    The analog flicker waveform is a sine in [-1, 1]; quantizing it with these
    equally spaced edges via :func:`numpy.digitize` maps it onto integer levels
    ``0 .. base - 1``. For ``base == 2`` the single edge at 0 turns the sine
    into a square wave (0 while negative, 1 while positive).

    Parameters
    ----------
    base : int
        Number of quantization levels (>= 2).

    Returns
    -------
    numpy.ndarray
        The ``base - 1`` inner edges ``-1 + step, ..., 1 - step`` with
        ``step = 2 / base``.
    """
    step = 2 / base
    return -1.0 + step * np.arange(1, base)


def _optimize_ssvep_frequencies(target_freqs, fps_resolution):
    """Snap requested frequencies to *distinct* frame-lockable frequencies.

    On a frame-based display a flicker is only strictly periodic (no frame
    jitter) when one cycle spans a whole number of frames, so the realizable
    frequencies are ``fps_resolution / p`` for integer periods ``p >= 2``
    (``p == 2`` is the Nyquist limit ``fps / 2``). Each requested frequency is
    snapped to the closest such frequency (minimum absolute error in Hz); if
    that one is already taken by another command, the nearest still-free
    frame-lockable frequency is used so every command keeps a distinct code.

    Parameters
    ----------
    target_freqs : array_like
        Requested frequencies, in Hz.
    fps_resolution : float
        Display refresh rate, in Hz.

    Returns
    -------
    realized_freqs : numpy.ndarray
        Frame-lockable frequencies actually used (``fps_resolution / periods``).
    periods : numpy.ndarray
        Integer number of frames per cycle assigned to each command.
    """
    target_freqs = np.asarray(target_freqs, dtype=float)
    n = target_freqs.size
    ideal_periods = fps_resolution / target_freqs   # fractional frames per cycle
    # Candidate integer periods, wide enough that every command can still get a
    # distinct value after resolving collisions (>= 2 enforces Nyquist).
    p_hi = int(np.ceil(ideal_periods.max())) + n
    candidates = np.arange(2, p_hi + 1)
    candidate_freqs = fps_resolution / candidates   # frame-lockable frequencies
    used = np.zeros(candidates.shape, dtype=bool)
    periods = np.empty(n, dtype=int)
    for i in range(n):
        # still-free frame-lockable frequency closest (in Hz) to the target
        cost = np.abs(candidate_freqs - target_freqs[i])
        cost[used] = np.inf
        k = int(np.argmin(cost))
        used[k] = True
        periods[i] = candidates[k]
    return fps_resolution / periods, periods


def get_optimal_frequencies(fps_resolution, freq_range=None):
    """Return every flicker frequency a display can render exactly at ``fps``.

    A flicker is rendered exactly — no frame jitter, no frequency error — only
    when one cycle spans a whole number of frames, so the exactly-displayable
    frequencies are ``fps_resolution / p`` for integer periods ``p >= 2``
    (``p == 2`` is the Nyquist limit ``fps / 2``, the fastest real flicker). The
    result is restricted to ``freq_range``.

    Parameters
    ----------
    fps_resolution : float
        Display refresh rate, in Hz (> 0).
    freq_range : (float, float) or None, optional
        ``(f_min, f_max)`` band, in Hz, to keep (inclusive). ``f_min`` must be
        > 0 and may be below 1 Hz; ``f_max`` is effectively capped at the
        Nyquist limit ``fps / 2`` (nothing faster can flicker). Defaults to
        ``None``, i.e. ``(1, fps_resolution / 2)``.

    Returns
    -------
    freqs : numpy.ndarray
        Exactly-displayable frequencies in Hz (``fps_resolution / p`` for integer
        periods ``p``) that fall within ``freq_range``, sorted ascending. Each
        renders with zero frame-quantization error. May be empty if no such
        frequency lies in the band.

    Raises
    ------
    ValueError
        If ``fps_resolution`` is not > 0, or if ``freq_range`` is not a pair
        satisfying ``0 < f_min <= f_max``.

    Examples
    --------
    >>> get_optimal_frequencies(60, freq_range=(8, 15)).round(2).tolist()
    [8.57, 10.0, 12.0, 15.0]
    >>> get_optimal_frequencies(60)[-5:].round(3).tolist()
    [10.0, 12.0, 15.0, 20.0, 30.0]
    """
    if not fps_resolution > 0:
        raise ValueError(f"fps_resolution must be > 0, got {fps_resolution!r}.")
    # Default band: 1 Hz up to the Nyquist limit (the fastest real flicker).
    if freq_range is None:
        freq_range = (1.0, fps_resolution / 2)
    freq_range = tuple(freq_range)
    if len(freq_range) != 2:
        raise ValueError(
            f"freq_range must be a (f_min, f_max) pair, got {freq_range!r}.")
    f_min, f_max = float(freq_range[0]), float(freq_range[1])
    if not 0 < f_min <= f_max:
        raise ValueError(
            f"freq_range must satisfy 0 < f_min <= f_max, got {freq_range!r}.")

    # freq = fps / p lies in [f_min, f_max]  <=>  p in [fps/f_max, fps/f_min].
    # p >= 2 enforces the Nyquist limit, so nothing above fps/2 is returned.
    p_lo = max(2, int(np.ceil(fps_resolution / f_max)))
    p_hi = int(np.floor(fps_resolution / f_min))
    periods = np.arange(p_lo, p_hi + 1)
    return np.sort(fps_resolution / periods)


def generate_freq_codebook(
        n_cmds,
        freq_range,
        t_stim,
        fps_resolution,
        base=2,
        commands_info=None
):
    """Generate the codebook of a simple frequency-coded speller.

    Each command flickers at its own frequency. ``n_cmds`` target frequencies
    are spread linearly across ``freq_range`` and each is *frame-locked*: it is
    snapped to the closest frequency whose period is a whole number of frames
    (``fps_resolution / p``, integer ``p``), so the on-screen flicker is
    periodic and jitter-free. Frequencies that cannot be represented exactly are
    approximated and a warning is emitted (see Notes). For every command a sine
    at its frequency is sampled once per frame over ``t_stim`` seconds and
    quantized into ``base`` levels (``base == 2`` -> a 0/1 square wave).

    Parameters
    ----------
    n_cmds : int
        Number of commands (> 0); one frequency is assigned per command.
    freq_range : (float, float)
        ``(f_min, f_max)`` frequency range in Hz. Must satisfy
        ``0 < f_min <= f_max <= fps_resolution / 2`` (Nyquist).
    t_stim : float
        Stimulation time, in seconds (> 0). The codebook has
        ``round(t_stim * fps_resolution)`` frames.
    fps_resolution : float
        Display refresh rate, in Hz (> 0).
    base : int, optional
        Number of quantization levels of the flicker waveform (>= 2). The
        default ``2`` yields a binary on/off square wave; larger values give a
        multi-level (e.g. grayscale) rendering of the sine.
    commands_info : dict of {str: CommandInfo} or None, optional
        Commands to attach the generated codes to. If a dict is given, each
        command's ``code`` is updated in place from the codebook, matched to
        rows in iteration order (the dict must hold exactly ``n_cmds``
        commands). If ``None`` (default), a fresh dict is built with one
        :class:`~spellers.data.CommandInfo` per command — ``uid = '0'`` ..
        ``str(n_cmds - 1)`` (ascending frequency), its ``code``, and the
        frame-locked frequency in ``extra`` as ``{'stim_freq': <Hz>}``.

    Returns
    -------
    commands_info : dict of {str: CommandInfo}
        Mapping ``uid -> CommandInfo`` — the dict passed in (with codes updated
        in place) or a freshly built default one. Each command's ``code`` is its
        quantized flicker (``round(t_stim * fps_resolution)`` frames, integer
        levels in ``[0, base - 1]``); commands are ordered by ascending
        frequency.

    Warns
    -----
    UserWarning
        If ``t_stim`` is approximated to a whole number of frames, or if any
        requested frequency is approximated to a frame-lockable one (including
        when ``freq_range`` / ``fps_resolution`` cannot supply ``n_cmds``
        distinct frame-locked frequencies within the range).

    Raises
    ------
    ValueError
        If any argument is out of range (e.g. ``f_max > fps_resolution / 2``),
        or if ``t_stim`` rounds to 0 frames at the given refresh rate.

    Notes
    -----
    Frame-locking: a display can only toggle stimuli at frame boundaries, so the
    cleanly renderable SSVEP frequencies are ``fps_resolution / p`` for integer
    ``p >= 2``. At 60 Hz these are 30, 20, 15, 12, 10, 8.57, ... Hz. A requested
    frequency is snapped to the nearest such value; e.g. 11 Hz at 60 Hz snaps to
    60 / 5 = 12 Hz.
    """
    # --- Validate inputs ---------------------------------------------------
    if not (isinstance(n_cmds, (int, np.integer)) and n_cmds > 0):
        raise ValueError(f"n_cmds must be a positive integer, got {n_cmds!r}.")
    if not fps_resolution > 0:
        raise ValueError(f"fps_resolution must be > 0, got {fps_resolution!r}.")
    if not t_stim > 0:
        raise ValueError(f"t_stim must be > 0, got {t_stim!r}.")
    if not (isinstance(base, (int, np.integer)) and base >= 2):
        raise ValueError(f"base must be an integer >= 2, got {base!r}.")
    freq_range = tuple(freq_range)
    if len(freq_range) != 2:
        raise ValueError(
            f"freq_range must be a (f_min, f_max) pair, got {freq_range!r}.")
    f_min, f_max = float(freq_range[0]), float(freq_range[1])
    nyquist = fps_resolution / 2
    if not 0 < f_min <= f_max:
        raise ValueError(
            f"freq_range must satisfy 0 < f_min <= f_max, got {freq_range!r}.")
    if f_max > nyquist:
        raise ValueError(
            f"f_max={f_max:g}Hz exceeds the Nyquist limit fps/2={nyquist:g}Hz.")

    # --- Stimulation length in frames (quantized, warns if inexact) --------
    n_frames = _seconds_to_frames(t_stim, fps_resolution, 't_stim')
    if n_frames == 0:
        raise ValueError(
            f"t_stim={t_stim:g}s rounds to 0 frames at {fps_resolution:g}Hz; "
            f"the resulting codebook would be empty."
        )

    # --- One frame-locked frequency per command ----------------------------
    target_freqs = np.linspace(f_min, f_max, n_cmds)
    freqs, _ = _optimize_ssvep_frequencies(target_freqs, fps_resolution)
    # Warn about any frequency that had to be approximated, and flag the case
    # where the range simply cannot hold n_cmds distinct frame-locked values.
    approx = ~np.isclose(freqs, target_freqs)
    if np.any(approx):
        max_err = float(np.max(np.abs(freqs - target_freqs)))
        # distinct frame-locked frequencies available inside the range
        p_lo = max(2, int(np.ceil(fps_resolution / f_max)))
        p_hi = int(np.floor(fps_resolution / f_min))
        capacity = max(0, p_hi - p_lo + 1)
        msg = (
            f"{int(approx.sum())}/{n_cmds} requested frequencies are not "
            f"frame-lockable at {fps_resolution:g}Hz and were approximated to "
            f"fps/round(fps/f) (max error {max_err:g}Hz)."
        )
        if n_cmds > capacity:
            msg += (
                f" Only {capacity} distinct frame-locked frequencies exist in "
                f"[{f_min:g}, {f_max:g}]Hz, fewer than n_cmds={n_cmds}; some "
                f"commands use frequencies outside this range. Widen freq_range "
                f"or raise fps_resolution."
            )
        warnings.warn(msg, stacklevel=2)

    # --- Build the codebook: one sampled sine per command, then quantize ---
    t = np.arange(n_frames) / fps_resolution                  # frame timestamps
    # (n_cmds, n_frames) analog sines in [-1, 1]
    analog = np.sin(2 * np.pi * freqs[:, None] * t[None, :])
    codebook = np.digitize(analog, bins=_get_quantization_bins(base),
                           right=False)                       # levels 0..base-1
    if base <= 256:
        codebook = codebook.astype(np.uint8)

    # Build a default {uid: CommandInfo} dict, or update the one provided.
    extras = [{'stim_freq': float(freqs[c])} for c in range(n_cmds)]
    return _make_commands_info(commands_info, codebook, extras)


def plot_codebook(commands_info, ax=None, cmap='gray', label='uid', title=None):
    """Visualize the codebook held in a ``{uid: CommandInfo}`` dict.

    Each command's ``code`` becomes one row of a black-and-white image (dark =
    level 0, light = the maximum level), which makes the encoding produced by
    :func:`generate_row_col_codebook` / :func:`generate_freq_codebook` easy to
    eyeball.

    Parameters
    ----------
    commands_info : dict of {str: CommandInfo}
        Commands whose ``code`` lists are stacked, in iteration order, into the
        rows of the image. Every code must have the same length.
    ax : matplotlib.axes.Axes or None, optional
        Axes to draw on. When ``None`` (default) a new figure and axes are
        created, sized to the number of commands.
    cmap : str, optional
        Matplotlib colormap. Defaults to ``'gray'`` (0 -> black, max -> white).
    label : {'uid', 'content'} or callable or None, optional
        How to label each command on the y-axis: ``'uid'`` (default) or
        ``'content'`` reads that attribute, a callable receives the
        ``CommandInfo`` and returns the label string, and ``None`` hides the
        per-command ticks.
    title : str or None, optional
        Axes title.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes the codebook was drawn on (its figure is ``ax.figure``).

    Raises
    ------
    ValueError
        If ``commands_info`` is empty, the codes are not all the same length,
        or ``label`` is not a recognized option.
    """
    import matplotlib.pyplot as plt

    if not commands_info:
        raise ValueError("commands_info is empty; nothing to plot.")
    commands = list(commands_info.values())
    lengths = {len(cmd.code) for cmd in commands}
    if len(lengths) != 1:
        raise ValueError(
            f"all commands must have equal-length codes, got lengths "
            f"{sorted(lengths)}.")
    codebook = np.array([cmd.code for cmd in commands])

    if ax is None:
        _, ax = plt.subplots(figsize=(10, max(2.0, 0.4 * len(commands))))

    ax.imshow(codebook, cmap=cmap, aspect='auto', interpolation='nearest',
              vmin=0, vmax=max(1, int(codebook.max())))
    ax.set_xlabel('Frame')
    ax.set_ylabel('Command')
    if title is not None:
        ax.set_title(title)

    # Per-command y-axis labels.
    if label is None:
        ax.set_yticks([])
    else:
        if callable(label):
            labels = [str(label(cmd)) for cmd in commands]
        elif label == 'uid':
            labels = [str(cmd.uid) for cmd in commands]
        elif label == 'content':
            labels = [str(cmd.content) for cmd in commands]
        else:
            raise ValueError(
                f"label must be 'uid', 'content', a callable or None, got "
                f"{label!r}.")
        ax.set_yticks(np.arange(len(commands)))
        ax.set_yticklabels(labels)
    return ax