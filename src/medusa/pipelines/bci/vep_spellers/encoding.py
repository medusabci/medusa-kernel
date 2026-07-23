"""Stimulation encoding for VEP spellers: the codebook framework.

A *codebook* is the one idea that unifies c-VEP, SSVEP and ERP/P300 spellers. Each
command owns a ``(n_codes, n_frames)`` ``code`` (its
:attr:`~medusa.pipelines.bci.vep_spellers.data.CommandInfo.code`): one row per selectable code,
each a per-frame pattern (a frame is one display refresh). Single-code paradigms (c-VEP,
SSVEP) have ``n_codes == 1``. A row-column speller that re-randomises its flash order
stores one code per sequence. The paradigm is simply *which generator built the codebook*:

* :func:`generate_row_col_codebook` -- ERP/P300 (sparse binary, row and column groups),
* :func:`generate_freq_codebook` -- SSVEP (frame-locked flicker, quantized),
* :func:`generate_mseq_codebook` -- c-VEP (circular shifts of one m-sequence),
* :func:`generate_gold_codebook` -- c-VEP (distinct Gold codes).

Each one returns a ``{uid: CommandInfo}`` dict, ready for
:class:`~medusa.pipelines.bci.vep_spellers.data.SpellerData`. Stimulation is frame-based:
durations are rounded to whole display frames (see :func:`get_optimal_frequencies`).
This module is pure stimulation design. It has no EEG and no models.
"""

import warnings
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from medusa.pipelines.bci.vep_spellers.data import CommandInfo

if TYPE_CHECKING:
    from collections.abc import Callable

    from matplotlib.axes import Axes

__all__ = [
    "generate_row_col_codebook",
    "generate_freq_codebook",
    "generate_mseq_codebook",
    "generate_gold_codebook",
    "generate_random_codebook",
    "get_optimal_frequencies",
    "plot_codebook",
    "LFSR",
    "GOLD_CODES",
]

#: Color of the block separators :func:`plot_codebook` draws ON TOP of the codebook
#: image between multi-code commands. A kernel-local default (like the one in
#: ``plots/time_heatmap.py``): there is no medusa-style palette role for "a line over a
#: colormap", and this saturated red is the rare color that stays visible over *both*
#: the pure-black (level 0) and pure-white (max level) cells of a binary ``'gray'`` code.
_SEPARATOR_COLOR = "#E24A4A"


def _seconds_to_frames(t_seconds, fps_resolution, param_name):
    """Quantize a duration in seconds to a whole number of display frames.

    A monitor updates only at frame boundaries. So a requested duration is rounded to the
    nearest multiple of the frame period ``1 / fps_resolution``. If the duration cannot be
    represented exactly, it is approximated and a :class:`UserWarning` reports the value
    actually used.
    """
    n_frames_exact = t_seconds * fps_resolution
    n_frames = int(round(n_frames_exact))
    if not np.isclose(n_frames_exact, n_frames):
        actual_t = n_frames / fps_resolution
        warnings.warn(
            f"{param_name}={t_seconds:g}s is not a multiple of the frame "
            f"period (1/{fps_resolution:g}Hz = {1e3 / fps_resolution:.3f}ms): "
            f"approximating to {n_frames} frame(s) = {actual_t:g}s.",
            stacklevel=2,
        )
    return n_frames


def _make_commands_info(commands_info, codebook, extras):
    """Attach codebook rows to a ``{uid: CommandInfo}`` dict.

    Each codebook row is matched to one command, in iteration order. When
    ``commands_info`` is ``None``, a fresh dict is built: one :class:`CommandInfo` per
    row, with ``uid = str(index)``, ``code`` from the row, and ``extras[index]`` in
    ``extra``. Otherwise only each existing command's ``code`` is updated in place
    (``extras`` is ignored), and the number of commands must match the number of rows.
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
    # ``codebook[i]`` is 1-D (single-code paradigms) or 2-D (n_codes, n_frames); either
    # way store a 2-D code so there is one representation (CommandInfo coerces 1-D too).
    for cmd, row in zip(commands_info.values(), codebook):
        cmd.code = np.atleast_2d(row).tolist()
    return commands_info


# --------------------------------------------------------------------------- #
# ERP / P300 -- row-column codebook
# --------------------------------------------------------------------------- #
def generate_row_col_codebook(
        n_rows: int, n_cols: int, n_sequences: int, t_stim: float, t_isi: float,
        fps_resolution: float, seed: "int | np.random.Generator | None" = None,
        commands_info: "dict[str, CommandInfo] | None" = None,
) -> "dict[str, CommandInfo]":
    """Generate the codebook of a row-column (RCP) speller.

    Commands sit on an ``n_rows`` x ``n_cols`` grid and are stimulated by *groups*. Every
    row and every column is one stimulation group (``n_rows + n_cols`` groups in total),
    activated one after another. While a group is active, every command in it is ``1`` for
    ``t_stim`` seconds, followed by a silent gap (the ISI) of ``t_isi`` seconds. Durations
    are rounded to whole frames at ``fps_resolution``.

    An RCP re-randomises the group order **every cycle**, so this returns one code per
    sequence. Each command's ``code`` is ``(n_sequences, n_frames_cycle)``, and the
    speller cycles through the sequences (the event ``code_idx`` picks one). The c-VEP and
    SSVEP generators, in contrast, return a single code (``n_codes == 1``).

    Parameters
    ----------
    n_rows, n_cols :
        Grid size. Both must be > 0. The speller exposes ``n_rows * n_cols`` commands.
    n_sequences :
        Number of distinct random group orders to build (> 0). One code per order.
    t_stim :
        Stimulation time per group, in seconds (> 0).
    t_isi :
        Silent gap after each stimulation, in seconds (>= 0).
    fps_resolution :
        Display refresh rate, in Hz (> 0).
    seed :
        Controls the random group orders. Give a value for reproducible output.
    commands_info :
        Commands to update in place (must be exactly ``n_rows * n_cols``). Pass ``None``
        to build a fresh dict, with ``extra = {'row': ..., 'col': ...}`` (row-major).

    Returns
    -------
    dict of {str: CommandInfo}
        ``uid -> CommandInfo``. Each ``code`` has shape ``(n_sequences, n_frames)`` with
        ``n_frames = (n_rows + n_cols) * (n_frames_stim + n_frames_isi)``.

    Raises
    ------
    ValueError
        If an argument is out of range, or if ``t_stim`` rounds to 0 frames.

    Examples
    --------
    >>> import numpy as np
    >>> from medusa.pipelines.bci.vep_spellers.encoding import generate_row_col_codebook
    >>> cmds = generate_row_col_codebook(2, 3, n_sequences=4, t_stim=0.1,
    ...                                  t_isi=0.0, fps_resolution=60, seed=0)
    >>> len(cmds)                              # 2 * 3 commands
    6
    >>> np.asarray(cmds["0"].code).shape       # (n_sequences, n_frames)
    (4, 30)
    """
    if not (isinstance(n_rows, (int, np.integer)) and n_rows > 0):
        raise ValueError(f"n_rows must be a positive integer, got {n_rows!r}.")
    if not (isinstance(n_cols, (int, np.integer)) and n_cols > 0):
        raise ValueError(f"n_cols must be a positive integer, got {n_cols!r}.")
    if not (isinstance(n_sequences, (int, np.integer)) and n_sequences > 0):
        raise ValueError(f"n_sequences must be a positive integer, got {n_sequences!r}.")
    if not fps_resolution > 0:
        raise ValueError(f"fps_resolution must be > 0, got {fps_resolution!r}.")
    if not t_stim > 0:
        raise ValueError(f"t_stim must be > 0, got {t_stim!r}.")
    if t_isi < 0:
        raise ValueError(f"t_isi must be >= 0, got {t_isi!r}.")

    n_cmd = n_rows * n_cols
    n_stim_groups = n_rows + n_cols
    cmd_layout = np.arange(n_cmd).reshape(n_rows, n_cols)

    n_frames_stim = _seconds_to_frames(t_stim, fps_resolution, 't_stim')
    n_frames_isi = _seconds_to_frames(t_isi, fps_resolution, 't_isi')
    if n_frames_stim == 0:
        raise ValueError(
            f"t_stim={t_stim:g}s rounds to 0 frames at {fps_resolution:g}Hz; "
            f"the codebook would contain no stimulation.")
    n_frames_cycle = n_frames_stim + n_frames_isi
    code_len = n_stim_groups * n_frames_cycle

    rng = np.random.default_rng(seed)
    # (n_cmd, n_sequences, code_len): one random group order per sequence.
    codebook = np.zeros((n_cmd, n_sequences, code_len), dtype=np.uint8)
    for s in range(n_sequences):
        for slot, grp in enumerate(rng.permutation(n_stim_groups)):
            if grp < n_rows:
                stim_cmds = cmd_layout[grp, :]
            else:
                stim_cmds = cmd_layout[:, grp - n_rows]
            stim_start = slot * n_frames_cycle
            codebook[stim_cmds, s, stim_start:stim_start + n_frames_stim] = 1

    extras = [{'row': c // n_cols, 'col': c % n_cols} for c in range(n_cmd)]
    return _make_commands_info(commands_info, codebook, extras)


# --------------------------------------------------------------------------- #
# SSVEP -- frequency-coded codebook
# --------------------------------------------------------------------------- #
def _get_quantization_bins(base):
    """The ``base - 1`` inner edges that split ``[-1, 1]`` into ``base`` levels.

    Quantizing a sine in ``[-1, 1]`` with :func:`numpy.digitize` maps it to integer
    levels ``0 .. base - 1`` (``base == 2`` gives a 0/1 square wave).
    """
    step = 2 / base
    return -1.0 + step * np.arange(1, base)


def _optimize_ssvep_frequencies(target_freqs, fps_resolution):
    """Snap requested frequencies to *distinct* frame-lockable frequencies.

    A flicker has no jitter only when one cycle spans a whole number of frames. So the
    frequencies a display can show are ``fps_resolution / p`` for integer ``p >= 2``. Each
    requested frequency is snapped to the closest free one (smallest error in Hz).
    """
    target_freqs = np.asarray(target_freqs, dtype=float)
    n = target_freqs.size
    ideal_periods = fps_resolution / target_freqs
    p_hi = int(np.ceil(ideal_periods.max())) + n
    candidates = np.arange(2, p_hi + 1)
    candidate_freqs = fps_resolution / candidates
    used = np.zeros(candidates.shape, dtype=bool)
    periods = np.empty(n, dtype=int)
    for i in range(n):
        cost = np.abs(candidate_freqs - target_freqs[i])
        cost[used] = np.inf
        k = int(np.argmin(cost))
        used[k] = True
        periods[i] = candidates[k]
    return fps_resolution / periods, periods


def get_optimal_frequencies(fps_resolution: float,
                            freq_range: "tuple[float, float] | None" = None) -> NDArray:
    """Every flicker frequency a display can show exactly at ``fps_resolution``.

    A flicker is exact (no frame jitter) only when one cycle spans a whole number of
    frames. So the frequencies a display can show are ``fps_resolution / p`` for integer
    ``p >= 2`` (``p == 2`` is the Nyquist limit ``fps / 2``), kept within ``freq_range``.

    Parameters
    ----------
    fps_resolution :
        Display refresh rate, in Hz (> 0).
    freq_range :
        ``(f_min, f_max)`` band in Hz, both ends included. Defaults to ``(1, fps / 2)``.

    Returns
    -------
    numpy.ndarray
        The frequencies in the band the display can show exactly, sorted ascending. May
        be empty.

    Raises
    ------
    ValueError
        If ``fps_resolution <= 0``, if ``freq_range`` is not a ``(f_min, f_max)`` pair,
        or if it does not satisfy ``0 < f_min <= f_max``.

    Examples
    --------
    >>> from medusa.pipelines.bci.vep_spellers.encoding import get_optimal_frequencies
    >>> get_optimal_frequencies(60, freq_range=(8, 15)).round(2).tolist()
    [8.57, 10.0, 12.0, 15.0]
    """
    if not fps_resolution > 0:
        raise ValueError(f"fps_resolution must be > 0, got {fps_resolution!r}.")
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
    p_lo = max(2, int(np.ceil(fps_resolution / f_max)))
    p_hi = int(np.floor(fps_resolution / f_min))
    periods = np.arange(p_lo, p_hi + 1)
    return np.sort(fps_resolution / periods)


def generate_freq_codebook(
        n_cmds: int, freq_range: "tuple[float, float]", t_stim: float,
        fps_resolution: float, base: int = 2,
        commands_info: "dict[str, CommandInfo] | None" = None,
) -> "dict[str, CommandInfo]":
    """Generate the codebook of a frequency-coded (SSVEP) speller.

    ``n_cmds`` target frequencies are spread evenly across ``freq_range``. Each one is
    *frame-locked* (snapped to ``fps_resolution / p`` for integer ``p``). A sine at each
    frequency is sampled once per frame over ``t_stim`` seconds and quantized into
    ``base`` levels (``base == 2`` gives a 0/1 square wave).

    Parameters
    ----------
    n_cmds :
        Number of commands (> 0). One frequency per command.
    freq_range :
        ``(f_min, f_max)`` in Hz, with ``0 < f_min <= f_max <= fps_resolution / 2``.
    t_stim :
        Stimulation time, in seconds (> 0).
    fps_resolution :
        Display refresh rate, in Hz (> 0).
    base :
        Number of quantization levels (>= 2). Default ``2`` (binary square wave).
    commands_info :
        Commands to update in place (exactly ``n_cmds``). Pass ``None`` to build a fresh
        dict, with ``extra = {'stim_freq': <Hz>}`` (ascending frequency).

    Returns
    -------
    dict of {str: CommandInfo}
        ``uid -> CommandInfo``. Each ``code`` has ``round(t_stim * fps_resolution)``
        frames, with integer levels in ``[0, base - 1]``.

    Raises
    ------
    ValueError
        If an argument is out of range, or if ``t_stim`` rounds to 0 frames.

    Examples
    --------
    >>> from medusa.pipelines.bci.vep_spellers.encoding import generate_freq_codebook
    >>> cmds = generate_freq_codebook(3, freq_range=(10, 20), t_stim=1.0,
    ...                               fps_resolution=60)
    >>> len(cmds)
    3
    >>> len(cmds["0"].code[0])          # round(t_stim * fps_resolution) frames
    60
    >>> sorted(round(c.extra["stim_freq"]) for c in cmds.values())
    [10, 15, 20]
    """
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

    n_frames = _seconds_to_frames(t_stim, fps_resolution, 't_stim')
    if n_frames == 0:
        raise ValueError(
            f"t_stim={t_stim:g}s rounds to 0 frames at {fps_resolution:g}Hz; "
            f"the codebook would be empty.")

    target_freqs = np.linspace(f_min, f_max, n_cmds)
    freqs, _ = _optimize_ssvep_frequencies(target_freqs, fps_resolution)
    approx = ~np.isclose(freqs, target_freqs)
    if np.any(approx):
        max_err = float(np.max(np.abs(freqs - target_freqs)))
        p_lo = max(2, int(np.ceil(fps_resolution / f_max)))
        p_hi = int(np.floor(fps_resolution / f_min))
        capacity = max(0, p_hi - p_lo + 1)
        msg = (f"{int(approx.sum())}/{n_cmds} requested frequencies are not "
               f"frame-lockable at {fps_resolution:g}Hz and were approximated "
               f"(max error {max_err:g}Hz).")
        if n_cmds > capacity:
            msg += (f" Only {capacity} distinct frame-locked frequencies exist in "
                    f"[{f_min:g}, {f_max:g}]Hz, fewer than n_cmds={n_cmds}; widen "
                    f"freq_range or raise fps_resolution.")
        warnings.warn(msg, stacklevel=2)

    t = np.arange(n_frames) / fps_resolution
    analog = np.sin(2 * np.pi * freqs[:, None] * t[None, :])
    codebook = np.digitize(analog, bins=_get_quantization_bins(base), right=False)
    if base <= 256:
        codebook = codebook.astype(np.uint8)

    extras = [{'stim_freq': float(freqs[c])} for c in range(n_cmds)]
    return _make_commands_info(commands_info, codebook, extras)


# --------------------------------------------------------------------------- #
# c-VEP -- m-sequence / Gold-code generators
# --------------------------------------------------------------------------- #
class LFSR:
    """A Linear-Feedback Shift Register (LFSR) sequence.

    Maximal-length (m-)sequences need a *primitive* generator polynomial. There is a
    table of primitive polynomials by order at
    https://en.wikipedia.org/wiki/Linear-feedback_shift_register. A seed of all zeros
    gives an all-zero output.

    Parameters
    ----------
    polynomial :
        Generator polynomial (the taps), for example ``[0, 0, 0, 0, 1, 1]`` for
        ``1 + x^5 + x^6``.
    base :
        Galois-field base. Default ``2`` gives events in ``{0, 1}``.
    seed :
        Initial state. Default: all ones, with length equal to the polynomial order.
    center :
        If ``True``, center the output around zero (``{0, 1}`` becomes ``{-1, 1}``).

    Examples
    --------
    >>> from medusa.pipelines.bci.vep_spellers.encoding import LFSR
    >>> len(LFSR([0, 0, 0, 0, 1, 1]).sequence)   # base ** order - 1
    63
    """

    def __init__(self, polynomial: list, base: int = 2, seed: "list | None" = None,
                 center: bool = False):
        self.polynomial = polynomial
        self.base = base
        self.seed = seed
        self.order = len(polynomial)
        self.N = base ** self.order - 1
        self.sequence = self.lfsr(polynomial, base, seed, center)

    @staticmethod
    def lfsr(polynomial: list, base: int = 2, seed: "list | None" = None,
             center: bool = False) -> "list | NDArray":
        """Return the LFSR sequence (max length ``base ** order - 1``).

        Raises
        ------
        ValueError
            If the polynomial order is larger than the seed length.
        """
        order = len(polynomial)
        if seed is None:
            seed = [1 for _ in range(order)]
        if order > len(seed):
            raise ValueError(
                f"[LFSR] polynomial order ({order}) exceeds the seed length "
                f"({len(seed)}).")
        sequence = list(seed).copy()
        polynom = np.array(polynomial)
        while len(sequence) < base ** order - 1:
            new_bit = int(np.matmul(polynom, np.array(sequence[:order])) % base)
            sequence.insert(0, new_bit)
        if center:
            if base == 2:
                sequence = np.array(sequence) * 2 - 1
            else:
                sequence = np.array(sequence) - int(np.floor(base / 2))
        return sequence


class GOLD_CODES:
    """A set of ``base ** order - 1`` Gold codes from two preferred m-sequences.

    Parameters
    ----------
    polynomial_1, polynomial_2 :
        The two generator polynomials. They must have the same degree, and they must be a
        *preferred pair* for good cross-correlation.
    base :
        Galois-field base (default ``2``).
    center :
        Kept for API symmetry with :class:`LFSR`. The codes are ``{0, 1}``.
    """

    def __init__(self, polynomial_1: list, polynomial_2: list, base: int = 2,
                 center: bool = False):
        self.polynomial_1 = polynomial_1
        self.polynomial_2 = polynomial_2
        self.base = base
        self.order = len(polynomial_1)
        self.N = base ** self.order - 1
        self.sequences = self.gold_codes(polynomial_1, polynomial_2, base,
                                         self.order)

    @staticmethod
    def gold_codes(pol_1: list, pol_2: list, base: int,
                   order: int) -> "list[list[int]]":
        """Return the list of Gold codes (each of length ``base ** order - 1``).

        Raises
        ------
        ValueError
            If ``pol_1`` and ``pol_2`` do not have the same degree.
        """
        if pol_1[-1] != pol_2[-1]:
            raise ValueError("pol_1 and pol_2 must have the same degree.")
        sequences = []
        seq_1 = list(LFSR.lfsr(pol_1))
        seq_2 = list(LFSR.lfsr(pol_2))
        seq_1.reverse()
        seq_2.reverse()
        for i in range(0, base ** order - 1):
            seq_2_rolled = list(np.roll(np.array(seq_2), i))
            seq = [int(bool(seq_1[j]) ^ bool(seq_2_rolled[j]))
                   for j in range(len(seq_1))]
            sequences.append(seq)
        return sequences


def generate_mseq_codebook(
        n_commands: int, polynomial: list, base: int = 2, seed: "list | None" = None,
        shift: "int | None" = None,
        commands_info: "dict[str, CommandInfo] | None" = None,
) -> "dict[str, CommandInfo]":
    """Generate a c-VEP codebook where each command is a circular shift of one m-sequence.

    This is the classic *circular-shifting* c-VEP paradigm. One m-sequence (length
    ``base ** order - 1``) is shown, and each command's code is that sequence rolled by a
    different lag. So one trained template can decode the whole matrix.

    Parameters
    ----------
    n_commands :
        Number of commands (> 0).
    polynomial :
        Primitive generator polynomial for the base m-sequence (see :class:`LFSR`).
    base :
        Galois-field base (default ``2``).
    seed :
        LFSR seed (default: all ones).
    shift :
        Lag, in frames, between the codes of consecutive commands. ``None`` spreads the
        commands as evenly as possible (``len(seq) // n_commands``).
    commands_info :
        Commands to update in place (exactly ``n_commands``). Pass ``None`` to build a
        fresh dict, with ``extra = {'lag': ...}``.

    Returns
    -------
    dict of {str: CommandInfo}
        ``uid -> CommandInfo``. Each ``code`` has length ``base ** order - 1``.

    Raises
    ------
    ValueError
        If an argument is out of range, or if the requested lags are not distinct.

    Examples
    --------
    >>> from medusa.pipelines.bci.vep_spellers.encoding import generate_mseq_codebook
    >>> cmds = generate_mseq_codebook(4, polynomial=[0, 0, 0, 0, 1, 1])
    >>> len(cmds)
    4
    >>> len(cmds["0"].code[0])          # base ** order - 1
    63
    """
    if not (isinstance(n_commands, (int, np.integer)) and n_commands > 0):
        raise ValueError(f"n_commands must be a positive integer, got {n_commands!r}.")
    seq = np.asarray(LFSR(polynomial, base=base, seed=seed).sequence)
    length = len(seq)
    if n_commands > length:
        raise ValueError(
            f"n_commands={n_commands} exceeds the m-sequence length {length}.")
    if shift is None:
        shift = length // n_commands
    if shift <= 0:
        raise ValueError(f"shift must be a positive integer, got {shift!r}.")
    lags = (np.arange(n_commands) * shift) % length
    if len(set(lags.tolist())) != n_commands:
        raise ValueError(
            f"shift={shift} does not yield {n_commands} distinct lags over a "
            f"length-{length} sequence.")
    codebook = np.stack([np.roll(seq, -int(lag)) for lag in lags]).astype(np.uint8)
    extras = [{'lag': int(lag), 'polynomial': list(polynomial)} for lag in lags]
    return _make_commands_info(commands_info, codebook, extras)


def generate_gold_codebook(
        n_commands: int, polynomial_1: list, polynomial_2: list, base: int = 2,
        commands_info: "dict[str, CommandInfo] | None" = None,
) -> "dict[str, CommandInfo]":
    """Generate a c-VEP codebook where each command is a distinct Gold code.

    Uses Gold codes, which have good, bounded cross-correlation. They fit a TRCA-style
    decoder that trains one template per command.

    Parameters
    ----------
    n_commands :
        Number of commands (> 0). Must not exceed the Gold-code family size
        ``base ** order - 1``.
    polynomial_1, polynomial_2 :
        The preferred-pair generator polynomials (see :class:`GOLD_CODES`).
    base :
        Galois-field base (default ``2``).
    commands_info :
        Commands to update in place (exactly ``n_commands``). Pass ``None`` to build a
        fresh dict, with ``extra = {'gold_index': ...}``.

    Returns
    -------
    dict of {str: CommandInfo}
        ``uid -> CommandInfo``. Each ``code`` has length ``base ** order - 1``.

    Raises
    ------
    ValueError
        If ``n_commands`` is not a positive integer or exceeds the Gold-code family size,
        or if the two polynomials are not a valid pair (they must have the same degree).

    Examples
    --------
    >>> from medusa.pipelines.bci.vep_spellers.encoding import generate_gold_codebook
    >>> cmds = generate_gold_codebook(4, polynomial_1=[1, 0, 0, 0, 0, 1],
    ...                               polynomial_2=[1, 1, 0, 0, 1, 1])
    >>> len(cmds)
    4
    >>> len(cmds["0"].code[0])          # base ** order - 1
    63
    """
    if not (isinstance(n_commands, (int, np.integer)) and n_commands > 0):
        raise ValueError(f"n_commands must be a positive integer, got {n_commands!r}.")
    codes = GOLD_CODES(polynomial_1, polynomial_2, base=base).sequences
    if n_commands > len(codes):
        raise ValueError(
            f"n_commands={n_commands} exceeds the Gold-code family size {len(codes)}.")
    codebook = np.array(codes[:n_commands], dtype=np.uint8)
    extras = [{'gold_index': i} for i in range(n_commands)]
    return _make_commands_info(commands_info, codebook, extras)


def generate_random_codebook(
        n_commands: int, n_frames: int, base: int = 2, p: float = 0.5,
        seed: "int | np.random.Generator | None" = None,
        commands_info: "dict[str, CommandInfo] | None" = None,
) -> "dict[str, CommandInfo]":
    """Generate a c-VEP codebook of independent random ("noise") codes, one per command.

    In random-code c-VEP every command owns its own randomly drawn per-frame code, with
    no circular-shift (m-sequence) or frequency (SSVEP) structure. Independent random
    codes have low mutual cross-correlation, so a bit-wise-reconstruction decoder
    (:class:`~medusa.pipelines.bci.vep_spellers.BWRLDAPipeline`) tells them apart from the
    reconstructed frame response alone, without a per-command template. A calibrated
    template-matching decoder also works, but then every command must be attended during
    calibration (each distinct code learns its own template), so BWR is the natural fit.

    For a binary code (``base == 2``) each frame is ``1`` with probability ``p`` and ``0``
    otherwise; for ``base > 2`` each frame is a uniform integer level in ``[0, base - 1]``
    and ``p`` is ignored. Codes are drawn to be **distinct** and **non-constant** (a
    constant code has zero variance and can never be decoded), redrawing on a collision.

    Parameters
    ----------
    n_commands :
        Number of commands (> 0). One random code each.
    n_frames :
        Code length, in frames (> 0). It must be large enough to draw ``n_commands``
        distinct non-constant codes (``base ** n_frames`` well above ``n_commands``).
    base :
        Number of stimulation levels (>= 2). Default ``2`` (binary).
    p :
        Probability of a ``1`` per frame for a binary code (``0 < p < 1``, default
        ``0.5`` for a balanced code). Ignored when ``base > 2``.
    seed :
        Controls the random draw. Give a value for reproducible output.
    commands_info :
        Commands to update in place (exactly ``n_commands``). Pass ``None`` to build a
        fresh dict, with ``extra = {'code_index': ...}``.

    Returns
    -------
    dict of {str: CommandInfo}
        ``uid -> CommandInfo``. Each ``code`` has length ``n_frames``.

    Raises
    ------
    ValueError
        If an argument is out of range, or if ``n_commands`` distinct non-constant codes
        cannot be drawn at the given ``n_frames`` / ``base``.

    Examples
    --------
    >>> import numpy as np
    >>> from medusa.pipelines.bci.vep_spellers.encoding import generate_random_codebook
    >>> cmds = generate_random_codebook(4, n_frames=63, seed=0)
    >>> len(cmds)
    4
    >>> np.asarray(cmds["0"].code).shape       # (n_codes, n_frames)
    (1, 63)
    >>> {len(set(np.asarray(c.code)[0].tolist())) for c in cmds.values()} == {2}
    True
    """
    if not (isinstance(n_commands, (int, np.integer)) and n_commands > 0):
        raise ValueError(f"n_commands must be a positive integer, got {n_commands!r}.")
    if not (isinstance(n_frames, (int, np.integer)) and n_frames > 0):
        raise ValueError(f"n_frames must be a positive integer, got {n_frames!r}.")
    if not (isinstance(base, (int, np.integer)) and base >= 2):
        raise ValueError(f"base must be an integer >= 2, got {base!r}.")
    if base == 2 and not 0 < p < 1:
        raise ValueError(f"p must satisfy 0 < p < 1, got {p!r}.")

    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)

    def draw_row():
        if base == 2:
            return (rng.random(n_frames) < p).astype(np.uint8)
        return rng.integers(0, base, size=n_frames).astype(np.uint8)

    codebook = np.zeros((n_commands, n_frames), dtype=np.uint8)
    seen: "set[bytes]" = set()
    max_tries = 1000 * n_commands
    for i in range(n_commands):
        for _ in range(max_tries):
            row = draw_row()
            if row.min() == row.max():
                continue                        # constant code: undecodable (zero variance)
            key = row.tobytes()
            if key in seen:
                continue                        # keep codes distinct
            seen.add(key)
            codebook[i] = row
            break
        else:
            raise ValueError(
                f"could not draw {n_commands} distinct non-constant random codes at "
                f"n_frames={n_frames}, base={base}; increase n_frames or lower n_commands.")
    extras = [{'code_index': i} for i in range(n_commands)]
    return _make_commands_info(commands_info, codebook, extras)


# --------------------------------------------------------------------------- #
# Visualisation
# --------------------------------------------------------------------------- #
def plot_codebook(commands_info: "dict[str, CommandInfo]",
                  ax: "Axes | None" = None, cmap: str = 'gray',
                  label: "str | Callable | None" = 'uid', title: "str | None" = None,
                  n_codes: "int | None" = None) -> "Axes":
    """Draw a ``{uid: CommandInfo}`` codebook as a black-and-white image.

    Every command owns a ``(n_codes, n_frames)`` ``code``: one row per selectable code,
    each a per-frame pattern (dark = level 0, light = max level). Each command's codes are
    stacked into one block of image rows. A single-code paradigm (c-VEP, SSVEP; ``n_codes
    == 1``) draws one row per command. A multi-code paradigm (an RCP/P300 speller, one code
    per re-randomised cycle) draws a labelled block per command. Commands need not have the
    same number of codes. When any command has more than one row, thin separators mark the
    block boundaries.

    Parameters
    ----------
    commands_info :
        Commands whose codes are stacked into image rows (in iteration order). All shown
        codes must have the same frame length (``n_frames``). The number of codes per
        command may differ.
    ax :
        Target axes. A new figure is created when this is ``None``.
    cmap :
        Matplotlib colormap (default ``'gray'``).
    label :
        Where each command's y-axis label comes from (one label centred on each command's
        block). Use ``'uid'``, ``'content'``, or a callable that takes a command and
        returns a string. ``None`` hides the ticks.
    title :
        Axes title.
    n_codes :
        How many of each command's codes to draw, counted from the first. Clamped to the
        codes a command actually has, so a command with fewer simply shows fewer. ``None``
        (default) draws *all* codes of every command.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn on.

    Raises
    ------
    ValueError
        If ``commands_info`` is empty, ``n_codes`` is not a positive integer or ``None``,
        the shown codes differ in frame length, or ``label`` is invalid.
    """
    import matplotlib.pyplot as plt

    if not commands_info:
        raise ValueError("commands_info is empty; nothing to plot.")
    if n_codes is not None and not (
            isinstance(n_codes, (int, np.integer)) and n_codes > 0):
        raise ValueError(
            f"n_codes must be a positive integer or None, got {n_codes!r}.")

    commands = list(commands_info.values())
    # Stack each command's first ``n_codes`` codes (all if None) into a block of rows.
    rows, block_sizes = [], []
    for cmd in commands:
        code = np.atleast_2d(np.asarray(cmd.code))       # (n_codes, n_frames)
        k = code.shape[0] if n_codes is None else min(int(n_codes), code.shape[0])
        rows.extend(code[:k])
        block_sizes.append(k)

    lengths = {len(r) for r in rows}
    if len(lengths) != 1:
        raise ValueError(
            f"all displayed codes must have equal frame length, got lengths "
            f"{sorted(lengths)}.")
    codebook = np.array(rows)

    if ax is None:
        _, ax = plt.subplots(figsize=(10, max(2.0, 0.4 * len(rows))))

    ax.imshow(codebook, cmap=cmap, aspect='auto', interpolation='nearest',
              vmin=0, vmax=max(1, int(codebook.max())))
    ax.set_xlabel('Frame')
    ax.set_ylabel('Command')
    if title is not None:
        ax.set_title(title)

    # Row offsets of every command block; one centre tick per command, separators
    # between blocks only when at least one command contributes more than one row.
    offsets = np.cumsum([0, *block_sizes])
    centres = (offsets[:-1] + offsets[1:] - 1) / 2
    if max(block_sizes) > 1:
        for edge in offsets[1:-1]:
            ax.axhline(edge - 0.5, color=_SEPARATOR_COLOR, linewidth=0.8)

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
                f"label must be 'uid', 'content', a callable or None, got {label!r}.")
        ax.set_yticks(centres)
        ax.set_yticklabels(labels)
    return ax
