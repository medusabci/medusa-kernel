"""Live-acquisition streaming + HDF5 persistence for a
:class:`~medusa.core.data.recording.Recording`.

The :class:`Recorder` handles a recording's data **as it is acquired**. Each
:meth:`Recorder.append_data` chunk goes to two optional sinks:

* an in-RAM **buffer** (on by default): it grows ``recording.signals[key]`` in
  place and trims it to the last ``buffer`` seconds, so ``recording.signals[key]``
  is a live, bounded :class:`Signal` you process directly. RAM stays flat over a
  long run.
* **continuous persistence** to one self-contained ``.h5`` (enabled by passing a
  ``file``): a binary, growable, crash-safe format.

Reading a file back is ``Recording.load(path)`` (engine: :func:`read_recording`);
there is no reader class. In the ``.h5`` each ``Signal``'s ``signal``/``times``
are native chunked, growable, gzip datasets; everything else (BIDS identity,
sidecars, channel sets, ``experiment``, events) rides as one JSON blob in
``/meta``. Only :class:`Signal` streams are supported.

On-disk layout (versioned by ``schema_version``)::

    recording.h5
    |-- attrs: schema_version, subject, session, task, run, acquisition
    |-- /meta                JSON string = to_serializable_obj() minus stream arrays
    |-- /data/<key>/signal   [n_samples x n_channels]  chunked, growable, gzip
    |-- /data/<key>/times    [n_samples]               growable; present iff appended
    +-- /events              [n_events] vlen-UTF8; one JSON record per event, growable

Crash-resilience: ``/meta`` is written up front and every append flushes, so a
file left unclosed by a crash reads back as a shorter-but-valid recording.
"""

import json
import math
from collections.abc import Mapping

import numpy as np
import h5py

from medusa.core.data.recording import Recording
from medusa.core.data.signal import Signal

__all__ = ["Recorder", "read_recording", "DEFAULT_BUFFER_SECONDS"]

_META = "meta"
_EVENTS = "events"

#: Default in-RAM buffer length (seconds) when ``buffer`` is left unset. Cheap
#: (~5 MB at 1 kHz x 64 ch x float64) and a safe margin for typical processing
#: windows; override per :class:`Recorder`, or pass ``buffer=None`` to disable.
DEFAULT_BUFFER_SECONDS = 10.0


def _json_default(o):
    """Fallback encoder: numpy scalars/arrays -> native Python for ``json.dumps``."""
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"object of type {type(o).__name__} is not JSON-serializable")


def _none_if_nan(v):
    """Map a NaN float (incl. numpy) to ``None`` so events round-trip as JSON null."""
    if v is None:
        return None
    try:
        if isinstance(v, (float, np.floating)) and math.isnan(float(v)):
            return None
    except (TypeError, ValueError):
        pass
    return v


def _read_text(dataset) -> str:
    """Read a scalar string dataset as ``str`` (h5py returns ``bytes`` for vlen)."""
    value = dataset[()]
    return value.decode("utf-8") if isinstance(value, (bytes, bytearray)) else value


class Recorder:
    """Live recorder: buffers into the recording's Signals + optional file write.

    The :class:`Recording` is the single source of structure -- every stream must
    already be registered (e.g. with :meth:`Signal.empty` before acquisition, or
    full signals for a one-shot save), and the events schema set via
    ``recording.set_events(Events(...))`` before any :meth:`append_events`.

    When buffering is on, the recorder **fills the recording's own Signals in
    place**, trimmed to the last ``buffer`` seconds -- so ``recording.signals[key]``
    is the live, bounded window you process. The complete stream goes to the file
    (when recording); load it back with ``Recording.load(path)``.

    Parameters
    ----------
    recording
        The :class:`Recording` whose structure defines the streams. Only
        :class:`Signal` entries are supported. Its Signals are mutated in place as
        the bounded buffer (when buffering is on).
    buffer
        In-RAM buffer length in **seconds** (per stream; the recording keeps the
        last ``round(seconds * fs)`` samples). Default
        :data:`DEFAULT_BUFFER_SECONDS`; pass ``None`` (or ``0``) to disable
        buffering (the Signals stay empty -- e.g. for pure file logging).
    file
        Output ``.h5`` path. Pass a path to **persist** the full stream
        continuously to disk; leave it ``None`` for an in-RAM buffer only.
    compression, overwrite, autoflush
        HDF5 dataset compression (default ``"gzip"``); refuse to clobber an
        existing file unless ``overwrite``; flush after every append (default).

    Raises
    ------
    ValueError
        If neither buffering nor a ``file`` is given (the recorder would do
        nothing).
    NotImplementedError
        If ``recording.data`` holds a non-:class:`Signal` modality.
    """

    def __init__(self, recording: Recording, *,
                 buffer: "float | None" = DEFAULT_BUFFER_SECONDS,
                 file=None, compression="gzip", overwrite: bool = False,
                 autoflush: bool = True) -> None:
        if not isinstance(recording, Recording):
            raise TypeError("`recording` must be a Recording instance.")
        for key, data in recording.data.items():
            if "/" in key:
                raise ValueError(
                    f"data key {key!r} must not contain '/' (it names an HDF5 "
                    "group).")
            if not isinstance(data, Signal):
                raise NotImplementedError(
                    f"Recorder supports Signal streams only in 2.0; data[{key!r}] "
                    f"is a {type(data).__name__}.")

        self._buffering = buffer is not None and buffer > 0
        self._persisting = file is not None     # persistence is on iff a file is given
        if not self._buffering and not self._persisting:
            raise ValueError(
                "Recorder does nothing: enable buffering (buffer=<seconds>) "
                "and/or persistence (pass a `file`).")

        self.recording = recording
        self._compression = compression
        self._autoflush = autoflush
        self._closed = False

        # Per-stream buffer capacity (samples), derived from seconds x fs.
        self._capacity: "dict[str, int]" = {}
        if self._buffering:
            self._capacity = {key: max(1, round(buffer * sig.fs))
                              for key, sig in recording.signals.items()}

        # File datasets (only when persisting).
        self._file = None
        self._signal_ds: "dict[str, h5py.Dataset]" = {}
        self._times_ds: "dict[str, h5py.Dataset]" = {}
        self._tracks_times: "dict[str, bool]" = {}
        self._events_ds: "h5py.Dataset | None" = None
        self._event_columns: "list[str] | None" = None
        if self._persisting:
            self._file = h5py.File(str(file), "w" if overwrite else "x")
            try:
                self._init_file()
            except Exception:
                self._file.close()
                self._closed = True
                raise

    # ------------------------------------------------------------------
    # File layout (only reached when recording)
    # ------------------------------------------------------------------
    def _init_file(self) -> None:
        rec, f = self.recording, self._file

        f.attrs["schema_version"] = rec.schema_version
        for name, value in rec.bids.entities.items():
            f.attrs[name] = value

        meta = rec.to_serializable_obj()
        for key in rec.signals:
            component_data = meta["data"][key]["component_data"]
            component_data.pop("signal", None)
            component_data.pop("times", None)
        if meta.get(_EVENTS) is not None:
            meta[_EVENTS]["data"] = {}
        f.create_dataset(_META, data=json.dumps(meta))

        for key, sig in rec.signals.items():
            n_ch = sig.channel_set.n_channels
            dataset = f.create_dataset(
                f"data/{key}/signal", shape=(0, n_ch), maxshape=(None, n_ch),
                dtype=sig.signal.dtype, chunks=True, compression=self._compression)
            dataset.attrs["fs"] = sig.fs
            self._signal_ds[key] = dataset

        if rec.events is not None:
            self._event_columns = list(rec.events.column_names)
            self._events_ds = f.create_dataset(
                _EVENTS, shape=(0,), maxshape=(None,),
                dtype=h5py.string_dtype("utf-8"), chunks=True,
                compression=self._compression)

        # Persist whatever the recording already holds (one-shot save path).
        for key, sig in rec.signals.items():
            if sig.n_samples > 0:
                self._append_to_disk(key, sig.signal, sig.times)
        if rec.events is not None and rec.events.n_events > 0:
            existing = rec.events.df.where(rec.events.df.notna(), None)
            self._append_event_records(existing.to_dict("records"))
        self.flush()

    # ------------------------------------------------------------------
    # Streaming append (fills the in-RAM Signal and/or the file)
    # ------------------------------------------------------------------
    def append_data(self, key: str, chunk, times=None) -> "Recorder":
        """Append a ``[n_new x n_channels]`` block to stream ``key``.

        When buffering, grows ``recording.signals[key]`` in place and trims it to
        the last ``buffer`` seconds (so the recording's own Signal is the live
        window). When persisting, also writes the block to the file. ``times``
        (per-sample seconds) is used by both: **the buffered Signal keeps the real
        timestamps you pass** -- key for precise cross-stream synchronization --
        and only synthesizes a continuation of its own clock when ``times`` is
        omitted; the file stores a ``times`` dataset iff you pass it on the first
        append.
        """
        self._check_open()
        if key not in self.recording.signals:
            raise KeyError(
                f"unknown stream {key!r}; streams: {list(self.recording.signals)}.")
        chunk = np.asarray(chunk)
        n_ch = self.recording.signals[key].channel_set.n_channels
        if chunk.ndim != 2:
            raise ValueError(
                f"chunk for {key!r} must be 2-D [n_samples x n_channels]; got "
                f"shape {chunk.shape} (reshape a single sample to (1, n_ch)).")
        if chunk.shape[1] != n_ch:
            raise ValueError(
                f"chunk for {key!r} has {chunk.shape[1]} channels but the stream "
                f"has {n_ch}.")

        if self._buffering:
            self._buffer_append(key, chunk, times)
        if self._persisting:
            self._append_to_disk(key, chunk, times)
            if self._autoflush:
                self.flush()
        return self

    def _buffer_append(self, key: str, chunk: np.ndarray, times) -> None:
        """Grow ``recording.signals[key]`` in place, trimmed to the buffer length."""
        sig = self.recording.data[key]
        cap = self._capacity[key]
        has = sig.signal.shape[0] > 0
        if times is None:                              # continue the stream's clock
            step = 1.0 / sig.fs
            start = (sig.times[-1] + step) if has else 0.0
            times = start + np.arange(chunk.shape[0]) / sig.fs
        else:
            times = np.asarray(times)

        new_signal = np.vstack([sig.signal, chunk]) if has else np.array(chunk)
        new_times = np.concatenate([sig.times, times]) if has else np.asarray(times)
        if new_signal.shape[0] > cap:
            new_signal = new_signal[-cap:].copy()
            new_times = new_times[-cap:].copy()
        sig.signal = new_signal
        sig.times = new_times

    def _append_to_disk(self, key: str, chunk, times) -> None:
        dataset = self._signal_ds[key]
        chunk = np.asarray(chunk)
        n_new = chunk.shape[0]

        t = None
        if times is not None:
            t = np.asarray(times)
            if t.ndim != 1 or t.shape[0] != n_new:
                raise ValueError(
                    f"times for {key!r} must be 1-D of length {n_new}; got shape "
                    f"{t.shape}.")

        tracks = self._tracks_times.get(key)
        if tracks is None:                       # first recorded append decides
            tracks = t is not None
            self._tracks_times[key] = tracks
            if tracks:
                self._times_ds[key] = self._file.create_dataset(
                    f"data/{key}/times", shape=(0,), maxshape=(None,),
                    dtype=t.dtype, chunks=True, compression=self._compression)
        if tracks and t is None:
            raise ValueError(
                f"stream {key!r} tracks times; pass `times` to every append_data.")
        if not tracks and t is not None:
            raise ValueError(
                f"stream {key!r} was first persisted without times; it cannot start "
                "tracking them now.")

        n0 = dataset.shape[0]
        dataset.resize(n0 + n_new, axis=0)
        dataset[n0:] = chunk
        if tracks:
            times_ds = self._times_ds[key]
            m0 = times_ds.shape[0]
            times_ds.resize(m0 + n_new, axis=0)
            times_ds[m0:] = t

    def append_events(self, events: "Mapping | list | None" = None,
                      **fields) -> "Recorder":
        """Append one event (or a list) to the ``/events`` table (recording only).

        Pass a single record as keyword args (``append_events(onset=1.0,
        duration=0.5, trial_type="target")``), a dict, or a list of dicts. Each
        record must hold **exactly** the columns of the ``recording.events`` schema.
        Requires a ``file`` (events are written to the file, not buffered).
        """
        self._check_open()
        if not self._persisting:
            raise ValueError(
                "append_events needs a `file` (events are written to the file, "
                "not the in-RAM buffer).")
        if self._events_ds is None:
            raise ValueError(
                "no events schema in the recording; call "
                "recording.set_events(Events(...)) before writing events.")
        if events is None:
            records = [fields]
        elif isinstance(events, Mapping):
            records = [dict(events)]
        else:
            records = [dict(r) for r in events]
        self._append_event_records(records)
        if self._autoflush:
            self.flush()
        return self

    def _append_event_records(self, records: "list[Mapping]") -> None:
        expected = set(self._event_columns)
        rows = []
        for record in records:
            if set(record) != expected:
                raise ValueError(
                    f"each event must have exactly the columns "
                    f"{sorted(expected)}; got {sorted(record)} "
                    f"(missing={sorted(expected - set(record))}, "
                    f"unexpected={sorted(set(record) - expected)}).")
            ordered = {c: _none_if_nan(record[c]) for c in self._event_columns}
            rows.append(json.dumps(ordered, default=_json_default))
        if not rows:
            return
        dataset = self._events_ds
        n0 = dataset.shape[0]
        dataset.resize(n0 + len(rows), axis=0)
        dataset[n0:] = rows

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def flush(self) -> None:
        """Flush buffered file writes to disk (no-op when not recording)."""
        if self._file is not None and not self._closed:
            self._file.flush()

    def close(self) -> None:
        """Flush and close the file, if any (idempotent)."""
        if not self._closed:
            if self._file is not None:
                self._file.flush()
                self._file.close()
            self._closed = True

    def __enter__(self) -> "Recorder":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False

    def _check_open(self) -> None:
        if self._closed:
            raise ValueError("Recorder is closed.")


def read_recording(path) -> Recording:
    """Read an HDF5 file written by :class:`Recorder` into a Recording.

    The single read engine behind ``Recording.load(path)`` for ``.h5``/``.hdf5``
    (there is no reader class). Loads ``/meta``, splices each stream's ``signal``
    (and ``times`` if stored) datasets back into the serialized dict, rebuilds the
    event rows, and defers to :meth:`Recording.from_serializable_obj` -- so every
    component is reconstructed by its own deserializer, with only the arrays
    special-cased. Works on an unfinalized (crashed) file too: it just reads fewer
    samples.
    """
    with h5py.File(str(path), "r") as f:
        meta = json.loads(_read_text(f[_META]))

        for key in list((meta.get("data") or {})):
            component_data = meta["data"][key]["component_data"]
            component_data["signal"] = f[f"data/{key}/signal"][()]
            times_path = f"data/{key}/times"
            component_data["times"] = (
                f[times_path][()] if times_path in f else None)

        events_meta = meta.get(_EVENTS)
        if events_meta is not None:
            records = _read_event_records(f)
            columns = list(events_meta.get("dtypes") or {})
            events_meta["data"] = {c: [r.get(c) for r in records] for c in columns}

        return Recording.from_serializable_obj(meta)


def _read_event_records(f) -> "list[dict]":
    if _EVENTS not in f:
        return []
    records = []
    for row in f[_EVENTS][:]:
        if isinstance(row, (bytes, bytearray)):
            row = row.decode("utf-8")
        records.append(json.loads(row))
    return records
