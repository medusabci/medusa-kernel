"""BIDS ``events.tsv``-aligned event timeline.

An :class:`Events` wraps a :class:`pandas.DataFrame` whose rows are events and
whose columns are the always-present ``onset``/``duration`` (seconds, against the
recording origin shared by every :class:`~medusa.core.data.signal.Signal`) plus
the typed optional columns declared at construction. The DataFrame is the source
of truth (:attr:`Events.df`); the class adds a fixed, typed column schema,
record-oriented :meth:`append`, value-based :meth:`select`, an explicit
:meth:`sort` (out-of-order onsets are warned, never forced), the ``events.json``
``descriptions`` dict, and BIDS/serialization projections.
"""

import warnings
from collections.abc import Mapping

import numpy as np
import pandas as pd

from medusa.core.serialization import SerializableComponent, ensure_list

__all__ = ["Events"]

# Always-present required columns (float seconds); never in ``optional_columns``.
_REQUIRED_COLUMNS = ("onset", "duration")


def _normalize_dtype(dtype) -> str:
    """Normalise a user dtype spec to a pandas-friendly dtype string.

    Python ``str`` -> ``"object"`` (avoids numpy fixed-width truncation on
    append); ``int``/``float``/``bool`` -> their 64-bit numpy names; numpy dtypes
    -> their name; pandas extension strings (``"Int64"``, ``"category"`` ...)
    pass through unchanged.
    """
    if dtype is str or dtype in ("str", "unicode"):
        return "object"
    if dtype is int:
        return "int64"
    if dtype is float:
        return "float64"
    if dtype is bool:
        return "bool"
    if isinstance(dtype, str):
        return dtype
    npd = np.dtype(dtype)
    return "object" if npd.kind in ("U", "S") else npd.name


class Events(SerializableComponent):
    """A DataFrame-backed event timeline: ``onset`` + ``duration`` + typed columns.

    Parameters
    ----------
    optional_columns
        ``{name: dtype}`` for the optional columns, in ``events.tsv`` order.
        ``dtype`` is a Python type (``str``/``int``/``float``/``bool``), a numpy
        dtype, or a pandas dtype string (``"Int64"``, ``"category"`` ...);
        ``str`` maps to ``object``. ``onset``/``duration`` are implicit (float)
        and must not be listed.
    descriptions
        Optional ``{column: descriptor}`` for the BIDS ``events.json`` sidecar
        (a BIDS column object or plain string). Keys must be existing columns.
        Stored verbatim, separate from the dtype schema.

    Raises
    ------
    ValueError
        On reserved column names, or a ``descriptions`` key that is not a column.

    Examples
    --------
    >>> ev = Events(optional_columns={"trial_type": str, "value": int},
    ...             descriptions={"value": "Stimulus code"})
    >>> _ = ev.append({"onset": 1.0, "duration": 0.5,
    ...                "trial_type": "target", "value": 1})
    >>> _ = ev.append([
    ...     {"onset": 2.5, "duration": 0.5, "trial_type": "nontarget", "value": 0},
    ...     {"onset": 4.0, "duration": 0.5, "trial_type": "target", "value": 1}])
    >>> ev.select(trial_type="target").n_events
    2
    >>> ev.df["value"].dtype.name
    'int64'
    """

    def __init__(self, optional_columns: "dict | None" = None,
                 descriptions: "dict | None" = None) -> None:
        optional_columns = dict(optional_columns) if optional_columns else {}
        reserved = [c for c in optional_columns if c in _REQUIRED_COLUMNS]
        if reserved:
            raise ValueError(
                f"{reserved} are always-present and must not be in "
                f"`optional_columns`.")
        # Declared dtype per column (string form); the single schema source.
        self._dtypes = {c: "float64" for c in _REQUIRED_COLUMNS}
        self._dtypes.update({name: _normalize_dtype(dt)
                             for name, dt in optional_columns.items()})
        self._df = pd.DataFrame({c: pd.Series(dtype=dt)
                                 for c, dt in self._dtypes.items()})

        descriptions = dict(descriptions) if descriptions else {}
        unknown = set(descriptions) - set(self._dtypes)
        if unknown:
            raise ValueError(
                f"`descriptions` reference undeclared columns {sorted(unknown)}; "
                f"declare them in `optional_columns` first.")
        self.descriptions = descriptions

    def _apply_dtypes(self) -> None:
        """Cast every column to its declared dtype (enforces the schema)."""
        for column, dtype in self._dtypes.items():
            self._df[column] = self._df[column].astype(dtype)

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------
    @property
    def df(self) -> pd.DataFrame:
        """The underlying DataFrame (source of truth; full pandas available).

        Editing it directly is allowed but bypasses the schema/dtype/order checks
        that :meth:`append` applies.
        """
        return self._df

    @property
    def n_events(self) -> int:
        """Number of events (rows)."""
        return len(self._df)

    def __len__(self) -> int:
        return len(self._df)

    @property
    def column_names(self) -> "list[str]":
        """Columns in ``events.tsv`` order (``onset``/``duration`` first)."""
        return list(self._dtypes)

    def __repr__(self) -> str:
        return f"Events(n_events={self.n_events}, columns={self.column_names})"

    # ------------------------------------------------------------------
    # Build / query
    # ------------------------------------------------------------------
    def append(self, events: "Mapping | list[Mapping]") -> "Events":
        """Append one record (a ``{column: value}`` dict) or a list of them.

        Pandas "records" orientation (``df.to_dict("records")``). Each record must
        hold **exactly** the columns; pass an explicit ``None``/``NaN`` for an
        unknown value. Values are cast to the declared dtypes. Returns ``self``.

        Raises
        ------
        ValueError
            If a record's keys do not match the columns exactly.

        Warns
        -----
        UserWarning
            If the result is not ordered by onset (see :meth:`sort`).
        """
        records = [events] if isinstance(events, Mapping) else list(events)
        expected = set(self._dtypes)
        for r in records:
            if set(r) != expected:
                raise ValueError(
                    f"each event must have exactly the columns {sorted(expected)}; "
                    f"got {sorted(r)} (missing={sorted(expected - set(r))}, "
                    f"unexpected={sorted(set(r) - expected)}).")
        if not records:
            return self
        new = pd.DataFrame(records, columns=self.column_names)
        self._df = new if self._df.empty else pd.concat(
            [self._df, new], ignore_index=True)
        self._apply_dtypes()
        self._warn_if_unordered()
        return self

    def select(self, **conditions) -> "Events":
        """Return a new :class:`Events` of rows matching ``column=value`` filters.

        Multiple conditions combine with AND; a list/tuple/set value matches any
        of its members (``isin``). The typed schema and ``descriptions`` are
        preserved.

        Examples
        --------
        >>> ev = Events(optional_columns={"trial_type": str}).append(
        ...     [{"onset": 0.0, "duration": 0.1, "trial_type": "target"},
        ...      {"onset": 1.0, "duration": 0.1, "trial_type": "nontarget"}])
        >>> ev.select(trial_type="target").n_events
        1
        """
        mask = pd.Series(True, index=self._df.index)
        for column, value in conditions.items():
            if column not in self._dtypes:
                raise KeyError(f"unknown column {column!r}")
            series = self._df[column]
            mask &= (series.isin(value) if isinstance(value, (list, tuple, set))
                     else series == value)
        out = Events(optional_columns={c: self._dtypes[c]
                                       for c in self.column_names
                                       if c not in _REQUIRED_COLUMNS},
                     descriptions=self.descriptions)
        out._df = self._df[mask].reset_index(drop=True)
        return out

    def sort(self) -> "Events":
        """Sort rows in place by non-decreasing onset (stable); returns ``self``."""
        self._df = self._df.sort_values("onset", kind="stable").reset_index(
            drop=True)
        return self

    def _warn_if_unordered(self) -> None:
        if not self._df["onset"].is_monotonic_increasing:
            warnings.warn("events are not ordered by onset; call .sort() to "
                          "order them.", stacklevel=3)

    # ------------------------------------------------------------------
    # BIDS projection
    # ------------------------------------------------------------------
    def to_dataframe(self) -> pd.DataFrame:
        """BIDS ``events.tsv`` view: a copy with missing values filled ``"n/a"``.

        Columns are cast to ``object`` so ``"n/a"`` fits regardless of the stored
        dtype (text view; the typed data stays on :attr:`df`).
        """
        return self._df.astype(object).where(self._df.notna(), "n/a")

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame,
                       descriptions: "dict | None" = None) -> "Events":
        """Build from a BIDS ``events.tsv`` table (``"n/a"`` restored to ``NaN``).

        Optional-column dtypes are taken from ``df`` (it already carries them); an
        unsorted source warns (does not raise) -- call :meth:`sort`.
        """
        df = df.replace("n/a", np.nan).reset_index(drop=True)
        extra = [c for c in df.columns if c not in _REQUIRED_COLUMNS]
        events = cls(optional_columns={c: df[c].dtype for c in extra},
                     descriptions=descriptions)
        events._df = df[[*_REQUIRED_COLUMNS, *extra]]
        events._apply_dtypes()
        events._warn_if_unordered()
        return events

    # ------------------------------------------------------------------
    # SerializableComponent contract
    # ------------------------------------------------------------------
    def to_serializable_obj(self) -> dict:
        return {
            "dtypes": dict(self._dtypes),
            "data": self._df.where(self._df.notna(), None).to_dict("list"),
            "descriptions": self.descriptions,
        }

    @classmethod
    def from_serializable_obj(cls, data: dict) -> "Events":
        data = dict(data)
        dtypes = data.get("dtypes") or {}
        optional = {c: dt for c, dt in dtypes.items()
                    if c not in _REQUIRED_COLUMNS}
        events = cls(optional_columns=optional,
                     descriptions=data.get("descriptions"))
        columns_data = data.get("data") or {}
        if columns_data:
            # ensure_list: a single-row table round-trips through .mat with each
            # column collapsed to a scalar (squeeze) -- wrap back to 1-element lists.
            columns_data = {c: ensure_list(v) for c, v in columns_data.items()}
            events._df = pd.DataFrame(columns_data)[events.column_names]
            events._apply_dtypes()
        return events
