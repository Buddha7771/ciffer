from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import ArrayLike, NDArray

    RowIndex: TypeAlias = (
        int | slice | Sequence[int] | NDArray[np.integer] | NDArray[np.bool_]
    )
    """Anything that selects rows: an integer, a slice, indices, or a mask."""


class Table:
    """Named numpy columns of equal length, with edge lists between rows.

    Each column is a numpy array whose first axis indexes rows.

    Parameters
    ----------
    columns: Mapping[str, ArrayLike] | None, optional
        Column name to array. All arrays must have the same length.
    edges: Mapping[str, EdgeList] | None, optional
        Edge lists by name, connecting rows of this table.

    Examples
    --------
    .. code-block:: python

        bond = EdgeList([0, 1], [1, 2], {"order": [1, 2]})
        atoms = Table({"name": ["N", "CA", "C"]}, edges={"bond": bond})

        atoms["name"]            # the stored array
        atoms["b_factor"] = 0.0  # add a column
        del atoms["b_factor"]

    """

    __module__ = "ciffer"

    def __init__(
        self,
        columns: Mapping[str, ArrayLike] | None = None,
        *,
        edges: Mapping[str, EdgeList] | None = None,
    ) -> None:
        self._columns: dict[str, NDArray[Any]] = {}
        self.edges: dict[str, EdgeList] = dict(edges or {})

        for name, value in (columns or {}).items():
            self[name] = value

        self.check_edges()

    @property
    def columns(self) -> list[str]:
        """Column names in insertion order."""
        return list(self._columns)

    def __len__(self) -> int:
        """Return the number of rows."""
        if not self._columns:
            return 0
        first = next(iter(self._columns.values()))
        return len(first)

    def __contains__(self, name: str) -> bool:
        """Check if a column exists."""
        return name in self._columns

    def __getitem__(self, name: str) -> NDArray[Any]:
        """Return the stored array of a column."""
        return self._columns[name]

    def __setitem__(self, name: str, value: ArrayLike) -> None:
        """Add or replace a column."""
        array = np.asarray(value)

        if array.ndim == 0:
            if not self._columns:
                msg = f"Column {name!r} needs an array; the table has no columns yet."
                raise ValueError(msg)
            if name in self._columns:
                array = np.full_like(self._columns[name], array)
            else:
                array = np.full(len(self), array)

        if self._columns and len(array) != len(self):
            msg = (
                f"Column {name!r} has {len(array)} rows, "
                f"but the table has {len(self)} rows."
            )
            raise ValueError(msg)

        self._columns[name] = array

    def __delitem__(self, name: str) -> None:
        """Remove a column. A table with rows keeps at least one column."""
        if name in self._columns and len(self._columns) == 1 and len(self):
            msg = f"Cannot delete {name!r}, the last column of a table with rows."
            raise ValueError(msg)
        del self._columns[name]

    def take(self, indices: RowIndex) -> Table:
        """Return a new table with the rows at ``indices``.

        Parameters
        ----------
        indices: RowIndex
            Rows to take: integers, a slice, or a boolean mask. Rows may repeat
            and come in any order.

        Returns
        -------
        Table
            A table that shares no arrays with this one.

        Notes
        -----
        Edges with both ends taken are kept and renumbered; others are dropped.

        Examples
        --------
        .. code-block:: python

            atoms.take([2, 0])
            atoms.take(atoms["name"] == "CA")

        """

        def _take_edges(edges: EdgeList) -> EdgeList:
            src = old_to_new[edges.src]
            dst = old_to_new[edges.dst]
            kept = (src >= 0) & (dst >= 0)
            columns = {name: edges[name][kept] for name in edges.columns}
            return EdgeList(src[kept], dst[kept], columns)

        self.check_edges()  # edges may have been added after construction
        rows = as_indices(np.atleast_1d(np.arange(len(self))[indices]))

        # Old row index to new row index: -1 if not taken, the first copy if repeated.
        old_to_new = np.full(len(self), -1)
        unique_rows, first = np.unique(rows, return_index=True)
        old_to_new[unique_rows] = first

        return Table(
            {name: column[rows] for name, column in self._columns.items()},
            edges={name: _take_edges(edges) for name, edges in self.edges.items()},
        )

    def copy(self) -> Table:
        """Return a deep copy of the table."""
        return self.take(slice(None))

    def check_edges(self) -> None:
        """Raise if an edge group is not an EdgeList or points outside the rows."""
        for name, edges in self.edges.items():
            if not isinstance(edges, EdgeList):
                kind = type(edges).__name__
                msg = f"Edge group {name!r} must be an EdgeList, got {kind}."
                raise TypeError(msg)

            for end in (edges.src, edges.dst):
                if end.size and (end.min() < 0 or end.max() >= len(self)):
                    msg = f"Edge group {name!r} points outside the {len(self)} rows."
                    raise IndexError(msg)

    def __repr__(self) -> str:
        """Return the number of rows and the schema."""
        lines = _schema({name: self[name] for name in self.columns})
        if self.edges:
            lines.append("  edges")
        for name, edges in self.edges.items():
            lines.append(f"    {name}  {len(edges)} edges")
            lines.extend(f"    {line}" for line in repr(edges).splitlines()[1:])

        header = f"<{self.__class__.__name__} rows={len(self)}>"
        return "\n".join([header, *lines])


class EdgeList:
    """Edges between rows of a table, such as bonds between atoms.

    Edge ``i`` connects row ``src[i]`` to row ``dst[i]`` and can carry values,
    such as a bond order.

    Parameters
    ----------
    src: ArrayLike
        Row index of the first end of each edge.
    dst: ArrayLike
        Row index of the second end of each edge.
    columns: Mapping[str, ArrayLike] | None, optional
        Column name to array, one value per edge.

    Examples
    --------
    .. code-block:: python

        bond = EdgeList([0, 1], [1, 2], {"order": [1, 2]})
        bond.src                 # the stored array
        bond["order"]

    """

    __module__ = "ciffer"

    def __init__(
        self,
        src: ArrayLike,
        dst: ArrayLike,
        columns: Mapping[str, ArrayLike] | None = None,
    ) -> None:
        self._src = as_indices(src)
        self._dst = self._check_length("dst", as_indices(dst))
        self._values: dict[str, NDArray[Any]] = {}

        for name, value in (columns or {}).items():
            self[name] = value

    def __setattr__(self, name: str, value: Any) -> None:
        """Refuse new attributes, so that a misspelled field does not pass silently."""
        in_init = name in ("_src", "_dst", "_values")  # what __init__ sets
        # Fields of subclasses, such as Bond.order, define __set__.
        settable = hasattr(getattr(type(self), name, None), "__set__")
        if not in_init and not settable:
            msg = (
                f"{type(self).__name__} has no {name!r} field; "
                f"set other columns with edges[{name!r}] = value."
            )
            raise AttributeError(msg)
        super().__setattr__(name, value)

    @property
    def src(self) -> NDArray[np.intp]:
        """Row index of the first end of each edge."""
        return self._src

    @property
    def dst(self) -> NDArray[np.intp]:
        """Row index of the second end of each edge."""
        return self._dst

    @property
    def columns(self) -> list[str]:
        """Value column names in insertion order."""
        return list(self._values)

    def __len__(self) -> int:
        """Return the number of edges."""
        return len(self._src)

    def __contains__(self, name: str) -> bool:
        """Check if a value column exists."""
        return name in self._values

    def __getitem__(self, name: str) -> NDArray[Any]:
        """Return the stored array of a value column."""
        return self._values[name]

    def __setitem__(self, name: str, value: ArrayLike) -> None:
        """Add or replace a value column."""
        array = np.asarray(value)
        if array.ndim == 0:
            if name in self._values:
                array = np.full_like(self._values[name], array)
            else:
                array = np.full(len(self), array)
        self._values[name] = self._check_length(name, array)

    def __delitem__(self, name: str) -> None:
        """Remove a value column."""
        del self._values[name]

    def __repr__(self) -> str:
        """Return the number of edges and the schema of the value columns."""
        lines = _schema({name: self[name] for name in self.columns})
        return "\n".join([f"<{type(self).__name__} edges={len(self)}>", *lines])

    def _check_length(self, name: str, array: NDArray[Any]) -> NDArray[Any]:
        if len(array) != len(self):
            msg = f"{name} has {len(array)} entries, but there are {len(self)} edges."
            raise ValueError(msg)
        return array


def as_indices(value: ArrayLike) -> NDArray[np.intp]:
    """Return ``value`` as a 1-D intp array, or raise TypeError."""
    index = np.asarray(value)
    if index.size == 0:  # an empty list arrives as float
        index = index.astype(np.intp)

    if index.ndim != 1 or not np.issubdtype(index.dtype, np.integer):
        msg = f"Expected 1-D integer indices, got {index.ndim}-D {index.dtype}."
        raise TypeError(msg)
    return index.astype(np.intp, copy=False)


def _schema(columns: Mapping[str, NDArray[Any]]) -> list[str]:
    """Return one line per column with its dtype, and its shape after the first axis."""
    width = max((len(name) for name in columns), default=0)
    lines = []
    for name, column in columns.items():
        shape = f" {column.shape[1:]}" if column.ndim > 1 else ""
        lines.append(f"  {name:<{width}}  {column.dtype}{shape}")
    return lines
