from __future__ import annotations

import copy
from enum import IntEnum
from typing import TYPE_CHECKING, Any, Self, overload

import numpy as np

from .table import as_indices

if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy.typing import ArrayLike, NDArray

    from .structure import Structure
    from .table import EdgeList, RowIndex


class Level(IntEnum):
    """Levels of a structure, from the bottom up."""

    ATOM = 0
    RESIDUE = 1
    CHAIN = 2


class View:
    """A selection of atoms, residues, or chains of a structure.

    A view of a whole level, such as ``st.atoms``, reads and writes the stored
    arrays. A view of selected rows, such as ``st.atoms[idx]``, reads copies, and
    assigning to its columns writes the selected rows of the stored arrays.

    Parameters
    ----------
    structure: Structure
        The structure the rows belong to.
    level: Level
        The level of the rows.
    indices: NDArray[np.integer] | None, optional
        Row indices of the view. None means the whole level in stored order.

    Examples
    --------
    .. code-block:: python

        atoms = st.atoms
        ca = atoms[atoms["name"] == "CA"]
        ca["xyz"]                # a copy
        ca["xyz"] = 0.0          # writes into st
        ca.residues              # residues that contain these atoms
        ca.extract()             # a new structure

    """

    def __init__(
        self,
        structure: Structure,
        level: Level,
        indices: NDArray[np.integer] | None = None,
    ) -> None:
        self.structure = structure
        self.level = level
        self.table = structure.tables[level]
        self._indices = indices

    @property
    def indices(self) -> NDArray[np.integer]:
        """Row indices of the view in the level's table."""
        if self._indices is None:
            return np.arange(len(self.table))
        return self._indices

    @property
    def columns(self) -> list[str]:
        """Column names of the level."""
        return self.table.columns

    @property
    def edges(self) -> dict[str, EdgeList]:
        """Edge groups of the level.

        For a view of selected rows, only edges between those rows are kept, as copies
        whose ``src`` and ``dst`` index the rows of the view.
        """
        if self._indices is None:
            return self.table.edges
        return self.table.take(self._indices).edges

    @property
    def atoms(self) -> View:
        """Atoms of the selection, each listed once in stored order."""
        return self._convert(Level.ATOM)

    @property
    def residues(self) -> View:
        """Residues of the selection, each listed once in stored order."""
        return self._convert(Level.RESIDUE)

    @property
    def chains(self) -> View:
        """Chains of the selection, each listed once in stored order."""
        return self._convert(Level.CHAIN)

    def extract(self) -> Structure:
        """Extract a new structure of the rows in the view.

        Returns
        -------
        Structure
            A structure that shares no arrays or metadata with this one.

        Notes
        -----
        Rows follow the stored order, not the selection order.

        Examples
        --------
        .. code-block:: python

            pocket = st.atoms[np.linalg.norm(st.atoms["xyz"] - center, axis=1) < 10]
            pocket.extract()

        """
        atoms, residues, chains = self.atoms, self.residues, self.chains
        st = self.structure

        # Indices are sorted, so a parent's new index is its position among them.
        return st.__class__(
            atoms=atoms.table.take(atoms.indices),
            residues=residues.table.take(residues.indices),
            chains=chains.table.take(chains.indices),
            atom_to_residue=np.searchsorted(
                residues.indices, st.atom_to_residue[atoms.indices]
            ),
            residue_to_chain=np.searchsorted(
                chains.indices, st.residue_to_chain[residues.indices]
            ),
            metadata=copy.deepcopy(st.metadata),
        )

    @overload
    def __getitem__(self, key: str) -> NDArray[Any]: ...

    @overload
    def __getitem__(self, key: RowIndex) -> Self: ...

    def __getitem__(self, key: str | RowIndex) -> NDArray[Any] | Self:
        """Return a column by name, or a new view of the rows selected by key."""
        if isinstance(key, str):
            column = self.table[key]
            return column if self._indices is None else column[self._indices]
        indices = as_indices(np.atleast_1d(self.indices[key]))
        return self.__class__(self.structure, self.level, indices)

    def __setitem__(self, name: str, value: ArrayLike) -> None:
        """Set a whole column, or the selected rows of an existing column."""
        if self._indices is None:
            self.table[name] = value
        elif name in self.table:
            self.table[name][self._indices] = value
        else:
            msg = f"Column {name!r} does not exist; add it on the whole level first."
            raise TypeError(msg)

    def __delitem__(self, name: str) -> None:
        """Remove a column. Only a view of a whole level can do this."""
        if self._indices is not None:
            msg = "Delete columns from a whole level, such as st.atoms."
            raise TypeError(msg)
        del self.table[name]

    def __contains__(self, name: str) -> bool:
        """Check if the level has a column."""
        return name in self.table

    def __len__(self) -> int:
        """Return the number of rows in the view."""
        return len(self.table) if self._indices is None else len(self._indices)

    def __iter__(self) -> Iterator[Self]:
        """Iterate over the rows, yielding single-row views."""
        for index in self.indices:
            yield self.__class__(self.structure, self.level, np.array([index]))

    def __repr__(self) -> str:
        """Return the level, the number of rows, and the columns."""
        header = f"<View {self.level.name.lower()}s={len(self)}"
        if self._indices is not None:
            header += f" of {len(self.table)}"
        # The table lists its columns right after the header, then its edges.
        columns = repr(self.table).splitlines()[1 : 1 + len(self.table.columns)]
        return "\n".join([f"{header}>", *columns])

    def _convert(self, level: Level) -> View:
        st = self.structure
        parent_index = (st.atom_to_residue, st.residue_to_chain)

        indices = self.indices
        for lower in range(self.level, level):  # up: the parent of each row
            indices = parent_index[lower][indices]
        for lower in reversed(range(level, self.level)):  # down: the children
            indices = np.flatnonzero(np.isin(parent_index[lower], indices))

        return View(st, level, np.unique(indices))
