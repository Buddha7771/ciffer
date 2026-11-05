from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, Self

import numpy as np

from .serialize import dump_bytes, load_bytes
from .table import EdgeList, Table, as_indices
from .view import Level, View

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike, NDArray

FORMAT_VERSION = 1


class Structure:
    """A biomolecular structure of atoms, residues, and chains.

    Each level is a ``Table`` in ``tables``, indexed by ``Level``. Atoms are grouped
    into residues and residues into chains by parent index arrays.

    Parameters
    ----------
    atoms: Table
        Atom-level columns and edges.
    residues: Table
        Residue-level columns and edges.
    chains: Table
        Chain-level columns and edges.
    atom_to_residue: ArrayLike
        Residue index of each atom.
    residue_to_chain: ArrayLike
        Chain index of each residue.
    metadata: dict[str, Any] | None, optional
        Values that describe the whole structure.

    Examples
    --------
    .. code-block:: python

        st = Structure(
            atoms=Table({"name": ["N", "CA", "C"], "xyz": xyz}),
            residues=Table({"name": ["ALA"]}),
            chains=Table({"id": ["A"]}),
            atom_to_residue=[0, 0, 0],
            residue_to_chain=[0],
            metadata={"id": "1abc"},
        )
        st.atoms["xyz"]                       # the stored array
        st.atoms[st.atoms["name"] == "CA"]    # a view of selected atoms

    """

    def __init__(
        self,
        *,
        atoms: Table,
        residues: Table,
        chains: Table,
        atom_to_residue: ArrayLike,
        residue_to_chain: ArrayLike,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.tables = (atoms, residues, chains)
        for level, table in zip(Level, self.tables, strict=True):
            if not isinstance(table, Table):
                name = f"{level.name.lower()}s"
                msg = f"{name} must be a Table; use view.extract() for a new structure."
                raise TypeError(msg)

        def _as_parent_index(
            name: str, value: ArrayLike, n_children: int, n_parents: int
        ) -> NDArray[np.integer]:
            index = as_indices(value)

            if len(index) != n_children:
                msg = f"{name} has {len(index)} entries for {n_children} rows."
                raise ValueError(msg)

            if index.size and (index.min() < 0 or index.max() >= n_parents):
                msg = f"{name} points outside the {n_parents} parent rows."
                raise IndexError(msg)

            return index

        self.atom_to_residue = _as_parent_index(
            "atom_to_residue", atom_to_residue, len(atoms), len(residues)
        )
        self.residue_to_chain = _as_parent_index(
            "residue_to_chain", residue_to_chain, len(residues), len(chains)
        )
        self.metadata: dict[str, Any] = metadata if metadata is not None else {}

    @property
    def atoms(self) -> View:
        """View of all atoms."""
        return View(self, Level.ATOM)

    @property
    def residues(self) -> View:
        """View of all residues."""
        return View(self, Level.RESIDUE)

    @property
    def chains(self) -> View:
        """View of all chains."""
        return View(self, Level.CHAIN)

    def copy(self) -> Self:
        """Return a deep copy of the structure, including its metadata."""
        atoms, residues, chains = (table.copy() for table in self.tables)
        return self.__class__(
            atoms=atoms,
            residues=residues,
            chains=chains,
            atom_to_residue=self.atom_to_residue.copy(),
            residue_to_chain=self.residue_to_chain.copy(),
            metadata=copy.deepcopy(self.metadata),
        )

    def to_bytes(self) -> bytes:
        """Serialize the structure to zstd-compressed bytes.

        Raises
        ------
        TypeError
            If a column or metadata value cannot be saved.
        """

        def _columns(table: Table | EdgeList) -> dict[str, Any]:
            return {name: table[name] for name in table.columns}

        def _table_to_dict(table: Table) -> dict[str, Any]:
            table.check_edges()  # edges may have been added after construction
            return {
                "columns": _columns(table),
                "edges": {
                    name: {
                        "src": edges.src,
                        "dst": edges.dst,
                        "columns": _columns(edges),
                    }
                    for name, edges in table.edges.items()
                },
            }

        return dump_bytes(
            {
                "version": FORMAT_VERSION,
                "atoms": _table_to_dict(self.tables[Level.ATOM]),
                "residues": _table_to_dict(self.tables[Level.RESIDUE]),
                "chains": _table_to_dict(self.tables[Level.CHAIN]),
                "atom_to_residue": self.atom_to_residue,
                "residue_to_chain": self.residue_to_chain,
                "metadata": self.metadata,
            }
        )

    @classmethod
    def from_bytes(cls, byte_data: bytes) -> Self:
        """Deserialize a structure saved by ``to_bytes``."""

        def _table_from_dict(data: dict[str, Any]) -> Table:
            return Table(
                data["columns"],
                edges={
                    name: EdgeList(edges["src"], edges["dst"], edges["columns"])
                    for name, edges in data["edges"].items()
                },
            )

        data = load_bytes(byte_data)
        if data["version"] != FORMAT_VERSION:
            msg = (
                f"Format version {data['version']} is not supported "
                f"(expected {FORMAT_VERSION})."
            )
            raise ValueError(msg)

        return cls(
            atoms=_table_from_dict(data["atoms"]),
            residues=_table_from_dict(data["residues"]),
            chains=_table_from_dict(data["chains"]),
            atom_to_residue=data["atom_to_residue"],
            residue_to_chain=data["residue_to_chain"],
            metadata=data["metadata"],
        )

    def __repr__(self) -> str:
        """Return the size of each level and the schema."""
        tables = [
            (f"{level.name.lower()}s", table)
            for level, table in zip(Level, self.tables, strict=True)
        ]
        counts = " ".join(f"{name}={len(table)}" for name, table in tables)
        lines = [f"<Structure {counts}>"]

        for name, table in tables:
            lines.append(name)
            lines.extend(repr(table).splitlines()[1:])  # the schema, without header

        lines.append(f"metadata: {', '.join(self.metadata) or '-'}")
        return "\n".join(lines)


def concat(structures: Sequence[Structure]) -> Structure:
    """Concatenate structures.

    Parameters
    ----------
    structures: Sequence[Structure]
        Structures whose levels have the same columns and edge groups.

    Returns
    -------
    Structure
        A new structure that shares no arrays with ``structures``, with empty
        metadata.

    Examples
    --------
    .. code-block:: python

        structure = concat([protein, ligand])
        structure.metadata["id"] = "1abc"

    """
    if not structures:
        msg = "Nothing to concatenate."
        raise ValueError(msg)

    residue_counts = [len(st.residues) for st in structures]
    chain_counts = [len(st.chains) for st in structures]
    residue_offsets = np.cumsum([0, *residue_counts[:-1]])
    chain_offsets = np.cumsum([0, *chain_counts[:-1]])

    atoms, residues, chains = (
        _concat_tables([st.tables[level] for st in structures]) for level in Level
    )
    return Structure(
        atoms=atoms,
        residues=residues,
        chains=chains,
        atom_to_residue=np.concatenate(
            [
                st.atom_to_residue + offset
                for st, offset in zip(structures, residue_offsets, strict=True)
            ]
        ),
        residue_to_chain=np.concatenate(
            [
                st.residue_to_chain + offset
                for st, offset in zip(structures, chain_offsets, strict=True)
            ]
        ),
    )


def _concat_tables(tables: Sequence[Table]) -> Table:
    """Stack tables row-wise into a new table.

    Parameters
    ----------
    tables: Sequence[Table]
        Tables with the same columns and edge groups.

    Returns
    -------
    Table
        A table whose edges are shifted to point at the stacked rows.
    """

    def _check_same_names(what: str, first: list[str], other: list[str]) -> None:
        missing = sorted(set(first) - set(other))
        extra = sorted(set(other) - set(first))
        if missing or extra:
            msg = f"{what} differ: missing {missing}, extra {extra}."
            raise ValueError(msg)

    def _concat_edges(name: str) -> EdgeList:
        groups = [table.edges[name] for table in tables]
        for edges in groups:
            _check_same_names(
                f"Columns of edge group {name!r}", groups[0].columns, edges.columns
            )

        shift = np.repeat(offsets, [len(edges) for edges in groups])
        return EdgeList(
            np.concatenate([edges.src for edges in groups]) + shift,
            np.concatenate([edges.dst for edges in groups]) + shift,
            {
                c: np.concatenate([edges[c] for edges in groups])
                for c in groups[0].columns
            },
        )

    first = tables[0]
    for table in tables:
        table.check_edges()
        _check_same_names("Columns", first.columns, table.columns)
        _check_same_names("Edge groups", list(first.edges), list(table.edges))

    offsets = np.cumsum([0, *(len(table) for table in tables[:-1])])

    return Table(
        {
            name: np.concatenate([table[name] for table in tables])
            for name in first.columns
        },
        edges={name: _concat_edges(name) for name in first.edges},
    )
