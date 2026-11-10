from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ciffer.table import EdgeList

from .category import Category, group_rows

if TYPE_CHECKING:
    from numpy.typing import NDArray

ORDERS = {"sing": 1, "doub": 2, "trip": 3, "quad": 4}


def bond_edges(
    site: Category,
    atom_to_residue: NDArray[np.intp],
    residue_names: NDArray[np.str_],
    comp_bond: Category,
) -> EdgeList:
    """Connect the atoms of each residue as ``_chem_comp_bond`` lists for its type.

    ``site`` rows are the atoms. The ``order`` column is 1 to 4, or 0 when missing or
    for any other order such as ``arom``.

    Examples
    --------
    .. code-block:: python

        comp_bond = Category(block, "_chem_comp_bond.")
        bond = bond_edges(site, atom_to_residue, residues["name"], comp_bond)

    """
    comp = comp_bond.get("comp_id")

    # Each bond row applies to every residue of its type.
    residue, row = [np.zeros(0, np.intp)], [np.zeros(0, np.intp)]
    for name in np.unique(comp).tolist():
        holders = np.flatnonzero(residue_names == name)
        rows = np.flatnonzero(comp == name)
        residue.append(np.repeat(holders, len(rows)))
        row.append(np.tile(rows, len(holders)))
    residue, row = np.concatenate(residue), np.concatenate(row)

    any_altloc = np.full(len(row), "")
    src, dst, found = _match(
        [atom_to_residue, site["label_atom_id"]],
        site["label_alt_id"],
        ([residue, comp_bond.get("atom_id_1")[row]], any_altloc),
        ([residue, comp_bond.get("atom_id_2")[row]], any_altloc),
    )
    row = row[found]
    order = [ORDERS.get(v.lower(), 0) for v in comp_bond.get("value_order")[row]]
    return EdgeList(
        src,
        dst,
        {
            "order": np.array(order, dtype=np.int64),
            "aromatic": comp_bond.get("pdbx_aromatic_flag")[row] == "Y",
        },
    )


def struct_conn_edges(site: Category, conn: Category) -> EdgeList:
    """Connect the atoms that ``_struct_conn`` lists, where ``site`` rows are the atoms.

    Every connection type is kept, such as ``covale``, ``metalc``, and ``hydrog``.
    Connections to another symmetry copy are dropped, since that copy has no atoms. The
    ``order`` column is 1 to 4, or 0 when missing, as in most files.

    Examples
    --------
    .. code-block:: python

        conn = Category(block, "_struct_conn.")
        struct_conn = struct_conn_edges(site, conn)

    """
    # A missing symmetry is the identity.
    symmetry = [conn.get(f"ptnr{p}_symmetry") for p in (1, 2)]
    symmetry = [np.where(s == "", "1_555", s) for s in symmetry]
    row = np.flatnonzero(symmetry[0] == symmetry[1])

    # Non-polymers have no label_seq_id, so partners are found by auth_seq_id.
    def _partner(p: int) -> tuple[list[NDArray[np.str_]], NDArray[np.str_]]:
        keys = [
            conn.get(f"ptnr{p}_label_asym_id")[row],
            conn.get(f"ptnr{p}_auth_seq_id")[row],
            conn.get(f"pdbx_ptnr{p}_PDB_ins_code")[row],
            conn.get(f"ptnr{p}_label_comp_id")[row],
            conn.get(f"ptnr{p}_label_atom_id")[row],
        ]
        return keys, conn.get(f"pdbx_ptnr{p}_label_alt_id")[row]

    atoms = [
        site["label_asym_id"],
        site.get("auth_seq_id"),
        site.get("pdbx_PDB_ins_code"),
        site["label_comp_id"],
        site["label_atom_id"],
    ]
    src, dst, found = _match(atoms, site["label_alt_id"], _partner(1), _partner(2))
    row = row[found]
    order = [ORDERS.get(v.lower(), 0) for v in conn.get("pdbx_value_order")[row]]
    return EdgeList(
        src,
        dst,
        {
            "type": conn.get("conn_type_id")[row],
            "order": np.array(order, dtype=np.int64),
        },
    )


def _match(
    atoms: list[NDArray],
    altloc: NDArray[np.str_],
    end_1: tuple[list[NDArray], NDArray[np.str_]],
    end_2: tuple[list[NDArray], NDArray[np.str_]],
) -> tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.intp]]:
    """Find the atoms at both ends of each edge, within one altloc at a time.

    ``atoms`` and each end hold key columns that must all match. An atom without an
    altloc belongs to every altloc, and so does an end without one. Returns the two
    atoms and the edge index of each match, sorted by atom.
    """
    keys_1, altloc_1 = end_1
    keys_2, altloc_2 = end_2
    columns = zip(atoms, keys_1, keys_2, strict=True)
    _, key = group_rows(*(np.concatenate(column) for column in columns))
    atom_key, key_1, key_2 = np.split(key, [len(altloc), len(altloc) + len(altloc_1)])

    found = [np.zeros((3, 0), np.intp)]
    for letter in np.unique(altloc[altloc != ""]).tolist() or [""]:
        seen = np.flatnonzero((altloc == "") | (altloc == letter))
        atom_of = np.full(len(key), -1)
        atom_of[atom_key[seen]] = seen
        src, dst = atom_of[key_1], atom_of[key_2]
        fits = np.isin(altloc_1, ["", letter]) & np.isin(altloc_2, ["", letter])
        ok = np.flatnonzero((src >= 0) & (dst >= 0) & fits)
        found.append(np.stack([src[ok], dst[ok], ok]))

    # Edges between atoms without an altloc are found in every altloc.
    src, dst, edge = np.unique(np.concatenate(found, axis=1), axis=1)
    return src, dst, edge
