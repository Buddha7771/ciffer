from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ciffer.table import EdgeList

from .category import Category, filled, group_rows, to_cif

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ciffer.structure import Structure

    from .category import Columns, Items

ORDERS = {"sing": 1, "doub": 2, "trip": 3, "quad": 4}
# PDB entries write these in lower case in _struct_conn, and upper in _chem_comp_bond.
ORDER_NAMES = {order: text for text, order in ORDERS.items()}


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


def comp_bond_category(
    st: Structure, atoms: Columns, residues: Columns
) -> dict[str, Items]:
    """Return ``_chem_comp_bond`` with one row per residue type and atom pair of bonds.

    Bonds between residues are left out, and so is the category when no bonds remain.

    Raises
    ------
    ValueError
        If residues of one type disagree on a bond: they give it different ``order`` or
        ``aromatic`` values, or one with both atoms lacks it.
    """
    bond = st.atoms.edges.get("bond")
    if bond is None:
        return {}
    a2r = st.atom_to_residue
    within = np.flatnonzero(a2r[bond.src] == a2r[bond.dst])
    if not within.size:
        return {}
    residue = a2r[bond.src[within]]
    comp = residues["name"][residue]
    first, second = atoms["name"][bond.src[within]], atoms["name"][bond.dst[within]]
    values = filled(bond, order=0, aromatic=False)
    order, aromatic = values["order"][within], values["aromatic"][within]

    # A bond is the same whichever end comes first.
    low, high = np.sort([first, second], axis=0)
    rows, pair = group_rows(comp, low, high)
    variants = group_rows(pair, order, aromatic)[0]
    differing = np.flatnonzero(np.bincount(pair[variants]) > 1)
    if differing.size:
        row = rows[differing[0]]
        msg = (
            f"Residues {comp[row]} give the bond {first[row]}-{second[row]} different "
            "order or aromatic values."
        )
        raise ValueError(msg)

    # Readers give a listed bond to every residue of its type with both atoms in one
    # altloc, and an atom without an altloc is in every altloc.
    altlocs: dict[tuple[int, str], set[str]] = {}
    for key in zip(
        a2r.tolist(), atoms["name"].tolist(), atoms["altloc"].tolist(), strict=True
    ):
        altlocs.setdefault(key[:2], set()).add(key[2])
    bonded = set(zip(residue.tolist(), pair.tolist(), strict=True))
    holders = {
        name: np.flatnonzero(residues["name"] == name).tolist()
        for name in np.unique(comp).tolist()
    }
    for k, row in enumerate(rows.tolist()):
        a, b = low[row], high[row]
        for r in holders[comp[row]]:
            ends = altlocs.get((r, a), set()), altlocs.get((r, b), set())
            meet = all(ends) and ("" in ends[0] | ends[1] or ends[0] & ends[1])
            if meet and (r, k) not in bonded:
                msg = (
                    f"Residue {r} lacks the bond {a}-{b} that other {comp[row]} "
                    "residues have."
                )
                raise ValueError(msg)

    return {
        "_chem_comp_bond.": {
            "comp_id": to_cif(comp[rows]),
            "atom_id_1": to_cif(first[rows]),
            "atom_id_2": to_cif(second[rows]),
            "value_order": to_cif(
                [ORDER_NAMES.get(o, "").upper() for o in order[rows].tolist()]
            ),
            "pdbx_aromatic_flag": np.where(aromatic[rows], "Y", "N").tolist(),
            "pdbx_ordinal": to_cif(np.arange(1, len(rows) + 1)),
        }
    }


def struct_conn_category(
    st: Structure, atoms: Columns, residues: Columns, chains: Columns
) -> dict[str, Items]:
    """Return ``_struct_conn`` with one row per ``struct_conn`` edge.

    A connection without a type is written as ``covale``.
    """
    conn = st.atoms.edges.get("struct_conn")
    if conn is None or not len(conn):
        return {}
    values = filled(conn, type="", order=0)
    types = np.where(values["type"] == "", "covale", values["type"])
    order = values["order"]

    # Ids number connections within each type, such as disulf1 and disulf2.
    counts: dict[str, int] = {}
    ids = []
    for kind in types.tolist():
        counts[kind] = counts.get(kind, 0) + 1
        ids.append(f"{kind}{counts[kind]}")

    items: Items = {"id": ids, "conn_type_id": to_cif(types)}
    for p, atom in ((1, conn.src), (2, conn.dst)):
        residue = st.atom_to_residue[atom]
        chain = st.residue_to_chain[residue]
        seq = residues["label_seq_id"][residue]
        items |= {
            f"ptnr{p}_label_asym_id": to_cif(chains["label_asym_id"][chain]),
            f"ptnr{p}_label_comp_id": to_cif(residues["name"][residue]),
            f"ptnr{p}_label_seq_id": to_cif(
                np.where(seq > 0, seq.astype(str), ""), null=False
            ),
            f"ptnr{p}_label_atom_id": to_cif(atoms["name"][atom]),
            f"pdbx_ptnr{p}_label_alt_id": to_cif(atoms["altloc"][atom]),
            f"pdbx_ptnr{p}_PDB_ins_code": to_cif(residues["ins_code"][residue]),
            # Non-polymers have no label_seq_id, so readers find them by auth_seq_id.
            f"ptnr{p}_auth_asym_id": to_cif(chains["auth_asym_id"][chain]),
            f"ptnr{p}_auth_comp_id": to_cif(residues["name"][residue]),
            f"ptnr{p}_auth_seq_id": to_cif(residues["auth_seq_id"][residue]),
            f"ptnr{p}_symmetry": ["1_555"] * len(conn),
        }
    items["pdbx_value_order"] = to_cif([ORDER_NAMES.get(o, "") for o in order.tolist()])
    return {"_struct_conn.": items}


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
