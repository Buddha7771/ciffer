from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ciffer.structure import Structure
from ciffer.table import Table

from .bonds import bond_edges, struct_conn_edges
from .category import Category, group_rows

if TYPE_CHECKING:
    from numpy.typing import NDArray


def from_categories(
    site: Category,
    *,
    entity: Category,
    entity_poly: Category,
    chem_comp: Category,
    scheme: Category,
    comp_bond: Category,
    conn: Category,
) -> Structure:
    """Build the atoms, residues, and chains of one model.

    Polymer residues follow the sequence, including those with no atoms.

    Parameters
    ----------
    site: Category
        ``_atom_site`` rows of one model.
    entity: Category
        ``_entity``, for the entity_type and description of each chain.
    entity_poly: Category
        ``_entity_poly``, for the polymer_type of each chain.
    chem_comp: Category
        ``_chem_comp``, for the comp_type of each residue.
    scheme: Category
        ``_pdbx_poly_seq_scheme``, for polymer residues with no atoms.
    comp_bond: Category
        ``_chem_comp_bond``, for the bonds within each residue.
    conn: Category
        ``_struct_conn``, for the connections between residues.

    Examples
    --------
    .. code-block:: python

        site = Category(block, "_atom_site.")
        entity = Category(block, "_entity.")
        ...
        st = from_categories(site.take(rows), entity=entity, ..., conn=conn)

    """
    per_atom = {
        "label_asym_id": site["label_asym_id"],
        "auth_asym_id": site["auth_asym_id"],
        "entity_id": site["label_entity_id"],
        "name": site["label_comp_id"],
        "label_seq_id": site["label_seq_id"],
        "auth_seq_id": site.get("auth_seq_id"),
        "ins_code": site.get("pdbx_PDB_ins_code"),
    }

    listed = {
        "label_asym_id": scheme.get("asym_id"),
        "auth_asym_id": scheme.get("pdb_strand_id"),
        "entity_id": scheme.get("entity_id"),
        "name": scheme.get("mon_id"),
        "label_seq_id": scheme.get("seq_id"),
        "auth_seq_id": scheme.get("pdb_seq_num"),
        "ins_code": scheme.get("pdb_ins_code"),
    }

    # Non-polymers have no label_seq_id, so auth_seq_id and ins_code tell waters apart;
    # name splits two residue types at one position (microheterogeneity).
    keys = ["label_asym_id", "label_seq_id", "auth_seq_id", "ins_code", "name"]
    rows = {name: np.concatenate([listed[name], per_atom[name]]) for name in per_atom}
    from_atoms = np.arange(len(rows["name"])) >= len(scheme)

    keep = _one_type_per_position(
        rows["label_asym_id"], rows["label_seq_id"], rows["name"], from_atoms
    )
    rows = {name: values[keep] for name, values in rows.items()}

    first, group = group_rows(*(rows[key] for key in keys))
    residues = {name: values[first] for name, values in rows.items()}
    atom_to_residue = group[from_atoms[keep]]

    first, residue_to_chain = group_rows(residues["label_asym_id"])
    chains = {name: values[first] for name, values in residues.items()}

    return Structure(
        atoms=_atom_table(site, atom_to_residue, residues["name"], comp_bond, conn),
        residues=Table(_residue_table(residues, chem_comp)),
        chains=Table(_chain_table(chains, entity, entity_poly)),
        atom_to_residue=atom_to_residue,
        residue_to_chain=residue_to_chain,
    )


def _one_type_per_position(
    asym: NDArray[np.str_],
    seq: NDArray[np.str_],
    name: NDArray[np.str_],
    from_atoms: NDArray[np.bool_],
) -> NDArray[np.bool_]:
    """Return which rows to keep where the sequence lists several residue types.

    One position can hold several residue types (microheterogeneity). The types with
    atoms are kept, and a position without atoms keeps its first listed type.

    Examples
    --------
    >>> asym = np.array(["A", "A", "A", "A", "A"])
    >>> seq = np.array(["22", "22", "23", "23", "22"])
    >>> name = np.array(["PRO", "SER", "ALA", "GLY", "PRO"])
    >>> from_atoms = np.array([False, False, False, False, True])
    >>> _one_type_per_position(asym, seq, name, from_atoms)
    array([ True, False,  True, False,  True])

    """
    first, position = group_rows(asym, seq)
    _, kind = group_rows(asym, seq, name)
    observed = np.isin(position, position[from_atoms])
    return np.where(
        observed,
        np.isin(kind, kind[from_atoms]),
        kind == kind[first][position],
    )


def _chain_table(
    chains: dict[str, NDArray[np.str_]], entity: Category, entity_poly: Category
) -> dict[str, NDArray]:
    entity_type = dict(zip(entity.get("id"), entity.get("type"), strict=True))
    description = dict(
        zip(entity.get("id"), entity.get("pdbx_description"), strict=True)
    )
    polymer_type = dict(
        zip(entity_poly.get("entity_id"), entity_poly.get("type"), strict=True)
    )
    entity_id = chains["entity_id"]
    return {
        "label_asym_id": chains["label_asym_id"],
        "auth_asym_id": chains["auth_asym_id"],
        "entity_id": entity_id,
        "entity_type": np.array([entity_type.get(e, "") for e in entity_id], str),
        "polymer_type": np.array([polymer_type.get(e, "") for e in entity_id], str),
        "description": np.array([description.get(e, "") for e in entity_id], str),
    }


def _residue_table(
    residues: dict[str, NDArray[np.str_]], chem_comp: Category
) -> dict[str, NDArray]:
    comp_type = dict(zip(chem_comp.get("id"), chem_comp.get("type"), strict=True))
    names = residues["name"]
    seq = residues["label_seq_id"]
    return {
        "name": names,
        "label_seq_id": np.where(seq == "", "-1", seq).astype(int),
        "auth_seq_id": residues["auth_seq_id"],
        "ins_code": residues["ins_code"],
        "comp_type": np.array([comp_type.get(n, "") for n in names], str),
    }


def _atom_table(
    site: Category,
    atom_to_residue: NDArray[np.intp],
    residue_names: NDArray[np.str_],
    comp_bond: Category,
    conn: Category,
) -> Table:
    """Return the atom table, with its bond and struct_conn edges."""
    occupancy = site.get("occupancy")
    b_factor = site.get("B_iso_or_equiv")
    charge = site.get("pdbx_formal_charge")
    columns = {
        "name": site["label_atom_id"],
        "element": site["type_symbol"],
        "altloc": site["label_alt_id"],
        "occupancy": np.where(occupancy == "", "nan", occupancy).astype(float),
        "b_factor": np.where(b_factor == "", "nan", b_factor).astype(float),
        "charge": np.where(charge == "", "0", charge).astype(int),
        "xyz": np.stack([site[f"Cartn_{a}"] for a in "xyz"], axis=1).astype(float),
    }
    edges = {
        "bond": bond_edges(site, atom_to_residue, residue_names, comp_bond),
        "struct_conn": struct_conn_edges(site, conn),
    }
    return Table(columns, edges=edges)
