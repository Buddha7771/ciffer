from __future__ import annotations

from typing import TYPE_CHECKING

import gemmi
import numpy as np

from ciffer.structure import Structure
from ciffer.table import Table

from .bonds import (
    bond_edges,
    comp_bond_category,
    struct_conn_category,
    struct_conn_edges,
)
from .category import Category, group_rows, to_cif
from .metadata import metadata_categories
from .writable import in_sequence

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .category import Columns, Items


def from_categories(
    site: Category,
    *,
    entity: Category,
    entity_poly: Category,
    chem_comp: Category,
    struct_asym: Category,
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
    struct_asym: Category
        ``_struct_asym``, for the order of chains.
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

    # Rows of each chain go together, in the order _struct_asym lists chains (others
    # last), so residues and chains come out in that order.
    listed_at = {asym: i for i, asym in enumerate(struct_asym.get("id").tolist())}
    chain_ids, chain_of_row = np.unique(rows["label_asym_id"], return_inverse=True)
    rank = [listed_at.get(asym, len(listed_at)) for asym in chain_ids.tolist()]
    order = np.argsort(np.array(rank)[chain_of_row], kind="stable")

    first, group = group_rows(*(rows[key][order] for key in keys))
    residues = {name: values[order][first] for name, values in rows.items()}
    atom_to_residue = group[np.argsort(order)][from_atoms[keep]]  # back to row order

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


def to_categories(
    st: Structure, atoms: Columns, residues: Columns, chains: Columns
) -> dict[str, Items]:
    """Return the gemmi items of each mmCIF category, keyed like ``"_atom_site."``.

    ``atoms``, ``residues``, and ``chains`` are what ``writable_columns`` returns.

    Raises
    ------
    ValueError
        If residues of one type disagree on a bond.
    """
    entity_chains = group_rows(chains["entity_id"])[0]
    names, first_of_name = np.unique(residues["name"], return_index=True)
    return {
        **metadata_categories(st.metadata),
        "_atom_site.": _atom_site(st, atoms, residues, chains),
        "_chem_comp.": {
            "id": to_cif(names),
            "type": to_cif(residues["comp_type"][first_of_name]),
        },
        "_entity.": {
            "id": to_cif(chains["entity_id"][entity_chains]),
            "type": to_cif(chains["entity_type"][entity_chains]),
            "pdbx_description": to_cif(chains["description"][entity_chains]),
        },
        "_struct_asym.": {
            "id": to_cif(chains["label_asym_id"]),
            "entity_id": to_cif(chains["entity_id"]),
        },
        **_sequences(st, residues, chains),
        **comp_bond_category(st, atoms, residues),
        **struct_conn_category(st, atoms, residues, chains),
    }


def _sequences(st: Structure, residues: Columns, chains: Columns) -> dict[str, Items]:
    """Return _entity_poly, _entity_poly_seq, and _pdbx_poly_seq_scheme of polymers.

    Polymer residues with a label_seq_id are listed, with or without atoms, and the
    first chain listing residues of an entity gives the entity sequence. Residue types
    sharing a position (microheterogeneity) are marked hetero, and the first one is in
    the one-letter sequence.
    """
    polymer_chains = np.flatnonzero(chains["entity_type"] == "polymer")
    if not polymer_chains.size:
        return {}
    # The first polymer chain of each entity gives its type.
    entity_chains = polymer_chains[group_rows(chains["entity_id"][polymer_chains])[0]]
    entity_ids = chains["entity_id"][entity_chains].tolist()
    items: dict[str, Items] = {
        "_entity_poly.": {
            "entity_id": to_cif(entity_ids),
            "type": to_cif(chains["polymer_type"][entity_chains]),
        }
    }

    scheme_residues = np.flatnonzero(in_sequence(st, residues, chains))
    if not scheme_residues.size:
        return items
    chain = st.residue_to_chain[scheme_residues]
    seq = residues["label_seq_id"][scheme_residues]
    name = residues["name"][scheme_residues]
    position_first, position = group_rows(chain, seq)
    # A position listed more than once holds several residue types.
    hetero = np.where(np.bincount(position)[position] > 1, "y", "n")

    # The first chain listing residues of each entity gives its sequence.
    sequence_chains = chain[group_rows(chains["entity_id"][chain])[0]]
    in_sequence_chain = np.isin(chain, sequence_chains)
    code, canonical = {}, {}
    for sequence_chain in sequence_chains.tolist():
        sequence_names = name[position_first[chain[position_first] == sequence_chain]]
        entity = chains["entity_id"][sequence_chain]
        polymer_type = str(chains["polymer_type"][sequence_chain])
        code[entity], canonical[entity] = _one_letter(sequence_names, polymer_type)
    items["_entity_poly."] |= {
        "pdbx_seq_one_letter_code": [code.get(e) for e in entity_ids],
        "pdbx_seq_one_letter_code_can": [canonical.get(e) for e in entity_ids],
    }

    observed = np.bincount(st.atom_to_residue, minlength=len(residues["name"])) > 0
    observed = observed[scheme_residues]
    auth_seq = residues["auth_seq_id"][scheme_residues]
    items["_entity_poly_seq."] = {
        "entity_id": to_cif(chains["entity_id"][chain[in_sequence_chain]]),
        "num": to_cif(seq[in_sequence_chain]),
        "mon_id": to_cif(name[in_sequence_chain]),
        "hetero": hetero[in_sequence_chain].tolist(),
    }
    items["_pdbx_poly_seq_scheme."] = {
        "asym_id": to_cif(chains["label_asym_id"][chain]),
        "entity_id": to_cif(chains["entity_id"][chain]),
        "seq_id": to_cif(seq),
        "mon_id": to_cif(name),
        "pdb_seq_num": to_cif(auth_seq),
        "auth_seq_num": to_cif(np.where(observed, auth_seq, "")),
        "pdb_mon_id": to_cif(np.where(observed, name, "")),
        "auth_mon_id": to_cif(np.where(observed, name, "")),
        "pdb_strand_id": to_cif(chains["auth_asym_id"][chain]),
        "pdb_ins_code": to_cif(residues["ins_code"][scheme_residues], null=False),
        "hetero": hetero.tolist(),
    }
    return items


def _one_letter(names: NDArray[np.str_], polymer_type: str) -> tuple[str, str]:
    """Return the one-letter sequence and its canonical form, as PDB entries write them.

    Standard amino acids and ribonucleotides are letters, and other residues are their
    names in parentheses. The canonical form uses the parent letter gemmi tabulates, or
    N for nucleic acids and X otherwise where it has none.

    >>> _one_letter(np.array(["MET", "MSE", "UNK"]), "polypeptide(L)")
    ('M(MSE)(UNK)', 'MMX')
    >>> _one_letter(np.array(["DA", "PSU", "G"]), "polydeoxyribonucleotide")
    ('(DA)(PSU)G', 'AUG')

    """
    unknown = "N" if "nucleotide" in polymer_type else "X"
    kinds_with_letters = (gemmi.ResidueKind.AA, gemmi.ResidueKind.RNA)
    code, canonical = "", ""
    for name in names.tolist():
        info = gemmi.find_tabulated_residue(name)
        # gemmi gives a modified residue its parent letter in lowercase, or " " if none.
        letter = info.one_letter_code if info is not None else " "
        lettered = (
            info is not None and info.is_standard() and info.kind in kinds_with_letters
        )
        code += letter if lettered and name != "UNK" else f"({name})"
        canonical += letter.upper() if letter.strip() else unknown
    return code, canonical


def _atom_site(
    st: Structure, atoms: Columns, residues: Columns, chains: Columns
) -> Items:
    """Return one _atom_site row per atom."""
    a2r = st.atom_to_residue
    a2c = st.residue_to_chain[a2r]
    name = residues["name"][a2r]
    seq = residues["label_seq_id"][a2r]

    # Standard residues of polymers are ATOM, as gemmi tabulates them.
    standard = []
    for n in np.unique(name).tolist():
        info = gemmi.find_tabulated_residue(n)
        if info is not None and info.is_standard():
            standard.append(n)
    polymer = chains["entity_type"][a2c] == "polymer"
    group = np.where(polymer & np.isin(name, standard), "ATOM", "HETATM")

    xyz = atoms["xyz"]
    return {
        "group_PDB": group.tolist(),
        "id": to_cif(np.arange(1, len(a2r) + 1)),
        "type_symbol": to_cif(atoms["element"]),
        "label_atom_id": to_cif(atoms["name"]),
        "label_alt_id": to_cif(atoms["altloc"], null=False),
        "label_comp_id": to_cif(name),
        "label_asym_id": to_cif(chains["label_asym_id"][a2c]),
        "label_entity_id": to_cif(chains["entity_id"][a2c]),
        "label_seq_id": to_cif(np.where(seq > 0, seq.astype(str), ""), null=False),
        "pdbx_PDB_ins_code": to_cif(residues["ins_code"][a2r]),
        "Cartn_x": _decimals(xyz[:, 0], 3),
        "Cartn_y": _decimals(xyz[:, 1], 3),
        "Cartn_z": _decimals(xyz[:, 2], 3),
        "occupancy": _decimals(atoms["occupancy"], 2),
        "B_iso_or_equiv": _decimals(atoms["b_factor"], 2),
        "pdbx_formal_charge": to_cif(atoms["charge"]),
        "auth_seq_id": to_cif(residues["auth_seq_id"][a2r]),
        "auth_comp_id": to_cif(name),
        "auth_asym_id": to_cif(chains["auth_asym_id"][a2c]),
        "auth_atom_id": to_cif(atoms["name"]),
        "pdbx_PDB_model_num": [str(st.metadata.get("model", 1))] * len(a2r),
    }


def _decimals(values: NDArray[np.float64], digits: int) -> list[str | bool | None]:
    """Return ``values`` as text, all with the same number of decimals.

    That is the fewest decimals from ``digits`` to five that keep every value, or
    ``digits`` when the values need more.

    >>> _decimals(np.array([1.5, 2.12345, np.nan]), 3)
    ['1.50000', '2.12345', None]
    >>> _decimals(np.array([1.23456789]), 3)
    ['1.235']

    """
    for kept in range(digits, 6):
        # Float noise below 10**-(kept + 3) does not count as another decimal.
        rounded = np.round(values, kept)
        noise = 10.0 ** -(kept + 3)
        if np.allclose(rounded, values, rtol=0, atol=noise, equal_nan=True):
            return to_cif(np.char.mod(f"%.{kept}f", values))
    return to_cif(np.char.mod(f"%.{digits}f", values))
