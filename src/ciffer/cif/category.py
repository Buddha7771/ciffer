from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import gemmi
import numpy as np

from ciffer.structure import Structure
from ciffer.table import Table

if TYPE_CHECKING:
    from numpy.typing import NDArray


class Category:
    """One mmCIF table, with a string array per item.

    Bare ``?`` and ``.`` become "", and quotes are removed.

    Examples
    --------
    .. code-block:: python

        site = Category(block, "_atom_site.")
        site["label_atom_id"]  # array(['N', 'CA', ...]), KeyError if absent
        site.get("pdbx_formal_charge")  # array(['', '', ...]) if absent
        len(site)  # number of atoms
        site.take(rows)  # a Category with only these rows

    """

    def __init__(self, block: gemmi.cif.Block, name: str) -> None:
        table = block.find_mmcif_category(name)
        self.name = name
        self.rows = len(table)
        self.columns: dict[str, NDArray[np.str_]] = {}

        for i, tag in enumerate(table.tags):
            raw = np.array(list(table.column(i)), dtype=str)
            values = raw.copy()

            # gemmi keeps the quotes; remove them only where a value starts with one.
            quoted = np.isin(raw.astype("U1"), ["'", '"', ";"])
            for j in np.flatnonzero(quoted):
                values[j] = gemmi.cif.as_string(raw[j])

            # A quoted '?' is text, so nulls are found in the raw values.
            values[(raw == "?") | (raw == ".")] = ""

            item = tag.removeprefix(name)
            self.columns[item] = values

    def __getitem__(self, item: str) -> NDArray[np.str_]:
        """Return the column of ``item``; raise KeyError if the file lacks it."""
        if item not in self.columns:
            msg = f"The file has no {self.name}{item}."
            raise KeyError(msg)
        return self.columns[item]

    def get(self, item: str) -> NDArray[np.str_]:
        """Return the column of ``item``, or a column of "" if the file lacks it."""
        if item not in self.columns:
            return np.full(self.rows, "")
        return self.columns[item]

    def take(self, rows: NDArray[np.intp]) -> Category:
        """Return a new Category with the rows at ``rows``."""
        taken = copy.copy(self)
        taken.rows = len(rows)
        taken.columns = {item: values[rows] for item, values in self.columns.items()}
        return taken

    def __contains__(self, item: str) -> bool:
        """Return whether the file has ``item``."""
        return item in self.columns

    def __len__(self) -> int:
        """Return the number of rows."""
        return self.rows


def from_categories(
    site: Category,
    entity: Category,
    entity_poly: Category,
    chem_comp: Category,
    scheme: Category,
) -> Structure:
    """Build the atoms, residues, and chains of one model.

    Polymer residues follow the sequence, including those with no atoms.

    Parameters
    ----------
    site: Category
        ``_atom_site`` rows of one model.
    entity: Category
        ``_entity``, for the entity_type of each chain.
    entity_poly: Category
        ``_entity_poly``, for the polymer_type of each chain.
    chem_comp: Category
        ``_chem_comp``, for the comp_type of each residue.
    scheme: Category
        ``_pdbx_poly_seq_scheme``, for polymer residues with no atoms.

    Examples
    --------
    .. code-block:: python

        site = Category(block, "_atom_site.")
        entity = Category(block, "_entity.")
        ...
        st = from_categories(site.take(rows), entity, entity_poly, chem_comp, scheme)

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

    first, residue_to_chain = group_rows(residues["label_asym_id"])
    chains = {name: values[first] for name, values in residues.items()}

    return Structure(
        atoms=Table(_atom_table(site)),
        residues=Table(_residue_table(residues, chem_comp)),
        chains=Table(_chain_table(chains, entity, entity_poly)),
        atom_to_residue=group[from_atoms[keep]],
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


def _atom_table(site: Category) -> dict[str, NDArray]:
    occupancy = site.get("occupancy")
    b_factor = site.get("B_iso_or_equiv")
    charge = site.get("pdbx_formal_charge")
    return {
        "name": site["label_atom_id"],
        "element": site["type_symbol"],
        "altloc": site["label_alt_id"],
        "occupancy": np.where(occupancy == "", "nan", occupancy).astype(float),
        "b_factor": np.where(b_factor == "", "nan", b_factor).astype(float),
        "charge": np.where(charge == "", "0", charge).astype(int),
        "xyz": np.stack([site[f"Cartn_{a}"] for a in "xyz"], axis=1).astype(float),
    }


def group_rows(*columns: NDArray) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Group rows that are equal in every column.

    Returns the first row of each group and the group of each row, numbered by first
    appearance.

    Examples
    --------
    >>> chain = np.array(["A", "A", "A", "B"])
    >>> seq = np.array(["1", "1", "2", "1"])
    >>> group_rows(chain, seq)
    (array([0, 2, 3]), array([0, 0, 1, 2]))

    """
    if len(columns[0]) == 0:
        return np.zeros(0, dtype=np.intp), np.zeros(0, dtype=np.intp)

    rows = np.stack(columns, axis=1)
    _, first, group = np.unique(rows, axis=0, return_index=True, return_inverse=True)

    # np.unique numbers groups by sorted value; renumber them by first row.
    order = np.argsort(first)
    return first[order], np.argsort(order)[group.ravel()]
