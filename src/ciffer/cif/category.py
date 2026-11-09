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
) -> Structure:
    """Build the atoms, residues, and chains of one model.

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

    Examples
    --------
    .. code-block:: python

        site = Category(block, "_atom_site.")
        entity = Category(block, "_entity.")
        ...
        st = from_categories(site.take(rows), entity, entity_poly, chem_comp)

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

    # Non-polymers have no label_seq_id, so auth_seq_id and ins_code tell waters apart;
    # name splits two residue types at one position (microheterogeneity).
    keys = ["label_asym_id", "label_seq_id", "auth_seq_id", "ins_code", "name"]
    first, atom_to_residue = _group_rows(*(per_atom[key] for key in keys))
    residues = {name: values[first] for name, values in per_atom.items()}
    first, residue_to_chain = _group_rows(residues["label_asym_id"])

    entity_type = dict(zip(entity.get("id"), entity.get("type"), strict=True))
    polymer_type = dict(
        zip(entity_poly.get("entity_id"), entity_poly.get("type"), strict=True)
    )
    comp_type = dict(zip(chem_comp.get("id"), chem_comp.get("type"), strict=True))

    entity_id = residues["entity_id"][first]
    chains = {
        "label_asym_id": residues["label_asym_id"][first],
        "auth_asym_id": residues["auth_asym_id"][first],
        "entity_id": entity_id,
        "entity_type": np.array([entity_type.get(e, "") for e in entity_id], str),
        "polymer_type": np.array([polymer_type.get(e, "") for e in entity_id], str),
    }

    names = residues["name"]
    seq = residues["label_seq_id"]
    residues = {
        "name": names,
        "label_seq_id": np.where(seq == "", "-1", seq).astype(int),
        "auth_seq_id": residues["auth_seq_id"],
        "ins_code": residues["ins_code"],
        "comp_type": np.array([comp_type.get(n, "") for n in names], str),
    }

    occupancy = site.get("occupancy")
    b_factor = site.get("B_iso_or_equiv")
    charge = site.get("pdbx_formal_charge")
    atoms = {
        "name": site["label_atom_id"],
        "element": site["type_symbol"],
        "altloc": site["label_alt_id"],
        "occupancy": np.where(occupancy == "", "nan", occupancy).astype(float),
        "b_factor": np.where(b_factor == "", "nan", b_factor).astype(float),
        "charge": np.where(charge == "", "0", charge).astype(int),
        "xyz": np.stack([site[f"Cartn_{a}"] for a in "xyz"], axis=1).astype(float),
    }

    return Structure(
        atoms=Table(atoms),
        residues=Table(residues),
        chains=Table(chains),
        atom_to_residue=atom_to_residue,
        residue_to_chain=residue_to_chain,
    )


def _group_rows(*columns: NDArray) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Group rows that are equal in every column.

    Returns the first row of each group and the group of each row, numbered by first
    appearance.

    Examples
    --------
    >>> chain = np.array(["A", "A", "A", "B"])
    >>> seq = np.array(["1", "1", "2", "1"])
    >>> _group_rows(chain, seq)
    (array([0, 2, 3]), array([0, 0, 1, 2]))

    """
    if len(columns[0]) == 0:
        return np.zeros(0, dtype=np.intp), np.zeros(0, dtype=np.intp)

    rows = np.stack(columns, axis=1)
    _, first, group = np.unique(rows, axis=0, return_index=True, return_inverse=True)

    # np.unique numbers groups by sorted value; renumber them by first row.
    order = np.argsort(first)
    return first[order], np.argsort(order)[group.ravel()]
