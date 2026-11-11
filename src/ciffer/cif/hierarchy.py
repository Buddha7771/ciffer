from __future__ import annotations

from typing import TYPE_CHECKING, Any

import gemmi
import numpy as np

from ciffer.structure import Structure
from ciffer.table import Table
from ciffer.view import Field

from .bonds import bond_edges, struct_conn_edges
from .category import Category, group_rows, to_cif

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

    from ciffer.view import View

    Columns = dict[str, NDArray[Any]]
    Items = dict[str, list[str | bool | None]]

REQUIRED = {
    "atoms": ["name", "element", "xyz"],
    "residues": ["name"],
    "chains": ["label_asym_id", "entity_type"],
}
STANDARD_EDGES = {"atoms": ["bond", "struct_conn"]}


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


def to_categories(st: Structure) -> dict[str, Items]:
    """Return the _atom_site, _chem_comp, _entity, and _struct_asym items of ``st``.

    Missing optional fields are filled as ``ciffer.write`` describes.

    Raises
    ------
    ValueError
        If a required field is missing, or two chains share a label_asym_id.
    """
    missing = [
        f"{level}.{name}"
        for level, names in REQUIRED.items()
        for name in names
        if name not in getattr(st, level)
    ]
    if missing:
        msg = f"Cannot write without {missing}."
        raise ValueError(msg)
    labels, counts = np.unique(st.chains.label_asym_id, return_counts=True)
    if (counts > 1).any():
        msg = f"Chains share label_asym_id {labels[counts > 1].tolist()}."
        raise ValueError(msg)

    atoms, residues, chains = _filled_tables(st)
    entities = group_rows(chains["entity_id"])[0]
    names, first = np.unique(residues["name"], return_index=True)
    return {
        "_atom_site.": _atom_site(st, atoms, residues, chains),
        "_chem_comp.": {
            "id": to_cif(names),
            "type": to_cif(residues["comp_type"][first]),
        },
        "_entity.": {
            "id": to_cif(chains["entity_id"][entities]),
            "type": to_cif(chains["entity_type"][entities]),
            "pdbx_description": to_cif(chains["description"][entities]),
        },
        "_struct_asym.": {
            "id": to_cif(chains["label_asym_id"]),
            "entity_id": to_cif(chains["entity_id"]),
        },
    }


def unwritten(st: Structure) -> list[str]:
    """Name the columns and edges of ``st`` that are not standard fields.

    Standard columns are the Fields of each view class, such as ``AtomView.xyz``.
    """
    names = []
    for view in (st.atoms, st.residues, st.chains):
        level = f"{view.level.name.lower()}s"
        fields = vars(type(view))
        names += [
            f"{level}[{name!r}]"
            for name in view.columns
            if not isinstance(fields.get(name), Field)
        ]
        names += [
            f"{level}.edges[{name!r}]"
            for name in view.edges
            if name not in STANDARD_EDGES.get(level, [])
        ]
    return names


def _filled_tables(st: Structure) -> tuple[Columns, Columns, Columns]:
    """Return the standard columns of atoms, residues, and chains, filling defaults."""
    chains = {
        "label_asym_id": st.chains.label_asym_id,
        "entity_type": st.chains.entity_type,
        **_filled(
            st.chains,
            auth_asym_id=st.chains.label_asym_id,
            entity_id=np.arange(1, len(st.chains) + 1).astype(str),
            description="",
        ),
    }

    # Missing ids number each residue by its place among the residues of its chain:
    # sorted by chain, a residue's place is its distance from the first of its chain.
    r2c = st.residue_to_chain
    order = np.argsort(r2c, kind="stable")
    first_of_chain = np.searchsorted(r2c[order], r2c[order])
    place = np.empty(len(r2c), np.intp)
    place[order] = np.arange(len(r2c)) - first_of_chain + 1

    polymer = chains["entity_type"][r2c] == "polymer"
    residues = {
        "name": st.residues.name,
        **_filled(
            st.residues,
            label_seq_id=np.where(polymer, place, -1),
            ins_code="",
            comp_type="",
        ),
    }
    # auth_seq_id follows label_seq_id, so it is filled after it.
    seq = residues["label_seq_id"]
    residues |= _filled(st.residues, auth_seq_id=np.where(seq > 0, seq, place))

    atoms = {
        "name": st.atoms.name,
        "element": st.atoms.element,
        "xyz": st.atoms.xyz,
        **_filled(st.atoms, altloc="", occupancy=1.0, b_factor=np.nan, charge=0),
    }
    return atoms, residues, chains


def _filled(view: View, **defaults: ArrayLike) -> Columns:
    """Return the named columns of ``view``, with the default of each missing one."""
    return {
        name: view[name] if name in view else np.broadcast_to(default, len(view))
        for name, default in defaults.items()
    }


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

    def _decimals(values: NDArray[np.float64], digits: int) -> list[str | bool | None]:
        text = np.char.mod(f"%.{digits}f", values)
        return to_cif(np.where(np.isnan(values), "", text))

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
