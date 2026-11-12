from __future__ import annotations

from typing import TYPE_CHECKING

import gemmi
import numpy as np

from ciffer.view import Bond, Field, StructConn

from .category import chain_names, filled, group_rows
from .metadata import ENTRY_KEYS

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ciffer.structure import Structure

    from .category import Columns

REQUIRED = {"atoms": ["name", "element", "xyz"], "residues": ["name"]}
WRITTEN_EDGES = {("atoms", "bond"): Bond, ("atoms", "struct_conn"): StructConn}
# Metadata that read sets, written in _atom_site or not at all.
READ_KEYS = {"model", "models", "assembly", "assemblies"}
# The values of _entity.type.
ENTITY_TYPES = {"polymer", "non-polymer", "branched", "macrolide", "water"}
# The values of _entity_poly.type.
POLYMER_TYPES = {
    "polypeptide(L)",
    "polypeptide(D)",
    "polyribonucleotide",
    "polydeoxyribonucleotide",
    "polydeoxyribonucleotide/polyribonucleotide hybrid",
    "cyclic-pseudo-peptide",
    "peptide nucleic acid",
    "other",
}
# The polymer types gemmi infers from residues.
GEMMI_POLYMER_TYPES = {
    gemmi.PolymerType.PeptideL: "polypeptide(L)",
    gemmi.PolymerType.PeptideD: "polypeptide(D)",
    gemmi.PolymerType.Rna: "polyribonucleotide",
    gemmi.PolymerType.Dna: "polydeoxyribonucleotide",
    gemmi.PolymerType.DnaRnaHybrid: "polydeoxyribonucleotide/polyribonucleotide hybrid",
}


def writable_columns(st: Structure) -> tuple[Columns, Columns, Columns]:
    """Return the standard columns of atoms, residues, and chains, filling defaults.

    Raises
    ------
    ValueError
        If ``st`` cannot be written, naming why.
    """
    _check_structure(st)
    chains = filled(
        st.chains,
        label_asym_id=chain_names(len(st.chains)),
        entity_type="",
        entity_id=np.arange(1, len(st.chains) + 1).astype(str),
        polymer_type="",
        description="",
    )
    # Empty types are inferred, the polymer_type after the entity_type it depends on.
    entity_type, polymer_type = _inferred_types(st)
    given_entity, given_polymer = chains["entity_type"], chains["polymer_type"]
    chains["entity_type"] = np.where(given_entity == "", entity_type, given_entity)
    untyped = (chains["entity_type"] == "polymer") & (given_polymer == "")
    chains["polymer_type"] = np.where(untyped, polymer_type, given_polymer)
    _check_types(chains)
    # auth_asym_id follows label_asym_id, so it is filled after it.
    chains |= filled(st.chains, auth_asym_id=chains["label_asym_id"])

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
        **filled(
            st.residues,
            label_seq_id=np.where(polymer, place, -1),
            ins_code="",
            comp_type="",
        ),
    }
    # auth_seq_id follows label_seq_id, so it is filled after it.
    seq = residues["label_seq_id"]
    residues |= filled(st.residues, auth_seq_id=np.where(seq > 0, seq, place))

    atoms = {
        "name": st.atoms.name,
        "element": st.atoms.element,
        "xyz": st.atoms.xyz,
        **filled(st.atoms, altloc="", occupancy=1.0, b_factor=np.nan, charge=0),
    }
    return atoms, residues, chains


def _check_structure(st: Structure) -> None:
    """Raise ValueError naming the first reason ``st`` cannot be written."""
    if not len(st.atoms):
        msg = "Cannot write a structure with no atoms."
        raise ValueError(msg)
    missing = [
        f"{level}.{name}"
        for level, names in REQUIRED.items()
        for name in names
        if name not in getattr(st, level)
    ]
    if missing:
        msg = f"Cannot write without {missing}."
        raise ValueError(msg)

    xyz = st.atoms.xyz
    if xyz.shape != (len(st.atoms), 3):
        msg = f"xyz must have shape (n_atoms, 3), not {xyz.shape}."
        raise ValueError(msg)
    unplaced = ~np.isfinite(xyz).all(axis=1)
    if unplaced.any():
        msg = (
            f"{int(unplaced.sum())} atoms have NaN or infinite coordinates; drop them "
            "with st.atoms[np.isfinite(st.atoms.xyz).all(axis=1)].extract()."
        )
        raise ValueError(msg)

    if "label_asym_id" in st.chains:
        labels, counts = np.unique(st.chains.label_asym_id, return_counts=True)
        if (counts > 1).any():
            msg = f"Chains share label_asym_id {labels[counts > 1].tolist()}."
            raise ValueError(msg)


def _check_types(chains: Columns) -> None:
    """Raise ValueError if a chain type, given or inferred, cannot be written."""
    unknown = sorted(set(chains["entity_type"].tolist()) - ENTITY_TYPES)
    if unknown:
        msg = f"entity_type {unknown} is not one of {sorted(ENTITY_TYPES)}."
        raise ValueError(msg)
    polymer = chains["entity_type"] == "polymer"
    lacking = polymer & ~np.isin(chains["polymer_type"], sorted(POLYMER_TYPES))
    if lacking.any():
        msg = (
            f"Polymer chains {chains['label_asym_id'][lacking].tolist()} need a "
            f"polymer_type, one of {sorted(POLYMER_TYPES)}."
        )
        raise ValueError(msg)


def _inferred_types(st: Structure) -> tuple[NDArray[np.str_], NDArray[np.str_]]:
    """Return the entity_type and polymer_type of each chain, judged from its residues.

    A chain is a polymer of the type gemmi's check_polymer_type finds, else water if all
    its residues are, and non-polymer otherwise. A single residue is never a polymer,
    and branched and macrolide chains are never inferred.
    """
    residues = []
    for name in st.residues.name.tolist():
        residue = gemmi.Residue()
        residue.name = name
        residues.append(residue)
    # gemmi counts residues it does not tabulate as amino acids if they have CA, and as
    # nucleotides if they have P.
    for i in np.flatnonzero(np.isin(st.atoms.name, ["CA", "P"])).tolist():
        atom = gemmi.Atom()
        atom.name = str(st.atoms.name[i])
        atom.element = gemmi.Element(str(st.atoms.element[i]))
        residues[st.atom_to_residue[i]].add_atom(atom)

    chains = [gemmi.Chain("") for _ in range(len(st.chains))]
    for residue, chain in zip(residues, st.residue_to_chain.tolist(), strict=True):
        chains[chain].add_residue(residue)

    entity_type, polymer_type = [], []
    for chain in chains:
        found = GEMMI_POLYMER_TYPES.get(chain.whole().check_polymer_type(), "")
        water = len(chain) > 0 and all(residue.is_water() for residue in chain)
        entity_type.append("polymer" if found else "water" if water else "non-polymer")
        polymer_type.append(found)
    return np.array(entity_type), np.array(polymer_type)


def unwritten(st: Structure, residues: Columns, chains: Columns) -> list[str]:
    """Name what ``st`` holds that an mmCIF file cannot.

    Those are columns that are not standard fields, edge groups other than ``bond`` and
    ``struct_conn``, bonds between residues, residues with no atoms outside polymer
    sequences, and metadata keys that ``read`` does not set. Standard columns are the
    Fields of each class, such as ``AtomView.xyz`` and ``Bond.order``. ``residues`` and
    ``chains`` are the columns ``writable_columns`` returns.
    """
    observed = np.bincount(st.atom_to_residue, minlength=len(st.residues)) > 0
    lost = int((~observed & ~in_sequence(st, residues, chains)).sum())
    names = [f"{lost} residues with no atoms outside polymer sequences"] if lost else []
    for view in (st.atoms, st.residues, st.chains):
        level = f"{view.level.name.lower()}s"
        names += [
            f"{level}[{name!r}]"
            for name in view.columns
            if not isinstance(getattr(type(view), name, None), Field)
        ]
        for name, edges in view.edges.items():
            if not len(edges):
                continue
            kind = WRITTEN_EDGES.get((level, name))
            if kind is None:
                names.append(f"{level}.edges[{name!r}]")
                continue
            names += [
                f"{level}.edges[{name!r}][{column!r}]"
                for column in edges.columns
                if not isinstance(getattr(kind, column, None), Field)
            ]

    # _chem_comp_bond lists bonds by residue type, so it cannot hold these.
    bond = st.atoms.edges.get("bond")
    if bond is not None:
        a2r = st.atom_to_residue
        between = int((a2r[bond.src] != a2r[bond.dst]).sum())
        names += [f"{between} bonds between residues"] if between else []

    known = ENTRY_KEYS | READ_KEYS
    names += [f"metadata[{key!r}]" for key in st.metadata if key not in known]
    return names


def in_sequence(st: Structure, residues: Columns, chains: Columns) -> NDArray[np.bool_]:
    """Return which residues polymer sequences list.

    Those are polymer residues with a label_seq_id, in entities whose chains each
    number them from 1 to the same end without gaps.
    """
    r2c = st.residue_to_chain
    entity = chains["entity_id"]
    listed = (chains["entity_type"][r2c] == "polymer") & (residues["label_seq_id"] > 0)
    rows = np.flatnonzero(listed)
    chain, seq = r2c[rows], residues["label_seq_id"][rows]
    # A chain numbered 1 to n without gaps has n positions.
    positions = np.bincount(chain[group_rows(chain, seq)[0]], minlength=len(entity))
    last = np.zeros(len(entity), positions.dtype)
    np.maximum.at(last, chain, seq)

    unsequenced = set(entity[positions != last].tolist())
    ends: dict[str, set[int]] = {}
    held = last > 0
    for entity_id, end in zip(entity[held].tolist(), last[held].tolist(), strict=True):
        ends.setdefault(entity_id, set()).add(end)
    unsequenced |= {entity_id for entity_id, end in ends.items() if len(end) > 1}
    return listed & ~np.isin(entity[r2c], sorted(unsequenced))
