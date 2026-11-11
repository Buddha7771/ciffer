from __future__ import annotations

from typing import TYPE_CHECKING, Any

import gemmi
import numpy as np

from ciffer.structure import Structure
from ciffer.table import Table
from ciffer.view import Bond, Field, StructConn

from .bonds import (
    bond_edges,
    comp_bond_category,
    struct_conn_category,
    struct_conn_edges,
)
from .category import Category, group_rows, to_cif
from .metadata import ENTRY_KEYS, metadata_categories

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

    from ciffer.view import View

    Columns = dict[str, NDArray[Any]]
    Items = dict[str, list[str | bool | None]]

REQUIRED = {"atoms": ["name", "element", "xyz"], "residues": ["name"]}
WRITTEN_EDGES = {("atoms", "bond"): Bond, ("atoms", "struct_conn"): StructConn}
# PDB files hold struct_conn as LINK and SSBOND records, and no bonds.
PDB_EDGES = {("atoms", "struct_conn"): StructConn}
# Entry metadata a PDB file keeps; its HEADER holds an id of up to four letters.
PDB_ENTRY_KEYS = ENTRY_KEYS - {"authors", "r_free", "release_date", "doi"}
PDB_ID_LENGTH = 4
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


def to_categories(st: Structure) -> dict[str, Items]:
    """Return the gemmi items of each mmCIF category, keyed like ``"_atom_site."``.

    Raises
    ------
    ValueError
        If ``st`` cannot be written, naming why.
    """
    _check_writable(st)
    atoms, residues, chains = _filled_tables(st)
    _check_types(chains)
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


def _check_writable(st: Structure) -> None:
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

    scheme_residues = np.flatnonzero(_in_sequence(st, residues, chains))
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


def _in_sequence(
    st: Structure, residues: Columns, chains: Columns
) -> NDArray[np.bool_]:
    """Return which residues polymer sequences list: polymer ones with a position."""
    polymer = chains["entity_type"][st.residue_to_chain] == "polymer"
    return polymer & (residues["label_seq_id"] > 0)


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


def unwritten(st: Structure, *, pdb: bool = False) -> list[str]:
    """Name what ``st`` holds that the file cannot.

    Those are columns that are not standard fields, edge groups other than ``bond`` and
    ``struct_conn``, bonds between residues, residues with no atoms outside polymer
    sequences, and metadata keys that ``read`` does not set. Standard columns are the
    Fields of each class, such as ``AtomView.xyz`` and ``Bond.order``. A PDB file also
    cannot hold ``bond`` or the metadata ``authors``, ``r_free``, ``release_date``, and
    ``doi``, nor what ``_pdb_losses`` names.
    """
    written_edges = PDB_EDGES if pdb else WRITTEN_EDGES
    _, residues, chains = _filled_tables(st)
    observed = np.bincount(st.atom_to_residue, minlength=len(st.residues)) > 0
    lost = int((~observed & ~_in_sequence(st, residues, chains)).sum())
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
            kind = written_edges.get((level, name))
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
    if bond is not None and not pdb:
        a2r = st.atom_to_residue
        between = int((a2r[bond.src] != a2r[bond.dst]).sum())
        names += [f"{between} bonds between residues"] if between else []

    known = (PDB_ENTRY_KEYS if pdb else ENTRY_KEYS) | READ_KEYS
    names += [f"metadata[{key!r}]" for key in st.metadata if key not in known]
    return names + _pdb_losses(st) if pdb else names


def _pdb_losses(st: Structure) -> list[str]:
    """Name what a PDB file loses beyond what an mmCIF file does.

    Those are chain descriptions, hydrogen bonds and orders of ``struct_conn``, an id
    longer than four letters, and a model number other than 1.
    """
    names = []
    if "description" in st.chains and (st.chains.description != "").any():
        names.append("chains['description']")
    conn = st.atoms.edges.get("struct_conn")
    if conn is not None and "type" in conn:
        hydrog = int((conn["type"] == "hydrog").sum())
        names += [f"{hydrog} hydrogen bonds in struct_conn"] if hydrog else []
    if conn is not None and "order" in conn and (conn["order"] != 0).any():
        names.append("atoms.edges['struct_conn']['order']")
    if len(str(st.metadata.get("id", ""))) > PDB_ID_LENGTH:
        names.append("metadata['id']")
    # A file of one model has no MODEL record, so it reads back as model 1.
    if st.metadata.get("model", 1) != 1:
        names.append("metadata['model']")
    return names


def _filled_tables(st: Structure) -> tuple[Columns, Columns, Columns]:
    """Return the standard columns of atoms, residues, and chains, filling defaults."""
    chains = _filled(
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
    # auth_asym_id follows label_asym_id, so it is filled after it.
    chains |= _filled(st.chains, auth_asym_id=chains["label_asym_id"])

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


def chain_names(n: int) -> list[str]:
    """Return n chain ids as PDB entries name them, the first letter changing fastest.

    >>> names = chain_names(703)
    >>> names[24:28], names[50:54], names[-1]
    (['Y', 'Z', 'AA', 'BA'], ['YA', 'ZA', 'AB', 'BB'], 'AAA')

    """
    names = []
    for i in range(n):
        name, rest = "", i
        while rest >= 0:
            name += chr(ord("A") + rest % 26)
            rest = rest // 26 - 1
        names.append(name)
    return names


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
