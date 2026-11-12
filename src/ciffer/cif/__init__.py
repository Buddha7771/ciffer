from __future__ import annotations

import gzip
import re
import uuid
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Literal, overload

import gemmi
import numpy as np

from ciffer.structure import Structure

from .altloc import conformer_rows
from .assembly import assemble, chosen_assembly
from .category import Category
from .hierarchy import from_categories, to_categories
from .metadata import entry_metadata
from .pdb import pdb_block, pdb_text
from .rcsb import download_mmcif
from .writable import unwritten, writable_columns

if TYPE_CHECKING:
    import os


@overload
def read(
    path: str | os.PathLike[str],
    *,
    model: int | None = None,
    altloc: Literal["occupancy", "all"] | None = None,
    assembly: bool | int | str = False,
) -> Structure: ...


@overload
def read(
    path: str | os.PathLike[str],
    *,
    model: list[int] | Literal["all"],
    altloc: Literal["occupancy", "all"] | None = None,
    assembly: bool | int | str = False,
) -> list[Structure]: ...


def read(
    path: str | os.PathLike[str],
    *,
    model: int | list[int] | Literal["all"] | None = None,
    altloc: Literal["occupancy", "all"] | None = None,
    assembly: bool | int | str = False,
) -> Structure | list[Structure]:
    """Read an mmCIF or PDB file.

    A PDB file is read as gemmi converts it to mmCIF, so it has no residues without
    atoms and no bonds within residues. Its chains get label ids A, B, ..., Z, AA, BA,
    and so on, in the order of the file. Without SEQRES records, its polymer residues
    are numbered from 1 in the order of the file.

    Parameters
    ----------
    path: str | os.PathLike[str]
        A ``.cif``, ``.mmcif``, ``.pdb``, or ``.ent`` file, optionally gzipped.
    model: int | list[int] | Literal["all"] | None, optional
        Model number to read, or a list of them. None reads the first model, and
        ``"all"`` reads every model.
    altloc: Literal["occupancy", "all"] | None, optional
        None keeps the first conformer as gemmi does. ``"occupancy"`` keeps, at each
        residue position, the altloc with the highest total occupancy as biotite does.
        ``"all"`` keeps every alternative location.
    assembly: bool | int | str, optional
        Assembly id to build, such as ``"1"`` or ``1``. True builds the first assembly
        of the file, or keeps the deposited chains if the file has none;
        ``metadata["assembly"]`` tells which.

    Returns
    -------
    Structure | list[Structure]
        A list, in the given order, when ``model`` is a list or ``"all"``.

    Examples
    --------
    .. code-block:: python

        st = ciffer.read("1abc.cif.gz")
        st = ciffer.read("design.pdb")
        pair = ciffer.read("2k9q.cif", model=[1, 3])
        ensemble = ciffer.read("2k9q.cif", model="all")
        likely = ciffer.read("1ejg.cif", altloc="occupancy")
        every = ciffer.read("1ejg.cif", altloc="all")
        tetramer = ciffer.read("1stp.cif", assembly="1")
        biological = ciffer.read("1stp.cif", assembly=True)

    """
    path = Path(path)
    if _file_format(path) == "pdb":
        block = pdb_block(path)
    else:
        block = gemmi.cif.read(str(path)).sole_block()
    return _from_block(block, model=model, altloc=altloc, assembly=assembly)


@overload
def fetch(
    pdb_id: str,
    directory: str | os.PathLike[str] | None = None,
    *,
    model: int | None = None,
    altloc: Literal["occupancy", "all"] | None = None,
    assembly: bool | int | str = False,
) -> Structure: ...


@overload
def fetch(
    pdb_id: str,
    directory: str | os.PathLike[str] | None = None,
    *,
    model: list[int] | Literal["all"],
    altloc: Literal["occupancy", "all"] | None = None,
    assembly: bool | int | str = False,
) -> list[Structure]: ...


def fetch(
    pdb_id: str,
    directory: str | os.PathLike[str] | None = None,
    *,
    model: int | list[int] | Literal["all"] | None = None,
    altloc: Literal["occupancy", "all"] | None = None,
    assembly: bool | int | str = False,
) -> Structure | list[Structure]:
    """Read an entry of the PDB, downloaded from RCSB as mmCIF.

    Parameters
    ----------
    pdb_id: str
        A 4-character or extended id, such as ``"1stp"`` or ``"pdb_00001stp"``.
    directory: str | os.PathLike[str] | None, optional
        Directory to save the file in, as ``1stp.cif`` with the id in lowercase. A file
        saved there before is read instead of downloading; delete it to download again.
        None keeps the file in memory only.
    model, altloc, assembly
        As in ``read``.

    Returns
    -------
    Structure | list[Structure]
        A list, in the given order, when ``model`` is a list or ``"all"``.

    Raises
    ------
    ValueError
        If the id has characters other than letters, digits, and underscores.
    FileNotFoundError
        If RCSB has no entry of the id.

    Examples
    --------
    .. code-block:: python

        st = ciffer.fetch("1stp")
        st = ciffer.fetch("pdb_00001stp", assembly="1")
        st = ciffer.fetch("1stp", "data")

    """
    if not re.fullmatch(r"[A-Za-z0-9_]+", pdb_id):
        msg = f"{pdb_id!r} is not a PDB id."
        raise ValueError(msg)

    path = None if directory is None else Path(directory) / f"{pdb_id.lower()}.cif"
    if path is not None and path.exists():
        return read(path, model=model, altloc=altloc, assembly=assembly)

    data = download_mmcif(pdb_id)
    # Read before saving, so that a file gemmi cannot read is not kept.
    block = gemmi.cif.read_string(data).sole_block()
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # A name of its own keeps other processes saving the same entry apart.
        temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_bytes(data)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    return _from_block(block, model=model, altloc=altloc, assembly=assembly)


def _from_block(
    block: gemmi.cif.Block,
    *,
    model: int | list[int] | Literal["all"] | None,
    altloc: Literal["occupancy", "all"] | None,
    assembly: bool | int | str,
) -> Structure | list[Structure]:
    """Build the structures that ``read`` returns from an mmCIF block."""
    if altloc not in (None, "occupancy", "all"):
        msg = f"altloc must be None, 'occupancy', or 'all', got {altloc!r}."
        raise ValueError(msg)

    site = Category(block, "_atom_site.")
    entity = Category(block, "_entity.")
    entity_poly = Category(block, "_entity_poly.")
    chem_comp = Category(block, "_chem_comp.")
    struct_asym = Category(block, "_struct_asym.")
    scheme = Category(block, "_pdbx_poly_seq_scheme.")
    comp_bond = Category(block, "_chem_comp_bond.")
    conn = Category(block, "_struct_conn.")
    gen = Category(block, "_pdbx_struct_assembly_gen.")
    oper = Category(block, "_pdbx_struct_oper_list.")

    # A file without model numbers holds one model, numbered 1.
    numbers = site.get("pdbx_PDB_model_num")
    numbers = np.where(numbers == "", "1", numbers).astype(int)
    models = list(dict.fromkeys(numbers.tolist()))

    if model is None:
        chosen = models[:1]
    elif model == "all":
        chosen = models
    elif isinstance(model, int):
        chosen = [model]
    else:
        chosen = model
    missing = [number for number in chosen if number not in models]
    if missing:
        msg = f"Models {missing} are not in the file; models are {models}."
        raise ValueError(msg)

    assemblies = list(dict.fromkeys(gen.get("assembly_id").tolist()))
    assembly_id = chosen_assembly(assemblies, assembly=assembly)

    structures = []
    for number in chosen:
        model_site = site.take(np.flatnonzero(numbers == number))
        model_site = model_site.take(conformer_rows(model_site, altloc))
        st = from_categories(
            model_site,
            entity=entity,
            entity_poly=entity_poly,
            chem_comp=chem_comp,
            struct_asym=struct_asym,
            scheme=scheme,
            comp_bond=comp_bond,
            conn=conn,
        )
        metadata = {**entry_metadata(block), "model": number, "models": models.copy()}
        if assembly_id is not None:
            st = assemble(st, assembly_id, gen=gen, oper=oper)
            metadata["assembly"] = assembly_id
        if assemblies:
            metadata["assemblies"] = assemblies.copy()
        st.metadata = metadata
        structures.append(st)

    return structures if isinstance(model, list) or model == "all" else structures[0]


def write(st: Structure, path: str | os.PathLike[str]) -> None:
    """Write a structure to an mmCIF or PDB file, gzipped if the name ends in .gz.

    Parameters
    ----------
    st: Structure
        Needs atom ``name``, ``element``, and ``xyz``, and residue ``name``. Missing
        standard columns are filled, and other columns and metadata keys are skipped
        with a warning.
    path: str | os.PathLike[str]
        A ``.cif``, ``.mmcif``, ``.pdb``, or ``.ent`` file, optionally gzipped.

    Notes
    -----
    Missing chain types are judged from residues, so give them for branched and
    macrolide chains. Missing ``label_seq_id`` numbers each polymer from 1, and an
    entity whose chains do not each run from 1 to the same end without gaps, as after
    cropping, is written without its sequence. Readers may give no bonds to residue
    types that ``bond`` leaves out, so give bonds for every residue or for none. Chains
    sharing an ``entity_id`` but holding other molecules, as after ``concat``, are
    written as separate entities.

    A PDB file loses, without a warning, bonds, chain descriptions, hydrogen bonds,
    connection orders, and the metadata ``authors``, ``r_free``, ``release_date``, and
    ``doi``.

    Examples
    --------
    .. code-block:: python

        st = ciffer.read("input.cif")
        st.atoms.xyz = predicted
        st.atoms.b_factor = plddt
        ciffer.write(st, "predicted.cif.gz")

    """
    if not isinstance(st, Structure):
        msg = (
            f"write takes a Structure, not {type(st).__name__}; extract() a view "
            "first, or write each model of a list to its own file."
        )
        raise TypeError(msg)
    path = Path(path)
    as_pdb = _file_format(path) == "pdb"

    atoms, residues, chains = writable_columns(st)
    doc = gemmi.cif.Document()
    block = doc.add_new_block(str(st.metadata.get("id", "ciffer")))
    for category, items in to_categories(st, atoms, residues, chains).items():
        block.set_mmcif_category(category, items)

    text = pdb_text(block) if as_pdb else doc.as_string()
    if path.name.lower().endswith(".gz"):
        with gzip.open(path, "wt") as file:
            file.write(text)
    else:
        path.write_text(text)

    skipped = unwritten(st, residues, chains)
    if skipped:
        warnings.warn(f"Not written: {', '.join(skipped)}.", stacklevel=2)


def _file_format(path: Path) -> Literal["mmcif", "pdb"]:
    """Return the format the file name tells, or raise ValueError."""
    stem = path.name.lower().removesuffix(".gz")
    if stem.endswith((".cif", ".mmcif")):
        return "mmcif"
    if stem.endswith((".pdb", ".ent")):
        return "pdb"
    msg = f"{path} is not a .cif, .mmcif, .pdb, or .ent file, optionally gzipped."
    raise ValueError(msg)
