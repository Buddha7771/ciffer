from __future__ import annotations

import gzip
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Literal, overload

import gemmi
import numpy as np

from ciffer.structure import Structure

from .altloc import conformer_rows
from .assembly import assemble, chosen_assembly
from .category import Category
from .hierarchy import from_categories, to_categories, unwritten
from .metadata import entry_metadata

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
    atoms and no bonds within residues, and its chains have label ids gemmi assigns.
    Without SEQRES records, its polymer residues have a label_seq_id of -1.

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
    if altloc not in (None, "occupancy", "all"):
        msg = f"altloc must be None, 'occupancy', or 'all', got {altloc!r}."
        raise ValueError(msg)

    block = _mmcif_block(path)
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
    """Write a structure to an mmCIF file, gzipped if the name ends in .gz.

    Parameters
    ----------
    st: Structure
        Needs atom ``name``, ``element``, and ``xyz``, and residue ``name``. Missing
        standard columns are filled, and other columns are skipped with a warning.
    path: str | os.PathLike[str]
        A ``.cif`` or ``.mmcif`` file, optionally gzipped.

    Notes
    -----
    Missing chain types are judged from residues, so give them for branched and
    macrolide chains. Missing ``label_seq_id`` numbers each polymer from 1, so give it
    when a polymer has gaps.

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
    if not path.name.lower().removesuffix(".gz").endswith((".cif", ".mmcif")):
        msg = f"{path} is not a .cif or .mmcif file, optionally gzipped."
        raise ValueError(msg)

    doc = gemmi.cif.Document()
    block = doc.add_new_block(str(st.metadata.get("id", "ciffer")))
    for category, items in to_categories(st).items():
        block.set_mmcif_category(category, items)

    text = doc.as_string()
    if path.name.lower().endswith(".gz"):
        with gzip.open(path, "wt") as file:
            file.write(text)
    else:
        path.write_text(text)

    skipped = unwritten(st)
    if skipped:
        warnings.warn(f"Not written: {', '.join(skipped)}.", stacklevel=2)


def _mmcif_block(path: str | os.PathLike[str]) -> gemmi.cif.Block:
    """Return the mmCIF block of a file, converting a PDB file as gemmi does."""
    path = Path(path)
    stem = path.name.lower().removesuffix(".gz")
    if stem.endswith((".cif", ".mmcif")):
        return gemmi.cif.read(str(path)).sole_block()
    if stem.endswith((".pdb", ".ent")):
        pdb = gemmi.read_structure(str(path))
        pdb.setup_entities()
        pdb.assign_label_seq_id()
        return pdb.make_mmcif_document().sole_block()
    msg = f"{path} is not a .cif, .mmcif, .pdb, or .ent file, optionally gzipped."
    raise ValueError(msg)
