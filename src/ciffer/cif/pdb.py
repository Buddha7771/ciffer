from __future__ import annotations

from typing import TYPE_CHECKING

import gemmi
import numpy as np

from .category import Category, chain_names

if TYPE_CHECKING:
    from pathlib import Path

# PDB column widths of the _atom_site items gemmi would cut; gemmi writes chain names
# of two letters in columns 21 and 22.
PDB_WIDTHS = {
    "auth_asym_id": ("chain names", 2),
    "auth_atom_id": ("atom names", 4),
    "label_alt_id": ("altlocs", 1),
    "pdbx_PDB_ins_code": ("insertion codes", 1),
}


def pdb_block(path: Path) -> gemmi.cif.Block:
    """Return the mmCIF block gemmi converts a PDB file to, naming chains A, B, C."""
    pdb = gemmi.read_structure(str(path))
    pdb.setup_entities()
    # gemmi names chains like "Axp" for the polymer of chain A; they are renamed once
    # over all models, so a chain keeps its name in each.
    residues = [res for model in pdb for chain in model for res in chain]
    old = list(dict.fromkeys(res.subchain for res in residues))
    names = dict(zip(old, chain_names(len(old)), strict=True))
    for res in residues:
        res.subchain = names[res.subchain]
    for entity in pdb.entities:
        entity.subchains = [names[name] for name in entity.subchains]
    pdb.assign_label_seq_id()
    block = pdb.make_mmcif_document().sole_block()
    # gemmi reads the resolution of REMARK 2 but writes it only with REMARK 3.
    if pdb.resolution and not len(block.find_values("_refine.ls_d_res_high")):
        block.set_pair("_refine.ls_d_res_high", str(pdb.resolution))
    return block


def pdb_text(block: gemmi.cif.Block) -> str:
    """Return the PDB text of an mmCIF block, converting it as gemmi does."""
    site = Category(block, "_atom_site.")
    for item, (label, width) in PDB_WIDTHS.items():
        values = np.unique(site.get(item))
        too_long = values[np.char.str_len(values) > width]
        if too_long.size:
            msg = (
                f"PDB files hold {label} of up to {width} characters, not "
                f"{too_long[:5].tolist()}; write mmCIF."
            )
            raise ValueError(msg)

    pdb = gemmi.make_structure_from_block(block)
    # Residue names longer than three letters get short codes, with the real ones in
    # HET records, which gemmi reads back.
    pdb.shorten_ccd_codes()
    options = gemmi.PdbWriteOptions()
    options.cryst1_record = False  # a structure has no unit cell
    return pdb.make_pdb_string(options)
