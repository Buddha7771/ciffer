from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .category import Category, group_rows

if TYPE_CHECKING:
    from numpy.typing import NDArray


def conformer_rows(site: Category, altloc: str | None) -> NDArray[np.intp]:
    """Return the ``_atom_site`` rows to keep for an altloc choice.

    None keeps the first conformer as gemmi does, ``"occupancy"`` keeps the altloc
    with the highest total occupancy at each residue position, and ``"all"`` keeps
    every row.

    Examples
    --------
    .. code-block:: python

        site = site.take(conformer_rows(site, "occupancy"))

    """
    if altloc == "all":
        return np.arange(len(site))
    if altloc == "occupancy":
        return _highest_occupancy(site)
    return _first(site)


def _first(site: Category) -> NDArray[np.intp]:
    """Return the rows gemmi keeps when it removes alternative conformations.

    Each residue position keeps the residue type of its first row, and that residue
    keeps the first row of each atom name.
    """
    name, atom = site["label_comp_id"], site["label_atom_id"]
    first, position = _positions(site)
    candidates = np.flatnonzero(name == name[first][position])
    first_atoms, _ = group_rows(position[candidates], atom[candidates])
    return candidates[first_atoms]


def _highest_occupancy(site: Category) -> NDArray[np.intp]:
    """Return the rows of the altloc with the highest total occupancy at each position.

    Rows without an altloc are kept, and ties go to the earlier letter. A missing
    occupancy counts as 0.
    """
    altloc = site["label_alt_id"]
    occupancy = site.get("occupancy")
    occupancy = np.where(occupancy == "", "0", occupancy).astype(float)
    _, position = _positions(site)

    rows = np.flatnonzero(altloc != "")
    first, pair = group_rows(position[rows], altloc[rows])
    total = np.bincount(pair, weights=occupancy[rows])

    # Sort the (position, altloc) pairs so that the best one leads each position.
    pair_position, pair_altloc = position[rows][first], altloc[rows][first]
    order = np.lexsort((pair_altloc, -total, pair_position))
    best = order[np.unique(pair_position[order], return_index=True)[1]]

    keep = altloc == ""
    keep[rows[np.isin(pair, best)]] = True
    return np.flatnonzero(keep)


def _positions(site: Category) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Return the first row of each residue position, and the position of each row."""
    return group_rows(
        site["label_asym_id"],
        site["label_seq_id"],
        site.get("auth_seq_id"),
        site.get("pdbx_PDB_ins_code"),
    )
