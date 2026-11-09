from __future__ import annotations

import os
from typing import TYPE_CHECKING, Literal, overload

import gemmi
import numpy as np

from .category import Category, from_categories

if TYPE_CHECKING:
    from ciffer.structure import Structure


@overload
def read(
    path: str | os.PathLike[str],
    *,
    model: int | None = None,
) -> Structure: ...


@overload
def read(
    path: str | os.PathLike[str],
    *,
    model: Literal["all"],
) -> list[Structure]: ...


def read(
    path: str | os.PathLike[str],
    *,
    model: int | Literal["all"] | None = None,
) -> Structure | list[Structure]:
    """Read an mmCIF file.

    Parameters
    ----------
    path: str | os.PathLike[str]
        An mmCIF file, optionally gzipped.
    model: int | Literal["all"] | None, optional
        Model number to read. None reads the first model.
        ``"all"`` returns a list with one structure per model.

    Returns
    -------
    Structure | list[Structure]
        A list only when ``model="all"``.

    Examples
    --------
    .. code-block:: python

        st = ciffer.read("1abc.cif.gz")
        ensemble = ciffer.read("2k9q.cif", model="all")

    """
    block = gemmi.cif.read(os.fspath(path)).sole_block()
    site = Category(block, "_atom_site.")
    entity = Category(block, "_entity.")
    entity_poly = Category(block, "_entity_poly.")
    chem_comp = Category(block, "_chem_comp.")

    # A file without model numbers holds one model, numbered 1.
    numbers = site.get("pdbx_PDB_model_num")
    numbers = np.where(numbers == "", "1", numbers).astype(int)
    models = list(dict.fromkeys(numbers.tolist()))

    if model is None:
        chosen = models[:1]
    elif model == "all":
        chosen = models
    elif model in models:
        chosen = [model]
    else:
        msg = f"Model {model!r} is not in the file; models are {models}."
        raise ValueError(msg)

    structures = []
    for number in chosen:
        rows = np.flatnonzero(numbers == number)
        st = from_categories(site.take(rows), entity, entity_poly, chem_comp)
        st.metadata["model"] = number
        structures.append(st)

    return structures if model == "all" else structures[0]
