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
    model: list[int] | Literal["all"],
) -> list[Structure]: ...


def read(
    path: str | os.PathLike[str],
    *,
    model: int | list[int] | Literal["all"] | None = None,
) -> Structure | list[Structure]:
    """Read an mmCIF file.

    Parameters
    ----------
    path: str | os.PathLike[str]
        An mmCIF file, optionally gzipped.
    model: int | list[int] | Literal["all"] | None, optional
        Model number to read, or a list of them. None reads the first model, and
        ``"all"`` reads every model.

    Returns
    -------
    Structure | list[Structure]
        A list, in the given order, when ``model`` is a list or ``"all"``.

    Examples
    --------
    .. code-block:: python

        st = ciffer.read("1abc.cif.gz")
        pair = ciffer.read("2k9q.cif", model=[1, 3])
        ensemble = ciffer.read("2k9q.cif", model="all")

    """
    block = gemmi.cif.read(os.fspath(path)).sole_block()
    site = Category(block, "_atom_site.")
    entity = Category(block, "_entity.")
    entity_poly = Category(block, "_entity_poly.")
    chem_comp = Category(block, "_chem_comp.")
    scheme = Category(block, "_pdbx_poly_seq_scheme.")

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

    structures = []
    for number in chosen:
        rows = np.flatnonzero(numbers == number)
        st = from_categories(site.take(rows), entity, entity_poly, chem_comp, scheme)
        st.metadata["model"] = number
        structures.append(st)

    return structures if isinstance(model, list) or model == "all" else structures[0]
