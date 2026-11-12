from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, Literal

import gemmi
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

    from ciffer.table import EdgeList
    from ciffer.view import View

    Columns = dict[str, NDArray[Any]]
    Items = dict[str, list[str | bool | None]]


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


def group_rows(*columns: NDArray) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Group rows that are equal in every column.

    Returns the first row of each group and the group of each row, numbered by first
    appearance.

    Examples
    --------
    >>> chain = np.array(["A", "A", "A", "B"])
    >>> seq = np.array(["1", "1", "2", "1"])
    >>> group_rows(chain, seq)
    (array([0, 2, 3]), array([0, 0, 1, 2]))

    """
    if len(columns[0]) == 0:
        return np.zeros(0, dtype=np.intp), np.zeros(0, dtype=np.intp)

    rows = np.stack(columns, axis=1)
    _, first, group = np.unique(rows, axis=0, return_index=True, return_inverse=True)

    # np.unique numbers groups by sorted value; renumber them by first row.
    order = np.argsort(first)
    return first[order], np.argsort(order)[group.ravel()]


def to_cif(
    values: ArrayLike, *, null: Literal[False] | None = None
) -> list[str | bool | None]:
    """Return ``values`` as text for gemmi, with "" and NaN written as ``null``.

    gemmi writes None as ``?`` (unknown) and False as ``.`` (not applicable).

    >>> to_cif(np.array(["A", ""]))
    ['A', None]
    >>> to_cif(np.array([1.5, np.nan]), null=False)
    ['1.5', False]

    """
    text = np.asarray(values).astype(str).tolist()
    return [null if value in ("", "nan") else value for value in text]


def filled(holder: View | EdgeList, **defaults: ArrayLike) -> Columns:
    """Return the named columns of ``holder``, with the default of each missing one."""
    return {
        name: holder[name] if name in holder else np.broadcast_to(default, len(holder))
        for name, default in defaults.items()
    }


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
