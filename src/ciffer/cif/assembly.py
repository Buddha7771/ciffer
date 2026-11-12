from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import numpy as np

from ciffer.structure import Structure, concat

if TYPE_CHECKING:
    from .category import Category


def chosen_assembly(assemblies: list[str], *, assembly: bool | int | str) -> str | None:
    """Return the assembly id to build, or None to keep the deposited chains.

    >>> chosen_assembly(["1", "2"], assembly=True)
    '1'
    >>> chosen_assembly(["1", "2"], assembly=2)
    '2'
    >>> chosen_assembly([], assembly=True) is None
    True

    """
    if assembly is False:
        return None
    if assembly is True:
        return assemblies[0] if assemblies else None
    return str(assembly)  # ids are text, but users write 1 for "1"


def assemble(
    st: Structure, assembly_id: str, *, gen: Category, oper: Category
) -> Structure:
    """Copy chains by the operators of an assembly.

    Copies are named as gemmi names them: the copy of author chain A made by the n-th
    operator is A{n}, or the next free number, and its label_asym_id gets the same
    number. Connections between chains of different copies are not made.

    Examples
    --------
    .. code-block:: python

        gen = Category(block, "_pdbx_struct_assembly_gen.")
        oper = Category(block, "_pdbx_struct_oper_list.")
        assemble(st, "1", gen=gen, oper=oper)

    """
    rows = np.flatnonzero(gen.get("assembly_id") == assembly_id)
    if not rows.size:
        ids = list(dict.fromkeys(gen.get("assembly_id").tolist()))
        msg = f"Assembly {assembly_id!r} is not in the file; assemblies are {ids}."
        raise ValueError(msg)

    rotations = np.array(
        [[oper[f"matrix[{i}][{j}]"] for j in "123"] for i in "123"], dtype=float
    ).transpose(2, 0, 1)
    shifts = np.array([oper[f"vector[{i}]"] for i in "123"], dtype=float).T
    operators = dict(
        zip(oper["id"].tolist(), zip(rotations, shifts, strict=True), strict=True)
    )

    copies, used, counter = [], set(), 0
    for row in rows:
        chains = gen["asym_id_list"][row].split(",")
        selected = np.isin(st.chains.label_asym_id, chains)

        for steps in _operator_steps(gen["oper_expression"][row]):
            counter += 1
            if not selected.any():
                continue
            part = st.chains[selected].extract()
            xyz = part.atoms.xyz
            for step in steps:
                rotation, shift = operators[step]
                xyz = xyz @ rotation.T + shift
            part.atoms.xyz = xyz

            # gemmi numbers copies by operator, skipping names already taken.
            auth = part.chains.auth_asym_id.tolist()
            number = {}
            for name in dict.fromkeys(auth):
                n = counter
                while f"{name}{n}" in used:
                    n += 1
                used.add(f"{name}{n}")
                number[name] = str(n)
            suffix = [number[name] for name in auth]
            part.chains.auth_asym_id = np.char.add(auth, suffix)
            part.chains.label_asym_id = np.char.add(part.chains.label_asym_id, suffix)
            copies.append(part)

    if not copies:
        return st.chains[np.zeros(len(st.chains), bool)].extract()
    return concat(copies)


def _operator_steps(expression: str) -> list[tuple[str, ...]]:
    """Return the operator ids of each copy, in the order they are applied.

    Operators of a product apply from right to left, as biotite applies them.

    >>> _operator_steps("1,3-4")
    [('1',), ('3',), ('4',)]
    >>> _operator_steps("(X0)(1-2)")
    [('1', 'X0'), ('2', 'X0')]

    """
    groups = [group for group in expression.replace(")", "").split("(") if group]
    steps = []
    for group in reversed(groups):
        ids = []
        for part in group.split(","):
            first, _, last = part.partition("-")
            if last:
                ids += [str(n) for n in range(int(first), int(last) + 1)]
            else:
                ids.append(part)
        steps.append(ids)
    return list(itertools.product(*steps))
