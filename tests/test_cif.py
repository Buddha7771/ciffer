from pathlib import Path
from typing import Any

import gemmi
import numpy as np
import pytest

import ciffer
from ciffer import Structure

DATA = Path(__file__).parent / "data"


def gemmi_atoms(path: Path, model: int = 1) -> list:
    st = gemmi.read_structure(str(path))
    st.remove_alternative_conformations()
    model_ = next(m for m in st if m.num == model)
    return sorted(
        (
            res.subchain,
            str(res.seqid.num),
            res.seqid.icode.strip(),
            res.name,
            atom.name,
            *np.round(atom.pos.tolist(), 2),
        )
        for chain in model_
        for res in chain
        for atom in res
    )


def ciffer_atoms(st: Structure) -> list:
    residue = st.atom_to_residue
    chain = st.residue_to_chain[residue]
    return sorted(
        zip(
            st.chains["label_asym_id"][chain].tolist(),
            st.residues["auth_seq_id"][residue].tolist(),
            st.residues["ins_code"][residue].tolist(),
            st.residues["name"][residue].tolist(),
            st.atoms["name"].tolist(),
            *np.round(st.atoms["xyz"], 2).T.tolist(),
            strict=True,
        )
    )


@pytest.mark.parametrize(
    ("name", "options"),
    [
        ("1STP.cif.gz", {}),
        ("1L2Y_3models.cif.gz", {"model": 2}),
        ("1EJG.cif.gz", {}),
    ],
)
def test_read_matches_gemmi(name: str, options: dict[str, Any]):
    st = ciffer.read(DATA / name, **options)

    assert ciffer_atoms(st) == gemmi_atoms(DATA / name, **options)


def test_read_adds_unobserved_residues():
    st = ciffer.read(DATA / "1STP.cif.gz")

    observed = np.bincount(st.atom_to_residue, minlength=len(st.residues)) > 0
    assert (~observed).sum() == 38
    polymer = st.residues[st.residue_to_chain == 0]
    assert polymer["label_seq_id"].tolist() == list(range(1, 160))
    gemmi_st = gemmi.read_structure(str(DATA / "1STP.cif.gz"))
    gemmi_st.setup_entities()
    assert polymer["name"].tolist() == gemmi_st.entities[0].full_sequence


def test_read_altloc():
    one = ciffer.read(DATA / "1EJG.cif.gz")
    every = ciffer.read(DATA / "1EJG.cif.gz", altloc="all")

    # Positions 22 and 25 hold two residue types; one is kept by default.
    polymer = one.residues[one.residue_to_chain == 0]
    assert polymer["label_seq_id"].tolist() == list(range(1, 47))
    at_22 = every.residues["label_seq_id"] == 22
    assert every.residues["name"][at_22].tolist() == ["PRO", "SER"]
    assert len(every.atoms) == 843  # every row of atom_site

    # Asn 12 has more occupancy in B, though A comes first in the file.
    likely = ciffer.read(DATA / "1EJG.cif.gz", altloc="occupancy")
    for st, letter in [(one, "A"), (likely, "B")]:
        at_12 = st.residues["label_seq_id"][st.atom_to_residue] == 12
        assert set(st.atoms["altloc"][at_12].tolist()) - {""} == {letter}
    at_22 = likely.residues["label_seq_id"] == 22
    assert likely.residues["name"][at_22].tolist() == ["PRO"]  # 0.57 against 0.43


def test_read_keeps_first_type_without_atoms(tmp_path: Path):
    doc = gemmi.cif.read(str(DATA / "1EJG.cif.gz"))
    table = doc.sole_block().find_mmcif_category("_atom_site.")
    seq = table.find_column("label_seq_id")
    for row in reversed(range(len(table))):
        if seq[row] == "22":
            table.remove_row(row)
    path = tmp_path / "1EJG.cif"
    doc.write_file(str(path))

    st = ciffer.read(path, altloc="all")

    at_22 = st.residues["label_seq_id"] == 22
    assert st.residues["name"][at_22].tolist() == ["PRO"]  # the first listed type


def test_read_all_models():
    models = ciffer.read(DATA / "1L2Y_3models.cif.gz", model="all")

    assert [st.metadata["model"] for st in models] == [1, 2, 3]
    second = ciffer.read(DATA / "1L2Y_3models.cif.gz", model=2)
    assert np.array_equal(models[1].atoms["xyz"], second.atoms["xyz"])


def test_read_model_list():
    models = ciffer.read(DATA / "1L2Y_3models.cif.gz", model=[3, 1])

    assert [st.metadata["model"] for st in models] == [3, 1]


def test_read_types():
    st = ciffer.read(DATA / "1STP.cif.gz")

    assert st.chains["entity_type"].tolist() == ["polymer", "non-polymer", "water"]
    assert st.chains["polymer_type"].tolist() == ["polypeptide(L)", "", ""]
    assert set(st.residues["comp_type"].tolist()) == {
        "L-peptide linking",
        "peptide linking",
        "non-polymer",
    }


def test_read_without_optional_items(tmp_path: Path):
    doc = gemmi.cif.read(str(DATA / "1STP.cif.gz"))
    block = doc.sole_block()
    loop = block.find_loop("_atom_site.id").get_loop()
    items = ["pdbx_PDB_model_num", "occupancy", "B_iso_or_equiv", "pdbx_formal_charge"]
    for item in items:
        loop.remove_column(f"_atom_site.{item}")
    block.find_mmcif_category("_pdbx_poly_seq_scheme.").erase()
    path = tmp_path / "1STP.cif"
    doc.write_file(str(path))

    st = ciffer.read(path)

    assert st.metadata["model"] == 1
    assert len(st.residues) == 206  # observed residues only
    assert np.isnan(st.atoms["occupancy"]).all()
    assert np.isnan(st.atoms["b_factor"]).all()
    assert (st.atoms["charge"] == 0).all()


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"model": 9}, "models are"),
        ({"altloc": "A"}, "altloc must be"),
    ],
)
def test_bad_options_raise(options: dict[str, Any], message: str):
    with pytest.raises(ValueError, match=message):
        ciffer.read(DATA / "1STP.cif.gz", **options)
