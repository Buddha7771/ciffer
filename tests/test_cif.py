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
    ],
)
def test_read_matches_gemmi(name: str, options: dict[str, Any]):
    st = ciffer.read(DATA / name, **options)

    assert ciffer_atoms(st) == gemmi_atoms(DATA / name, **options)


def test_read_all_models():
    models = ciffer.read(DATA / "1L2Y_3models.cif.gz", model="all")

    assert [st.metadata["model"] for st in models] == [1, 2, 3]
    second = ciffer.read(DATA / "1L2Y_3models.cif.gz", model=2)
    assert np.array_equal(models[1].atoms["xyz"], second.atoms["xyz"])


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
    loop = doc.sole_block().find_loop("_atom_site.id").get_loop()
    items = ["pdbx_PDB_model_num", "occupancy", "B_iso_or_equiv", "pdbx_formal_charge"]
    for item in items:
        loop.remove_column(f"_atom_site.{item}")
    path = tmp_path / "1STP.cif"
    doc.write_file(str(path))

    st = ciffer.read(path)

    assert st.metadata["model"] == 1
    assert np.isnan(st.atoms["occupancy"]).all()
    assert np.isnan(st.atoms["b_factor"]).all()
    assert (st.atoms["charge"] == 0).all()


def test_missing_model_raises():
    with pytest.raises(ValueError, match="models are"):
        ciffer.read(DATA / "1STP.cif.gz", model=9)
