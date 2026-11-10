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
        ("4DWA.cif.gz", {}),
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


def test_read_bonds():
    st = ciffer.read(DATA / "1EJG.cif.gz", altloc="all")
    bond, conn = st.atoms.edges["bond"], st.atoms.edges["struct_conn"]
    name, altloc = st.atoms["name"], st.atoms["altloc"]
    seq = st.residues["label_seq_id"][st.atom_to_residue]

    # Cys 3 has no altloc; its HG is gone, since SG bonds to Cys 40.
    at_3 = seq[bond.src] == 3
    names = zip(name[bond.src][at_3], name[bond.dst][at_3], strict=True)
    assert sorted(names) == [
        ("C", "O"),
        ("CA", "C"),
        ("CA", "CB"),
        ("CA", "HA"),
        ("CB", "HB2"),
        ("CB", "HB3"),
        ("CB", "SG"),
        ("N", "CA"),
        ("N", "H"),
    ]
    assert bond["order"][at_3].tolist().count(2) == 1  # C=O

    # Atoms of different altlocs are never bonded.
    ends = altloc[bond.src], altloc[bond.dst]
    assert ((ends[0] == ends[1]) | (ends[0] == "") | (ends[1] == "")).all()

    pairs = sorted(zip(seq[conn.src].tolist(), seq[conn.dst].tolist(), strict=True))
    assert pairs == [(3, 40), (4, 32), (16, 26)]
    assert conn["type"].tolist() == ["disulf"] * 3


def test_read_glycans():
    st = ciffer.read(DATA / "4DWA.cif.gz")

    # Each glycan is a chain of its own, and its sugars are told apart by auth_seq_id.
    glycans = st.chains[st.chains["entity_type"] == "branched"]
    assert len(glycans) == 5
    first = glycans[0].residues
    assert first["name"].tolist() == ["NAG", "NAG", "BMA", "BMA", "BMA"]
    assert first["auth_seq_id"].tolist() == ["1", "2", "3", "4", "5"]

    # Sugars link to each other as covale.
    links = glycans.atoms.edges["struct_conn"]
    assert links["type"].tolist() == ["covale"] * 12


def test_read_metadata():
    st = ciffer.read(DATA / "1EJG.cif.gz")

    assert st.metadata == {
        "id": "1EJG",
        "title": "CRAMBIN AT ULTRA-HIGH RESOLUTION: VALENCE ELECTRON DENSITY.",
        "keywords": "PLANT PROTEIN",
        "deposit_date": "2000-03-02",
        "methods": ["X-RAY DIFFRACTION"],
        "authors": [
            "Jelsch, C.",
            "Teeter, M.M.",
            "Lamzin, V.",
            "Pichon-Lesme, V.",
            "Blessing, B.",
            "Lecomte, C.",
        ],
        "resolution": 0.54,
        "r_free": 0.094,
        "release_date": "2000-04-05",  # the first of six revisions
        "doi": "10.1073/pnas.97.7.3171",
        "model": 1,
    }

    # Only values in the file are kept: NMR has no resolution, and 1STP has no DOI.
    assert "resolution" not in ciffer.read(DATA / "1L2Y_3models.cif.gz").metadata
    st = ciffer.read(DATA / "1STP.cif.gz")
    assert "doi" not in st.metadata
    assert st.chains["description"][1] == "BIOTIN"


def test_read_pdb_format():
    pdb = ciffer.read(DATA / "1STP.pdb.gz")
    cif = ciffer.read(DATA / "1STP.cif.gz")

    # Atoms are the same; residues with no atoms and bonds are not in PDB files.
    assert pdb.atoms["name"].tolist() == cif.atoms["name"].tolist()
    assert np.array_equal(pdb.atoms["xyz"], cif.atoms["xyz"])
    assert pdb.chains["entity_type"].tolist() == cif.chains["entity_type"].tolist()
    assert len(pdb.residues) == 206
    assert len(pdb.atoms.edges["bond"]) == 0

    with pytest.raises(ValueError, match="optionally gzipped"):
        ciffer.read(DATA / "1STP.txt")


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
