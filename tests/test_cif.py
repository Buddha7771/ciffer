import warnings
from pathlib import Path
from typing import Any

import gemmi
import numpy as np
import pytest

import ciffer
from ciffer import EdgeList, Structure, Table

DATA = Path(__file__).parent / "data"


def gemmi_atoms(path: Path, model: int = 1, assembly: str | None = None) -> list:
    st = gemmi.read_structure(str(path))
    st.remove_alternative_conformations()
    if assembly is not None:
        how = gemmi.HowToNameCopiedChain.AddNumber
        st.transform_to_assembly(assembly, how, merge_dist=0)
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
        ("1STP.cif.gz", {"assembly": "1"}),
        ("1G2J.cif.gz", {"assembly": "1"}),
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
        "models": [1],
        "assemblies": ["1"],
    }

    # Only values in the file are kept: NMR has no resolution, and 1STP has no DOI.
    assert "resolution" not in ciffer.read(DATA / "1L2Y_3models.cif.gz").metadata
    st = ciffer.read(DATA / "1STP.cif.gz")
    assert "doi" not in st.metadata
    assert st.chains["description"][1] == "BIOTIN"


def test_read_assembly():
    deposited = ciffer.read(DATA / "1G2J.cif.gz")
    st = ciffer.read(DATA / "1G2J.cif.gz", assembly="1")

    assert st.chains.label_asym_id.tolist() == [
        *("A1", "B1", "C1", "D1"),
        *("A2", "B2", "C2", "D2"),
    ]
    assert st.chains.auth_asym_id.tolist() == ["A1"] * 4 + ["A2"] * 4
    assert len(st.atoms.struct_conn) == 2 * len(deposited.atoms.struct_conn)
    assert st.metadata["assembly"] == "1"
    assert "assembly" not in deposited.metadata
    assert ciffer.read(DATA / "1G2J.cif.gz", assembly=True).metadata["assembly"] == "1"
    assert ciffer.read(DATA / "1G2J.cif.gz", assembly=1).metadata["assembly"] == "1"

    # (P)(61-88) moves the unit made by operators 61 to 88 as a whole.
    unit = ciffer.read(DATA / "1M4X.cif.gz", assembly="2").atoms.xyz
    moved = ciffer.read(DATA / "1M4X.cif.gz", assembly="7").atoms.xyz
    assert not np.allclose(moved, unit)
    assert np.allclose(
        np.linalg.norm(moved - moved[0], axis=1),
        np.linalg.norm(unit - unit[0], axis=1),
    )


def test_read_pdb_format():
    pdb = ciffer.read(DATA / "1STP.pdb.gz")
    cif = ciffer.read(DATA / "1STP.cif.gz")

    # Atoms are the same; residues with no atoms and bonds are not in PDB files.
    assert pdb.atoms["name"].tolist() == cif.atoms["name"].tolist()
    assert np.array_equal(pdb.atoms["xyz"], cif.atoms["xyz"])
    assert pdb.chains["entity_type"].tolist() == cif.chains["entity_type"].tolist()
    assert pdb.chains.label_asym_id.tolist() == cif.chains.label_asym_id.tolist()
    assert len(pdb.residues) == 206
    assert len(pdb.atoms.edges["bond"]) == 0

    # Assembly copies are named as in the mmCIF file.
    pdb = ciffer.read(DATA / "1STP.pdb.gz", assembly="1")
    cif = ciffer.read(DATA / "1STP.cif.gz", assembly="1")
    assert pdb.chains.label_asym_id.tolist() == cif.chains.label_asym_id.tolist()
    assert pdb.chains.auth_asym_id.tolist() == cif.chains.auth_asym_id.tolist()
    assert np.array_equal(pdb.atoms.xyz, cif.atoms.xyz)

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
    block.find_mmcif_category("_pdbx_struct_assembly_gen.").erase()
    path = tmp_path / "1STP.cif"
    doc.write_file(str(path))

    st = ciffer.read(path, assembly=True)

    assert st.metadata["model"] == 1
    assert "assembly" not in st.metadata  # the deposited chains
    assert "assemblies" not in st.metadata
    assert len(st.chains) == 3
    assert len(st.residues) == 206  # observed residues only
    assert np.isnan(st.atoms["occupancy"]).all()
    assert np.isnan(st.atoms["b_factor"]).all()
    assert (st.atoms["charge"] == 0).all()


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"model": 9}, "models are"),
        ({"altloc": "A"}, "altloc must be"),
        ({"assembly": "9"}, "assemblies are"),
    ],
)
def test_bad_options_raise(options: dict[str, Any], message: str):
    with pytest.raises(ValueError, match=message):
        ciffer.read(DATA / "1STP.cif.gz", **options)


@pytest.mark.parametrize(
    ("name", "options"),
    [
        ("1G2J.cif.gz", {}),  # RNA and ligands
        ("1STP.cif.gz", {}),  # residues with no atoms
        ("1EJG.cif.gz", {}),  # two residue types at one position
        ("1STP.cif.gz", {"assembly": True}),  # ligand chains before polymer chains
        ("4DWA.cif.gz", {}),  # glycans, disulfides, and hydrogen bonds
        ("1EJG.cif.gz", {"altloc": None}),  # Thr 39 keeps CB of A and HB of B
    ],
)
def test_write_round_trip(tmp_path: Path, name: str, options: dict[str, Any]):
    options = {"altloc": "all"} | options
    st = ciffer.read(DATA / name, **options)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # nothing read is left unwritten
        ciffer.write(st, tmp_path / "out.cif.gz")
    back = ciffer.read(tmp_path / "out.cif.gz", altloc="all")

    # Keys about the source file, such as its models and assemblies, may differ.
    source = ("models", "assembly", "assemblies")
    ours = {k: v for k, v in st.metadata.items() if k not in source}
    assert {k: v for k, v in back.metadata.items() if k not in source} == ours

    for ours, theirs in zip(st.tables, back.tables, strict=True):
        for column in ours.columns:
            floats = ours[column].dtype.kind == "f"
            assert np.array_equal(ours[column], theirs[column], equal_nan=floats)
    assert back.atom_to_residue.tolist() == st.atom_to_residue.tolist()
    assert back.residue_to_chain.tolist() == st.residue_to_chain.tolist()

    for group in ("bond", "struct_conn"):
        ours, theirs = st.atoms.edges[group], back.atoms.edges[group]
        assert theirs.src.tolist() == ours.src.tolist()
        assert theirs.dst.tolist() == ours.dst.tolist()
        for column in ours.columns:
            assert theirs[column].tolist() == ours[column].tolist()


def test_write_metadata(tmp_path: Path):
    path = tmp_path / "out.cif"
    st = ciffer.read(DATA / "1EJG.cif.gz")
    ciffer.write(st, path)
    assert gemmi.read_structure(str(path)).resolution == 0.54

    # An EM resolution has a category of its own.
    st.metadata |= {"methods": ["ELECTRON MICROSCOPY"], "resolution": 3.1}
    ciffer.write(st, path)
    block = gemmi.cif.read(str(path)).sole_block()
    assert block.find_value("_em_3d_reconstruction.resolution") == "3.1"
    assert ciffer.read(path).metadata["resolution"] == 3.1

    st.metadata["doi"] = None
    ciffer.write(st, path)
    assert "doi" not in ciffer.read(path).metadata

    st.metadata["ptm"] = 0.8
    with pytest.warns(UserWarning, match=r"Not written: metadata\['ptm'\]"):
        ciffer.write(st, path)

    st.metadata["authors"] = "Kim, J."
    with pytest.raises(TypeError, match=r"metadata\['authors'\] must be a list"):
        ciffer.write(st, path)


def test_write_pdb(tmp_path: Path):
    st = ciffer.read(DATA / "4DWA.cif.gz")
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # what PDB files cannot hold goes unwarned
        ciffer.write(st, tmp_path / "out.pdb")
    back = ciffer.read(tmp_path / "out.pdb")
    assert back.metadata["resolution"] == st.metadata["resolution"]  # REMARK 2

    # PDB files list each chain with its ligands and waters, so atoms are reordered.
    def atoms(st: Structure) -> list:
        a2r = st.atom_to_residue
        a2c = st.residue_to_chain[a2r]
        return sorted(
            zip(
                st.chains.auth_asym_id[a2c].tolist(),
                st.residues.auth_seq_id[a2r].tolist(),
                st.atoms.name.tolist(),
                st.atoms.altloc.tolist(),
                np.round(st.atoms.xyz, 3).tolist(),
                strict=True,
            )
        )

    assert atoms(back) == atoms(st)
    assert len(back.atoms.struct_conn) == len(st.atoms.struct_conn)
    assert "CRYST1" not in (tmp_path / "out.pdb").read_text()

    st.atoms["plddt"] = 0.0
    with pytest.warns(UserWarning, match=r"Not written: atoms\['plddt'\]\.$"):
        ciffer.write(st, tmp_path / "out.pdb")


def test_write_pdb_names(tmp_path: Path):
    # Assembly copies keep two-letter chain names, and long residue names come back.
    st = ciffer.read(DATA / "1G2J.cif.gz", assembly=True)
    st.residues.name = np.where(st.residues.name == "PYY", "A1LIG", st.residues.name)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ciffer.write(st, tmp_path / "out.ent.gz")
    back = ciffer.read(tmp_path / "out.ent.gz")
    assert sorted(set(back.chains.auth_asym_id.tolist())) == ["A1", "A2"]
    assert "A1LIG" in back.residues.name

    # Values wider than PDB columns are refused rather than cut.
    st.atoms.name = np.where(st.atoms.name == "OP1", "OP1XY", st.atoms.name)
    with pytest.raises(ValueError, match=r"atom names of up to 4 .* \['OP1XY'\]"):
        ciffer.write(st, tmp_path / "out.pdb")
    st.chains.auth_asym_id = np.full(len(st.chains), "ABC")
    with pytest.raises(ValueError, match=r"chain names of up to 2 .* \['ABC'\]"):
        ciffer.write(st, tmp_path / "out.pdb")


def test_write_fills_optional_fields(tmp_path: Path):
    st = Structure(
        atoms=Table(
            {
                "name": ["N", "CA", "N", "C1", "O", "O"],
                "element": ["N", "C", "N", "C", "O", "O"],
                "xyz": np.zeros((6, 3)),
                "plddt": np.ones(6),
            },
            edges={
                "bond": EdgeList([0, 1], [1, 2]),  # the second joins two residues
                "struct_conn": EdgeList([2], [3]),
            },
        ),
        residues=Table({"name": ["GLY", "MSE", "LIG", "HOH", "HOH"]}),
        chains=Table({"description": ["peptide", "ligand", "water"]}),
        atom_to_residue=[0, 0, 1, 2, 3, 4],
        residue_to_chain=[0, 0, 1, 2, 2],
    )
    with pytest.warns(
        UserWarning, match=r"Not written: atoms\['plddt'\], 1 bonds between residues"
    ):
        ciffer.write(st, tmp_path / "out.cif")
    back = ciffer.read(tmp_path / "out.cif")

    assert back.residues.label_seq_id.tolist() == [1, 2, -1, -1, -1]
    assert back.residues.auth_seq_id.tolist() == ["1", "2", "1", "1", "2"]
    assert back.chains.label_asym_id.tolist() == ["A", "B", "C"]
    assert back.chains.auth_asym_id.tolist() == ["A", "B", "C"]
    assert back.chains.entity_id.tolist() == ["1", "2", "3"]
    assert back.chains.entity_type.tolist() == ["polymer", "non-polymer", "water"]
    assert back.chains.polymer_type.tolist() == ["polypeptide(L)", "", ""]
    assert back.atoms.occupancy.tolist() == [1.0] * 6
    assert np.isnan(back.atoms.b_factor).all()
    assert back.atoms.bond.src.tolist() == [0]
    assert back.atoms.bond.order.tolist() == [0]
    assert back.atoms.struct_conn.type.tolist() == ["covale"]

    # A modified residue in a polymer is HETATM, as in PDB entries.
    block = gemmi.cif.read(str(tmp_path / "out.cif")).sole_block()
    groups = list(block.find_values("_atom_site.group_PDB"))
    assert groups == ["ATOM", "ATOM", "HETATM", "HETATM", "HETATM", "HETATM"]


def test_write_checks_inputs(tmp_path: Path):
    st = ciffer.read(DATA / "1G2J.cif.gz")

    with pytest.raises(ValueError, match=r"not a \.cif, \.mmcif, \.pdb, or \.ent file"):
        ciffer.write(st, tmp_path / "out.txt")
    with pytest.raises(ValueError, match=r"share label_asym_id \['A', 'B'"):
        ciffer.write(ciffer.concat([st, st]), tmp_path / "out.cif")

    with pytest.raises(ValueError, match="no atoms"):
        ciffer.write(st.atoms[st.atoms.name == "XX"].extract(), tmp_path / "out.cif")

    with pytest.raises(TypeError, match="write each model of a list to its own"):
        ciffer.write([st, st], tmp_path / "out.cif")  # ty: ignore[invalid-argument-type]

    proteins = st.copy()
    proteins.chains.entity_type = "protein"
    with pytest.raises(ValueError, match=r"entity_type \['protein'\] is not one of"):
        ciffer.write(proteins, tmp_path / "out.cif")

    # Ligand residues tell no polymer type.
    untyped = st.copy()
    untyped.chains.entity_type = "polymer"
    untyped.chains.polymer_type = ""
    with pytest.raises(ValueError, match=r"Polymer chains \['B', 'C', 'D'\] need a"):
        ciffer.write(untyped, tmp_path / "out.cif")

    # Bonds are listed by residue type, so residues of a type must agree.
    double = st.copy()
    double.atoms.bond.order[0] = 2
    with pytest.raises(ValueError, match=r"Residues C give the bond O5'-C5' different"):
        ciffer.write(double, tmp_path / "out.cif")
    bond = st.atoms.bond
    lacking = st.copy()
    lacking.atoms.edges["bond"] = EdgeList(bond.src[1:], bond.dst[1:])
    with pytest.raises(
        ValueError, match=r"lacks the bond C5'-O5' that other C residues"
    ):
        ciffer.write(lacking, tmp_path / "out.cif")

    four_columns = st.copy()
    four_columns.atoms.xyz = np.zeros((len(st.atoms), 4))
    with pytest.raises(ValueError, match=r"shape \(n_atoms, 3\)"):
        ciffer.write(four_columns, tmp_path / "out.cif")

    st.atoms.xyz[0] = np.nan
    st.atoms.xyz[1] = np.inf
    with pytest.raises(ValueError, match="2 atoms have NaN or infinite coordinates"):
        ciffer.write(st, tmp_path / "out.cif")

    del st.atoms["element"]
    with pytest.raises(ValueError, match=r"without \['atoms.element'\]"):
        ciffer.write(st, tmp_path / "out.cif")


def test_write_warns_what_the_file_cannot_hold(tmp_path: Path):
    st = ciffer.read(DATA / "1G2J.cif.gz")
    st.atoms["plddt"] = 0.0
    st.atoms.bond["length"] = 1.5
    st.atoms.edges["contact"] = EdgeList([0], [1])
    with pytest.warns(
        UserWarning,
        match=r"atoms\['plddt'\], atoms.edges\['bond'\]\['length'\], "
        r"atoms.edges\['contact'\]",
    ):
        ciffer.write(st, tmp_path / "out.cif")

    # A ligand with no atoms has no place in the file.
    lonely = Structure(
        atoms=Table({"name": ["CA"], "element": ["C"], "xyz": np.zeros((1, 3))}),
        residues=Table({"name": ["GLY", "LIG"]}),
        chains=Table(
            {
                "label_asym_id": ["A", "B"],
                "entity_type": ["polymer", "non-polymer"],
                "polymer_type": ["polypeptide(L)", ""],
            }
        ),
        atom_to_residue=[0],
        residue_to_chain=[0, 1],
    )
    with pytest.warns(UserWarning, match="1 residues with no atoms outside polymer"):
        ciffer.write(lonely, tmp_path / "out.cif")


@pytest.mark.parametrize("name", ["1G2J.cif.gz", "1STP.cif.gz"])
def test_write_infers_chain_types(tmp_path: Path, name: str):
    st = ciffer.read(DATA / name)
    untyped = st.copy()
    untyped.chains.entity_type = ""  # as from a file without _entity
    del untyped.chains["polymer_type"]
    ciffer.write(untyped, tmp_path / "out.cif")
    back = ciffer.read(tmp_path / "out.cif")

    assert back.chains.entity_type.tolist() == st.chains.entity_type.tolist()
    assert back.chains.polymer_type.tolist() == st.chains.polymer_type.tolist()

    # A given entity_type keeps the polymer_type inferred.
    untyped = st.copy()
    untyped.chains.polymer_type = ""
    ciffer.write(untyped, tmp_path / "out.cif")
    back = ciffer.read(tmp_path / "out.cif")
    assert back.chains.polymer_type.tolist() == st.chains.polymer_type.tolist()


def test_write_names_chains_past_z(tmp_path: Path):
    waters = Structure(
        atoms=Table(
            {"name": ["O"] * 28, "element": ["O"] * 28, "xyz": np.zeros((28, 3))}
        ),
        residues=Table({"name": ["HOH"] * 28}),
        chains=Table({"description": ["water"] * 28}),
        atom_to_residue=range(28),
        residue_to_chain=range(28),
    )
    ciffer.write(waters, tmp_path / "out.cif")
    back = ciffer.read(tmp_path / "out.cif")

    assert back.chains.label_asym_id.tolist()[24:] == ["Y", "Z", "AA", "BA"]


def test_write_polymer_without_sequence_positions(tmp_path: Path):
    # As from a PDB file without SEQRES records.
    st = ciffer.read(DATA / "1G2J.cif.gz")
    st.residues.label_seq_id = -1
    ciffer.write(st, tmp_path / "out.cif")
    back = ciffer.read(tmp_path / "out.cif")

    assert back.chains.polymer_type.tolist() == st.chains.polymer_type.tolist()
    assert back.residues.label_seq_id.tolist() == st.residues.label_seq_id.tolist()
