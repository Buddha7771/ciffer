from typing import Any

import numpy as np
import pytest

from ciffer import EdgeList, Structure, Table, concat


@pytest.fixture
def st() -> Structure:
    """Chain A has residues 0 and 1, chain B has residue 2; six atoms in total."""
    bond = EdgeList([0, 1, 3], [1, 2, 4], {"order": [1, 1, 2]})
    atoms = Table(
        {
            "name": ["N", "CA", "C", "N", "CA", "ZN"],
            "xyz": np.arange(18.0).reshape(6, 3),
        },
        edges={"bond": bond},
    )
    return Structure(
        atoms=atoms,
        residues=Table({"name": ["ALA", "GLY", "ZN"]}),
        chains=Table({"id": ["A", "B"]}),
        atom_to_residue=[0, 0, 0, 1, 1, 2],
        residue_to_chain=[0, 0, 1],
        metadata={"id": "1abc"},
    )


def test_inputs_are_checked(st: Structure):
    atoms = Table({"name": ["N", "CA"]})
    residues = Table({"name": ["ALA"]})
    chains = Table({"id": ["A"]})

    with pytest.raises(ValueError, match="1 entries for 2 rows"):
        Structure(
            atoms=atoms,
            residues=residues,
            chains=chains,
            atom_to_residue=[0],
            residue_to_chain=[0],
        )
    with pytest.raises(IndexError, match="outside the 1 parent rows"):
        Structure(
            atoms=atoms,
            residues=residues,
            chains=chains,
            atom_to_residue=[0, 1],
            residue_to_chain=[0],
        )
    with pytest.raises(TypeError, match="must be a Table"):
        Structure(
            atoms=st.atoms[[0, 1]],  # ty: ignore[invalid-argument-type]
            residues=residues,
            chains=chains,
            atom_to_residue=[0, 0],
            residue_to_chain=[0],
        )


def test_whole_view_reads_stored_array(st: Structure):
    st.atoms["xyz"][0] = -1.0

    assert st.atoms["xyz"][0].tolist() == [-1.0, -1.0, -1.0]


def test_selected_view_reads_copies_and_takes_no_assignments(st: Structure):
    selected = st.atoms[[1, 2]]

    selected["xyz"][:] = 0.0
    assert st.atoms["xyz"][1].tolist() == [3.0, 4.0, 5.0]

    message = r"takes no assignments; st\.atoms\['xyz'\]\[view\.indices\] = value"
    with pytest.raises(TypeError, match=message):
        selected["xyz"] = 0.0
    with pytest.raises(TypeError, match="takes no assignments"):
        selected.xyz = 0.0
    with pytest.raises(TypeError, match=r"add the column first, .* = value"):
        selected["b_factor"] = 0.0

    st.atoms["xyz"][selected.indices] = 0.0
    assert st.atoms["xyz"][1].tolist() == [0.0, 0.0, 0.0]


def test_assignment_widens_columns(st: Structure):
    # A longer string or a float among integers widens the column rather than being cut.
    st.chains["id"] = "ABC"
    assert st.chains["id"].tolist() == ["ABC", "ABC"]
    st.atoms.edges["bond"]["order"] = 1.5
    assert st.atoms.edges["bond"]["order"].tolist() == [1.5, 1.5, 1.5]

    # A value that fits keeps the dtype, booleans stay masks, and numbers stay numbers.
    st.atoms["b_factor"] = np.zeros(6, np.float32)
    st.atoms["b_factor"] = 0
    assert st.atoms["b_factor"].dtype == np.float32
    st.atoms["hetero"] = np.zeros(6, bool)
    st.atoms["hetero"] = 1
    assert st.atoms["hetero"].dtype == bool
    with pytest.raises(ValueError, match="could not convert"):
        st.atoms["b_factor"] = "x"


def test_add_and_delete_columns(st: Structure):
    st.atoms["b_factor"] = np.zeros(6)
    assert "b_factor" in st.atoms

    with pytest.raises(TypeError):
        del st.atoms[[0]]["b_factor"]

    del st.atoms["b_factor"]
    assert "b_factor" not in st.atoms

    with pytest.raises(ValueError, match="last column"):
        del st.chains["id"]


def test_standard_columns_as_attributes(st: Structure):
    assert st.atoms.xyz is st.atoms["xyz"]
    st.atoms.xyz = np.zeros((6, 3))
    assert st.atoms["xyz"].tolist() == [[0.0, 0.0, 0.0]] * 6
    assert st.chains[1].residues.name.tolist() == ["ZN"]

    # A missing column raises AttributeError, so hasattr works.
    assert not hasattr(st.atoms, "b_factor")
    with pytest.raises(AttributeError, match="no 'label_asym_id' column"):
        _ = st.chains.label_asym_id

    # Other names are columns only through [], so a misspelled field is not lost.
    with pytest.raises(AttributeError, match=r"no 'plddt' field; .* view\['plddt'\]"):
        st.atoms.plddt = 0.0
    with pytest.raises(AttributeError, match="no 'metdata' attribute"):
        st.metdata = {}


def test_standard_edges_as_attributes(st: Structure):
    st.atoms.bond.order[0] = 3
    assert st.atoms.edges["bond"]["order"].tolist() == [3, 1, 2]

    bond = st.atoms[[4, 3, 0]].bond
    assert list(zip(bond.src, bond.dst, strict=True)) == [(1, 0)]
    assert not hasattr(st.atoms, "struct_conn")

    with pytest.raises(AttributeError, match=r"no 'ordr' field; .* edges\['ordr'\]"):
        st.atoms.bond.ordr = 2


def test_level_conversion_uses_stored_order(st: Structure):
    assert st.atoms[[5, 3, 0, 4]].residues.indices.tolist() == [0, 1, 2]
    assert st.atoms[[5, 3, 0, 4]].chains.indices.tolist() == [0, 1]
    assert st.chains[[1, 0]].atoms.indices.tolist() == [0, 1, 2, 3, 4, 5]
    assert st.residues[[1, 1]].atoms.indices.tolist() == [3, 4]
    assert [r.atoms.indices.tolist() for r in st.residues] == [[0, 1, 2], [3, 4], [5]]


def test_extract_uses_stored_order(st: Structure):
    sub = st.atoms[[5, 3, 0, 4]].extract()

    assert sub.atoms["name"].tolist() == ["N", "N", "CA", "ZN"]
    assert sub.residues["name"].tolist() == ["ALA", "GLY", "ZN"]
    assert sub.chains["id"].tolist() == ["A", "B"]
    assert sub.atom_to_residue.tolist() == [0, 1, 1, 2]
    assert sub.residue_to_chain.tolist() == [0, 0, 1]

    bond = sub.atoms.edges["bond"]
    assert list(zip(bond.src, bond.dst, strict=True)) == [(1, 2)]


@pytest.mark.parametrize("make", [lambda st: st.chains[[0]].extract(), Structure.copy])
def test_new_structures_share_nothing(st: Structure, make: Any):
    sub = make(st)
    sub.atoms["xyz"][:] = 0.0
    sub.metadata["id"] = "changed"

    assert st.atoms["xyz"][1].tolist() == [3.0, 4.0, 5.0]
    assert st.metadata["id"] == "1abc"


def test_extract_nothing(st: Structure):
    empty = st.atoms[st.atoms["name"] == "FE"].extract()

    assert len(empty.atoms) == len(empty.residues) == len(empty.chains) == 0
    assert empty.atoms["xyz"].shape == (0, 3)
    assert len(empty.atoms.edges["bond"]) == 0


def test_selected_view_renumbers_edges(st: Structure):
    bond = st.atoms[[4, 3, 0]].edges["bond"]

    assert list(zip(bond.src, bond.dst, strict=True)) == [(1, 0)]
    assert bond["order"].tolist() == [2]


def test_concat_shifts_parents_and_clears_metadata(st: Structure):
    both = concat([st, st])

    assert both.atom_to_residue.tolist() == [0, 0, 0, 1, 1, 2, 3, 3, 3, 4, 4, 5]
    assert both.residue_to_chain.tolist() == [0, 0, 1, 2, 2, 3]
    assert both.metadata == {}

    bond = both.atoms.edges["bond"]
    assert list(zip(bond.src, bond.dst, strict=True))[3:] == [(6, 7), (7, 8), (9, 10)]


def test_concat_checks_inputs(st: Structure):
    other = st.copy()
    del other.atoms["name"]
    with pytest.raises(ValueError, match="missing \\['name'\\]"):
        concat([st, other])


def test_bad_edges_added_later_are_caught(st: Structure):
    other = st.copy()
    other.atoms.edges["bond"] = EdgeList([0], [9])
    with pytest.raises(IndexError, match="outside the 6 rows"):
        concat([other, st])
    with pytest.raises(IndexError, match="outside the 6 rows"):
        other.to_bytes()


def test_bytes_roundtrip(st: Structure):
    st.metadata["resolution"] = np.float32(1.5)
    st.metadata["operators"] = {"rotation": np.eye(3)}

    back = Structure.from_bytes(st.to_bytes())

    assert [repr(t) for t in back.tables] == [repr(t) for t in st.tables]
    assert back.atoms["name"].tolist() == st.atoms["name"].tolist()
    assert np.array_equal(back.atoms["xyz"], st.atoms["xyz"])
    assert back.atom_to_residue.tolist() == st.atom_to_residue.tolist()
    assert back.atoms.edges["bond"]["order"].tolist() == [1, 1, 2]
    assert back.metadata["resolution"] == 1.5
    assert np.array_equal(back.metadata["operators"]["rotation"], np.eye(3))


def test_to_bytes_names_unsavable_value(st: Structure):
    st.metadata["source"] = {"date": object()}

    with pytest.raises(TypeError, match=r"metadata\.source\.date"):
        st.to_bytes()


def test_repr_shows_schema(st: Structure):
    text = repr(st)

    assert "<Structure atoms=6 residues=3 chains=2>" in text
    assert "    id  '1abc'" in text


def test_view_repr_shows_columns_and_edges(st: Structure):
    assert repr(st.atoms).startswith("<AtomView atoms=6>\n  name")
    assert "    bond  3 edges" in repr(st.atoms)
    assert repr(st.atoms.bond).startswith("<Bond edges=3>")

    # Only the edges between the selected rows are counted.
    assert repr(st.atoms[:2]).startswith("<AtomView atoms=2 of 6>")
    assert "    bond  1 edges" in repr(st.atoms[:2])
