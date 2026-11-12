from typing import Any

import numpy as np
import pytest

from ciffer import EdgeList, Table


@pytest.fixture
def chain() -> Table:
    """Four atoms in a line, bonded 0-1, 1-2, 2-3."""
    bond = EdgeList([0, 1, 2], [1, 2, 3], {"order": [1, 2, 1]})
    return Table(
        {"xyz": np.arange(12.0).reshape(4, 3), "name": ["N", "CA", "C", "O"]},
        edges={"bond": bond},
    )


def test_columns_must_have_equal_length():
    table = Table({"a": [1, 2, 3]})

    with pytest.raises(ValueError, match="2 rows, but the table has 3"):
        table["b"] = [1, 2]


def test_scalar_fills_like_numpy(chain: Table):
    chain["xyz"] = 0
    assert chain["xyz"].dtype == np.float64
    assert chain["xyz"].shape == (4, 3)

    chain["occupancy"] = 1.0
    assert chain["occupancy"].tolist() == [1.0, 1.0, 1.0, 1.0]

    with pytest.raises(ValueError, match="no columns yet"):
        Table()["a"] = 1


def test_take_renumbers_edges_and_drops_cut_ones(chain: Table):
    taken = chain.take([2, 1, 3])

    assert taken["name"].tolist() == ["C", "CA", "O"]
    bond = taken.edges["bond"]
    assert list(zip(bond.src, bond.dst, strict=True)) == [(1, 0), (0, 2)]
    assert bond["order"].tolist() == [2, 1]


def test_take_repeated_rows_point_edges_to_first(chain: Table):
    bond = chain.take([0, 1, 0]).edges["bond"]

    assert list(zip(bond.src, bond.dst, strict=True)) == [(0, 1)]


def test_take_accepts_mask_and_slice(chain: Table):
    assert chain.take(chain["name"] == "CA")["name"].tolist() == ["CA"]
    assert len(chain.take(slice(1, 3))) == 2

    with pytest.raises(TypeError, match="1-D"):
        chain.take(np.array([[0, 1], [2, 3]]))


def test_edges_check_their_ends():
    assert EdgeList([], []).src.dtype == np.intp

    with pytest.raises(TypeError, match="got 1-D float64"):
        EdgeList([0], [0.5])
    with pytest.raises(ValueError, match="dst has 1 entries, but there are 2 edges"):
        EdgeList([0, 1], [1])
    with pytest.raises(ValueError, match="order has 1 entries, but there are 2 edges"):
        EdgeList([0, 1], [1, 2], {"order": [1]})


@pytest.mark.parametrize(
    ("bond", "error"),
    [
        (Table({"a": [0]}), TypeError),
        (EdgeList([0], [9]), IndexError),
    ],
)
def test_edges_are_checked_against_rows(bond: Any, error: type[Exception]):
    with pytest.raises(error):
        Table({"a": [1, 2]}, edges={"bond": bond})


def test_repr_lists_columns_and_edges(chain: Table):
    text = repr(chain)

    assert "<Table rows=4>" in text
    assert "xyz   float64 (3,)" in text
    assert "  edges\n    bond  3 edges\n      order  int64" in text
    assert repr(chain.edges["bond"]) == "<EdgeList edges=3>\n  order  int64"
