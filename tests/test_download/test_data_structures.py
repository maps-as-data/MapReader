from __future__ import annotations

import pytest

from mapreader.download.data_structures import Coordinate, GridBoundingBox, GridIndex


@pytest.fixture
def init_coordinate():
    coordinate = Coordinate(0, 90)
    assert coordinate.lat == 0
    assert coordinate.lon == 90
    assert str(coordinate) == "(0, 90)"

    coordinate = Coordinate(0, -180)
    assert coordinate.lat == 0
    assert coordinate.lon == -180
    assert str(coordinate) == "(0, -180)"

    with pytest.raises(ValueError, match="must be in range"):
        coordinate = Coordinate(-180, 90)

    with pytest.raises(ValueError, match="must be in range"):
        coordinate = Coordinate(-180, 90)

    with pytest.raises(ValueError, match="must be in range"):
        coordinate = Coordinate(0, -181)


@pytest.fixture
def init_grid_index():
    grid_index = GridIndex(1, 2, 3)
    assert grid_index.x == 1
    assert grid_index.y == 2
    assert grid_index.z == 3

    with pytest.raises(
        ValueError, match="Zoom level must be greater than or equal to 0"
    ):
        grid_index = GridIndex(1, 2, -1)

    with pytest.raises(ValueError, match=r"X value must be in range \[0, 8\]"):
        grid_index = GridIndex(9, 2, 3)

    with pytest.raises(ValueError, match=r"Y value must be in range \[0, 8\]"):
        grid_index = GridIndex(1, 9, 3)


@pytest.fixture
def init_grid_bounding_box():
    cell1 = GridIndex(1, 2, 3)
    cell2 = GridIndex(2, 3, 4)

    with pytest.raises(
        NotImplementedError, match="Can't calculate a grid on different scales yet"
    ):
        bb = GridBoundingBox(cell1, cell2)

    cell2 = GridIndex(2, 3, 3)

    bb = GridBoundingBox(cell1, cell2)

    assert bb.z == 3
    assert str(bb.lower_corner) == "(3, 1, 2)"
    assert str(bb.upper_corner) == "(3, 2, 3)"
    assert bb.x_range == range(1, 3)
    assert bb.y_range == range(2, 4)
    assert bb.covered_cells == 4
