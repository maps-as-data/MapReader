from __future__ import annotations

import math

import pytest

from mapreader.download.downloader_utils import (
    _check_z,
    _get_coordinate_from_index,
    _get_index_from_coordinate,
)


def test_check_z():
    assert _check_z(14) is True
    with pytest.raises(ValueError, match="Zoom level must be positive"):
        _check_z(-1)


def test_get_index_from_coordinate():
    lon, lat, z = 0, 0, 1
    expected = (1, 1)
    assert _get_index_from_coordinate(lon, lat, z) == expected

    lon, lat, z = 180, 85.05112878, 1
    expected = (2, 0)
    assert _get_index_from_coordinate(lon, lat, z) == expected

    lon, lat, z = -180, -85.05112878, 1
    expected = (0, 2)
    assert _get_index_from_coordinate(lon, lat, z) == expected


def test_get_coordinate_from_index():
    x, y, z = 1, 1, 1
    expected = (0.0, 0.0)
    result = _get_coordinate_from_index(x, y, z)
    assert math.isclose(result[0], expected[0], rel_tol=1e-9)
    assert math.isclose(result[1], expected[1], rel_tol=1e-9)

    x, y, z = 2, 0, 1
    expected = (180.0, 85.05112878)
    result = _get_coordinate_from_index(x, y, z)
    assert math.isclose(result[0], expected[0], rel_tol=1e-9)
    assert math.isclose(result[1], expected[1], rel_tol=1e-9)

    x, y, z = 0, 2, 1
    expected = (-180.0, -85.05112878)
    result = _get_coordinate_from_index(x, y, z)
    assert math.isclose(result[0], expected[0], rel_tol=1e-9)
    assert math.isclose(result[1], expected[1], rel_tol=1e-9)
