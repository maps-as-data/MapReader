from __future__ import annotations

import math

from shapely.geometry import LineString, Polygon, box

from .data_structures import Coordinate, GridBoundingBox, GridIndex


def create_polygon_from_latlons(
    min_lat: float,
    min_lon: float,
    max_lat: float,
    max_lon: float,
) -> Polygon:
    """Creates a polygon from latitudes and longitudes.

    Parameters
    ----------
    min_lat : float
        minimum latitude
    min_lon : float
        minimum longitude
    max_lat : float
        maximum latitude
    max_lon : float
        maximum longitude

    Returns
    -------
    Polygon
        shapely Polgyon
    """
    min_y, max_y = min_lat, max_lat  # for clarity - can delete?
    min_x, max_x = min_lon, max_lon  # for clarity - can delete?

    polygon = box(min_x, min_y, max_x, max_y)
    return polygon


def create_line_from_latlons(
    lat1_lon1: tuple,
    lat2_lon2: tuple,
) -> LineString:
    """Creates a line between two points.

    Parameters
    ----------
    lat1_lon1 : tuple
        Tuple defining first point
    lat2 : tuple
        Tuple defining second point

    Returns
    -------
    LineString
        shapely LineString
    """

    y1, x1 = lat1, lon1 = lat1_lon1  # for clarity - can delete?
    y2, x2 = lat2, lon2 = lat2_lon2  # for clarity - can delete?
    return LineString([(x1, y1), (x2, y2)])


def get_grid_bb_from_polygon(polygon: Polygon, zoom_level: int):
    """
    Create GridBoundingBox object from shapely.Polygon

    Parameters
    ----------
    polygon : shapely.Polygon
        shapely.Polygon to convert.
    zoom_level : int
        Zoom level to use when creating GridBoundingBox

    Returns
    -------
    GridBoundingBox
    """
    min_x, min_y, max_x, max_y = polygon.bounds
    start = Coordinate(min_y, min_x)  # (lat, lon), lower left
    end = Coordinate(max_y, max_x)  # (lat, lon), upper right
    start_idx = get_index_from_coordinate(start, zoom_level)
    end_idx = get_index_from_coordinate(end, zoom_level)
    return GridBoundingBox(start_idx, end_idx)


def get_polygon_from_grid_bb(grid_bb: GridBoundingBox):
    """
    Create shapely.Polygon object from GridBoundingBox

    Parameters
    ----------
    grid_bb : GridBoundingBox
        GridBoundingBox to convert.

    Returns
    -------
    shapely.Polygon
    """
    lower_corner = grid_bb.lower_corner  # SW
    upper_corner = grid_bb.upper_corner  # SW

    # for NE corner of upper right tile, do x+1 and y+1
    upper_corner_NE = GridIndex(upper_corner.x + 1, upper_corner.y + 1, upper_corner.z)

    SW_coord = get_coordinate_from_index(lower_corner)
    NE_coord = get_coordinate_from_index(upper_corner_NE)

    polygon = create_polygon_from_latlons(
        SW_coord.lat, SW_coord.lon, NE_coord.lat, NE_coord.lon
    )
    return polygon


# The code below converts lon-lat requests to the respective tile indices.
# Code adapted from https://github.com/baurls/TileStitcher.
# Conversions are taken from https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames.


def get_index_from_coordinate(coordinate: Coordinate, zoom: int) -> GridIndex:
    """Create GridIndex object from Coordinate.

    Parameters
    ----------
    coordinate : Coordinate
        Coordinate to convert
    zoom : int
        Zoom level to use when creating GridIndex

    Returns
    -------
    GridIndex
    """
    (x, y) = _get_index_from_coordinate(coordinate.lon, coordinate.lat, zoom)
    return GridIndex(x, y, zoom)


def get_coordinate_from_index(grid_index: GridIndex) -> Coordinate:
    """Create Coordinate object from GridIndex.

    Parameters
    ----------
    grid_index : GridIndex
        GridIndex to convert

    Returns
    -------
    Coordinate
        The upper left corner of the tile.


    """
    lon, lat = _get_coordinate_from_index(grid_index.x, grid_index.y, grid_index.z)
    return Coordinate(lat, lon)


def _check_z(z):
    if not z >= 0:
        raise ValueError("Zoom level must be positive")
    return True


def _get_index_from_coordinate(lon: float, lat: float, z: int) -> tuple[(int, int)]:
    """Generate (x,y) tuple from Coordinate latitudes and longitudes.

    Returns
    -------
    Tuple
        (x,y) tuple.
    """
    _check_z(z)
    n = 2**z
    x = int((lon + 180) / 360 * n)
    lat_rad = math.radians(lat)
    y = int((1 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def _get_coordinate_from_index(x: int, y: int, z: int) -> tuple[(float, float)]:
    """Generate (lon, lat) tuple from GridIndex x, y and zoom level (z).

    Returns
    -------
    Tuple
        (lon, lat) tuple representing the upper left corner of the tile.
    """
    _check_z(z)
    n = 2**z
    lon = (x / n) * 360 - 180
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    lat = math.degrees(lat_rad)
    return lon, lat
