from shapely.geometry import Polygon, LineString, box

import math
from typing import Tuple

from .data_structures import Coordinate, GridIndex


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
    
    y1,x1 = lat1,lon1 = lat1_lon1 # for clarity - can delete?
    y2,x2 = lat2,lon2 = lat2_lon2 # for clarity - can delete?

    line = LineString([(x1,y1),(x2,y2)])
    return line


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


def _get_index_from_coordinate(lon: float, lat: float, z: int) -> Tuple[(int, int)]:
    """Generate (x,y) tuple from Coordinate latitutes and longitutes.

    Returns
    -------
    Tuple
        (x,y) tuple.
    """
    assert z >= 0, "Zoom level must be positive"
    n = 2**z
    x = int((lon + 180) / 360 * n)
    lat_rad = math.radians(lat)
    y = int((1 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def _get_coordinate_from_index(x: int, y: int, z: int) -> Tuple[(float, float)]:
    """Generate (lon, lat) tuple from GridIndex x, y and zoom level (z).

    Returns
    -------
    Tuple
        (lon, lat) tuple representing the upper left corner of the tile.
    """
    assert z >= 0, "Zoom level must be positive"
    n = 2**z
    lon = (x / n) * 360 - 180
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    lat = math.degrees(lat_rad)
    return lon, lat
