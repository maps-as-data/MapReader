from shapely.geometry import Polygon, box

import math
from typing import Tuple

from .data_structures import Coordinate, GridIndex


def create_polygon_from_latlons(min_lat: float, min_lon: float, max_lat: float, max_lon: float) -> Polygon:
    min_y, max_y = min_lat, max_lat # for clarity - can delete?
    min_x, max_x = min_lon, max_lon # for clarity - can delete?
    
    polygon = box(min_x, min_y, max_x, max_y)
    return polygon

# The code below converts lon-lat requests to the respective tile indices.
# Code taken from https://github.com/baurls/TileStitcher. 
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
    """Create Coordiante object from GridIndex.

    Parameters
    ----------
    grid_index : GridIndex
        GridIndex to convert

    Returns
    -------
    Coordinate
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
    x = int(((lon + 180) / 360) * (2 ** z))
    scaled_lat = lat * math.pi / 180
    y = int((1 - (math.log(math.tan(scaled_lat) + 1 / math.cos(scaled_lat))) / math.pi) * 2 ** (z - 1))
    return x, y


def _get_coordinate_from_index(x: int, y: int, z: int) -> Tuple[(float, float)]:
    """Generate (lon, lat) tuple from GridIndex x, y and zoom level (z).

    Returns
    -------
    Tuple
        (lon, lat) tuple.
    """
    assert z >= 0, "Zoom level must be positive"
    divisor = (2 ** z)
    lon = (x / divisor) * 360 - 180
    lat = math.degrees(math.atan(math.sinh(math.pi - (y / divisor) * 2 * math.pi)))
    return lon, lat
