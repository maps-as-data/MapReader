#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import os
import time

from typing import List, Tuple, Optional


def create_hf(geom: str) -> Tuple[str, str]:
    """
    Creates a header and footer for a GeoJSON file of a specified geometry
    type.

    Parameters
    ----------
    geom : str
        The geometry type for the GeoJSON file. Currently only ``"polygon"`` is
        implemented.

    Returns
    -------
    Tuple[str, str]
        A tuple of strings representing the header and footer of the GeoJSON
        file.

    Raises
    ------
    NotImplementedError
        If the specified geometry type is not implemented.
    """

    if geom == "polygon":
        header = """
        {
        "type": "FeatureCollection",
        "features": [
        {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [
        """

        footer = """
                ]
            }
        }
        ]
        }"""
    else:
        raise NotImplementedError(f"{geom} is not implemented.")

    return header, footer


class input_class:
    """
    A simple class with one property, ``name``, used by MapReader's TileServer
    scraper's and stitcher's ``runner`` and ``scraper`` functions respectively.

    Parameters
    ----------
    name : str
        The name of the input.

    Attributes
    ----------
    name : str
        The name of the input.
    """

    def __init__(self, name: str):
        self.name = name


def latlon2tile(lat: float, lon: float, zoom: int) -> Tuple[int, int]:
    """
    Convert latitude and longitude coordinates to tile indices at a given zoom
    level.

    Parameters
    ----------
    lat : float
        Latitude in decimal degrees.
    lon : float
        Longitude in decimal degrees.
    zoom : int
        Zoom level, which determines the resolution of the tile.

    Returns
    -------
    Tuple[int, int]
        The x and y tile indices corresponding to the input ``latitude`` and
        ``longitude`` coordinates at the provided ``zoom`` (zoom level).

    Notes
    -----
    From OSM Slippy Tile definitions & https://github.com/Caged/tile-stitch.

    Reference: https://github.com/stamen/the-ultimate-tile-stitcher.
    """

    lat_radians = lat * math.pi / 180.0
    n = 1 << zoom

    return (
        n * ((lon + 180.0) / 360.0),
        n
        * (
            1
            - (
                math.log(math.tan(lat_radians) + 1 / math.cos(lat_radians))
                / math.pi
            )
        )
        / 2.0,
    )


def tile2latlon(x: int, y: int, zoom: int) -> Tuple[float, float]:
    """
    Convert tile coordinates (``x``, ``y``) and ``zoom`` (zoom level) to
    latitude and longitude coordinates.

    Parameters
    ----------
    x : int
        Tile X coordinate.
    y : int
        Tile Y coordinate.
    zoom : int
        Zoom level.

    Returns
    -------
    tuple
        A tuple containing latitude and longitude coordinates in degrees.

    Notes
    -----
    Reference: https://github.com/stamen/the-ultimate-tile-stitcher.
    """
    n = 1 << zoom
    lat_radians = math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n)))
    lat = lat_radians * 180 / math.pi
    lon = 360 * x / n - 180.0
    return (lat, lon)


def collect_coord_info(
    list_files: List[str],
) -> Tuple[float, float, float, float]:
    """
    Collects the minimum and maximum latitude and longitude from a list of
    tiles.

    Parameters
    ----------
    list_files : list of str
        List of file paths to be read.

    Returns
    -------
    tuple
        A tuple containing the minimum longitude, maximum longitude,
        minimum latitude, and maximum latitude of the tiles.
    """
    # initialize lat/lon
    min_lat = None
    max_lat = None
    min_lon = None
    max_lon = None

    for file in list_files:
        z, x, y = os.path.basename(file).split("_")
        y = y.split(".")[0]
        lat, lon = tile2latlon(int(x), int(y), int(z))
        if min_lat is None:
            min_lat = lat
            min_lon = lon
            max_lat = lat
            max_lon = lon
        else:
            min_lat = min(min_lat, lat)
            min_lon = min(min_lon, lon)
            max_lat = max(max_lat, lat)
            max_lon = max(max_lon, lon)

    return min_lon, max_lon, min_lat, max_lat


def check_par_jobs(jobs: List, sleep_time: Optional[int] = 1) -> None:
    """
    Wait for all processes in a list of parallel jobs to finish.

    Parameters
    ----------
    jobs : list
        A list of processes.
    sleep_time : float, optional
        Time to wait before checking the status of processes. Defaults to
        ``1``.

    Returns
    -------
    None

    ..
        TODO: This function's documentation needs a type for the List[...]
        type provided for the jobs parameter above. What is it?
    """
    pp_flag = True
    while pp_flag:
        for proc in jobs:
            if proc.is_alive():
                time.sleep(sleep_time)
                pp_flag = True
                break
            else:
                pp_flag = False
    if not pp_flag:
        sep = "================================"
        print("\n\n{sep}")
        print("All %s processes are finished..." % len(jobs))
        print(sep)
