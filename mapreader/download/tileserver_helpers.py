#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import os
import time

# -------
def create_hf(geom):
    """Create header and footer for different types of geometries

    Args:
        geom (str): geometry type, e.g., polygone 
    """
    
    if geom == "polygone":
        header = '''
        {
        "type": "FeatureCollection",
        "features": [
        {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [
        '''

        footer = '''
                ]
            }
        }
        ]
        }'''
    else:
        raise ValueError(f"{geom} is not implemented.")

    return header, footer

# -------
class input_class:
    """initialize input class"""
    def __init__(self, name):
        self.name = name

# -------
def latlon2tile(lat, lon, zoom):
    """Convert lat/lon/zoom to tiles

    from OSM Slippy Tile definitions & https://github.com/Caged/tile-stitch
    Reference: https://github.com/stamen/the-ultimate-tile-stitcher
    """
    lat_radians = lat * math.pi / 180.0
    n = 1 << zoom
    return (
        n * ((lon + 180.0) / 360.0),
        n * (1 - (math.log(math.tan(lat_radians) + 1 / math.cos(lat_radians)) / math.pi)) / 2.0
    )

# -------
def tile2latlon(x, y, zoom):
    """Convert x/y/zoom to lat/lon

    Reference: https://github.com/stamen/the-ultimate-tile-stitcher
    """
    n = 1 << zoom
    lat_radians = math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n)))
    lat = lat_radians * 180 / math.pi
    lon = 360 * x / n - 180.0
    return (lat, lon)

# -------
def collect_coord_info(list_files):
    """Collect min/max lat/lon from a list of tiles

    Args:
        list_files (list): list of files to be read
    """
    # initialize lat/lon
    min_lat = None; max_lat = None
    min_lon = None; max_lon = None
                          
    for one_file in list_files:
        z, x, y = os.path.basename(one_file).split("_")
        y = y.split(".")[0]
        lat, lon = tile2latlon(int(x), int(y), int(z))
        if min_lat == None:
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

# -------
def check_par_jobs(jobs, sleep_time=1):
    """
    check if all the parallel jobs are finished
    :param jobs:
    :param sleep_time:
    :return:
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
        print('\n\n================================')
        print('All %s processes are finished...' % len(jobs))
        print('================================')

